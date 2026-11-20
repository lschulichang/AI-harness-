"""B、C 回归训练师的白名单参数和候选反馈。"""
from .llm_client import Planner

TRAINING_ACTIONS = {"INCREMENTAL_TRAINING", "MIXED_DATA_TRAINING"}
LIMITS = {"learning_rate": (1e-6, 0.01), "batch_size": (8, 512),
          "epochs": (1, 50), "window_size": (1, 8)}
DEFAULTS = {"learning_rate": 0.001, "batch_size": 128, "epochs": 5, "window_size": 4}
SYSTEM = (
    "current_validation 始终是本批入口模型的验证指标，不能当作任何候选的结果。"
    "batch_feedback.attempts 为空时意味着本批尚未试训，不能据此判定续训没有收益或没有通过选模门槛。"
    "有标签和预算时，通常先试一个保守候选；也可根据具体证据决定跳过，例如标签不足。"
    "首批没有漂移参考属于正常情况。选模门槛只用于候选训练后的实际验证。"
    "每次提案只训练一个候选，都从同一批初模型出发，只使用当前和已开放的训练数据。"
    "仅返回简洁 JSON，字段为 decision、reason、training_config。"
    "动作可为 NO_UPDATE、INCREMENTAL_TRAINING、MIXED_DATA_TRAINING。"
    "跳过或停止时 training_config 为 {}；训练须给出白名单全部四个参数。"
    "仅本批续训使用 window_size=1；有历史回放才可混合续训，window_size 至少为 2。"
)
FEEDBACK = (
    " 本批已有候选结果。查看实际当前验证、历史退化和剩余预算。"
    "如有值得尝试的不同配置，可以继续；返回 NO_UPDATE 时只结束搜索，程序仍会采用已有的最佳合格候选。"
    "若有当前验证目标，优先达到该 RMSE；未达目标时，程序仍可采用优于批初且通过历史门槛的候选。"
)


def training_defaults(dataset):
    return {**DEFAULTS, "batch_size": 256} if dataset == "cmapss_rul" else DEFAULTS


def rules(context, defaults):
    replay = context["replay_batches"]
    feedback = context["batch_feedback"]
    attempts = feedback["attempts"]
    target = feedback.get("validation_target_rmse")
    stable = target is not None and context["current_validation"]["rmse"] <= target
    if attempts:
        reason = "结束本批候选搜索"
    elif context["new_rows"] < 80:
        reason = "本批有标签样本不足"
    elif stable and context["drift"]["feature_count"] < 20:
        reason = "当前验证达到目标，特征漂移较少"
    else:
        return {"decision": "MIXED_DATA_TRAINING" if replay else "INCREMENTAL_TRAINING",
                "reason": "尝试保守续训，由实际候选验证结果决定是否采用",
                "training_config": {**defaults, "window_size": defaults["window_size"] if replay else 1}}
    return {"decision": "NO_UPDATE", "reason": reason, "training_config": {}}


def validate(proposal, history_count):
    if not isinstance(proposal, dict) or set(proposal) != {"decision", "reason", "training_config"}:
        raise ValueError("方案字段必须为 decision、reason、training_config")
    action, reason, params = (proposal[key] for key in ("decision", "reason", "training_config"))
    if not isinstance(action, str) or action not in TRAINING_ACTIONS | {"NO_UPDATE"}:
        raise ValueError("未知训练动作")
    if not isinstance(reason, str) or not reason.strip() or not isinstance(params, dict):
        raise ValueError("方案缺少理由或参数对象")
    if action == "NO_UPDATE":
        if params:
            raise ValueError("保留模型时训练参数必须为空")
        return proposal
    if set(params) != LIMITS.keys():
        raise ValueError("训练配置不符合 RUL 白名单")
    for name, (low, high) in LIMITS.items():
        value = params[name]
        types = (int, float) if name == "learning_rate" else (int,)
        if type(value) not in types or not low <= value <= high:
            raise ValueError(f"{name} 超出允许范围")
    if action == "INCREMENTAL_TRAINING" and params["window_size"] != 1:
        raise ValueError("仅本批续训的窗口必须为 1")
    if action == "MIXED_DATA_TRAINING" and (not history_count or params["window_size"] < 2):
        raise ValueError("混合训练须有历史数据且窗口至少为 2")
    return proposal


class RulTrainer(Planner):
    def decide(self, key, context):
        saved = self.ledger.get("trainer_decision", key)
        if saved is not None:
            return saved
        count = len(context["replay_batches"])
        previous = context["batch_feedback"]["attempts"]
        output, error = None, ""
        source = "rules"
        if self.config.strategy == "llm":
            def check(proposal):
                validated = validate(proposal, count)
                params = validated["training_config"]
                if params:
                    effective = {**params, "window_size": min(params["window_size"], count + 1)}
                    if any(attempt["effective_config"] == effective for attempt in previous):
                        raise ValueError("该有效训练配置已尝试")
                return validated

            task, unit = ("涡扇发动机", "循环") if self.config.dataset == "cmapss_rul" else ("轴承", "分钟")
            prompt = (
                f"/no_think 你为{task}剩余寿命回归决定是否尝试续训。RMSE 和 MAE 单位为{unit}，越低越好。"
                + SYSTEM + (FEEDBACK if previous else "")
            )
            output, error = self._request_proposal(
                key, prompt, context, check, output_budget=self.config.max_api_output_tokens)
            source = "rules_fallback" if output is None else "trainer_llm"
        if output is None:
            output = rules(context, training_defaults(self.config.dataset))
            if previous and error:
                output["reason"] = "反馈请求未得到有效方案，结束本批搜索"
        plan = {"id": key, "source": source, "fallback_reason": error, "output": output}
        self.ledger.put("trainer_decision", key, plan)
        return plan


BearingTrainer = RulTrainer  # 保留既有 B 实验脚本的导入路径。
