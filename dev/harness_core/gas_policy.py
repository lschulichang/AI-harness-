"""滑动窗口训练师的结构化提案、校验与规则兜底。"""

from .llm_client import OpenAITransport, Planner
from .runner import MIXED_TRAINING_WINDOW, window_size_for_action

DECISIONS = {"NO_UPDATE", "INCREMENTAL_TRAINING", "MIXED_DATA_TRAINING"}
TRAINING_ACTIONS = {"INCREMENTAL_TRAINING", "MIXED_DATA_TRAINING"}
WINDOW_POLICY = "trainer_window_v4"  # 决定训练时的数据批次数量
WINDOW_DRIFT_BATCHES = 3  # 漂移检测批次数量
WINDOW_TARGET_BATCHES = 4  # 求训练目标所用的批次数量

WINDOW_TRAINER_CONFIG = {
    "drift": {
        "bins": 10,
        "psi_warning": 0.1,  # 漂移指标PSI
        "ks_fdr": 0.05,  # KS检验的显著性水平
        "mmd_samples": 300,  # MMD样本数量
        "mmd_permutations": 99,  # MMD排列数量
        "label_js_warning": 0.05,  # 标签JS警告阈值
        "min_samples": 80,  # 待确认
    },
    "training": {  # 训练参数
        "learning_rate": 0.0003,  # 学习率
        "batch_size": 64,  # 批次大小
        "epochs": 20,  # 训练轮数
        "early_stopping_patience": 5,  # 早停patience
        "weight_decay": 0.0001,  # 权重衰减
        "class_weighting": False,  # 类别权重
        "window_size": MIXED_TRAINING_WINDOW,  # 固定混合窗口
    },
    "limits": { #给大模型设置的训练参数限制
        "learning_rate": [1e-6, 0.01],
        "batch_size": [8, 512],
        "epochs": [1, 50],
        "early_stopping_patience": [1, 15],
        "weight_decay": [0, 0.01],
    },
}

# 验证大模型返回的训练参数是否符合限制
def validate(raw, cfg, *, require_complete=False):
    if not isinstance(raw, dict) or set(raw) != {
        "decision",
        "reason",
        "training_config",
        "evaluation_requirements",
    }:
        raise ValueError("decision must have exactly the required JSON keys")
    if (
        not isinstance(raw["decision"], str)
        or raw["decision"] not in DECISIONS
        or not isinstance(raw["reason"], str)
        or not raw["reason"].strip()
    ):
        raise ValueError("invalid decision or reason")
    requirements = raw["evaluation_requirements"]
    if not isinstance(requirements, list) or not all(
        isinstance(value, str) for value in requirements
    ):
        raise ValueError("invalid evaluation requirements")
    supplied = raw["training_config"]
    training_keys = cfg["training"].keys()
    if not isinstance(supplied, dict) or set(supplied) - training_keys:
        raise ValueError("invalid training configuration keys")
    if raw["decision"] not in TRAINING_ACTIONS:
        if supplied:
            raise ValueError("nontraining decision must have empty training_config")
        return raw
    if require_complete and set(supplied) - {"window_size"} != training_keys - {"window_size"}:
        raise ValueError(
            "training_config must explicitly contain every training parameter"
        )
    config = {**cfg["training"], **supplied}
    # Older model replies may still include a width. It never controls replay.
    if "window_size" in supplied and (type(supplied["window_size"]) is not int
                                     or not 1 <= supplied["window_size"] <= 8):
        raise ValueError("invalid legacy window_size")
    for key, (lower, upper) in cfg["limits"].items():
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not lower <= value <= upper
        ):
            raise ValueError(f"{key} is outside configured bounds")
    for key in ("batch_size", "epochs", "early_stopping_patience"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if type(config["class_weighting"]) is not bool:
        raise ValueError("class_weighting must be boolean")
    config["window_size"] = window_size_for_action(raw["decision"])
    raw["training_config"] = config
    return raw


def rules(context, cfg): #兜底规则
    current = context["performance"]["current"]
    target = context.get("validation_target")
    stable = target is not None and current["macro_f1"] >= target
    if context["data"]["new_labeled_samples"] < cfg["drift"]["min_samples"]:
        decision, reason = (
            "NO_UPDATE",
            "Insufficient labeled samples; keep the current model",
        )
    elif stable and len(context["drift_report"]["drifted_features"]) < 20:
        decision, reason = "NO_UPDATE", "Current performance is stable"
    elif context["data"]["historical_replay_available"]:
        decision, reason = (
            "MIXED_DATA_TRAINING",
            "Drift or performance decline; historical replay exists",
        )
    else:
        decision, reason = (
            "INCREMENTAL_TRAINING",
            "Drift or performance decline; replay unavailable",
        )
    return validate(
        {
            "decision": decision,
            "reason": reason,
            "training_config": dict(cfg["training"])
            if decision in TRAINING_ACTIONS
            else {},
            "evaluation_requirements": [
                "new_data_macro_f1",
                "per_class_recall",
            ],
        },
        cfg,
    )


WINDOW_PROMPT = ( #大模型训练参数搜索提示词
    "/no_think Return only concise JSON with exactly decision, reason, training_config, "
    "evaluation_requirements. Choose NO_UPDATE, INCREMENTAL_TRAINING or "
    "MIXED_DATA_TRAINING. Use training_config={} when skipping. For a candidate explicitly return "
    "learning_rate, batch_size, epochs, early_stopping_patience, weight_decay and class_weighting "
    "within training_limits. Omit window_size; the harness sets it from the action. "
    f"Incremental training uses the current batch. Mixed training uses the current batch and up to {MIXED_TRAINING_WINDOW - 1} newest historical batches. "
    "Do not choose mixed training without any replay batches. evaluation_requirements must be an array of "
    "strings. Missing drift reference or labels mean the corresponding evidence is unavailable. "
    "Use validation evidence only. Do not claim held-out results or output code or commands."
)

#计算实际的window_size
def effective_params(params, replay_count):
    return {**params, "window_size": min(params["window_size"], replay_count + 1)}

#整理发给大模型的输入
def compact_input(context, cfg):
    def brief(metrics):#简要的性能指标
        if metrics is None:
            return None
        return {
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "recall": {
                name: round(value["recall"], 3)
                for name, value in metrics["per_class"].items()
            },
        }

    drift = context["drift_report"]
    return {
        "drift": {#漂移指标
            "feature_count": len(drift["drifted_features"]),
            "top_feature_indices": drift["drifted_features"][:8],
            "mmd_p": (drift["mmd"] or {}).get("permutation_p"),
            "mmd_squared": (drift["mmd"] or {}).get("mmd_squared"),
            "label_js": drift["label_js_divergence"],
            "label_status": drift.get("label_status"),
            "status": drift["status"],
            "reference_batches": drift["reference_batches"],
        },
        "current_validation": brief(context["performance"]["current"]),#旧模型在新数据上的性能指标
        "new_rows": context["data"]["new_labeled_samples"],#训练集和验证集的新样本数量
        "replay_available": context["data"]["historical_replay_available"],#是否存在历史回放数据
        "replay_batches": context["data"]["replay_batches"],#历史回放数据
        "recent_history": [#最近3批的训练参数和性能指标
            {
                "batch": row["batch"],
                "decision": row["decision"],
                "deployed": row["deployed"],
                "macro_f1_gain": round(row["macro_f1_gain"], 4),
                "drifted_feature_count": row["drifted_feature_count"],
            }
            for row in context["history"][:3]
        ],
        "training_defaults": {key: value for key, value in cfg["training"].items() if key != "window_size"},
        "training_limits": cfg["limits"],
    }


class GasTrainer(Planner):
    def __init__(self, config, ledger, transport=None):
        if transport is None and config.strategy == "llm":
            transport = OpenAITransport(config, temperature=0)
        super().__init__(config, ledger, transport)

    def decide(self, key, context, cfg, *, feedback=None):
        saved = self.ledger.get("trainer_decision", key)
        if saved is not None:
            return saved
        payload = compact_input(context, cfg)
        prompt = WINDOW_PROMPT
        attempts = []
        if feedback is not None:
            payload["batch_feedback"] = feedback
            attempts = feedback["attempts"]
            prompt += (
                " Each decision requests at most one candidate. All candidates start from the same batch-entry model. "
                "Review batch_feedback for actual validation, gate failures and remaining budgets. "
                "If another distinct configuration is worth trying, return a training decision with changed parameters. "
                "Otherwise return NO_UPDATE to finish; the best eligible candidate so far will still be adopted."
            )
            if feedback.get("allow_best_below_target"):
                prompt += (
                    " If no candidate reaches the validation target, the best below-target candidate "
                    "may be selected at the end if it improves over "
                    "the batch-entry model. Use min_current_validation_macro_f1 as the priority target."
                )
        decision, error = None, ""
        if self.config.strategy == "llm":

            def check(value):
                decision = validate(value, cfg, require_complete=True)
                if (
                    decision["decision"] == "MIXED_DATA_TRAINING"
                    and not payload["replay_available"]
                ):
                    raise ValueError("mixed training requires replay samples")
                previous_count = len(payload.get("replay_batches", ()))
                if decision["decision"] in TRAINING_ACTIONS and any(
                    effective_params(decision["training_config"], previous_count)
                    == effective_params(attempt["training_config"], previous_count)
                    for attempt in attempts
                ):
                    raise ValueError(
                        "this training configuration has already been tried in the current batch"
                    )
                return decision

            decision, error = self._request_proposal(
                key,
                prompt,
                payload,
                check,
                output_budget=self.config.max_api_output_tokens,
            )
        if decision is None:
            if attempts:
                decision = {
                    "decision": "NO_UPDATE",
                    "reason": "反馈请求未得到有效方案，结束本批搜索",
                    "training_config": {},
                    "evaluation_requirements": [],
                }
            else:
                decision = rules(context, cfg)
            source = "rules_fallback" if self.config.strategy == "llm" else "rules"
        else:
            source = "trainer_llm"
        result = {
            "input": payload,
            "output": decision,
            "metadata": {
                "source": source,
                "fallback": source == "rules_fallback",
                "error": error or None,
            },
        }
        self.ledger.put("trainer_decision", key, result)
        return result
