"""运行配置：阈值和预算都在这里集中管理。"""

from dataclasses import asdict, dataclass


@dataclass
class RunConfig:
    scenario: str = "stream"
    strategy: str = "llm"
    provider: str = "host_gateway_rpc"
    model: str | None = None
    swanlab_mode: str = "offline"
    swanlab_project: str = "AI+harness"
    swanlab_workspace: str | None = None
    seed: int = 42
    trials_per_batch: int | None = None
    max_experiments: int | None = None
    max_api_calls: int | None = None
    gas_sample_budget: int | None = None
    max_api_output_tokens: int | None = None
    api_timeout: float | None = None
    train_timeout: float | None = None
    max_output_tokens: int | None = None
    min_gain: float = 0.01
    evaluate_history: bool = False
    max_history_drop: float = 0.20  # Retained for saved options and reports.
    gas_trainer_min_validation_f1: float | None = None
    rul_min_relative_gain: float = 0.01
    rul_max_history_increase: float = 0.50  # Legacy configuration field.
    dataset: str = "gas_mlp"

    def validate(self):
        if type(self.evaluate_history) is not bool:
            raise ValueError("evaluate_history 必须是布尔值")
        if self.scenario != "stream" or self.dataset not in {"gas_mlp", "bearing_rul", "cmapss_rul"}:
            raise ValueError("当前 Harness 只支持 gas_mlp、bearing_rul 和 cmapss_rul 的 stream 回放")
        if self.dataset in {"bearing_rul", "cmapss_rul"}:
            if self.gas_sample_budget is not None or self.gas_trainer_min_validation_f1 is not None:
                raise ValueError("气体样本预算和 Macro-F1 阈值仅适用于 gas_mlp")
            for name in ("rul_min_relative_gain", "rul_max_history_increase"):
                value = getattr(self, name)
                if type(value) not in (float, int) or not 0 <= value <= 1:
                    raise ValueError(f"{name} 必须是 0 到 1 的比例")
        if self.strategy not in {"rules", "llm"}:
            raise ValueError("strategy 只能是 llm 或 rules；旧对照入口已移除")
        if self.swanlab_mode not in {"offline", "disabled"}:
            raise ValueError("SwanLab 模式无效")
        if self.trials_per_batch is not None and (type(self.trials_per_batch) is not int or self.trials_per_batch < 1):
            raise ValueError("单批实验预算无效")
        if self.max_api_calls is not None and (type(self.max_api_calls) is not int or self.max_api_calls < 0):
            raise ValueError("API 预算无效")
        if self.max_experiments is not None and (type(self.max_experiments) is not int or self.max_experiments < 0):
            raise ValueError("实验预算无效")
        for name in ("gas_sample_budget", "max_api_output_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} 必须为非负整数或 null")
        for timeout in (self.api_timeout, self.train_timeout):
            if timeout is not None and (type(timeout) not in (int, float) or not 0 < timeout < float("inf")):
                raise ValueError("超时必须为正的有限秒数或 null")
        if self.max_output_tokens is not None and (type(self.max_output_tokens) is not int or self.max_output_tokens < 1):
            raise ValueError("max_output_tokens 必须为正整数或 null")
        trainer_target = self.gas_trainer_min_validation_f1
        if trainer_target is not None and (type(trainer_target) not in (int, float) or not 0 < trainer_target <= 1):
            raise ValueError("气体 trainer 的最低验证 Macro-F1 应大于 0 且不超过 1，或设为 null")
        if any(type(value) not in (int, float) or not 0 <= value <= 1
               for value in (self.min_gain, self.max_history_drop)):
            raise ValueError("阈值应使用 0 到 1 的比例")
        return self

    @property
    def candidate_limit(self):
        return (self.trials_per_batch or 3) if self.strategy == "llm" else 1

    def to_dict(self):
        return asdict(self)
