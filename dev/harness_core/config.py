"""运行配置：阈值和预算都在这里集中管理。"""

from dataclasses import asdict, dataclass
from pathlib import Path

PROVIDER_DEFAULTS = {
    "pytrio": ("Qwen/Qwen3.8-27B", "https://pytrio.com/api/openai/v1", "PYTRIO_API_KEY"),
}


@dataclass
class RunConfig:
    scenario: str = "stream"
    strategy: str = "llm"
    provider: str = "pytrio"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    swanlab_mode: str = "online"
    swanlab_project: str = "AI+harness"
    swanlab_workspace: str | None = "ls_chulichang"
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
    max_history_drop: float = 0.20
    gas_trainer_min_validation_f1: float | None = None
    rul_min_relative_gain: float = 0.01
    rul_max_history_increase: float = 0.50
    competition: bool = False
    data_dir: str = ""
    dataset: str = "gas_mlp"
    baseline_dir: str = ""

    @classmethod
    def from_saved_options(cls, options):
        """读取主线配置，并兼容已保存的滑动窗口运行快照。"""
        options = dict(options)
        if options.pop("gas_policy", None) not in (None, "trainer"):
            raise ValueError("旧搜索策略已移除，历史结果可用 status 查看")
        for field in (
            "drift_drop", "gas_sensor_groups", "gas_target_macro_f1",
            "gas_max_extension_stages", "depths", "leaves", "windows",
        ):
            options.pop(field, None)
        try:
            return cls(**options)
        except TypeError as exc:
            raise ValueError(f"运行配置包含无法识别的字段：{exc}") from exc

    def __post_init__(self):
        defaults = PROVIDER_DEFAULTS.get(self.provider)
        if defaults is not None:
            for field, value in zip(("model", "base_url", "api_key_env"), defaults):
                if getattr(self, field) is None:
                    setattr(self, field, value)

    def validate(self):
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
        if self.provider not in PROVIDER_DEFAULTS:
            raise ValueError("provider 只支持 PyTRiO")
        if self.swanlab_mode not in {"offline", "online", "disabled"}:
            raise ValueError("SwanLab 模式无效")
        if self.competition:
            raise ValueError("官方 SDK 尚未接入：练习入口不能作为正式比赛入口。请先实现官方数据适配器。")
        if not self.base_url or not self.api_key_env:
            raise ValueError("PyTRiO 需要配置 API 地址和密钥环境变量")
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


def project_root():
    return Path(__file__).resolve().parent.parent
