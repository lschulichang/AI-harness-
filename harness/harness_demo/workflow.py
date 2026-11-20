"""校验运行配置，分发 A 气体分类及 B、C 寿命回放。"""

from pathlib import Path

from .config import RunConfig
from .credentials import get_secret
from .storage import read_json


def training_window(batch, completed, window_size):
    if type(window_size) is not int or not 1 <= window_size <= 8:
        raise ValueError("window_size must be an integer from 1 to 8")
    previous_count = window_size - 1
    return [*(completed[-previous_count:] if previous_count else ()), batch]


def run_workflow(config=None, *, runs_root=None, resume=None, transport=None, stop_after=None, no_api_call_limit=False):
    if resume:
        root = Path(resume).resolve()
        try:
            options = read_json(root / "config.json")
        except FileNotFoundError as exc:
            raise ValueError("运行缺少配置或使用旧版格式，已有结果可用 status 查看") from exc
        manifest = read_json(root / "dataset.json")
        dataset = options.get("dataset", "gas_mlp")
        if dataset in {"bearing_rul", "cmapss_rul"}:
            from .bearing_trainer_workflow import RUL_POLICIES

            key, version = RUL_POLICIES[dataset]
            if manifest.get(key, {}).get("version") != version:
                raise ValueError("当前只恢复对应任务的滑窗运行，旧版运行可用 status 查看")
        else:
            from .gas_trainer_policy import WINDOW_POLICY

            if manifest.get("gas_trainer_policy", {}).get("version") != WINDOW_POLICY:
                raise ValueError("当前只恢复滑动窗口主线，旧版运行可用 status 查看")
            options.setdefault("api_timeout", 60.0)
            options.setdefault("train_timeout", 60.0)
        config = RunConfig.from_saved_options(options).validate()
    else:
        config = (config or RunConfig()).validate()
    if no_api_call_limit:
        config.max_api_calls = None
    if config.strategy == "llm" and transport is None and not get_secret(config.api_key_env):
        raise ValueError(f"尚未配置 {config.api_key_env}。请在环境变量或 local_secrets.py 中设置后再运行 llm。")

    if config.dataset in {"bearing_rul", "cmapss_rul"}:
        from .bearing_trainer_workflow import run_rul as run_dataset
    else:
        from .gas_mlp import run_gas_mlp as run_dataset

    return run_dataset(config, runs_root=runs_root, resume=resume, transport=transport,
                       stop_after=stop_after, persist_config=no_api_call_limit)
