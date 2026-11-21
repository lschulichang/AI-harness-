"""从项目根目录运行 python -m harness_core --help。"""

import argparse
import json
import sys
from pathlib import Path

from .config import RunConfig
from .runner import run_workflow
from .storage import read_json


def parser():
    p = argparse.ArgumentParser(description="智训引擎 Harness：工业数据的批次实验闭环",
                                epilog="气体格式识别与 DAT 转换：python -m harness_core.gas_import --help")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="运行或恢复实验")
    run.add_argument("--config", type=Path)
    run.add_argument("--dataset", choices=("gas_mlp", "bearing_rul", "cmapss_rul"))
    run.add_argument("--data-dir", help="项目 data、原始数据目录或官方基线包目录，支持项目相对路径")
    run.add_argument("--baseline-dir", help="官方基线包目录，默认自动匹配任务")
    run.add_argument("--scenario", choices=["stream"])
    run.add_argument("--strategy", choices=["rules", "llm"])
    run.add_argument("--provider", choices=["pytrio"])
    run.add_argument("--model")
    run.add_argument("--base-url")
    run.add_argument("--api-key-env")
    run.add_argument("--swanlab-mode", choices=["offline", "online", "disabled"])
    api_budget = run.add_mutually_exclusive_group()
    api_budget.add_argument("--max-api-calls", type=int)
    api_budget.add_argument("--no-api-call-limit", action="store_true", help="取消整轮 API 请求上限，也可用于恢复运行")
    run.add_argument("--trials-per-batch", type=int, help="每批训练阶段上限；大模型反馈策略默认 3 次")
    run.add_argument("--gas-sample-budget", type=int, help="气体本地回放的可选样本遍历量上限，默认不限")
    run.add_argument("--max-api-output-tokens", type=int, help="整轮模型提案的可选输出 Token 上限，默认不限")
    budget = run.add_mutually_exclusive_group()
    budget.add_argument("--max-experiments", type=int)
    budget.add_argument("--no-experiment-limit", action="store_true", help="取消显式设置的整轮候选次数上限")
    run.add_argument("--resume", type=Path)
    run.add_argument("--stop-after-trials", type=int)
    run.add_argument("--runs-root", type=Path)
    status = sub.add_parser("status", help="查看运行结果")
    status.add_argument("run_dir", type=Path)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.command == "run":
            if args.resume:
                explicit = [k for k in ("config", "dataset", "data_dir", "baseline_dir", "scenario", "strategy", "provider", "model", "base_url", "api_key_env", "swanlab_mode", "max_api_calls", "max_experiments", "trials_per_batch", "gas_sample_budget", "max_api_output_tokens") if getattr(args, k) is not None]
                if args.no_experiment_limit:
                    explicit.append("no_experiment_limit")
                if explicit:
                    raise ValueError("恢复时沿用原配置，请勿同时传入：" + ", ".join(explicit))
                run_workflow(resume=args.resume, stop_after=args.stop_after_trials, no_api_call_limit=args.no_api_call_limit)
            else:
                options = read_json(args.config) if args.config else {}
                if args.provider is not None and args.provider != options.get("provider", RunConfig.provider):
                    for field in ("model", "base_url", "api_key_env"):
                        options.pop(field, None)
                for field in ("dataset", "data_dir", "baseline_dir", "scenario", "strategy", "provider", "model", "base_url", "api_key_env", "swanlab_mode", "max_api_calls", "max_experiments", "trials_per_batch", "gas_sample_budget", "max_api_output_tokens"):
                    value = getattr(args, field)
                    if value is not None:
                        options[field] = value
                if args.no_experiment_limit:
                    options["max_experiments"] = None
                run_workflow(RunConfig.from_saved_options(options), runs_root=args.runs_root, stop_after=args.stop_after_trials, no_api_call_limit=args.no_api_call_limit)
        elif args.command == "status":
            print(json.dumps(read_json(args.run_dir/"summary.json"), ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"运行未完成：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
