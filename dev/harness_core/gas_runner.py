"""气体滑动窗口 Harness 的运行入口、数据核验和结果汇总。"""

import hashlib
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .config import project_root
from .gas_data import GAS_DATA_DIR, load_gas_model, open_gas_batches
from .gas_policy import (
    WINDOW_DRIFT_BATCHES,
    WINDOW_POLICY,
    WINDOW_TARGET_BATCHES,
    WINDOW_TRAINER_CONFIG,
    WINDOW_VALIDATION_BATCHES,
)
from .gas_workflow import checked_path, run_trainer_batches
from .llm_client import request_usage_known
from .portability import saved_data_path
from .storage import Ledger, RunLock, atomic_json, read_json
from .tracking import Tracker


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_manifest(bundle):
    report = read_json(bundle / "training.json")
    info = read_json(bundle / "baseline-info.json")
    files = {
        "checkpoint_sha256": bundle / "checkpoint.npz",
        "train_sha256": bundle / "public/train.jsonl",
        "validation_sha256": bundle / "public/validation.jsonl",
    }
    hashes = {key: digest(path) for key, path in files.items()}
    if info["dataset_key"] != "A" or report["profile"] != "A" or report["task"] != "classification":
        raise ValueError("基线包必须是官方气体分类任务 A")
    if info["checkpoint_sha256"] != hashes["checkpoint_sha256"] or any(
        report[key] != value for key, value in hashes.items()
    ):
        raise ValueError("官方检查点或公开数据摘要与原训练报告不一致")
    return {
        "baseline_id": info["id"], **hashes,
        "train_rows": report["train_rows"], "validation_rows": report["validation_rows"],
        "baseline_validation": report["validation_after"],
        "baseline_steps": report["final_steps"], "baseline_epochs": report["final_epochs"],
    }


def batch_comparison_columns(row):
    old_f1, new_f1 = row["before_test_macro_f1"], row["test_macro_f1"]
    old_accuracy, new_accuracy = row["before_test_accuracy"], row["test_accuracy"]
    updated = row["before_model_path"] != row["model_path"]
    decision = "采用新模型" if updated else "沿用旧模型"
    return (
        row["source_file"], str(row["test_samples"]),
        f"{old_f1:.4f} → {new_f1:.4f}（{new_f1 - old_f1:+.4f}）",
        f"{old_accuracy:.2%} → {new_accuracy:.2%}（{100 * (new_accuracy - old_accuracy):+.2f} 个百分点）",
        decision,
    )


def write_batch_comparison(path, batches):
    lines = [
        "# 逐批留出测试对比",
        "",
        "每行在同一批留出测试集上评估批次开始时的旧模型和批次结束时实际采用的模型。",
        "",
        "| 批次 | 测试样本 | 测试 Macro-F1（旧 → 批末） | 测试准确率（旧 → 批末） | 批末决定 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend("| " + " | ".join(batch_comparison_columns(row)) + " |" for row in batches)
    temp = path.with_suffix(".md.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(path)



def batch_decision_label(row):
    selection = f"采用 {row['selected_trial_id']}" if row["promoted"] else "保留批初模型"
    if row.get("selected_below_target"):
        selection = f"未达目标，采用通过历史门槛且优于批初模型的 {row['selected_trial_id']}"
    outcome = {
        "NO_UPDATE": "保留现用模型",
        "trainer_candidate_adopted": "采用候选", "trainer_candidate_rejected": "候选未通过验证门槛",
        "experiment_budget": "实验预算不足", "training_budget": "样本预算不足",
        "training_failed": "候选训练失败", "interrupted": "候选训练中断",
        "candidate_limit": "达到每批候选上限", "trainer_stop": "训练师停止尝试",
        "feedback_fallback": "反馈请求失败，结束本批",
    }.get(row["stop_reason"], row["stop_reason"])
    return f"{row['experiment_count']} 次训练，{outcome}，{selection}"


def write_batch_results(path, batches):
    lines = [
        "# 逐批回放结果", "",
        "| 批次 | 批初验证 Macro-F1 | 批初验证准确率 | 批次决定 | 批末验证 Macro-F1 | 批末验证准确率 | 留出测试 Macro-F1 | 留出测试准确率 | 特征 PSI / 漂移 |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in batches:
        label = Path(row["source_file"]).stem
        if row["test_protocol"] == "development_holdout_previously_public":
            label += "（开发留出）"
        drift = row["feature_drift"]
        drift_label = "无历史参考" if drift["mean_psi"] is None else (
            f"{drift['mean_psi']:.4f} / {'是' if drift['detected'] else '否'}")
        values = [
            label, f"{row['before_validation_macro_f1']:.2%}", f"{row['before_validation_accuracy']:.2%}",
            batch_decision_label(row), f"{row['validation_macro_f1']:.2%}", f"{row['validation_accuracy']:.2%}",
            f"{row['test_macro_f1']:.2%}", f"{row['test_accuracy']:.2%}", drift_label,
        ]
        lines.append("| " + " | ".join(values) + " |")
    report = "\n".join(lines) + "\n"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)
    return report


def replay_batches(data, manifest):
    return tuple(i for i in data.batch_indices if i > 0)


def write_report(root, config, ledger, manifest, active, *, data):
    trials, batches = ledger.items("trial"), ledger.items("batch")
    requests, plans = ledger.requests(), ledger.items("trainer_decision")
    policy = manifest["gas_trainer_policy"]
    summary = {
        "run_id": root.name, "dataset": "gas_mlp", "baseline_id": manifest["baseline_id"],
        "strategy": config.strategy, "provider": config.provider if config.strategy == "llm" else None,
        "complete": len(batches) == len(replay_batches(data, manifest)),
        "experiments": len(trials), "accepted_models": sum(row["promoted"] for row in batches),
        "llm_proposals": sum(t["source"] in {"deepseek", "pytrio"} for t in trials),
        "fallback_experiments": sum(t["source"] == "rules_fallback" for t in trials),
        "final_checkpoint": active["model_path"], "final_checkpoint_sha256": active["checkpoint_sha256"],
        "training_seconds": sum(t["training_seconds"] for t in trials),
        "api_calls": len(requests),
        "requests_with_unknown_usage": sum(not request_usage_known(r) for r in requests),
        "failed_stages": sum(t["status"] == "failed" for t in trials),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in requests),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in requests),
        "tracking_errors": ledger.items("tracking"),
        "data_dir": config.data_dir, "dataset_id": manifest["raw_gas"]["dataset_id"],
        "batches_completed": len(batches), "target_batches": len(replay_batches(data, manifest)),
        "batch_selection_policy": manifest["batch_selection_policy"],
        "evaluation_protocol": "local_holdout", "strict_blind_test": False,
        "gas_trainer_policy": policy["version"],
        "max_candidates_per_batch": policy["max_candidates_per_batch"],
        "gas_trainer_min_validation_f1": policy.get("min_validation_macro_f1"),
        "trainer_decisions": len(plans),
        "llm_decisions": sum(p["metadata"]["source"] == "trainer_llm" for p in plans),
        "fallback_decisions": sum(p["metadata"]["fallback"] for p in plans),
        "training_sample_passes": sum(t.get("sample_passes", 0) for t in trials),
        "reserved_training_sample_passes": sum(r["sample_passes"] for r in ledger.items("training_reservation")),
        "gas_trainer_max_history_drop": policy["max_history_drop"],
        "development_holdout_batches": [
            row["source_file"] for row in batches if row["test_protocol"] == "development_holdout_previously_public"
        ],
    }
    if "validation_target_window" in policy:
        summary["gas_trainer_validation_target_window"] = policy["validation_target_window"]
    if batches:
        for name in ("macro_f1", "accuracy"):
            summary[f"mean_before_test_{name}"] = sum(b[f"before_test_{name}"] for b in batches) / len(batches)
            summary[f"mean_test_{name}"] = sum(b[f"test_{name}"] for b in batches) / len(batches)
        summary["worst_before_test_macro_f1"] = min(b["before_test_macro_f1"] for b in batches)
        summary["worst_test_macro_f1"] = min(b["test_macro_f1"] for b in batches)
    drift = [row["feature_drift"] for row in batches if row["feature_drift"]["mean_psi"] is not None]
    summary.update(
        gas_drift_policy=manifest["gas_drift_policy"],
        drift_no_reference_batches=len(batches) - len(drift),
        drift_profiled_batches=len(drift),
        drift_detected_batches=sum(item["detected"] for item in drift),
    )
    if drift:
        summary["mean_feature_psi"] = sum(item["mean_psi"] for item in drift) / len(drift)
    measured = [b for b in batches if b["profile_complete"]]
    summary["profiled_batches"] = len(measured)
    if measured:
        summary["total_decision_seconds"] = sum(b["decision_seconds"] for b in measured)
        summary["total_batch_cpu_seconds"] = sum(b["cpu_seconds"] for b in measured)
        summary["peak_batch_rss_mib"] = max(b["peak_rss_mib"] for b in measured)
    atomic_json(root / "audit_test_results.json", batches)
    write_batch_comparison(root / "batch_comparison.md", batches)
    if batches:
        print(write_batch_results(root / "batch_results.md", batches), end="", flush=True)
    atomic_json(root / "history.json", {
        "proposals": ledger.items("proposal"), "trials": trials, "decisions": ledger.items("decision"),
    })
    atomic_json(root / "api_requests.json", requests)
    atomic_json(root / "summary.json", summary)
    return summary


def run_gas_mlp(config, *, runs_root=None, resume=None, transport=None, stop_after=None, persist_config=False):
    root = Path(resume).resolve() if resume else None
    saved_manifest = read_json(root / "dataset.json") if root is not None else None
    saved_policy = (saved_manifest or {}).get("gas_trainer_policy", {})
    if saved_manifest is not None and saved_policy.get("version") != WINDOW_POLICY:
        raise ValueError("当前只恢复滑动窗口主线，旧版运行可用 status 查看")
    input_root = (project_root() / Path(config.data_dir or GAS_DATA_DIR).expanduser()).resolve()
    if (input_root / "baseline-info.json").is_file():
        raise ValueError("主线需要原始气体批次或导入目录，请用 --baseline-dir 指定初始模型包")
    bundle = (project_root() / (config.baseline_dir or "baseline/gas")).resolve()
    model_meta = load_gas_model(bundle / "checkpoint.npz").meta
    data = open_gas_batches(input_root, config.seed, model_meta=model_meta)
    manifest = baseline_manifest(bundle)
    manifest["raw_gas"] = data.manifest
    manifest["gas_schema"] = {"feature_count": len(model_meta["paths"]), "class_labels": model_meta["classes"]}
    policy = {
        "version": WINDOW_POLICY, "drift": WINDOW_TRAINER_CONFIG["drift"],
        "training_defaults": WINDOW_TRAINER_CONFIG["training"],
        "training_limits": WINDOW_TRAINER_CONFIG["limits"],
        "prompt_macro_f1": saved_policy.get("prompt_macro_f1", "checkpoint_class_set"),
        "history_gain_source": "validation_only",
        "max_candidates_per_batch": config.candidate_limit,
        "max_history_drop": saved_policy.get("max_history_drop", config.max_history_drop),
        "history_validation_batches": WINDOW_VALIDATION_BATCHES,
    }
    if "min_validation_macro_f1" in saved_policy or (
            saved_manifest is None and config.gas_trainer_min_validation_f1 is not None):
        policy["min_validation_macro_f1"] = saved_policy.get(
            "min_validation_macro_f1", config.gas_trainer_min_validation_f1)
    if "validation_target_window" in saved_policy or (
            saved_manifest is None and config.gas_trainer_min_validation_f1 is None):
        policy["validation_target_window"] = saved_policy.get("validation_target_window", WINDOW_TARGET_BATCHES)
    if saved_manifest is None or "allow_best_below_target" in saved_policy:
        policy["allow_best_below_target"] = saved_policy.get("allow_best_below_target", True)
    manifest.update(
        gas_trainer_policy=policy,
        gas_drift_policy={"method": "psi_ks_fdr_wasserstein_mmd_label_js",
                          "reference": "previous_completed_batches", "input_split": "train",
                          "history_batches": WINDOW_DRIFT_BATCHES},
        batch_selection_policy="fixed_entry_v1", reserved_public_validation=[],
    )
    if root is None:
        config.data_dir = saved_data_path(data.root, project_root())
        if config.baseline_dir:
            config.baseline_dir = saved_data_path(bundle, project_root())
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (Path(runs_root or project_root() / "runs") / f"{stamp}_gas_mlp_{config.strategy}_{uuid.uuid4().hex[:8]}").resolve()
        root.mkdir(parents=True)
        atomic_json(root / "config.json", config.to_dict())
        atomic_json(root / "dataset.json", manifest)
    elif saved_manifest != manifest:
        raise ValueError("恢复运行时官方基线或公开数据已经改变")
    print(f"运行目录：{root}", flush=True)
    with RunLock(root / "run.lock"):
        if persist_config:
            atomic_json(root / "config.json", config.to_dict())
        ledger = Ledger(root / "experiments.sqlite")
        tracker = None
        complete = False
        try:
            project = project_root()
            sources = sorted(path for folder in (project / "harness_core", project / "wrappers", bundle / "baselines")
                             for path in folder.glob("*.py"))
            code_digest = hashlib.sha256("".join(digest(path) for path in sources).encode()).hexdigest()
            tracker = Tracker(config, root, ledger, metric="macro_f1", run_metadata={
                "code_sha256": code_digest, "baseline_id": manifest["baseline_id"],
                "baseline_checkpoint_sha256": manifest["checkpoint_sha256"],
                "dataset_id": manifest["raw_gas"]["dataset_id"],
                "data_manifest_sha256": digest(root / "dataset.json"),
                "evaluation_protocol": "local_holdout_after_batch_update",
                "gas_drift_policy": manifest["gas_drift_policy"], "gas_trainer_policy": policy,
            })
            baseline = root / "models/baseline.npz"
            if not baseline.exists():
                baseline.parent.mkdir(exist_ok=True)
                temporary = baseline.with_suffix(".tmp")
                shutil.copyfile(bundle / "checkpoint.npz", temporary)
                temporary.replace(baseline)
            if digest(baseline) != manifest["checkpoint_sha256"]:
                raise ValueError("运行目录中的基线检查点摘要不匹配")
            active = {
                "model_path": "models/baseline.npz", "checkpoint_sha256": manifest["checkpoint_sha256"],
                "validation": manifest["baseline_validation"], "steps": manifest["baseline_steps"],
                "epochs": manifest["baseline_epochs"],
            }
            active = run_trainer_batches(config, bundle, root, ledger, tracker, transport, baseline,
                                         active, data, replay_batches(data, manifest), manifest, stop_after)
            checked_path(root, active)
            atomic_json(root / "active.json", active)
            summary = write_report(root, config, ledger, manifest, active, data=data)
            complete = summary["complete"]
            tracker.log_final(summary)
            print(f"完成 {summary['batches_completed']}/{summary['target_batches']} 批，"
                  f"训练 {summary['experiments']} 轮。结果：{root / 'summary.json'}", flush=True)
            return root, summary
        finally:
            if tracker is not None:
                tracker.close(complete)
            ledger.close()
