"""回放 B、C 公开数据，按当前与历史 RMSE 门槛选模。"""
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
from threadpoolctl import threadpool_limits
from wrappers.bearing_baseline import PACKAGE, BearingBaseline, digest
from wrappers.bearing_pairs import REPLAY_DIR, PairedBearingBaseline
from wrappers.cmapss_baseline import PACKAGE as C_PACKAGE
from wrappers.cmapss_baseline import RAW_DATA, CmapssBaseline, test_scores

from .config import project_root
from .feature_drift import detect_features
from .llm_client import request_usage_known
from .history_evaluation import HISTORY_WINDOW, history_batches, rul_references, rul_scores
from .portability import resolve_model_path, saved_data_path
from .rul_policy import LIMITS, TRAINING_ACTIONS, RulTrainer, training_defaults
from .runner import candidate_feedback, training_window
from .storage import Ledger, RunLock, atomic_json, read_json
from .tracking import Tracker

POLICY = "bearing_window_v2"
RUL_POLICIES = {
    "bearing_rul": ("bearing_trainer_policy", POLICY),
    "cmapss_rul": ("cmapss_trainer_policy", "cmapss_window_v1"),
}
DRIFT_WINDOW = 3
VALIDATION_TARGET_WINDOW = 4
DRIFT = {"bins": 10, "psi_warning": 0.1, "ks_fdr": 0.05, "mmd_samples": 300, "mmd_permutations": 99}


def checkpoint(root, path, model):
    return {"model_path": path.relative_to(root).as_posix(), "checkpoint_sha256": digest(path),
            "steps": model.meta["steps"], "epochs": model.meta["epochs"]}


def checked_path(root, record, adapter):
    path = resolve_model_path(root, record["model_path"])
    if digest(path) != record["checkpoint_sha256"]:
        raise ValueError("RUL 检查点摘要不匹配")
    model = adapter.load_model(path)
    if (model.meta["steps"], model.meta["epochs"]) != (record["steps"], record["epochs"]):
        raise ValueError("RUL 检查点步数与运行记录不一致")
    return path


def gate_failures(config, before, candidate, *, target=None):
    failures = []
    if candidate >= before - 1e-12 or (
            target is None and candidate > before * (1 - config.rul_min_relative_gain) + 1e-12):
        failures.append("current_gain_insufficient")
    if target is not None and candidate > target + 1e-12:
        failures.append("validation_target_not_met")
    return failures


def choose_candidate(attempts):
    successful = [trial for trial in attempts if trial["status"] == "success"]
    preferred = [trial for trial in successful if not trial["gate_failures"]]
    fallback = [trial for trial in successful if trial["gate_failures"] == ["validation_target_not_met"]]
    return min(preferred or fallback, key=lambda trial: trial["rmse"], default=None)


def train_candidate(config, adapter, parent, output, key, params, inputs):
    env = {
        name: value for name, value in os.environ.items()
        if name not in {config.api_key_env, "PYTHONPATH"} and not name.endswith("_API_KEY")
    }
    env.update(BASELINE_CPU_THREADS="1", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-B", "-m", "baselines.train", "--profile", adapter.profile,
               "--train", str(inputs[0]), "--validation", str(inputs[1]), "--resume", str(parent),
               "--output", str(output), "--stage-id", key, "--learning-rate", str(params["learning_rate"]),
               "--batch-size", str(params["batch_size"]), "--epochs", str(params["epochs"])]
    if "early_stopping_patience" in params:
        command.extend(["--early-stopping-patience", str(params["early_stopping_patience"])])
    result = subprocess.run(command, cwd=adapter.bundle, env=env, capture_output=True,
                            encoding="utf-8", errors="replace", timeout=config.train_timeout)
    if result.returncode:
        raise RuntimeError("RUL 基线训练失败：" + result.stderr[-1500:])


def candidate_metrics(adapter, root, output, parent, parent_path, params, inputs, key, train_rows, validation):
    path = output / "checkpoint.npz"
    report = read_json(output / "training.json")
    model = adapter.load_model(path)
    artifact = checkpoint(root, path, model)
    requested_epochs = params["epochs"]
    patience = params.get("early_stopping_patience")
    epochs_run = report.get("epochs_run", requested_epochs if patience is None else None)
    best_epoch = report.get("best_epoch", requested_epochs if patience is None else None)
    if (type(epochs_run) is not int or type(best_epoch) is not int
            or not 1 <= best_epoch <= epochs_run <= requested_epochs
            or report.get("early_stopping_patience") != patience
            or (patience is None and best_epoch != requested_epochs)):
        raise ValueError("RUL 检查点的早停轮次与训练方案不一致")
    steps = (train_rows + params["batch_size"] - 1) // params["batch_size"]
    expected = {
        "stage_id": key, "profile": adapter.profile, "task": "regression",
        "parent_checkpoint_sha256": digest(parent_path), "checkpoint_sha256": artifact["checkpoint_sha256"],
        "train_sha256": digest(inputs[0]), "validation_sha256": digest(inputs[1]),
        "train_rows": train_rows, "validation_rows": len(validation["y"]),
        "learning_rate": params["learning_rate"], "batch_size": params["batch_size"],
        "initial_steps": parent.meta["steps"], "initial_epochs": parent.meta["epochs"],
        "final_steps": parent.meta["steps"] + steps * best_epoch,
        "final_epochs": parent.meta["epochs"] + best_epoch,
    }
    if adapter.profile == "B":
        expected["target_unit"] = adapter.unit
    if (any(report.get(name) != value for name, value in expected.items())
            or artifact["steps"] != expected["final_steps"] or artifact["epochs"] != expected["final_epochs"]
            or not np.array_equal(model.mean, parent.mean) or not np.array_equal(model.scale, parent.scale)
            or any(model.meta[name] != parent.meta[name] for name in ("target_mean", "target_scale", "seed"))):
        raise ValueError("RUL 候选与父检查点、训练输入或缩放状态不一致")
    if not any(not np.array_equal(model.params[name], parent.params[name]) for name in parent.params):
        raise ValueError("RUL 候选没有更新模型权重")
    return model, artifact, {**report, "epochs_run": epochs_run, "best_epoch": best_epoch}


def run_candidate(config, adapter, root, ledger, tracker, batch, plan, window, entry, before, history, target,
                  *, measure_candidate=None):
    key = plan["id"]
    trial = ledger.get("trial", key)
    if trial is not None:
        if trial["status"] == "success":
            checked_path(root, trial["artifact"], adapter)
        return trial
    params = plan["output"]["training_config"]
    names = [adapter.group_names[index] for index in window]
    train_rows = len(adapter.training_rows(window))
    sample_passes = train_rows * params["epochs"]
    reservation = ledger.get("training_reservation", key)
    if reservation is None:
        ledger.put("training_reservation", key, {"id": key, "batch": batch, "sample_passes": sample_passes})
    directory = root / "experiments" / key
    output = directory / "result"
    trial = {"id": key, "batch": batch, "source": config.provider if plan["source"] == "trainer_llm" else plan["source"],
             "stage": "trainer", "status": "failed", "params": params, "training_batches": names,
             "effective_config": {**params, "window_size": len(window)}, "train_rows": train_rows,
             "parent_checkpoint_sha256": entry["checkpoint_sha256"], "training_seconds": 0.0, "sample_passes": 0}
    ledger.put("proposal", key, {"source": trial["source"], "params": params,
               "hypothesis": plan["output"]["reason"], "fallback_reason": plan["fallback_reason"]})
    decision = {"id": key, "accepted": False, "eligible": False, "scope": "candidate_eligibility",
                "reason": "training_failed", "selected": entry}
    started = time.perf_counter()
    try:
        if reservation is not None and not output.is_dir():
            decision["reason"] = "interrupted"
            raise RuntimeError("上次尝试中断且没有完整候选")
        inputs = adapter.write_inputs(batch, window, directory / "inputs")
        parent_path = checked_path(root, entry, adapter)
        parent = adapter.load_model(parent_path)
        if reservation is None:
            train_candidate(config, adapter, parent_path, output, key, params, inputs)
        validation = adapter.samples(batch)
        model, artifact, report = candidate_metrics(
            adapter, root, output, parent, parent_path, params, inputs, key, train_rows, validation)
        trial.update(training_seconds=report["training_seconds"], epochs_run=report["epochs_run"],
                     best_epoch=report["best_epoch"], sample_passes=train_rows * report["epochs_run"])
        score = adapter.metrics(model, validation)
        if measure_candidate is not None:
            score = measure_candidate(output, score, plan, model.meta["seed"], names)
        scores = rul_scores(adapter, model, history)
        failures = gate_failures(config, before, score["rmse"], target=target)
        gain = (before - score["rmse"]) / max(before, 1e-12)
        trial.update(status="success", **score, artifact=artifact, checkpoint_sha256=artifact["checkpoint_sha256"],
                     current_relative_gain=gain, history_validation=scores, gate_failures=failures,
                     history_drop=max((row["relative_increase"] for row in scores), default=None))
        reasons = {
            "current_gain_insufficient": (
                f"当前批 RMSE 降幅未达到 {config.rul_min_relative_gain:.2%}" if target is None
                else "当前批 RMSE 未低于批初模型"
            ),
            "validation_target_not_met": f"当前批 RMSE 未达到本轮目标 {target}",
        }
        decision.update(accepted=not failures, eligible=not failures, selected=artifact if not failures else entry,
                        reason="、".join(reasons[key] for key in failures) if failures else "达到当前批验证门槛",
                        history_validation=scores)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, BadZipFile, subprocess.TimeoutExpired) as exc:
        trial.update(error_type=type(exc).__name__, error=str(exc)[-1500:])
    trial["attempt_seconds"] = time.perf_counter() - started
    ledger.put_trial_decision(trial, decision)
    tracker.log_trial(trial, decision)
    return trial


def feature_drift(adapter, batch, previous, seed):
    references = previous[-DRIFT_WINDOW:]
    names = [adapter.group_names[index] for index in references]
    if not references:
        return {"status": "no_reference", "reference_batches": names, "features": [], "global": None,
                "drifted_features": [], "most_drifted_feature": None, "mean_psi": None, "detected": False}
    reference = np.concatenate([adapter.samples(index, "train")["x"] for index in references])
    current = adapter.samples(batch, "train")["x"]
    result = detect_features(reference, current, DRIFT, seed)
    result.update(status="available", reference_batches=names,
                  mean_psi=mean(row["psi"] for row in result["features"]),
                  detected=bool(result["drifted_features"]))
    return result


def run_batch(config, adapter, root, ledger, tracker, planner, batch, entry, history, stop_after,
              *, measure_candidate=None):
    prefix = f"b{batch + 1:02d}"
    name = adapter.group_names[batch]
    entry_model = adapter.load_model(checked_path(root, entry, adapter))
    validation = adapter.samples(batch)
    before = adapter.metrics(entry_model, validation)
    completed = [row["batch"] for row in history]
    historical_batches = history_batches(history, enabled=config.evaluate_history)
    references = rul_references(adapter, historical_batches, entry_model)
    target_batches = history[-VALIDATION_TARGET_WINDOW:]
    validation_target = mean(row["validation_rmse"] for row in target_batches) if target_batches else None
    drift = feature_drift(adapter, batch, completed, config.seed + batch)
    train_samples = len(adapter.rows["train"][name])
    context = {
        "condition": name, "unit": adapter.unit, "new_rows": train_samples,
        "current_validation": {key: round(before[key], 4) for key in ("rmse", "mae")},
        "replay_available": bool(completed),
        "replay_batches": [adapter.group_names[index] for index in completed[-7:]],
        "drift": {
            "status": drift["status"], "reference_batches": drift["reference_batches"],
            "feature_count": len(drift["drifted_features"]),
            "top_feature_indices": drift["drifted_features"][:8],
            "mmd_p": (drift["global"] or {}).get("permutation_p"),
            "mmd_squared": (drift["global"] or {}).get("mmd_squared"),
        },
        "recent_history": [
            {"condition": row["source_file"], "decision": row["trainer_action"], "deployed": row["promoted"],
             "relative_rmse_gain": round((row["before_validation_rmse"] - row["validation_rmse"])
                                         / max(row["before_validation_rmse"], 1e-12), 4),
             "drifted_feature_count": len(row["feature_drift"]["drifted_features"])}
            for row in reversed(history[-3:])
        ],
        "training_defaults": training_defaults(config.dataset), "training_limits": LIMITS,
    }
    attempts, plans = [], []
    stop_reason = "candidate_limit"
    for number in range(1, config.candidate_limit + 1):
        key = f"{prefix}-trainer" if number == 1 else f"{prefix}-trainer-t{number:02d}"
        reserved = ledger.get("training_reservation", key)
        if stop_after is not None and ledger.count("trial") >= stop_after and reserved is None:
            return None
        if config.max_experiments is not None and ledger.count("training_reservation") >= config.max_experiments and reserved is None:
            stop_reason = "experiment_budget"
            break
        feedback = {
            "attempts": [
                {field: value for field, value in trial.items() if field != "history_validation"}
                for trial in attempts
            ],
            "remaining_candidates": config.candidate_limit - number + 1,
            "remaining_experiments": None if config.max_experiments is None else
            max(0, config.max_experiments - ledger.count("training_reservation")),
            "validation_target_rmse": validation_target,
        }
        if validation_target is None:
            feedback["min_relative_rmse_gain"] = config.rul_min_relative_gain
        plan = planner.decide(key, {**context, "batch_feedback": feedback})
        plans.append(plan)
        if plan["output"]["decision"] not in TRAINING_ACTIONS:
            stop_reason = "feedback_fallback" if plan["fallback_reason"] else "trainer_no_update"
            break
        window = training_window(batch, completed, plan["output"]["training_config"]["window_size"])
        trial = run_candidate(config, adapter, root, ledger, tracker, batch, plan, window,
                              entry, before["rmse"], references, validation_target, measure_candidate=measure_candidate)
        attempts.append({field: trial[field] for field in (
            "id", "status", "effective_config", "training_batches", "rmse", "mae", "gate_failures",
            "current_relative_gain", "history_validation", "error_type") if field in trial})
        decision = ledger.get("decision", trial["id"])
        attempts[-1].update(candidate_feedback(
            trial, decision, ("rmse", "mae"),
            can_fallback=trial.get("gate_failures") == ["validation_target_not_met"],
        ))

    best = choose_candidate(attempts)
    selected = ledger.get("trial", best["id"])["artifact"] if best else entry
    below_target = best is not None and bool(best["gate_failures"])
    chosen_plan = next((plan for plan in plans if best and plan["id"] == best["id"]),
                       plans[0] if plans else None)
    score = {key: best[key] for key in ("rmse", "mae")} if best else before
    selected_model = adapter.load_model(checked_path(root, selected, adapter)) if best and references else entry_model
    past = rul_scores(adapter, selected_model, references)
    row = {"batch": batch, "source_file": name, "train_samples": train_samples,
           "validation_samples": len(validation["y"]), "entry": entry, "selected": selected,
           "before_model_path": entry["model_path"], "promoted": best is not None,
           "experiment_count": len(attempts), "selected_trial_id": best["id"] if best else None,
           "before_validation_rmse": before["rmse"], "before_validation_mae": before["mae"],
           "validation_rmse": score["rmse"], "validation_mae": score["mae"],
           "feature_drift": drift, "drift_reference_batches": drift["reference_batches"],
           "history_validation_batches": [group for group, _, _ in references],
           "validation_target_rmse": validation_target,
           "trainer_validation_target_batches": [row["source_file"] for row in target_batches],
           "selected_below_target": below_target,
           "history_validation": past, "max_history_relative_increase":
               max((item["relative_increase"] for item in past), default=None),
           "trainer_attempts": attempts, "training_batches": best["training_batches"] if best else [],
           "training_sample_passes": sum(ledger.get("trial", item["id"])["sample_passes"] for item in attempts),
           "stop_reason": stop_reason,
           "selection_reason": ("best_eligible_below_target" if below_target else
                                "best_eligible_rmse" if best else "keep_entry"),
           "trainer_last_action": plans[-1]["output"]["decision"] if plans else None,
           "trainer_last_reason": plans[-1]["output"]["reason"] if plans else None,
           "trainer_action": chosen_plan["output"]["decision"] if chosen_plan else None,
           "trainer_reason": chosen_plan["output"]["reason"] if chosen_plan else None,
           "trainer_source": chosen_plan["source"] if chosen_plan else None}
    ledger.put("batch", prefix, row)
    tracker.log_batch(row)
    unit = "循环" if adapter.unit == "cycles" else "分钟"
    print(f"{name}: RMSE {before['rmse']:.3f} → {score['rmse']:.3f} {unit}", flush=True)
    return row


def evaluate_public_test(adapter, data_dir, baseline, final):
    groups = []
    for batch, name in enumerate(adapter.group_names):
        samples = adapter.test_samples(batch, data_dir)
        groups.append({"group": name, "static": test_scores(baseline, samples),
                       "final": test_scores(final, samples)})

    pooled = {}
    for stage in ("static", "final"):
        rows = [group[stage] for group in groups]
        count = sum(row["samples"] for row in rows)
        pooled[stage] = {
            "samples": count,
            "rmse": float(np.sqrt(sum(row["rmse"] ** 2 * row["samples"] for row in rows) / count)),
            "mae": sum(row["mae"] * row["samples"] for row in rows) / count,
            "nasa_score": sum(row["nasa_score"] for row in rows)
            if all(row["nasa_score"] is not None for row in rows) else None,
        }
    return {"scope": "last_observation_per_public_test_engine", "groups": groups, "pooled": pooled}


def write_summary(root, ledger, manifest, active, public_test=None):
    batches, trials, requests = ledger.items("batch"), ledger.items("trial"), ledger.requests()
    plans = ledger.items("trainer_decision")
    policy_key, _ = RUL_POLICIES[manifest["dataset"]]
    summary = {"dataset": manifest["dataset"], "metric": "rmse", "unit": manifest["unit"],
               "evaluation_protocol": manifest["protocol"], "independent_test_available": "test" in manifest,
               "initial_checkpoint_already_saw_all_public_training": True,
               "initial_checkpoint_saw_replay_training": manifest.get("initial_checkpoint_saw_replay_training", True),
               "complete": len(batches) == len(manifest["groups"]), "batches_completed": len(batches),
               "experiments": len(trials), "accepted_models": sum(row["promoted"] for row in batches),
               "failed_stages": sum(trial["status"] == "failed" for trial in trials),
               "training_seconds": sum(trial["training_seconds"] for trial in trials),
               "training_sample_passes": sum(trial["sample_passes"] for trial in trials),
               "reserved_training_sample_passes": sum(r["sample_passes"] for r in ledger.items("training_reservation")),
               "api_calls": len(requests), "requests_with_unknown_usage": sum(not request_usage_known(r) for r in requests),
               "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in requests),
               "completion_tokens": sum(r.get("completion_tokens", 0) for r in requests),
               "trainer_decisions": len(plans), "llm_decisions": sum(p["source"] == "trainer_llm" for p in plans),
               "fallback_decisions": sum(p["source"] == "rules_fallback" for p in plans),
               "final_checkpoint": active["model_path"], "final_checkpoint_sha256": active["checkpoint_sha256"],
               policy_key: manifest[policy_key], "batches": batches}
    if public_test is not None:
        summary["public_test"] = public_test
    if batches:
        for metric in ("rmse", "mae"):
            for stage in ("before_", ""):
                key = f"{stage}validation_{metric}"
                summary[f"mean_{key}"] = mean(row[key] for row in batches)
        summary["worst_validation_rmse"] = max(row["validation_rmse"] for row in batches)
        summary["max_history_relative_increase"] = max(
            (row["max_history_relative_increase"] for row in batches if row["max_history_relative_increase"] is not None),
            default=None)
    atomic_json(root / "summary.json", summary)
    atomic_json(root / "active.json", active)
    if manifest["profile"] == "C":
        title, unit = "C 发动机公开验证回放", "循环"
        provenance = "公开训练与验证按整台发动机划分，初始检查点已经使用全部公开训练轨迹。"
    else:
        title, unit = "B 轴承公开验证回放", "分钟"
        provenance = ("每个工况用清单指定的两根轴承分别续训和验证。"
                      if not summary["initial_checkpoint_saw_replay_training"] else
                      "初始检查点已使用本次回放的公开训练轴承。")
    lines = [f"# {title}", "", f"单位为{unit}，表内为 RMSE / MAE。{provenance}", "",
             "| 工况 | 批初模型 | 选中模型 | 决策 |",
             "|---|---:|---:|---|"]
    for row in batches:
        pairs = [f"{row[f'{stage}validation_rmse']:.3f} / {row[f'{stage}validation_mae']:.3f}"
                 for stage in ("before_", "")]
        lines.append("| " + " | ".join([row["source_file"], *pairs, "更新" if row["promoted"] else "保留"]) + " |")
    if public_test is not None:
        lines += ["", "## NASA 公开测试", "",
                  "全部选模结束后，用每台发动机最后一个可见循环评估静态基线和最终模型。", "",
                  "| 子集 | 发动机 | 静态 RMSE / MAE | 最终 RMSE / MAE | 静态 NASA 分 | 最终 NASA 分 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in [*public_test["groups"], {"group": "合计", **public_test["pooled"]}]:
            def nasa(value):
                return f"{value:.3g}" if value is not None else "溢出"
            old, new = row["static"], row["final"]
            lines.append(f"| {row['group']} | {old['samples']} | {old['rmse']:.3f} / {old['mae']:.3f} | "
                         f"{new['rmse']:.3f} / {new['mae']:.3f} | {nasa(old['nasa_score'])} | {nasa(new['nasa_score'])} |")
    (root / "batch_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_rul(config, *, runs_root=None, resume=None, transport=None, stop_after=None, persist_config=False):
    project = project_root()
    saved = read_json(Path(resume).resolve() / "dataset.json") if resume else {}
    explicit_data = (project / config.data_dir).resolve() if config.data_dir else None
    raw_test = None
    if config.dataset == "cmapss_rul":
        bundle = (project / (config.baseline_dir or f"baseline/{C_PACKAGE}")).resolve()
        raw_test = explicit_data if explicit_data is not None else project / RAW_DATA
        if (raw_test / "CMAPSSData").is_dir():
            raw_test = raw_test / "CMAPSSData"
        adapter = CmapssBaseline(bundle)
    else:
        # Historical B runs with the public split continue using that same split.
        bundle = (project / (config.baseline_dir or f"baseline/{PACKAGE}")).resolve()
        public_bundle = explicit_data is not None and (explicit_data / "baseline-info.json").is_file()
        if public_bundle:
            if config.baseline_dir and bundle != explicit_data:
                raise ValueError("公开包回放的数据和检查点必须来自同一个公开基线包")
            bundle = explicit_data
        legacy = (saved.get("protocol") == "bearing_public_condition_replay" if resume else public_bundle)
        replay_dir = explicit_data or project / REPLAY_DIR
        adapter = BearingBaseline(bundle) if legacy else PairedBearingBaseline(bundle, replay_dir)

    policy_key, policy_version = RUL_POLICIES[config.dataset]
    manifest = {**adapter.manifest, policy_key: {
        "version": policy_version, "drift_settings": DRIFT, "drift_window": DRIFT_WINDOW,
        "history_window": HISTORY_WINDOW if config.evaluate_history else 0, "validation_target_window": VALIDATION_TARGET_WINDOW,
        "history_reference": "batch_entry", "max_candidates_per_batch": config.candidate_limit,
        "min_relative_gain": config.rul_min_relative_gain, "max_history_increase": config.rul_max_history_increase}}
    if raw_test is not None:
        manifest["test"] = adapter.test_manifest(raw_test)
    if resume:
        root = Path(resume).resolve()
        if read_json(root / "dataset.json") != manifest:
            raise ValueError("恢复 RUL 运行时数据、基线或策略与原运行不一致")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (Path(runs_root or project / "runs") /
                f"{stamp}_{config.dataset}_{config.strategy}_{uuid4().hex[:8]}").resolve()
        root.mkdir(parents=True)
        config.baseline_dir = saved_data_path(bundle, project)
        if raw_test is not None:
            config.data_dir = saved_data_path(raw_test, project)
        elif not legacy:
            config.data_dir = saved_data_path(replay_dir, project)
        elif config.data_dir:
            config.data_dir = config.baseline_dir
        atomic_json(root / "config.json", config.to_dict())
        atomic_json(root / "dataset.json", manifest)
    print(f"运行目录：{root}", flush=True)
    with RunLock(root / "run.lock"), threadpool_limits(limits=1):
        if persist_config:
            atomic_json(root / "config.json", config.to_dict())
        path = root / "models/baseline.npz"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix(".tmp")
            shutil.copyfile(bundle / "checkpoint.npz", temporary)
            temporary.replace(path)
        if digest(path) != manifest["checkpoint_sha256"]:
            raise ValueError("运行内 RUL 初始检查点摘要不匹配")
        baseline = adapter.load_model(path)
        active = checkpoint(root, path, baseline)
        ledger = Ledger(root / "experiments.sqlite")
        tracker, complete = None, False
        try:
            tracker = Tracker(config, root, ledger, metric="rmse", run_metadata={
                "unit": adapter.unit, "baseline_id": manifest["baseline_id"], "evaluation_protocol": manifest["protocol"],
                policy_key: manifest[policy_key]})
            planner = RulTrainer(config, ledger, transport)
            history = []
            for batch in range(len(adapter.group_names)):
                row = ledger.get("batch", f"b{batch + 1:02d}")
                if row is None:
                    row = run_batch(config, adapter, root, ledger, tracker, planner, batch,
                                    active, history, stop_after)
                if row is None:
                    break
                if row["entry"] != active:
                    raise ValueError("批初检查点与上一个批次末尾不一致")
                checked_path(root, row["selected"], adapter)
                history.append(row)
                active = row["selected"]
            public_test = None
            if raw_test is not None and len(history) == len(adapter.group_names):
                final = adapter.load_model(checked_path(root, active, adapter))
                public_test = evaluate_public_test(adapter, raw_test, baseline, final)
            summary = write_summary(root, ledger, manifest, active, public_test)
            complete = summary["complete"]
            tracker.log_final(summary)
            return summary
        finally:
            if tracker is not None:
                tracker.close(success=complete)
            ledger.close()


run_bearing_rul = run_rul  # 保留原 B 调用入口。
