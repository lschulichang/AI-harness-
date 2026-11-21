"""Compare trainer candidates against the batch-entry model using validation only."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from zipfile import BadZipFile

import numpy as np
from baseline.gas.baselines.model import Model, checkpoint_digest

from .gas_data import evaluate, evaluate_samples, gas_samples
from .gas_drift import detect, label_distribution
from .gas_metrics import evaluate as trainer_evaluate
from .gas_policy import (
    TRAINING_ACTIONS,
    WINDOW_DRIFT_BATCHES,
    WINDOW_TRAINER_CONFIG,
    WINDOW_VALIDATION_BATCHES,
    GasTrainer,
)
from .portability import resolve_model_path
from .resource_monitor import BatchMeter
from .runner import training_window
from .storage import atomic_json, read_json


# 找到检查点
def checked_path(root, entry):
    path = resolve_model_path(root, entry["model_path"])
    if checkpoint_digest(path) != entry["checkpoint_sha256"]:
        raise ValueError("气体模型检查点摘要不匹配")
    return path

#启动一次 A 气体分类候选模型的训练
def train_candidate(config, parent, output, train, validation, request):
    output.parent.mkdir(exist_ok=True)
    env = {
        name: value for name, value in os.environ.items()
        if name not in {config.api_key_env, "PYTHONPATH"} and not name.endswith("_API_KEY")
    }
    env.update(
        BASELINE_CPU_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with tempfile.TemporaryDirectory(prefix="gas-trainer-input-") as temp:
        arrays = Path(temp) / "samples.npz"
        work = Path(temp) / "request.json"
        np.savez(
            arrays,
            **{
                f"{split}_{key}": value
                for split, samples in (("train", train), ("validation", validation))
                if samples is not None
                for key, value in samples.items()
                if key in {"x", "y"}
            },
        )
        atomic_json(work, request)
        response = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "harness_core.gas_training",
                str(arrays),
                str(work),
                str(parent),
                str(output),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.train_timeout,
        )
        if response.returncode:
            raise RuntimeError(response.stderr[-1500:])


def history_gate_failed(history_drop, limit):
    return history_drop is not None and history_drop > limit + 1e-12


def candidate_gate_failures(
    config,
    macro_f1,
    gain,
    history_drop,
    min_validation_f1,
    *,
    max_history_drop=None,
    require_improvement=False,
):
    failures = []
    if min_validation_f1 is None:
        if gain <= 0 or gain + 1e-12 < config.min_gain:
            failures.append("当前验证增益不足")
    else:
        if macro_f1 + 1e-12 < min_validation_f1:
            failures.append(f"当前验证 Macro-F1 未达到 {min_validation_f1:.2%}")
        if require_improvement and gain <= 1e-12:
            failures.append("当前验证 Macro-F1 未优于批初模型")
    limit = config.max_history_drop if max_history_drop is None else max_history_drop
    if history_gate_failed(history_drop, limit):
        failures.append("历史验证退化超限")
    return failures


def evaluate_candidate(
    config,
    root,
    parent_model,
    output,
    request,
    train_rows,
    validation,
    current,
    history,
    *,
    min_validation_f1,
    max_history_drop=None,
    require_improvement=False,
    measure_candidate=None,
):
    path = output / "checkpoint.npz"
    report = read_json(output / "training.json")
    digest = checkpoint_digest(path)
    model = Model.load(path)
    params = request["params"]
    steps_per_epoch = (train_rows + params["batch_size"] - 1) // params["batch_size"]
    if (
        report["request"] != request
        or report["checkpoint_sha256"] != digest
        or report["train_rows"] != train_rows
        or not 1 <= report["best_epoch"] <= report["epochs_run"] <= params["epochs"]
        or model.meta["steps"]
        != parent_model.meta["steps"] + report["best_epoch"] * steps_per_epoch
        or model.meta["epochs"] != parent_model.meta["epochs"] + report["best_epoch"]
        or model.meta["paths"] != parent_model.meta["paths"]
        or model.meta["classes"] != parent_model.meta["classes"]
        or model.meta["learning_rate"] != params["learning_rate"]
        or model.meta["batch_size"] != params["batch_size"]
        or not np.array_equal(model.mean, parent_model.mean)
        or not np.array_equal(model.scale, parent_model.scale)
    ):
        raise ValueError("训练产物与大模型决策、训练起点或标准化状态不一致")
    score = evaluate_samples(path, validation)
    if measure_candidate is not None:
        score = measure_candidate(output, score)
    gain = score["macro_f1"] - current["macro_f1"]
    history_scores = []
    for name, samples, previous in history:
        candidate_f1 = evaluate_samples(path, samples)["macro_f1"]
        history_scores.append(
            {
                "source": name,
                "entry_macro_f1": previous["macro_f1"],
                "candidate_macro_f1": candidate_f1,
                "drop": previous["macro_f1"] - candidate_f1,
            }
        )
    worst = max(history_scores, key=lambda item: item["drop"], default=None)
    drop = worst["drop"] if worst else None
    failures = candidate_gate_failures(
        config,
        score["macro_f1"],
        gain,
        drop,
        min_validation_f1,
        max_history_drop=max_history_drop,
        require_improvement=require_improvement,
    )
    accepted = not failures
    candidate = {
        "model_path": path.relative_to(root).as_posix(),
        "checkpoint_sha256": digest,
        "validation": score,
        "steps": model.meta["steps"],
        "epochs": model.meta["epochs"],
    }
    result = {
        "macro_f1": score["macro_f1"],
        "accuracy": score["accuracy"],
        "history_macro_f1": worst["candidate_macro_f1"] if worst else None,
        "history_drop": drop,
        "checkpoint_sha256": digest,
        "model_path": candidate["model_path"],
        "training_seconds": report["training_seconds"],
        "epochs_run": report["epochs_run"],
        "best_epoch": report["best_epoch"],
        "sample_passes": train_rows * report["epochs_run"],
        "gate_failures": failures,
    }
    return candidate, result, accepted, gain, history_scores


def run_batch_trial(
    config,
    root,
    ledger,
    tracker,
    batch,
    key,
    active,
    entry_path,
    model,
    trainer_decision,
    train,
    validation,
    current,
    history,
    data_id,
    *,
    feedback_enabled,
    min_validation_f1,
    max_history_drop=None,
    training_batches=(),
    allow_best_below_target=False,
    require_improvement=False,
    measure_candidate=None,
):
    decision = ledger.get("decision", key)
    if decision is not None:
        checked_path(root, decision["selected"])
        return decision, decision["status"]
    params = trainer_decision["output"]["training_config"]
    n_train = len(train["y"])
    max_passes = n_train * params["epochs"]
    reservation = ledger.get("training_reservation", key)
    if reservation is None:
        if (
            config.max_experiments is not None
            and ledger.count("training_reservation") >= config.max_experiments
        ):
            return None, "experiment_budget"
        used = sum(r["sample_passes"] for r in ledger.items("training_reservation"))
        if (
            config.gas_sample_budget is not None
            and used + max_passes > config.gas_sample_budget
        ):
            return None, "training_budget"
        ledger.put(
            "training_reservation",
            key,
            {"id": key, "batch": batch, "sample_passes": max_passes},
        )

    output = root / "experiments" / key
    request = {
        "id": key,
        "data_id": data_id,
        "seed": config.seed + batch + 1,
        "params": params,
        "parent_checkpoint_sha256": active["checkpoint_sha256"],
    }
    request["training_batches"] = training_batches
    metadata = trainer_decision["metadata"]
    source = (
        config.provider if metadata["source"] == "trainer_llm" else metadata["source"]
    )
    trial = {
        "id": key,
        "batch": batch,
        "source": source,
        "status": "failed",
        "stage": "trainer",
        "params": params,
        "parent_checkpoint_sha256": active["checkpoint_sha256"],
        "training_seconds": 0.0,
        "sample_passes": 0,
    }
    decision = {
        "id": key,
        "accepted": False,
        "eligible": False,
        "selected": active,
        "status": "training_failed",
        "reason": "候选训练或校验失败",
    }
    trial.update(training_batches=training_batches, train_rows=n_train)
    if feedback_enabled:
        decision["scope"] = "candidate_eligibility"
    ledger.put(
        "proposal",
        key,
        {
            "source": source,
            "params": params,
            "hypothesis": trainer_decision["output"]["reason"],
            "fallback_reason": metadata["error"],
        },
    )

    def measure(directory, local_score):
        return measure_candidate(
            directory, local_score, trainer_decision, request["seed"], training_batches
        )

    started = time.perf_counter()
    try:
        if reservation is None:
            train_candidate(config, entry_path, output, train, validation, request)
        elif not output.is_dir():
            raise RuntimeError("上次尝试中断且没有完整产物")
        candidate, result, accepted, gain, scores = evaluate_candidate(
            config,
            root,
            model,
            output,
            request,
            n_train,
            validation,
            current,
            history,
            min_validation_f1=min_validation_f1,
            max_history_drop=max_history_drop,
            require_improvement=require_improvement,
            measure_candidate=measure if measure_candidate is not None else None,
        )
        trial.update(status="success", **result)
        reason = f"当前验证 Macro-F1 变化 {gain:+.4f}"
        if result["history_drop"] is not None:
            reason += f"，历史最大下降 {result['history_drop']:+.4f}"
        if result["gate_failures"]:
            reason += "，" + "、".join(result["gate_failures"])
        decision.update(
            accepted=accepted,
            eligible=accepted,
            gain=gain,
            reason=reason,
            selected=candidate if accepted else active,
            history_validation=scores,
            status=(
                "trainer_candidate_eligible"
                if feedback_enabled
                else "trainer_candidate_adopted"
            )
            if accepted
            else "trainer_candidate_rejected",
        )
        if allow_best_below_target and min_validation_f1 is not None and not accepted:
            history_limit = (
                config.max_history_drop
                if max_history_drop is None
                else max_history_drop
            )
            if (
                gain > 1e-12
                and result["macro_f1"] + 1e-12 < min_validation_f1
                and not history_gate_failed(result["history_drop"], history_limit)
            ):
                decision.update(
                    fallback_candidate=candidate,
                    status="trainer_candidate_below_target",
                )
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        BadZipFile,
        subprocess.TimeoutExpired,
    ) as exc:
        trial.update(error_type=type(exc).__name__, error=str(exc)[-1500:])
        if reservation is not None and not output.is_dir():
            decision["status"] = "interrupted"
    trial["attempt_seconds"] = time.perf_counter() - started
    ledger.put_trial_decision(trial, decision)
    tracker.log_trial(trial, decision)
    return decision, decision["status"]


def combine_training(data, batches):
    parts = [gas_samples(data, batch, "train") for batch in batches]
    return {key: np.concatenate([part[key] for part in parts]) for key in ("x", "y")}


def rolling_drift(data, references, train, drift_config, seed, classes):
    if references:
        result = detect(
            combine_training(data, references),
            train,
            drift_config,
            seed,
            classes=classes,
        )
        result["status"] = "available"
    else:
        result = {
            "status": "no_reference",
            "features": [],
            "global": None,
            "drifted_features": [],
            "most_drifted_feature": None,
            "label_distribution": label_distribution({}, train, classes),
            "elapsed_seconds": 0.0,
        }
    result["reference_batches"] = [data.batch_names[i] for i in references]
    return result


def run_batch_search(
    config,
    root,
    ledger,
    tracker,
    planner,
    batch,
    entry,
    entry_path,
    model,
    train,
    validation,
    current,
    histories,
    data_id,
    context,
    directory,
    policy,
    stop_after,
    *,
    min_validation_f1,
    data,
    completed,
    measure_candidate=None,
    combine_candidate_training=combine_training,
):
    feedback_enabled = config.strategy == "llm"
    trainer_config = WINDOW_TRAINER_CONFIG
    max_history_drop = policy.get("max_history_drop", config.max_history_drop)
    limit = policy["max_candidates_per_batch"]
    require_improvement = "validation_target_window" in policy
    history = None
    plans, attempts = [], []
    best = None
    best_below_target = None
    allow_best_below_target = policy.get("allow_best_below_target", False)
    stop_reason = "candidate_limit"
    for number in range(1, limit + 1):
        key = f"b{batch + 1:02d}-trainer"
        file = "trainer_decision.json"
        if number > 1:
            key += f"-t{number:02d}"
            file = f"trainer_decision_t{number:02d}.json"

        remaining = None
        if config.max_experiments is not None:
            remaining = max(
                0, config.max_experiments - ledger.count("training_reservation")
            )
            if (
                attempts
                and remaining == 0
                and ledger.get("trainer_decision", key) is None
            ):
                stop_reason = "experiment_budget"
                break
        feedback = None
        if feedback_enabled:
            feedback = {
                "remaining_candidates": limit - len(attempts),
                "remaining_global_experiments": remaining,
                "max_historical_macro_f1_drop": max_history_drop,
                "batch_entry_macro_f1": current["macro_f1"],
                "attempts": attempts.copy(),
            }
            if min_validation_f1 is None:
                feedback["min_current_macro_f1_gain"] = config.min_gain
            else:
                feedback["min_current_validation_macro_f1"] = min_validation_f1
            if allow_best_below_target and min_validation_f1 is not None:
                feedback["allow_best_below_target"] = True
        plan = planner.decide(key, context, trainer_config, feedback=feedback)
        atomic_json(directory / file, plan)
        plans.append(plan)
        action = plan["output"]["decision"]
        if action not in TRAINING_ACTIONS:
            if attempts and plan["metadata"]["fallback"]:
                stop_reason = "feedback_fallback"
            elif attempts and action == "NO_UPDATE":
                stop_reason = "trainer_stop"
            else:
                stop_reason = action
            break

        if history is None:
            history = [
                (group, rows, evaluate_samples(entry_path, rows))
                for group, rows in histories
            ]
        window = training_window(
            batch, completed, plan["output"]["training_config"]["window_size"]
        )
        training_batches = [data.batch_names[i] for i in window]
        candidate_train = (
            combine_candidate_training(data, window) if len(window) > 1 else train
        )
        decision, status = run_batch_trial(
            config,
            root,
            ledger,
            tracker,
            batch,
            key,
            entry,
            entry_path,
            model,
            plan,
            candidate_train,
            validation,
            current,
            history,
            data_id,
            feedback_enabled=feedback_enabled,
            min_validation_f1=min_validation_f1,
            max_history_drop=max_history_drop,
            training_batches=training_batches,
            allow_best_below_target=allow_best_below_target,
            require_improvement=require_improvement,
            measure_candidate=measure_candidate,
        )
        if decision is None:
            stop_reason = status
            break
        trial = ledger.get("trial", key)
        attempts.append(
            {
                "trial_id": key,
                "training_config": trial["params"],
                "source": plan["metadata"]["source"],
                "status": trial["status"],
                "validation_macro_f1": trial.get("macro_f1"),
                "validation_accuracy": trial.get("accuracy"),
                "current_gain": decision.get("gain"),
                "worst_historical_group": max(
                    decision.get("history_validation", ()),
                    key=lambda row: row["drop"],
                    default=None,
                ),
                "eligible": decision["eligible"],
                "reason": decision["reason"],
                "error_type": trial.get("error_type"),
                "sample_passes": trial["sample_passes"],
            }
        )
        attempts[-1].update(
            training_batches=trial["training_batches"], train_rows=trial["train_rows"]
        )
        if allow_best_below_target:
            attempts[-1]["fallback_eligible"] = "fallback_candidate" in decision
        if decision["eligible"] and (best is None or decision["gain"] > best["gain"]):
            best = decision
        elif "fallback_candidate" in decision and (
            best_below_target is None or decision["gain"] > best_below_target["gain"]
        ):
            best_below_target = decision
        if not feedback_enabled or limit == 1:
            stop_reason = (
                "trainer_candidate_adopted" if decision["eligible"] else status
            )
        if (
            number < limit
            and stop_after is not None
            and ledger.count("trial") >= stop_after
        ):
            # No batch decision or held-out score until the feedback loop finishes.
            return None
    if best is None and best_below_target is not None:
        best = {
            **best_below_target,
            "selected": best_below_target["fallback_candidate"],
            "selected_below_target": True,
        }
    return best, plans, attempts, stop_reason


def run_trainer_batches(
    config,
    bundle,
    root,
    ledger,
    tracker,
    transport,
    baseline,
    active,
    data,
    batches,
    manifest,
    stop_after,
):
    meter = BatchMeter(ledger)
    planner = GasTrainer(config, ledger, transport)
    model_meta = Model.load(baseline).meta
    policy = manifest["gas_trainer_policy"]
    target_window = policy.get("validation_target_window")
    histories, completed, decision_history = [], [], []
    try:
        for batch in batches:
            batch_key = f"{batch:02d}"
            validation = gas_samples(data, batch, "validation")
            name = data.batch_names[batch]
            saved = ledger.get("batch", batch_key)
            if saved is not None:
                active = saved["selected"]
                decision_history.insert(0, saved["trainer_history"])
                histories.append((name, validation))
                completed.append(batch)
                continue
            trial_id = f"b{batch + 1:02d}-trainer"
            meter.start(
                batch, at_entry=ledger.get("trainer_decision", trial_id) is None
            )
            entry = active
            entry_path = checked_path(root, entry)
            model = Model.load(entry_path)
            train = gas_samples(data, batch, "train")
            current = evaluate_samples(entry_path, validation)
            batch_size = data.batch_size(batch)
            drift = ledger.get("trainer_drift", batch_key)
            if drift is None:
                drift = rolling_drift(
                    data,
                    completed[-WINDOW_DRIFT_BATCHES:],
                    train,
                    WINDOW_TRAINER_CONFIG["drift"],
                    config.seed + batch + 1,
                    model_meta["classes"],
                )
                ledger.put("trainer_drift", batch_key, drift)
            directory = root / "reports" / Path(name).stem
            atomic_json(directory / "drift.json", drift)
            label_status = drift["label_distribution"].get("status", "available")
            drift_report = {
                "mmd": drift["global"],
                "drifted_features": drift["drifted_features"],
                "label_status": label_status,
                "label_js_divergence": drift["label_distribution"]["js_divergence"],
                "status": drift["status"],
                "reference_batches": drift["reference_batches"],
            }
            candidate_histories = histories[-WINDOW_VALIDATION_BATCHES:]
            replay_batches = [data.batch_names[i] for i in completed[-7:]]
            context = {
                "drift_report": drift_report,
                "performance": {
                    "current": trainer_evaluate(model, validation),
                },
                "data": {
                    "new_labeled_samples": batch_size,
                    "replay_batches": replay_batches,
                    "historical_replay_available": bool(replay_batches),
                },
                "history": decision_history,
            }
            target_batches = completed[-target_window:] if target_window else []
            validation_target = policy.get("min_validation_macro_f1")
            if target_batches:
                validation_target = mean(
                    ledger.get("batch", f"{i:02d}")["validation_macro_f1"]
                    for i in target_batches
                )
            context["validation_target"] = validation_target
            result = run_batch_search(
                config,
                root,
                ledger,
                tracker,
                planner,
                batch,
                entry,
                entry_path,
                model,
                train,
                validation,
                current,
                candidate_histories,
                manifest["raw_gas"]["dataset_id"],
                context,
                directory,
                policy,
                stop_after,
                min_validation_f1=validation_target,
                data=data,
                completed=completed,
            )
            if result is None:
                return entry
            best, plans, attempts, stop_reason = result
            if best is not None:
                active = best["selected"]
            selected_id = best["id"] if best else None
            selected_number = next(
                (
                    i
                    for i, attempt in enumerate(attempts)
                    if attempt["trial_id"] == selected_id
                ),
                0,
            )
            plan = plans[selected_number]
            action = plan["output"]["decision"]
            chosen_path = checked_path(root, active)
            final = evaluate_samples(chosen_path, validation)
            promoted = active["model_path"] != entry["model_path"]
            trainer_gain = (
                trainer_evaluate(Model.load(chosen_path), validation)["macro_f1"]
                - context["performance"]["current"]["macro_f1"]
            )
            # Persist the chosen model before evaluating held-out samples.
            active = {**active, "validation": final}
            atomic_json(root / "active.json", active)
            profile = meter.finish(batch)
            before, after = (
                evaluate(path, data, batch, "test")
                for path in (entry_path, chosen_path)
            )
            row = {
                "batch": batch,
                "source_file": name,
                "selected": active,
                "promoted": promoted,
                "experiment_count": len(attempts),
                "train_samples": len(train["y"]),
                "validation_samples": len(validation["y"]),
                "test_samples": batch_size - len(train["y"]) - len(validation["y"]),
                "before_model_path": entry["model_path"],
                "model_path": active["model_path"],
                "training_sample_passes": sum(t["sample_passes"] for t in attempts),
                "trainer_feedback_enabled": config.strategy == "llm",
                "selected_trial_id": selected_id,
                "trainer_attempts": attempts,
                "selected_below_target": best is not None
                and best.get("selected_below_target", False),
                "trainer_min_validation_f1": validation_target,
                "trainer_last_action": plans[-1]["output"]["decision"],
                "trainer_last_reason": plans[-1]["output"]["reason"],
                "validation_drop": current["macro_f1"] - final["macro_f1"],
                "update_test_gain": after["macro_f1"] - before["macro_f1"],
                "update_test_accuracy_gain": after["accuracy"] - before["accuracy"],
                "trainer_action": action,
                "trainer_source": plan["metadata"]["source"],
                "trainer_reason": plan["output"]["reason"],
                "stop_reason": stop_reason,
                "trainer_history": {
                    "batch": batch + 1,
                    "decision": action,
                    "deployed": promoted,
                    "macro_f1_gain": trainer_gain,
                    "drifted_feature_count": len(drift["drifted_features"]),
                },
                "feature_drift": {
                    "mean_psi": float(np.mean([f["psi"] for f in drift["features"]]))
                    if drift["features"]
                    else None,
                    "detected": bool(drift["drifted_features"]),
                    "mmd": drift["global"],
                    "label_status": label_status,
                    "label_js_divergence": drift["label_distribution"]["js_divergence"],
                    "status": drift["status"],
                    "reference_batches": drift["reference_batches"],
                },
                "drift_reference_batches": drift["reference_batches"],
                "history_validation_batches": [
                    group for group, _ in candidate_histories
                ],
                "test_protocol": "development_holdout_previously_public"
                if batch == 1
                else "local_holdout",
                **profile,
            }
            for metric in ("macro_f1", "accuracy"):
                row.update(
                    {
                        f"before_validation_{metric}": current[metric],
                        f"validation_{metric}": final[metric],
                        f"before_test_{metric}": before[metric],
                        f"test_{metric}": after[metric],
                    }
                )
            if target_window:
                row["trainer_validation_target_batches"] = [
                    data.batch_names[i] for i in target_batches
                ]
            ledger.put("batch", batch_key, row)
            tracker.log_batch(row)
            histories.append((name, validation))
            completed.append(batch)
            decision_history.insert(0, row["trainer_history"])
            print(
                f"{name} {row['trainer_source']} → {action} → {stop_reason}，"
                f"验证 Macro-F1 {current['macro_f1']:.4f} → {final['macro_f1']:.4f}",
                flush=True,
            )
            if (
                attempts
                and stop_after is not None
                and ledger.count("trial") >= stop_after
            ):
                break
        return active
    finally:
        meter.close()
