"""Run the existing A/B/C trainers through contest_harness workspace.v2."""

import math
import os
import shutil
import tempfile
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from baseline.gas.baselines.model import Model, checkpoint_digest
from baseline.gas.baselines.predict import predict as predict_checkpoint
from contest_data import (
    BUNDLES,
    DATASETS,
    ROOT,
    TASK_TYPES,
    UNITS,
    BearingProjection,
    CmapssProjection,
    GasProjection,
    Projection,
    combine_gas_candidate,
    sha256_matches,
    validate_model,
)
from contest_harness.errors import HarnessError
from harness_core.config import RunConfig
from harness_core.gas_data import evaluate_samples, gas_samples
from harness_core.gas_metrics import evaluate as trainer_evaluate
from harness_core.gas_policy import (
    WINDOW_DRIFT_BATCHES,
    WINDOW_POLICY,
    WINDOW_TARGET_BATCHES,
    WINDOW_TRAINER_CONFIG,
    WINDOW_VALIDATION_BATCHES,
    GasTrainer,
)
from harness_core.gas_workflow import rolling_drift, run_batch_search
from harness_core.rul_policy import RulTrainer
from harness_core.rul_workflow import RUL_POLICIES, checkpoint, run_batch
from harness_core.storage import Ledger, atomic_json
from harness_core.tracking import Tracker

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
SCORE_KEYS = {"A": ("macro_f1", "accuracy"), "B": ("rmse", "mae"), "C": ("rmse", "mae")}
_PREDICT_CACHE = None


def text_limit(value, size=4000):
    return str(value).encode("utf-8")[:size].decode("utf-8", errors="ignore").strip()


class Gateway:
    def __init__(self, rnd):
        self.rnd = rnd

    def __call__(self, messages):
        receipt = self.rnd.model(
            messages, temperature=0, response_format={"type": "json_object"},
            max_tokens=self.rnd.max_output_tokens,
        )
        usage = receipt.response.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        return {
            "content": receipt.text, "call_id": receipt.call_id,
            "prompt_tokens": prompt or 0, "completion_tokens": completion or 0,
            "usage_known": prompt is not None and completion is not None,
        }


def measured_scores(summary, keys):
    if not isinstance(summary, dict):
        return {}
    # SDK documents a public summary; accept direct metrics or a nested validation map.
    maps = []
    for key in ("validation", "public_validation", "validation_metrics", "public_metrics", "metrics"):
        group = summary.get(key)
        if isinstance(group, dict):
            maps.extend((group.get("metrics"), group))
    maps.append(summary)
    result = {}
    for key in keys:
        for mapping in maps:
            if not isinstance(mapping, dict):
                continue
            value = mapping.get(key)
            if isinstance(value, dict):
                value = value.get("value")
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                if key in {"macro_f1", "accuracy"} and value > 1:
                    continue
                result[key] = float(value)
                break
    return result


class CandidateMeasurements:
    def __init__(self, rnd, root, policy):
        self.rnd = rnd
        self.root = root
        self.policy = policy
        self.results = {}

    def __call__(self, output, local_score, plan, seed, batches):
        checkpoint_path = output / "checkpoint.npz"
        name = checkpoint_path.relative_to(self.root).as_posix()
        if name in self.results:
            return self.results[name]["decision_scores"]
        source = ROOT / "baseline" / BUNDLES[self.rnd.dataset_key] / "baselines"
        (output / "baselines").mkdir(exist_ok=True)
        code_paths = ["baselines/model.py", "baselines/data.py"]
        for path in code_paths:
            shutil.copyfile(source / Path(path).name, output / path)
        configuration = {
            "policy": self.policy, "training_config": plan["output"]["training_config"],
            "training_batches": batches, "public_validation_only": True,
        }
        receipt = self.rnd.submit_candidate(
            output, hypothesis=text_limit(plan["output"]["reason"]),
            change_summary="从本轮父检查点保留权重、Adam 和标准化状态，按选定滑窗对公开训练集续训。",
            configuration=configuration, seed=seed, checkpoint_paths=["checkpoint.npz"],
            code_paths=code_paths, predict_argv=PREDICT_ARGV,
        )
        host_scores = measured_scores(receipt.get("summary"), SCORE_KEYS[self.rnd.dataset_key])
        score = {**local_score, **host_scores}
        self.results[name] = {
            "artifact_id": receipt["artifact_id"], "configuration": configuration,
            "platform_public_validation": host_scores, "local_public_validation": local_score,
            "decision_scores": score,
        }
        return score


def train_gas(config, rnd, root, ledger, tracker, projection, entry, parent_model, measure, transport):
    batch = projection.batch
    history = projection.history
    completed = [row["batch"] for row in history]
    data = GasProjection(projection, parent_model)
    train = gas_samples(data, batch, "train")
    validation = gas_samples(data, batch, "validation")
    entry_path = root / entry["model_path"]
    current = evaluate_samples(entry_path, validation)
    drift = rolling_drift(
        data, completed[-WINDOW_DRIFT_BATCHES:], train, WINDOW_TRAINER_CONFIG["drift"],
        config.seed + batch + 1, parent_model.meta["classes"],
    )
    atomic_json(root / "reports" / "drift.json", drift)
    target_batches = history[-WINDOW_TARGET_BATCHES:]
    target = mean(row["validation_macro_f1"] for row in target_batches) if target_batches else None
    replay = [data.batch_names[i] for i in completed]
    context = {
        "drift_report": {
            "mmd": drift["global"], "drifted_features": drift["drifted_features"],
            "label_status": drift["label_distribution"].get("status", "available"),
            "label_js_divergence": drift["label_distribution"]["js_divergence"],
            "status": drift["status"], "reference_batches": drift["reference_batches"],
        },
        "performance": {"current": trainer_evaluate(parent_model, validation)},
        "data": {"new_labeled_samples": len(train["y"]) + len(validation["y"]),
                 "replay_batches": replay, "historical_replay_available": bool(replay)},
        "history": [row["trainer_history"] for row in reversed(history)],
        "validation_target": target,
    }
    histories = [(data.batch_names[i], gas_samples(data, i, "validation"))
                 for i in completed[-WINDOW_VALIDATION_BATCHES:]]
    policy = {"max_candidates_per_batch": config.candidate_limit,
              "max_history_drop": config.max_history_drop,
              "validation_target_window": WINDOW_TARGET_BATCHES, "allow_best_below_target": True}
    best, plans, attempts, stop_reason = run_batch_search(
        config, root, ledger, tracker, GasTrainer(config, ledger, transport), batch, entry, entry_path,
        parent_model, train, validation, current, histories, f"A-round-{rnd.round_index}",
        context, root / "reports", policy, None,
        min_validation_f1=target, data=data, completed=completed, measure_candidate=measure,
        combine_candidate_training=combine_gas_candidate,
    )
    selected = best["selected"] if best else entry
    trial_id = best["id"] if best else None
    winner = next((i for i, item in enumerate(attempts) if item["trial_id"] == trial_id), 0)
    plan = plans[winner]
    selected_score = measure.results[selected["model_path"]]["decision_scores"] if best else current
    prompt_gain = (trainer_evaluate(Model.load(root / selected["model_path"]), validation)["macro_f1"]
                   - context["performance"]["current"]["macro_f1"]) if best else 0.0
    row = {
        "batch": batch, "source_file": data.batch_names[batch], "selected": selected, "promoted": best is not None,
        "experiment_count": len(attempts), "train_samples": len(train["y"]),
        "validation_samples": len(validation["y"]), "before_model_path": entry["model_path"],
        "before_validation_macro_f1": current["macro_f1"], "validation_macro_f1": selected_score["macro_f1"],
        "before_validation_accuracy": current["accuracy"], "validation_accuracy": selected_score["accuracy"],
        "trainer_attempts": attempts, "trainer_source": plan["metadata"]["source"],
        "trainer_action": plan["output"]["decision"], "trainer_reason": plan["output"]["reason"],
        "trainer_last_action": plans[-1]["output"]["decision"], "trainer_last_reason": plans[-1]["output"]["reason"],
        "trainer_min_validation_f1": target, "selected_trial_id": trial_id,
        "selected_below_target": bool(best and best.get("selected_below_target")), "stop_reason": stop_reason,
        "training_sample_passes": sum(trial["sample_passes"] for trial in attempts),
        "drift_reference_batches": drift["reference_batches"],
        "history_validation_batches": [name for name, _ in histories],
        "trainer_validation_target_batches": [item["source_file"] for item in target_batches],
        "feature_drift": {
            "mean_psi": mean(item["psi"] for item in drift["features"]) if drift["features"] else None,
            "detected": bool(drift["drifted_features"]), "status": drift["status"],
            "mmd": drift["global"], "label_js_divergence": drift["label_distribution"]["js_divergence"],
        },
        "trainer_history": {"batch": batch + 1, "decision": plan["output"]["decision"],
                            "deployed": best is not None, "macro_f1_gain": prompt_gain,
                            "drifted_feature_count": len(drift["drifted_features"])},
    }
    ledger.put("batch", f"{batch:02d}", row)
    tracker.log_batch(row)
    return row


def append_audit(state, root, row, ledger, measure, provenance):
    atomic_json(root / "round.json", {
        "result": row, "platform": provenance, "measurements": measure.results,
        "planner_requests": ledger.requests(), "tracking_errors": ledger.items("tracking"),
        "trials": ledger.items("trial"), "decisions": ledger.items("decision"),
        "proposals": ledger.items("proposal"),
    })
    with ZipFile(state / "audit.zip", "a", compression=ZIP_DEFLATED, compresslevel=5) as archive:
        name = row["source_file"]
        archive.write(root / "round.json", f"{name}/round.json")
        for path in sorted((root / "swanlog").rglob("*")):
            if path.is_file() and not path.is_symlink():
                archive.write(path, f"{name}/{path.relative_to(root).as_posix()}")


def train(rnd):
    key = rnd.dataset_key
    task = rnd.task.get("task")
    if task != TASK_TYPES[key]:
        raise ValueError(f"{key} 的任务类型与比赛接口不一致")
    if key in UNITS and rnd.task.get("target_unit", UNITS[key]) != UNITS[key]:
        raise ValueError(f"{key} 的目标单位与比赛接口不一致")

    checkpoints = rnd.parent.descriptor["checkpoint_paths"]
    if len(checkpoints) != 1:
        raise ValueError("CPU MLP 训练需要父产物声明一个 NPZ 检查点")
    parent = rnd.parent.path(checkpoints[0])
    parent_sha = checkpoint_digest(parent)
    if not sha256_matches(parent_sha, rnd.parent.checkpoint_sha256):
        raise ValueError("父检查点摘要与轮次合同不一致")
    policy = WINDOW_POLICY if key == "A" else RUL_POLICIES[DATASETS[key]][1]
    config = RunConfig(dataset=DATASETS[key], model=rnd.model_name, swanlab_mode="offline",
                       swanlab_workspace=None, max_output_tokens=rnd.max_output_tokens).validate()

    workspace = Path(os.environ.get("CONTEST_WORK", "/workspace")).resolve()
    with tempfile.TemporaryDirectory(prefix=f"{key}-round-{rnd.round_index}-", dir=workspace) as directory:
        root = Path(directory)
        os.environ["SWANLAB_ROOT"] = str(root / "swan-config")
        state = rnd.state_workspace(root / "state")
        entry_path = root / "parent.npz"
        shutil.copyfile(parent, entry_path)
        model = Model.load(entry_path)
        validate_model(model, key, task)
        entry = checkpoint(root, entry_path, model)
        projection = Projection(rnd, state, model, entry_path)
        history_records = rnd.history()
        measure = CandidateMeasurements(rnd, root, policy)
        transport = Gateway(rnd)
        ledger = Ledger(root / "experiments.sqlite")
        try:
            tracker = Tracker(
                config, root, ledger, metric=SCORE_KEYS[key][0],
                run_metadata={"dataset_key": key, "policy": policy, "stage_id": rnd.stage_id},
            )
            successful = False
            try:
                if key == "A":
                    row = train_gas(config, rnd, root, ledger, tracker, projection, entry, model, measure, transport)
                else:
                    adapter = (BearingProjection if key == "B" else CmapssProjection)(projection, model)
                    static = adapter.load_model(state / "static.npz")
                    planner = RulTrainer(config, ledger, transport)
                    row = run_batch(config, adapter, root, ledger, tracker, planner, projection.batch, entry,
                                    static, projection.history, None, measure_candidate=measure)
                if not rnd.call_ids:
                    raise HarnessError("MODEL_CALL_REQUIRED", "宿主未返回真实模型调用回执")
                successful = True
            finally:
                tracker.close(successful)

            chosen = measure.results.get(row["selected"]["model_path"])
            picked = chosen["artifact_id"] if chosen else rnd.parent.artifact_id
            provenance = {"stage_id": rnd.stage_id, "round_index": rnd.round_index,
                          "policy": policy, "selected_artifact_id": picked,
                          "parent_artifact_id": rnd.parent.artifact_id,
                          "model_call_ids": list(rnd.call_ids), "history_ids": [item["id"] for item in history_records]}
            projection.save(row)
            append_audit(state, root, row, ledger, measure, provenance)
            rnd.save_state(state)

            result = "选择通过历史门槛且最优的候选" if chosen else "保留本轮父模型"
            if chosen and row.get("selected_below_target"):
                result = "采用通过历史门槛且优于父模型的最佳未达目标候选"
            reason = text_limit(f"{result}。{row['trainer_last_reason'] or row['trainer_reason']}（{row['stop_reason']}）")
            rnd.finish(
                selected_artifact_id=picked,
                hypothesis=text_limit(row["trainer_reason"] or reason),
                change_summary="使用已测量的续训检查点。" if chosen else "本轮继续使用同一个父检查点。",
                reason=reason, seed=model.meta["seed"],
                configuration={
                    **provenance, "candidate_count": len(measure.results), "stop_reason": row["stop_reason"],
                    "training_batches": chosen["configuration"]["training_batches"] if chosen else [],
                    "training_config": chosen["configuration"]["training_config"] if chosen else {},
                    "selected_below_target": row.get("selected_below_target", False),
                },
                history_ids=provenance["history_ids"], model_call_ids=provenance["model_call_ids"],
            )
        finally:
            ledger.close()


def predict(sample, model_dir):
    global _PREDICT_CACHE
    root = Path(model_dir).resolve()
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH", str(root / "checkpoint.npz"))).resolve()
    if not path.is_relative_to(root):
        raise ValueError("检查点路径不在平台选中的产物目录中")
    if _PREDICT_CACHE is None or _PREDICT_CACHE[0] != path:
        _PREDICT_CACHE = path, Model.load(path)
    model = _PREDICT_CACHE[1]
    key = sample.dataset_key or model.meta.get("profile")
    validate_model(model, key, sample.task)
    output = predict_checkpoint(model, sample.inputs)["outputs"]
    if sample.task == "classification":
        best = max(output, key=lambda item: item["score"])
        if sum(item["score"] == best["score"] for item in output) > 1:
            best["score"] = float(np.nextafter(best["score"], 1.0))
    return output
