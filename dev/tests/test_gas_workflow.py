"""Trainer contract through the actual gas entry point and CPU training subprocess."""
import json
from statistics import mean

from harness_core.config import RunConfig
from harness_core.runner import run_workflow
from harness_core.storage import Ledger, read_json


def test_trainer_executes_one_complete_decision_and_resumes(dataset, tmp_path, monkeypatch):
    import harness_core.gas_workflow as workflow

    params = {"learning_rate": .0003, "batch_size": 96, "epochs": 2, "early_stopping_patience": 1,
              "weight_decay": .0002, "window_size": 4, "class_weighting": True}
    payloads = []
    actions = [
        ("INCREMENTAL_TRAINING", {**params, "window_size": 1}),
        ("MIXED_DATA_TRAINING", params),
        ("NO_UPDATE", {}),
        ("NO_UPDATE", {}),
    ]

    def transport(messages):
        payloads.append(json.loads(messages[1]["content"]))
        action, plan = actions[min(len(payloads) - 1, len(actions) - 1)]
        return {"content": json.dumps({"decision": action, "reason": "synthetic validation",
                                        "training_config": plan, "evaluation_requirements": ["new_data_macro_f1"]}),
                "prompt_tokens": 10, "completion_tokens": 20}

    real_eval = workflow.evaluate

    def audit(path, data, batch, split):
        assert split == "test"
        root = next(tmp_path.iterdir())
        ledger = Ledger(root / "experiments.sqlite")
        try:
            key = f"b{batch + 1:02d}-trainer"
            assert ledger.get("trainer_decision", key)
            if batch in (1, 2):
                assert ledger.get("decision", key)
        finally:
            ledger.close()
        return real_eval(path, data, batch, split)

    monkeypatch.setattr(workflow, "evaluate", audit)
    config = RunConfig(strategy="llm", data_dir=str(dataset.root), swanlab_mode="disabled", trials_per_batch=1)
    root, partial = run_workflow(config, runs_root=tmp_path, transport=transport, stop_after=1)
    assert not partial["complete"] and partial["experiments"] == 1 and partial["api_calls"] == 1
    trial = read_json(root / "history.json")["trials"][0]
    assert trial["status"] == "success"
    assert trial["params"] == {**params, "window_size": 1}
    report = root / "experiments/b02-trainer/training.json"
    stamp = report.stat().st_mtime_ns
    assert read_json(report)["request"]["training_batches"] == ["batch2.dat"]
    assert read_json(report)["train_rows"] == len(dataset.load(1, "train")[1])

    _, result = run_workflow(resume=root, transport=transport)
    assert result["complete"] and result["experiments"] == 2
    assert result["llm_decisions"] == 9 and result["api_calls"] == len(payloads) == 9
    assert report.stat().st_mtime_ns == stamp
    first, second = payloads[:2]
    assert "historical_validation" not in first and not first["replay_batches"]
    assert first["drift"]["status"] == "no_reference" and first["drift"]["mmd_p"] is None
    assert all("historical_validation" not in payload and "historical_validation_batch" not in payload
               for payload in payloads)
    assert second["drift"]["reference_batches"] == second["replay_batches"] == ["batch2.dat"]
    assert second["recent_history"][0]["decision"] == "INCREMENTAL_TRAINING"
    assert "sensor" not in json.dumps(payloads)
    trained = read_json(root / "experiments/b03-trainer/training.json")
    assert trained["train_rows"] == sum(len(dataset.load(i, "train")[1]) for i in (1, 2))
    assert "test" not in json.dumps(payloads)
    batches = read_json(root / "audit_test_results.json")
    assert [b["trainer_action"] for b in batches[:4]] == [action for action, _ in actions]
    assert all(b["experiment_count"] == 0 for b in batches[2:])
    assert result["gas_trainer_validation_target_window"] == 4
    assert "min_current_validation_macro_f1" not in first["batch_feedback"]
    assert first["batch_feedback"]["min_current_macro_f1_gain"] == config.min_gain
    for index, batch in enumerate(batches):
        prior = batches[max(0, index - 4):index]
        target = mean(row["validation_macro_f1"] for row in prior) if prior else None
        assert batch["trainer_validation_target_batches"] == [row["source_file"] for row in prior]
        assert batch["trainer_min_validation_f1"] == target
        assert payloads[index]["batch_feedback"].get("min_current_validation_macro_f1") == target
    by_name = {b["source_file"]: b for b in batches}
    assert by_name["batch2.dat"]["test_protocol"] == "development_holdout_previously_public"
    assert by_name["batch5.dat"]["drift_reference_batches"] == ["batch2.dat", "batch3.dat", "batch4.dat"]
    assert by_name["batch6.dat"]["drift_reference_batches"] == ["batch3.dat", "batch4.dat", "batch5.dat"]
    assert all(batch["history_validation_batches"] == [] for batch in batches)
    assert all(row["history_drop"] is None for row in read_json(root / "history.json")["trials"])
    assert result["drift_profiled_batches"] == 8 and result["drift_no_reference_batches"] == 1
    drift = read_json(root / "reports/batch3/drift.json")
    assert len(drift["features"]) == dataset.feature_count
    assert "sensor" not in json.dumps(drift)
    assert drift["label_distribution"]["class_labels"] == list(dataset.classes)
    assert {"psi", "ks_p", "ks_bh_significant", "wasserstein"} <= drift["features"][0].keys()
    assert "permutation_p" in drift["global"] and "js_divergence" in drift["label_distribution"]


def test_incomplete_llm_plan_is_rejected_before_training(dataset, tmp_path):
    def transport(messages):
        return {"content": json.dumps({"decision": "MIXED_DATA_TRAINING", "reason": "incomplete",
                                        "training_config": {}, "evaluation_requirements": []})}

    config = RunConfig(strategy="llm", data_dir=str(dataset.root), swanlab_mode="disabled",
                       max_api_calls=1, max_experiments=0)
    root, result = run_workflow(config, runs_root=tmp_path, transport=transport)
    assert result["complete"] and result["experiments"] == 0
    assert result["api_calls"] == 1 and result["llm_decisions"] == 0
    assert result["fallback_decisions"] == result["batches_completed"]
    plan = read_json(root / "reports/batch2/trainer_decision.json")
    assert plan["metadata"]["source"] == "rules_fallback"


def test_feedback_resumes_and_keeps_best_eligible_candidate(dataset, tmp_path, monkeypatch):
    import harness_core.gas_workflow as workflow

    real_candidate = workflow.evaluate_candidate
    real_evaluate = workflow.evaluate
    payloads = []
    params = {"learning_rate": .0003, "batch_size": 96, "epochs": 2, "early_stopping_patience": 1,
              "weight_decay": .0002, "window_size": 4, "class_weighting": True}

    def transport(messages):
        payload = json.loads(messages[1]["content"])
        payloads.append(payload)
        attempts = payload["batch_feedback"]["attempts"]
        # Warm up the history with batch2, then try three candidates on batch3.
        if len(payloads) == 4:
            assert "already been tried" in messages[-1]["content"]
        should_train = 2 <= len(payloads) <= 5
        plan = {**params, "batch_size": 96 + len(attempts), "epochs": 2 + len(attempts)}
        if len(payloads) == 3:
            # The reported legacy width is ignored, leaving the same effective plan.
            plan = {**params, "window_size": 8}
        return {"content": json.dumps({
            "decision": "MIXED_DATA_TRAINING" if should_train else "NO_UPDATE",
            "reason": "test response", "evaluation_requirements": [],
            "training_config": plan if should_train else {},
        })}

    def evaluate_candidate(*args, **kwargs):
        candidate, result, _, _, scores = real_candidate(*args, **kwargs)
        trial_id = args[4]["id"]
        score, drop = {"b03-trainer": (.98, .35), "b03-trainer-t02": (.97, .02),
                       "b03-trainer-t03": (.96, .02)}[trial_id]
        gain = score - args[7]["macro_f1"]
        scores = [{**scores[0], "candidate_macro_f1": scores[0]["entry_macro_f1"] - drop, "drop": drop}]
        failures = workflow.candidate_gate_failures(args[0], score, gain, kwargs["min_validation_f1"])
        return {**candidate, "validation": {**candidate["validation"], "macro_f1": score}}, {
            **result, "macro_f1": score, "history_drop": drop, "gate_failures": failures,
        }, not failures, gain, scores

    def test_evaluate(path, data, batch, split):
        if batch == 2:
            assert len(payloads) >= 6
            assert (next(tmp_path.iterdir()) / "reports/batch3/trainer_decision_t04.json").exists()
        return real_evaluate(path, data, batch, split)

    monkeypatch.setattr(workflow, "evaluate_candidate", evaluate_candidate)
    monkeypatch.setattr(workflow, "evaluate", test_evaluate)
    config = RunConfig(strategy="llm", data_dir=str(dataset.root), swanlab_mode="disabled",
                       trials_per_batch=4, gas_trainer_min_validation_f1=.95, evaluate_history=True)
    root, partial = run_workflow(config, runs_root=tmp_path, transport=transport, stop_after=1)
    assert not partial["complete"]
    assert partial["batches_completed"] == partial["experiments"] == 1 and partial["api_calls"] == 2
    first_report = root / "experiments/b03-trainer/training.json"
    first_stamp = first_report.stat().st_mtime_ns
    assert read_json(root / "active.json")["model_path"] == "models/baseline.npz"

    _, result = run_workflow(resume=root, transport=transport)
    assert result["complete"] and result["experiments"] == 3 and result["accepted_models"] == 1
    assert first_report.stat().st_mtime_ns == first_stamp
    assert result["max_candidates_per_batch"] == 4
    feedback = payloads[2]["batch_feedback"]
    assert feedback["remaining_candidates"] == 3
    assert "max_historical_macro_f1_drop" not in feedback
    assert feedback["min_current_validation_macro_f1"] == .95
    assert "min_current_macro_f1_gain" not in feedback
    assert feedback["attempts"][0]["training_config"] == {**params, "window_size": 3}
    assert feedback["attempts"][0]["eligible"]
    assert "worst_historical_group" not in feedback["attempts"][0]
    assert "test" not in json.dumps(payloads)

    batch = read_json(root / "audit_test_results.json")[1]
    assert batch["experiment_count"] == 3 and batch["stop_reason"] == "trainer_stop"
    assert batch["selected_trial_id"] == "b03-trainer"
    assert batch["history_validation_batches"] == ["batch2.dat"]
    assert batch["model_path"] == "experiments/b03-trainer/checkpoint.npz"
    trials = read_json(root / "history.json")["trials"]
    assert trials[0]["history_drop"] == .35 and trials[0]["gate_failures"] == []
    assert len({trial["parent_checkpoint_sha256"] for trial in trials}) == 1
    assert [trial["params"]["batch_size"] for trial in trials] == [96, 97, 98]
    assert [trial["params"]["epochs"] for trial in trials] == [2, 3, 4]


def test_trainer_absolute_gate_uses_current_validation():
    from harness_core.gas_workflow import candidate_gate_failures

    cfg = RunConfig(gas_trainer_min_validation_f1=.95)
    target = cfg.gas_trainer_min_validation_f1
    assert candidate_gate_failures(cfg, .95, .005, target) == []
    assert candidate_gate_failures(cfg, .96, -.02, target) == []
    assert candidate_gate_failures(cfg, .96, -.02, target, require_improvement=True) == [
        "当前验证 Macro-F1 未优于批初模型"]
    assert candidate_gate_failures(cfg, .949, .24, target) == ["当前验证 Macro-F1 未达到 95.00%"]
    assert candidate_gate_failures(cfg, .94, .24, None) == []
    assert candidate_gate_failures(cfg, .94, .005, None) == ["当前验证增益不足"]

def test_trainer_reads_alternate_checkpoint_dimensions(tmp_path):
    import numpy as np
    from baseline.gas.baselines.data import flatten, sha256
    from baseline.gas.baselines.model import Model, checkpoint_digest
    from baseline.gas.baselines.train import metrics
    from harness_core.gas_import import dat_line
    from harness_core.storage import atomic_json

    classes = ["gas-a", "gas-b", "gas-c"]
    groups = {"sensor-left": [1, 4], "sensor-right": [2, 3, 5]}
    rng = np.random.default_rng(82)
    labels = np.tile(np.arange(len(classes)), 30)
    x = rng.normal(size=(len(labels), 5)) + labels[:, None]
    paths = flatten({"features": x[0].tolist()})[0]
    model = Model("classification", paths, classes, x, labels, hidden=4)
    model.meta["sensor_groups"] = groups
    bundle = tmp_path / "bundle"
    public = bundle / "public"
    public.mkdir(parents=True)
    path = bundle / "checkpoint.npz"
    model.save(path)
    digest = checkpoint_digest(path)
    fingerprints = {}
    for split, source_batch, rows in (("train", 1, range(60)), ("validation", 2, range(60, 90))):
        contents = "".join(json.dumps({"id": f"public-{idx}", "inputs": {"features": x[idx].tolist()},
                                       "answer": classes[labels[idx]], "meta": {"source_batch": source_batch}}) + "\n"
                           for idx in rows).encode()
        (public / f"{split}.jsonl").write_bytes(contents)
        fingerprints[f"{split}_sha256"] = sha256(contents)
    atomic_json(bundle / "baseline-info.json", {"dataset_key": "A", "id": "alternate-shape", "checkpoint_sha256": digest})
    atomic_json(bundle / "training.json", {"profile": "A", "task": "classification", "checkpoint_sha256": digest,
                "train_rows": 60, "validation_rows": 30,
                "validation_after": metrics(model, {"x": x[60:], "y": labels[60:]}),
                "final_steps": 0, "final_epochs": 0, **fingerprints})
    raw_dir = tmp_path / "data"
    raw_dir.mkdir()
    raw = "".join(dat_line(features, int(label + 1)) for features, label in zip(x + .2, labels)).encode()
    (raw_dir / "batch3.dat").write_bytes(raw)
    atomic_json(raw_dir / "gas-import.json", {"version": 1, "batches": {"3": {
        "dat_sha256": sha256(raw).split(":")[1],
    }}})
    captured = []

    def transport(messages):
        captured.append(json.loads(messages[1]["content"]))
        return {"content": json.dumps({
            "decision": "INCREMENTAL_TRAINING", "reason": "testing alternate schema", "evaluation_requirements": [],
            "training_config": {"learning_rate": .0003, "batch_size": 16, "epochs": 2, "early_stopping_patience": 1,
                                "weight_decay": .0001, "window_size": 1, "class_weighting": True},
        })}

    config = RunConfig(strategy="llm", data_dir=str(raw_dir), baseline_dir=str(bundle),
                       swanlab_mode="disabled", trials_per_batch=1)
    root, summary = run_workflow(config, runs_root=tmp_path / "runs", transport=transport)
    assert summary["complete"] and summary["experiments"] == 1 and summary["failed_stages"] == 0
    assert read_json(root / "dataset.json")["gas_schema"] == {
        "feature_count": 5, "class_labels": classes,
    }
    drift = read_json(root / "reports/batch3/drift.json")
    # A single imported batch has no reference even when its checkpoint had sensor metadata.
    assert drift["status"] == "no_reference" and "sensors" not in drift
    assert drift["label_distribution"]["class_labels"] == classes
    assert len(drift["label_distribution"]["current"]) == 3
    assert set(captured[0]["current_validation"]["recall"]) == set(classes)


def test_drift_with_missing_labels_and_unequal_feature_dimensions():
    import numpy as np
    import pytest
    from harness_core.gas_drift import detect
    from harness_core.gas_policy import WINDOW_TRAINER_CONFIG

    ref = {"x": np.arange(60, dtype=float).reshape(20, 3), "y": np.tile([0, 1], 10)}
    cur = {"x": ref["x"] + 2, "y": None}
    cfg = {**WINDOW_TRAINER_CONFIG["drift"], "mmd_permutations": 3}
    classes = ["one", "two", "absent"]
    drift = detect(ref, cur, cfg, 42, classes=classes)
    assert len(drift["features"]) == 3
    assert "sensors" not in drift
    assert drift["label_distribution"]["status"] == "labels_unavailable"
    assert drift["label_distribution"]["current"] is None
    assert drift["label_distribution"]["js_divergence"] is None
    assert len(drift["label_distribution"]["reference"]) == 3
    with pytest.raises(ValueError, match="特征维度"):
        detect(ref, {"x": cur["x"][:, :2]}, cfg, 42, classes=classes)


def test_below_target_selects_best_safe_candidate_above_entry(dataset, tmp_path, monkeypatch):
    import harness_core.gas_workflow as workflow

    real_evaluate_candidate = workflow.evaluate_candidate
    responses = []
    params = {"learning_rate": .0003, "batch_size": 96, "epochs": 2, "early_stopping_patience": 1,
              "weight_decay": .0002, "window_size": 2, "class_weighting": True}

    def transport(messages):
        payload = json.loads(messages[1]["content"])
        responses.append(payload)
        count = len(responses)
        train = 2 <= count <= 5  # Skip batch2, then request four candidates for batch3.
        return {"content": json.dumps({
            "decision": "MIXED_DATA_TRAINING" if train else "NO_UPDATE",
            "reason": "testing the batch-end fallback", "evaluation_requirements": [],
            "training_config": {**params, "batch_size": 96 + count} if train else {},
        })}

    def evaluate_candidate(*args, **kwargs):
        candidate, trial, _, _, histories = real_evaluate_candidate(*args, **kwargs)
        trial_id = args[4]["id"]
        entry_score = args[7]["macro_f1"]
        score, drop = {
            "b03-trainer": (.83, .05),
            "b03-trainer-t02": (.88, .21),  # Highest score with the largest historical drop.
            "b03-trainer-t03": (.86, .12),
            "b03-trainer-t04": (entry_score, .01),  # A tie with the entry model.
        }[trial_id]
        gain = score - entry_score
        failures = workflow.candidate_gate_failures(args[0], score, gain, kwargs["min_validation_f1"])
        return {**candidate, "validation": {**candidate["validation"], "macro_f1": score}}, {
            **trial, "macro_f1": score, "history_drop": drop, "gate_failures": failures,
        }, not failures, gain, histories

    monkeypatch.setattr(workflow, "evaluate_candidate", evaluate_candidate)
    config = RunConfig(strategy="llm", data_dir=str(dataset.root), swanlab_mode="disabled",
                       trials_per_batch=4, gas_trainer_min_validation_f1=.90)
    root, summary = run_workflow(config, runs_root=tmp_path, transport=transport)
    assert summary["complete"] and summary["accepted_models"] == 1
    batch = next(row for row in read_json(root / "audit_test_results.json") if row["source_file"] == "batch3.dat")
    assert [row["fallback_eligible"] for row in batch["trainer_attempts"]] == [True, True, True, False]
    assert batch["selected_below_target"] and batch["selected_trial_id"] == "b03-trainer-t02"
    assert "未达目标" in (root / "batch_results.md").read_text()
