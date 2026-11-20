import json

from harness_core.config import RunConfig
from harness_core.rul_policy import BearingTrainer
from harness_core.rul_workflow import choose_candidate, gate_failures
from harness_core.runner import run_workflow
from harness_core.storage import Ledger, read_json
from wrappers.bearing_baseline import digest


def test_bearing_relative_thresholds():
    config = RunConfig(dataset="bearing_rul").validate()
    assert gate_failures(config, 100, 99, [{"entry_rmse": 80, "candidate_rmse": 120}]) == []
    assert gate_failures(config, 100, 98, [{"entry_rmse": 80, "candidate_rmse": 120.01}]) == [
        "history_regression"]
    assert gate_failures(config, 100, 99.01, []) == ["current_gain_insufficient"]
    assert gate_failures(config, 0, 0, []) == ["current_gain_insufficient"]
    assert gate_failures(config, 100, 99.5, [], target=99.5) == []
    assert gate_failures(config, 100, 99.5, [], target=90) == ["validation_target_not_met"]
    assert gate_failures(config, 100, 101, [], target=200) == ["current_gain_insufficient"]


def test_bearing_prefers_target_and_only_falls_back_to_safe_improvement():
    safe_below_target = {"status": "success", "rmse": 95, "gate_failures": ["validation_target_not_met"]}
    eligible = {"status": "success", "rmse": 89, "gate_failures": []}
    damages_history = {"status": "success", "rmse": 80, "gate_failures": ["history_regression"]}
    assert choose_candidate([damages_history, safe_below_target, eligible]) is eligible
    assert choose_candidate([damages_history, safe_below_target]) is safe_below_target
    assert choose_candidate([damages_history]) is None


def test_bearing_fixed_responses_resume_feedback_budget_and_history_gate(tmp_path):
    seen = []

    def transport(messages):
        context = json.loads(messages[1]["content"])
        seen.append(context)
        assert "static_validation" not in context
        assert "本批入口模型" in messages[0]["content"]
        if context["batch_feedback"]["attempts"]:
            output = {"decision": "NO_UPDATE", "reason": "结束反馈搜索", "training_config": {}}
        else:
            mixed = bool(context["replay_batches"])
            output = {"decision": "MIXED_DATA_TRAINING" if mixed else "INCREMENTAL_TRAINING",
                      "reason": "验证固定窗口方案", "training_config": {
                          "learning_rate": 0.001, "batch_size": 128, "epochs": 5,
                          "window_size": 4 if mixed else 1}}
        return {"content": json.dumps(output), "prompt_tokens": 10, "completion_tokens": 8}

    config = RunConfig(dataset="bearing_rul", strategy="llm", swanlab_mode="disabled",
                       data_dir="baseline/cpu-b-rul-20261110-v1",
                       max_experiments=2, max_api_calls=3, rul_max_history_increase=0.05)
    partial = run_workflow(config, runs_root=tmp_path, transport=transport, stop_after=1)
    assert not partial["complete"] and partial["batches_completed"] == 0
    assert partial["experiments"] == len(seen) == 1
    root = next(tmp_path.iterdir())
    path = root / "experiments/b01-trainer/result/checkpoint.npz"
    fingerprint = digest(path), path.stat().st_mtime_ns
    result = run_workflow(resume=root, transport=transport)
    assert (digest(path), path.stat().st_mtime_ns) == fingerprint
    assert result["complete"] and result["experiments"] == 2
    assert result["api_calls"] == len(seen) == 3
    assert seen[0]["batch_feedback"]["attempts"] == []
    assert seen[0]["batch_feedback"]["validation_target_rmse"] is None
    feedback = seen[1]["batch_feedback"]["attempts"][0]
    assert feedback["status"] == "success"
    assert feedback["rmse"] < seen[1]["current_validation"]["rmse"]
    one, two, three = result["batches"]
    assert one["promoted"] and one["stop_reason"] == "trainer_no_update"
    assert not two["promoted"] and not three["promoted"]
    assert two["validation_target_rmse"] == one["validation_rmse"]
    assert three["validation_target_rmse"] == (one["validation_rmse"] + two["validation_rmse"]) / 2
    assert "history_regression" in two["trainer_attempts"][0]["gate_failures"]
    historical = two["trainer_attempts"][0]["history_validation"][0]
    assert historical["entry_rmse"] == one["validation_rmse"]
    assert historical["relative_increase"] > config.rul_max_history_increase
    assert two["selected"] == one["selected"] == three["entry"]
    assert three["stop_reason"] == "experiment_budget" and three["trainer_attempts"] == []
    first_report = read_json(path.with_name("training.json"))
    second_path = root / "experiments/b02-trainer/result/training.json"
    second_report = read_json(second_path)
    assert second_report["parent_checkpoint_sha256"] == first_report["checkpoint_sha256"]
    assert second_report["initial_steps"] == first_report["final_steps"]
    assert two["trainer_attempts"][0]["training_batches"] == ["condition1", "condition2"]
    manifest = read_json(root / "dataset.json")
    assert sum(len(group["train_entities"]) for group in manifest["groups"]) == 9
    assert sum(len(group["validation_entities"]) for group in manifest["groups"]) == 3
    train = {bearing for group in manifest["groups"] for bearing in group["train_entities"]}
    valid = {bearing for group in manifest["groups"] for bearing in group["validation_entities"]}
    assert not train & valid
    assert "label_distribution" not in two["feature_drift"]


def test_bearing_api_budget_fallback_is_ledgered(tmp_path):
    def unexpected_call(messages):
        raise AssertionError("API budget is exhausted")

    ledger = Ledger(tmp_path / "budget.sqlite")
    try:
        planner = BearingTrainer(RunConfig(dataset="bearing_rul", max_api_calls=0), ledger, unexpected_call)
        context = {"replay_batches": ["condition1"], "batch_feedback": {"attempts": []},
                   "new_rows": 100, "current_validation": {"rmse": 100}, "drift": {"feature_count": 1}}
        first = planner.decide("b02-trainer", context)
        assert first["source"] == "rules_fallback"
        assert first["output"]["decision"] == "MIXED_DATA_TRAINING"
        feedback = {**context, "batch_feedback": {"attempts": [first["output"]]}}
        assert planner.decide("b02-trainer-t02", feedback)["output"]["decision"] == "NO_UPDATE"
        assert ledger.api_calls() == 0
    finally:
        ledger.close()


def test_bearing_can_skip_first_candidate_on_a_valid_model_decision(tmp_path):
    def transport(messages):
        assert "尚未试训" in messages[0]["content"]
        response = {"decision": "NO_UPDATE", "reason": "本批只有两条标签", "training_config": {}}
        return {"content": json.dumps(response)}

    ledger = Ledger(tmp_path / "skip.sqlite")
    try:
        planner = BearingTrainer(RunConfig(dataset="bearing_rul"), ledger, transport)
        plan = planner.decide("b01-trainer", {
            "replay_batches": [], "batch_feedback": {"attempts": []}, "new_rows": 2,
        })
        assert plan["source"] == "trainer_llm" and plan["output"]["decision"] == "NO_UPDATE"
        assert ledger.api_calls() == 1
    finally:
        ledger.close()
