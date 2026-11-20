import json

from harness_core.config import RunConfig
from harness_core.storage import Ledger
from harness_core.tracking import Tracker


def test_cloud_log_contains_batch_test_and_final_summary(tmp_path, monkeypatch):
    import swanlab

    events = []

    class Session:
        def log(self, values, step):
            events.append((step, values))

        def finish(self, **kwargs):
            pass

    monkeypatch.setattr(swanlab, "init", lambda **kwargs: Session())
    monkeypatch.setattr(swanlab, "Text", lambda value: value)
    ledger = Ledger(tmp_path / "experiments.sqlite")
    trial = {
        "id": "b03-trainer", "batch": 2, "status": "success", "source": "deepseek",
        "params": {"learning_rate": .001, "batch_size": 128, "epochs": 1, "window_size": 2},
        "training_seconds": .1, "macro_f1": .9,
    }
    decision = {"accepted": True, "scope": "candidate_eligibility", "eligible": True, "reason": "候选通过门槛"}
    batch = {
        "batch": 2, "source_file": "batch3.dat", "before_validation_macro_f1": .82, "validation_macro_f1": .9,
        "before_test_macro_f1": .8, "test_macro_f1": .85,
        "before_test_accuracy": .84, "test_accuracy": .88,
        "training_sample_passes": 120,
        "experiment_count": 1, "promoted": False, "stop_reason": "trainer_stop",
        "trainer_min_validation_f1": .95,
        "update_test_gain": .05, "decision_seconds": 1.2, "cpu_seconds": .6,
        "peak_rss_mib": 42.0, "profile_complete": True,
        "feature_drift": {"mean_psi": .25, "threshold": .2, "detected": True, "feature_psi": [.25]},
    }
    for request in (
        {"id": "first", "trial_id": "b03-trainer", "status": "received", "prompt_tokens": 20, "completion_tokens": 10},
        {"id": "second", "trial_id": "b03-w01", "status": "error"},
        {"id": "next", "trial_id": "b04-trainer", "status": "received", "prompt_tokens": 100, "completion_tokens": 50},
    ):
        ledger.reserve_request(request["id"], 12, request)
    ledger.put("trial", trial["id"], trial)
    ledger.put("batch", "02", batch)
    tracker = Tracker(RunConfig(swanlab_mode="online"), tmp_path, ledger, metric="macro_f1")
    tracker.log_trial(trial, decision)
    tracker.log_batch(batch)
    summary = {
        "complete": True, "mean_test_macro_f1": .85,
        "worst_test_macro_f1": .85, "accepted_models": 0, "experiments": 1, "batches_completed": 1,
        "mean_test_accuracy": .88, "prompt_tokens": 20, "completion_tokens": 10,
    }
    tracker.log_final({**summary, "complete": False})
    tracker.log_final(summary)
    tracker.log_final(summary)
    tracker.close()
    assert len(events) == 3
    assert [step for step, _ in events] == sorted({step for step, _ in events})
    trial_metrics, batch_metrics, final_metrics = (values for _, values in events)
    assert json.loads(trial_metrics["trial/details"])["eligible"] is True
    assert json.loads(trial_metrics["trial/details"])["trial_id"] == trial["id"]
    assert "budget/prompt_tokens" not in trial_metrics
    assert batch_metrics["batch/before_test_macro_f1"] == .8
    assert batch_metrics["batch/test_macro_f1"] == .85
    assert batch_metrics["batch/update_test_gain"] == .05
    assert batch_metrics["batch/before_validation_macro_f1"] == .82
    assert batch_metrics["batch/validation_macro_f1"] == .9
    assert batch_metrics["batch/test_accuracy"] == .88
    assert batch_metrics["batch/trainer_min_validation_f1"] == .95
    assert batch_metrics["batch/training_sample_passes"] == 120
    assert batch_metrics["batch/total_tokens"] == 30
    assert batch_metrics["batch/decision_seconds"] == 1.2
    assert batch_metrics["batch/api_calls"] == 2
    assert batch_metrics["batch/completion_tokens"] == 10
    assert batch_metrics["batch/requests_with_unknown_usage"] == 1
    details = json.loads(batch_metrics["batch/details"])
    assert details["stop_reason"] == "trainer_stop"
    assert details["cpu_seconds"] == .6
    assert details["peak_rss_mib"] == 42.0
    assert batch_metrics["batch/feature_psi"] == .25
    assert batch_metrics["batch/drift_detected"] == 1
    assert "feature_psi" not in details["feature_drift"]
    assert final_metrics["final/mean_test_macro_f1"] == .85
    assert final_metrics["final/worst_test_macro_f1"] == .85
    assert final_metrics["final/accepted_models"] == 0
    assert final_metrics["final/mean_test_accuracy"] == .88
    assert final_metrics["final/total_tokens"] == 30
    assert ledger.items("tracking") == []
    ledger.close()
