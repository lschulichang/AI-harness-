"""主线验收覆盖数据隔离、提案校验、失败恢复与请求计数。"""

import json

import numpy as np
import pytest

from harness_demo.config import RunConfig
from harness_demo.gas_trainer_policy import WINDOW_TRAINER_CONFIG, validate
from harness_demo.planner import OpenAITransport, Planner
from harness_demo.storage import Ledger, read_json
from harness_demo.tracking import Tracker
from harness_demo.workflow import run_workflow


def test_split_and_stream_are_reproducible_and_disjoint(dataset, gas_data, tmp_path):
    duplicate = gas_data(tmp_path / "copy")
    for batch in range(dataset.batch_count):
        all_ids, lengths = set(), []
        size = 250 if batch == 0 else 100
        for split in ("train", "validation", "test"):
            x, y, ids = dataset.load(batch, split)
            x2, y2, ids2 = duplicate.load(batch, split)
            np.testing.assert_array_equal(x, x2)
            np.testing.assert_array_equal(y, y2)
            np.testing.assert_array_equal(ids, ids2)
            assert not all_ids.intersection(ids)
            all_ids.update(ids)
            lengths.append(len(y))
        assert lengths == [int(size * .6), int(size * .2), int(size * .2)]


@pytest.mark.parametrize("changes", [
    {"learning_rate": True}, {"learning_rate": 999}, {"batch_size": 2.0},
    {"epochs": "future"}, {"window_size": 0}, {"window_size": 9}, {"window_size": True},
])
def test_invalid_parameters_are_rejected(changes):
    params = {**WINDOW_TRAINER_CONFIG["training"], **changes}
    proposal = {"decision": "MIXED_DATA_TRAINING", "reason": "test",
                "training_config": params, "evaluation_requirements": []}
    with pytest.raises(ValueError):
        validate(proposal, WINDOW_TRAINER_CONFIG, require_complete=True)


@pytest.mark.parametrize("strategy", ["static", "fixed"])
def test_retired_comparison_modes_are_rejected(strategy):
    with pytest.raises(ValueError, match="旧对照"):
        RunConfig(strategy=strategy).validate()


def test_saved_window_config_drops_retired_knobs():
    config = RunConfig.from_saved_options({
        "gas_policy": "trainer", "gas_target_macro_f1": .95, "gas_sensor_groups": None,
        "depths": [2, 3], "leaves": [1], "windows": ["recent"], "gas_max_extension_stages": 3,
        "drift_drop": .05, "max_api_calls": 4, "gas_sample_budget": 1200,
    }).validate()
    assert config.max_api_calls == 4 and config.gas_sample_budget == 1200
    assert "depths" not in config.to_dict()
    with pytest.raises(ValueError, match="已移除"):
        RunConfig.from_saved_options({"gas_policy": "legacy"})


def test_missing_api_key_fails_before_run_creation(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTRIO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="尚未配置 PYTRIO_API_KEY"):
        run_workflow(RunConfig(), runs_root=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_api_failure_counts_request_and_labels_fallback(dataset, tmp_path):
    def fail(messages):
        raise TimeoutError("secret_should_not_be_logged")

    config = RunConfig(data_dir=str(dataset.root), swanlab_mode="disabled",
                       max_api_calls=1, max_experiments=1)
    root, summary = run_workflow(config, runs_root=tmp_path, transport=fail)
    assert summary["complete"] and summary["api_calls"] == 1
    assert summary["fallback_experiments"] == 1 and summary["llm_proposals"] == 0
    assert summary["requests_with_unknown_usage"] == 1
    assert "secret_should_not_be_logged" not in (root / "api_requests.json").read_text()


def test_invalid_json_is_repaired_only_once(dataset, tmp_path):
    calls = []

    def invalid(messages):
        calls.append(messages)
        return {"content": "not json"}

    config = RunConfig(data_dir=str(dataset.root), swanlab_mode="disabled",
                       max_api_calls=2, max_experiments=0)
    _, summary = run_workflow(config, runs_root=tmp_path, transport=invalid)
    assert summary["complete"] and summary["api_calls"] == len(calls) == 2
    assert len(calls[1]) == 3 and summary["fallback_decisions"] == 9


def test_reserved_request_is_not_reissued(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    cfg = RunConfig(max_api_calls=1)
    ledger.reserve_request("x_request_0", 1, {"id": "x_request_0", "trial_id": "x", "status": "reserved"})
    calls = []

    def transport(messages):
        calls.append(messages)
        return {"content": "{}"}

    result, error = Planner(cfg, ledger, transport)._request_proposal("x", "json", {}, lambda value: value)
    assert result is None and "预算" in error
    assert ledger.api_calls() == 1 and not calls
    ledger.close()


def test_failed_candidate_keeps_model_and_continues(dataset, tmp_path, monkeypatch):
    import harness_demo.gas_trainer_workflow as workflow

    def fail(*args):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(workflow, "train_candidate", fail)
    config = RunConfig(strategy="rules", data_dir=str(dataset.root), swanlab_mode="disabled", max_experiments=1)
    root, summary = run_workflow(config, runs_root=tmp_path)
    assert summary["complete"] and summary["failed_stages"] == 1
    assert summary["experiments"] == 1 and summary["accepted_models"] == 0
    assert read_json(root / "active.json")["model_path"] == "models/baseline.npz"
    trial = read_json(root / "history.json")["trials"][0]
    assert trial["error_type"] == "RuntimeError" and trial["attempt_seconds"] >= 0


def test_empty_training_budget_finishes_without_attempts(dataset, tmp_path):
    config = RunConfig(strategy="rules", data_dir=str(dataset.root), swanlab_mode="disabled", gas_sample_budget=0)
    _, summary = run_workflow(config, runs_root=tmp_path)
    assert summary["complete"] and summary["experiments"] == summary["api_calls"] == 0
    assert summary["reserved_training_sample_passes"] == 0


def test_legacy_resume_is_rejected_without_changing_results(tmp_path):
    root = tmp_path / "old-run"
    root.mkdir()
    dataset = root / "dataset.json"
    dataset.write_text('{"gas_search_policy":"sample_budget_v1"}')
    before = dataset.read_bytes()
    with pytest.raises(ValueError, match="旧版"):
        run_workflow(resume=root)
    assert dataset.read_bytes() == before
    assert sorted(path.name for path in root.iterdir()) == ["dataset.json"]


def test_competition_mode_cannot_accidentally_call_deepseek():
    with pytest.raises(ValueError, match="官方 SDK"):
        RunConfig(competition=True).validate()
    with pytest.raises(ValueError, match="PyTRiO"):
        RunConfig(provider="pytrio", base_url="https://api.deepseek.com").validate()


def test_default_pytrio_configuration():
    cfg = RunConfig().validate()
    assert (cfg.provider, cfg.model) == ("pytrio", "Qwen/Qwen3.8-27B")
    assert cfg.base_url == "https://pytrio.com/api/openai/v1"
    assert cfg.api_key_env == "PYTRIO_API_KEY"


def test_deepseek_client_uses_json_and_no_hidden_retries(monkeypatch):
    import httpx
    from openai import OpenAI
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"id":"mock", "object":"chat.completion", "created":0, "model":"deepseek-flash",
            "choices":[{"index":0,"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":13,"completion_tokens":5,"total_tokens":18}})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake_for_local_mock_only")
    transport = OpenAITransport(RunConfig(provider="deepseek"))
    transport.client = OpenAI(api_key="fake_for_local_mock_only", base_url="https://mock.invalid", max_retries=0,
                              http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    response = transport([{"role":"user","content":"json"}])
    assert response["prompt_tokens"] == 13 and response["completion_tokens"] == 5
    assert seen[0]["response_format"] == {"type":"json_object"}
    assert seen[0]["thinking"] == {"type":"disabled"}
    assert "max_tokens" not in seen[0]
    transport([{"role": "user", "content": "json"}], max_tokens=40)
    assert seen[1]["max_tokens"] == 40
    assert transport.client.max_retries == 0
    transport.client.close()


def test_swanlab_logs_metrics_without_hardware_probe(tmp_path, monkeypatch):
    import swanlab
    captured = {}
    class Session:
        def log(self, metrics, step):
            captured["metrics"], captured["step"] = metrics, step
        def finish(self, **kwargs):
            captured["finished"] = True
    def init(**kwargs):
        captured["settings"] = kwargs["settings"]
        return Session()
    monkeypatch.setattr(swanlab, "init", init)
    ledger = Ledger(tmp_path/"ledger.sqlite")
    tracker = Tracker(RunConfig(), tmp_path, ledger, metric="accuracy")
    trial = {"id":"x", "batch":0, "params":dict(WINDOW_TRAINER_CONFIG["training"]),
             "status":"success", "accuracy":.8, "history_accuracy":None, "training_seconds":.1, "source":"rules"}
    ledger.put("trial", "x", trial)
    tracker.log_trial(trial, {"accepted":False,"reason":"保持"})
    tracker.close()
    assert not captured["settings"].probe.monitor and not captured["settings"].probe.hardware
    assert captured["metrics"]["validation/accuracy"] == .8
    assert "budget/prompt_tokens" not in captured["metrics"]
    assert "trial/details" in captured["metrics"]
    assert captured["finished"]
    ledger.close()


@pytest.mark.parametrize("invalid_action,invalid_window", [
    ("COLLECT_MORE_DATA", None),
    ("INCREMENTAL_TRAINING", 0),
])
def test_invalid_action_or_zero_window_is_repaired_before_training(dataset, tmp_path, invalid_action, invalid_window):
    calls = []

    def transport(messages):
        calls.append(messages)
        params = {**WINDOW_TRAINER_CONFIG["training"], "epochs": 2,
                  "early_stopping_patience": 1, "window_size": 1}
        if len(calls) == 1:
            action = invalid_action
            params = {} if invalid_window is None else {**params, "window_size": invalid_window}
        else:
            action = "INCREMENTAL_TRAINING"
            assert len(messages) == 3
        return {"content": json.dumps({"decision": action, "reason": "test correction",
                                        "training_config": params, "evaluation_requirements": []})}

    config = RunConfig(data_dir=str(dataset.root), swanlab_mode="disabled",
                       trials_per_batch=1, max_api_calls=2, max_experiments=1)
    root, summary = run_workflow(config, runs_root=tmp_path, transport=transport)
    assert summary["complete"] and summary["api_calls"] == len(calls) == 2
    assert summary["experiments"] == summary["llm_proposals"] == 1
    trial = read_json(root / "history.json")["trials"][0]
    assert trial["params"]["window_size"] == 1
    for path in (root / "reports").glob("*/trainer_decision*.json"):
        assert read_json(path)["output"]["decision"] in {
            "NO_UPDATE", "INCREMENTAL_TRAINING", "MIXED_DATA_TRAINING",
        }


def test_rule_fallback_uses_dynamic_target_and_keeps_small_batches():
    from harness_demo.gas_trainer_policy import rules

    context = {
        "performance": {"current": {"macro_f1": .9}},
        "validation_target": .85,
        "data": {"new_labeled_samples": 100, "historical_replay_available": True},
        "drift_report": {"drifted_features": []},
    }
    assert rules(context, WINDOW_TRAINER_CONFIG)["decision"] == "NO_UPDATE"
    context["performance"]["current"]["macro_f1"] = .7
    assert rules(context, WINDOW_TRAINER_CONFIG)["decision"] == "MIXED_DATA_TRAINING"
    context["data"]["new_labeled_samples"] = 20
    assert rules(context, WINDOW_TRAINER_CONFIG)["decision"] == "NO_UPDATE"
