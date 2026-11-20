import json

from harness_demo.config import RunConfig
from harness_demo.storage import read_json
from harness_demo.workflow import run_workflow
from wrappers.cmapss_baseline import CmapssBaseline


def test_cmapss_uses_engine_splits_and_tests_only_after_selection(tmp_path, monkeypatch):
    observed = []
    evaluated = []
    original_test_samples = CmapssBaseline.test_samples

    def test_samples(adapter, batch, root):
        result = original_test_samples(adapter, batch, root)
        evaluated.append((adapter.group_names[batch], len(result["y"])))
        return result

    monkeypatch.setattr(CmapssBaseline, "test_samples", test_samples)

    def proposer(messages):
        context = json.loads(messages[1]["content"])
        assert "涡扇发动机" in messages[0]["content"]
        assert context["unit"] == "cycles" and context["condition"] == "FD001"
        assert "public_test" not in context and not context["batch_feedback"]["attempts"]
        observed.append(context)
        return {"content": json.dumps({
            "decision": "INCREMENTAL_TRAINING", "reason": "用 FD001 当前批训练",
            "training_config": {"learning_rate": 0.001, "batch_size": 256, "epochs": 1, "window_size": 1},
        })}

    config = RunConfig(dataset="cmapss_rul", strategy="llm", swanlab_mode="disabled",
                       max_experiments=1, max_api_calls=1)
    partial = run_workflow(config, runs_root=tmp_path, transport=proposer, stop_after=1)
    assert not partial["complete"] and "public_test" not in partial and not evaluated
    root = next(tmp_path.iterdir())
    candidate = root / "experiments/b01-trainer/result/training.json"
    before = candidate.stat().st_mtime_ns

    result = run_workflow(resume=root, transport=proposer)
    assert len(observed) == 1 and candidate.stat().st_mtime_ns == before
    assert result["complete"] and result["experiments"] == 1 and result["failed_stages"] == 0
    assert result["unit"] == "cycles"
    assert evaluated == [("FD001", 100), ("FD002", 259), ("FD003", 100), ("FD004", 248)]
    assert result["public_test"]["pooled"]["final"]["samples"] == sum(count for _, count in evaluated)
    assert result["public_test"]["pooled"]["final"]["nasa_score"] > 0

    data = read_json(root / "dataset.json")
    assert data["initial_checkpoint_already_saw_all_public_training"]
    for group in data["groups"]:
        assert set(group["train_entities"]).isdisjoint(group["validation_entities"])

    report = read_json(candidate)
    assert report["profile"] == "C" and report["input_features"] == 24
    assert report["parent_checkpoint_sha256"] == data["checkpoint_sha256"]
    assert report["final_steps"] > report["initial_steps"]
    assert result["batches"][0]["trainer_attempts"][0]["training_batches"] == ["FD001"]
