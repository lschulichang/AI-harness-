import shutil

import pytest

from harness_demo.config import RunConfig
from harness_demo.portability import resolve_model_path, saved_data_path
from harness_demo.storage import atomic_json, read_json
from harness_demo.workflow import run_workflow


def test_project_data_is_saved_as_relative_path(tmp_path):
    project = tmp_path / "AI+harness"
    assert saved_data_path(project / "data/gas", project) == "data/gas"


def test_modified_baseline_is_rejected_on_resume(tmp_path, official_bundle, dataset):
    bundle = shutil.copytree(official_bundle, tmp_path / "gas")
    config = RunConfig(data_dir=str(dataset.root), baseline_dir=str(bundle),
                       swanlab_mode="disabled", strategy="rules")
    root, _ = run_workflow(config, runs_root=tmp_path, stop_after=1)
    with (bundle / "public/train.jsonl").open("a", encoding="utf-8") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="不一致"):
        run_workflow(resume=root)


def test_windows_model_separator_and_escape_check(tmp_path):
    root = tmp_path / "run"
    assert resolve_model_path(root, r"models\baseline.npz") == (root / "models/baseline.npz").resolve()
    with pytest.raises(ValueError, match="超出"):
        resolve_model_path(root, r"..\outside.npz")


def test_moved_window_run_resumes_saved_config_without_retraining(tmp_path, monkeypatch, official_bundle, dataset):
    import harness_demo.gas_mlp as workflow

    project = tmp_path / "AI+harness"
    shutil.copytree(official_bundle, project / "baseline/gas")
    shutil.copytree(dataset.root, project / "data/gas")
    monkeypatch.setattr(workflow, "project_root", lambda: project)
    config = RunConfig(data_dir="data/gas", swanlab_mode="disabled", strategy="rules", max_experiments=3)
    root, paused = run_workflow(config, runs_root=project / "runs", stop_after=1)
    assert not paused["complete"] and paused["experiments"] == 1
    assert read_json(root / "config.json")["data_dir"] == "data/gas"
    frozen_history_drop = read_json(root / "dataset.json")["gas_trainer_policy"]["max_history_drop"]
    saved = read_json(root / "config.json")
    saved.update(gas_policy="trainer", gas_target_macro_f1=.95, gas_sensor_groups=None,
                 depths=[2, 3, 4], leaves=[1, 2], windows=["recent", "all"],
                 gas_max_extension_stages=3, drift_drop=.05, max_history_drop=.3)
    atomic_json(root / "config.json", saved)
    moved = tmp_path / "renamed_AI+harness"
    shutil.move(str(project), str(moved))
    monkeypatch.setattr(workflow, "project_root", lambda: moved)
    root = moved / "runs" / root.name
    model = root / "experiments/b02-trainer/checkpoint.npz"
    old_mtime = model.stat().st_mtime_ns
    _, completed = run_workflow(resume=root)
    assert completed["complete"] and completed["experiments"] >= 1
    assert completed["gas_trainer_max_history_drop"] == frozen_history_drop == config.max_history_drop
    assert model.stat().st_mtime_ns == old_mtime
