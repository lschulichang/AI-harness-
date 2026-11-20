"""将实验账本中的关键指标和审计详情写入 SwanLab。"""

import json
from pathlib import Path

from .credentials import get_secret
from .planner import request_usage_known


class Tracker:
    def __init__(self, config, run_dir, ledger, *, metric=None, run_metadata=None):
        self.root, self.ledger = Path(run_dir), ledger
        self.metric = metric or "macro_f1"
        self.unlimited = config.max_experiments is None
        metadata = run_metadata or {}
        trainer_policy = (metadata.get("gas_trainer_policy") or metadata.get("bearing_trainer_policy")
                          or metadata.get("cmapss_trainer_policy") or {})
        batch_slots = trainer_policy.get("max_candidates_per_batch", config.candidate_limit)
        self.stride = (batch_slots if self.unlimited else config.max_experiments) + 2
        self.session = None
        if config.swanlab_mode != "disabled":
            import swanlab

            fields = (
                "dataset", "strategy", "seed", "min_gain", "max_history_drop",
                "gas_sample_budget", "max_experiments", "trials_per_batch", "max_api_calls",
                "max_api_output_tokens", "max_output_tokens", "api_timeout", "train_timeout", "swanlab_mode",
            )
            settings = config.to_dict()
            logged_config = {key: settings[key] for key in fields}
            if config.strategy == "llm":
                logged_config.update(provider=config.provider, model=config.model)
            logged_config.update(run_metadata or {})
            try:
                self.session = swanlab.init(
                    project=config.swanlab_project, workspace=config.swanlab_workspace, name=self.root.name,
                    id=self.root.name, resume="allow", mode=config.swanlab_mode,
                    log_dir=str(self.root / "swanlog"), config=logged_config, reinit=True,
                    on_online_error="offline", settings=swanlab.Settings(api_key=get_secret("SWANLAB_API_KEY"), probe={
                        "hardware": False, "monitor": False, "requirements": False, "git": False}),
                )
            except Exception as exc:
                self.ledger.put("tracking", "init_error", {"error_type": type(exc).__name__, "local_records_available": True})
                print("SwanLab 初始化失败，本地完整记录仍会保存。", flush=True)

    def _log(self, values, step, event_id):
        try:
            self.session.log(values, step=step)
            return True
        except Exception as exc:
            self.ledger.put("tracking", event_id, {"error_type": type(exc).__name__})
            return False

    def log_trial(self, trial, decision):
        if self.session is None:
            return
        import swanlab

        batch = trial["batch"]
        if self.unlimited:
            offset = sum(row["batch"] == batch for row in self.ledger.items("trial"))
        else:
            offset = self.ledger.count("trial")
        step = batch * self.stride + offset
        fields = (
            "batch", "source", "status", "params", "stage", "training_batches", "train_rows",
            "parent_checkpoint_sha256", "checkpoint_sha256", "training_seconds", "attempt_seconds",
            "sample_passes", "history_drop", "history_validation", "current_relative_gain",
            "error_type", f"history_{self.metric}",
        )
        details = {key: trial[key] for key in fields if trial.get(key) is not None}
        proposal = self.ledger.get("proposal", trial["id"]) or {}
        details.update(
            trial_id=trial["id"], reason=decision["reason"], eligible=decision.get("eligible"),
            decision_scope=decision.get("scope"), hypothesis=proposal.get("hypothesis"),
            fallback_reason=proposal.get("fallback_reason") or None,
        )
        values = {"experiment/success": int(trial["status"] == "success")}
        if trial["status"] == "success":
            values[f"validation/{self.metric}"] = trial[self.metric]
            if "mae" in trial:
                values["validation/mae"] = trial["mae"]
        if decision.get("scope") != "candidate_eligibility":
            details["accepted"] = decision["accepted"]
        values["trial/details"] = swanlab.Text(json.dumps(details, ensure_ascii=False))
        self._log(values, step, trial["id"])

    def log_batch(self, result):
        """报告批末指标。评估和 API 调用详情按批次记录。"""
        if self.session is None:
            return
        import swanlab

        batch = result["batch"]
        values = {"batch/id": batch, "batch/experiment_count": result["experiment_count"]}
        fields = (
            f"before_validation_{self.metric}", f"validation_{self.metric}",
            "static_validation_rmse", "static_validation_mae", "before_validation_mae", "validation_mae",
            "max_history_relative_increase", "validation_target_rmse",
            f"before_test_{self.metric}", f"test_{self.metric}",
            "before_test_accuracy", "test_accuracy",
            "update_test_gain",
            "test_samples", "promoted", "decision_seconds", "training_sample_passes",
            "trainer_min_validation_f1",
        )
        for key in fields:
            if result.get(key) is not None:
                values[f"batch/{key}"] = int(result[key]) if key == "promoted" else result[key]

        def is_batch_request(request):
            trial_id = request.get("trial_id", "")
            return trial_id.startswith(f"b{batch + 1:02d}-")

        requests = [request for request in self.ledger.requests() if is_batch_request(request)]
        values["batch/api_calls"] = len(requests)
        for key in ("prompt_tokens", "completion_tokens"):
            values[f"batch/{key}"] = sum(request.get(key, 0) for request in requests)
        values["batch/total_tokens"] = values["batch/prompt_tokens"] + values["batch/completion_tokens"]
        values["batch/requests_with_unknown_usage"] = sum(not request_usage_known(request) for request in requests)

        detail_fields = (
            "source_file", "train_samples", "validation_samples", f"before_validation_{self.metric}",
            f"validation_{self.metric}", "before_test_accuracy", "test_accuracy",
            "stop_reason",
            "cpu_seconds", "peak_rss_mib", "profile_complete",
            "trainer_action", "trainer_source", "trainer_reason", "selected_trial_id", "selected_below_target",
            "trainer_last_action", "trainer_last_reason", "trainer_min_validation_f1",
            "trainer_attempts", "drift_reference_batches", "history_validation_batches", "test_protocol",
            "trainer_validation_target_batches", "selection_reason", "training_batches", "history_validation",
        )
        details = {key: result[key] for key in detail_fields if result.get(key) is not None}
        drift = result.get("feature_drift")
        if drift is not None:
            if drift["mean_psi"] is not None:
                values["batch/feature_psi"] = drift["mean_psi"]
                values["batch/drift_detected"] = int(drift["detected"])
            details["feature_drift"] = {key: value for key, value in drift.items() if key != "feature_psi"}
        details["batch_id"] = batch
        details["entry_model"] = result.get("before_model_path")
        if result.get("selected") is not None:
            details["selected_model"] = result["selected"].get("checkpoint_sha256")
        details["api_requests"] = [
            {**{key: request[key] for key in (
                "id", "model", "status", "elapsed_seconds", "error_type", "prompt_tokens", "completion_tokens",
            ) if key in request}, "usage_known": request_usage_known(request)}
            for request in requests
        ]
        values["batch/details"] = swanlab.Text(json.dumps(details, ensure_ascii=False))
        self._log(values, batch * self.stride + self.stride - 1, f"batch_{batch}")

    def log_final(self, summary):
        if self.session is None or not summary["complete"] or self.ledger.get("tracking_export", "final"):
            return
        metric = self.metric
        fields = (
            "batches_completed",
            f"mean_before_test_{metric}", f"mean_test_{metric}",
            f"worst_before_test_{metric}", f"worst_test_{metric}",
            "mean_before_test_accuracy", "mean_test_accuracy",
            "mean_static_validation_rmse", "mean_before_validation_rmse", "mean_validation_rmse",
            "worst_validation_rmse", "mean_static_validation_mae", "mean_before_validation_mae",
            "mean_validation_mae", "max_history_relative_increase",
            "experiments", "accepted_models", "failed_stages",
            "training_sample_passes", "training_seconds", "total_decision_seconds",
            "mean_feature_psi", "drift_profiled_batches", "drift_detected_batches",
            "api_calls", "llm_proposals", "fallback_experiments", "prompt_tokens", "completion_tokens",
            "trainer_decisions", "llm_decisions", "fallback_decisions", "reserved_training_sample_passes",
            "requests_with_unknown_usage",
        )
        values = {f"final/{key}": summary[key] for key in fields if summary.get(key) is not None}
        if "prompt_tokens" in summary and "completion_tokens" in summary:
            values["final/total_tokens"] = summary["prompt_tokens"] + summary["completion_tokens"]
        if public_test := summary.get("public_test"):
            for stage in ("static", "final"):
                for metric, value in public_test["pooled"][stage].items():
                    if value is not None:
                        values[f"public_test/{stage}_{metric}"] = value
        last_batch = max((row["batch"] for row in self.ledger.items("trial") + self.ledger.items("batch")), default=0)
        final_step = (last_batch + 2) * self.stride
        if self._log(values, final_step, "final_summary"):
            self.ledger.put("tracking_export", "final", {"logged": True})

    def close(self, success=True):
        if self.session:
            try:
                self.session.finish(state="success" if success else "aborted")
            except Exception as exc:
                self.ledger.put("tracking", "finish_error", {"error_type": type(exc).__name__})
            self.session = None
