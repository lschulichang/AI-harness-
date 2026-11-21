"""滑动窗口训练师的结构化提案、校验与规则兜底。"""

from .llm_client import Planner
from .runner import (
    EARLY_STOPPING_PATIENCE, TRAINING_DEFAULTS, TRAINING_LIMITS, effective_search_parameters,
    reference_window, search_parameters, validate_search_parameters,
)

DECISIONS = {"NO_UPDATE", "INCREMENTAL_TRAINING", "MIXED_DATA_TRAINING"}
TRAINING_ACTIONS = {"INCREMENTAL_TRAINING", "MIXED_DATA_TRAINING"}
WINDOW_POLICY = "trainer_window_v4"
WINDOW_DRIFT_BATCHES = 3
WINDOW_TARGET_BATCHES = 4

WINDOW_TRAINER_CONFIG = {
    "drift": {"bins": 10, "psi_warning": 0.1, "ks_fdr": 0.05,
              "mmd_samples": 300, "mmd_permutations": 99,
              "label_js_warning": 0.05, "min_samples": 80},
    "training": {**TRAINING_DEFAULTS, "batch_size": 64,
                 "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                 "weight_decay": 0.0001, "class_weighting": False},
    "limits": TRAINING_LIMITS,
}


def validate(raw, cfg):
    if not isinstance(raw, dict) or set(raw) != {"decision", "reason", "training_config", "evaluation_requirements"}:
        raise ValueError("decision must have exactly the required JSON keys")
    if (not isinstance(raw["decision"], str) or raw["decision"] not in DECISIONS
            or not isinstance(raw["reason"], str) or not raw["reason"].strip()):
        raise ValueError("invalid decision or reason")
    requirements = raw["evaluation_requirements"]
    if not isinstance(requirements, list) or not all(isinstance(value, str) for value in requirements):
        raise ValueError("invalid evaluation requirements")
    supplied = raw["training_config"]
    if not isinstance(supplied, dict):
        raise ValueError("invalid training configuration")
    if raw["decision"] not in TRAINING_ACTIONS:
        if supplied:
            raise ValueError("nontraining decision must have empty training_config")
        return raw
    validate_search_parameters(supplied, raw["decision"])
    return {**raw, "training_config": {**cfg["training"], **supplied}}


def rules(context, cfg):
    current = context["performance"]["current"]
    target = context.get("validation_target")
    stable = target is not None and current["macro_f1"] >= target
    if context["data"]["new_labeled_samples"] < cfg["drift"]["min_samples"]:
        decision, reason = "NO_UPDATE", "Insufficient labeled samples; keep the current model"
    elif stable and len(context["drift_report"]["drifted_features"]) < 20:
        decision, reason = "NO_UPDATE", "Current performance is stable"
    elif context["data"]["historical_replay_available"]:
        decision, reason = "MIXED_DATA_TRAINING", "Drift or performance decline; historical replay exists"
    else:
        decision, reason = "INCREMENTAL_TRAINING", "Drift or performance decline; replay unavailable"
    return validate({
        "decision": decision, "reason": reason,
        "training_config": {**search_parameters(cfg["training"]), "window_size": reference_window(decision)} if decision in TRAINING_ACTIONS else {},
        "evaluation_requirements": ["new_data_macro_f1", "per_class_recall"],
    }, cfg)


WINDOW_PROMPT = (
    "/no_think Return only concise JSON with exactly decision, reason, training_config, "
    "evaluation_requirements. Choose NO_UPDATE, INCREMENTAL_TRAINING or "
    "MIXED_DATA_TRAINING. Use training_config={} when skipping. For a candidate explicitly return "
    "only learning_rate, epochs and window_size within training_limits. "
    "A window counts the current batch. Use 1 for incremental training and 2–6 for mixed training. "
    "For mixed training use the latest available history, up to window_size - 1 batches. "
    "Do not choose mixed training without any replay batches. evaluation_requirements must be an array of "
    "strings. Missing drift reference or labels mean the corresponding evidence is unavailable. "
    "Use validation evidence only. Do not claim held-out results or output code or commands."
)


def compact_input(context, cfg):
    def brief(metrics):
        if metrics is None:
            return None
        return {"accuracy": round(metrics["accuracy"], 4), "macro_f1": round(metrics["macro_f1"], 4),
                "recall": {name: round(value["recall"], 3) for name, value in metrics["per_class"].items()}}

    drift = context["drift_report"]
    return {
        "drift": {
            "feature_count": len(drift["drifted_features"]),
            "top_feature_indices": drift["drifted_features"][:8],
            "mmd_p": (drift["mmd"] or {}).get("permutation_p"),
            "mmd_squared": (drift["mmd"] or {}).get("mmd_squared"),
            "label_js": drift["label_js_divergence"],
            "label_status": drift.get("label_status"),
            "status": drift["status"], "reference_batches": drift["reference_batches"],
        },
        "current_validation": brief(context["performance"]["current"]),
        "new_rows": context["data"]["new_labeled_samples"],
        "replay_available": context["data"]["historical_replay_available"],
        "replay_batches": context["data"]["replay_batches"],
        "recent_history": [
            {"batch": row["batch"], "decision": row["decision"], "deployed": row["deployed"],
             "macro_f1_gain": round(row["macro_f1_gain"], 4),
             "drifted_feature_count": row["drifted_feature_count"]}
            for row in context["history"][:3]
        ],
        "training_defaults": search_parameters(cfg["training"]), "training_limits": cfg["limits"],
    }


class GasTrainer(Planner):
    def decide(self, key, context, cfg, *, feedback=None):
        saved = self.ledger.get("trainer_decision", key)
        if saved is not None:
            return saved
        payload = compact_input(context, cfg)
        prompt = WINDOW_PROMPT
        attempts = []
        if feedback is not None:
            attempts = feedback["attempts"]
            payload["batch_feedback"] = {**feedback, "attempts": [
                {**attempt, "training_config": search_parameters(attempt["training_config"])}
                for attempt in attempts
            ]}
            prompt += (
                " Each decision requests at most one candidate. All candidates start from the same batch-entry model. "
                "Review batch_feedback for actual validation, selection status, gate failures, sample passes and remaining budgets. "
                "If another distinct configuration is worth trying, return a training decision with changed parameters. "
                "Otherwise return NO_UPDATE to finish; the best eligible candidate so far will still be adopted."
            )
            if feedback.get("allow_best_below_target"):
                prompt += (
                    " If no candidate reaches the validation target, the best below-target candidate "
                    "may be selected at the end if it improves over "
                    "the batch-entry model. Use min_current_validation_macro_f1 as the priority target."
                )
        decision, error = None, ""
        if self.config.strategy == "llm":
            def check(value):
                decision = validate(value, cfg)
                if decision["decision"] == "MIXED_DATA_TRAINING" and not payload["replay_available"]:
                    raise ValueError("mixed training requires replay samples")
                previous_count = len(payload.get("replay_batches", ()))
                if decision["decision"] in TRAINING_ACTIONS and any(
                        effective_search_parameters(decision["training_config"], previous_count)
                        == effective_search_parameters(attempt["training_config"], previous_count) for attempt in attempts):
                    raise ValueError("this training configuration has already been tried in the current batch")
                return decision

            decision, error = self._request_proposal(
                key, prompt, payload, check, output_budget=self.config.max_api_output_tokens)
        if decision is None:
            if attempts:
                decision = {"decision": "NO_UPDATE", "reason": "反馈请求未得到有效方案，结束本批搜索",
                            "training_config": {}, "evaluation_requirements": []}
            else:
                decision = rules(context, cfg)
            source = "rules_fallback" if self.config.strategy == "llm" else "rules"
        else:
            source = "trainer_llm"
        result = {"input": payload, "output": decision, "metadata": {
            "source": source, "fallback": source == "rules_fallback", "error": error or None}}
        self.ledger.put("trainer_decision", key, result)
        return result
