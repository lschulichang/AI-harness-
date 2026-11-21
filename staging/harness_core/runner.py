"""比赛训练闭环共用的历史窗口。"""

MIXED_TRAINING_WINDOW = 3


def window_size_for_action(action):
    """Resolve replay width from the chosen action, independent of model parameters."""
    return MIXED_TRAINING_WINDOW if action == "MIXED_DATA_TRAINING" else 1


def candidate_feedback(trial, decision, metric_names, *, can_fallback=False):
    if decision["eligible"]:
        status = "达到优先门槛"
    elif can_fallback:
        status = "可作备选"
    else:
        status = "未通过"
    return {
        "validation_metrics": {name: trial.get(name) for name in metric_names},
        "selection_status": status,
        "reason": decision["reason"],
        "gate_failures": trial.get("gate_failures", [decision["reason"]]),
        "sample_passes": trial["sample_passes"],
    }


def training_window(batch, completed, window_size):
    if type(window_size) is not int or not 1 <= window_size <= 8:
        raise ValueError("window_size must be an integer from 1 to 8")
    previous_count = window_size - 1
    return [*(completed[-previous_count:] if previous_count else ()), batch]
