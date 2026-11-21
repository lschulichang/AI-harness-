"""比赛训练闭环共用的历史窗口。"""

TRAINING_DEFAULTS = {"learning_rate": 0.0003, "epochs": 20, "window_size": 3}
TRAINING_LIMITS = {"learning_rate": [1e-6, 0.01], "epochs": [1, 50], "window_size": [1, 6]}
EARLY_STOPPING_PATIENCE = 5


def reference_window(action):
    return TRAINING_DEFAULTS["window_size"] if action == "MIXED_DATA_TRAINING" else 1


def search_parameters(config):
    return {key: config[key] for key in TRAINING_LIMITS}


def validate_search_parameters(params, action):
    if set(params) != TRAINING_LIMITS.keys():
        raise ValueError("training_config must contain only learning_rate, epochs, and window_size")
    for key, (lower, upper) in TRAINING_LIMITS.items():
        allowed_types = (int, float) if key == "learning_rate" else (int,)
        if type(params[key]) not in allowed_types or not lower <= params[key] <= upper:
            raise ValueError(f"{key} is outside configured bounds")
    if (params["window_size"] == 1) != (action == "INCREMENTAL_TRAINING"):
        raise ValueError("incremental training requires window_size=1; mixed training requires 2–6")


def effective_search_parameters(params, replay_count):
    return {**search_parameters(params), "window_size": min(params["window_size"], replay_count + 1)}


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
    if type(window_size) is not int or not 1 <= window_size <= TRAINING_LIMITS["window_size"][1]:
        raise ValueError("window_size must be an integer from 1 to 6")
    previous_count = window_size - 1
    return [*(completed[-previous_count:] if previous_count else ()), batch]
