"""比赛训练闭环共用的历史窗口。"""

MIXED_TRAINING_WINDOW = 3


def window_size_for_action(action):
    """Resolve replay width from the chosen action, independent of model parameters."""
    return MIXED_TRAINING_WINDOW if action == "MIXED_DATA_TRAINING" else 1


def training_window(batch, completed, window_size):
    if type(window_size) is not int or not 1 <= window_size <= 8:
        raise ValueError("window_size must be an integer from 1 to 8")
    previous_count = window_size - 1
    return [*(completed[-previous_count:] if previous_count else ()), batch]
