"""Use the project gas data with the official MLP input format."""

import numpy as np
from wrappers.gas_dataset import load_gas_model


def evaluate_samples(model_path, samples):
    from .gas_metrics import evaluate as evaluate_model

    model = load_gas_model(model_path)
    if "paths" in samples and samples["paths"] != model.meta["paths"]:
        raise ValueError("公开验证特征与检查点不一致")
    score = evaluate_model(model, samples)
    return {key: score[key] for key in ("macro_f1", "accuracy")}


def gas_samples(data, batch, split):
    x, labels, _ = data.load(batch, split)
    return {"x": x, "y": np.asarray(labels) - 1}
