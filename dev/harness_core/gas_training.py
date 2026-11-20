"""Resume the original NumPy MLP and select epochs on validation only."""
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from baseline.gas.baselines.model import Model, checkpoint_digest

from .gas_metrics import evaluate
from .storage import atomic_json, read_json


def train(parent_checkpoint, splits, config, seed):
    started = time.perf_counter()
    model = Model.load(parent_checkpoint)
    initial = {key: value.copy() for key, value in model.params.items()}
    train_data = splits["train"]
    x = model.normalized(train_data["x"])
    y = train_data["y"]
    rng = np.random.default_rng(seed)
    n_classes = len(model.meta["classes"])
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("训练标签编号与模型类别不匹配")
    counts = np.bincount(y, minlength=n_classes)
    weighting = len(y) / (n_classes * np.maximum(counts, 1)) if config["class_weighting"] else np.ones(n_classes)
    # Keep the official checkpoint's moments and step count, including the best epoch on restore.
    initial_steps, initial_epochs = model.meta["steps"], model.meta["epochs"]
    step, best_score, stale, best, history = initial_steps, -1., 0, None, []
    first, second = model.first, model.second
    for epoch in range(config["epochs"]):
        order = rng.permutation(len(y))
        losses = []
        for start in range(0, len(y), config["batch_size"]):
            ids = order[start:start + config["batch_size"]]
            features, expected = x[ids], y[ids]
            hidden, logits = model.forward(features)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            weights = weighting[expected]
            losses.append(float(-np.mean(weights * np.log(np.maximum(probabilities[np.arange(len(ids)), expected], 1e-300)))))
            delta = probabilities
            delta[np.arange(len(ids)), expected] -= 1
            delta *= (weights / weights.sum())[:, None]
            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hdelta = (delta @ model.params["w2"].T) * (1 - hidden**2)
            gradients.update(w1=features.T @ hdelta, b1=hdelta.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient")
            step += 1
            for key, gradient in gradients.items():
                gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                first[key] = .9 * first[key] + .1 * gradient
                second[key] = .999 * second[key] + .001 * gradient**2
                update = (first[key] / (1 - .9**step)) / (np.sqrt(second[key] / (1 - .999**step)) + 1e-8)
                decay = config["weight_decay"] * model.params[key] if key.startswith("w") else 0
                model.params[key] -= config["learning_rate"] * (update + decay)
        score = evaluate(model, splits["validation"])["macro_f1"]
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best_score + 1e-8:
            best_score, stale = score, 0
            best = ([{k: v.copy() for k, v in state.items()}
                     for state in (model.params, first, second)], step, epoch + 1)
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]:
                break
    (model.params, model.first, model.second), best_step, best_epoch = best
    model.meta["epochs"] = initial_epochs + best_epoch
    model.meta["steps"] = best_step
    model.meta["learning_rate"] = config["learning_rate"]
    model.meta["batch_size"] = config["batch_size"]
    model.validate()
    changed = any(not np.array_equal(initial[k], model.params[k]) for k in initial)
    if not changed:
        raise ValueError("training did not update model weights")
    return model, {"epochs_run": len(history), "best_epoch": best_epoch,
                   "best_validation_macro_f1": best_score, "train_rows": len(y),
                   "history": history, "initial_steps": initial_steps, "initial_epochs": initial_epochs,
                   "final_steps": best_step, "final_epochs": initial_epochs + best_epoch,
                   "training_seconds": time.perf_counter() - started}

def main(input_path, request_path, parent_path, output_path):
    output = Path(output_path)
    request = read_json(request_path)
    if checkpoint_digest(parent_path) != request["parent_checkpoint_sha256"]:
        raise ValueError("训练起点的检查点摘要发生变化")
    with np.load(input_path, allow_pickle=False) as arrays:
        splits = {split: {key: arrays[f"{split}_{key}"] for key in ("x", "y")}
                  for split in ("train", "validation")}
    model, report = train(parent_path, splits, request["params"], request["seed"])
    with tempfile.TemporaryDirectory(prefix=".trainer-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        checkpoint = staging / "checkpoint.npz"
        model.save(checkpoint)
        atomic_json(staging / "training.json", {**report, "request": request,
                    "checkpoint_sha256": checkpoint_digest(checkpoint)})
        os.replace(staging, output)


if __name__ == "__main__":
    main(*sys.argv[1:])
