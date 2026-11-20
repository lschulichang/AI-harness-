"""Evaluate the classes declared by the checkpoint, including classes with zero support."""
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def evaluate(model, data):
    start = time.perf_counter()
    probabilities = model.predict(data["x"])
    latency = (time.perf_counter() - start) * 1000 / len(data["y"])
    pred = probabilities.argmax(axis=1)
    y = data["y"]
    classes = model.meta["classes"]
    class_ids = np.arange(len(classes))
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=class_ids, zero_division=0)
    return {"samples": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=class_ids, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, labels=class_ids, average="weighted", zero_division=0)),
            "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(support[i])}
                          for i, name in enumerate(classes)},
            "confusion_matrix": confusion_matrix(y, pred, labels=class_ids).tolist(),
            "latency_ms_per_sample": latency}
