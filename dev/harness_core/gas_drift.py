"""气体分类的特征及标签比例漂移诊断。"""
import time

import numpy as np
from scipy.spatial.distance import jensenshannon

from .feature_drift import detect_features


def label_distribution(reference, current, classes):
    def distribution(batch):
        labels = batch.get("y")
        if labels is None:
            return None
        labels = np.asarray(labels)
        if (labels.shape != (len(batch["x"]),) or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels >= len(classes))):
            raise ValueError("漂移诊断标签须与样本逐行对应，且编号须符合检查点类别定义")
        counts = np.bincount(labels, minlength=len(classes)).astype(float) + 1e-12
        return counts / counts.sum()

    ref, cur = distribution(reference), distribution(current)
    available = ref is not None and cur is not None
    return {"class_labels": list(classes),
            "status": "available" if available else "labels_unavailable",
            "reference": ref.tolist() if ref is not None else None,
            "current": cur.tolist() if cur is not None else None,
            "js_divergence": float(jensenshannon(ref, cur) ** 2) if available else None}


def detect(reference, current, cfg, seed, *, classes):
    started = time.perf_counter()
    result = detect_features(reference["x"], current["x"], cfg, seed)
    result["label_distribution"] = label_distribution(reference, current, classes)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result
