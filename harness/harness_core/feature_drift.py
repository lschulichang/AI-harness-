"""不依赖标签的逐特征及整体分布漂移诊断。"""
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance


def _psi(ref, cur, bins):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(cur, ref[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0].astype(float) + 1e-6
    b = np.histogram(cur, edges)[0].astype(float) + 1e-6
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def _bh(pvalues, q):
    order = np.argsort(pvalues)
    ranks = np.arange(1, len(pvalues) + 1)
    passed = np.asarray(pvalues)[order] <= q * ranks / len(pvalues)
    cutoff = np.max(np.where(passed)[0]) if passed.any() else -1
    out = np.zeros(len(pvalues), dtype=bool)
    if cutoff >= 0:
        out[order[:cutoff + 1]] = True
    return out


def _mmd(ref, cur, cfg, seed):
    rng = np.random.default_rng(seed)
    cap = cfg["mmd_samples"]
    a = ref[rng.choice(len(ref), min(cap, len(ref)), replace=False)]
    b = cur[rng.choice(len(cur), min(cap, len(cur)), replace=False)]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    stacked = np.vstack((a, b))
    scale = np.std(a, axis=0)
    scale[scale < 1e-12] = 1
    stacked = np.clip((stacked - np.mean(a, axis=0)) / scale, -10, 10)
    squared = cdist(stacked, stacked, "sqeuclidean")
    positive = squared[squared > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-squared / max(bandwidth, 1e-12))
    def score(indices):
        left, right = indices[:n], indices[n:]
        return float(kernel[np.ix_(left, left)].mean() + kernel[np.ix_(right, right)].mean()
                     - 2 * kernel[np.ix_(left, right)].mean())
    observed = score(np.arange(2*n))
    null = [score(rng.permutation(2*n)) for _ in range(cfg["mmd_permutations"])]
    p = (1 + sum(v >= observed for v in null)) / (len(null) + 1)
    return {"mmd_squared": observed, "permutation_p": p, "sample_per_group": n}


def detect_features(rx, cx, cfg, seed):
    started = time.perf_counter()
    if (rx.ndim != 2 or cx.ndim != 2 or not len(rx) or not len(cx)
            or not rx.shape[1] or rx.shape[1] != cx.shape[1]):
        raise ValueError("漂移参考集和当前批次须是非空矩阵，且特征维度一致")
    features = []
    for i in range(rx.shape[1]):
        a, b = rx[:, i], cx[:, i]
        ks = ks_2samp(a, b)
        features.append({"index": i + 1, "psi": _psi(a, b, cfg["bins"]),
                         "ks_statistic": float(ks.statistic),
                         "ks_p": float(ks.pvalue),
                         "wasserstein": float(wasserstein_distance(a, b))})
    flagged = _bh([f["ks_p"] for f in features], cfg["ks_fdr"])
    for f, flag in zip(features, flagged):
        f["ks_bh_significant"] = bool(flag)
        f["drifted"] = bool(f["psi"] >= cfg["psi_warning"] or flag)
    mmd = _mmd(rx, cx, cfg, seed)
    result = {"features": features, "global": mmd,
              "drifted_features": [f["index"] for f in features if f["drifted"]],
              "most_drifted_feature": max(features, key=lambda x: x["psi"])["index"],
              "elapsed_seconds": time.perf_counter() - started}
    return result
