"""C-MAPSS 剩余寿命基线与最后观测循环公开测试。"""
from pathlib import Path

import numpy as np

from .rul_baseline import RulBaseline, digest


PACKAGE = "cpu-c-guide-20261110-v1"
RAW_DATA = "data/CMAPSSData"


class CmapssBaseline(RulBaseline):
    profile = "C"
    unit = "cycles"
    # 旧版 C 指导包的检查点没有 target_unit，任务合同定义单位为 cycles。
    checkpoint_default_unit = "cycles"
    feature_count = 24
    group_names = ("FD001", "FD002", "FD003", "FD004")
    dataset = "cmapss_rul"
    protocol = "cmapss_public_subset_replay"

    @staticmethod
    def identity(row):
        meta = row["meta"]
        group, unit = meta["source_subset"], meta["unit"]
        if meta.get("source_split") != "train" or type(unit) is not int or unit < 1:
            raise ValueError("C 公开训练/验证必须来自完整训练轨迹，并按发动机划分")
        return group, f"{group}:{unit}"

    def test_manifest(self, data_root):
        root = Path(data_root)
        paths = [root / f"{prefix}_{subset}.txt" for subset in self.group_names for prefix in ("test", "RUL")]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError("C-MAPSS 独立测试需要四组 test_FD00x.txt 与 RUL_FD00x.txt")
        return {"available": True, "protocol": "last_observed_cycle_per_test_engine_uncapped_RUL",
                "files": {p.name: digest(p) for p in paths}}

    def test_samples(self, batch, root):
        subset = self.group_names[batch]
        raw = np.loadtxt(Path(root) / f"test_{subset}.txt", ndmin=2)
        if raw.shape[1] != 26 or not np.isfinite(raw).all() or (raw[:, :2] < 1).any() or (raw[:, :2] != np.floor(raw[:, :2])).any():
            raise ValueError("C-MAPSS 测试文件格式无效")
        raw = raw[np.lexsort((raw[:, 1], raw[:, 0]))]
        ids, first, counts = np.unique(raw[:, 0].astype(int), return_index=True, return_counts=True)
        if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
            raise ValueError("测试发动机编号必须与 RUL 行号连续对应")
        cycles = np.arange(len(raw)) - np.repeat(first, counts) + 1
        if not np.array_equal(raw[:, 1], cycles):
            raise ValueError("测试周期不连续或重复")
        labels = np.loadtxt(Path(root) / f"RUL_{subset}.txt", ndmin=2)
        if labels.shape != (len(ids), 1) or not np.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("RUL 真值数量/格式与测试发动机不一致")
        features = []
        for row in raw[first + counts - 1]:
            paths, values = self.flatten({"settings": row[2:5].tolist(), "sensors": row[5:].tolist()})
            if paths != self.model.meta["paths"]:
                raise ValueError("测试特征路径与官方检查点不一致")
            features.append(values)
        return {"x": np.asarray(features), "y": labels[:, 0],
                "ids": [f"{subset}_test_{unit}" for unit in ids]}


def test_scores(model, samples):
    error = model.predict(samples["x"]) - samples["y"]
    if error.shape != samples["y"].shape or not error.size or not np.isfinite(error).all():
        raise ValueError("C 测试预测必须与目标同形状且为有限数值")
    exponent = np.where(error < 0, -error / 13, error / 10)
    with np.errstate(over="ignore"):
        nasa = float(np.expm1(exponent).sum())
    return {"samples": len(error), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "nasa_score": nasa if np.isfinite(nasa) else None}
