"""Map each SDK stage to one batch in the existing sliding windows."""

import gzip
import json
import math
import shutil
from pathlib import Path

import numpy as np
from baseline.gas.baselines.data import flatten
from harness_core.storage import atomic_json, read_json
from wrappers.bearing_baseline import BearingBaseline
from wrappers.cmapss_baseline import CmapssBaseline

ROOT = Path(__file__).resolve().parent
BUNDLES = {
    "A": "gas",
    "B": "cpu-b-rul-20261110-v1",
    "C": "cpu-c-guide-20261110-v1",
}
DATASETS = {"A": "gas_mlp", "B": "bearing_rul", "C": "cmapss_rul"}
TASK_TYPES = {"A": "classification", "B": "regression", "C": "regression"}
UNITS = {"B": "minutes", "C": "cycles"}
FEATURE_COUNTS = {"A": 128, "B": 34, "C": 24}
GASES = {"Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"}
TRAIN_HISTORY = 7
VALIDATION_HISTORY = 6
STATE_SCHEMA = "contest.sliding-window.v1"


def sha256_matches(actual, declared):
    return actual.removeprefix("sha256:") == declared.removeprefix("sha256:")


def validate_model(model, key, task):
    meta = model.meta
    if (task != TASK_TYPES[key] or meta["task"] != task
            or meta.get("profile") != key or len(meta["paths"]) != FEATURE_COUNTS[key]):
        raise ValueError(f"{key} 的父检查点任务、输入维度或 profile 不匹配")
    if key == "A" and set(meta["classes"]) != GASES:
        raise ValueError("A 检查点的气体类别不匹配")
    if key in UNITS and meta.get("target_unit", "cycles" if key == "C" else None) != UNITS[key]:
        raise ValueError(f"{key} 检查点的寿命单位不匹配")


def checked_rows(rows, model, split):
    seen = {}
    for row in rows:
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"{split} 含空样本 ID 或重复样本 ID")
        paths, _ = flatten(row["inputs"])
        if paths != model.meta["paths"]:
            raise ValueError(f"{split} 的输入结构与父模型不一致")
        answer = row.get("answer")
        if model.meta["task"] == "classification":
            if not isinstance(answer, str) or answer not in model.meta["classes"]:
                raise ValueError(f"{split} 含未识别的气体标签")
        elif type(answer) not in (int, float) or not math.isfinite(answer) or answer < 0:
            raise ValueError(f"{split} 寿命标签必须是有限非负数")
        seen[identity] = {"id": identity, "inputs": row["inputs"], "answer": answer}
    if not seen:
        raise ValueError(f"{split} 没有公开样本")
    return list(seen.values())


class Projection:
    """Public rows retained only for the original replay and validation windows."""

    def __init__(self, rnd, state, model, parent):
        self.state = state
        self.key = rnd.dataset_key
        self.batch = rnd.round_index - 1
        head = state / "window.json"
        if head.is_file():
            saved = read_json(head)
            if (saved["schema"] != STATE_SCHEMA or saved["dataset"] != self.key
                    or saved["last_batch"] != self.batch - 1):
                raise ValueError("历史状态与当前数据集或轮次不连续")
            self.history = saved["history"]
            self.names = {row["batch"]: row["source_file"] for row in self.history}
        elif self.batch == 0:
            self.history, self.names = [], {}
            if self.key != "A":
                shutil.copyfile(parent, state / "static.npz")
        else:
            raise ValueError("平台未挂载上一轮的滑窗状态")

        self.names[self.batch] = f"round-{rnd.round_index:03d}"
        self.rows = {"train": {}, "validation": {}}
        for split, width, raw in (
            ("train", TRAIN_HISTORY, rnd.data.training_rows()),
            ("validation", VALIDATION_HISTORY, rnd.data.validation_rows()),
        ):
            for index in range(max(0, self.batch - width), self.batch):
                with gzip.open(self.path(index, split), "rt", encoding="utf-8") as stream:
                    self.rows[split][index] = [json.loads(line) for line in stream]
            self.rows[split][self.batch] = checked_rows(raw, model, split)

        train_ids = {row["id"] for rows in self.rows["train"].values() for row in rows}
        validation_ids = {row["id"] for rows in self.rows["validation"].values() for row in rows}
        if train_ids & validation_ids:
            raise ValueError("公开样本在滑窗内同时出现在训练集和验证集")

    def path(self, index, split):
        return self.state / f"{split}-{index + 1:03d}.jsonl.gz"

    def window_rows(self, window):
        # A cumulative public projection may repeat an earlier training row.
        # Include it once and keep the latest public value as authoritative.
        latest = {}
        for index in window:
            for row in self.rows["train"][index]:
                latest[row["id"]] = row
        return list(latest.values())

    def save(self, row):
        for split, width in (("train", TRAIN_HISTORY), ("validation", VALIDATION_HISTORY)):
            with gzip.open(self.path(self.batch, split), "wt", encoding="utf-8", compresslevel=5) as stream:
                for item in self.rows[split][self.batch]:
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            expired = self.batch - width
            if expired >= 0:
                self.path(expired, split).unlink(missing_ok=True)
        keys = (
            "batch", "source_file", "validation_macro_f1", "trainer_history",
            "validation_rmse", "before_validation_rmse", "trainer_action", "promoted", "feature_drift",
        )
        entry = {key: row[key] for key in keys if key in row}
        if self.key != "A":
            entry["feature_drift"] = {"drifted_features": row["feature_drift"]["drifted_features"]}
        atomic_json(self.state / "window.json", {
            "schema": STATE_SCHEMA, "dataset": self.key, "last_batch": self.batch,
            "history": [*self.history, entry][-TRAIN_HISTORY:],
        })


class GasProjection:
    def __init__(self, projection, model):
        self.projection = projection
        self.batch_names = projection.names
        self.classes = model.meta["classes"]
        self._cache = {}

    def samples(self, rows):
        return {
            "x": np.asarray([flatten(row["inputs"])[1] for row in rows], dtype=np.float64),
            "y": np.asarray([self.classes.index(row["answer"]) for row in rows], dtype=np.int64),
        }

    def load(self, index, split):
        key = (index, split)
        if key not in self._cache:
            self._cache[key] = self.samples(self.projection.rows[split][index])
        samples = self._cache[key]
        return samples["x"], samples["y"] + 1, None


def combine_gas_candidate(data, window):
    return data.samples(data.projection.window_rows(window))


class PublicRulProjection:
    def __init__(self, projection, model):
        self.projection = projection
        self.bundle = ROOT / "baseline" / BUNDLES[projection.key]
        self.model_class = type(model)
        self.model = model
        self.flatten = flatten
        self.group_names = projection.names
        self.rows = {
            split: {self.group_names[index]: rows for index, rows in batches.items()}
            for split, batches in projection.rows.items()
        }

    def training_rows(self, window):
        return self.projection.window_rows(window)


class BearingProjection(PublicRulProjection, BearingBaseline):
    pass


class CmapssProjection(PublicRulProjection, CmapssBaseline):
    pass
