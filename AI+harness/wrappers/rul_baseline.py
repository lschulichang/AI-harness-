"""读取 RUL 基线检查点与按设备隔离的公开数据。"""
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class RulBaseline:
    def __init__(self, bundle):
        self.bundle = Path(bundle).resolve()
        # 基线包使用独立命名空间，避免和 A 的 baselines.model 混用。
        name = "_rul_baseline_" + hashlib.sha256(str(self.bundle).encode()).hexdigest()[:16]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, self.bundle / "baselines/__init__.py",
                submodule_search_locations=[str(self.bundle / "baselines")])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self.model_class = importlib.import_module(name + ".model").Model
        self.reader = importlib.import_module(name + ".data").read_dataset
        self.flatten = importlib.import_module(name + ".data").flatten
        report = json.loads((self.bundle / "training.json").read_text())
        info = json.loads((self.bundle / "baseline-info.json").read_text())
        if (report["profile"] != self.profile or report["task"] != "regression"
                or info["dataset_key"] != self.profile
                or info.get("target_unit", self.checkpoint_default_unit) != self.unit):
            raise ValueError("需要匹配任务与单位的 RUL 回归包")
        hashes = {key: digest(self.bundle / path) for key, path in (
            ("checkpoint_sha256", "checkpoint.npz"), ("train_sha256", "public/train.jsonl"),
            ("validation_sha256", "public/validation.jsonl"))}
        if any(report[key] != value for key, value in hashes.items()) or info["checkpoint_sha256"] != hashes["checkpoint_sha256"]:
            raise ValueError("基线检查点或公开数据与 training.json 摘要不一致")
        self.model = self.load_model(self.bundle / "checkpoint.npz")
        self.rows = {}
        entities, ids = {}, {}
        for split in ("train", "validation"):
            rows = read_rows(self.bundle / f"public/{split}.jsonl")
            self.reader(self.bundle / f"public/{split}.jsonl", "regression", [], self.model.meta["paths"])
            self.rows[split] = {group: [] for group in self.group_names}
            entities[split], ids[split] = set(), set()
            for row in rows:
                group, entity = self.identity(row)
                if group not in self.rows[split]:
                    raise ValueError("公开样本含未知任务组")
                self.rows[split][group].append(row)
                entities[split].add(entity)
                ids[split].add(row["id"])
            if any(not value for value in self.rows[split].values()):
                raise ValueError("每个任务组都需要公开训练和验证样本")
        if entities["train"] & entities["validation"] or ids["train"] & ids["validation"]:
            raise ValueError("训练和验证的设备或样本 ID 发生重叠")
        self.manifest = {
            "protocol": self.protocol, "dataset": self.dataset, "profile": self.profile,
            "baseline_id": info["id"], "unit": self.unit, "metric": "rmse", "higher_is_better": False,
            "initial_checkpoint_already_saw_all_public_training": True,
            "batch_order_is_local_group_order_not_real_time": True,
            "validation_is_public_not_blind": True, **hashes,
            "groups": [{"name": group,
                        "train_rows": len(self.rows["train"][group]),
                        "validation_rows": len(self.rows["validation"][group]),
                        "train_entities": sorted({self.identity(r)[1] for r in self.rows["train"][group]}),
                        "validation_entities": sorted({self.identity(r)[1] for r in self.rows["validation"][group]})}
                       for group in self.group_names],
            "baseline_sources": {p.name: digest(p) for p in sorted((self.bundle / "baselines").glob("*.py"))},
            "profile_sha256": digest(self.bundle / f"baselines/configs/{self.profile}.json"),
        }

    def load_model(self, path):
        model = self.model_class.load(path)
        if (model.meta.get("profile") != self.profile or model.meta["task"] != "regression"
                or len(model.meta["paths"]) != self.feature_count):
            raise ValueError("检查点任务/输入维度不匹配")
        if model.meta.get("target_unit", self.checkpoint_default_unit) != self.unit:
            raise ValueError(f"检查点目标单位必须为 {self.unit}")
        if hasattr(self, "model") and model.meta["paths"] != self.model.meta["paths"]:
            raise ValueError("候选检查点特征顺序不匹配")
        return model

    def samples(self, batch, split="validation"):
        rows = self.rows[split][self.group_names[batch]]
        x = [self.flatten(row["inputs"])[1] for row in rows]
        return {"x": np.asarray(x), "y": np.asarray([r["answer"] for r in rows], dtype=float)}

    def metrics(self, model, samples):
        expected = samples["y"]
        predicted = model.predict(samples["x"])
        if expected.shape != predicted.shape or not expected.size or not np.isfinite(predicted).all():
            raise ValueError("回归预测必须与目标同形状且为有限数值")
        error = predicted - expected
        return {"rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))), "samples": int(expected.size)}

    def training_rows(self, window):
        return [row for index in window for row in self.rows["train"][self.group_names[index]]]

    def write_inputs(self, batch, window, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        train = self.training_rows(window)
        validation = self.rows["validation"][self.group_names[batch]]
        for name, rows in (("train", train), ("validation", validation)):
            path = directory / f"{name}.jsonl"
            content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
            if path.exists():
                if path.read_text(encoding="utf-8") != content:
                    raise ValueError("已保存的训练输入发生变化")
            else:
                path.write_text(content, encoding="utf-8")
        return directory / "train.jsonl", directory / "validation.jsonl"
