"""读取 RUL 基线检查点与按设备隔离的公开数据。"""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RulBaseline:
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
