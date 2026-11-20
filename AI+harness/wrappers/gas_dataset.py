"""读取 UCI 气体传感器漂移数据的时间批次。"""

import hashlib
import json
from functools import cached_property
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


IMPORT_MANIFEST = "gas-import.json"


def load_gas_model(checkpoint=None):
    from baseline.gas.baselines.model import Model

    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parents[1] / "baseline/gas/checkpoint.npz"
    model = Model.load(checkpoint)
    if model.meta["task"] != "classification":
        raise ValueError("气体任务需要分类模型检查点")
    return model


class GasDriftData:
    batch_names = tuple(f"batch{i}.dat" for i in range(1, 11))
    batch_count = len(batch_names)

    def __init__(self, root, seed, *, model_meta=None):
        self.root = Path(root)
        self.seed = seed
        meta = load_gas_model().meta if model_meta is None else model_meta
        self.feature_count = len(meta["paths"])
        self.classes = tuple(meta["classes"])
        self.splits = {}
        import_path = self.root / IMPORT_MANIFEST
        self.imported = None
        if import_path.is_file():
            manifest = json.loads(import_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("version") != 1:
                raise ValueError(f"导入清单版本无效：{import_path}")
            records = manifest.get("batches")
            allowed = {str(i) for i in range(2, self.batch_count + 1)}
            if not isinstance(records, dict) or not records or set(records) - allowed:
                raise ValueError(f"导入清单须列出第 2 至 {self.batch_count} 批内的批次")
            if any(not isinstance(item, dict) or not isinstance(item.get("dat_sha256"), str)
                   for item in records.values()):
                raise ValueError(f"导入清单缺少 DAT 摘要：{import_path}")
            self.imported = records
            self.batch_indices = tuple(sorted(int(i) - 1 for i in records))
        else:
            self.batch_indices = tuple(range(self.batch_count))
        for i in self.batch_indices:
            if not (self.root / self.batch_names[i]).is_file():
                raise FileNotFoundError(f"缺少气体传感器批次文件：{self.root / self.batch_names[i]}")

    @cached_property
    def manifest(self):
        files = {self.batch_names[i]: hashlib.sha256((self.root / self.batch_names[i]).read_bytes()).hexdigest()
                 for i in self.batch_indices}
        if self.imported is not None:
            for i in self.batch_indices:
                if files[self.batch_names[i]] != self.imported[str(i + 1)]["dat_sha256"]:
                    raise ValueError(f"{self.batch_names[i]} 与导入清单的摘要不一致")
            signature = json.dumps([self.seed, self.imported], sort_keys=True).encode()
            return {"dataset_id": hashlib.sha256(signature).hexdigest()[:20],
                    "source": "gas-import-v1", "source_files": files,
                    "imports": self.imported, "seed": self.seed,
                    "split": {"train": 0.6, "validation": 0.2, "test": 0.2}}
        signature = repr(("uci-gas-drift-v1", self.seed, files)).encode()
        return {"dataset_id": hashlib.sha256(signature).hexdigest()[:20],
                "source": "https://archive.ics.uci.edu/dataset/224/gas",
                "source_files": files, "seed": self.seed,
                "split": {"train": 0.6, "validation": 0.2, "test": 0.2}}

    def load(self, batch, split):
        if batch not in self.batch_indices or split not in {"train", "validation", "test"}:
            raise ValueError("无效的气体传感器批次或数据划分")
        if batch not in self.splits:
            name = self.batch_names[batch]
            source = str(self.root / name)
            if self.imported is not None:
                import io

                raw = (self.root / name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != self.imported[str(batch + 1)]["dat_sha256"]:
                    raise ValueError(f"{name} 与导入清单的摘要不一致")
                source = io.BytesIO(raw)
            features, labels = load_svmlight_file(source, n_features=self.feature_count, zero_based=False)
            if not np.isfinite(features.data).all() or not np.isin(labels, np.arange(1, len(self.classes) + 1)).all():
                raise ValueError(f"{name} 的特征须为有限数值，类别须为 1 至 {len(self.classes)} 的编号")
            x, y = features.toarray(), labels.astype(int)
            rows = np.arange(len(y))
            seed = self.seed + batch
            train, rest = train_test_split(rows, test_size=0.4, stratify=y, random_state=seed)
            validation, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
            self.splits[batch] = {
                key: (x[index], y[index], index)
                for key, index in (("train", train), ("validation", validation), ("test", test))
            }
        return self.splits[batch][split]

    def batch_size(self, batch):
        self.load(batch, "train")
        return sum(len(part[0]) for part in self.splits[batch].values())
