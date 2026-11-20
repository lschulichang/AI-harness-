"""Load the B replay with a training bearing and a validation bearing per condition."""
import json
from pathlib import Path

from .bearing_baseline import BearingBaseline, digest, read_rows


REPLAY_DIR = "data/bearing_rul_replay"
PROTOCOL = "bearing_paired_condition_replay"


class PairedBearingBaseline(BearingBaseline):
    def __init__(self, bundle, replay_dir):
        super().__init__(bundle)
        replay_dir = Path(replay_dir)
        replay_manifest = replay_dir / "replay.json"
        if not replay_manifest.is_file():
            raise FileNotFoundError(f"找不到 B 配对回放数据：{replay_manifest}")
        manifest = json.loads(replay_manifest.read_text(encoding="utf-8"))
        if (manifest.get("protocol"), manifest.get("checkpoint_sha256")) != (
            PROTOCOL, self.manifest["checkpoint_sha256"]
        ):
            raise ValueError("配对回放数据与初始 B 检查点不匹配")

        official_training = {row["meta"]["bearing"] for rows in self.rows["train"].values() for row in rows}
        official_public = self.rows["validation"]
        groups = self.manifest["groups"]
        batches = manifest.get("batches")
        if not isinstance(batches, list) or len(batches) != len(self.group_names):
            raise ValueError("配对回放清单的工况数量不匹配")
        pairing = {}
        for index, (group, batch) in enumerate(zip(self.group_names, batches), 1):
            if (batch.get("condition") != group or
                    {batch.get("train_bearing"), batch.get("validation_bearing")}
                    != {f"Bearing{index}_4", f"Bearing{index}_5"}):
                raise ValueError(f"配对回放清单的轴承分工不正确：{group}")
            pairing[group] = batch
        source_digests = {}
        sample_ids = set()
        for split in ("train", "validation"):
            path = replay_dir / f"{split}.jsonl"
            source_digests[f"{split}_sha256"] = digest(path)
            if source_digests[f"{split}_sha256"] != manifest[f"{split}_sha256"]:
                raise ValueError(f"配对回放 {split} 文件摘要不匹配")
            self.reader(path, "regression", [], self.model.meta["paths"])
            grouped = {group: [] for group in self.group_names}
            for row in read_rows(path):
                group, bearing = self.identity(row)
                if group not in grouped or bearing != pairing[group][f"{split}_bearing"]:
                    raise ValueError(f"未知工况或 {split} 轴承：{bearing}")
                if bearing in official_training or row["id"] in sample_ids:
                    raise ValueError("回放样本重复或来自初始模型的训练轴承")
                sample_ids.add(row["id"])
                grouped[group].append(row)
            for index, group in enumerate(self.group_names, 1):
                records = grouped[group]
                count = len(records)
                if not count or [row["meta"]["snapshot"] for row in records] != list(range(1, count + 1)):
                    raise ValueError(f"{split} 工况 {group} 的快照序列不完整")
                if [row["answer"] for row in records] != list(range(count - 1, -1, -1)):
                    raise ValueError(f"{split} 工况 {group} 的 RUL 分钟数与快照顺序不一致")
                bearing = pairing[group][f"{split}_bearing"]
                if count != pairing[group][f"{split}_rows"]:
                    raise ValueError(f"{split} 工况 {group} 的样本数与清单不一致")
                if bearing.endswith("_4") and records != sorted(official_public[group],
                                                                 key=lambda row: row["meta"]["snapshot"]):
                    raise ValueError(f"{bearing} 的记录必须与公开包一致")
                groups[index - 1][f"{split}_rows"] = count
                groups[index - 1][f"{split}_entities"] = [bearing]
            self.rows[split] = grouped

        self.manifest.update(
            protocol=PROTOCOL, replay_data=source_digests,
            initial_checkpoint_already_saw_all_public_training=True,
            initial_checkpoint_saw_replay_training=False,
            validation_is_public=True,
            validation_is_public_not_blind=True,
        )
