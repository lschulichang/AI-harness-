"""轴承剩余寿命基线及按轴承分组规则。"""
from .rul_baseline import RulBaseline
from .rul_baseline import digest as digest
from .rul_baseline import read_rows as read_rows

PACKAGE = "cpu-b-rul-20261110-v1"


class BearingBaseline(RulBaseline):
    profile = "B"
    unit = "minutes"
    checkpoint_default_unit = None
    feature_count = 34
    group_names = ("condition1", "condition2", "condition3")
    dataset = "bearing_rul"
    protocol = "bearing_public_condition_replay"

    @staticmethod
    def identity(row):
        bearing = row["meta"]["bearing"]
        prefix = bearing.split("_")[0]
        if not prefix.startswith("Bearing") or not prefix[7:].isdigit():
            raise ValueError("无效轴承来源")
        return "condition" + prefix[7:], bearing
