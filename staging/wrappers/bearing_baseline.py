"""轴承剩余寿命基线及按轴承分组规则。"""
from .rul_baseline import RulBaseline


PACKAGE = "cpu-b-rul-20261110-v1"


class BearingBaseline(RulBaseline):
    profile = "B"
    unit = "minutes"
    checkpoint_default_unit = None
    feature_count = 34
    group_names = ("condition1", "condition2", "condition3")
    dataset = "bearing_rul"
    protocol = "bearing_public_condition_replay"
