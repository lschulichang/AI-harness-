"""C-MAPSS 剩余寿命基线与最后观测循环公开测试。"""
from .rul_baseline import RulBaseline

PACKAGE = "cpu-c-guide-20261110-v1"


class CmapssBaseline(RulBaseline):
    profile = "C"
    unit = "cycles"
    # 旧版 C 指导包的检查点没有 target_unit，任务合同定义单位为 cycles。
    checkpoint_default_unit = "cycles"
    feature_count = 24
    group_names = ("FD001", "FD002", "FD003", "FD004")
    dataset = "cmapss_rul"
    protocol = "cmapss_public_subset_replay"
