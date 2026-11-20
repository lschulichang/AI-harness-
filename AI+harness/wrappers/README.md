# 气体数据适配层

`gas_drift.py` 负责读取原始或转换后的 DAT 批次，校验数据摘要，并按固定随机种子分层划分训练、验证和留出测试。

特征维度和类别顺序来自官方模型检查点。导入目录的批次清单由 gas_import 生成。

本目录对外导出 GasDriftData。批次调度和窗口组装由 harness_demo/gas_trainer_workflow.py 完成，MLP 与官方评分实现来自 baseline/gas/baselines。

输入转换方法见 ../docs/气体输入转DAT.md。
