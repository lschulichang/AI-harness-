# 数据与模型适配层

| 文件 | 职责 |
|---|---|
| gas_dataset.py | 读取气体 DAT 批次，校验摘要并固定训练、验证和留出测试划分 |
| bearing_baseline.py | 对接官方轴承回归模型 |
| bearing_pairs.py | 组织轴承训练与验证配对 |
| cmapss_baseline.py | 对接发动机模型、发动机分组和 NASA 测试 |
| rul_baseline.py | B/C 共用的回归适配逻辑 |

harness_core 负责调度、预算、候选反馈、历史门槛、选模和日志。气体输入转换方法见 ../docs/气体输入转DAT.md。
