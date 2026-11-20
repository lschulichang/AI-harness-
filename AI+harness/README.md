# 工业模型训练 Harness

项目位置为 `/home/adc/桌面/LXB/AI+harness`。默认的 A 气体分类使用滑动窗口训练师 `trainer_window_v4`。

## 启动

```bash
cd '/home/adc/桌面/LXB/AI+harness'
bash run_linux.sh --help

# 正常实验，默认使用 PyTRiO 千问与 SwanLab 云端记录
bash run_linux.sh

# 同一个闭环的规则调试入口
bash run_linux.sh run --strategy rules --swanlab-mode disabled

# 开发时选用 DeepSeek
bash run_linux.sh run --provider deepseek

# 查看已保存结果
bash run_linux.sh status 'runs/运行编号'

# 恢复滑动窗口运行
bash run_linux.sh run --resume 'runs/运行编号'
```

统一 Python 入口是 `run.py`，`python -m harness_demo` 调用同一主函数。启动脚本选择 `/home/adc/anaconda3/envs/AI+harness` 环境。密钥优先读环境变量，其次读 `local_secrets.py`。

## B 轴承剩余寿命回放

```bash
bash run_linux.sh run --dataset bearing_rul --strategy rules --swanlab-mode disabled
bash run_linux.sh run --dataset bearing_rul --strategy llm --max-api-calls 9 --max-experiments 9 --swanlab-mode offline
```

B 从原始检查点启动。三批训练与验证轴承分别为 Bearing1_4 → Bearing1_5、Bearing2_5 → Bearing2_4、Bearing3_4 → Bearing3_5。每根轴承使用完整快照。默认目录为 data/bearing_rul_replay，批次见其中的 replay.json。可用 --data-dir 指定另一份相同协议的配对数据，用 --baseline-dir 指定匹配的基线包。提示词使用回归指标和训练特征漂移摘要。训练窗口包含本批及已开放训练批，漂移参考取最近 3 批。所有候选从本批入口模型续训。

B 按分钟计算 RMSE 和 MAE。模型在试训前看到批初模型的验证分数、漂移与可用样本，可以决定是否尝试。试训后的结果才用于选模。首批默认要求 RMSE 比批初至少降低 1%。后续以最近 4 批批末 RMSE 均值为优先目标，候选须优于批初模型。若均未达到目标，仍可选最好的合格改善候选。最近最多 6 个历史工况均以本批入口模型在同一验证集上的 RMSE 为对照，默认增幅不得超过 50%。这两个阈值分别通过 JSON 配置的 `rul_min_relative_gain`、`rul_max_history_increase` 设置。每个运行的 `batch_results.md` 列出同一验证集上的静态基线、批初与选中模型指标。初始检查点已使用全部公开训练轴承，公开验证结果用于本地回放选模。

## C 涡扇发动机剩余寿命回放

~~~bash
bash run_linux.sh run --dataset cmapss_rul --strategy rules --swanlab-mode disabled
bash run_linux.sh run --dataset cmapss_rul --strategy llm --provider pytrio --max-api-calls 12 --max-experiments 12 --swanlab-mode offline
~~~

C 从 baseline/cpu-c-guide-20261110-v1 的开发检查点出发，按 FD001、FD002、FD003、FD004 回放。训练和验证沿用基线包按整台发动机划分的公开文件，输入是 3 项运行设置和 21 项传感器值。初始检查点已使用过全部公开训练轨迹，所以这是开发回放。C 与 B 共用滑窗、候选反馈和基于批初模型的历史门槛。RMSE 和 MAE 以循环数计。

全部批次选模结束后，使用 data/CMAPSSData 下的 test_FD00x.txt 和 RUL_FD00x.txt 测试静态基线与最终模型。每台测试发动机只评估其最后可见循环。NASA 分数按样本累计，越低越好。公开测试结果写入 summary.json 和 batch_results.md，供本地比较。--data-dir 可指定 NASA 原始测试数据目录，--baseline-dir 可指定兼容的 C 基线包。

## A 气体分类闭环

1. 加载官方 NumPy MLP 检查点，该模型已使用 batch1 训练。
2. 顺序回放 batch2–10，每批按 60% 训练、20% 验证、20% 留出测试分层划分。导入目录也可提供自己的批次。
3. 用最近 3 个已完成批次的训练集诊断当前训练数据，计算 PSI、KS 多重检验校正、Wasserstein、MMD 和标签 JS 散度。无历史时记录 no_reference，缺标签时 JS 为 null。
4. 把当前验证表现、漂移摘要、最近 3 批决策、可用回放批次及候选反馈发给大模型。
5. 大模型选择保留模型、仅本批续训或混合历史续训，并给出完整训练参数。程序校验动作、范围、重复方案和预算。
6. 训练窗口包含本批。window_size 为 1 时仅用本批，为 2–8 时合并窗口内各批的全部训练样本。历史不足时截短。同批所有候选从同一个批初模型出发。
7. 本批优先目标默认为最近 4 个已完成批次最终验证 Macro-F1 的均值。首批要求至少提高 min_gain，默认 1 个百分点。历史门槛检查最近 6 批验证集，每组默认最多下降 20 个百分点。候选须优于批初模型。未达到优先目标时，也可采用通过历史门槛且有改善的最佳候选。
8. 大模型根据候选真实结果决定继续或结束，每批默认最多 3 次训练。首轮提案失败使用规则兜底，已有候选后的反馈失败则结束搜索。
9. 批末确定模型后再评估留出测试，保存决定、模型和日志，继续下一批。

默认不设置整轮候选、API、Token 和单次超时上限。需要限制时通过 `--config` 或 `--max-experiments`、`--max-api-calls`、`--max-api-output-tokens`、`--gas-sample-budget`、`--trials-per-batch` 配置。

完整 batch2 此前用于官方公开验证，其测试成绩标记为开发留出。当前 DAT 训练回放需要真实标签，属于本地留出评估。正式平台入口须按 SDK 契约另行验收。

## 文件职责

```text
AI+harness/
├── run.py、run_linux.sh           统一运行入口
├── harness_demo/
│   ├── __main__.py               命令行
│   ├── config.py                 当前方案配置
│   ├── workflow.py               配置校验与恢复入口
│   ├── gas_mlp.py                模型包核验、运行组织与汇总
│   ├── gas_trainer_workflow.py    批次闭环、候选反馈与选模
│   ├── gas_trainer_policy.py      大模型提示词、方案校验、规则兜底
│   ├── gas_trainer_drift.py       漂移诊断
│   ├── gas_trainer_training.py    MLP 续训与早停
│   ├── gas_trainer_metrics.py     全类别诊断指标
│   ├── gas_mlp_data.py            数据与官方评分对接
│   ├── gas_import.py              CSV、JSONL、DAT 检查、转换与预测
│   ├── planner.py                 API 请求、重试与请求预算
│   ├── storage.py                 SQLite 账本、文件锁与原子保存
│   ├── tracking.py                SwanLab 日志
│   ├── metrics.py                 决策耗时、CPU 与内存
│   ├── portability.py             路径迁移
│   └── credentials.py             本机密钥读取
├── wrappers/gas_drift.py          批次读取与固定数据划分
├── tests/                        当前方案验收
├── docs/                         开发说明与气体输入格式说明
├── baseline/                     官方模型包
├── data/                         原始数据
├── runs/                         实验记录、模型和检查点
├── contest-harness-sdk-1.0.2/     官方 SDK 原包
├── scripts/setup_linux.sh         环境安装
├── requirements-harness*.txt      依赖
├── Dockerfile、.dockerignore      本地容器打包
└── local_secrets.py               本机配置，排除在 Git 和镜像之外
```

## 结果与恢复

每次运行使用日期、时间、任务、策略和唯一后缀命名。主要结果有：

- `summary.json` 保存平均和最差留出分、训练次数、API、Token、耗时及资源用量。
- `batch_results.md` 保存逐批验证、测试与更新决定。
- `batch_comparison.md` 保存本批更新前后在同一留出测试集上的结果。
- `audit_test_results.json` 保存详细逐批指标与三个窗口实际使用的批次。
- `history.json`、`api_requests.json`、`reports/` 保存方案、反馈、请求与诊断。
- `active.json` 指向选中的检查点，`experiments.sqlite` 和 `experiments/` 支持恢复。

旧结果仍可通过 status 查看。恢复支持 A 的 trainer_window_v4 和 B 的 bearing_window_v2 和 C 的 cmapss_window_v1，并校验该次运行的数据摘要、检查点与门槛。更早的运行可通过 status 查看。

主线已移除独立 static、fixed、legacy 对照、旧参数网格、固定参考漂移、固定比例抽样回放、旧模型搜索回滚和早期训练师分支。规则调试与失败兜底共用当前滑动窗口闭环。

## 安装与验收

```bash
bash scripts/setup_linux.sh
PYTHONDONTWRITEBYTECODE=1 /home/adc/anaconda3/envs/AI+harness/bin/python -m pytest tests -q -p no:cacheprovider
docker build -t harness-demo .
```

容器的现有 CMD 使用 rules 调试模式。模型包、数据和结果通过挂载提供。正式 SDK 协议见 `contest-harness-sdk-1.0.2/HARNESS_SDK.md`。

详细说明见 [开发与比赛适配](docs/开发与比赛适配.md)、[气体输入转 DAT](docs/气体输入转DAT.md) 和 [数据适配层](wrappers/README.md)。
