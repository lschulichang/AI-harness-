# A、B、C 比赛 Harness

提交目录为 `/home/adc/桌面/LXB/staging`。一个 Linux/amd64 镜像接入 SDK 1.0.2 的 `contest.harness.workspace.v2` 协议。平台启动 `python3 /app/harness.py describe|train|predict`，训练时按 `dataset_key` 分发。

| 任务 | 训练闭环 | 当前验证主指标 |
| --- | --- | --- |
| A 气体分类 | `trainer_window_v4`。保留气体漂移检测、可选混合回放、续训早停、候选反馈和当前批验证选模 | Macro-F1 |
| B 轴承寿命 | `bearing_window_v2`。保留回归训练、工况漂移、候选反馈和当前批验证选模 | RMSE（分钟） |
| C 发动机寿命 | `cmapss_window_v1`。复用原有回归闭环及 C 的模型和默认批大小 | RMSE（循环） |

平台每阶段作为原有闭环的一批。A、B、C 的混合续训固定用当前批和最近两批历史训练数据，历史不足时按可用批数执行。数据来自 SDK 提供的公开训练和验证文件，原项目的 DAT 划分、轴承配对和 NASA 测试回放入口留在开发项目。每个候选都从平台本轮指定的父检查点续训。候选提交后，程序使用宿主返回的当前批公开指标继续反馈和选模。历史验证回测默认关闭。需要诊断时可在本地容器设置 HARNESS_EVALUATE_HISTORY=1，计算最近最多 6 批历史验证指标并写入实验记录。公开摘要缺失的指标只在内部使用同一公开验证文件计算，并在审计记录中区分两种来源。隐藏集和正式评分由平台管理。

模型请求走 `rnd.model()` 网关。参赛模型身份和 PyTRiO API Key 在平台表单中填写。原有 SwanLab 记录使用离线模式，随状态中的 `audit.zip` 保存。滑窗状态仅保留最近七轮公开训练、最近六轮公开验证和最近阶段的决策摘要。构建上下文只收入代码、依赖清单和 SDK。

## 构建并导出

在可以访问基础镜像和 Python 依赖的 Docker 环境中运行：

```bash
cd '/home/adc/桌面/LXB/staging'
bash build_submission.sh
```

脚本构建并在只读根文件系统运行 `describe`，随后用 `docker image save` 导出并核验单个 Linux/amd64 镜像。上传生成的 `mirror/5.tar` 到比赛平台，再使用平台提供的试运行检查真实宿主。

## 更新开发代码

A、B、C 的算法文件由同级 `dev` 开发目录维护。算法有更新时先运行 `bash sync_sources.sh`，将当前闭环同步为仅使用比赛宿主网关的版本，再构建。比赛适配代码集中在 `contest_data.py` 和 `contest_submission.py`。
