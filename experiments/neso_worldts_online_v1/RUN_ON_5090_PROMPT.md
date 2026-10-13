# Codex 执行任务：NESO Vintage-Aware WorldTS（真实数据、真实训练、在线回放）

你当前在一张 RTX 5090 的研究服务器上。请**直接执行**我提供的 `neso_worldts_online_v1` 工程，不要仅写实验方案或脚本。最终目标是得到 NESO 真实预测版本数据上的 WorldTS 模型、复现日志、在线回放指标及 GO/NO-GO 研究结论。

## 0. 工作边界和文件

- 工程来自 `neso_worldts_online_v1.zip`，优先在当前研究仓库中找到已经解压的 `neso_worldts_online_v1/`。若仅有压缩包，先解压；不要把工程误认成官方 WorldTS 仓库。若两者均不存在，说明缺少输入工程，而非从头编造一个相似实现。
- 先阅读 `README.md`、`CODEX_EXECUTION.md`、`STATUS.md`、`TEST_REPORT.md`、`configs/worldts_neso.json`，检查 `neso_online/{download,prepare,native_worldts}.py` 和测试用例。
- 原作者源码：`https://github.com/decisionintelligence/WorldTS`，固定 commit `269426ea6217d7152a244da0e209d922acd46d4f`。必须导入作者的 `WorldTSModel` 和 `LatentStateLoss`，不得偷偷改成另一个 Transformer。
- 本轮**不加 Loop、不引入 Flow Matching、不改原 Encoder/Predictor/Decoder 架构**。可以为 NESO 数据做必要接口适配，但所有改动均有 diff 与理由。
- 种子 2021，先做真实数据、Stage1/Stage2 WorldTS 和强基线；只有达到预设的更新改进空间才运行轻量单步 latent updater。
- 如果已有研究仓库 Git 工作区，请新建独立实验分支或目录，保留现有实验。大数据、checkpoint 和中间数组不要提交 Git，只提交源码、配置、日志摘要、结果 CSV/JSON、文件哈希和报告。

## 1. 环境与 CUDA

检查 `nvidia-smi`、Python、CUDA/PyTorch、RAM、磁盘空间和联网状况。优先沿用服务器已经验证能运行 RTX 5090 的 PyTorch CUDA 版本（例如 2.8+cu128），**禁止用 WorldTS/DAG 旧版 requirements 把其降级到不支持 Blackwell 的 torch 2.4.1**。隔离环境，先执行一次 GPU forward/backward，并记录实际硬件和 `pip freeze`。

在工程根目录执行（命令按当前 Bash/PowerShell 自动改写，不得把多行 Bash 续行符原样放入 PowerShell）：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
git clone https://github.com/decisionintelligence/WorldTS.git upstream/WorldTS
git -C upstream/WorldTS checkout 269426ea6217d7152a244da0e209d922acd46d4f
```

如果 `upstream/WorldTS` 已存在，检查 commit 而不是覆盖。根据实际 ImportError 补充与当前 CUDA 兼容的轻量依赖。运行项目里的 synthetic fixture 测试，但**合成数据的通过不算 NESO 真实实验结果**。在跑全年数据之前，还要用真实记录进行官方核心模型的最小 forward/backward + stage1/stage2 smoke test。

## 2. 下载与验证真正的 NESO 档案

使用工程内下载器：

```bash
python -m neso_online.download --resources forecast_2025 forecast_2026_jan_jun forecast_2026_jun_dec demand_2025 demand_2026 --out data/raw
```

2026 的档案可能分批、覆盖重叠，且 2026 年后半段尚未完整结束。应按资源 ID、文件 SHA256、行数、时间覆盖、字段名和重复键实际审计，不得根据预期行数宣称完成。若 `/datastore/dump/{id}` 不可用，切换 NESO 官方数据门户的 Download CSV，不得从不明镜像获取替代数据。

验证 `Forecast_Datetime`、`DATE_GMT`、`TIME_GMT`、`SETTLEMENT_DATE`、`SETTLEMENT_PERIOD`、`EMBEDDED_WIND_FORECAST`、`EMBEDDED_SOLAR_FORECAST`，以及需求中的 `ND`、`FORECAST_ACTUAL_INDICATOR`。检查预报生成时间是否可作为可用时间、UTC/BST、每个结算日 46/48/50 个半小时、重复版本冲突、缺失值与极值。严禁用较晚预测版本或未来 ND 填补较早预测缺失。如果真实档案字段或时间定义与程序假设不符，修复解析逻辑并写回归测试。

## 3. 构建严格的 A/B 同目标版本对

运行：

```bash
python -m neso_online.prepare --forecast data/raw/forecast_2025.csv data/raw/forecast_2026_jan_jun.csv data/raw/forecast_2026_jun_dec.csv --demand data/raw/demand_2025.csv data/raw/demand_2026.csv --lookback 336 --horizon 48 --revision-hours 6 --out outputs/dataset
```

任务是过去 336 个半小时的 ND、风电/光伏序列，预测未来 48 个半小时 ND。对相同 48 个目标时间点，构建 A 时刻协变量预测 `cov_a[48,2]` 与 B 时刻更新预测 `cov_b[48,2]`；只在目标时间晚于 B 的位置计分。保存 `issue_a`,`issue_b`,`valid_times`,`update_mask`；必须证明 A/B 预测对应完全相同的真实目标时间。

**第一轮固定 A 时刻的目标历史**，B 仅替换真正于 B 已发布、且对应未结束时段的风/光预测，隔离 covariate vintage 修订。将其命名为 `fixed-A-history covariate-vintage replay`。不得声称这是所有新增观测均实时可见的完整线上系统。

按目标标签完整覆盖区间分割：2025 train；2026 Q1 validation；2026-04-01 至 2026-08-31 test。跨边界的窗口丢弃，不让任一真实目标时刻同时进入训练、验证和测试。标准化只 fit TRAIN。输出真实样本数、每目标的版本数量、版本差大小、版本间隔、A/B 交集长度、缺失与排除原因；并检验严格同目标、时序单调和无未来标签泄漏。

**特别重要的因果限制：** NESO Historic Demand 汇总数据可能延后发布，也可能事后修订；`Forecast_Datetime` 记录预报生成时间，未必是第一公开时间。如果无法证实 ND 在 A 的实时可用性，本轮应明确称 `retrospective forecast-vintage replay with assumed ND availability`；不得写成严格真实上线可用的 Online Learning / Operational Online Forecasting。若能找到相应的历史实时已发布 ND 源，可单列严格 as-of 协议，不得混在 V1 结果中。

## 4. 真正训练官方 WorldTS 数值主干

先检查代码是否与固定 commit 的 `WorldTSModel`、`LatentStateLoss` 接口完全兼容。使用 NESO V1 配置（不是作者 NP/PJM 官方超参），训练两阶段：

```bash
python -m neso_online.native_worldts train --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021 --epochs1 20 --epochs2 20 --patience 5
```

Stage1 更新 Encoder、Condition Encoder 和 Predictor，优化作者 latent matching + cosine + regularizer；Stage2 冻结上述模块，仅训练原始 Decoder，采用预测状态 MAE 和真实编码状态辅助重构 MAE。两个阶段分别按 validation loss 选 checkpoint，不能使用 TEST 选择参数。必须确认 Stage2 冻结参数的 hash 不变、Decoder 梯度非零、Stage1/2 最佳轮次、训练曲线和早停行为。若 Stage1 最优在 Epoch1，真实记录而非为了成绩篡改 Early Stopping。

从 NP/PJM 的现有结果不能推断 NESO 成绩；NESO 是新任务、采用适配配置，必须写作 `WorldTS native-core NESO adaptation`，而非“官方 WorldTS 在 NESO 的论文复现”。

## 5. 基线：先验证是否真的存在在线更新问题

在**验证集**运行并独立核对：

- `Old`：A 历史 + A 版未来协变量生成并缓存的旧预测。
- `Recompute`：保持 A 历史完全相同，在 B 用 B 版未来协变量执行**完整 WorldTS 重新计算**，这是核心强基线，但准确称为 `covariate-only recompute`，不是使用 B 时刻新增 ND 的完整在线重算。
- `Seasonal Naive`：相同目标时间的上周负荷预测，检查半小时槽位是否正确对齐。
- `Output Correction`：仅用 TRAIN 训练的输出空间 Ridge/线性修正，输入仅为新旧协变量差；禁止读取验证或测试未来标签。

评估：B 时刻仍未结束的相同 target slots 上的 MAE、MSE、按窗口胜负比例、条件更新幅度分组、预测提前量分组、更新反而恶化比例；同时测量 B 时刻新增计算的 latency、peak CUDA memory、参数量，能实测再报 FLOPs。推理计时 GPU 必须同步并做 warmup。检验代码中逆标准化后是 MW/ MW² 还是 normalized MSE，不得将两者混写。检查 Seasonal Naive 的索引是否存在半小时错位。连续窗口高度重叠，用按周/跨多天时间块的 bootstrap 做描述性区间，不把每窗口当独立样本。

优先执行 `train` 自动生成 `outputs/model/train_validation.json`；分析该结果以后再决定下一项，不要一开始无条件运行 updater。

## 6. 条件性训练轻量 Latent Updater

只有当真实的版本差非平凡，验证集上 old/recompute 存在有意义的差异，且简单 output correction 不能解释所有收益/退化，才运行：

```bash
python -m neso_online.native_worldts train_updater --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021 --updater-epochs 20 --patience 5
```

冻结 WorldTS，训练工程内的单步 `LatentRevisionUpdater`。只提供缓存的 A latent、A/B 协变量和 update mask；其推理不能读取 B 之后的未来标签。比较 `Updater`、`Output Correction` 和 `Recompute` 的准确率—增量计算成本权衡，记录相同信息预算。若 Updater 不优于简单修正，也不能用“参数更少”替代实际任务收益。暂时不加 Loop。

## 7. 终端测试，只运行已冻结的方案

在记录验证集选择依据、固定超参和 checkpoint 之后才执行：

```bash
python -m neso_online.native_worldts replay --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021
```

TEST 只执行一次最终评估，不用 TEST 调参或重新挑选方法。若此前已经观察过测试结果，明确测试不再完全独立。需保存 test predictions、mask、时间索引、各方法 metrics、环境和 hash。若原工具没有导出必要的逐样本数组或版本时间，先做最小扩展并加测试。

## 8. 必须交付的文件与判断

在工程或仓库 `experiments/neso_worldts_online_seed2021_v1/` 下生成：

- `DATA_AUDIT.md`：五个真实原始资源的版本、SHA256、时间范围、重叠和缺失统计，A/B 预测版本配对率、版本变化幅度、样本数量、时区与 as-of 信息可用性。
- `ENVIRONMENT.md`、`pip_freeze.txt`、`ISSUE_LOG.md`：设备、CUDA、真实执行命令、完整报错和修复 diff。
- `CONFIG_PROVENANCE.md`、`train_history.csv`：NESO 改编配置 vs WorldTS 官方原配置，Stage1/Stage2 checkpoint 和收敛曲线。
- `VALIDATION_RESULTS.csv`、`TEST_RESULTS.csv`、`REVISION_ANALYSIS.csv`、`COMPUTE_RESULTS.csv`：旧预测、完整重算、朴素法、输出修正，以及有条件训练的 Updater 全部结果。
- `REPRODUCE.md`：一键运行/从中断恢复的真实命令，支持 Windows PowerShell 或 Bash，注明实际平台。
- `FINAL_REPORT.md`：真实数据与训练是否完成；在什么条件下 B 版重算改善/恶化；Update Headroom 是否显著；普通输出修正是否已足够；Latent Updater 是否有独立价值；最终 GO/NO-GO。

报告还必须区分 `REAL_NESO`、`SYNTHETIC_TEST`、`PAPER_REFERENCE` 三类结果，禁止混淆。如果因为接口、数据或计算问题无法完成某一步，明确 `BLOCKED` 并提交已完成的结果与全部诊断；不要虚构分数，也不要把通过 unit test 写成训练完成。

## 9. 执行风格

这是执行任务，**不是让你再给我写一个研究建议**。按 `环境检查 → 真实数据下载 → 数据审计 → 构建样本 → CUDA smoke → WorldTS Stage1/Stage2 → validation 在线回放 → 条件性 updater → 冻结后 TEST → FINAL_REPORT` 逐项真正运行。出错先定位和修复，并继续执行；不得仅因脚本第一次报错就停止，也不得为得到正结果擅自降低数据检查或修改目标定义。

**最后明确回答：WorldTS 是否在 NESO 的真实预报版本更新环境中表现出可修复的性能/计算瓶颈？如果没有，给出 NO-GO，不要强行设计新模型。**
