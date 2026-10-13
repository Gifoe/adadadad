# Codex Research Task: Why Does WorldTS Stage 1 Peak at Epoch 1?

## 0. 研究目标

请在现有 GitHub 仓库中开展一项完整、可复现的研究可行性实验。

**Experiment name:** `worldts_stage1_state_selection_audit_seed2021_v1`

**Repository:** https://github.com/Gifoe/adadadad

**Official WorldTS:** https://github.com/decisionintelligence/WorldTS

**Official source commit:** `269426ea6217d7152a244da0e209d922acd46d4f`

请阅读以下已有研究结果：

1. `experiments/baseline_reproduction_seed2021_v1/FINAL_REPORT.md`
2. `experiments/baseline_reproduction_seed2021_v1/results/worldts_stage1_metrics.csv`
3. `experiments/baseline_reproduction_seed2021_v1/results/worldts_stage2_metrics.csv`
4. `experiments/worldts_readout_alignment_feasibility_seed2021_v1/outputs/FINAL_RESEARCH_REPORT.md`
5. `experiments/worldts_future_response_loop/FINAL_RESEARCH_REPORT.md`，如当前分支缺失，请从 `codex/worldts-future-response-loop` 分支读取。

### 0.1 已有实验发现

WorldTS 官方模型已经完成数值复现，在 NP/PJM 的 H24/H360 四项任务上，Stage 1 最佳 Validation Latent Loss 都出现在 Epoch 1。

尤其是两个长预测任务：

| Dataset | Metric | Epoch 1 | Epoch 6 |
|---|---|---:|---:|
| NP-H360 | Training Latent Loss | 5.1806 | 3.1248 |
| NP-H360 | Validation Latent Loss | 6.8009 | 8.6687 |
| PJM-H360 | Training Latent Loss | 4.0953 | 2.6438 |
| PJM-H360 | Validation Latent Loss | 4.4428 | 7.7197 |

训练 Loss 改善，但验证 Loss 明显恶化。

同时，之前的 Predictor-only 审计发现，在冻结 Encoder 和 Decoder 后，继续优化 Predictor 可以改善 NP 的验证 Forecasting MAE。

然而之前没有直接比较：Stage 1 不同 Epoch 的状态经过公平 Decoder 训练后，最终预测表现究竟如何。

### 0.2 核心 Research Question

**Does minimizing the Stage 1 latent validation objective select the most useful predictive representations for downstream forecasting?**

我们要回答：

1. Epoch 1 最佳是否是真实且稳定的验证现象？
2. Validation Latent Loss 恶化究竟是预测能力下降，还是潜在目标分布和尺度发生变化？
3. Epoch 1 的 Latent 最优状态，是否也是最终 Forecasting 最优状态？
4. 继续训练是否产生了仍有预测价值、但未被 Stage 1 Checkpoint Selection 利用的表示？

### 0.3 竞争性假设

**H1：Ordinary Generalization Failure**

Stage 1 在 Epoch 1 之后过拟合或优化不稳定，导致 Latent Prediction 和 Downstream Forecasting 同时恶化。

预期：Epoch 1 在统一评估协议和公平 Decoder 适配之后仍然最好。

**H2：Latent Objective / Model Selection Mismatch**

后续 Epoch 的 latent validation objective 更差，但对应的状态经过公平 Decoder 适配后，反而产生更好的最终 Forecasting。

预期：最佳 Latent Epoch 与最佳 Forecast Epoch 系统性不一致。

**H3：Moving Target / Latent Evaluation Instability**

因为 Stage 1 同时训练 Encoder 与 Predictor，Target Latent 本身不断变化，加上 VICReg 的 Batch 依赖性，导致跨 Epoch Latent Loss 不完全具有直接可比的含义。

预期：表示尺度、有效秩、Target Statistics 或 Batch Grouping 可以解释部分验证 Loss 变化。

这些机制可能同时存在，不要强行假设它们互斥。

本轮只验证问题，不预先设计解决方法。

---

## 1. 实验总体要求

### 1.1 数据与模型

第一轮只使用：

| Dataset | Input Length | Forecast Horizon | Seed |
|---|---:|---:|---:|
| NP | 720 | 360 | 2021 |
| PJM | 720 | 360 | 2021 |

保持官方 WorldTS：

- Causal Target Encoder。
- Independent Covariate Encoder。
- PatchTST State Predictor。
- Future Covariate Residual。
- Causal Patch Decoder。
- RevIN。
- 原始 Latent Loss。
- 原始 Stage 2 Decoder Objective。
- 原始 Future Covariate Availability Protocol。

不修改网络结构、Patch、Latent Dimension、Predictor Width、Attention、Loss 权重或任何基础设置。

现有权重和数据位于服务器：

`D:\baseline_reproduction_seed2021_v1`

请读取原始配置、Manifest、Checkpoint 路径和训练日志，不要直接假设某个文件一定存在。

### 1.2 本轮禁止

- 不加入 Loop。
- 不加入 Jacobian Loss。
- 不加入新 Regularizer。
- 不设计 Adapter、Stitching 或新的 World Model。
- 不根据 Test Performance 挑选 Epoch。
- 不直接使用旧 Decoder 评价未经对齐的新 Epoch Latent 并将其解释成状态质量。
- 不覆盖任何历史模型、Checkpoint 或实验结果。
- 不因为某次结果不好而重新挑 Seed。
- 不将先前已经看过的 NP/PJM Test Set 当成全新独立测试集。

所有结果必须保存在新的独立实验目录。

---

# Phase 0：恢复训练现场并审计 Checkpoint

首先检查已有 WorldTS Stage 1 的 Epoch Snapshot。

重点确认：

- Epoch 1、3、6 的模型权重是否仍然存在。
- Epoch Snapshot 是当轮训练完成后的真实权重，还是 Early Stopping 恢复的最佳权重。
- Decoder 在 Stage 1 期间是否始终保持冻结。
- Snapshot 是否包含完整 Target Encoder、Condition Encoder、Predictor 与相关模块。
- 每个 Snapshot 是否能正确恢复模型状态。
- 对应的 Optimizer、Scheduler、Random State 和 Config 是否可恢复。
- 原始 Stage 1 Loss 与 Snapshot 是否匹配。

原版训练在 Stage 1 完成后会恢复最佳 Checkpoint。因此绝不能误把恢复后的 Epoch 1 权重标记为 Epoch 6。

### 如果已有 Snapshot 完整

直接复用，不重新训练 Stage 1。

### 如果缺少 Epoch 3 或 6

使用官方初始化、官方代码、官方训练配置重新运行 Stage 1。

要求：

1. 完整保存 Epoch 1–6 的原始权重。
2. 不允许 Early Stopping 在 Epoch 6 之前终止。
3. 维持官方学习率、Scheduler、Batch、Loss、随机种子及训练数据顺序。
4. 只改变实验所需的 Snapshot 保存和提前停止行为。
5. 如果新训练轨迹与历史结果不同，保留差异，不强行调整训练来匹配旧数值。

建立模型完整性测试，确认 Stage 1 的 Decoder 未被意外更新。

输出：

`outputs/PREFLIGHT_REPORT.md`

`outputs/checkpoint_inventory.csv`

`outputs/checkpoint_hashes.json`

---

# Phase 1：为什么 Stage 1 在 Epoch 1 最好？

本阶段不修改模型，不设计方法。

## 2.1 统一 Train / Validation Evaluation

原始 Training Loss 是训练过程中每个 Batch 的在线平均，而 Validation Loss 来自 Epoch 结束后的模型。

这不是完全对称的比较。

请在每个 Epoch Snapshot 上，重新执行同样的 `eval()` Forward。

同时计算：

- Train Probe Latent Loss。
- Validation Latent Loss。
- Latent L1。
- Trajectory Cosine Loss。
- Variance Regularization。
- Covariance Regularization。
- Total Latent Objective。

确保 Train 与 Validation 使用：

- 相同的 Forward 逻辑。
- 相同的评估模式。
- 相同的 Batch Size。
- 相同的 Loss Reduction。
- 相同的归一化规则。
- 相同的固定样本与评估顺序。

Train Probe 必须从原始训练集按预设规则抽样，也可以在资源允许时计算完整 Train Loss，但不能混用在线 Training Loss 和 Snapshot Evaluation Loss 来计算 Generalization Gap。

记录：

\[
G_e=
L_{\mathrm{val}}(e)-L_{\mathrm{train}}(e)
\]

并展示 Epoch 1–6 的完整曲线。

## 2.2 检查 Batch-dependent Evaluation

WorldTS 的 Variance/Covariance Regularizer 涉及 Batch Statistics。

因此对同一组模型、同一组验证窗口，比较：

1. 固定顺序与固定 Batch Partition。
2. 五个预先声明的随机 Shuffle Partition。
3. 仅 L1 + Cosine，不包含 Batch Regularizer。
4. 原始完整 Objective。

必须保持相同样本集合。

目的：检查 Epoch 1 最优是否对评估顺序或 Batch 分组敏感。

需要注意，Shuffle 不应改变样本集合，也不能在多次 Shuffle 中挑一个有利结果作为正式指标。

报告各个 Epoch 的：

- 完整 Objective 均值与分散程度。
- 不含 Batch Regularizer 的 Objective。
- 分量的变化趋势。
- 最佳 Epoch 排名是否稳定。

如果 Epoch 1 的优势主要来自评估时 Batch Statistics，而不是稳定的预测状态差异，这将改变后续研究判断。

## 2.3 检查 Latent Target 是否发生变化

WorldTS 的 Target Latent 是：

\[
Z_e^*=E_e(Y_{\mathrm{future}})
\]

预测状态是：

\[
\hat Z_e=P_e(X_{\mathrm{history}},C)
\]

因为 \(E_e\) 随着 Stage 1 更新，不能假设不同 Epoch 的 Latent Coordinate System 不变。

对于 Epoch 1、3、6，使用完全相同的输入样本，分别提取：

- `h_hist`
- `h_target`
- `h_pred`

计算：

**表示统计：**

- Latent Mean。
- Latent Standard Deviation。
- Latent L2 Norm。
- Effective Rank。
- Singular Value Spectrum。
- Target/Prediction Variance。
- Prediction–Target Cosine。
- 每个 Epoch 的官方 Latent Objective 分量。

**跨 Epoch 表示比较：**

- Linear CKA。
- Centered CKA（如果实现不同，应说明）。
- 基于训练样本拟合的 Orthogonal Procrustes Alignment。
- 对齐后的验证集表示误差。

历史状态、真实未来目标状态、预测未来状态应分开分析。

注意：

1. CKA 衡量的是表示结构相似性，不是预测信息的直接度量。
2. Procrustes 映射只可使用训练样本拟合，验证样本只能评估。
3. 不允许未经对齐就直接比较不同 Epoch 的 latent 坐标。
4. 不能把表示发生变化直接称为 Representation Collapse 或 Information Loss。
5. 不允许使用真实未来标签拟合任何部署时会使用的预测映射。

本阶段只做诊断，不通过这些指标挑选新的最佳模型。

## 2.4 结果分析

必须分别判断：

**A. 是否存在真实的 Train–Validation Divergence？**

**B. Validation Loss 上升主要由哪个 Loss 分量解释？**

**C. 不同 Epoch 的 Target Latent 是否发生显著的尺度或几何变化？**

**D. Epoch 1 的优势是否对 Batch Partition 稳定？**

**E. 是否存在可以排除的评估实现问题？**

完成：

`outputs/STAGE1_DYNAMICS_AUDIT.md`

`outputs/stage1_epoch_metrics.csv`

`outputs/latent_geometry.csv`

`outputs/batch_partition_sensitivity.csv`

`outputs/figures/`

---

# Phase 2：Latent 最优 Epoch 是否等于 Forecast 最优 Epoch？

这是本次研究最关键的实验。

**研究问题：Stage 1 的验证 Latent Loss，是否能够正确选择下游预测最有价值的状态？**

只选择三个预先确定的 Epoch：

- Epoch 1。
- Epoch 3。
- Epoch 6。

不要根据 Phase 1 的表现重新挑 Epoch。

## 3.1 统一 Decoder Training Protocol

对于每个 Epoch，构造完整的 Stage 1 模型：

\[
S_e=(E_e,P_e,C_e)
\]

其中 \(C_e\) 包括相应的外部条件编码分支。

冻结全部 Stage 1 模块，仅训练对应的 Decoder：

\[
D_e
\]

三个 Decoder 必须使用：

- 完全相同的 Decoder Architecture。
- 完全相同的初始化权重。
- 相同的训练数据。
- 相同的 Batch Size。
- 相同的 Optimizer。
- 相同的 Learning Rate。
- 相同的 Scheduler。
- 相同的随机数据顺序。
- 相同的最大 Optimizer Steps。
- 相同的 Validation Frequency。
- 相同的 Checkpoint Selection Criterion。

非常重要：

**不能把原版 Stage 2 已经训练完成的 Decoder 当作所有 Epoch 的共同初始 Decoder。**

因为那个 Decoder 已经针对 Epoch 1 的状态学习过，使用它初始化会天然偏向 Epoch 1。

请使用原始 Stage 2 开始之前的未训练 Decoder 初始化权重，或者按完全相同的随机状态重新初始化 Decoder。

每一组实验启动前验证 Decoder 的 Tensor Hash 一致。

对所有组使用相同的 Stage 2 Training Objective：

\[
L_{\mathrm{decoder}}
=
L_{\mathrm{predicted-state}}
+
\alpha L_{\mathrm{target-reconstruction}}
\]

其中 \(\alpha\) 使用实际复现配置中的官方数值；现有 NP/PJM 复现为 0.3。

保持官方 Stage 2 的 History Context、RevIN、Causal Patch Decoder 和完整预测接口。

验证时只能使用真实部署预测路径，不得将 Future Target Latent 送入预测器。

## 3.2 正式训练矩阵

| Dataset | Stage 1 Snapshot | Decoder |
|---|---|---|
| NP-H360 | Epoch 1 | 新训练，相同初始化 |
| NP-H360 | Epoch 3 | 新训练，相同初始化 |
| NP-H360 | Epoch 6 | 新训练，相同初始化 |
| PJM-H360 | Epoch 1 | 新训练，相同初始化 |
| PJM-H360 | Epoch 3 | 新训练，相同初始化 |
| PJM-H360 | Epoch 6 | 新训练，相同初始化 |

共六次正式 Decoder 训练。

Stage 2 基于原始官方训练配置：

- Batch Size：32。
- Epoch Budget：20。
- NP Learning Rate：1e-4。
- PJM Learning Rate：2e-4。
- 使用官方 Optimizer 与 Scheduler。
- 使用官方 `stage2_decoder_mix` 损失。
- Seed：2021。

为了控制等预算比较，建议六组都训练完整的预设最大更新预算，并保存逐 Epoch 权重；不要让不同早停步数造成不对称的训练机会。

Checkpoint Selection 可以在统一预算内根据相同的验证准则完成。

所有方法都必须记录训练过程中的完整曲线，不能只保留最终结果。

## 3.3 预先锁定 Evaluation Protocol

不能再将整个历史验证区间直接用于所有决策。

在原有 Validation 时间区间内部，按时间顺序建立：

- `V_select`：用于 Decoder Checkpoint Selection。
- `V_diagnostic`：只用于最终诊断比较。

两者之间设置不小于 `T + H - 1` 的原始时间步 Embargo，防止两组滚动预测窗口的原始输入和目标时间区间直接重叠。

必须确保：

1. 两组都来自原始 Validation Period。
2. 不使用原始 Test Period。
3. 不重新拟合原始 StandardScaler。
4. 未来协变量仅来自原官方协议允许的输入。
5. 不使用 `V_diagnostic` 选择 Loss 权重、训练 Epoch、模型结构或学习率。

需要明确标注：因为我们过去已经使用过整个原始验证区间进行 Baseline Selection，`V_diagnostic` 只是新的时间隔离诊断集，不能声称它对历史研究流程而言是完全未接触过的独立测试集。

一旦第一轮出现值得继续的结果，必须另外引入未参与此前方法选择的新数据集进行确认。

## 3.4 核心比较

分别记录每组模型在 `V_select` 和 `V_diagnostic` 上的：

- Forecast MAE。
- Forecast MSE。
- Decoder Reconstruction MAE/MSE。
- Decoder Training MAE。
- Best Decoder Epoch。
- Last Decoder Epoch。
- Decoder 参数量。
- Training Time。
- Peak GPU Memory。

定义：

\[
e_L^*=\arg\min_e L_{\mathrm{latent,val}}(e)
\]

以及：

\[
e_Y^*=\arg\min_e L_{\mathrm{forecast,val}}(e)
\]

其中 \(e\in\{1,3,6\}\)。

`e_L*` 使用预先固定的 Latent 评估协议确定；`e_Y*` 使用相同训练预算与模型选择程序产生的 Decoder 结果确定。

除比较 Best Decoder Checkpoint 外，还要比较相同训练步数处的 Forecasting Error，排除某个 Epoch 的状态只是 Decoder 学习速度不同。

特别关注以下情况：

**Case 1：**

Epoch 1 的 Latent Loss 最优，同时 Forecasting Performance 也最优。

这支持继续训练后状态质量或泛化性能下降，但不能单凭此结果区分过拟合、移动目标或其他优化问题。

**Case 2：**

Epoch 1 的 Latent Loss 最优，但 Epoch 3/6 的 Forecasting Performance 更好。

这支持 Stage 1 Checkpoint Selection 与下游预测能力不一致。

然而仍需判断收益是否源自 Decoder 收敛、训练预算或表示尺度差异。

**Case 3：**

不同 Epoch 的 Forecasting Performance 基本一致，但 Latent Loss 明显不同。

说明当前 Latent Validation Loss 的变化可能没有同等大小的下游影响。

这不一定需要新方法，但值得进一步确认 Loss 是否适合作为模型选择准则。

**Case 4：**

NP 与 PJM 的 Epoch 排序不同。

需报告任务依赖性，不允许简单平均后宣称普适模型选择问题。

---

# Phase 3：统计分析与研究决策

本次 Phase 3 只进行结果分析，不训练新模型。

## 4.1 必须提供的图表

1. Training/Validation Latent Loss vs Epoch。
2. Latent L1/Cosine/Regularization vs Epoch。
3. Prediction/Target Latent Norm 与 Effective Rank vs Epoch。
4. Cross-Epoch CKA / Procrustes Alignment。
5. Decoder Training Curves，Epoch 1/3/6 三组重叠比较。
6. Stage 1 Latent Loss vs Downstream Forecasting MAE。
7. 按时间区块比较 Epoch 1/3/6 的 Forecast Errors。

请保留所有数值的原始 CSV，而不只是图片。

由于滚动预测窗口高度重叠，使用连续时间块分析差值，不把全部窗口当成独立样本进行常规显著性检验。

报告至少三个 Block Length 的描述性 Bootstrap 结果，并说明有效独立信息量的限制。

## 4.2 训练预算与结果可靠性

必须报告：

- 每组实际 Optimizer Steps。
- 每组 Epoch 数。
- 每组训练耗时。
- 峰值 GPU Memory。
- 每组完整随机状态和 Checkpoint Hash。
- 冻结 Stage 1 权重是否保持不变。
- Decoder 初始化是否完全相同。
- 模型是否存在数据泄漏。
- 是否进行了任何因结果不佳而增加的训练。

如果某组 Decoder 未收敛，不允许直接断言对应潜在状态缺乏预测信息。需报告收敛证据、固定步数对照及相应限制。

---

# 5. GO / NO-GO 决策标准

请在实际实验前固定以下研究判断标准。

## GO：值得继续研究 State Selection

至少需要观察到：

1. Stage 1 Latent Loss 在 Epoch 1 最优的现象经过统一评估后仍成立。
2. Epoch 3 或 6 在相同 Decoder Training Budget 下获得明显更好的下游预测。
3. 优势不能简单解释为 Decoder 初始化、训练不足或评估协议变化。
4. 结果在时间区块分析中具有一定稳定性，而非只来自极少量窗口。
5. NP/PJM 结果具有明确的一致性，或者数据集差异能够提出具体可验证的新机制。

可将超过最强比较对象约 2% 的验证 Forecast MAE 改善作为值得关注的探索性效果尺度，但它不是统计显著性标准，不允许为跨越门槛而调参。

即便满足这些条件，也只能进入下一阶段的独立数据集验证，不代表已有完整论文贡献。

## GO：转向 Generalization / Optimization Audit

如果后续 Epoch 的 Latent 和 Forecasting Performance 都明显变差，则不要研究 Readout-aware Loss。

下一轮才调查：

- Stage 1 Learning Rate。
- Early Stopping。
- Encoder / Predictor Freeze Schedule。
- Representation Scale Drift。
- Target Representation Stability。
- Train–Validation Distribution Shift。

此时尚不能直接认定是接口漂移。

## NO-GO：不继续开发新方法

如果：

- Epoch 1 最优主要是评估协议造成的假象；
- 或不同 Epoch 的下游预测差异很小；
- 或改善来自 Decoder 训练预算不匹配；
- 或 NP/PJM 的结果无法支持稳定的研究命题；

则明确判定当前方向缺乏方法创新依据。

不要因为结果不支持假设，就自动建议增加复杂模块。

---

# 6. 文件结构与交付要求

请建立：

`experiments/worldts_stage1_state_selection_audit_seed2021_v1/`

建议结构：

    README.md
    REPRODUCE.md
    configs/
    scripts/
    src/
    tests/
    outputs/
        PREFLIGHT_REPORT.md
        STAGE1_DYNAMICS_AUDIT.md
        CROSS_EPOCH_DECODER_REPORT.md
        FINAL_RESEARCH_REPORT.md
        NEXT_RESEARCH_DECISION.md
        checkpoint_inventory.csv
        stage1_epoch_metrics.csv
        latent_geometry.csv
        batch_partition_sensitivity.csv
        decoder_training_results.csv
        decoder_training_curves.csv
        paired_block_statistics.csv
        research_decision.json
        figures/
        checkpoints/
        logs/

如果权重较大，可以只保留在服务器，但 Git 中必须记录准确路径、SHA256 与恢复方法。

## 6.1 自动化测试

至少覆盖：

1. 完整恢复每个 Stage 1 Snapshot。
2. Epoch 1/3/6 的 Encoder/Predictor 权重确实不同。
3. 防止把 Early Stopping 恢复的最佳权重误认为后续 Epoch。
4. Decoder 使用相同初始权重。
5. Stage 2 只更新 Decoder。
6. Future Target 不进入预测路径。
7. 所有方法使用相同 Train/Validation Origin。
8. 原始 RevIN 与 Inverse Normalization 保持一致。
9. 同一模型的验证预测可重复。
10. Batch Partition Sensitivity 只改变 Batch Grouping。
11. Snapshot 与新训练 Checkpoint 均可正确加载。
12. Forecast MAE/MSE 与独立数值计算一致。

任何实验预检失败，必须先修复，不允许继续生成看似完整的研究结论。

---

# 7. 计算环境与执行

使用现有 Windows 服务器、单张 RTX 5090 32GB。

优先复用：

`D:\baseline_reproduction_seed2021_v1`

中的 Python 环境、WorldTS 官方源码、数据文件和模型权重。

不要为了本次实验重新安装或降级 PyTorch/CUDA。

原版 WorldTS Stage 1、Stage 2 已经在当前服务器上成功运行，不需要重新部署完整 Benchmark。

优先级：

1. 发现已有 Stage 1 的 Epoch 1/3/6 Snapshot。
2. 完成 Phase 0 预检。
3. 完成 Phase 1 统一 Loss 与表示几何审计。
4. 完成 Phase 2 六组 Decoder 训练。
5. 完成 Phase 3 统计与研究判断。
6. 整理结果，提交代码与完整报告。

如果缺失某个 Snapshot，仅补做必要的 Stage 1 训练。

不要因为 Snapshot 缺失而重新运行所有官方 Baseline。

---

# 8. 最终报告必须回答的问题

请使用中文，并按照下面顺序报告。

**问题一：为什么 Epoch 1 的 Latent Validation Loss 最低？**

请指出具体是哪几个 Loss 分量导致 Epoch 1–6 的变化；判断是否存在稳定的训练与验证分离。

**问题二：不同 Epoch 的 Latent State 是否发生显著变化？**

给出尺度、有效秩、CKA 和对齐结果；不得将状态变化直接视为信息丢失。

**问题三：哪个 Epoch 最适合最终 Forecasting？**

展示 NP/PJM 的完整表格：

| Dataset | Stage1 Epoch | Latent Validation Loss | Decoder Best Epoch | Forecast MAE | Forecast MSE |
|---|---|---|---|---|---|

**问题四：Latent Loss 最优是否等于最终 Forecasting 最优？**

请分别给出 NP 和 PJM 的答案。

**问题五：我们发现了什么真正值得研究的失败机制？**

只能从已经获得的证据中得出，不允许根据预期的论文故事倒推机制。

**问题六：下一轮是否应该继续？**

明确：

- GO：State Selection。
- GO：Generalization / Optimization Audit。
- NO-GO。
- INCONCLUSIVE。

如果发现 Latent 与 Forecast 最优 Epoch 不一致，还需要提出下一轮用什么独立数据和干预排除简单解释。

如果 Epoch 1 两方面均最好，则重点转向为什么继续训练会损害泛化，而不是强行研究 Decoder Compatibility。

**最终要求：请直接完成代码实现和实际实验，不要只给出方案。结果不必为正，但必须完整、真实、可复现。**

本次实验成功的标准是可靠地判断研究方向，而不是提高 WorldTS 的 MSE 排名。