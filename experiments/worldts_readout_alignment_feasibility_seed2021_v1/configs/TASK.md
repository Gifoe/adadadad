# Codex Research Task: WorldTS Readout–Latent Alignment Feasibility Audit

## 0. 研究背景与目标

请在现有仓库中完成一项独立研究实验：

**Experiment:** `worldts_readout_alignment_feasibility_seed2021_v1`

**Repository:** https://github.com/Gifoe/adadadad

**WorldTS Official Code:** https://github.com/decisionintelligence/WorldTS

**Baseline Report:**
`experiments/baseline_reproduction_seed2021_v1/FINAL_REPORT.md`

**Previous Negative Evidence:** 请检索仓库中的 `conditional_latent_loop_seed2021_v1` 与 `worldts_future_response_loop` 等实验，读取对应的最终报告和失败结论。

我们的前期实验已经发现：

1. LatentTSF-inspired 模型可以获得很低的未来状态重构误差，但这不保证优秀的最终预测表现。
2. WorldTS 的四个任务中，Stage 1 最佳验证 Latent Loss 均出现在 Epoch 1。
3. 对 WorldTS 继续优化和加入多种 Loop 后，并没有在 NP/PJM 上获得稳定且具有独立价值的性能提升。
4. 原版 WorldTS Stage 1 主要训练 Encoder 和 State Predictor，Stage 2 冻结这些模块并训练 Decoder；最终 Decoder 使用预测的未来状态进行预测。

由此产生新的研究问题：

> Does the latent-space learning objective in WorldTS provide an optimization signal that is aligned with downstream forecasting performance?

更具体地：

**RQ1:** Latent Loss 与最终 Forecasting Loss 的梯度是否存在系统性不一致？

**RQ2:** 如果存在不一致，它是否构成实际可利用的性能瓶颈？

**RQ3:** 这种瓶颈能否通过简单的 decoded supervision 修正，还是有必要研究更复杂的 readout-aware representation learning？

请不要预设上述假设成立。

本实验的目标不是发表一个新的 Loss，而是决定是否值得继续研究 Readout-Aligned Predictive States。

---

## 1. 执行原则

### 1.1 必须沿用已经复现成功的官方 WorldTS

我们的官方复现已经完成，使用以下配置：

| Dataset | T | H | Seed | Original Test MSE |
|---|---:|---:|---:|---:|
| NP | 720 | 360 | 2021 | 0.430395 |
| PJM | 720 | 360 | 2021 | 0.115082 |
| NP | 168 | 24 | 2021 | 0.167258 |
| PJM | 168 | 24 | 2021 | 0.057516 |

这些数值仅用于确认历史复现记录，不允许作为本轮新方法的超参数选择依据。

优先使用 NP-H360、PJM-H360 进行完整审计与训练干预。

H24 仅作为可选的辅助诊断，不增加第一轮训练数量。

此前训练的原始数据、模型权重和预测结果保存在服务器：

`D:\baseline_reproduction_seed2021_v1`

请先检查实际目录、文件、Checkpoint、Config、Scaler、数据划分和代码版本。

不得凭猜测构造文件路径。

如果路径不一致，应在仓库中检索原始报告、Artifact Manifest、REPRODUCE.md 和实际目录，找到正确文件后继续。

原始复现使用 Windows、RTX 5090 32GB、PyTorch CUDA 环境。应复用已经验证可用的运行环境，避免重新安装大型依赖或破坏原有实验。

### 1.2 严格禁止

- 不重新训练 WorldTS Baseline，除非现有 Checkpoint 确认不可用，且需要在报告中说明。
- 不修改原始 NP/PJM 数据划分。
- 不修改未来协变量使用方式。
- 不使用 Test Set 选择超参数。
- 不修改 WorldTS Backbone、Patch Encoder、Decoder 或 Predictor 结构。
- 不加入 Loop、Attention Adapter、Jacobian-aware Loss 或新模型。
- 不为了获得正结果更换 Seed、筛选测试窗口或隐藏失败实验。
- 不覆盖之前的实验目录和权重。
- 不将结果没有支持的解释写成已证实结论。

所有实验代码和结果保存在新建目录：

`experiments/worldts_readout_alignment_feasibility_seed2021_v1/`

---

## 2. Phase 0：Baseline 与 Checkpoint 预检

首先确认我们复用的是完整的 Stage 2 训练完成的 WorldTS 模型，而不只是 Stage 1 Encoder/Predictor Checkpoint。

原因是本次审计需要使用已经学习了预测任务的 Decoder。

请检查官方实现：

- `WorldTSModel.forward`
- `WorldTSModel.decode`
- `WorldTSModel.forecast`
- `WorldTS._decode_model_states`
- `WorldTS._process`
- `LatentStateLoss.forward`
- `configure_for_stage1`
- `configure_for_stage2`

尤其需要确认：

1. `h_hist`：历史序列的 latent state。
2. `h_target`：真实未来序列编码得到的 latent target。
3. `h_pred`：模型根据历史与允许的外部条件预测的未来 latent state。
4. `target_revin_stats`：原版 RevIN 使用的统计量。
5. Causal Patch Decoder 使用 `h_hist` 和 `h_pred` 的正确方式。

建立完整的 Checkpoint 加载与验证流程。

必须验证：

- 原版 Stage 2 Checkpoint 能够被正确加载。
- 原来的网络输出与复现记录一致，误差处于合理浮点容差内。
- 审计程序与官方预测程序使用完全相同的归一化和 Decoder 路径。
- 未来真实值只用于产生监督标签和审计目标，不得进入实际部署时的 `h_pred` 预测路径。
- 扰动未来标签不会改变由相同历史和合法未来协变量得到的预测状态。
- 保存加载前后的参数 Hash，以及全部模块的参数统计。

输出：

`outputs/PREFLIGHT_REPORT.md`

`outputs/preflight_metrics.json`

如果无法保证与官方预测一致，必须先修复，而不是继续进行梯度分析。

---

## 3. Phase 1：Latent–Readout Alignment Audit

本阶段不训练模型。

所有参数冻结，但保留对预测状态 `h_pred` 的自动微分。

审计时使用确定性的 `eval()` 模式，不得使用 `torch.no_grad()` 禁用所需的梯度。

### 3.1 提取相同样本的预测状态与目标状态

对于每个输入窗口：

\[
h^*=E(y_{\mathrm{future}})
\]

\[
\hat h=P(x_{\mathrm{history}},c)
\]

使用同一个训练完成的 Decoder，计算：

\[
\hat y=D(\hat h,h_{\mathrm{hist}})
\]

\[
y_{\mathrm{target-decoded}}
=D(h^*,h_{\mathrm{hist}})
\]

注意：

- `h_target` 必须使用原始 WorldTS 的 future-target encoding 方法。
- 必须保留历史前缀、Patch Padding、RevIN、外生协变量和 Decoder 的全部原始约束。
- 所有误差应明确标注是在归一化空间还是原始观测空间计算。
- 不允许将 Target Decoder Reconstruction Error 当作整个模型的预测误差下界。

记录：

**A. Latent Error**

\[
L_{\mathrm{latent}}=
\mathcal L_{\mathrm{WorldTS}}(\hat h,h^*)
\]

并单独记录 L1、Cosine、Regularization 各分量。

**B. Forecast Error**

\[
L_{\mathrm{forecast}}=
\mathcal L(\hat y,y_{\mathrm{future}})
\]

记录 MSE、MAE。

**C. Target Reconstruction Error**

\[
L_{\mathrm{reconstruction}}
=
\mathcal L(y_{\mathrm{target-decoded}},y_{\mathrm{future}})
\]

**D. Decoded Latent Discrepancy**

\[
L_{\mathrm{decoded-gap}}
=
\mathcal L(D(\hat h),D(h^*))
\]

其中两个 Decoder 调用必须使用相同的历史条件和归一化统计。

分别给出每个 Dataset、Horizon 和时间区间的结果。

### 3.2 核心实验：Latent 与 Forecast 梯度一致性

对于同一个 `h_pred`，分别计算：

\[
g_L=
\nabla_{\hat h}L_{\mathrm{latent}}
\]

\[
g_Y=
\nabla_{\hat h}L_{\mathrm{forecast}}
\]

计算：

\[
A=
\frac{\langle g_L,g_Y\rangle}
{\|g_L\|\|g_Y\|+\epsilon}
\]

这是本实验最重要的诊断。

请报告：

- Mean Gradient Cosine。
- Median Gradient Cosine。
- 10%、25%、75%、90% 分位数。
- Cosine 小于 0 的比例。
- Cosine 接近 0 的比例。
- 两种梯度的范数分布。
- 按预测误差分位组分层后的 Gradient Cosine。
- 按预测区间位置分层的 Gradient Cosine。

对 Latent L1、Cosine 和完整 WorldTS Latent Objective 分别进行分析。

注意：VICReg/variance-covariance regularizer 可能涉及 Batch 间耦合。对完整 Loss 的样本分组只能解释为 Batch Objective 对各样本状态的梯度分量，不得错误声称它们是相互独立的样本 Loss。

如果出现零梯度或数值不稳定，请单独计数，不要将它们强制设置为 Cosine=0 后参与均值。

### 3.3 进一步检查 Predictor Parameter-space Alignment

Hidden-state Gradient Conflict 并不一定意味着真实参数更新冲突。

因此额外检查：

\[
g_L^\theta=
\nabla_{\theta_P}L_{\mathrm{latent}}
\]

\[
g_Y^\theta=
\nabla_{\theta_P}L_{\mathrm{forecast}}
\]

其中 \(\theta_P\) 为 State Predictor 参数。

至少在固定抽样的验证 Mini-batches 上，比较：

- Predictor 参数空间的梯度余弦。
- Predictor 的梯度范数。
- 参数空间冲突与状态空间冲突是否一致。

必须使用相同的模型状态和样本批次。

不要把同一次 Backward 的多个 Loss 梯度意外累加在一起。

这一诊断用于回答：

**Latent–Forecast 不一致是否真的可能影响 Predictor 的优化方向？**

### 3.4 统计与可视化

输出：

1. Latent Error vs Forecast Error 散点图。
2. Gradient Cosine 分布图。
3. Gradient Cosine vs Forecast Error。
4. Gradient Norm 分布。
5. NP/PJM 两数据集的对比图。
6. 按时间区块的误差和梯度一致性变化。

报告 Pearson 与 Spearman 相关系数，并使用按连续时间区块聚合的统计方式，避免将高度重叠的滑动窗口误当作独立样本。

本阶段不以发现低 Cosine 为成功标准。

需要讨论：

- 是否存在真正的梯度冲突？
- 是否集中于高预测误差样本？
- 是否存在明显的数据集依赖？
- 是否可能只是损失尺度、RevIN 或 Decoder 非线性引起的表面现象？

输出：

`outputs/ALIGNMENT_AUDIT.md`

`outputs/gradient_alignment.csv`

`outputs/error_alignment.csv`

`outputs/figures/`

---

## 4. Phase 2：最小训练干预实验

只做三组对照。

不设计 Jacobian Loss。

### 4.1 固定训练起点

从 Phase 0 加载的同一个官方 Stage 2 WorldTS Checkpoint 初始化。

冻结：

- Target Encoder
- Condition Encoder
- Modal/Time Encoder（如存在）
- State Decoder
- 其他所有非 Predictor 模块

只允许：

`model.predictor`

参与训练。

需要明确区分 `requires_grad=False` 和 `eval()` 模式，避免冻结模块的 Dropout、Normalization 等状态变化。

对于 Predictor，可以按原训练协议使用训练模式；其余模块必须保持固定状态。

记录 Predictor 可训练参数量。

训练前后检查冻结模块的 Tensor Hash，确保完全没有变化。

### 4.2 三组训练方案

**A0：Latent-only**

使用官方 WorldTS 的 Latent Loss：

\[
L_{A0}=L_{\mathrm{latent}}
\]

包括原版的 L1/Cosine 和适用于 Predictor 的 regularization 分量。

必须解释冻结 Encoder 后，各 Loss 分量究竟是否还能产生 Predictor 梯度。

**A1：Decoded-only**

\[
L_{A1}=
L_{\mathrm{forecast}}
\]

只优化 Decoder 输出的预测误差。

此时 Decoder 参数仍然冻结，但梯度必须能够通过 Decoder 传递到 Predictor。

必须确认：

\[
\|\nabla_{\theta_P}L_{A1}\|>0
\]

不能把所有 Decoder Forward 包在 `torch.no_grad()` 中。

**A2：Latent + Decoded**

\[
L_{A2}
=
L_{\mathrm{latent}}
+
\lambda L_{\mathrm{forecast}}
\]

\(\lambda\) 必须在训练前用训练集上的梯度尺度确定，或使用预先声明的固定值。

建议用固定的训练集 Mini-batches 测量两个 Loss 对 Predictor 参数的初始梯度范数，然后确定一个数量级平衡的权重。

不可使用测试集寻找 \(\lambda\)。

第一轮只采用一个预先确定的权重，不做大规模 Grid Search。

### 4.3 公平比较

固定：

- Seed：2021。
- 训练数据划分。
- 模型初始化权重。
- Optimizer 类型和学习率。
- Batch Size。
- 数据顺序和随机状态。
- 最大 Optimizer Steps。
- Validation 频率。
- Checkpoint Selection。
- Early Stopping 规则。

建议从学习率 `1e-4` 开始进行统一预检；如不能稳定训练，必须在正式对照前统一修订所有方案。

所有方案训练相同的最大 1,000 个 Optimizer Steps，使用相同的验证频率和提前停止规则。

该预算是本次可行性实验预设，而不是原论文训练配置。

可以先执行 20–50 Step Smoke Test，确认三个 Loss 都能正确产生 Predictor 梯度，冻结参数没有改变，损失没有 NaN/Inf。

正式运行时：

| Dataset | A0 | A1 | A2 |
|---|---|---|---|
| NP-H360 | 训练 | 训练 | 训练 |
| PJM-H360 | 训练 | 训练 | 训练 |

总计六次训练。

每次必须包含 Step 0 原始 Checkpoint 的验证指标，并允许其作为最佳 Checkpoint，避免微调后性能退化却被隐瞒。

### 4.4 Validation 与选择

采用与原复现一致的 Validation Forecasting MAE 选择最佳 Checkpoint，并同时报告 MSE。

不能根据 Latent Loss 为 A0 选择 Checkpoint、根据 Forecast Loss 为 A1 选择 Checkpoint。这会产生不公平比较。

每组都以完全相同的最终 Forecasting Validation Criterion 选模。

报告：

- Step 0 Validation MSE/MAE。
- Best Validation MSE/MAE。
- Last-step Validation MSE/MAE。
- 相对 Step 0 的变化。
- A0/A1/A2 之间的差值。
- Training Latent Loss。
- Training Forecast Loss。
- Gradient Cosine 的训练轨迹（适当抽样）。
- 时间与峰值显存。

已经被之前实验查看过的 NP/PJM Test 集不用于本轮新方案的选择，也不将其结果用于宣称新方法的无偏泛化提升。

本轮以机制可行性为目的。若方案通过筛选，再新增未被方法选择使用的数据集进行独立验证。

输出：

`outputs/TRAINING_COMPARISON.md`

`outputs/training_results.csv`

`outputs/training_curves.csv`

`outputs/checkpoints/`

---

## 5. Phase 3：分析结果、判断原因

请不要只汇报某组训练的 MSE 更低。

必须回答以下五个问题。

### Q1：Latent Loss 与 Forecast Loss 是否冲突？

同时检查：

- 隐状态梯度。
- Predictor 参数梯度。
- 梯度冲突是否集中在高误差样本。

不要仅根据平均 Cosine 判断。

### Q2：这种冲突是否具有实际性能后果？

如果存在冲突，A1 或 A2 是否比 A0 改善验证预测？

如果没有改善，就不能因为梯度不一致而宣称找到了性能瓶颈。

### Q3：是否只需要普通任务监督？

如果 A1 或 A2 已经取得全部收益，那么直接 Decoded Loss 很可能已经能够解释改进。

这时不能立即提出 Jacobian-aware 新方法。

### Q4：是否存在明显的 NP/PJM 差异？

需要分别分析：

- 数据分布。
- Latent Error。
- Gradient Cosine。
- 预测误差。
- 训练后的变化。

不得仅根据单个任务上涨宣称机制普适。

### Q5：我们是否真的找到了新方法的空间？

重点判断：

- 是否存在真实的优化目标不一致？
- 不一致是否可以通过 Predictor 训练利用？
- 现有简单方法是否已经充分解决？
- 如果没有，剩余瓶颈更可能属于 Encoder 表征、Decoder 表征、时间分布变化，还是监督机制？

---

## 6. 最终决策：GO / NO-GO / INCONCLUSIVE

最终必须给出明确结论。

**NO-GO 情况：**

1. Latent 与 Forecast Gradient 基本一致，不存在明显的目标不一致证据。
2. 即使有梯度不一致，A1/A2 仍没有获得可重复的验证收益。
3. 所有改善都来自极少数样本，无法跨时间区间成立。
4. 数据流、Checkpoint 或原版 Decoder 路径未能正确验证。

**LIMITED POSITIVE / INCONCLUSIVE：**

1. 出现明显梯度冲突。
2. A1/A2 能提升预测表现。
3. 但提升可以被直接 decoded supervision 完整解释。
4. 或只有一个任务改善，另一个任务明显退化。

这只能说明任务监督值得继续研究，不足以证明我们需要新型 Readout-aware Loss。

**GO TO NEXT MECHANISM STUDY：**

需要同时满足：

1. 梯度不一致具有稳定、可量化的证据。
2. Predictor 参数空间的诊断支持这种不一致可能影响训练。
3. 改变监督能够在至少两个任务上获得一致的、有实际意义的验证收益，或提供明确的任务差异解释。
4. 训练过程和数据协议不存在泄漏或不公平因素。
5. 能指出简单 decoded supervision 尚未解决的具体问题。

即使 GO，也不代表已经有足够的新方法创新。

下一轮才允许研究：

- Readout-aware latent geometry。
- Decoder Jacobian supervision。
- Task-sensitive state representation。
- 更稳健的联合表示学习。

本轮不要实现这些方法。

---

## 7. 结果目录与交付物

请使用以下结构：

```text
experiments/
└── worldts_readout_alignment_feasibility_seed2021_v1/
    ├── README.md
    ├── configs/
    ├── scripts/
    ├── src/
    ├── tests/
    ├── outputs/
    │   ├── PREFLIGHT_REPORT.md
    │   ├── ALIGNMENT_AUDIT.md
    │   ├── TRAINING_COMPARISON.md
    │   ├── FINAL_RESEARCH_REPORT.md
    │   ├── NEXT_RESEARCH_DECISION.md
    │   ├── preflight_metrics.json
    │   ├── gradient_alignment.csv
    │   ├── error_alignment.csv
    │   ├── training_results.csv
    │   ├── training_curves.csv
    │   ├── figures/
    │   └── checkpoints/
    └── REPRODUCE.md
```

如有必要增加文件可以，但不得遗漏核心报告。

`REPRODUCE.md` 必须提供 Windows 环境中真实验证过的运行命令，说明所需输入文件、环境、相对/绝对路径及执行顺序。

增加自动化测试，至少涵盖：

1. Predictor-only 训练时，冻结参数完全不变。
2. A1/A2 能正确向 Predictor 反传梯度。
3. 原版 Prediction Path 不访问未来真实目标。
4. Forward 输出与原始 WorldTS 一致。
5. 梯度余弦实现与简单人工测试一致。
6. 原始 Stage 2 Checkpoint 可以无损恢复。
7. 原版 RevIN 和 Forecast Metric 计算一致。
8. 验证阶段不允许因 `torch.no_grad()` 导致梯度审计失效。

确保失败的 Smoke Test 不会被记录为正式实验结果。

---

## 8. 资源与执行要求

服务器为单张 RTX 5090 32GB，Windows 环境。

优先复用原有 WorldTS 依赖及 Checkpoint。

不要重新下载已经存在的数据或模型。

按以下顺序执行：

1. 完成环境、数据与 Checkpoint 预检。
2. 完成 NP/PJM 的梯度一致性审计。
3. 分析审计结果。
4. 完成六次小型 Predictor-only 对照训练。
5. 整理所有统计结果。
6. 撰写最终研究报告与下一轮建议。

每阶段如果遇到工程问题，应定位并修复，不要静默跳过。

如果存在无法修复的阻碍，应保留失败证据并说明哪些实验没有完成。

---

## 9. 最终回答格式

完成后，请首先用中文回答：

**1. 我们的假设是否成立？**

Latent Objective 与 Forecast Objective 是否存在明显的训练目标不一致？

**2. 这种不一致是否真的影响预测性能？**

给出 NP/PJM 的实际数值、梯度分析和训练结果。

**3. 简单的 Decoded Supervision 是否已经足够？**

明确比较 A0、A1、A2。

**4. Idea A 是否值得继续？**

给出 GO / NO-GO / INCONCLUSIVE，并说明依据。

**5. 下一轮究竟应该解决什么具体问题？**

如果发现新的可行动瓶颈，用一句话说明：

> 我们要解决的是______问题；当前 WorldTS 在______情况下失败；现有简单方法无法解决是因为______；下一轮需要验证______。

如果实验没有支持任何明确的新机制，请明确写出 NO-GO，不要为了继续项目而虚构改进方向。

最后提供完整文件路径、Git Commit、实际运行命令、实验耗时与显存，以及最终研究报告的位置。

**本次任务的成功标准是给出一个可靠的研究决策，而不是必须让模型涨点。请直接执行实验，不要只给出设计方案。**