# Why Does WorldTS Stage 1 Peak at Epoch 1?

实验：`worldts_stage1_state_selection_audit_seed2021_v1`。官方 WorldTS 源码固定在 `269426ea6217d7152a244da0e209d922acd46d4f`。NP/PJM 均为 T=720、H=360、seed 2021。本轮没有训练新结构、没有使用 Test 预测、没有根据结果追加训练或调参。

## 问题一：为什么 Epoch 1 的 Latent Validation Loss 最低？

第 1 轮在完整固定验证窗口、五个预先固定的分组和去除 batch regularizer 的目标下均最低。同轮快照 `eval()` 的 Train Probe 与 Validation 差距随轮数增大：
- NP：固定 Validation 6.83015 → 8.70218；匹配的 Train–Validation 差距 2.01326 → 5.62520。增量中加权轨迹余弦 1.84353，加权 L1 0.02762，regularizer 0.00087。
- PJM：固定 Validation 4.44739 → 7.71698；匹配的 Train–Validation 差距 0.13010 → 5.10038。增量中加权轨迹余弦 2.78453，加权 L1 0.48542，regularizer -0.00036。

在线训练损失逐 batch 测得，不能与轮末快照的 Validation 损失直接相减。固定 Train Probe 的轮末重评排除了这一比较口径错误。分组可改变绝对值，尤其最后 19 窗口在官方无权 batch 均值中仍占一个完整 batch 的权重，但不改变最优轮次。此实验显示目标函数的训练与验证分离，不能单凭损失上升认定预测信息丢失。

## 问题二：不同 Epoch 的 Latent State 是否发生显著变化？

| Dataset | State | Epoch 1/6 std | Epoch 1/6 effective rank | Epoch 6 vs 1 centered CKA | Train-fit Procrustes Val MSE / Epoch-1 variance |
|---|---|---:|---:|---:|---:|
| NP | h_hist | 0.519 / 0.517 | 12.41 / 25.19 | 0.576 | 0.558 |
| NP | h_target | 0.620 / 0.667 | 12.53 / 24.26 | 0.583 | 0.580 |
| NP | h_pred | 0.425 / 0.382 | 17.88 / 30.39 | 0.561 | 0.739 |
| PJM | h_hist | 0.486 / 0.432 | 20.30 / 59.24 | 0.740 | 0.444 |
| PJM | h_target | 0.567 / 0.587 | 22.13 / 58.63 | 0.675 | 0.556 |
| PJM | h_pred | 0.435 / 0.372 | 19.97 / 63.38 | 0.665 | 0.679 |

三个状态的几何均改变；有效秩普遍上升。CKA 量化结构相似性，不能等同于预测信息。Procrustes 只用 16 个预定训练窗口拟合旋转，在验证窗口上评估，且没有进入部署预测。因此不把几何变化直接称为 collapse、信息丢失或已确认的接口漂移。

## 问题三：哪个 Epoch 最适合最终 Forecasting？

下表 Decoder checkpoint 只由 `V_select` 选择。Forecast MAE/MSE 均来自所有六次训练及选择锁定后才计算的 `V_diagnostic`，不能把它再用于挑选部署模型。每组 Decoder 都训练完整 20 轮 (22,240 steps)，下表报告固定预算内的 best checkpoint。

| Dataset | Stage1 Epoch | Latent Validation Loss | Decoder Best Epoch | Forecast MAE | Forecast MSE | V_select weighted MAE |
|---|---:|---:|---:|---:|---:|---:|
| NP | 1 | 6.83015 | 2 | 0.22356 | 0.09898 | 0.18997 |
| NP | 3 | 8.18214 | 17 | 0.25881 | 0.11960 | 0.17187 |
| NP | 6 | 8.70218 | 5 | 0.23916 | 0.10662 | 0.19695 |
| PJM | 1 | 4.44739 | 19 | 0.18441 | 0.12054 | 0.14894 |
| PJM | 3 | 6.08781 | 17 | 0.18294 | 0.08231 | 0.15923 |
| PJM | 6 | 7.71698 | 17 | 0.20157 | 0.11945 | 0.16039 |

Checkpoint 选择及跨状态的选择集排序均按预注册的无权 batch 均值 MAE；表中 MAE/MSE 按全部预测元素等权计算，两个统计口径均保留。同一任务三组的 Decoder 初始 tensor hash 与每轮 Train origins 顺序 hash 一致，Encoder/Condition Encoder/Predictor 全程冻结。各组最后一轮及相同步数的曲线也保留在 `decoder_training_curves.csv`；最优点不是以不同训练机会取得。

## 问题四：Latent Loss 最优是否等于最终 Forecasting 最优？

- NP：latent 最优 E1；`V_select` 预测最优 E3，隔离诊断集上的描述性最优 E1。选择集 E3 相对 E1 的 MAE 改善 9.27%；同在 Decoder 第 20 轮的改善 9.48%；其诊断集相对 E1 的改善 -15.77%。
- PJM：latent 与 `V_select` 的预测最优均为 E1；诊断集描述性最优为 E3。E3 相对 E1 在选择集的改善 -7.21%，在诊断集的改善 0.80%。
- NP 的选择集与隔离诊断集最优 Stage-1 轮次发生反转：E3 相对 E1 的诊断 MAE 差值 0.03525，0/4 个不重叠 360-origin 区块支持 E3；360-origin block bootstrap 描述区间 [0.01623, 0.05658]。
- PJM 的选择集与隔离诊断集最优 Stage-1 轮次发生反转：E3 相对 E1 的诊断 MAE 差值 -0.00147，2/4 个不重叠 360-origin 区块支持 E3；360-origin block bootstrap 描述区间 [-0.01399, 0.01209]。

诊断集只用于事后描述，且历史研究已经接触过完整旧验证区间，故不是独立新测试集。其 1,363 个滚动窗口高度重叠；连续区块 bootstrap 仅是敏感性描述，不作独立样本显著性检验。

| Dataset | Candidate vs E1 | Block length | ΔMAE candidate−E1 | 95% descriptive interval | Better resample fraction |
|---|---|---:|---:|---:|---:|
| NP | E3 | 360 | 0.03525 | [0.01623, 0.05658] | 0.00% |
| NP | E3 | 720 | 0.03525 | [0.01802, 0.05300] | 0.00% |
| NP | E3 | 1080 | 0.03525 | [0.02100, 0.04940] | 0.00% |
| NP | E6 | 360 | 0.01560 | [-0.01365, 0.05338] | 21.95% |
| NP | E6 | 720 | 0.01560 | [-0.01140, 0.04169] | 17.10% |
| NP | E6 | 1080 | 0.01560 | [-0.01021, 0.04178] | 11.10% |
| PJM | E3 | 360 | -0.00147 | [-0.01399, 0.01209] | 58.35% |
| PJM | E3 | 720 | -0.00147 | [-0.01258, 0.01026] | 59.15% |
| PJM | E3 | 1080 | -0.00147 | [-0.01002, 0.00689] | 62.95% |
| PJM | E6 | 360 | 0.01717 | [0.00516, 0.02939] | 0.15% |
| PJM | E6 | 720 | 0.01717 | [0.00740, 0.02702] | 0.00% |
| PJM | E6 | 1080 | 0.01717 | [0.00884, 0.02534] | 0.00% |

在固定选择集上，NP 的下游最优状态与 E1 潜变量最优状态不一致；这是观察到的模型选择失配，但仍需看时间区块和另一任务是否支持可泛化机制。

## 问题五：真正值得研究的失败机制是什么？

严格可归因的事实是：Stage-1 轮末验证目标在两任务上均于 E1 最低，而其主要上升分量是轨迹余弦；与此同时 Encoder/target 与 predicted latent 坐标和有效秩持续变化。下游 Decoder 对比检验这种选择准则是否遗漏预测有用的状态。更直接的实证失败是时间切片上的轮次排序不稳定：NP 在选择集偏向 E3，在原始支持不重叠的诊断切片却偏向 E1；PJM 的排序方向也不同，且诊断 E3 优势很小。目前的消融设计不能区分“移动目标”“优化路径”“模型选择目标错配”各自的因果贡献，也不能推出某种新损失或模块有效。先前 Predictor-only 审计对复杂 readout-aware loss 已给 NO-GO，future-response Loop 对比为 INCONCLUSIVE；本实验不推翻那些结论，只定位 Stage-1 选择问题。

## 问题六：研究决策

**NO_GO_NEW_STATE_SELECTION_METHOD**。NP 的较晚状态只在选择集上明显更好，在时间隔离诊断集上方向反转；另一任务没有同向的选择集收益。未满足预设跨任务和区块稳定性门槛，不据此开发新状态选择方法。

这是 seed 2021、NP/PJM 两个任务内的机制可行性判断。若出现值得继续的状态选择信号，下一步必须在此前未用于方法选择的新数据集和多个种子上预注册确认；若没有一致信号，不增添 Jacobian、adapter、loop 或新损失。

## 可复核执行记录

- 自动检查：66 PASS、0 FAIL；12 个真实 raw Stage-1 checkpoint、126 个逐轮 Decoder checkpoint 均已记录 SHA-256。
- 六组 Adam/type-3/batch32/20 epoch；三组 NP LR=1e−4，三组 PJM LR=2e−4。官方 Decoder mix objective 为预测 MAE + 0.3×target-state 重构 MAE。
- `V_select` 原点 36,690–38,051；`V_diagnostic` 原点 40,210–41,572；两组原始时间支持中间有 1,079 小时空档。
- 服务器运行环境：NVIDIA GeForce RTX 5090，PyTorch 2.8.0+cu128，CUDA 12.8；原始基线输入的 16 个 SHA-256 未改变，官方源码 312 个文件 SHA-256 重验通过。所有大权重与诊断预测阵列留在服务器；Git 保存源码、配置、日志摘要、报告、原始 CSV 和完整路径哈希。

相关原始数据：`stage1_epoch_metrics.csv`、`batch_partition_sensitivity.csv`、`latent_geometry.csv`、`latent_geometry_alignment.csv`、`decoder_training_curves.csv`、`decoder_training_results.csv`、`paired_block_statistics.csv`。七类图在 `outputs/figures`，原始数值以 CSV 为准。
