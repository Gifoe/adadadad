# WorldTS Readout–Latent Alignment Feasibility Audit

Final decision: **NO-GO for a new complex readout-aware loss**. Ordinary decoded supervision: **INCONCLUSIVE**.

Q1 — Objective conflict: NP state cosine=0.0808, negative=0.0%; Predictor cosine=0.5784, negative=0.0%; PJM state cosine=0.0718, negative=0.0%; Predictor cosine=0.2542, negative=0.0%. Interpret together with quartiles, separate L1/cosine/regularization, and eight-batch parameter sampling. Near-zero cosine is not automatically conflict. Parameter gradients diagnose optimizer directions, while state gradients alone cannot. Positive loss scaling cannot explain cosine signs; history RevIN and Decoder geometry remain plausible contributors rather than proven causes.

Q2 — Performance consequences: NP A1 vs A0 full validation MAE gain=-3.079%, 360-hour descriptive95% interval=[-6.103,-0.249], positive blocks=5/14; NP A2 vs A0 full validation MAE gain=-1.727%, 360-hour descriptive95% interval=[-3.512,0.050], positive blocks=5/14; PJM A1 vs A0 full validation MAE gain=1.456%, 360-hour descriptive95% interval=[-1.032,3.997], positive blocks=9/14; PJM A2 vs A0 full validation MAE gain=1.278%, 360-hour descriptive95% interval=[-0.394,3.107], positive blocks=9/14. Also compare every method with Step0 above: beating deteriorated A0 alone does not establish improvement over the original model. This single-seed validation intervention supports only local feasibility; conflict and performance association do not prove a unique causal mechanism.

Q3 — Ordinary task supervision: A1 is ordinary MAE through a frozen Decoder; A2 adds only a train-calibrated copy of the same MAE. Thus any observed improvement is already obtained with simple supervision. A2 exceeding A1 would demonstrate a useful tradeoff in this budget, not a need for a new readout-aware loss. No Jacobian, adapters, loop or architecture changes were used.

Q4 — NP/PJM difference: the initial latent/decoded/forecast errors and gradient distributions differ as shown in the audit table; paired gains, error quartiles and all time blocks are reported separately. Neither one dataset nor their arithmetic mean establishes a universal mechanism. Decoder-conditioned target reconstruction can remain small while forecasting remains difficult, and it is not a forecasting lower bound. A causal explanation involving distribution shift, Encoder or Decoder requires another intervention.

Q5 — Novel mechanism space / Idea A: NO-GO for a new complex readout-aware loss. Ordinary supervision status=INCONCLUSIVE. The experiment has not isolated a concrete failure that simple decoded supervision cannot address. The next defensible action, if ordinary-supervision gains warrant follow-up, is a preregistered independent dataset and additional seeds with A0/A1/A2; retain time-block and Step0 controls. Do not invent a Jacobian mechanism from low cosine or residual forecast error.

## Initial diagnostics

| dataset | forecast_MAE | forecast_MSE | latent_L1 | latent_cosine_loss | target_reconstruction_MAE | decoded_gap_MAE | state_cosine | state_negative_fraction | parameter_cosine | parameter_negative_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 0.203775 | 0.0876107 | 0.252612 | 0.280604 | 0.00508913 | 0.202878 | 0.0807732 | 0 | 0.578366 | 0 |
| PJM | 0.166042 | 0.07058 | 0.198312 | 0.159674 | 0.0108706 | 0.168208 | 0.0717869 | 0 | 0.254162 | 0 |

## All six formal training outcomes

| dataset | method | seed | steps | best_step | step0_selection_MAE | best_selection_MAE | last_selection_MAE | step0_MAE | step0_MSE | best_MAE | best_MSE | last_MAE | last_MSE | MAE_gain_vs_step0_percent | MSE_gain_vs_step0_percent | lambda | seconds_training_and_periodic_validation | peak_allocated_GPU_MiB | predictor_parameters | frozen_unchanged | fresh_prediction_max_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | A0 | 2021 | 400 | 150 | 0.204109 | 0.195545 | 0.202274 | 0.203775 | 0.0876107 | 0.195255 | 0.0790825 | 0.201922 | 0.082472 | 4.18091 | 9.73415 | 12.0313 | 20.3794 | 1426.17 | 827944 | True | 0 |
| NP | A1 | 2021 | 350 | 100 | 0.204109 | 0.201566 | 0.212555 | 0.203775 | 0.0876107 | 0.201268 | 0.084493 | 0.212238 | 0.0904043 | 1.2302 | 3.55859 | 12.0313 | 17.7034 | 1427.25 | 827944 | True | 0 |
| NP | A2 | 2021 | 350 | 100 | 0.204109 | 0.198927 | 0.208678 | 0.203775 | 0.0876107 | 0.198626 | 0.0825864 | 0.208346 | 0.0873216 | 2.52654 | 5.73476 | 12.0313 | 18.0989 | 1427.1 | 827944 | True | 0 |
| PJM | A0 | 2021 | 500 | 250 | 0.166145 | 0.165922 | 0.169516 | 0.166042 | 0.07058 | 0.165931 | 0.0666824 | 0.169541 | 0.0668318 | 0.0672443 | 5.52222 | 5.716 | 45.9964 | 5283.69 | 1891176 | True | 0 |
| PJM | A1 | 2021 | 400 | 150 | 0.166145 | 0.163538 | 0.164207 | 0.166042 | 0.07058 | 0.163515 | 0.0653871 | 0.164201 | 0.0658464 | 1.52212 | 7.35749 | 5.716 | 36.9818 | 5283.13 | 1891176 | True | 0 |
| PJM | A2 | 2021 | 500 | 250 | 0.166145 | 0.163826 | 0.166295 | 0.166042 | 0.07058 | 0.163809 | 0.066523 | 0.166303 | 0.0655869 | 1.34474 | 5.74814 | 5.716 | 46.5562 | 5281.75 | 1891176 | True | 0 |

## Protocol, validity and limitations

All six fits start from the same task-specific official Stage2 checkpoint. Only Predictor parameters train; every other parameter/buffer is frozen and every other module remains eval. Adam lr=1e-4, batch32, identical seed/order/initial CUDA RNG, max1000 steps, validation every50, patience5 checks. Step0 eligible. Selection is the official unweighted mean of batch MAEs on a fixed chronological validation partition; element-weighted full MAE/MSE are reported separately. Original validator shuffles validation and gives the short tail batch equal weight, so its historical best value is not exactly reproducible without the missing shuffle order. Current fixed partitions are verified against the official validator. This canonicalization is applied to all methods before comparison; no metric or tolerance was relaxed. Phase0 independent test reproduction uses original batch64 and gives max prediction difference0 for all four checkpoints.

A0=official full latent objective; A1=decoded forecast MAE; A2=full latent + lambda*forecast MAE. Lambda is fixed before fits from the median Predictor gradient-norm ratio over16 TRAIN-only batches: {'NP': 12.031270805052078, 'PJM': 5.715996374349599}. No validation/test tuning. Target-Encoder regularizer is constant; predicted-state regularizer and L1/cosine retain Predictor gradients. Frozen Decoder autograd is enabled. Six separate20-step smoke checks passed and are not formal results.

Reported seconds include optimization and periodic validation; command wall times additionally include startup, Step0 validation, saving and fresh reload validation. Peak GPU is process-owned torch.cuda.max_memory_allocated MiB, not total device occupancy. All frozen tensor hashes, shared data-order prefixes, initialization/RNG hashes and fresh-reload prediction checks pass. No candidate test forecasts were generated. Different actual step counts follow the common early-stop rule.


Fixed single seed2021, NP/PJM H360 only for candidate training; H24 original-checkpoint preflight only. Maximum1000 steps is this audit budget, not the paper training protocol. Validation used for audit and selection; no unbiased candidate test/generalization claim. Historical validation-shuffle order is unavailable and disclosed; exact current official validation and original test prediction routes are verified. Best Step0 results are retained when fine-tuning hurts. No failures or negative comparisons are hidden. Preflight attempts1–4 remain in logs (test-batch mismatch and historical randomized validation mean); attempt5 passes without weakening exact-path checks. Prior conditional-latent and future-response-loop negative reports are copied as evidence in configs, and do not become new positive evidence here.

See ALIGNMENT_AUDIT.md, TRAINING_COMPARISON.md, research_decision.json, block_correlations.csv, paired_block_bootstrap.csv, improvement_concentration.csv and figures01–09. REPRODUCE.md provides actual Windows commands and pinned inputs. Final artifact manifest records file hashes and server-only checkpoint paths. Git publication provenance is recorded after export.

## Evidence-based answers (Chinese)

**1. 是否存在 latent/readout 不一致？** 有弱状态方向对齐，但未发现系统性反向冲突。NP/PJM 全预测窗口的完整目标状态梯度余弦均值为0.08077/0.07179，负余弦比例均为0；固定8个 batch 的 Predictor 参数梯度余弦为0.57837/0.25416，负比例也均为0。NP 最后120步切片出现2/4,883个负余弦（0.041%），其余完整目标预测段均为0。高误差四分位同样没有负余弦；NP 高误差组余弦反而更高（0.1033 vs低误差0.0698），PJM 高/低误差组为0.0682/0.0697。所有六个最佳 checkpoint 在相同8个 batch 上的参数余弦仍全部为正。不能把接近正交的状态梯度写成真实参数更新冲突。 训练模式下按步数稀疏抽样的106次参数诊断中有4次负余弦（NP A0为1/17，PJM A0为1/21、A2为2/21，其余0）。这说明部分更新存在短暂冲突；它既不是全训练batch的普查比例，也没有改变固定验证审计与跨任务干预的否定证据。原始轨迹及汇总均保留。

**2. 是否是可利用的 Predictor 瓶颈？** 没有跨任务的证据。相对A0，A1/A2在NP上MAE分别恶化3.079%/1.727%，在PJM上改善1.456%/1.278%。NP只有5/14个360小时区块改善，PJM为9/14；PJM两种改进的360/720/1680小时区块bootstrap区间均包含0。A1在NP的360小时描述性95%区间为[-6.103%,-0.249%]，PJM为[-1.032%,3.997%]。区间针对已选模验证集，不能作为无偏泛化检验。继续训练本身使NP的A0相对Step0改善4.181%，超过A1的1.230%和A2的2.527%；因此不能将任何Step0改善归功于decoded监督。

**3. 简单Decoded Supervision是否已经足够？** PJM的全部已观察小幅收益由普通decoded MAE（A1）取得；A2未超过A1，增加latent项并未提供额外收益。NP则A0最好。当前既没有必要性证据支持新Jacobian/readout-aware loss，也没有稳定跨任务收益支持直接替换原目标。

**4. Idea A是否值得继续？** 对新型复杂readout-aware loss给出NO-GO。普通任务监督单独为INCONCLUSIVE：PJM约1.5%局部改善，但NP约3.1%相对退化，单seed且区块不确定性未排除0。初始latent L1为NP0.2526/PJM0.1983，预测MAE为0.20377/0.16604，参数对齐也明显不同，说明不能宣称机制普适。

**5. 下一轮解决什么？** 本轮未找到简单监督无法解决的明确新机制瓶颈；不启动Jacobian loss或新表示机制。若继续验证普通任务监督，唯一具体问题是“PJM的小幅收益能否在未参与选择的数据集和额外预设seed上重复，同时解释NP为何A0更好”。冻结Encoder/Decoder的本轮实验无法因果区分Encoder表示不足、Decoder几何和时间分布变化，不能据此编造其中任何一个失败机制。

## Measured resources

{
  "formal_command_wall_seconds": 246.79332733154297,
  "training_and_periodic_validation_seconds": 185.71622157096863,
  "pipeline_wall_seconds": 291.58900904655457,
  "successful_preflight_seconds": 27.72576928138733,
  "initial_gradient_audit_seconds": 28.707732915878296,
  "peak_process_owned_allocated_gpu_mib": 5283.69384765625,
  "actual_optimizer_steps": 2500,
  "formal_fits": 6,
  "smoke_fits": 6,
  "seconds_do_not_include_manual_setup_and_failed_preflight_debug": true
}
