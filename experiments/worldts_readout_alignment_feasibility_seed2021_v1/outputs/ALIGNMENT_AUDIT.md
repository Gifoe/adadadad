# Latent/readout alignment audit

| dataset | forecast_MAE | forecast_MSE | latent_L1 | latent_cosine_loss | target_reconstruction_MAE | decoded_gap_MAE | state_cosine | state_negative_fraction | parameter_cosine | parameter_negative_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 0.203775 | 0.0876107 | 0.252612 | 0.280604 | 0.00508913 | 0.202878 | 0.0807732 | 0 | 0.578366 | 0 |
| PJM | 0.166042 | 0.07058 | 0.198312 | 0.159674 | 0.0108706 | 0.168208 | 0.0717869 | 0 | 0.254162 | 0 |

## Scope and interpretation

Original Stage2 checkpoints, frozen weights, eval mode, all 4,883 chronological validation origins per task. No optimizer updates. State gradients use a fresh leaf h_pred; Decoder autograd remains enabled. Parameter gradients use the identical eight batches and weights; Predictor requires_grad is temporarily enabled without changing eval mode or accumulating .grad. All model tensor hashes remain unchanged.

Official full objective is 10 L1 + 15 trajectory cosine + 0.05 legacy VC(pred) + 0.05 legacy VC(target). Target regularizer is constant with respect to the frozen-Encoder Predictor. Prediction variance/covariance term couples batch and time tokens; rows are batch-objective state-gradient components, not independent full sample losses. L1 and cosine are also analyzed separately. Zero/nonfinite gradients are counted and excluded, not assigned cosine zero. Epsilon=1e-12; near-zero means absolute cosine <0.1. Norms depend on the mean reduction and actual batch size, including the final batch of 19; compare matching batches, not unmatched scales. Positive rescaling changes norms but cannot change the cosine sign.

Forecast/reconstruction/gap metrics are in train-standardized observations after the official history RevIN inverse. Raw observation forecast errors are separately recorded. Target reconstruction is label-conditioned and is not a forecasting lower bound. Decoder nonlinearity and history-dependent RevIN can cause geometric mismatch; this observational audit alone does not establish causality.

## Gradient distributions

| dataset | space | objective | n_total | invalid | valid | mean | median | q10 | q25 | q75 | q90 | negative_fraction | near_zero_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | state | l1 | 4883 | 0 | 4883 | 0.0608217 | 0.0620248 | 0.0453464 | 0.0538269 | 0.0693582 | 0.0775026 | 0 | 0.994061 |
| NP | predictor_parameter | l1 | 8 | 0 | 8 | 0.599169 | 0.671986 | 0.376929 | 0.601421 | 0.711365 | 0.718814 | 0 | 0 |
| NP | state | cosine | 4883 | 0 | 4883 | 0.0803823 | 0.0728689 | 0.0535673 | 0.0628234 | 0.0925844 | 0.119765 | 0 | 0.791522 |
| NP | predictor_parameter | cosine | 8 | 0 | 8 | 0.55861 | 0.594956 | 0.397073 | 0.560527 | 0.692489 | 0.719499 | 0 | 0.125 |
| NP | state | full | 4883 | 0 | 4883 | 0.0807732 | 0.0754532 | 0.0600016 | 0.067268 | 0.0937749 | 0.114324 | 0 | 0.808519 |
| NP | predictor_parameter | full | 8 | 0 | 8 | 0.578366 | 0.640064 | 0.38694 | 0.595499 | 0.689757 | 0.724077 | 0 | 0.125 |
| PJM | state | l1 | 4883 | 0 | 4883 | 0.076323 | 0.0763279 | 0.0620414 | 0.0680564 | 0.0838337 | 0.0902138 | 0 | 0.97522 |
| PJM | predictor_parameter | l1 | 8 | 0 | 8 | 0.295212 | 0.277178 | 0.236652 | 0.250204 | 0.324085 | 0.382743 | 0 | 0 |
| PJM | state | cosine | 4883 | 0 | 4883 | 0.0611531 | 0.0618857 | 0.0371737 | 0.0473861 | 0.0732129 | 0.0839069 | 0 | 0.998771 |
| PJM | predictor_parameter | cosine | 8 | 0 | 8 | 0.231094 | 0.206058 | 0.116228 | 0.127416 | 0.3319 | 0.3787 | 0 | 0 |
| PJM | state | full | 4883 | 0 | 4883 | 0.0717869 | 0.0715917 | 0.0518424 | 0.0609553 | 0.0817124 | 0.0921148 | 0 | 0.969076 |
| PJM | predictor_parameter | full | 8 | 0 | 8 | 0.254162 | 0.22905 | 0.158102 | 0.167614 | 0.330345 | 0.385074 | 0 | 0 |

## Forecast-error quartiles

| dataset | error_quartile | n_origins | forecast_MAE | valid | mean | median | q10 | q25 | q75 | q90 | negative_fraction | near_zero_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | Q1 low | 1221 | 0.107587 | 1221 | 0.0697553 | 0.0702303 | 0.0560174 | 0.0648525 | 0.0762562 | 0.0820738 | 0 | 0.99181 |
| NP | Q2 | 1221 | 0.143761 | 1221 | 0.0735145 | 0.0736247 | 0.0599505 | 0.0685489 | 0.078977 | 0.0858263 | 0 | 0.968059 |
| NP | Q3 | 1220 | 0.201125 | 1220 | 0.0765246 | 0.0702122 | 0.0490713 | 0.0650193 | 0.0897837 | 0.108582 | 0 | 0.848361 |
| NP | Q4 high | 1221 | 0.362624 | 1221 | 0.103295 | 0.106094 | 0.0780676 | 0.0922085 | 0.118434 | 0.126245 | 0 | 0.42588 |
| PJM | Q1 low | 1221 | 0.0963364 | 1221 | 0.0697396 | 0.0696096 | 0.0581578 | 0.0644167 | 0.0744416 | 0.0805917 | 0 | 1 |
| PJM | Q2 | 1221 | 0.133966 | 1221 | 0.0701702 | 0.0705538 | 0.0522714 | 0.060136 | 0.0770962 | 0.0857321 | 0 | 0.963964 |
| PJM | Q3 | 1220 | 0.187788 | 1220 | 0.0790558 | 0.0812184 | 0.0587279 | 0.0750069 | 0.0880915 | 0.0931461 | 0 | 0.990984 |
| PJM | Q4 high | 1221 | 0.246098 | 1221 | 0.0681879 | 0.0625868 | 0.045553 | 0.0531453 | 0.0855178 | 0.097757 | 0 | 0.921376 |

## Continuous-time-block correlations

| dataset | block_length | n_blocks | x | y | Pearson | Spearman | p_value | last_block_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 360 | 14 | latent_l1 | forecast_mae | 0.637872 | 0.608791 | NOT_COMPUTED_overlapping_origins_not_IID | 203 |
| NP | 360 | 14 | latent_trajectory_cosine | forecast_mae | 0.397166 | 0.375824 | NOT_COMPUTED_overlapping_origins_not_IID | 203 |
| NP | 360 | 14 | decoded_gap_mae | forecast_mae | 0.999827 | 0.995604 | NOT_COMPUTED_overlapping_origins_not_IID | 203 |
| NP | 360 | 14 | full_cosine | forecast_mae | 0.646478 | 0.608791 | NOT_COMPUTED_overlapping_origins_not_IID | 203 |
| NP | 720 | 7 | latent_l1 | forecast_mae | 0.618981 | 0.357143 | NOT_COMPUTED_overlapping_origins_not_IID | 563 |
| NP | 720 | 7 | latent_trajectory_cosine | forecast_mae | 0.13429 | 0.178571 | NOT_COMPUTED_overlapping_origins_not_IID | 563 |
| NP | 720 | 7 | decoded_gap_mae | forecast_mae | 0.999882 | 0.964286 | NOT_COMPUTED_overlapping_origins_not_IID | 563 |
| NP | 720 | 7 | full_cosine | forecast_mae | 0.649427 | 0.571429 | NOT_COMPUTED_overlapping_origins_not_IID | 563 |
| NP | 1680 | 3 | latent_l1 | forecast_mae | 0.958843 | 0.5 | NOT_COMPUTED_overlapping_origins_not_IID | 1523 |
| NP | 1680 | 3 | latent_trajectory_cosine | forecast_mae | 0.808459 | 0.5 | NOT_COMPUTED_overlapping_origins_not_IID | 1523 |
| NP | 1680 | 3 | decoded_gap_mae | forecast_mae | 0.999898 | 1 | NOT_COMPUTED_overlapping_origins_not_IID | 1523 |
| NP | 1680 | 3 | full_cosine | forecast_mae | 0.935552 | 1 | NOT_COMPUTED_overlapping_origins_not_IID | 1523 |
| PJM | 360 | 14 | latent_l1 | forecast_mae | 0.615478 | 0.696703 | NOT_COMPUTED_overlapping_origins_not_IID | 203 |
| PJM | 360 | 14 | latent_trajectory_cosine | forecast_mae | 0.732901 | 0.793407 | NOT_COMPUTED_overlapping_origins_not_IID | 203 |
| PJM | 360 | 14 | decoded_gap_mae | forecast_mae | 0.998508 | 0.991209 | NOT_COMPUTED_overlapping_origins_not_IID | 203 |
| PJM | 360 | 14 | full_cosine | forecast_mae | -0.127297 | -0.032967 | NOT_COMPUTED_overlapping_origins_not_IID | 203 |
| PJM | 720 | 7 | latent_l1 | forecast_mae | 0.636997 | 0.607143 | NOT_COMPUTED_overlapping_origins_not_IID | 563 |
| PJM | 720 | 7 | latent_trajectory_cosine | forecast_mae | 0.753208 | 0.535714 | NOT_COMPUTED_overlapping_origins_not_IID | 563 |
| PJM | 720 | 7 | decoded_gap_mae | forecast_mae | 0.998989 | 0.964286 | NOT_COMPUTED_overlapping_origins_not_IID | 563 |
| PJM | 720 | 7 | full_cosine | forecast_mae | -0.237832 | 0.0714286 | NOT_COMPUTED_overlapping_origins_not_IID | 563 |
| PJM | 1680 | 3 | latent_l1 | forecast_mae | 0.400265 | 0.5 | NOT_COMPUTED_overlapping_origins_not_IID | 1523 |
| PJM | 1680 | 3 | latent_trajectory_cosine | forecast_mae | 0.839229 | 1 | NOT_COMPUTED_overlapping_origins_not_IID | 1523 |
| PJM | 1680 | 3 | decoded_gap_mae | forecast_mae | 0.999258 | 1 | NOT_COMPUTED_overlapping_origins_not_IID | 1523 |
| PJM | 1680 | 3 | full_cosine | forecast_mae | -0.935045 | -0.5 | NOT_COMPUTED_overlapping_origins_not_IID | 1523 |

Pearson and Spearman use block means at 360/720/1680 hours; partial final blocks are explicitly counted. Window overlap still crosses adjacent block boundaries. No IID-window p-values or causal interpretation. Horizon-slice statistics are in horizon_gradient_summary.csv; gradient-norm distributions in gradient_norm_summary.csv. Figures 01–06 contain all requested views. Correlation of full batch losses is not represented as per-sample latent loss correlation.

## What this resolves

Assess both spaces and high-error quartiles above. Negative cosine is directional conflict for the sampled first-order gradient, whereas cosine near zero alone is weak alignment. Parameter-space checks contain only eight fixed validation batches and do not prove universal conflict. Training intervention and paired time-block effects are required before identifying an exploitable performance bottleneck.

## Observed conflict and dataset dependence

**1. 是否存在 latent/readout 不一致？** 有弱状态方向对齐，但未发现系统性反向冲突。NP/PJM 全预测窗口的完整目标状态梯度余弦均值为0.08077/0.07179，负余弦比例均为0；固定8个 batch 的 Predictor 参数梯度余弦为0.57837/0.25416，负比例也均为0。NP 最后120步切片出现2/4,883个负余弦（0.041%），其余完整目标预测段均为0。高误差四分位同样没有负余弦；NP 高误差组余弦反而更高（0.1033 vs低误差0.0698），PJM 高/低误差组为0.0682/0.0697。所有六个最佳 checkpoint 在相同8个 batch 上的参数余弦仍全部为正。不能把接近正交的状态梯度写成真实参数更新冲突。 训练模式下按步数稀疏抽样的106次参数诊断中有4次负余弦（NP A0为1/17，PJM A0为1/21、A2为2/21，其余0）。这说明部分更新存在短暂冲突；它既不是全训练batch的普查比例，也没有改变固定验证审计与跨任务干预的否定证据。原始轨迹及汇总均保留。

Equal-batch mean official full latent objective: NP=6.830150442185745, PJM=4.447391026160297. Components and verified weighted reconstruction are saved in latent_objective_components.csv. This scalar mean refers to batch objectives, not independent sample full losses.

NP/PJM 360-hour latent-L1 vs forecast-MAE block Pearson=0.6379/0.6155, Spearman=0.6088/0.6967. Full state cosine vs error Pearson=0.6465/-0.1273. These differ across tasks and depend on block length; at1680 hours only3 blocks remain. Decoded gap and forecast error are almost identical because target reconstruction is small, not because a prediction lower bound has been established. Target reconstruction MAE=0.00509/0.01087, compared with forecast MAE=0.20377/0.16604.

