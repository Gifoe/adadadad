# WorldTS future-response Loop — research report

Final conclusion: **E**. Decision: **INCONCLUSIVE**.

Completed real-task training executions: 26; controlled synthetic fits:12. Real-task fits use common-checkpoint two-stage continuation; they are not independent from-scratch pretraining. Optional seed expansion=False, selected by the frozen validation-only rule.

Primary paired comparisons against validation-selected nonrecurrent controls:

| dataset   |   seed | nonrecurrent         | shared_loop          |   relative_mse_gain |   relative_mae_gain |   mse_difference_loop_minus_control |
|:----------|-------:|:---------------------|:---------------------|--------------------:|--------------------:|------------------------------------:|
| NP        |   2021 | NP_H360_C4_K1_s2021  | NP_H360_C5_K2_s2021  |         0.0214042   |           0.0178789 |                        -0.0105632   |
| PJM       |   2021 | PJM_H360_B2_K1_s2021 | PJM_H360_C5_K4_s2021 |        -0.000827757 |          -0.0112703 |                         0.000100471 |

Verified original checkpoint reference:

| dataset   |   seed | shared_loop          |   mse_gain_vs_verified_original |   mae_gain_vs_verified_original | scope                                                                                                         |
|:----------|-------:|:---------------------|--------------------------------:|--------------------------------:|:--------------------------------------------------------------------------------------------------------------|
| NP        |   2021 | NP_H360_C5_K2_s2021  |                      -0.122104  |                      -0.0721854 | verified original checkpoint, no added continuation budget; reference, separate from matched training control |
| PJM       |   2021 | PJM_H360_C5_K4_s2021 |                      -0.0555766 |                      -0.0337681 | verified original checkpoint, no added continuation budget; reference, separate from matched training control |

Measured primary comparison budgets:

| dataset   |   seed |   parameter_relative_diff |   batch32_latency_relative_diff | within_parameter_10pct   | within_latency_25pct   |
|:----------|-------:|--------------------------:|--------------------------------:|:-------------------------|:-----------------------|
| NP        |   2021 |               -0.0504125  |                       0.0733647 | True                     | True                   |
| PJM       |   2021 |                0.00237756 |                       0.0783219 | True                     | True                   |

All completed real-task results:

| run_id                     | dataset   |   horizon | variant   |   k |   seed |   validation_mae |   reproduced_mse |   reproduced_mae |   num_parameters |   train_seconds |
|:---------------------------|:----------|----------:|:----------|----:|-------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|
| NP_H24_B0_K1_s2021         | NP        |        24 | B0        |   1 |   2021 |         0.106693 |        0.164129  |         0.207332 |           345945 |         340.856 |
| NP_H24_B1_K1_s2021         | NP        |        24 | B1        |   1 |   2021 |         0.104927 |        0.162282  |         0.20569  |           404190 |         362.55  |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |   4 |   2021 |         0.107212 |        0.165772  |         0.20758  |           404190 |         408.956 |
| NP_H360_B0_K1_s2021        | NP        |       360 | B0        |   1 |   2021 |         0.222573 |        0.492817  |         0.451098 |          1026729 |         497.51  |
| NP_H360_B1_K1_s2021        | NP        |       360 | B1        |   1 |   2021 |         0.215863 |        0.483778  |         0.445734 |          1084974 |         443.337 |
| NP_H360_B2_K1_s2021        | NP        |       360 | B2        |   1 |   2021 |         0.213409 |        0.481943  |         0.45442  |          1084714 |         484.113 |
| NP_H360_C1_K4_s2021        | NP        |       360 | C1        |   4 |   2021 |         0.226455 |        0.495612  |         0.450897 |          1026730 |         562.451 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   4 |   2021 |         0.212747 |        0.47403   |         0.444638 |          1259706 |         414.339 |
| NP_H360_C4_K1_s2021        | NP        |       360 | C4        |   1 |   2021 |         0.20863  |        0.493511  |         0.454369 |          1142574 |         586.744 |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |   2 |   2021 |         0.216265 |        0.482948  |         0.446245 |          1084974 |         566.637 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |   4 |   2021 |         0.218404 |        0.482445  |         0.446074 |          1084974 |         494.744 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |   4 |   2021 |         0.213199 |        0.474227  |         0.443118 |          1084974 |         605.425 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |   4 |   2021 |         0.215158 |        0.46175   |         0.43719  |          1084974 |         628.455 |
| PJM_H24_B0_K1_s2021        | PJM       |        24 | B0        |   1 |   2021 |         0.109202 |        0.0566208 |         0.140923 |          1087481 |         321.885 |
| PJM_H24_B1_K1_s2021        | PJM       |        24 | B1        |   1 |   2021 |         0.108863 |        0.0560395 |         0.140117 |          1203774 |         301.402 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |   4 |   2021 |         0.109262 |        0.0563809 |         0.140548 |          1203774 |         339.132 |
| PJM_H360_B0_K1_s2021       | PJM       |       360 | B0        |   1 |   2021 |         0.16379  |        0.127503  |         0.223339 |          2448713 |         642.101 |
| PJM_H360_B1_K1_s2021       | PJM       |       360 | B1        |   1 |   2021 |         0.167901 |        0.12047   |         0.2179   |          2565006 |         492.414 |
| PJM_H360_B2_K1_s2021       | PJM       |       360 | B2        |   1 |   2021 |         0.160664 |        0.121377  |         0.213058 |          2558922 |         564.541 |
| PJM_H360_C1_K4_s2021       | PJM       |       360 | C1        |   4 |   2021 |         0.162155 |        0.132914  |         0.226038 |          2448714 |        1883.59  |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   4 |   2021 |         0.168792 |        0.115476  |         0.212404 |          2913882 |         633.648 |
| PJM_H360_C4_K1_s2021       | PJM       |       360 | C4        |   1 |   2021 |         0.163477 |        0.118256  |         0.217076 |          2679950 |         551.297 |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |   2 |   2021 |         0.167089 |        0.119239  |         0.215955 |          2565006 |         509.657 |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |   4 |   2021 |         0.164605 |        0.121478  |         0.21546  |          2565006 |         599.892 |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |   4 |   2021 |         0.161695 |        0.113359  |         0.207015 |          2565006 |         631.591 |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |   4 |   2021 |         0.162514 |        0.11037   |         0.211632 |          2565006 |         568.8   |

## Stage0

# Stage 0 — Baseline audit

## Research Question

Can the current four official checkpoints support controlled comparisons?

## Pre-experiment Hypothesis

All raw targets, scaler parameters, rolling origins and model predictions match the verified historical artifacts.

## Experiment Design

Read-only baseline artifacts; fresh GPU inference on every official test origin; fixed source 269426ea6217d7152a244da0e209d922acd46d4f.

## Actual Execution

Four configurations freshly reevaluated; no historical baseline file or checkpoint modified.

## Raw Results

| dataset   |   horizon |      mse |      mae |   max_prediction_abs_diff |   windows |
|:----------|----------:|---------:|---------:|--------------------------:|----------:|
| NP        |       360 | 0.430395 | 0.416201 |                         0 |     10125 |
| PJM       |       360 | 0.115082 | 0.208422 |                         0 |     10125 |
| NP        |        24 | 0.167258 | 0.213351 |                         0 |     10461 |
| PJM       |        24 | 0.057516 | 0.142788 |                         0 |     10461 |

## Interpretation

Checkpoint reuse verified by source/data/checkpoint/prediction hashes, exact raw targets and all test origins. This does not establish bitwise reproducibility of retraining.

## Scientific Contribution

This is an audit prerequisite, not a methodological contribution.

## Limitations and Alternative Explanations

Single pretrained seed; efficient GPU determinism may not reproduce training trajectories. Future covariate vintages remain unknown. Original PJM24 training peak was unavailable.

## Decision

Proceed with isolated, matched two-stage continued-training controls; keep historical checkpoint results separate from newly trained controls.

## Reproduction

Activate the verified baseline private venv and execute python scripts/stage0.py. Detailed verified rows: results/stage0_baseline_verification.csv.


## Stage1

# STAGE1_RESPONSE_DIAGNOSIS.md

## 1. Research Question

Does the exact source graph impose local future-condition support, and does this expressivity limitation matter in controlled systems?

## 2. Pre-experiment Hypothesis

A condition at s affects output h only for s<=h<=s+p_e+p_d-2. Short AR response a=.2 should differ from long AR response a=.95; synthetic observations do not establish an NP/PJM bottleneck.

## 3. Experiment Design

32 fixed validation origins per task; all output-by-input derivatives and central differences, fixed history RevIN. Four full validation error curves. Twelve from-scratch small pinned WorldTS synthetic fits: a in{.2,.95}, noise in{.05,.5}, B0/B1/C5K4.

## 4. Actual Execution

Four real-data sensitivity audits, 40 meaningful response-module preflights, and all12 toy fits completed. First toy attempt failed a strict reload assertion because freezing flags differed; equal weights had measured7.15e-7 CUDA kernel discrepancy. Restoring the same Stage2 configuration repaired the check, with the original failure retained.

## 5. Raw Results

| dataset   |   horizon |   pred_mse |   pred_mae |   encoded_future_diagnostic_mse |   encoded_future_diagnostic_mae |   latent_l1 |   correction_to_base_norm |   receptive_positions |   maximum_time_displacement |   outside_support_max_sensitivity |   finite_difference_max_abs_discrepancy | unit                                              | oracle_is_deployable   |
|:----------|----------:|-----------:|-----------:|--------------------------------:|--------------------------------:|------------:|--------------------------:|----------------------:|----------------------------:|----------------------------------:|----------------------------------------:|:--------------------------------------------------|:-----------------------|
| NP        |       360 |  0.0885682 |  0.202338  |                     6.71177e-05 |                      0.00507027 |    0.252898 |                   3.29807 |                     5 |                           4 |                                 0 |                             0.000168622 | train-standardized covariate; history RevIN fixed | False                  |
| PJM       |       360 |  0.0714451 |  0.168848  |                     0.000317145 |                      0.0107519  |    0.198396 |                   2.45269 |                     9 |                           8 |                                 0 |                             0.000287652 | train-standardized covariate; history RevIN fixed | False                  |
| NP        |        24 |  0.0125046 |  0.0807615 |                     3.81884e-05 |                      0.00474067 |    0.148604 |                   6.34199 |                     5 |                           4 |                                 0 |                             0.000112746 | train-standardized covariate; history RevIN fixed | False                  |
| PJM       |        24 |  0.0233557 |  0.0959376 |                     8.29967e-05 |                      0.00644357 |    0.125854 |                   5.44648 |                     9 |                           8 |                                 0 |                             0.000233077 | train-standardized covariate; history RevIN fixed | False                  |

Synthetic raw results:

|   system_a |   noise_std | variant   |   k |        mse |       mae |   training_seconds | status   | training_family        |   fixed_seed |   physical_response_above_5_steps | source_type                                   |   n_train_points |   n_validation_points |   n_test_points | checkpoint_sha256                                                | predictions_sha256                                               | checkpoint_reloaded_equal   |
|-----------:|------------:|:----------|----:|-----------:|----------:|-------------------:|:---------|:-----------------------|-------------:|----------------------------------:|:----------------------------------------------|-----------------:|----------------------:|----------------:|:-----------------------------------------------------------------|:-----------------------------------------------------------------|:----------------------------|
|       0.2  |        0.05 | B0        |   1 | 0.00596989 | 0.0592079 |            65.6435 | COMPLETE | synthetic_from_scratch |         2021 |                          0.00032  | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | fbc61548f87e8b9b219e61f3615aff85d474ec8be767f019ac32beb869475bbd | 7636f05ce90b143aefa211c4102c166793e5f76b4149ce6290360f2188c39434 | True                        |
|       0.2  |        0.05 | B1        |   1 | 0.00718696 | 0.0651866 |            71.8825 | COMPLETE | synthetic_from_scratch |         2021 |                          0.00032  | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | 0d220397f869e054665e1fb2410b86738a71c260f635332dc8f35360f1d5129b | 883fb6c7f90c785da0b1835c146480bac5c29ec7520c0fb6f23918c193bab72e | True                        |
|       0.2  |        0.05 | C5        |   4 | 0.00699002 | 0.0629946 |            81.3825 | COMPLETE | synthetic_from_scratch |         2021 |                          0.00032  | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | d11e6659ca90698045133bfe330c06ac41c6035522f3ab73a3255d06ca98778f | 21f7314b70fcaf6baeea243f69a1cc9becc65dff294dd30dc4a1eeeecaadd408 | True                        |
|       0.2  |        0.5  | B0        |   1 | 0.263706   | 0.406145  |            31.7538 | COMPLETE | synthetic_from_scratch |         2021 |                          0.00032  | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | 956e7d1dfee8eb0c708292a59114c8e81c3b4f1ba03876c71684642b1804f7d5 | 75a91015650182ee310e0d58fca52725d4600a07fd078b2c3c429e859c30c7d7 | True                        |
|       0.2  |        0.5  | B1        |   1 | 0.269388   | 0.410705  |            44.8722 | COMPLETE | synthetic_from_scratch |         2021 |                          0.00032  | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | 3aec0b69a16aa6262afb54d8f12f3b5bb1b536d536fa8d394fe07f5bc3ab4861 | 247b1a56ec6db9b9183e1d4359759a680cf893d90df66b091fd6e94ff557baf0 | True                        |
|       0.2  |        0.5  | C5        |   4 | 0.270725   | 0.410708  |            38.4988 | COMPLETE | synthetic_from_scratch |         2021 |                          0.00032  | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | 941b9bd70ad88ced380cb5e20324d3bab89b3ed8a289bb582eb0c57f7c34a3a4 | 94c38f33067fe5715e248b9518e4717b81cf4def6e3065ef1938cbd2dfedeeb4 | True                        |
|       0.95 |        0.05 | B0        |   1 | 0.842098   | 0.717846  |            44.7746 | COMPLETE | synthetic_from_scratch |         2021 |                          0.773781 | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | ecaf6e8b892eae68cc1b8d7cbcacc90ae3d2b427dda580fdb700f2c7bab8888e | 4e9eebf30a00906906b465bba224722f9a60e2509972200a344c0be2f7135705 | True                        |
|       0.95 |        0.05 | B1        |   1 | 0.228211   | 0.368966  |            57.8278 | COMPLETE | synthetic_from_scratch |         2021 |                          0.773781 | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | 692f2c6f72be74ae258181ba526ff0da9545e9f7d75905d4971976a4434719fe | e4fb9fb315fd128e9c0e0e4a1fbdf0ef6dafa7e513bfd4a372877344a1422956 | True                        |
|       0.95 |        0.05 | C5        |   4 | 0.193745   | 0.341716  |            63.287  | COMPLETE | synthetic_from_scratch |         2021 |                          0.773781 | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | 1add213028e36bb7c32d40c89cadfa9264a80aac1fce9f38fc948d9c2161c00d | 57cd57be47a25ab8d18f253916757f0d33bd8bbe2d07b70a53010b89fe1e798d | True                        |
|       0.95 |        0.5  | B0        |   1 | 1.25426    | 0.868934  |            35.8768 | COMPLETE | synthetic_from_scratch |         2021 |                          0.773781 | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | 4c1c47dba4ce7bbc6ec7433b69a09786513a045d43a78679fcaf0136e01b1bdf | 96572dc0f59d325257342236ab511fa701c98bfc0ce7bafad34864334e4f6d04 | True                        |
|       0.95 |        0.5  | B1        |   1 | 0.748716   | 0.674229  |            56.2305 | COMPLETE | synthetic_from_scratch |         2021 |                          0.773781 | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | 8ed5f8441bd8e4423164af165fc6038a0444d00955201a905a346e061ec346ea | d2f10245c73598a7e16c48de76ea2a6f541d837a85408c2690d9b668a77371be | True                        |
|       0.95 |        0.5  | C5        |   4 | 1.05768    | 0.805264  |            41.7426 | COMPLETE | synthetic_from_scratch |         2021 |                          0.773781 | pinned_WorldTS_small_controlled_configuration |             2867 |                   409 |             820 | b0b9d0ce8909d78700813066677b4aab12a7f60ec79349adcc0b89b508ad9ce3 | 6105db77f6372f16d2c2b861d21fe7d7bdbc10c133f651db1e4f4eac973eca6c | True                        |

## 6. Interpretation

Actual NP support is5positions (maximum displacement4), PJM support is9positions (maximum displacement8); outside gradients exactly zero. Finite-difference checks agree within measured float32 error. Conditional future-state decoding is a diagnostic using future labels, neither a deployable model nor a mathematical performance bound. The graph result proves a structural support limit only.

AR a=0.2, noise=0.05: one-shot versus B0 MSE gain=-20.39%; shared K4 versus B0=-17.09%; shared K4 versus one-shot=2.74%. The true five-step response multiplier is0.000320.

AR a=0.2, noise=0.5: one-shot versus B0 MSE gain=-2.15%; shared K4 versus B0=-2.66%; shared K4 versus one-shot=-0.50%. The true five-step response multiplier is0.000320.

AR a=0.95, noise=0.05: one-shot versus B0 MSE gain=72.90%; shared K4 versus B0=76.99%; shared K4 versus one-shot=15.10%. The true five-step response multiplier is0.773781.

AR a=0.95, noise=0.5: one-shot versus B0 MSE gain=40.31%; shared K4 versus B0=15.67%; shared K4 versus one-shot=-41.27%. The true five-step response multiplier is0.773781.

NPH360: on32 fixed validation origins, predicted-state decoded MSE=0.0885682, encoded-future reconstruction MSE=6.71177e-05, latent L1=0.252898, correction/base norm ratio=3.29807. The reconstruction gap is consistent with a state-prediction or state-distribution mismatch; it cannot exclude decoder error on predicted states. The nonzero correction norm alone does not measure useful information.

PJMH360: on32 fixed validation origins, predicted-state decoded MSE=0.0714451, encoded-future reconstruction MSE=0.000317145, latent L1=0.198396, correction/base norm ratio=2.45269. The reconstruction gap is consistent with a state-prediction or state-distribution mismatch; it cannot exclude decoder error on predicted states. The nonzero correction norm alone does not measure useful information.

NPH24: on32 fixed validation origins, predicted-state decoded MSE=0.0125046, encoded-future reconstruction MSE=3.81884e-05, latent L1=0.148604, correction/base norm ratio=6.34199. The reconstruction gap is consistent with a state-prediction or state-distribution mismatch; it cannot exclude decoder error on predicted states. The nonzero correction norm alone does not measure useful information.

PJMH24: on32 fixed validation origins, predicted-state decoded MSE=0.0233557, encoded-future reconstruction MSE=8.29967e-05, latent L1=0.125854, correction/base norm ratio=5.44648. The reconstruction gap is consistent with a state-prediction or state-distribution mismatch; it cannot exclude decoder error on predicted states. The nonzero correction norm alone does not measure useful information.

Learned response versus known physical derivative,16 fixed validation origins:

|   system_a |   noise_std | variant   |   n_validation_origins |   physical_long_response_abs_mean |   learned_long_response_abs_mean |   physical_jacobian_mse |   max_gradient_outside_baseline_support |   future_input_after_output_abs_mean |
|-----------:|------------:|:----------|-----------------------:|----------------------------------:|---------------------------------:|------------------------:|----------------------------------------:|-------------------------------------:|
|       0.2  |        0.05 | B0        |                     16 |                       1.77866e-05 |                       0          |             0.000149632 |                               0         |                           0          |
|       0.2  |        0.05 | B1        |                     16 |                       1.77866e-05 |                       0.00733042 |             0.000345434 |                               0.139885  |                           0.00794948 |
|       0.2  |        0.05 | C5        |                     16 |                       1.77866e-05 |                       0.0058452  |             0.000288331 |                               0.108857  |                           0.00549632 |
|       0.2  |        0.5  | B0        |                     16 |                       1.5939e-05  |                       0          |             0.0012643   |                               0         |                           0          |
|       0.2  |        0.5  | B1        |                     16 |                       1.5939e-05  |                       0.00935757 |             0.00254785  |                               0.0930917 |                           0.00851201 |
|       0.2  |        0.5  | C5        |                     16 |                       1.5939e-05  |                       0.00975495 |             0.00232031  |                               0.152538  |                           0.0092067  |
|       0.95 |        0.05 | B0        |                     16 |                       0.154681    |                       0          |             0.0139221   |                               0         |                           0          |
|       0.95 |        0.05 | B1        |                     16 |                       0.154681    |                       0.102849   |             0.00489789  |                               0.601387  |                           0.0133999  |
|       0.95 |        0.05 | C5        |                     16 |                       0.154681    |                       0.115917   |             0.00629685  |                               0.910481  |                           0.0283332  |
|       0.95 |        0.5  | B0        |                     16 |                       0.138838    |                       0          |             0.0114809   |                               0         |                           0          |
|       0.95 |        0.5  | B1        |                     16 |                       0.138838    |                       0.0720894  |             0.00584324  |                               0.526109  |                           0.0117571  |
|       0.95 |        0.5  | C5        |                     16 |                       0.138838    |                       0.0374616  |             0.00880137  |                               0.254683  |                           0.00972839 |

No synthetic result substitutes for real-task evidence.

## 7. Scientific Contribution

The reproducible graph boundary is established. Toy comparisons separate known response duration and SNR; a negative toy result alone does not identify optimization versus architecture without the gradient and structural checks.

## 8. Limitations and Alternative Explanations

Only32 validation origins for real-data Jacobians and16 for toy derivatives. AR data are generated with iid future exogenous input, unlike benchmark forecasts. Synthetic dimensions are deliberately small and independently labeled. Known future conditions may be mixed bidirectionally; historical encoding remains causal.

## 9. Decision

Retain H1 as a graph statement; H1 as a real-task bottleneck remains unproven. Execute fixed Stage2/3 even if toy or Stage2 results are negative.

## 10. Reproduction

python scripts/stage1.py; python scripts/synthetic.py; python scripts/summarize.py --stage 1. Raw Jacobians/plots and per-toy checkpoint/history/predictions are retained.


![Baseline response Jacobians](figures/stage1_response_heatmap.png)

Heatmap color is log10 of the mean, over32 fixed validation origins, of the Euclidean norm of the two train-scaled future-covariate derivatives, plus1e-12 for visualization. Both axes use zero-based hourly positions. Raw gradients outside the verified support are exactly zero; the log-scale floor is not a nonzero measured effect.

![Controlled synthetic errors](figures/stage1_synthetic_comparison.png)

Each panel holds response duration and observation noise fixed. Synthetic test MSE is computed in that system's train-scaled target units; magnitudes across different panels do not measure one common noise-free task.

## Stage2

# STAGE2_ONESHOT_FUSION.md

## 1. Research Question

Does one-shot future temporal attention or wide convolution improve forecasting over equal-budget continued WorldTS?

## 2. Pre-experiment Hypothesis

No improvement is presumed. Validation MAE selects checkpoints; all fixed methods and K values were preregistered before new test results.

## 3. Experiment Design

Same task-specific pretrained seed2021 initialization for every method; continuation seed2021; original complete two-stage objective and maximum20/20epochs, patience5, batch32 and published LR. Shared attention uses Values only from correction, no output bias, eta=sigmoid(logit) initially.5, zero output projection, bidirectional known-future mixing. Conv has nine bidirectional dilated layers, receptive field1023. C3 uses four unshared blocks; C4 uses width128 once. Parameters and actual latency are measured separately.

## 4. Actual Execution

Completed configurations below. B1 is mathematically identical to shared K1 and is reused as the K1 control, never falsely labeled separate training. Old baseline inference is kept separate from equally budgeted B0 continued training.

## 5. Raw Results

| run_id               | dataset   |   horizon | variant   |   k |   seed |   validation_mae |   reproduced_mse |   reproduced_mae |   num_parameters |   train_seconds |
|:---------------------|:----------|----------:|:----------|----:|-------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|
| NP_H360_B0_K1_s2021  | NP        |       360 | B0        |   1 |   2021 |         0.222573 |         0.492817 |         0.451098 |          1026729 |         497.51  |
| NP_H360_B1_K1_s2021  | NP        |       360 | B1        |   1 |   2021 |         0.215863 |         0.483778 |         0.445734 |          1084974 |         443.337 |
| NP_H360_B2_K1_s2021  | NP        |       360 | B2        |   1 |   2021 |         0.213409 |         0.481943 |         0.45442  |          1084714 |         484.113 |
| PJM_H360_B0_K1_s2021 | PJM       |       360 | B0        |   1 |   2021 |         0.16379  |         0.127503 |         0.223339 |          2448713 |         642.101 |
| PJM_H360_B1_K1_s2021 | PJM       |       360 | B1        |   1 |   2021 |         0.167901 |         0.12047  |         0.2179   |          2565006 |         492.414 |
| PJM_H360_B2_K1_s2021 | PJM       |       360 | B2        |   1 |   2021 |         0.160664 |         0.121377 |         0.213058 |          2558922 |         564.541 |

## 6. Interpretation

NP_H360_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=-14.50%. Validation MAE=0.222573; best Stage1 epoch=1/6, Stage2=12/17.

NP_H360_B1_K1_s2021: relative test MSE gain versus equally continued B0=1.83%, MAE gain=1.19%; versus verified original checkpoint MSE gain=-12.40%. Validation MAE=0.215863; best Stage1 epoch=1/6, Stage2=8/13.

NP_H360_B2_K1_s2021: relative test MSE gain versus equally continued B0=2.21%, MAE gain=-0.74%; versus verified original checkpoint MSE gain=-11.98%. Validation MAE=0.213409; best Stage1 epoch=1/6, Stage2=9/14.

PJM_H360_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=-10.79%. Validation MAE=0.163790; best Stage1 epoch=1/6, Stage2=8/13.

PJM_H360_B1_K1_s2021: relative test MSE gain versus equally continued B0=5.52%, MAE gain=2.44%; versus verified original checkpoint MSE gain=-4.68%. Validation MAE=0.167901; best Stage1 epoch=1/6, Stage2=2/7.

PJM_H360_B2_K1_s2021: relative test MSE gain versus equally continued B0=4.80%, MAE gain=4.60%; versus verified original checkpoint MSE gain=-5.47%. Validation MAE=0.160664; best Stage1 epoch=1/6, Stage2=4/9.

Direct comparison judgments:

NP: B1 improves MSE and improves MAE versus continued B0; B2 improves MSE and does not improve MAE versus continued B0. This is a measured single-seed direction, without a multi-seed stability claim.

PJM: B1 improves MSE and improves MAE versus continued B0; B2 improves MSE and improves MAE versus continued B0. This is a measured single-seed direction, without a multi-seed stability claim.

Test error by horizon band:

| dataset   | variant   | positions   |       mse |   mse_gain_vs_continued_B0 |
|:----------|:----------|:------------|----------:|---------------------------:|
| NP        | B0        | 1-120       | 0.309492  |                 0          |
| NP        | B0        | 121-240     | 0.495888  |                 0          |
| NP        | B0        | 241-360     | 0.673072  |                 0          |
| NP        | B1        | 1-120       | 0.299241  |                 0.0331216  |
| NP        | B1        | 121-240     | 0.485407  |                 0.0211358  |
| NP        | B1        | 241-360     | 0.666686  |                 0.00948764 |
| NP        | B2        | 1-120       | 0.298869  |                 0.034322   |
| NP        | B2        | 121-240     | 0.480838  |                 0.03035    |
| NP        | B2        | 241-360     | 0.666123  |                 0.0103248  |
| PJM       | B0        | 1-120       | 0.106572  |                 0          |
| PJM       | B0        | 121-240     | 0.131926  |                 0          |
| PJM       | B0        | 241-360     | 0.144012  |                 0          |
| PJM       | B1        | 1-120       | 0.0989558 |                 0.0714657  |
| PJM       | B1        | 121-240     | 0.124961  |                 0.0527945  |
| PJM       | B1        | 241-360     | 0.137493  |                 0.0452687  |
| PJM       | B2        | 1-120       | 0.0990452 |                 0.0706266  |
| PJM       | B2        | 121-240     | 0.127933  |                 0.0302662  |
| PJM       | B2        | 241-360     | 0.137153  |                 0.0476286  |

Measured costs:

| run_id                     | dataset   |   horizon | variant   |   k |   seed |   parameters |   training_seconds | training_time_scope                                                               |   process_attempt_wall_seconds |   failed_attempts |   peak_gpu_mb |   batch1_ms_median |   batch1_ms_p90 |   batch32_ms_median |   batch32_ms_p90 |
|:---------------------------|:----------|----------:|:----------|----:|-------:|-------------:|-------------------:|:----------------------------------------------------------------------------------|-------------------------------:|------------------:|--------------:|-------------------:|----------------:|--------------------:|-----------------:|
| NP_H24_B0_K1_s2021         | NP        |        24 | B0        |   1 |   2021 |       345945 |            340.856 | retained/resumed optimization including validation; failed partial epoch excluded |                        372.972 |                 0 |       275.64  |            4.7897  |         4.8833  |             4.99563 |          5.08548 |
| NP_H24_B1_K1_s2021         | NP        |        24 | B1        |   1 |   2021 |       404190 |            362.55  | retained/resumed optimization including validation; failed partial epoch excluded |                        394.973 |                 0 |       277.509 |            5.3287  |         5.52918 |             5.48747 |          5.60719 |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |   4 |   2021 |       404190 |            408.956 | retained/resumed optimization including validation; failed partial epoch excluded |                        442.121 |                 0 |       278.634 |            6.55069 |         6.67871 |             6.83038 |          6.98831 |
| NP_H360_B0_K1_s2021        | NP        |       360 | B0        |   1 |   2021 |      1026729 |            497.51  | retained/resumed optimization including validation; failed partial epoch excluded |                        531.594 |                 0 |      1352.24  |            4.84102 |         4.98212 |             6.28235 |          6.54088 |
| NP_H360_B1_K1_s2021        | NP        |       360 | B1        |   1 |   2021 |      1084974 |            443.337 | retained/resumed optimization including validation; failed partial epoch excluded |                        510.93  |                 1 |      1367.77  |            5.32664 |         5.54078 |             6.82717 |          7.00282 |
| NP_H360_B2_K1_s2021        | NP        |       360 | B2        |   1 |   2021 |      1084714 |            484.113 | retained/resumed optimization including validation; failed partial epoch excluded |                        523.77  |                 0 |      1366.61  |            5.81003 |         5.87871 |             7.27736 |          7.50081 |
| NP_H360_C1_K4_s2021        | NP        |       360 | C1        |   4 |   2021 |      1026730 |            562.451 | retained/resumed optimization including validation; failed partial epoch excluded |                        605.196 |                 0 |      3496.09  |            7.5223  |         7.62686 |            11.6809  |         12.2495  |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   4 |   2021 |      1259706 |            414.339 | retained/resumed optimization including validation; failed partial epoch excluded |                        454.619 |                 0 |      1388.6   |            6.62446 |         6.719   |             8.21768 |          8.38026 |
| NP_H360_C4_K1_s2021        | NP        |       360 | C4        |   1 |   2021 |      1142574 |            586.744 | retained/resumed optimization including validation; failed partial epoch excluded |                        626.562 |                 0 |      1367.67  |            5.36277 |         5.58045 |             6.84558 |          7.14604 |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |   2 |   2021 |      1084974 |            566.637 | retained/resumed optimization including validation; failed partial epoch excluded |                        606.37  |                 0 |      1374.24  |            5.75504 |         5.83049 |             7.34781 |          7.66894 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |   4 |   2021 |      1084974 |            494.744 | retained/resumed optimization including validation; failed partial epoch excluded |                        534.817 |                 0 |      1385.25  |            6.62907 |         6.76481 |             8.25526 |          8.45137 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |   4 |   2021 |      1084974 |            605.425 | retained/resumed optimization including validation; failed partial epoch excluded |                        645.581 |                 0 |      1385.42  |            6.81366 |         6.91846 |             8.21773 |          8.40072 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |   4 |   2021 |      1084974 |            628.455 | retained/resumed optimization including validation; failed partial epoch excluded |                        668.613 |                 0 |      1385.25  |            6.68781 |         6.88539 |             8.21891 |          8.37652 |
| PJM_H24_B0_K1_s2021        | PJM       |        24 | B0        |   1 |   2021 |      1087481 |            321.885 | retained/resumed optimization including validation; failed partial epoch excluded |                        355.305 |                 0 |       819.013 |            4.78787 |         4.9054  |             6.85574 |          7.132   |
| PJM_H24_B1_K1_s2021        | PJM       |        24 | B1        |   1 |   2021 |      1203774 |            301.402 | retained/resumed optimization including validation; failed partial epoch excluded |                        335.176 |                 0 |       822.365 |            5.27333 |         5.50914 |             7.35502 |          7.57155 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |   4 |   2021 |      1203774 |            339.132 | retained/resumed optimization including validation; failed partial epoch excluded |                        373.413 |                 0 |       824.428 |            6.60075 |         6.80572 |             8.59877 |          8.8858  |
| PJM_H360_B0_K1_s2021       | PJM       |       360 | B0        |   1 |   2021 |      2448713 |            642.101 | retained/resumed optimization including validation; failed partial epoch excluded |                        685.099 |                 0 |      4928.02  |            4.8533  |         5.01825 |            12.3261  |         12.6661  |
| PJM_H360_B1_K1_s2021       | PJM       |       360 | B1        |   1 |   2021 |      2565006 |            492.414 | retained/resumed optimization including validation; failed partial epoch excluded |                        535.505 |                 0 |      4950.33  |            5.3484  |         5.50171 |            12.8614  |         13.3615  |
| PJM_H360_B2_K1_s2021       | PJM       |       360 | B2        |   1 |   2021 |      2558922 |            564.541 | retained/resumed optimization including validation; failed partial epoch excluded |                        607.96  |                 0 |      4949.55  |            5.8375  |         5.93966 |            13.2489  |         13.59    |
| PJM_H360_C1_K4_s2021       | PJM       |       360 | C1        |   4 |   2021 |      2448714 |           1883.59  | retained/resumed optimization including validation; failed partial epoch excluded |                       1939.45  |                 0 |     11771.9   |            7.48936 |         7.83046 |            32.8504  |         33.3961  |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   4 |   2021 |      2913882 |            633.648 | retained/resumed optimization including validation; failed partial epoch excluded |                        678.612 |                 0 |      4989.32  |            6.64261 |         6.87483 |            14.3252  |         14.863   |
| PJM_H360_C4_K1_s2021       | PJM       |       360 | C4        |   1 |   2021 |      2679950 |            551.297 | retained/resumed optimization including validation; failed partial epoch excluded |                        594.39  |                 0 |      4954.23  |            5.35006 |         5.51689 |            12.93    |         13.4886  |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |   2 |   2021 |      2565006 |            509.657 | retained/resumed optimization including validation; failed partial epoch excluded |                        553.049 |                 0 |      4961.68  |            5.824   |         5.99645 |            13.3442  |         13.6558  |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |   4 |   2021 |      2565006 |            599.892 | retained/resumed optimization including validation; failed partial epoch excluded |                        643.892 |                 0 |      4982.65  |            6.67038 |         6.76193 |            14.2866  |         14.9325  |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |   4 |   2021 |      2565006 |            631.591 | retained/resumed optimization including validation; failed partial epoch excluded |                        675.571 |                 0 |      4983.41  |            6.75728 |         6.87882 |            14.3527  |         15.0718  |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |   4 |   2021 |      2565006 |            568.8   | retained/resumed optimization including validation; failed partial epoch excluded |                        612.747 |                 0 |      4981.5   |            6.66566 |         6.80171 |            14.4157  |         14.9812  |

A lower test value is not a reason to change the architecture or gate. Whole-model parameter and actual latency mismatches limit efficiency claims. No result is attributed to sharing before the appropriate controls are compared.

## 7. Scientific Contribution

One-shot attention improves both test metrics over continued B0 on both tasks, but every one-shot attention/convolution fit remains worse than the independently verified original checkpoint in test MSE. Conv and attention have closely matched parameter counts and show no consistent attention-specific advantage across both metrics. The largest relative gains over continued B0 occur in the first120 forecast positions, rather than uniquely in the most distant band. This comparison excludes a simple extra-training-budget explanation for the within-continuation point gains; it does not establish that the original local support is the principal real-data bottleneck, or that recurrence is necessary. Single-seed and continuation limitations remain.

## 8. Limitations and Alternative Explanations

Main family is checkpoint-initialized continuation, not from-scratch. Each continuation seed starts from the same original trained weights. Fixed-K decoders may mismatch states at other test depths. Masking/noise ablations are evaluation robustness tests, not causal interventions.

## 9. Decision

Proceed to the next minimum stage regardless of sign. Expand seeds only through the preregistered validation gate.

## 10. Reproduction

python scripts/train.py --run-id <preregisteredID>; python scripts/orchestrate.py. Per-run research_spec/config/command/history/freeze audit/checkpoint/predictions and numbered stdout attempts are saved.


## Stage3

# STAGE3_RECURRENT_PROPAGATION.md

## 1. Research Question

Does shared anchored recurrence improve beyond ordinary historical looping, one-shot mixing, unshared stacks and wider one-shot controls?

## 2. Pre-experiment Hypothesis

No improvement is presumed. Validation MAE selects checkpoints; all fixed methods and K values were preregistered before new test results.

## 3. Experiment Design

Same task-specific pretrained seed2021 initialization for every method; continuation seed2021; original complete two-stage objective and maximum20/20epochs, patience5, batch32 and published LR. Shared attention uses Values only from correction, no output bias, eta=sigmoid(logit) initially.5, zero output projection, bidirectional known-future mixing. Conv has nine bidirectional dilated layers, receptive field1023. C3 uses four unshared blocks; C4 uses width128 once. Parameters and actual latency are measured separately.

## 4. Actual Execution

Completed configurations below. B1 is mathematically identical to shared K1 and is reused as the K1 control, never falsely labeled separate training. Old baseline inference is kept separate from equally budgeted B0 continued training.

## 5. Raw Results

| run_id               | dataset   |   horizon | variant   |   k |   seed |   validation_mae |   reproduced_mse |   reproduced_mae |   num_parameters |   train_seconds |
|:---------------------|:----------|----------:|:----------|----:|-------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|
| NP_H360_C1_K4_s2021  | NP        |       360 | C1        |   4 |   2021 |         0.226455 |         0.495612 |         0.450897 |          1026730 |         562.451 |
| NP_H360_C3_K4_s2021  | NP        |       360 | C3        |   4 |   2021 |         0.212747 |         0.47403  |         0.444638 |          1259706 |         414.339 |
| NP_H360_C4_K1_s2021  | NP        |       360 | C4        |   1 |   2021 |         0.20863  |         0.493511 |         0.454369 |          1142574 |         586.744 |
| NP_H360_C5_K2_s2021  | NP        |       360 | C5        |   2 |   2021 |         0.216265 |         0.482948 |         0.446245 |          1084974 |         566.637 |
| NP_H360_C5_K4_s2021  | NP        |       360 | C5        |   4 |   2021 |         0.218404 |         0.482445 |         0.446074 |          1084974 |         494.744 |
| PJM_H360_C1_K4_s2021 | PJM       |       360 | C1        |   4 |   2021 |         0.162155 |         0.132914 |         0.226038 |          2448714 |        1883.59  |
| PJM_H360_C3_K4_s2021 | PJM       |       360 | C3        |   4 |   2021 |         0.168792 |         0.115476 |         0.212404 |          2913882 |         633.648 |
| PJM_H360_C4_K1_s2021 | PJM       |       360 | C4        |   1 |   2021 |         0.163477 |         0.118256 |         0.217076 |          2679950 |         551.297 |
| PJM_H360_C5_K2_s2021 | PJM       |       360 | C5        |   2 |   2021 |         0.167089 |         0.119239 |         0.215955 |          2565006 |         509.657 |
| PJM_H360_C5_K4_s2021 | PJM       |       360 | C5        |   4 |   2021 |         0.164605 |         0.121478 |         0.21546  |          2565006 |         599.892 |
| NP_H360_B0_K1_s2021  | NP        |       360 | B0        |   1 |   2021 |         0.222573 |         0.492817 |         0.451098 |          1026729 |         497.51  |
| NP_H360_B1_K1_s2021  | NP        |       360 | B1        |   1 |   2021 |         0.215863 |         0.483778 |         0.445734 |          1084974 |         443.337 |
| NP_H360_B2_K1_s2021  | NP        |       360 | B2        |   1 |   2021 |         0.213409 |         0.481943 |         0.45442  |          1084714 |         484.113 |
| PJM_H360_B0_K1_s2021 | PJM       |       360 | B0        |   1 |   2021 |         0.16379  |         0.127503 |         0.223339 |          2448713 |         642.101 |
| PJM_H360_B1_K1_s2021 | PJM       |       360 | B1        |   1 |   2021 |         0.167901 |         0.12047  |         0.2179   |          2565006 |         492.414 |
| PJM_H360_B2_K1_s2021 | PJM       |       360 | B2        |   1 |   2021 |         0.160664 |         0.121377 |         0.213058 |          2558922 |         564.541 |

## 6. Interpretation

NP_H360_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=-14.50%. Validation MAE=0.222573; best Stage1 epoch=1/6, Stage2=12/17.

NP_H360_B1_K1_s2021: relative test MSE gain versus equally continued B0=1.83%, MAE gain=1.19%; versus verified original checkpoint MSE gain=-12.40%. Validation MAE=0.215863; best Stage1 epoch=1/6, Stage2=8/13.

NP_H360_B2_K1_s2021: relative test MSE gain versus equally continued B0=2.21%, MAE gain=-0.74%; versus verified original checkpoint MSE gain=-11.98%. Validation MAE=0.213409; best Stage1 epoch=1/6, Stage2=9/14.

NP_H360_C1_K4_s2021: relative test MSE gain versus equally continued B0=-0.57%, MAE gain=0.04%; versus verified original checkpoint MSE gain=-15.15%. Validation MAE=0.226455; best Stage1 epoch=1/6, Stage2=6/11.

NP_H360_C3_K4_s2021: relative test MSE gain versus equally continued B0=3.81%, MAE gain=1.43%; versus verified original checkpoint MSE gain=-10.14%. Validation MAE=0.212747; best Stage1 epoch=1/6, Stage2=4/9.

NP_H360_C4_K1_s2021: relative test MSE gain versus equally continued B0=-0.14%, MAE gain=-0.73%; versus verified original checkpoint MSE gain=-14.66%. Validation MAE=0.208630; best Stage1 epoch=1/6, Stage2=16/20.

NP_H360_C5_K2_s2021: relative test MSE gain versus equally continued B0=2.00%, MAE gain=1.08%; versus verified original checkpoint MSE gain=-12.21%. Validation MAE=0.216265; best Stage1 epoch=1/6, Stage2=13/18.

NP_H360_C5_K4_s2021: relative test MSE gain versus equally continued B0=2.10%, MAE gain=1.11%; versus verified original checkpoint MSE gain=-12.09%. Validation MAE=0.218404; best Stage1 epoch=1/6, Stage2=8/13.

PJM_H360_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=-10.79%. Validation MAE=0.163790; best Stage1 epoch=1/6, Stage2=8/13.

PJM_H360_B1_K1_s2021: relative test MSE gain versus equally continued B0=5.52%, MAE gain=2.44%; versus verified original checkpoint MSE gain=-4.68%. Validation MAE=0.167901; best Stage1 epoch=1/6, Stage2=2/7.

PJM_H360_B2_K1_s2021: relative test MSE gain versus equally continued B0=4.80%, MAE gain=4.60%; versus verified original checkpoint MSE gain=-5.47%. Validation MAE=0.160664; best Stage1 epoch=1/6, Stage2=4/9.

PJM_H360_C1_K4_s2021: relative test MSE gain versus equally continued B0=-4.24%, MAE gain=-1.21%; versus verified original checkpoint MSE gain=-15.50%. Validation MAE=0.162155; best Stage1 epoch=1/6, Stage2=16/20.

PJM_H360_C3_K4_s2021: relative test MSE gain versus equally continued B0=9.43%, MAE gain=4.90%; versus verified original checkpoint MSE gain=-0.34%. Validation MAE=0.168792; best Stage1 epoch=1/6, Stage2=5/10.

PJM_H360_C4_K1_s2021: relative test MSE gain versus equally continued B0=7.25%, MAE gain=2.80%; versus verified original checkpoint MSE gain=-2.76%. Validation MAE=0.163477; best Stage1 epoch=1/6, Stage2=4/9.

PJM_H360_C5_K2_s2021: relative test MSE gain versus equally continued B0=6.48%, MAE gain=3.31%; versus verified original checkpoint MSE gain=-3.61%. Validation MAE=0.167089; best Stage1 epoch=1/6, Stage2=2/7.

PJM_H360_C5_K4_s2021: relative test MSE gain versus equally continued B0=4.73%, MAE gain=3.53%; versus verified original checkpoint MSE gain=-5.56%. Validation MAE=0.164605; best Stage1 epoch=1/6, Stage2=4/9.

Actual unshared-stack Stage1 gradient observations:

| dataset   | response_block   |   observed_parameters |   parameters_with_positive_gradient |   min_median_parameter_gradient_norm |   max_median_parameter_gradient_norm |
|:----------|:-----------------|----------------------:|------------------------------------:|-------------------------------------:|-------------------------------------:|
| NP        | 0                |                     6 |                                   6 |                           0.00330895 |                            0.0411123 |
| NP        | 1                |                     6 |                                   6 |                           0.0118833  |                            0.203746  |
| NP        | 2                |                     6 |                                   6 |                           0.0427838  |                            0.429781  |
| NP        | 3                |                     6 |                                   6 |                           0.0989323  |                            0.836205  |
| NP        | gate_or_scalar   |                     1 |                                   1 |                           0.0744179  |                            0.0744179 |
| PJM       | 0                |                     6 |                                   6 |                           0.00260337 |                            0.0136878 |
| PJM       | 1                |                     6 |                                   6 |                           0.00141367 |                            0.0278482 |
| PJM       | 2                |                     6 |                                   6 |                           0.00856258 |                            0.151771  |
| PJM       | 3                |                     6 |                                   6 |                           0.0370408  |                            0.370719  |
| PJM       | gate_or_scalar   |                     1 |                                   1 |                           0.0250944  |                            0.0250944 |

All saved epoch means were checked finite. Positive observed gradients rule out wholly disconnected blocks; unequal gradient magnitudes do not by themselves prove the cause of an accuracy difference. First-step delayed gradients from zero output initialization remain an optimization difference. Complete per-parameter observations are in results/response_gradient_audit.csv.

Checkpoint propagation parameters:

| run_id                     | dataset   |   horizon | variant   |   seed | block              |        eta |   output_projection_frobenius_norm |   output_projection_initial_norm |   output_projection_changed_from_zero_initialization |   historical_loop_effective_coefficient |
|:---------------------------|:----------|----------:|:----------|-------:|:-------------------|-----------:|-----------------------------------:|---------------------------------:|-----------------------------------------------------:|----------------------------------------:|
| NP_H360_B1_K1_s2021        | NP        |       360 | B1        |   2021 | 0                  |   0.516346 |                            5.12256 |                                0 |                                                    1 |                             nan         |
| NP_H360_B2_K1_s2021        | NP        |       360 | B2        |   2021 | 0                  |   0.527826 |                            7.06792 |                                0 |                                                    1 |                             nan         |
| NP_H360_C1_K4_s2021        | NP        |       360 | C1        |   2021 | historical_encoder | nan        |                          nan       |                              nan |                                                  nan |                               0.0810307 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   2021 | 0                  |   0.507127 |                            3.04536 |                                0 |                                                    1 |                             nan         |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   2021 | 1                  |   0.507127 |                            2.7977  |                                0 |                                                    1 |                             nan         |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   2021 | 2                  |   0.507127 |                            3.70504 |                                0 |                                                    1 |                             nan         |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   2021 | 3                  |   0.507127 |                            5.08394 |                                0 |                                                    1 |                             nan         |
| NP_H360_C4_K1_s2021        | NP        |       360 | C4        |   2021 | 0                  |   0.499303 |                            5.78253 |                                0 |                                                    1 |                             nan         |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |   2021 | 0                  |   0.50014  |                            4.28002 |                                0 |                                                    1 |                             nan         |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |   2021 | 0                  |   0.4934   |                            3.9803  |                                0 |                                                    1 |                             nan         |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |   2021 | 0                  |   0.498879 |                            4.1264  |                                0 |                                                    1 |                             nan         |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |   2021 | 0                  |   0.469651 |                            3.11217 |                                0 |                                                    1 |                             nan         |
| PJM_H360_B1_K1_s2021       | PJM       |       360 | B1        |   2021 | 0                  |   0.489648 |                            4.38976 |                                0 |                                                    1 |                             nan         |
| PJM_H360_B2_K1_s2021       | PJM       |       360 | B2        |   2021 | 0                  |   0.531826 |                            9.78218 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C1_K4_s2021       | PJM       |       360 | C1        |   2021 | historical_encoder | nan        |                          nan       |                              nan |                                                  nan |                               0.0709938 |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   2021 | 0                  |   0.479333 |                            3.39478 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   2021 | 1                  |   0.479333 |                            2.35623 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   2021 | 2                  |   0.479333 |                            3.74364 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   2021 | 3                  |   0.479333 |                            4.32356 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C4_K1_s2021       | PJM       |       360 | C4        |   2021 | 0                  |   0.471398 |                            5.09657 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |   2021 | 0                  |   0.482643 |                            3.74246 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |   2021 | 0                  |   0.486725 |                            3.96557 |                                0 |                                                    1 |                             nan         |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |   2021 | 0                  |   0.494056 |                            4.56466 |                                0 |                                                    1 |                             nan         |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |   2021 | 0                  |   0.451829 |                            3.14437 |                                0 |                                                    1 |                             nan         |

Nonzero final output-projection norms verify updates from their zero initialization. Eta and the historical-loop coefficient describe actual learned update strengths; finite coefficients alone do not prove useful prediction or contraction. Norms across differently sized blocks are descriptive, not a quality ranking.

Direct comparison judgments:

NP_H360_C5_K2_s2021 versus validation-selected nonrecurrent NP_H360_C4_K1_s2021: MSE gain=2.14%, MAE gain=1.79%. Both point estimates favor recurrence, pending seed/block/budget evidence.

NP_H360_C5_K4_s2021 versus validation-selected nonrecurrent NP_H360_C4_K1_s2021: MSE gain=2.24%, MAE gain=1.83%. Both point estimates favor recurrence, pending seed/block/budget evidence.

PJM_H360_C5_K2_s2021 versus validation-selected nonrecurrent PJM_H360_B2_K1_s2021: MSE gain=1.76%, MAE gain=-1.36%. The point estimates do not jointly favor recurrence on both metrics.

PJM_H360_C5_K4_s2021 versus validation-selected nonrecurrent PJM_H360_B2_K1_s2021: MSE gain=-0.08%, MAE gain=-1.13%. The point estimates do not jointly favor recurrence on both metrics.

Same checkpoint depth results (fixed decoder, no training between depths):

| run_id               | dataset   |   horizon |   seed |   trained_k |   test_k |      mse |      mae | same_checkpoint_sha256                                           |
|:---------------------|:----------|----------:|-------:|------------:|---------:|---------:|---------:|:-----------------------------------------------------------------|
| NP_H360_C5_K2_s2021  | NP        |       360 |   2021 |           2 |        1 | 0.481242 | 0.445533 | 1397317b200c14b4751b3fc45a13f81f82ecd9c1356da934d723a2978565dee1 |
| NP_H360_C5_K2_s2021  | NP        |       360 |   2021 |           2 |        2 | 0.482948 | 0.446245 | 1397317b200c14b4751b3fc45a13f81f82ecd9c1356da934d723a2978565dee1 |
| NP_H360_C5_K2_s2021  | NP        |       360 |   2021 |           2 |        4 | 0.48811  | 0.448166 | 1397317b200c14b4751b3fc45a13f81f82ecd9c1356da934d723a2978565dee1 |
| NP_H360_C5_K4_s2021  | NP        |       360 |   2021 |           4 |        1 | 0.480859 | 0.445073 | 059516f5c02b8c5239e0625431485d2ce349c2bfec9c9f46120bb9e4c001eb29 |
| NP_H360_C5_K4_s2021  | NP        |       360 |   2021 |           4 |        2 | 0.482041 | 0.446119 | 059516f5c02b8c5239e0625431485d2ce349c2bfec9c9f46120bb9e4c001eb29 |
| NP_H360_C5_K4_s2021  | NP        |       360 |   2021 |           4 |        4 | 0.482445 | 0.446074 | 059516f5c02b8c5239e0625431485d2ce349c2bfec9c9f46120bb9e4c001eb29 |
| PJM_H360_C5_K2_s2021 | PJM       |       360 |   2021 |           2 |        1 | 0.120441 | 0.219183 | ab6bf0504dbf86d21361a36c284c799eb857458ca7fa1a296758b4308f381752 |
| PJM_H360_C5_K2_s2021 | PJM       |       360 |   2021 |           2 |        2 | 0.119239 | 0.215955 | ab6bf0504dbf86d21361a36c284c799eb857458ca7fa1a296758b4308f381752 |
| PJM_H360_C5_K2_s2021 | PJM       |       360 |   2021 |           2 |        4 | 0.119242 | 0.216    | ab6bf0504dbf86d21361a36c284c799eb857458ca7fa1a296758b4308f381752 |
| PJM_H360_C5_K4_s2021 | PJM       |       360 |   2021 |           4 |        1 | 0.124691 | 0.218078 | 376a2ab8db4cf3f846643ec2890be565ff55048900362e6411376a3d374f5fbd |
| PJM_H360_C5_K4_s2021 | PJM       |       360 |   2021 |           4 |        2 | 0.121261 | 0.215543 | 376a2ab8db4cf3f846643ec2890be565ff55048900362e6411376a3d374f5fbd |
| PJM_H360_C5_K4_s2021 | PJM       |       360 |   2021 |           4 |        4 | 0.121478 | 0.21546  | 376a2ab8db4cf3f846643ec2890be565ff55048900362e6411376a3d374f5fbd |

NP_H360_C5_K2_s2021: K4 versus K1 MSE gain=-1.43%. More iterations do not improve this checkpoint. This remains a fixed-K decoder comparison, with possible state-distribution mismatch.

NP_H360_C5_K4_s2021: K4 versus K1 MSE gain=-0.33%. More iterations do not improve this checkpoint. This remains a fixed-K decoder comparison, with possible state-distribution mismatch.

PJM_H360_C5_K2_s2021: K4 versus K1 MSE gain=1.00%. More iterations improve this checkpoint. This remains a fixed-K decoder comparison, with possible state-distribution mismatch.

PJM_H360_C5_K4_s2021: K4 versus K1 MSE gain=2.58%. More iterations improve this checkpoint. This remains a fixed-K decoder comparison, with possible state-distribution mismatch.

Measured costs:

| run_id                     | dataset   |   horizon | variant   |   k |   seed |   parameters |   training_seconds | training_time_scope                                                               |   process_attempt_wall_seconds |   failed_attempts |   peak_gpu_mb |   batch1_ms_median |   batch1_ms_p90 |   batch32_ms_median |   batch32_ms_p90 |
|:---------------------------|:----------|----------:|:----------|----:|-------:|-------------:|-------------------:|:----------------------------------------------------------------------------------|-------------------------------:|------------------:|--------------:|-------------------:|----------------:|--------------------:|-----------------:|
| NP_H24_B0_K1_s2021         | NP        |        24 | B0        |   1 |   2021 |       345945 |            340.856 | retained/resumed optimization including validation; failed partial epoch excluded |                        372.972 |                 0 |       275.64  |            4.7897  |         4.8833  |             4.99563 |          5.08548 |
| NP_H24_B1_K1_s2021         | NP        |        24 | B1        |   1 |   2021 |       404190 |            362.55  | retained/resumed optimization including validation; failed partial epoch excluded |                        394.973 |                 0 |       277.509 |            5.3287  |         5.52918 |             5.48747 |          5.60719 |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |   4 |   2021 |       404190 |            408.956 | retained/resumed optimization including validation; failed partial epoch excluded |                        442.121 |                 0 |       278.634 |            6.55069 |         6.67871 |             6.83038 |          6.98831 |
| NP_H360_B0_K1_s2021        | NP        |       360 | B0        |   1 |   2021 |      1026729 |            497.51  | retained/resumed optimization including validation; failed partial epoch excluded |                        531.594 |                 0 |      1352.24  |            4.84102 |         4.98212 |             6.28235 |          6.54088 |
| NP_H360_B1_K1_s2021        | NP        |       360 | B1        |   1 |   2021 |      1084974 |            443.337 | retained/resumed optimization including validation; failed partial epoch excluded |                        510.93  |                 1 |      1367.77  |            5.32664 |         5.54078 |             6.82717 |          7.00282 |
| NP_H360_B2_K1_s2021        | NP        |       360 | B2        |   1 |   2021 |      1084714 |            484.113 | retained/resumed optimization including validation; failed partial epoch excluded |                        523.77  |                 0 |      1366.61  |            5.81003 |         5.87871 |             7.27736 |          7.50081 |
| NP_H360_C1_K4_s2021        | NP        |       360 | C1        |   4 |   2021 |      1026730 |            562.451 | retained/resumed optimization including validation; failed partial epoch excluded |                        605.196 |                 0 |      3496.09  |            7.5223  |         7.62686 |            11.6809  |         12.2495  |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   4 |   2021 |      1259706 |            414.339 | retained/resumed optimization including validation; failed partial epoch excluded |                        454.619 |                 0 |      1388.6   |            6.62446 |         6.719   |             8.21768 |          8.38026 |
| NP_H360_C4_K1_s2021        | NP        |       360 | C4        |   1 |   2021 |      1142574 |            586.744 | retained/resumed optimization including validation; failed partial epoch excluded |                        626.562 |                 0 |      1367.67  |            5.36277 |         5.58045 |             6.84558 |          7.14604 |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |   2 |   2021 |      1084974 |            566.637 | retained/resumed optimization including validation; failed partial epoch excluded |                        606.37  |                 0 |      1374.24  |            5.75504 |         5.83049 |             7.34781 |          7.66894 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |   4 |   2021 |      1084974 |            494.744 | retained/resumed optimization including validation; failed partial epoch excluded |                        534.817 |                 0 |      1385.25  |            6.62907 |         6.76481 |             8.25526 |          8.45137 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |   4 |   2021 |      1084974 |            605.425 | retained/resumed optimization including validation; failed partial epoch excluded |                        645.581 |                 0 |      1385.42  |            6.81366 |         6.91846 |             8.21773 |          8.40072 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |   4 |   2021 |      1084974 |            628.455 | retained/resumed optimization including validation; failed partial epoch excluded |                        668.613 |                 0 |      1385.25  |            6.68781 |         6.88539 |             8.21891 |          8.37652 |
| PJM_H24_B0_K1_s2021        | PJM       |        24 | B0        |   1 |   2021 |      1087481 |            321.885 | retained/resumed optimization including validation; failed partial epoch excluded |                        355.305 |                 0 |       819.013 |            4.78787 |         4.9054  |             6.85574 |          7.132   |
| PJM_H24_B1_K1_s2021        | PJM       |        24 | B1        |   1 |   2021 |      1203774 |            301.402 | retained/resumed optimization including validation; failed partial epoch excluded |                        335.176 |                 0 |       822.365 |            5.27333 |         5.50914 |             7.35502 |          7.57155 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |   4 |   2021 |      1203774 |            339.132 | retained/resumed optimization including validation; failed partial epoch excluded |                        373.413 |                 0 |       824.428 |            6.60075 |         6.80572 |             8.59877 |          8.8858  |
| PJM_H360_B0_K1_s2021       | PJM       |       360 | B0        |   1 |   2021 |      2448713 |            642.101 | retained/resumed optimization including validation; failed partial epoch excluded |                        685.099 |                 0 |      4928.02  |            4.8533  |         5.01825 |            12.3261  |         12.6661  |
| PJM_H360_B1_K1_s2021       | PJM       |       360 | B1        |   1 |   2021 |      2565006 |            492.414 | retained/resumed optimization including validation; failed partial epoch excluded |                        535.505 |                 0 |      4950.33  |            5.3484  |         5.50171 |            12.8614  |         13.3615  |
| PJM_H360_B2_K1_s2021       | PJM       |       360 | B2        |   1 |   2021 |      2558922 |            564.541 | retained/resumed optimization including validation; failed partial epoch excluded |                        607.96  |                 0 |      4949.55  |            5.8375  |         5.93966 |            13.2489  |         13.59    |
| PJM_H360_C1_K4_s2021       | PJM       |       360 | C1        |   4 |   2021 |      2448714 |           1883.59  | retained/resumed optimization including validation; failed partial epoch excluded |                       1939.45  |                 0 |     11771.9   |            7.48936 |         7.83046 |            32.8504  |         33.3961  |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   4 |   2021 |      2913882 |            633.648 | retained/resumed optimization including validation; failed partial epoch excluded |                        678.612 |                 0 |      4989.32  |            6.64261 |         6.87483 |            14.3252  |         14.863   |
| PJM_H360_C4_K1_s2021       | PJM       |       360 | C4        |   1 |   2021 |      2679950 |            551.297 | retained/resumed optimization including validation; failed partial epoch excluded |                        594.39  |                 0 |      4954.23  |            5.35006 |         5.51689 |            12.93    |         13.4886  |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |   2 |   2021 |      2565006 |            509.657 | retained/resumed optimization including validation; failed partial epoch excluded |                        553.049 |                 0 |      4961.68  |            5.824   |         5.99645 |            13.3442  |         13.6558  |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |   4 |   2021 |      2565006 |            599.892 | retained/resumed optimization including validation; failed partial epoch excluded |                        643.892 |                 0 |      4982.65  |            6.67038 |         6.76193 |            14.2866  |         14.9325  |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |   4 |   2021 |      2565006 |            631.591 | retained/resumed optimization including validation; failed partial epoch excluded |                        675.571 |                 0 |      4983.41  |            6.75728 |         6.87882 |            14.3527  |         15.0718  |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |   4 |   2021 |      2565006 |            568.8   | retained/resumed optimization including validation; failed partial epoch excluded |                        612.747 |                 0 |      4981.5   |            6.66566 |         6.80171 |            14.4157  |         14.9812  |

A lower test value is not a reason to change the architecture or gate. Whole-model parameter and actual latency mismatches limit efficiency claims. No result is attributed to sharing before the appropriate controls are compared.

## 7. Scientific Contribution

The completed historical-loop, unshared-stack, wider-one-shot and shared-K2/K4 comparisons constrain generic depth, sharing and capacity explanations. Same-checkpoint K1/2/4 results separately test iteration effects without new training. The direct judgments and measured budget table above identify which point comparisons favor recurrence. A single global attention application already removes the structural reach limit; any extra K gain concerns refinement/optimization, rather than uniquely accessing distant conditions. The unshared zero-initialized stack has delayed early-layer gradients, so its score alone cannot exclude optimization differences.

## 8. Limitations and Alternative Explanations

Main family is checkpoint-initialized continuation, not from-scratch. Each continuation seed starts from the same original trained weights. Fixed-K decoders may mismatch states at other test depths. Masking/noise ablations are evaluation robustness tests, not causal interventions.

## 9. Decision

Proceed to the next minimum stage regardless of sign. Expand seeds only through the preregistered validation gate.

## 10. Reproduction

python scripts/train.py --run-id <preregisteredID>; python scripts/orchestrate.py. Per-run research_spec/config/command/history/freeze audit/checkpoint/predictions and numbered stdout attempts are saved.


## Stage4

# STAGE4_MECHANISM_AND_GENERALIZATION.md

## 1. Research Question

Are anchoring, correction-only propagation, shared parameters, covariate quality and horizon responsible for any observed effect?

## 2. Pre-experiment Hypothesis

No improvement is presumed. Validation MAE selects checkpoints; all fixed methods and K values were preregistered before new test results.

## 3. Experiment Design

Same task-specific pretrained seed2021 initialization for every method; continuation seed2021; original complete two-stage objective and maximum20/20epochs, patience5, batch32 and published LR. Shared attention uses Values only from correction, no output bias, eta=sigmoid(logit) initially.5, zero output projection, bidirectional known-future mixing. Conv has nine bidirectional dilated layers, receptive field1023. C3 uses four unshared blocks; C4 uses width128 once. Parameters and actual latency are measured separately.

## 4. Actual Execution

Completed configurations below. B1 is mathematically identical to shared K1 and is reused as the K1 control, never falsely labeled separate training. Old baseline inference is kept separate from equally budgeted B0 continued training.

## 5. Raw Results

| run_id                     | dataset   |   horizon | variant   |   k |   seed |   validation_mae |   reproduced_mse |   reproduced_mae |   num_parameters |   train_seconds |
|:---------------------------|:----------|----------:|:----------|----:|-------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|
| NP_H24_B0_K1_s2021         | NP        |        24 | B0        |   1 |   2021 |         0.106693 |        0.164129  |         0.207332 |           345945 |         340.856 |
| NP_H24_B1_K1_s2021         | NP        |        24 | B1        |   1 |   2021 |         0.104927 |        0.162282  |         0.20569  |           404190 |         362.55  |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |   4 |   2021 |         0.107212 |        0.165772  |         0.20758  |           404190 |         408.956 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |   4 |   2021 |         0.213199 |        0.474227  |         0.443118 |          1084974 |         605.425 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |   4 |   2021 |         0.215158 |        0.46175   |         0.43719  |          1084974 |         628.455 |
| PJM_H24_B0_K1_s2021        | PJM       |        24 | B0        |   1 |   2021 |         0.109202 |        0.0566208 |         0.140923 |          1087481 |         321.885 |
| PJM_H24_B1_K1_s2021        | PJM       |        24 | B1        |   1 |   2021 |         0.108863 |        0.0560395 |         0.140117 |          1203774 |         301.402 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |   4 |   2021 |         0.109262 |        0.0563809 |         0.140548 |          1203774 |         339.132 |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |   4 |   2021 |         0.161695 |        0.113359  |         0.207015 |          2565006 |         631.591 |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |   4 |   2021 |         0.162514 |        0.11037   |         0.211632 |          2565006 |         568.8   |

## 6. Interpretation

NP_H24_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=1.87%. Validation MAE=0.106693; best Stage1 epoch=1/6, Stage2=6/11.

NP_H24_B1_K1_s2021: relative test MSE gain versus equally continued B0=1.13%, MAE gain=0.79%; versus verified original checkpoint MSE gain=2.97%. Validation MAE=0.104927; best Stage1 epoch=1/6, Stage2=6/11.

NP_H24_C5_K4_s2021: relative test MSE gain versus equally continued B0=-1.00%, MAE gain=-0.12%; versus verified original checkpoint MSE gain=0.89%. Validation MAE=0.107212; best Stage1 epoch=1/6, Stage2=6/11.

NP_H360_FULL_K4_s2021: relative test MSE gain versus equally continued B0=3.77%, MAE gain=1.77%; versus verified original checkpoint MSE gain=-10.18%. Validation MAE=0.213199; best Stage1 epoch=1/6, Stage2=13/18.

NP_H360_NOANCHOR_K4_s2021: relative test MSE gain versus equally continued B0=6.30%, MAE gain=3.08%; versus verified original checkpoint MSE gain=-7.29%. Validation MAE=0.215158; best Stage1 epoch=1/6, Stage2=14/19.

PJM_H24_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=1.56%. Validation MAE=0.109202; best Stage1 epoch=1/6, Stage2=3/8.

PJM_H24_B1_K1_s2021: relative test MSE gain versus equally continued B0=1.03%, MAE gain=0.57%; versus verified original checkpoint MSE gain=2.57%. Validation MAE=0.108863; best Stage1 epoch=1/6, Stage2=1/6.

PJM_H24_C5_K4_s2021: relative test MSE gain versus equally continued B0=0.42%, MAE gain=0.27%; versus verified original checkpoint MSE gain=1.97%. Validation MAE=0.109262; best Stage1 epoch=1/6, Stage2=1/6.

PJM_H360_FULL_K4_s2021: relative test MSE gain versus equally continued B0=11.09%, MAE gain=7.31%; versus verified original checkpoint MSE gain=1.50%. Validation MAE=0.161695; best Stage1 epoch=1/6, Stage2=5/10.

PJM_H360_NOANCHOR_K4_s2021: relative test MSE gain versus equally continued B0=13.44%, MAE gain=5.24%; versus verified original checkpoint MSE gain=4.09%. Validation MAE=0.162514; best Stage1 epoch=1/6, Stage2=3/8.

Actual unshared-stack Stage1 gradient observations:

| dataset   | response_block   |   observed_parameters |   parameters_with_positive_gradient |   min_median_parameter_gradient_norm |   max_median_parameter_gradient_norm |
|:----------|:-----------------|----------------------:|------------------------------------:|-------------------------------------:|-------------------------------------:|
| NP        | 0                |                     6 |                                   6 |                           0.00330895 |                            0.0411123 |
| NP        | 1                |                     6 |                                   6 |                           0.0118833  |                            0.203746  |
| NP        | 2                |                     6 |                                   6 |                           0.0427838  |                            0.429781  |
| NP        | 3                |                     6 |                                   6 |                           0.0989323  |                            0.836205  |
| NP        | gate_or_scalar   |                     1 |                                   1 |                           0.0744179  |                            0.0744179 |
| PJM       | 0                |                     6 |                                   6 |                           0.00260337 |                            0.0136878 |
| PJM       | 1                |                     6 |                                   6 |                           0.00141367 |                            0.0278482 |
| PJM       | 2                |                     6 |                                   6 |                           0.00856258 |                            0.151771  |
| PJM       | 3                |                     6 |                                   6 |                           0.0370408  |                            0.370719  |
| PJM       | gate_or_scalar   |                     1 |                                   1 |                           0.0250944  |                            0.0250944 |

All saved epoch means were checked finite. Positive observed gradients rule out wholly disconnected blocks; unequal gradient magnitudes do not by themselves prove the cause of an accuracy difference. First-step delayed gradients from zero output initialization remain an optimization difference. Complete per-parameter observations are in results/response_gradient_audit.csv.

Checkpoint propagation parameters:

| run_id                     | dataset   |   horizon | variant   |   seed | block              |        eta |   output_projection_frobenius_norm |   output_projection_initial_norm |   output_projection_changed_from_zero_initialization |   historical_loop_effective_coefficient |
|:---------------------------|:----------|----------:|:----------|-------:|:-------------------|-----------:|-----------------------------------:|---------------------------------:|-----------------------------------------------------:|----------------------------------------:|
| NP_H360_B1_K1_s2021        | NP        |       360 | B1        |   2021 | 0                  |   0.516346 |                            5.12256 |                                0 |                                                    1 |                             nan         |
| NP_H360_B2_K1_s2021        | NP        |       360 | B2        |   2021 | 0                  |   0.527826 |                            7.06792 |                                0 |                                                    1 |                             nan         |
| NP_H360_C1_K4_s2021        | NP        |       360 | C1        |   2021 | historical_encoder | nan        |                          nan       |                              nan |                                                  nan |                               0.0810307 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   2021 | 0                  |   0.507127 |                            3.04536 |                                0 |                                                    1 |                             nan         |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   2021 | 1                  |   0.507127 |                            2.7977  |                                0 |                                                    1 |                             nan         |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   2021 | 2                  |   0.507127 |                            3.70504 |                                0 |                                                    1 |                             nan         |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   2021 | 3                  |   0.507127 |                            5.08394 |                                0 |                                                    1 |                             nan         |
| NP_H360_C4_K1_s2021        | NP        |       360 | C4        |   2021 | 0                  |   0.499303 |                            5.78253 |                                0 |                                                    1 |                             nan         |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |   2021 | 0                  |   0.50014  |                            4.28002 |                                0 |                                                    1 |                             nan         |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |   2021 | 0                  |   0.4934   |                            3.9803  |                                0 |                                                    1 |                             nan         |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |   2021 | 0                  |   0.498879 |                            4.1264  |                                0 |                                                    1 |                             nan         |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |   2021 | 0                  |   0.469651 |                            3.11217 |                                0 |                                                    1 |                             nan         |
| PJM_H360_B1_K1_s2021       | PJM       |       360 | B1        |   2021 | 0                  |   0.489648 |                            4.38976 |                                0 |                                                    1 |                             nan         |
| PJM_H360_B2_K1_s2021       | PJM       |       360 | B2        |   2021 | 0                  |   0.531826 |                            9.78218 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C1_K4_s2021       | PJM       |       360 | C1        |   2021 | historical_encoder | nan        |                          nan       |                              nan |                                                  nan |                               0.0709938 |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   2021 | 0                  |   0.479333 |                            3.39478 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   2021 | 1                  |   0.479333 |                            2.35623 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   2021 | 2                  |   0.479333 |                            3.74364 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   2021 | 3                  |   0.479333 |                            4.32356 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C4_K1_s2021       | PJM       |       360 | C4        |   2021 | 0                  |   0.471398 |                            5.09657 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |   2021 | 0                  |   0.482643 |                            3.74246 |                                0 |                                                    1 |                             nan         |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |   2021 | 0                  |   0.486725 |                            3.96557 |                                0 |                                                    1 |                             nan         |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |   2021 | 0                  |   0.494056 |                            4.56466 |                                0 |                                                    1 |                             nan         |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |   2021 | 0                  |   0.451829 |                            3.14437 |                                0 |                                                    1 |                             nan         |

Nonzero final output-projection norms verify updates from their zero initialization. Eta and the historical-loop coefficient describe actual learned update strengths; finite coefficients alone do not prove useful prediction or contraction. Norms across differently sized blocks are descriptive, not a quality ranking.

Direct comparison judgments:

NP_H360_C5_K2_s2021 versus validation-selected nonrecurrent NP_H360_C4_K1_s2021: MSE gain=2.14%, MAE gain=1.79%. Both point estimates favor recurrence, pending seed/block/budget evidence.

NP_H360_C5_K4_s2021 versus validation-selected nonrecurrent NP_H360_C4_K1_s2021: MSE gain=2.24%, MAE gain=1.83%. Both point estimates favor recurrence, pending seed/block/budget evidence.

NP FULL versus anchored correction-only K4: MSE gain=1.70%, MAE gain=0.66%. Parameter count and iteration count are held fixed; this single-seed ablation cannot prove a general mechanism advantage.

NP NOANCHOR versus anchored correction-only K4: MSE gain=4.29%, MAE gain=1.99%. Parameter count and iteration count are held fixed; this single-seed ablation cannot prove a general mechanism advantage.

PJM_H360_C5_K2_s2021 versus validation-selected nonrecurrent PJM_H360_B2_K1_s2021: MSE gain=1.76%, MAE gain=-1.36%. The point estimates do not jointly favor recurrence on both metrics.

PJM_H360_C5_K4_s2021 versus validation-selected nonrecurrent PJM_H360_B2_K1_s2021: MSE gain=-0.08%, MAE gain=-1.13%. The point estimates do not jointly favor recurrence on both metrics.

PJM FULL versus anchored correction-only K4: MSE gain=6.68%, MAE gain=3.92%. Parameter count and iteration count are held fixed; this single-seed ablation cannot prove a general mechanism advantage.

PJM NOANCHOR versus anchored correction-only K4: MSE gain=9.14%, MAE gain=1.78%. Parameter count and iteration count are held fixed; this single-seed ablation cannot prove a general mechanism advantage.

Same checkpoint depth results (fixed decoder, no training between depths):

| run_id               | dataset   |   horizon |   seed |   trained_k |   test_k |      mse |      mae | same_checkpoint_sha256                                           |
|:---------------------|:----------|----------:|-------:|------------:|---------:|---------:|---------:|:-----------------------------------------------------------------|
| NP_H360_C5_K2_s2021  | NP        |       360 |   2021 |           2 |        1 | 0.481242 | 0.445533 | 1397317b200c14b4751b3fc45a13f81f82ecd9c1356da934d723a2978565dee1 |
| NP_H360_C5_K2_s2021  | NP        |       360 |   2021 |           2 |        2 | 0.482948 | 0.446245 | 1397317b200c14b4751b3fc45a13f81f82ecd9c1356da934d723a2978565dee1 |
| NP_H360_C5_K2_s2021  | NP        |       360 |   2021 |           2 |        4 | 0.48811  | 0.448166 | 1397317b200c14b4751b3fc45a13f81f82ecd9c1356da934d723a2978565dee1 |
| NP_H360_C5_K4_s2021  | NP        |       360 |   2021 |           4 |        1 | 0.480859 | 0.445073 | 059516f5c02b8c5239e0625431485d2ce349c2bfec9c9f46120bb9e4c001eb29 |
| NP_H360_C5_K4_s2021  | NP        |       360 |   2021 |           4 |        2 | 0.482041 | 0.446119 | 059516f5c02b8c5239e0625431485d2ce349c2bfec9c9f46120bb9e4c001eb29 |
| NP_H360_C5_K4_s2021  | NP        |       360 |   2021 |           4 |        4 | 0.482445 | 0.446074 | 059516f5c02b8c5239e0625431485d2ce349c2bfec9c9f46120bb9e4c001eb29 |
| PJM_H360_C5_K2_s2021 | PJM       |       360 |   2021 |           2 |        1 | 0.120441 | 0.219183 | ab6bf0504dbf86d21361a36c284c799eb857458ca7fa1a296758b4308f381752 |
| PJM_H360_C5_K2_s2021 | PJM       |       360 |   2021 |           2 |        2 | 0.119239 | 0.215955 | ab6bf0504dbf86d21361a36c284c799eb857458ca7fa1a296758b4308f381752 |
| PJM_H360_C5_K2_s2021 | PJM       |       360 |   2021 |           2 |        4 | 0.119242 | 0.216    | ab6bf0504dbf86d21361a36c284c799eb857458ca7fa1a296758b4308f381752 |
| PJM_H360_C5_K4_s2021 | PJM       |       360 |   2021 |           4 |        1 | 0.124691 | 0.218078 | 376a2ab8db4cf3f846643ec2890be565ff55048900362e6411376a3d374f5fbd |
| PJM_H360_C5_K4_s2021 | PJM       |       360 |   2021 |           4 |        2 | 0.121261 | 0.215543 | 376a2ab8db4cf3f846643ec2890be565ff55048900362e6411376a3d374f5fbd |
| PJM_H360_C5_K4_s2021 | PJM       |       360 |   2021 |           4 |        4 | 0.121478 | 0.21546  | 376a2ab8db4cf3f846643ec2890be565ff55048900362e6411376a3d374f5fbd |

NP_H360_C5_K2_s2021: K4 versus K1 MSE gain=-1.43%. More iterations do not improve this checkpoint. This remains a fixed-K decoder comparison, with possible state-distribution mismatch.

NP_H360_C5_K4_s2021: K4 versus K1 MSE gain=-0.33%. More iterations do not improve this checkpoint. This remains a fixed-K decoder comparison, with possible state-distribution mismatch.

PJM_H360_C5_K2_s2021: K4 versus K1 MSE gain=1.00%. More iterations improve this checkpoint. This remains a fixed-K decoder comparison, with possible state-distribution mismatch.

PJM_H360_C5_K4_s2021: K4 versus K1 MSE gain=2.58%. More iterations improve this checkpoint. This remains a fixed-K decoder comparison, with possible state-distribution mismatch.

Preregistered no-anchor alternative versus the primary control:

| dataset   | ablation                   | validation_selected_control   |   validation_mae_gain |   test_mse_gain |   test_mae_gain |
|:----------|:---------------------------|:------------------------------|----------------------:|----------------:|----------------:|
| NP        | NP_H360_NOANCHOR_K4_s2021  | NP_H360_C4_K1_s2021           |            -0.0312921 |       0.064356  |      0.0378074  |
| PJM       | PJM_H360_NOANCHOR_K4_s2021 | PJM_H360_B2_K1_s2021          |            -0.0115135 |       0.0906883 |      0.00669719 |

These fixed ablations can identify a competing update rule even when the anchored hypothesis is unsupported. A favorable single-seed test difference does not replace the frozen primary C5 comparison or its validation-only expansion gate. Confirming a new primary no-anchor hypothesis would require a new prospective selection and independent validation protocol.

Future-covariate evaluation sensitivity:

| run_id               | dataset   |   horizon | variant   |   k | condition   |       mse |      mae | scope                                                                        | noise_units                           |
|:---------------------|:----------|----------:|:----------|----:|:------------|----------:|---------:|:-----------------------------------------------------------------------------|:--------------------------------------|
| NP_H24_B0_K1_s2021   | NP        |        24 | B0        |   1 | full        | 0.164129  | 0.207332 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_B0_K1_s2021   | NP        |        24 | B0        |   1 | mask_first  | 1.72599   | 0.91933  | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_B0_K1_s2021   | NP        |        24 | B0        |   1 | no_future   | 0.538072  | 0.46059  | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_B0_K1_s2021   | NP        |        24 | B0        |   1 | noise_0.25  | 0.266774  | 0.299456 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| NP_H24_B1_K1_s2021   | NP        |        24 | B1        |   1 | full        | 0.162282  | 0.20569  | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_B1_K1_s2021   | NP        |        24 | B1        |   1 | mask_first  | 1.63535   | 0.888232 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_B1_K1_s2021   | NP        |        24 | B1        |   1 | no_future   | 0.537276  | 0.463938 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_B1_K1_s2021   | NP        |        24 | B1        |   1 | noise_0.25  | 0.257677  | 0.290764 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| NP_H24_C5_K4_s2021   | NP        |        24 | C5        |   4 | full        | 0.165772  | 0.20758  | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_C5_K4_s2021   | NP        |        24 | C5        |   4 | mask_first  | 1.78723   | 0.921794 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_C5_K4_s2021   | NP        |        24 | C5        |   4 | no_future   | 0.535746  | 0.461596 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H24_C5_K4_s2021   | NP        |        24 | C5        |   4 | noise_0.25  | 0.272587  | 0.298186 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| NP_H360_B0_K1_s2021  | NP        |       360 | B0        |   1 | full        | 0.492817  | 0.451098 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_B0_K1_s2021  | NP        |       360 | B0        |   1 | mask_first  | 1.49642   | 0.881212 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_B0_K1_s2021  | NP        |       360 | B0        |   1 | no_future   | 0.924297  | 0.637656 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_B0_K1_s2021  | NP        |       360 | B0        |   1 | noise_0.25  | 0.6294    | 0.518556 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| NP_H360_B1_K1_s2021  | NP        |       360 | B1        |   1 | full        | 0.483778  | 0.445734 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_B1_K1_s2021  | NP        |       360 | B1        |   1 | mask_first  | 1.70454   | 0.944797 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_B1_K1_s2021  | NP        |       360 | B1        |   1 | no_future   | 0.984917  | 0.669394 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_B1_K1_s2021  | NP        |       360 | B1        |   1 | noise_0.25  | 0.620901  | 0.512393 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| NP_H360_C5_K2_s2021  | NP        |       360 | C5        |   2 | full        | 0.482948  | 0.446245 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_C5_K2_s2021  | NP        |       360 | C5        |   2 | mask_first  | 1.76846   | 0.969619 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_C5_K2_s2021  | NP        |       360 | C5        |   2 | no_future   | 0.947696  | 0.649473 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_C5_K2_s2021  | NP        |       360 | C5        |   2 | noise_0.25  | 0.622968  | 0.51182  | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| NP_H360_C5_K4_s2021  | NP        |       360 | C5        |   4 | full        | 0.482445  | 0.446074 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_C5_K4_s2021  | NP        |       360 | C5        |   4 | mask_first  | 1.73771   | 0.96331  | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_C5_K4_s2021  | NP        |       360 | C5        |   4 | no_future   | 0.927367  | 0.64052  | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| NP_H360_C5_K4_s2021  | NP        |       360 | C5        |   4 | noise_0.25  | 0.619999  | 0.513687 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| PJM_H24_B0_K1_s2021  | PJM       |        24 | B0        |   1 | full        | 0.0566208 | 0.140923 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_B0_K1_s2021  | PJM       |        24 | B0        |   1 | mask_first  | 0.245486  | 0.327338 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_B0_K1_s2021  | PJM       |        24 | B0        |   1 | no_future   | 0.256554  | 0.368687 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_B0_K1_s2021  | PJM       |        24 | B0        |   1 | noise_0.25  | 0.113845  | 0.207207 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| PJM_H24_B1_K1_s2021  | PJM       |        24 | B1        |   1 | full        | 0.0560395 | 0.140117 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_B1_K1_s2021  | PJM       |        24 | B1        |   1 | mask_first  | 0.240331  | 0.323834 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_B1_K1_s2021  | PJM       |        24 | B1        |   1 | no_future   | 0.268974  | 0.377841 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_B1_K1_s2021  | PJM       |        24 | B1        |   1 | noise_0.25  | 0.111276  | 0.206704 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| PJM_H24_C5_K4_s2021  | PJM       |        24 | C5        |   4 | full        | 0.0563809 | 0.140548 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_C5_K4_s2021  | PJM       |        24 | C5        |   4 | mask_first  | 0.246811  | 0.329092 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_C5_K4_s2021  | PJM       |        24 | C5        |   4 | no_future   | 0.282134  | 0.387453 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H24_C5_K4_s2021  | PJM       |        24 | C5        |   4 | noise_0.25  | 0.106311  | 0.20239  | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| PJM_H360_B0_K1_s2021 | PJM       |       360 | B0        |   1 | full        | 0.127503  | 0.223339 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_B0_K1_s2021 | PJM       |       360 | B0        |   1 | mask_first  | 0.255231  | 0.321764 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_B0_K1_s2021 | PJM       |       360 | B0        |   1 | no_future   | 0.32144   | 0.393175 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_B0_K1_s2021 | PJM       |       360 | B0        |   1 | noise_0.25  | 1.2014    | 0.437349 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| PJM_H360_B1_K1_s2021 | PJM       |       360 | B1        |   1 | full        | 0.12047   | 0.2179   | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_B1_K1_s2021 | PJM       |       360 | B1        |   1 | mask_first  | 0.240238  | 0.311143 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_B1_K1_s2021 | PJM       |       360 | B1        |   1 | no_future   | 0.316202  | 0.392654 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_B1_K1_s2021 | PJM       |       360 | B1        |   1 | noise_0.25  | 1.40564   | 0.486594 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| PJM_H360_C5_K2_s2021 | PJM       |       360 | C5        |   2 | full        | 0.119239  | 0.215955 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_C5_K2_s2021 | PJM       |       360 | C5        |   2 | mask_first  | 0.246751  | 0.316375 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_C5_K2_s2021 | PJM       |       360 | C5        |   2 | no_future   | 0.312657  | 0.388607 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_C5_K2_s2021 | PJM       |       360 | C5        |   2 | noise_0.25  | 0.910705  | 0.462166 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |
| PJM_H360_C5_K4_s2021 | PJM       |       360 | C5        |   4 | full        | 0.121478  | 0.21546  | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_C5_K4_s2021 | PJM       |       360 | C5        |   4 | mask_first  | 0.231479  | 0.30765  | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_C5_K4_s2021 | PJM       |       360 | C5        |   4 | no_future   | 0.317048  | 0.389778 | evaluation sensitivity only, not a retrained ablation or causal intervention | nan                                   |
| PJM_H360_C5_K4_s2021 | PJM       |       360 | C5        |   4 | noise_0.25  | 0.938505  | 0.428057 | evaluation sensitivity only, not a retrained ablation or causal intervention | 0.25 train-scaled standard deviations |

Per-iteration validation diagnostics:

| run_id                     | dataset   |   horizon | variant   |   trained_k |   iteration |   delta_l2_mean |   delta_rms |   step_l2_mean |   validation_sample_mse |   validation_sample_mae |   validation_latent_l1 |   origins_count |
|:---------------------------|:----------|----------:|:----------|------------:|------------:|----------------:|------------:|---------------:|------------------------:|------------------------:|-----------------------:|----------------:|
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |           4 |           0 |         25.0853 |    0.454815 |     0          |               0.0119581 |               0.0792301 |               0.158794 |              32 |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |           4 |           1 |         25.6446 |    0.464784 |     2.82061    |               0.011252  |               0.0762652 |               0.159391 |              32 |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |           4 |           2 |         25.634  |    0.464683 |     1.81118    |               0.0114604 |               0.0764179 |               0.156992 |              32 |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |           4 |           3 |         25.6351 |    0.464642 |     0.973421   |               0.0113494 |               0.076107  |               0.157615 |              32 |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |           4 |           4 |         25.6275 |    0.464539 |     0.515927   |               0.0113802 |               0.0761716 |               0.157138 |              32 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |           4 |           0 |         74.4272 |    0.352159 |     0          |               0.0912987 |               0.208916  |               0.264693 |              32 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |           4 |           1 |         70.7239 |    0.337353 |    24.905      |               0.0863019 |               0.200402  |               0.289065 |              32 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |           4 |           2 |         72.0715 |    0.34224  |    27.1029     |               0.0903522 |               0.208754  |               0.274805 |              32 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |           4 |           3 |         74.8613 |    0.36025  |    33.8151     |               0.0936195 |               0.213495  |               0.298814 |              32 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |           4 |           4 |         94.2909 |    0.454292 |    45.466      |               0.0859213 |               0.205375  |               0.261236 |              32 |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |           2 |           0 |         75.9507 |    0.359941 |     0          |               0.0938793 |               0.210563  |               0.260132 |              32 |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |           2 |           1 |         89.0983 |    0.42839  |    21.4891     |               0.0891188 |               0.206548  |               0.257896 |              32 |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |           2 |           2 |         94.5805 |    0.456195 |    12.6179     |               0.09019   |               0.210233  |               0.262067 |              32 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |           4 |           0 |         79.4019 |    0.376453 |     0          |               0.093836  |               0.210465  |               0.259486 |              32 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |           4 |           1 |         89.4326 |    0.429334 |    18.1904     |               0.0901405 |               0.208584  |               0.259692 |              32 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |           4 |           2 |         91.9744 |    0.442501 |     8.62827    |               0.0910091 |               0.211156  |               0.262174 |              32 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |           4 |           3 |         92.9503 |    0.447377 |     4.76716    |               0.0909922 |               0.211501  |               0.263195 |              32 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |           4 |           4 |         93.1768 |    0.448075 |     3.20337    |               0.0909862 |               0.211602  |               0.263401 |              32 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |           4 |           0 |         75.6509 |    0.358979 |     0          |               0.0946725 |               0.21266   |               0.260059 |              32 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |           4 |           1 |         86.6619 |    0.416022 |    20.0122     |               0.0878526 |               0.205254  |               0.256588 |              32 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |           4 |           2 |         89.8106 |    0.432716 |     9.13323    |               0.0886079 |               0.206456  |               0.258607 |              32 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |           4 |           3 |         90.5542 |    0.435244 |     4.7858     |               0.0886817 |               0.207061  |               0.259159 |              32 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |           4 |           4 |         91.0503 |    0.437873 |     2.92903    |               0.0885831 |               0.207015  |               0.259554 |              32 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |           4 |           0 |         69.5249 |    0.328185 |     0          |               0.10137   |               0.219507  |               0.266879 |              32 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |           4 |           1 |         73.0283 |    0.345746 |     6.68469    |               0.0958502 |               0.212436  |               0.261592 |              32 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |           4 |           2 |         77.4749 |    0.368311 |     7.37906    |               0.0916266 |               0.207491  |               0.257983 |              32 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |           4 |           3 |         82.9058 |    0.396101 |     8.30256    |               0.0893059 |               0.205213  |               0.256966 |              32 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |           4 |           4 |         89.4266 |    0.429724 |     9.47153    |               0.0896982 |               0.207179  |               0.259777 |              32 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |           4 |           0 |         30.9364 |    0.425567 |     0          |               0.0267612 |               0.0956599 |               0.13168  |              32 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |           4 |           1 |         31.0377 |    0.426928 |     1.26881    |               0.0261011 |               0.0937065 |               0.130786 |              32 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |           4 |           2 |         31.041  |    0.426974 |     0.131348   |               0.0261262 |               0.0937228 |               0.130778 |              32 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |           4 |           3 |         31.0402 |    0.426964 |     0.019881   |               0.0261239 |               0.0937137 |               0.130773 |              32 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |           4 |           4 |         31.0404 |    0.426967 |     0.00440626 |               0.0261239 |               0.0937158 |               0.130772 |              32 |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |           4 |           0 |        110.356  |    0.393981 |     0          |               0.0729084 |               0.182113  |               0.223656 |              32 |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |           4 |           1 |        141.738  |    0.506094 |    85.8625     |               0.0981776 |               0.230515  |               0.343916 |              32 |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |           4 |           2 |        109.536  |    0.391176 |    86.7061     |               0.0720054 |               0.182818  |               0.233557 |              32 |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |           4 |           3 |        108.028  |    0.386073 |    38.8191     |               0.0692592 |               0.175063  |               0.252235 |              32 |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |           4 |           4 |        116.97   |    0.419154 |    41.6462     |               0.0661695 |               0.171164  |               0.218984 |              32 |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |           2 |           0 |        115.24   |    0.412538 |     0          |               0.0670114 |               0.174675  |               0.218792 |              32 |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |           2 |           1 |        120.231  |    0.431151 |    20.4753     |               0.0662348 |               0.163035  |               0.214181 |              32 |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |           2 |           2 |        121.019  |    0.433995 |     8.8582     |               0.072742  |               0.168685  |               0.215718 |              32 |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |           4 |           0 |        114.371  |    0.40869  |     0          |               0.0635451 |               0.170464  |               0.219204 |              32 |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |           4 |           1 |        120.304  |    0.431731 |    23.1051     |               0.0766165 |               0.164633  |               0.216497 |              32 |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |           4 |           2 |        119.783  |    0.429366 |    10.4326     |               0.066445  |               0.165766  |               0.216197 |              32 |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |           4 |           3 |        119.856  |    0.429633 |     5.77059    |               0.064665  |               0.162883  |               0.216012 |              32 |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |           4 |           4 |        119.986  |    0.430309 |     4.32707    |               0.0676576 |               0.165413  |               0.216238 |              32 |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |           4 |           0 |        107.792  |    0.384972 |     0          |               0.0724989 |               0.175802  |               0.221625 |              32 |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |           4 |           1 |        114.898  |    0.410797 |    18.4427     |               0.0691402 |               0.162477  |               0.214847 |              32 |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |           4 |           2 |        115.289  |    0.412428 |     5.80066    |               0.068258  |               0.163852  |               0.214809 |              32 |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |           4 |           3 |        115.506  |    0.413338 |     2.45606    |               0.068893  |               0.163417  |               0.214702 |              32 |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |           4 |           4 |        115.489  |    0.413262 |     1.39466    |               0.0683982 |               0.163784  |               0.214872 |              32 |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |           4 |           0 |        105.853  |    0.377195 |     0          |               0.0666077 |               0.176549  |               0.217231 |              32 |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |           4 |           1 |        108.204  |    0.385745 |     5.87983    |               0.0633011 |               0.171141  |               0.213538 |              32 |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |           4 |           2 |        111.032  |    0.396144 |     6.06784    |               0.0610214 |               0.16697   |               0.211042 |              32 |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |           4 |           3 |        114.324  |    0.40837  |     6.30045    |               0.0598935 |               0.164312  |               0.20986  |              32 |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |           4 |           4 |        118.076  |    0.422433 |     6.58455    |               0.0600454 |               0.163193  |               0.210122 |              32 |

Response outside the original decoder support:

| run_id                     |   output_position |   mean_gradient_norm_outside_baseline_window |   max_gradient_norm_outside_baseline_window |   n_origins |
|:---------------------------|------------------:|---------------------------------------------:|--------------------------------------------:|------------:|
| NP_H360_B0_K1_s2021        |                 0 |                                   0          |                                  0          |          32 |
| NP_H360_B0_K1_s2021        |                90 |                                   0          |                                  0          |          32 |
| NP_H360_B0_K1_s2021        |               180 |                                   0          |                                  0          |          32 |
| NP_H360_B0_K1_s2021        |               359 |                                   0          |                                  0          |          32 |
| NP_H360_B1_K1_s2021        |                 0 |                                   0.00120937 |                                  0.00570589 |          32 |
| NP_H360_B1_K1_s2021        |                90 |                                   0.00242835 |                                  0.0107837  |          32 |
| NP_H360_B1_K1_s2021        |               180 |                                   0.0024852  |                                  0.00888988 |          32 |
| NP_H360_B1_K1_s2021        |               359 |                                   0.00261563 |                                  0.00944607 |          32 |
| NP_H360_B2_K1_s2021        |                 0 |                                   0.00122072 |                                  0.0670015  |          32 |
| NP_H360_B2_K1_s2021        |                90 |                                   0.00251955 |                                  0.101462   |          32 |
| NP_H360_B2_K1_s2021        |               180 |                                   0.00264289 |                                  0.112553   |          32 |
| NP_H360_B2_K1_s2021        |               359 |                                   0.00147877 |                                  0.0321108  |          32 |
| NP_H360_C1_K4_s2021        |                 0 |                                   0          |                                  0          |          32 |
| NP_H360_C1_K4_s2021        |                90 |                                   0          |                                  0          |          32 |
| NP_H360_C1_K4_s2021        |               180 |                                   0          |                                  0          |          32 |
| NP_H360_C1_K4_s2021        |               359 |                                   0          |                                  0          |          32 |
| NP_H360_C3_K4_s2021        |                 0 |                                   0.00297542 |                                  0.0196401  |          32 |
| NP_H360_C3_K4_s2021        |                90 |                                   0.00604062 |                                  0.0185427  |          32 |
| NP_H360_C3_K4_s2021        |               180 |                                   0.00645122 |                                  0.0236861  |          32 |
| NP_H360_C3_K4_s2021        |               359 |                                   0.00688286 |                                  0.0360123  |          32 |
| NP_H360_C4_K1_s2021        |                 0 |                                   0.00164607 |                                  0.0076893  |          32 |
| NP_H360_C4_K1_s2021        |                90 |                                   0.00372283 |                                  0.0432444  |          32 |
| NP_H360_C4_K1_s2021        |               180 |                                   0.0033629  |                                  0.0135105  |          32 |
| NP_H360_C4_K1_s2021        |               359 |                                   0.00365011 |                                  0.0157655  |          32 |
| NP_H360_C5_K2_s2021        |                 0 |                                   0.00123447 |                                  0.00558577 |          32 |
| NP_H360_C5_K2_s2021        |                90 |                                   0.00353245 |                                  0.0183916  |          32 |
| NP_H360_C5_K2_s2021        |               180 |                                   0.00331315 |                                  0.0222703  |          32 |
| NP_H360_C5_K2_s2021        |               359 |                                   0.00324641 |                                  0.0109331  |          32 |
| NP_H360_C5_K4_s2021        |                 0 |                                   0.00146183 |                                  0.00942828 |          32 |
| NP_H360_C5_K4_s2021        |                90 |                                   0.00336393 |                                  0.0213933  |          32 |
| NP_H360_C5_K4_s2021        |               180 |                                   0.00300185 |                                  0.0150215  |          32 |
| NP_H360_C5_K4_s2021        |               359 |                                   0.00322347 |                                  0.00974272 |          32 |
| NP_H360_FULL_K4_s2021      |                 0 |                                   0.00160851 |                                  0.00852182 |          32 |
| NP_H360_FULL_K4_s2021      |                90 |                                   0.00365785 |                                  0.0399145  |          32 |
| NP_H360_FULL_K4_s2021      |               180 |                                   0.00318214 |                                  0.0123532  |          32 |
| NP_H360_FULL_K4_s2021      |               359 |                                   0.00374666 |                                  0.0256976  |          32 |
| NP_H360_NOANCHOR_K4_s2021  |                 0 |                                   0.002039   |                                  0.0100489  |          32 |
| NP_H360_NOANCHOR_K4_s2021  |                90 |                                   0.00411548 |                                  0.0340223  |          32 |
| NP_H360_NOANCHOR_K4_s2021  |               180 |                                   0.0037988  |                                  0.0185673  |          32 |
| NP_H360_NOANCHOR_K4_s2021  |               359 |                                   0.00384178 |                                  0.0118648  |          32 |
| PJM_H360_B0_K1_s2021       |                 0 |                                   0          |                                  0          |          32 |
| PJM_H360_B0_K1_s2021       |                90 |                                   0          |                                  0          |          32 |
| PJM_H360_B0_K1_s2021       |               180 |                                   0          |                                  0          |          32 |
| PJM_H360_B0_K1_s2021       |               359 |                                   0          |                                  0          |          32 |
| PJM_H360_B1_K1_s2021       |                 0 |                                   0.00365207 |                                  0.0687255  |          32 |
| PJM_H360_B1_K1_s2021       |                90 |                                   0.00496539 |                                  0.0206768  |          32 |
| PJM_H360_B1_K1_s2021       |               180 |                                   0.00525761 |                                  0.0174386  |          32 |
| PJM_H360_B1_K1_s2021       |               359 |                                   0.00552182 |                                  0.0144614  |          32 |
| PJM_H360_B2_K1_s2021       |                 0 |                                   0.00318426 |                                  0.293518   |          32 |
| PJM_H360_B2_K1_s2021       |                90 |                                   0.00435245 |                                  0.207876   |          32 |
| PJM_H360_B2_K1_s2021       |               180 |                                   0.00479975 |                                  0.210028   |          32 |
| PJM_H360_B2_K1_s2021       |               359 |                                   0.00217962 |                                  0.0190468  |          32 |
| PJM_H360_C1_K4_s2021       |                 0 |                                   0          |                                  0          |          32 |
| PJM_H360_C1_K4_s2021       |                90 |                                   0          |                                  0          |          32 |
| PJM_H360_C1_K4_s2021       |               180 |                                   0          |                                  0          |          32 |
| PJM_H360_C1_K4_s2021       |               359 |                                   0          |                                  0          |          32 |
| PJM_H360_C3_K4_s2021       |                 0 |                                   0.0132362  |                                  0.284845   |          32 |
| PJM_H360_C3_K4_s2021       |                90 |                                   0.0104714  |                                  0.0482518  |          32 |
| PJM_H360_C3_K4_s2021       |               180 |                                   0.0117919  |                                  0.0343181  |          32 |
| PJM_H360_C3_K4_s2021       |               359 |                                   0.0112768  |                                  0.03512    |          32 |
| PJM_H360_C4_K1_s2021       |                 0 |                                   0.003702   |                                  0.0808943  |          32 |
| PJM_H360_C4_K1_s2021       |                90 |                                   0.00496263 |                                  0.0159979  |          32 |
| PJM_H360_C4_K1_s2021       |               180 |                                   0.00592248 |                                  0.023579   |          32 |
| PJM_H360_C4_K1_s2021       |               359 |                                   0.00583739 |                                  0.0254507  |          32 |
| PJM_H360_C5_K2_s2021       |                 0 |                                   0.00400782 |                                  0.0306376  |          32 |
| PJM_H360_C5_K2_s2021       |                90 |                                   0.0058641  |                                  0.0173588  |          32 |
| PJM_H360_C5_K2_s2021       |               180 |                                   0.00608386 |                                  0.0211915  |          32 |
| PJM_H360_C5_K2_s2021       |               359 |                                   0.00588204 |                                  0.0181061  |          32 |
| PJM_H360_C5_K4_s2021       |                 0 |                                   0.00381675 |                                  0.0229502  |          32 |
| PJM_H360_C5_K4_s2021       |                90 |                                   0.00565007 |                                  0.0217512  |          32 |
| PJM_H360_C5_K4_s2021       |               180 |                                   0.00696498 |                                  0.0221823  |          32 |
| PJM_H360_C5_K4_s2021       |               359 |                                   0.00684886 |                                  0.0276151  |          32 |
| PJM_H360_FULL_K4_s2021     |                 0 |                                   0.00282968 |                                  0.00740906 |          32 |
| PJM_H360_FULL_K4_s2021     |                90 |                                   0.00539817 |                                  0.0145327  |          32 |
| PJM_H360_FULL_K4_s2021     |               180 |                                   0.00594722 |                                  0.0131492  |          32 |
| PJM_H360_FULL_K4_s2021     |               359 |                                   0.00623657 |                                  0.0234052  |          32 |
| PJM_H360_NOANCHOR_K4_s2021 |                 0 |                                   0.00454898 |                                  0.0792052  |          32 |
| PJM_H360_NOANCHOR_K4_s2021 |                90 |                                   0.00519279 |                                  0.0315235  |          32 |
| PJM_H360_NOANCHOR_K4_s2021 |               180 |                                   0.00633784 |                                  0.0154808  |          32 |
| PJM_H360_NOANCHOR_K4_s2021 |               359 |                                   0.00625909 |                                  0.0173793  |          32 |

Nonzero off-window sensitivity confirms that a learned cross-time pathway is used; it does not establish that this pathway improves prediction. Finite stable norm sequences alone do not prove forecasting value. Quality changes are robustness probes with unchanged weights, not retrained ablations or causal interventions.

Measured costs:

| run_id                     | dataset   |   horizon | variant   |   k |   seed |   parameters |   training_seconds | training_time_scope                                                               |   process_attempt_wall_seconds |   failed_attempts |   peak_gpu_mb |   batch1_ms_median |   batch1_ms_p90 |   batch32_ms_median |   batch32_ms_p90 |
|:---------------------------|:----------|----------:|:----------|----:|-------:|-------------:|-------------------:|:----------------------------------------------------------------------------------|-------------------------------:|------------------:|--------------:|-------------------:|----------------:|--------------------:|-----------------:|
| NP_H24_B0_K1_s2021         | NP        |        24 | B0        |   1 |   2021 |       345945 |            340.856 | retained/resumed optimization including validation; failed partial epoch excluded |                        372.972 |                 0 |       275.64  |            4.7897  |         4.8833  |             4.99563 |          5.08548 |
| NP_H24_B1_K1_s2021         | NP        |        24 | B1        |   1 |   2021 |       404190 |            362.55  | retained/resumed optimization including validation; failed partial epoch excluded |                        394.973 |                 0 |       277.509 |            5.3287  |         5.52918 |             5.48747 |          5.60719 |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |   4 |   2021 |       404190 |            408.956 | retained/resumed optimization including validation; failed partial epoch excluded |                        442.121 |                 0 |       278.634 |            6.55069 |         6.67871 |             6.83038 |          6.98831 |
| NP_H360_B0_K1_s2021        | NP        |       360 | B0        |   1 |   2021 |      1026729 |            497.51  | retained/resumed optimization including validation; failed partial epoch excluded |                        531.594 |                 0 |      1352.24  |            4.84102 |         4.98212 |             6.28235 |          6.54088 |
| NP_H360_B1_K1_s2021        | NP        |       360 | B1        |   1 |   2021 |      1084974 |            443.337 | retained/resumed optimization including validation; failed partial epoch excluded |                        510.93  |                 1 |      1367.77  |            5.32664 |         5.54078 |             6.82717 |          7.00282 |
| NP_H360_B2_K1_s2021        | NP        |       360 | B2        |   1 |   2021 |      1084714 |            484.113 | retained/resumed optimization including validation; failed partial epoch excluded |                        523.77  |                 0 |      1366.61  |            5.81003 |         5.87871 |             7.27736 |          7.50081 |
| NP_H360_C1_K4_s2021        | NP        |       360 | C1        |   4 |   2021 |      1026730 |            562.451 | retained/resumed optimization including validation; failed partial epoch excluded |                        605.196 |                 0 |      3496.09  |            7.5223  |         7.62686 |            11.6809  |         12.2495  |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   4 |   2021 |      1259706 |            414.339 | retained/resumed optimization including validation; failed partial epoch excluded |                        454.619 |                 0 |      1388.6   |            6.62446 |         6.719   |             8.21768 |          8.38026 |
| NP_H360_C4_K1_s2021        | NP        |       360 | C4        |   1 |   2021 |      1142574 |            586.744 | retained/resumed optimization including validation; failed partial epoch excluded |                        626.562 |                 0 |      1367.67  |            5.36277 |         5.58045 |             6.84558 |          7.14604 |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |   2 |   2021 |      1084974 |            566.637 | retained/resumed optimization including validation; failed partial epoch excluded |                        606.37  |                 0 |      1374.24  |            5.75504 |         5.83049 |             7.34781 |          7.66894 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |   4 |   2021 |      1084974 |            494.744 | retained/resumed optimization including validation; failed partial epoch excluded |                        534.817 |                 0 |      1385.25  |            6.62907 |         6.76481 |             8.25526 |          8.45137 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |   4 |   2021 |      1084974 |            605.425 | retained/resumed optimization including validation; failed partial epoch excluded |                        645.581 |                 0 |      1385.42  |            6.81366 |         6.91846 |             8.21773 |          8.40072 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |   4 |   2021 |      1084974 |            628.455 | retained/resumed optimization including validation; failed partial epoch excluded |                        668.613 |                 0 |      1385.25  |            6.68781 |         6.88539 |             8.21891 |          8.37652 |
| PJM_H24_B0_K1_s2021        | PJM       |        24 | B0        |   1 |   2021 |      1087481 |            321.885 | retained/resumed optimization including validation; failed partial epoch excluded |                        355.305 |                 0 |       819.013 |            4.78787 |         4.9054  |             6.85574 |          7.132   |
| PJM_H24_B1_K1_s2021        | PJM       |        24 | B1        |   1 |   2021 |      1203774 |            301.402 | retained/resumed optimization including validation; failed partial epoch excluded |                        335.176 |                 0 |       822.365 |            5.27333 |         5.50914 |             7.35502 |          7.57155 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |   4 |   2021 |      1203774 |            339.132 | retained/resumed optimization including validation; failed partial epoch excluded |                        373.413 |                 0 |       824.428 |            6.60075 |         6.80572 |             8.59877 |          8.8858  |
| PJM_H360_B0_K1_s2021       | PJM       |       360 | B0        |   1 |   2021 |      2448713 |            642.101 | retained/resumed optimization including validation; failed partial epoch excluded |                        685.099 |                 0 |      4928.02  |            4.8533  |         5.01825 |            12.3261  |         12.6661  |
| PJM_H360_B1_K1_s2021       | PJM       |       360 | B1        |   1 |   2021 |      2565006 |            492.414 | retained/resumed optimization including validation; failed partial epoch excluded |                        535.505 |                 0 |      4950.33  |            5.3484  |         5.50171 |            12.8614  |         13.3615  |
| PJM_H360_B2_K1_s2021       | PJM       |       360 | B2        |   1 |   2021 |      2558922 |            564.541 | retained/resumed optimization including validation; failed partial epoch excluded |                        607.96  |                 0 |      4949.55  |            5.8375  |         5.93966 |            13.2489  |         13.59    |
| PJM_H360_C1_K4_s2021       | PJM       |       360 | C1        |   4 |   2021 |      2448714 |           1883.59  | retained/resumed optimization including validation; failed partial epoch excluded |                       1939.45  |                 0 |     11771.9   |            7.48936 |         7.83046 |            32.8504  |         33.3961  |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   4 |   2021 |      2913882 |            633.648 | retained/resumed optimization including validation; failed partial epoch excluded |                        678.612 |                 0 |      4989.32  |            6.64261 |         6.87483 |            14.3252  |         14.863   |
| PJM_H360_C4_K1_s2021       | PJM       |       360 | C4        |   1 |   2021 |      2679950 |            551.297 | retained/resumed optimization including validation; failed partial epoch excluded |                        594.39  |                 0 |      4954.23  |            5.35006 |         5.51689 |            12.93    |         13.4886  |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |   2 |   2021 |      2565006 |            509.657 | retained/resumed optimization including validation; failed partial epoch excluded |                        553.049 |                 0 |      4961.68  |            5.824   |         5.99645 |            13.3442  |         13.6558  |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |   4 |   2021 |      2565006 |            599.892 | retained/resumed optimization including validation; failed partial epoch excluded |                        643.892 |                 0 |      4982.65  |            6.67038 |         6.76193 |            14.2866  |         14.9325  |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |   4 |   2021 |      2565006 |            631.591 | retained/resumed optimization including validation; failed partial epoch excluded |                        675.571 |                 0 |      4983.41  |            6.75728 |         6.87882 |            14.3527  |         15.0718  |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |   4 |   2021 |      2565006 |            568.8   | retained/resumed optimization including validation; failed partial epoch excluded |                        612.747 |                 0 |      4981.5   |            6.66566 |         6.80171 |            14.4157  |         14.9812  |

A lower test value is not a reason to change the architecture or gate. Whole-model parameter and actual latency mismatches limit efficiency claims. No result is attributed to sharing before the appropriate controls are compared.

## 7. Scientific Contribution

The anchored/no-anchor and correction/full-state pairs change one update choice while keeping width and K fixed. Their measured MSE/MAE differences above test whether either design choice improves this continuation setup. Parameter sharing is evaluated alongside C3 and wider controls. The short/long and covariate-quality tables delimit the observed scope; stable norms or nonzero distant sensitivity alone do not count as forecasting success. Masking/noise are sensitivity probes, not causal evidence.

## 8. Limitations and Alternative Explanations

Main family is checkpoint-initialized continuation, not from-scratch. Each continuation seed starts from the same original trained weights. Fixed-K decoders may mismatch states at other test depths. Masking/noise ablations are evaluation robustness tests, not causal interventions.

## 9. Decision

Proceed to the next minimum stage regardless of sign. Expand seeds only through the preregistered validation gate.

## 10. Reproduction

python scripts/train.py --run-id <preregisteredID>; python scripts/orchestrate.py. Per-run research_spec/config/command/history/freeze audit/checkpoint/predictions and numbered stdout attempts are saved.


## Stage5

# STAGE5_STATISTICAL_SYNTHESIS.md

## 1. Research Question

What conclusion survives structural, accuracy, sharing, depth, cost and uncertainty controls?

## 2. Pre-experiment Hypothesis

Loop success requires two-task gains over strongest reasonable nonrecurrent controls, about3%meanMSE improvement, no MAE loss, about1%same-checkpoint K4 overK1, consistent seed direction, and meaningful measured budgets.

## 3. Experiment Design

Preregistered validation selection and expansion; every rolling test origin independently verified; paired per-seed differences;2000 moving-block bootstrap draws with three block lengths; measured batch1/32 latency.

## 4. Actual Execution

All mandatory Stage0–4 experiments executed; optional expansion decision: False. Any optional NOT_RUN is shown in the matrix, without fabricated scores.

## 5. Raw Results

| run_id                     | dataset   |   horizon | variant   |   k |   seed |   validation_mae |   reproduced_mse |   reproduced_mae |   num_parameters |   train_seconds |
|:---------------------------|:----------|----------:|:----------|----:|-------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|
| NP_H24_B0_K1_s2021         | NP        |        24 | B0        |   1 |   2021 |         0.106693 |        0.164129  |         0.207332 |           345945 |         340.856 |
| NP_H24_B1_K1_s2021         | NP        |        24 | B1        |   1 |   2021 |         0.104927 |        0.162282  |         0.20569  |           404190 |         362.55  |
| NP_H24_C5_K4_s2021         | NP        |        24 | C5        |   4 |   2021 |         0.107212 |        0.165772  |         0.20758  |           404190 |         408.956 |
| NP_H360_B0_K1_s2021        | NP        |       360 | B0        |   1 |   2021 |         0.222573 |        0.492817  |         0.451098 |          1026729 |         497.51  |
| NP_H360_B1_K1_s2021        | NP        |       360 | B1        |   1 |   2021 |         0.215863 |        0.483778  |         0.445734 |          1084974 |         443.337 |
| NP_H360_B2_K1_s2021        | NP        |       360 | B2        |   1 |   2021 |         0.213409 |        0.481943  |         0.45442  |          1084714 |         484.113 |
| NP_H360_C1_K4_s2021        | NP        |       360 | C1        |   4 |   2021 |         0.226455 |        0.495612  |         0.450897 |          1026730 |         562.451 |
| NP_H360_C3_K4_s2021        | NP        |       360 | C3        |   4 |   2021 |         0.212747 |        0.47403   |         0.444638 |          1259706 |         414.339 |
| NP_H360_C4_K1_s2021        | NP        |       360 | C4        |   1 |   2021 |         0.20863  |        0.493511  |         0.454369 |          1142574 |         586.744 |
| NP_H360_C5_K2_s2021        | NP        |       360 | C5        |   2 |   2021 |         0.216265 |        0.482948  |         0.446245 |          1084974 |         566.637 |
| NP_H360_C5_K4_s2021        | NP        |       360 | C5        |   4 |   2021 |         0.218404 |        0.482445  |         0.446074 |          1084974 |         494.744 |
| NP_H360_FULL_K4_s2021      | NP        |       360 | FULL      |   4 |   2021 |         0.213199 |        0.474227  |         0.443118 |          1084974 |         605.425 |
| NP_H360_NOANCHOR_K4_s2021  | NP        |       360 | NOANCHOR  |   4 |   2021 |         0.215158 |        0.46175   |         0.43719  |          1084974 |         628.455 |
| PJM_H24_B0_K1_s2021        | PJM       |        24 | B0        |   1 |   2021 |         0.109202 |        0.0566208 |         0.140923 |          1087481 |         321.885 |
| PJM_H24_B1_K1_s2021        | PJM       |        24 | B1        |   1 |   2021 |         0.108863 |        0.0560395 |         0.140117 |          1203774 |         301.402 |
| PJM_H24_C5_K4_s2021        | PJM       |        24 | C5        |   4 |   2021 |         0.109262 |        0.0563809 |         0.140548 |          1203774 |         339.132 |
| PJM_H360_B0_K1_s2021       | PJM       |       360 | B0        |   1 |   2021 |         0.16379  |        0.127503  |         0.223339 |          2448713 |         642.101 |
| PJM_H360_B1_K1_s2021       | PJM       |       360 | B1        |   1 |   2021 |         0.167901 |        0.12047   |         0.2179   |          2565006 |         492.414 |
| PJM_H360_B2_K1_s2021       | PJM       |       360 | B2        |   1 |   2021 |         0.160664 |        0.121377  |         0.213058 |          2558922 |         564.541 |
| PJM_H360_C1_K4_s2021       | PJM       |       360 | C1        |   4 |   2021 |         0.162155 |        0.132914  |         0.226038 |          2448714 |        1883.59  |
| PJM_H360_C3_K4_s2021       | PJM       |       360 | C3        |   4 |   2021 |         0.168792 |        0.115476  |         0.212404 |          2913882 |         633.648 |
| PJM_H360_C4_K1_s2021       | PJM       |       360 | C4        |   1 |   2021 |         0.163477 |        0.118256  |         0.217076 |          2679950 |         551.297 |
| PJM_H360_C5_K2_s2021       | PJM       |       360 | C5        |   2 |   2021 |         0.167089 |        0.119239  |         0.215955 |          2565006 |         509.657 |
| PJM_H360_C5_K4_s2021       | PJM       |       360 | C5        |   4 |   2021 |         0.164605 |        0.121478  |         0.21546  |          2565006 |         599.892 |
| PJM_H360_FULL_K4_s2021     | PJM       |       360 | FULL      |   4 |   2021 |         0.161695 |        0.113359  |         0.207015 |          2565006 |         631.591 |
| PJM_H360_NOANCHOR_K4_s2021 | PJM       |       360 | NOANCHOR  |   4 |   2021 |         0.162514 |        0.11037   |         0.211632 |          2565006 |         568.8   |

Paired results:

| dataset   |   seed | nonrecurrent         | shared_loop          |   relative_mse_gain |   relative_mae_gain |   mse_difference_loop_minus_control |
|:----------|-------:|:---------------------|:---------------------|--------------------:|--------------------:|------------------------------------:|
| NP        |   2021 | NP_H360_C4_K1_s2021  | NP_H360_C5_K2_s2021  |         0.0214042   |           0.0178789 |                        -0.0105632   |
| PJM       |   2021 | PJM_H360_B2_K1_s2021 | PJM_H360_C5_K4_s2021 |        -0.000827757 |          -0.0112703 |                         0.000100471 |

Budgets:

| dataset   |   seed |   parameter_relative_diff |   batch32_latency_relative_diff | within_parameter_10pct   | within_latency_25pct   |
|:----------|-------:|--------------------------:|--------------------------------:|:-------------------------|:-----------------------|
| NP        |   2021 |               -0.0504125  |                       0.0733647 | True                     | True                   |
| PJM       |   2021 |                0.00237756 |                       0.0783219 | True                     | True                   |

## 6. Interpretation

Conclusion type E; engineering decision INCONCLUSIVE. Mean two-task MSE gain=1.0288%. Validation-selected long-task checkpoint meanK4/K1gain=0.5750%. These measured effects do not automatically imply independent causal propagation or efficiency.

## 7. Scientific Contribution

H1 graph support is established. Any claimed forecasting contribution depends on the complete controlled table, not graph support alone. If controls are equally strong, recurrence is not necessary.

## 8. Limitations and Alternative Explanations

Main experiments are common-checkpoint continuation, finite tasks, conditional future-covariate benchmark. Fixed-K decoder mismatch is an alternative explanation for depth effects. Synthetic data and real data have different semantics. Multi-seed continuation is not three independent pretrained baselines. Unmatched compute/parameters preclude efficiency claims.

## 9. Decision

Final next decision: INCONCLUSIVE. Preserve every negative and failure; do not revise thresholds based on test outcomes.

## 10. Reproduction

python scripts/orchestrate.py; python scripts/post_analysis.py; python scripts/summarize.py --stage 5. Every config and actual status is in EXPERIMENT_MATRIX.csv.



## Current validation-selected models including original references

| dataset   | validation_selected_best_run   |   validation_mae |   test_mse |   test_mae | scope                                                                                                            |
|:----------|:-------------------------------|-----------------:|-----------:|-----------:|:-----------------------------------------------------------------------------------------------------------------|
| NP        | original_NP_H360_s2021         |         0.20365  |   0.430395 |   0.416201 | current model selected on validation including original reference; separate from matched continuation comparison |
| PJM       | PJM_H360_B2_K1_s2021           |         0.160664 |   0.121377 |   0.213058 | current model selected on validation including original reference; separate from matched continuation comparison |

This descriptive current-model selection is separate from the frozen primary continuation-family comparison. See NEXT_RESEARCH_DECISION.md for the original validation scores and history hashes.
