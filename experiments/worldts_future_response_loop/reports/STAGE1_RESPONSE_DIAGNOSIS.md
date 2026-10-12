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


![Baseline response Jacobians](../figures/stage1_response_heatmap.png)

Heatmap color is log10 of the mean, over32 fixed validation origins, of the Euclidean norm of the two train-scaled future-covariate derivatives, plus1e-12 for visualization. Both axes use zero-based hourly positions. Raw gradients outside the verified support are exactly zero; the log-scale floor is not a nonzero measured effect.

![Controlled synthetic errors](../figures/stage1_synthetic_comparison.png)

Each panel holds response duration and observation noise fixed. Synthetic test MSE is computed in that system's train-scaled target units; magnitudes across different panels do not measure one common noise-free task.
