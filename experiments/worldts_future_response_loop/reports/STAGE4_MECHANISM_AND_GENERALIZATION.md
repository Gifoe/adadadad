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

