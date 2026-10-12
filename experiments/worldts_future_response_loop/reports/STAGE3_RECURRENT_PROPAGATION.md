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

