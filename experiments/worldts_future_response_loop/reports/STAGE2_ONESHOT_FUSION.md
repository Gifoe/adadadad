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

