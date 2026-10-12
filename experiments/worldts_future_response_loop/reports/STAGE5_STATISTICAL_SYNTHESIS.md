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

