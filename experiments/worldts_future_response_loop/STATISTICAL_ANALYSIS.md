# Statistical analysis

| dataset   |   horizon | variant   |   k |   n_seeds |   mse_mean |   mse_std |   mae_mean |   mae_std |
|:----------|----------:|:----------|----:|----------:|-----------:|----------:|-----------:|----------:|
| NP        |        24 | B0        |   1 |         1 |  0.164129  |       nan |   0.207332 |       nan |
| NP        |        24 | B1        |   1 |         1 |  0.162282  |       nan |   0.20569  |       nan |
| NP        |        24 | C5        |   4 |         1 |  0.165772  |       nan |   0.20758  |       nan |
| NP        |       360 | B0        |   1 |         1 |  0.492817  |       nan |   0.451098 |       nan |
| NP        |       360 | B1        |   1 |         1 |  0.483778  |       nan |   0.445734 |       nan |
| NP        |       360 | B2        |   1 |         1 |  0.481943  |       nan |   0.45442  |       nan |
| NP        |       360 | C1        |   4 |         1 |  0.495612  |       nan |   0.450897 |       nan |
| NP        |       360 | C3        |   4 |         1 |  0.47403   |       nan |   0.444638 |       nan |
| NP        |       360 | C4        |   1 |         1 |  0.493511  |       nan |   0.454369 |       nan |
| NP        |       360 | C5        |   2 |         1 |  0.482948  |       nan |   0.446245 |       nan |
| NP        |       360 | C5        |   4 |         1 |  0.482445  |       nan |   0.446074 |       nan |
| NP        |       360 | FULL      |   4 |         1 |  0.474227  |       nan |   0.443118 |       nan |
| NP        |       360 | NOANCHOR  |   4 |         1 |  0.46175   |       nan |   0.43719  |       nan |
| PJM       |        24 | B0        |   1 |         1 |  0.0566208 |       nan |   0.140923 |       nan |
| PJM       |        24 | B1        |   1 |         1 |  0.0560395 |       nan |   0.140117 |       nan |
| PJM       |        24 | C5        |   4 |         1 |  0.0563809 |       nan |   0.140548 |       nan |
| PJM       |       360 | B0        |   1 |         1 |  0.127503  |       nan |   0.223339 |       nan |
| PJM       |       360 | B1        |   1 |         1 |  0.12047   |       nan |   0.2179   |       nan |
| PJM       |       360 | B2        |   1 |         1 |  0.121377  |       nan |   0.213058 |       nan |
| PJM       |       360 | C1        |   4 |         1 |  0.132914  |       nan |   0.226038 |       nan |
| PJM       |       360 | C3        |   4 |         1 |  0.115476  |       nan |   0.212404 |       nan |
| PJM       |       360 | C4        |   1 |         1 |  0.118256  |       nan |   0.217076 |       nan |
| PJM       |       360 | C5        |   2 |         1 |  0.119239  |       nan |   0.215955 |       nan |
| PJM       |       360 | C5        |   4 |         1 |  0.121478  |       nan |   0.21546  |       nan |
| PJM       |       360 | FULL      |   4 |         1 |  0.113359  |       nan |   0.207015 |       nan |
| PJM       |       360 | NOANCHOR  |   4 |         1 |  0.11037   |       nan |   0.211632 |       nan |

Paired selected comparisons:

| dataset   |   seed | nonrecurrent         | shared_loop          |   relative_mse_gain |   relative_mae_gain |   mse_difference_loop_minus_control |
|:----------|-------:|:---------------------|:---------------------|--------------------:|--------------------:|------------------------------------:|
| NP        |   2021 | NP_H360_C4_K1_s2021  | NP_H360_C5_K2_s2021  |         0.0214042   |           0.0178789 |                        -0.0105632   |
| PJM       |   2021 | PJM_H360_B2_K1_s2021 | PJM_H360_C5_K4_s2021 |        -0.000827757 |          -0.0112703 |                         0.000100471 |

Contiguous-origin moving-block bootstrap (2000 draws, lengths360/720/1680):

| dataset   |   seed |   block_length_origins |   mean_mse_difference |   ci95_low |   ci95_high |   bootstrap_repetitions | overlap_note                                                                                        |
|:----------|-------:|-----------------------:|----------------------:|-----------:|------------:|------------------------:|:----------------------------------------------------------------------------------------------------|
| NP        |   2021 |                    360 |          -0.0105632   | -0.0325418 |  0.00777454 |                    2000 | Contiguous rolling-origin blocks; horizon360 overlaps handled, sensitivity to block length reported |
| NP        |   2021 |                    720 |          -0.0105632   | -0.0345837 |  0.0105834  |                    2000 | Contiguous rolling-origin blocks; horizon360 overlaps handled, sensitivity to block length reported |
| NP        |   2021 |                   1680 |          -0.0105632   | -0.0401923 |  0.0135956  |                    2000 | Contiguous rolling-origin blocks; horizon360 overlaps handled, sensitivity to block length reported |
| PJM       |   2021 |                    360 |           0.000100471 | -0.0155076 |  0.01199    |                    2000 | Contiguous rolling-origin blocks; horizon360 overlaps handled, sensitivity to block length reported |
| PJM       |   2021 |                    720 |           0.000100471 | -0.0166861 |  0.012688   |                    2000 | Contiguous rolling-origin blocks; horizon360 overlaps handled, sensitivity to block length reported |
| PJM       |   2021 |                   1680 |           0.000100471 | -0.0173239 |  0.0107891  |                    2000 | Contiguous rolling-origin blocks; horizon360 overlaps handled, sensitivity to block length reported |

Three seeds, when present, are continuation RNG seeds from a common pretrained checkpoint; they do not establish broad independent pretraining generality. A single seed has no across-seed SD and no claim of multi-seed stability. Bootstrap treats contiguous time dependence, rather than iid overlapping windows; block-length sensitivity and remaining stationarity assumptions are disclosed. Selecting a comparator on validation avoids test-driven model selection, but multiple descriptive comparisons are exploratory rather than independent confirmatory hypotheses.
