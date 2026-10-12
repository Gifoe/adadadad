# Next research decision

**INCONCLUSIVE**

## 1. Is WorldTS+Loop worth continuing?

Decision **INCONCLUSIVE**, categoryE. Mean two-task relative MSE gain against the fixed validation-selected strongest nonrecurrent controls=1.03%; positive direction in every paired seed=False; matched parameters/latency=True; all block-length confidence intervals exclude zero in favor of Loop=False.

## 2. What is the strongest current model?

| dataset   | validation_selected_best_run   |   validation_mae |   test_mse |   test_mae | scope                                                                                                            |
|:----------|:-------------------------------|-----------------:|-----------:|-----------:|:-----------------------------------------------------------------------------------------------------------------|
| NP        | original_NP_H360_s2021         |         0.20365  |   0.430395 |   0.416201 | current model selected on validation including original reference; separate from matched continuation comparison |
| PJM       | PJM_H360_B2_K1_s2021           |         0.160664 |   0.121377 |   0.213058 | current model selected on validation including original reference; separate from matched continuation comparison |

Original checkpoint validation and verified test references:

| dataset   |   validation_mae |   test_mse |   test_mae |
|:----------|-----------------:|-----------:|-----------:|
| NP        |         0.20365  |   0.430395 |   0.416201 |
| PJM       |         0.166074 |   0.115082 |   0.208422 |

Best within the continuation family:

| dataset   | validation_selected_best_run   |   validation_mae |   test_mse |   test_mae |
|:----------|:-------------------------------|-----------------:|-----------:|-----------:|
| NP        | NP_H360_C4_K1_s2021            |         0.20863  |   0.493511 |   0.454369 |
| PJM       | PJM_H360_B2_K1_s2021           |         0.160664 |   0.121377 |   0.213058 |

Selection uses validation scores, never the lowest observed test score. Original references have no added continuation budget and are excluded from the formal matched control and expansion gate.

## 3. What extra contribution does Loop provide?

| dataset   |   seed | nonrecurrent         | shared_loop          |   relative_mse_gain |   relative_mae_gain |   mse_difference_loop_minus_control |
|:----------|-------:|:---------------------|:---------------------|--------------------:|--------------------:|------------------------------------:|
| NP        |   2021 | NP_H360_C4_K1_s2021  | NP_H360_C5_K2_s2021  |         0.0214042   |           0.0178789 |                        -0.0105632   |
| PJM       |   2021 | PJM_H360_B2_K1_s2021 | PJM_H360_C5_K4_s2021 |        -0.000827757 |          -0.0112703 |                         0.000100471 |

Mean same-checkpoint testK4 versus testK1 MSE gain on validation-selected long-task checkpoints=0.57%. A favorable separately trained K4 does not by itself demonstrate test-time scaling.

## 4. Is there innovation beyond the strongest nonrecurrent model?

The comparisons above and measured budget table constrain the claim. CategoryE does not support a claim of independently useful, efficient recurrence. More capacity, unshared depth and wider one-shot propagation remain explicit competing controls.

## 5. What paper claim is currently defensible?

WorldTS has a verified local future-condition support boundary (NP5/PJM9 positions), with controlled short/long synthetic response evidence. Real-data recurrence verdict=E. Do not publish a general superiority claim from structural support alone or from one favorable seed.

## 6. Which missing evidence prevents a broader paper?

Seed expansion was not run because the frozen validation gate failed. Independent pretraining seeds and a legally sourced third comparable future-covariate task remain absent. The selected primary comparisons meet both parameter/latency tolerances=True; budget matching alone does not establish an efficiency benefit without sufficient accuracy gain. Other control mismatches are listed in the complete cost table. Future covariate deployment availability is unverified.

## 7. What is the reasonable alternative direction?

Investigate two-stage latent-objective/decoded-error alignment and covariate information quality before adding recurrent complexity. Preserve the negative controls and original baseline as reference.