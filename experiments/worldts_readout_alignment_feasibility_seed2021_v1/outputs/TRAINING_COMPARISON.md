# Predictor-only supervision comparison

All six fits start from the same task-specific official Stage2 checkpoint. Only Predictor parameters train; every other parameter/buffer is frozen and every other module remains eval. Adam lr=1e-4, batch32, identical seed/order/initial CUDA RNG, max1000 steps, validation every50, patience5 checks. Step0 eligible. Selection is the official unweighted mean of batch MAEs on a fixed chronological validation partition; element-weighted full MAE/MSE are reported separately. Original validator shuffles validation and gives the short tail batch equal weight, so its historical best value is not exactly reproducible without the missing shuffle order. Current fixed partitions are verified against the official validator. This canonicalization is applied to all methods before comparison; no metric or tolerance was relaxed. Phase0 independent test reproduction uses original batch64 and gives max prediction difference0 for all four checkpoints.

A0=official full latent objective; A1=decoded forecast MAE; A2=full latent + lambda*forecast MAE. Lambda is fixed before fits from the median Predictor gradient-norm ratio over16 TRAIN-only batches: {'NP': 12.031270805052078, 'PJM': 5.715996374349599}. No validation/test tuning. Target-Encoder regularizer is constant; predicted-state regularizer and L1/cosine retain Predictor gradients. Frozen Decoder autograd is enabled. Six separate20-step smoke checks passed and are not formal results.

Reported seconds include optimization and periodic validation; command wall times additionally include startup, Step0 validation, saving and fresh reload validation. Peak GPU is process-owned torch.cuda.max_memory_allocated MiB, not total device occupancy. All frozen tensor hashes, shared data-order prefixes, initialization/RNG hashes and fresh-reload prediction checks pass. No candidate test forecasts were generated. Different actual step counts follow the common early-stop rule.

## Complete results

| dataset | method | seed | steps | best_step | step0_selection_MAE | best_selection_MAE | last_selection_MAE | step0_MAE | step0_MSE | best_MAE | best_MSE | last_MAE | last_MSE | MAE_gain_vs_step0_percent | MSE_gain_vs_step0_percent | lambda | seconds_training_and_periodic_validation | peak_allocated_GPU_MiB | predictor_parameters | frozen_unchanged | fresh_prediction_max_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | A0 | 2021 | 400 | 150 | 0.204109 | 0.195545 | 0.202274 | 0.203775 | 0.0876107 | 0.195255 | 0.0790825 | 0.201922 | 0.082472 | 4.18091 | 9.73415 | 12.0313 | 20.3794 | 1426.17 | 827944 | True | 0 |
| NP | A1 | 2021 | 350 | 100 | 0.204109 | 0.201566 | 0.212555 | 0.203775 | 0.0876107 | 0.201268 | 0.084493 | 0.212238 | 0.0904043 | 1.2302 | 3.55859 | 12.0313 | 17.7034 | 1427.25 | 827944 | True | 0 |
| NP | A2 | 2021 | 350 | 100 | 0.204109 | 0.198927 | 0.208678 | 0.203775 | 0.0876107 | 0.198626 | 0.0825864 | 0.208346 | 0.0873216 | 2.52654 | 5.73476 | 12.0313 | 18.0989 | 1427.1 | 827944 | True | 0 |
| PJM | A0 | 2021 | 500 | 250 | 0.166145 | 0.165922 | 0.169516 | 0.166042 | 0.07058 | 0.165931 | 0.0666824 | 0.169541 | 0.0668318 | 0.0672443 | 5.52222 | 5.716 | 45.9964 | 5283.69 | 1891176 | True | 0 |
| PJM | A1 | 2021 | 400 | 150 | 0.166145 | 0.163538 | 0.164207 | 0.166042 | 0.07058 | 0.163515 | 0.0653871 | 0.164201 | 0.0658464 | 1.52212 | 7.35749 | 5.716 | 36.9818 | 5283.13 | 1891176 | True | 0 |
| PJM | A2 | 2021 | 500 | 250 | 0.166145 | 0.163826 | 0.166295 | 0.166042 | 0.07058 | 0.163809 | 0.066523 | 0.166303 | 0.0655869 | 1.34474 | 5.74814 | 5.716 | 46.5562 | 5281.75 | 1891176 | True | 0 |

## Paired validation comparisons

| dataset | candidate | reference | MAE_gain_percent | CI360_low | CI360_high | CI720_low | CI720_high | CI1680_low | improved_blocks_360 | total_blocks_360 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | A1 | A0 | -3.07946 | -6.10327 | -0.249349 | -5.59555 | -0.460932 | -4.84465 | 5 | 14 |
| NP | A2 | A0 | -1.72655 | -3.5123 | 0.0495347 | -3.2792 | -0.0876193 | -2.83251 | 5 | 14 |
| PJM | A1 | A0 | 1.45585 | -1.03153 | 3.99738 | -1.17758 | 4.13258 | -1.43271 | 9 | 14 |
| PJM | A2 | A0 | 1.27835 | -0.394005 | 3.10702 | -0.495914 | 3.39027 | -0.334797 | 9 | 14 |

Circular moving-block bootstrap2000 repetitions, seed9281, block360/720/1680. Compare matching origins; intervals concern already selected validation checkpoints and cannot justify unbiased generalization claims. Every time block and every origin is retained.

| dataset | candidate | fraction_origins_improved | top5_percent_share_of_positive_MAE_reduction | note |
| --- | --- | --- | --- | --- |
| NP | A1 | 0.344051 | 0.432624 | share of positive reductions, not fraction of net effect; no window filtering |
| NP | A2 | 0.347737 | 0.453306 | share of positive reductions, not fraction of net effect; no window filtering |
| PJM | A1 | 0.589801 | 0.291047 | share of positive reductions, not fraction of net effect; no window filtering |
| PJM | A2 | 0.601679 | 0.30961 | share of positive reductions, not fraction of net effect; no window filtering |

## Selected-checkpoint alignment on the same fixed8 batches

| dataset | method | parameter_cosine_mean | parameter_negative_fraction | state_cosine_mean_equal_batch_weight | state_negative_fraction_equal_batch_weight |
| --- | --- | --- | --- | --- | --- |
| NP | A0 | 0.695206 | 0 | 0.0866794 | 0 |
| NP | A1 | 0.662593 | 0 | 0.0856121 | 0 |
| NP | A2 | 0.693081 | 0 | 0.0893785 | 0 |
| PJM | A0 | 0.27752 | 0 | 0.0801005 | 0 |
| PJM | A1 | 0.235662 | 0 | 0.0723175 | 0 |
| PJM | A2 | 0.265921 | 0 | 0.0770334 | 0 |

Training latent/forecast component losses and sampled parameter-gradient trajectories are saved in training_curves.csv/training_gradient_trajectory.csv. Best/last full-model checkpoints and full optimizer/RNG resume files remain on the server; SHA paths are in each run metrics.json and the final artifact manifest. No baseline or old experiment was modified.

