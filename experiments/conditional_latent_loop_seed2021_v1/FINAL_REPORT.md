# Conditional Latent Loop Feasibility Audit — Seed 2021 V1



**Decision: NO-GO.**

Thirty forecasting runs (four official DAG, twenty new K2/single-pass models, six long-horizon K4 models), plus two validation-selected frozen AEs. No test-selected depth or hyperparameters.



| dataset | horizon | B2_MSE | B3_K2_MSE | B5_K2_MSE | B5_vs_B2_reduction_percent | B5_vs_B3_reduction_percent |
| --- | --- | --- | --- | --- | --- | --- |
| NP | 24 | 0.416019 | 0.404238 | 0.509058 | -22.364176 | -25.930069 |
| NP | 360 | 0.964152 | 1.501805 | 1.409159 | -46.155171 | 6.168993 |
| PJM | 24 | 0.094686 | 0.093551 | 0.098646 | -4.182976 | -5.446737 |
| PJM | 360 | 0.146745 | 0.125046 | 0.126541 | 13.768369 | -1.195528 |

Positive reduction means B5 is better; negative reduction means regression. The continuation threshold is fixed K2 and evaluated on point estimates; task-specific uncertainty is reported below and must qualify the strength of any GO signal.



## Implementation and configuration



Exact source commits: DAG `0758990e2c73bb54138ea3e7b11a35cbc5476bcc`; LatentTSF `7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2`.

Official DAG NP/PJM published script arguments are preserved. The original benchmark runner is executed with observational timing/checkpoint hooks and a sequential backend. Architecture files are unchanged.

New models share a 64-wide, four-head, two-layer patch Transformer with patch length24, dataset-specific feed-forward width128/256, zero dropout, 32-dimensional future states and one-time future conditioning. Refinement adds state and condition patch projections, one pre-norm Transformer layer and a 0.1-scaled residual state readout. B3 uses independently parameterized copies; B4/B5 reuse one block K times. B4 still retains the initial future condition; only direct later access is zeroed.

All new models: Adam1e-3, batch64, max50 epochs, patience5; NP type3/PJM type1 schedules as in DAG. Latent loss=10*MSE+15*mean pointwise cosine loss, exact LatentTSF reduction; observation baseline uses decoded MSE. AEs are fully frozen. No intermediate supervision.

AE uses exact official MLP class via AST extraction without unrelated training imports. Train unique normalized target timestamps, batch512, Adam5e-4, MAE, max500/patience20; validation MAE selects AE. This pointwise expansion does not identify true hidden physical state.

Detailed data boundaries, scalers, column identity, availability caveats and leakage assertions are in data_audit.md/data_audit.json. Runtime packages and source hashes are in environment.json/source_manifest.json.



## Official baseline reproduction



| dataset | horizon | mse | mae | reference_mse | reference_mae | mse_relative_difference | official_runner_mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 24 | 0.201758 | 0.236808 | 0.202000 | 0.237000 | -0.001197 | 0.201758 |
| NP | 360 | 0.527719 | 0.456922 | 0.521000 | 0.451000 | 0.012897 | 0.527720 |
| PJM | 24 | 0.057442 | 0.143281 | 0.057000 | 0.143000 | 0.007762 | 0.057442 |
| PJM | 360 | 0.129724 | 0.217574 | 0.130000 | 0.218000 | -0.002122 | 0.129724 |



The independent normalized calculation agrees with the untouched official TFB metric output within2e-5. The paper values are references, not overwritten experimental measurements. Windows and target scaling are identical across models.

Large reference deviations (>20% relative MSE): []. Configuration, exact data bytes, boundary rounding, training-only scaling and rolling-origin equality were inspected before new-model runs. Current CUDA/PyTorch/numerical packages differ from the published environment; this remains a replication caveat rather than grounds for test-driven retuning.



## AE reconstruction



| dataset | split | mae | mse | best_epoch | epochs |
| --- | --- | --- | --- | --- | --- |
| NP | train | 0.000970 | 2.450e-06 | 42 | 62 |
| NP | validation | 0.000512 | 4.442e-07 | 42 | 62 |
| PJM | train | 0.000709 | 8.775e-07 | 13 | 33 |
| PJM | validation | 0.000700 | 7.173e-07 | 13 | 33 |



| dataset | parameters | train_seconds | epochs | best_epoch | latent_forecaster_configurations_using_AE |
| --- | --- | --- | --- | --- | --- |
| NP | 4385 | 20.714589 | 62 | 42 | 11 |
| PJM | 4385 | 10.803614 | 33 | 13 | 11 |

AE preparation is a separate one-time cost per dataset. Forecast-model training seconds below exclude this preparation, setup, downloads, final diagnostic evaluation and profiling.



| dataset | horizon | mse | latent_mse | latent_cosine | decoded_true_latent_mse |
| --- | --- | --- | --- | --- | --- |
| NP | 24 | 0.416019 | 0.105694 | 0.013722 | 9.618e-06 |
| NP | 360 | 0.964152 | 0.222664 | 0.040907 | 9.417e-06 |
| PJM | 24 | 0.094686 | 0.020117 | 0.014860 | 7.340e-07 |
| PJM | 360 | 0.146745 | 0.029389 | 0.024199 | 7.360e-07 |

Decoded ground-truth latents have much smaller held-out MSE than the forecasting models. AE reconstruction is therefore not the dominant error bottleneck in this run; good reconstruction alone does not imply useful forecasting geometry.



## All model comparisons



| dataset | horizon | variant | K | mse | mae | epochs | best_val_mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 24 | B0 | 0 | 0.201758 | 0.236808 | 22 | nan |
| NP | 24 | B1 | 0 | 0.331593 | 0.364881 | 12 | 0.036013 |
| NP | 24 | B2 | 0 | 0.416019 | 0.420932 | 15 | 0.034080 |
| NP | 24 | B3 | 2 | 0.404238 | 0.408011 | 14 | 0.035211 |
| NP | 24 | B4 | 2 | 0.438147 | 0.434921 | 15 | 0.035836 |
| NP | 24 | B5 | 2 | 0.509058 | 0.458036 | 21 | 0.034692 |
| NP | 360 | B0 | 0 | 0.527719 | 0.456922 | 6 | nan |
| NP | 360 | B1 | 0 | 1.528368 | 0.895526 | 11 | 0.097892 |
| NP | 360 | B2 | 0 | 0.964152 | 0.717684 | 6 | 0.085808 |
| NP | 360 | B3 | 2 | 1.501805 | 0.903360 | 9 | 0.075654 |
| NP | 360 | B3 | 4 | 0.999199 | 0.749437 | 6 | 0.084331 |
| NP | 360 | B4 | 2 | 1.435325 | 0.859000 | 8 | 0.077002 |
| NP | 360 | B4 | 4 | 0.912106 | 0.693500 | 6 | 0.080212 |
| NP | 360 | B5 | 2 | 1.409159 | 0.864113 | 11 | 0.080262 |
| NP | 360 | B5 | 4 | 1.010141 | 0.743845 | 6 | 0.082887 |
| PJM | 24 | B0 | 0 | 0.057442 | 0.143281 | 14 | nan |
| PJM | 24 | B1 | 0 | 0.094356 | 0.182444 | 10 | 0.066073 |
| PJM | 24 | B2 | 0 | 0.094686 | 0.176776 | 15 | 0.060888 |
| PJM | 24 | B3 | 2 | 0.093551 | 0.175223 | 17 | 0.063042 |
| PJM | 24 | B4 | 2 | 0.107931 | 0.176509 | 17 | 0.060767 |
| PJM | 24 | B5 | 2 | 0.098646 | 0.173667 | 17 | 0.061500 |
| PJM | 360 | B0 | 0 | 0.129724 | 0.217574 | 9 | nan |
| PJM | 360 | B1 | 0 | 0.159579 | 0.244320 | 8 | 0.100890 |
| PJM | 360 | B2 | 0 | 0.146745 | 0.229639 | 7 | 0.118718 |
| PJM | 360 | B3 | 2 | 0.125046 | 0.209185 | 8 | 0.086785 |
| PJM | 360 | B3 | 4 | 0.120862 | 0.219125 | 8 | 0.074403 |
| PJM | 360 | B4 | 2 | 0.129228 | 0.211712 | 8 | 0.086566 |
| PJM | 360 | B4 | 4 | 0.124335 | 0.208767 | 8 | 0.085817 |
| PJM | 360 | B5 | 2 | 0.126541 | 0.211024 | 8 | 0.087977 |
| PJM | 360 | B5 | 4 | 0.121822 | 0.219019 | 7 | 0.085628 |



## Refinement and horizon diagnostics



| dataset | horizon | variant | K | iteration | mse | mean_gain_from_previous | fraction_improved | fraction_worsened |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 24 | B3 | 2 | 0 | 0.440710 | 0.000000 | nan | nan |
| NP | 24 | B3 | 2 | 1 | 0.424611 | 0.016099 | 0.672402 | 0.327598 |
| NP | 24 | B3 | 2 | 2 | 0.404238 | 0.020373 | 0.684734 | 0.315266 |
| NP | 24 | B4 | 2 | 0 | 0.472043 | 0.000000 | nan | nan |
| NP | 24 | B4 | 2 | 1 | 0.442533 | 0.029510 | 0.702514 | 0.297486 |
| NP | 24 | B4 | 2 | 2 | 0.438147 | 0.004385 | 0.654622 | 0.345378 |
| NP | 24 | B5 | 2 | 0 | 0.517630 | 0.000000 | nan | nan |
| NP | 24 | B5 | 2 | 1 | 0.507296 | 0.010334 | 0.686168 | 0.313832 |
| NP | 24 | B5 | 2 | 2 | 0.509058 | -0.001762 | 0.433228 | 0.566772 |
| NP | 360 | B3 | 2 | 0 | 1.872274 | 0.000000 | nan | nan |
| NP | 360 | B3 | 2 | 1 | 1.586199 | 0.286074 | 0.923160 | 0.076840 |
| NP | 360 | B3 | 2 | 2 | 1.501805 | 0.084395 | 0.839309 | 0.160691 |
| NP | 360 | B3 | 4 | 0 | 1.227503 | 0.000000 | nan | nan |
| NP | 360 | B3 | 4 | 1 | 1.151541 | 0.075962 | 0.765333 | 0.234667 |
| NP | 360 | B3 | 4 | 2 | 1.102903 | 0.048638 | 0.594173 | 0.405827 |
| NP | 360 | B3 | 4 | 3 | 1.039718 | 0.063185 | 0.865086 | 0.134914 |
| NP | 360 | B3 | 4 | 4 | 0.999199 | 0.040520 | 0.847605 | 0.152395 |
| NP | 360 | B4 | 2 | 0 | 1.704123 | 0.000000 | nan | nan |
| NP | 360 | B4 | 2 | 1 | 1.500717 | 0.203406 | 0.814815 | 0.185185 |
| NP | 360 | B4 | 2 | 2 | 1.435325 | 0.065392 | 0.726914 | 0.273086 |
| NP | 360 | B4 | 4 | 0 | 1.213877 | 0.000000 | nan | nan |
| NP | 360 | B4 | 4 | 1 | 1.059390 | 0.154487 | 0.937679 | 0.062321 |
| NP | 360 | B4 | 4 | 2 | 0.979898 | 0.079492 | 0.895605 | 0.104395 |
| NP | 360 | B4 | 4 | 3 | 0.938539 | 0.041359 | 0.851654 | 0.148346 |
| NP | 360 | B4 | 4 | 4 | 0.912106 | 0.026433 | 0.810667 | 0.189333 |
| NP | 360 | B5 | 2 | 0 | 1.595013 | 0.000000 | nan | nan |
| NP | 360 | B5 | 2 | 1 | 1.487624 | 0.107389 | 0.864988 | 0.135012 |
| NP | 360 | B5 | 2 | 2 | 1.409159 | 0.078465 | 0.644642 | 0.355358 |
| NP | 360 | B5 | 4 | 0 | 1.278929 | 0.000000 | nan | nan |
| NP | 360 | B5 | 4 | 1 | 1.153149 | 0.125781 | 0.891062 | 0.108938 |
| NP | 360 | B5 | 4 | 2 | 1.078166 | 0.074983 | 0.906765 | 0.093235 |
| NP | 360 | B5 | 4 | 3 | 1.035890 | 0.042277 | 0.749333 | 0.250667 |
| NP | 360 | B5 | 4 | 4 | 1.010141 | 0.025748 | 0.629235 | 0.370765 |
| PJM | 24 | B3 | 2 | 0 | 0.489638 | 0.000000 | nan | nan |
| PJM | 24 | B3 | 2 | 1 | 0.208831 | 0.280807 | 0.978778 | 0.021222 |
| PJM | 24 | B3 | 2 | 2 | 0.093551 | 0.115281 | 0.886531 | 0.113469 |
| PJM | 24 | B4 | 2 | 0 | 0.728863 | 0.000000 | nan | nan |
| PJM | 24 | B4 | 2 | 1 | 0.283783 | 0.445080 | 0.988624 | 0.011376 |
| PJM | 24 | B4 | 2 | 2 | 0.107931 | 0.175853 | 0.928879 | 0.071121 |
| PJM | 24 | B5 | 2 | 0 | 0.605025 | 0.000000 | nan | nan |
| PJM | 24 | B5 | 2 | 1 | 0.243915 | 0.361110 | 0.989198 | 0.010802 |
| PJM | 24 | B5 | 2 | 2 | 0.098646 | 0.145269 | 0.919702 | 0.080298 |
| PJM | 360 | B3 | 2 | 0 | 0.222165 | 0.000000 | nan | nan |
| PJM | 360 | B3 | 2 | 1 | 0.269275 | -0.047110 | 0.325827 | 0.674173 |
| PJM | 360 | B3 | 2 | 2 | 0.125046 | 0.144229 | 0.899852 | 0.100148 |
| PJM | 360 | B3 | 4 | 0 | 0.297447 | 0.000000 | nan | nan |
| PJM | 360 | B3 | 4 | 1 | 0.438155 | -0.140709 | 0.018864 | 0.981136 |
| PJM | 360 | B3 | 4 | 2 | 0.457011 | -0.018856 | 0.080395 | 0.919605 |
| PJM | 360 | B3 | 4 | 3 | 0.284357 | 0.172654 | 1.000000 | 0.000000 |
| PJM | 360 | B3 | 4 | 4 | 0.120862 | 0.163495 | 0.907951 | 0.092049 |
| PJM | 360 | B4 | 2 | 0 | 0.270562 | 0.000000 | nan | nan |
| PJM | 360 | B4 | 2 | 1 | 0.183838 | 0.086724 | 0.925728 | 0.074272 |
| PJM | 360 | B4 | 2 | 2 | 0.129228 | 0.054610 | 0.839704 | 0.160296 |
| PJM | 360 | B4 | 4 | 0 | 0.339048 | 0.000000 | nan | nan |
| PJM | 360 | B4 | 4 | 1 | 0.327156 | 0.011892 | 0.500543 | 0.499457 |
| PJM | 360 | B4 | 4 | 2 | 0.267515 | 0.059642 | 0.994568 | 0.005432 |
| PJM | 360 | B4 | 4 | 3 | 0.155439 | 0.112076 | 0.993975 | 0.006025 |
| PJM | 360 | B4 | 4 | 4 | 0.124335 | 0.031104 | 0.767210 | 0.232790 |
| PJM | 360 | B5 | 2 | 0 | 0.209550 | 0.000000 | nan | nan |
| PJM | 360 | B5 | 2 | 1 | 0.316267 | -0.106717 | 0.117136 | 0.882864 |
| PJM | 360 | B5 | 2 | 2 | 0.126541 | 0.189726 | 0.941235 | 0.058765 |
| PJM | 360 | B5 | 4 | 0 | 0.216579 | 0.000000 | nan | nan |
| PJM | 360 | B5 | 4 | 1 | 0.274932 | -0.058353 | 0.272395 | 0.727605 |
| PJM | 360 | B5 | 4 | 2 | 0.336646 | -0.061714 | 0.041580 | 0.958420 |
| PJM | 360 | B5 | 4 | 3 | 0.229590 | 0.107056 | 0.998025 | 0.001975 |
| PJM | 360 | B5 | 4 | 4 | 0.121822 | 0.107769 | 0.843457 | 0.156543 |

Horizon quartiles and reconstruction/latent prediction errors are saved in horizon_error.csv and refinement_results.csv. Intermediate outputs are evaluated from the selected final model, not independently selected checkpoints. Each window has one fixed trained loop depth.



## Covariate dependency diagnostics



| dataset | horizon | variant | K | high_change_windows | high_change_mse | zero_future_mse | shuffled_future_mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 24 | B0 | 0 | 1485 | 0.156410 | nan | nan |
| NP | 24 | B1 | 0 | 1485 | 0.221019 | 0.777288 | 0.724576 |
| NP | 24 | B2 | 0 | 1485 | 0.243513 | 0.830407 | 0.873339 |
| NP | 24 | B3 | 2 | 1485 | 0.259903 | 0.770959 | 0.855859 |
| NP | 24 | B4 | 2 | 1485 | 0.275903 | 0.789454 | 0.920151 |
| NP | 24 | B5 | 2 | 1485 | 0.282051 | 0.776677 | 0.959420 |
| NP | 360 | B0 | 0 | 1214 | 0.642789 | nan | nan |
| NP | 360 | B1 | 0 | 1214 | 0.436267 | 1.225547 | 2.162938 |
| NP | 360 | B2 | 0 | 1214 | 0.335815 | 1.291905 | 1.396431 |
| NP | 360 | B3 | 2 | 1214 | 0.749843 | 1.431689 | 2.086773 |
| NP | 360 | B3 | 4 | 1214 | 0.398360 | 1.161023 | 1.415310 |
| NP | 360 | B4 | 2 | 1214 | 0.518096 | 1.359406 | 2.143826 |
| NP | 360 | B4 | 4 | 1214 | 0.309716 | 1.170819 | 1.394676 |
| NP | 360 | B5 | 2 | 1214 | 0.530351 | 1.441331 | 2.080347 |
| NP | 360 | B5 | 4 | 1214 | 0.351588 | 1.176078 | 1.490431 |
| PJM | 24 | B0 | 0 | 1040 | 0.042064 | nan | nan |
| PJM | 24 | B1 | 0 | 1040 | 0.149869 | 0.295108 | 0.361776 |
| PJM | 24 | B2 | 0 | 1040 | 0.111946 | 0.317961 | 0.403666 |
| PJM | 24 | B3 | 2 | 1040 | 0.156764 | 0.296974 | 0.454852 |
| PJM | 24 | B4 | 2 | 1040 | 0.108636 | 0.287191 | 0.443168 |
| PJM | 24 | B5 | 2 | 1040 | 0.131033 | 0.291412 | 0.428242 |
| PJM | 360 | B0 | 0 | 835 | 0.037517 | nan | nan |
| PJM | 360 | B1 | 0 | 835 | 0.154483 | 0.267609 | 0.431303 |
| PJM | 360 | B2 | 0 | 835 | 0.161175 | 0.272682 | 0.337770 |
| PJM | 360 | B3 | 2 | 835 | 0.121000 | 0.306191 | 0.480513 |
| PJM | 360 | B3 | 4 | 835 | 0.080110 | 0.304149 | 0.496179 |
| PJM | 360 | B4 | 2 | 835 | 0.102712 | 0.308620 | 0.427998 |
| PJM | 360 | B4 | 4 | 835 | 0.077298 | 0.308085 | 0.418556 |
| PJM | 360 | B5 | 2 | 835 | 0.118810 | 0.315129 | 0.488357 |
| PJM | 360 | B5 | 4 | 835 | 0.096863 | 0.306819 | 0.450906 |

High-change thresholds use training-only90th-percentile mean absolute standardized covariate increments. Zeroing and fixed within-window time shuffling diagnose reliance and distribution shift; neither establishes causality. Forecast issue times/vintages are unverified.



## Compute comparison



| dataset | horizon | variant | K | trainable_parameters | profiled_flops_batch1 | inference_ms_batch1 | train_seconds | peak_cuda_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 24 | B0 | 0 | 491719 | 1577554 | 8.370281 | 481.474174 | 37282304 |
| NP | 24 | B1 | 0 | 227337 | 4497216 | 1.430897 | 71.324780 | 39528960 |
| NP | 24 | B2 | 0 | 227240 | 5295936 | 1.612000 | 95.058032 | 39630336 |
| NP | 24 | B3 | 2 | 590888 | 6020288 | 2.613807 | 136.516392 | 46632448 |
| NP | 24 | B4 | 2 | 409064 | 6021056 | 2.599465 | 144.745487 | 43714048 |
| NP | 24 | B5 | 2 | 409064 | 6021056 | 2.631822 | 202.600756 | 43714048 |
| NP | 360 | B0 | 0 | 6058591 | 15605362 | 9.485268 | 133.460868 | 170899968 |
| NP | 360 | B1 | 0 | 229266 | 24124800 | 1.457451 | 67.001010 | 123801600 |
| NP | 360 | B2 | 0 | 229169 | 28617600 | 1.682905 | 38.947115 | 138541056 |
| NP | 360 | B3 | 2 | 592817 | 39482880 | 2.674689 | 90.430015 | 186311680 |
| NP | 360 | B3 | 4 | 956465 | 50348160 | 3.704765 | 81.374229 | 233754624 |
| NP | 360 | B4 | 2 | 410993 | 39494400 | 2.677411 | 79.264489 | 186836480 |
| NP | 360 | B4 | 4 | 410993 | 50359680 | 3.728820 | 78.455028 | 229809664 |
| NP | 360 | B5 | 2 | 410993 | 39494400 | 2.640894 | 108.758256 | 186836480 |
| NP | 360 | B5 | 4 | 410993 | 50359680 | 3.669431 | 78.721681 | 229809664 |
| PJM | 24 | B0 | 0 | 557767 | 1577554 | 9.413630 | 305.477417 | 45300736 |
| PJM | 24 | B1 | 0 | 260361 | 4955968 | 1.371294 | 59.326841 | 40057344 |
| PJM | 24 | B2 | 0 | 260264 | 5754688 | 1.615555 | 95.167911 | 40158720 |
| PJM | 24 | B3 | 2 | 656936 | 6544576 | 2.604693 | 166.081782 | 47689216 |
| PJM | 24 | B4 | 2 | 458600 | 6545344 | 2.692881 | 164.502815 | 45274112 |
| PJM | 24 | B5 | 2 | 458600 | 6545344 | 2.636971 | 163.694212 | 45274112 |
| PJM | 360 | B0 | 0 | 6124639 | 15605362 | 9.496501 | 201.158223 | 179418112 |
| PJM | 360 | B1 | 0 | 262290 | 26090880 | 1.445302 | 49.050587 | 128293888 |
| PJM | 360 | B2 | 0 | 262193 | 30583680 | 1.638961 | 45.543918 | 143688704 |
| PJM | 360 | B3 | 2 | 658865 | 42432000 | 2.647957 | 80.481040 | 193821696 |
| PJM | 360 | B3 | 4 | 1055537 | 54280320 | 3.646825 | 108.664307 | 243463168 |
| PJM | 360 | B4 | 2 | 460529 | 42443520 | 2.677240 | 79.606693 | 194088960 |
| PJM | 360 | B4 | 4 | 460529 | 54291840 | 3.701394 | 105.085822 | 239775744 |
| PJM | 360 | B5 | 2 | 460529 | 42443520 | 2.832994 | 79.451338 | 194088960 |
| PJM | 360 | B5 | 4 | 460529 | 54291840 | 3.668354 | 91.912204 | 239775744 |

B3/B5 profiled FLOP counts match within1%, and analytical linear/dense-attention FLOPs match exactly; B4/B5 parameter counts are identical. PyTorch profiling uses an unfused equivalent path and counts supported matmul/conv operations, omitting some operations. The additional analytical count includes attention products but excludes normalization/activation/bias/residual elementwise operations. Latencies use the normal optimized path, warmed synchronized CUDA batch1 final-only decoding; diagnostic intermediate decodes are excluded from deployment timing. Actual wall time and own-process allocated peak memory are recorded. Other user jobs may share the GPU. Early stopping is common but realized epoch/update counts can differ.



## Dependence-aware uncertainty and hypotheses



| dataset | horizon | hypothesis | relative_reduction | relative_ci_low | relative_ci_high | blocks | assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NP | 24 | H1 | -0.254606 | -0.426739 | -0.106515 | 63 | FALSIFIED |
| NP | 24 | H2 | -0.223642 | -0.308386 | -0.128881 | 63 | FALSIFIED |
| NP | 24 | H3 | -0.259301 | -0.374842 | -0.135701 | 63 | FALSIFIED |
| NP | 24 | H4 | -0.161841 | -0.242981 | -0.066762 | 63 | FALSIFIED |
| NP | 360 | H1 | 0.369162 | 0.216206 | 0.459728 | 29 | SUPPORTED |
| NP | 360 | H2 | -0.461552 | -0.684902 | -0.169745 | 29 | FALSIFIED |
| NP | 360 | H3 | 0.061690 | -0.018593 | 0.185465 | 29 | UNRESOLVED |
| NP | 360 | H4 | 0.018230 | -0.034150 | 0.070224 | 29 | UNRESOLVED |
| NP | 360 | H2_K4 | -0.047699 | -0.195667 | 0.051788 | 29 | UNRESOLVED |
| NP | 360 | H3_K4 | -0.010951 | -0.051693 | 0.042008 | 29 | UNRESOLVED |
| NP | 360 | H4_K4 | -0.107482 | -0.157111 | -0.060551 | 29 | FALSIFIED |
| PJM | 24 | H1 | -0.003493 | -0.112185 | 0.132278 | 63 | UNRESOLVED |
| PJM | 24 | H2 | -0.041830 | -0.116613 | 0.070320 | 63 | UNRESOLVED |
| PJM | 24 | H3 | -0.054467 | -0.278424 | 0.192810 | 63 | UNRESOLVED |
| PJM | 24 | H4 | 0.086020 | -0.042381 | 0.150184 | 63 | UNRESOLVED |
| PJM | 360 | H1 | 0.080422 | -0.023395 | 0.208511 | 29 | UNRESOLVED |
| PJM | 360 | H2 | 0.137684 | 0.089773 | 0.205578 | 29 | SUPPORTED |
| PJM | 360 | H3 | -0.011955 | -0.052972 | 0.024404 | 29 | UNRESOLVED |
| PJM | 360 | H4 | 0.020795 | -0.045495 | 0.060462 | 29 | UNRESOLVED |
| PJM | 360 | H2_K4 | 0.169842 | 0.060158 | 0.246478 | 29 | SUPPORTED |
| PJM | 360 | H3_K4 | -0.007946 | -0.063122 | 0.046948 | 29 | UNRESOLVED |
| PJM | 360 | H4_K4 | 0.020214 | -0.199535 | 0.151221 | 29 | UNRESOLVED |

H1=B2 versus B1 (latent recipe versus observation recipe); H2=B5 versus B2 (loop versus single pass); H3=B5 versus B3 (sharing versus matched non-shared depth); H4=B5 versus B4 (repeated direct future-condition access). Relative reductions and confidence limits in this table are fractions, not percentages.

Paired chronological block bootstrap uses2000 replicates and block lengths max(H,168) and twice that length; short-horizon hourly tasks therefore use at least one week to account for additional calendar dependence. Full sensitivity is in bootstrap_comparisons.csv. These intervals reflect dependent evaluation windows under one trained seed. They do not measure training-seed variation. A positive interval supports a task-specific hypothesis; an entirely negative interval falsifies it on that task; an interval crossing zero is unresolved.

The intervals are nominal 95% exploratory comparisons without a multiple-testing correction. They do not provide simultaneous coverage across all tasks and hypotheses.

```json

{
  "H1": {
    "positive_tasks": 2,
    "negative_tasks": 2,
    "supported_tasks": 1,
    "falsified_tasks": 1,
    "uncertain_tasks": 2
  },
  "H2": {
    "positive_tasks": 1,
    "negative_tasks": 3,
    "supported_tasks": 1,
    "falsified_tasks": 2,
    "uncertain_tasks": 1
  },
  "H3": {
    "positive_tasks": 1,
    "negative_tasks": 3,
    "supported_tasks": 0,
    "falsified_tasks": 1,
    "uncertain_tasks": 3
  },
  "H4": {
    "positive_tasks": 3,
    "negative_tasks": 1,
    "supported_tasks": 0,
    "falsified_tasks": 1,
    "uncertain_tasks": 3
  }
}

```



## Failures, changes and interpretation



Setup failures: Google download initially inherited an unusable proxy; corrected with explicit proxy and bounded range reads. Central-directory offsets were corrected before any data CRC verification. Windows affinity integer signatures were corrected and applied process-locally. The original NumPy2 environment is incompatible with upstream np.Inf; isolated NumPy1.26 and missing official imports were installed without modifying shared environments. These setup failures produced no reported scores.

B0 preserves official MAE checkpoint selection and multivariate training drop_last=True, while B1-B5 share decoded-MSE selection and retain all training windows. B0 is an external reference; the mechanism evidence is the matched B1-B5 comparisons. AE batching and K4 long-only scope are declared changes authorized by the task.

Per-iteration latent diagnostics were added after some completed runs. Their selected checkpoints were rescored without retraining, and final decoded MSE was asserted unchanged. Training losses and checkpoint-selection rules were not modified.

Scalar AE expansion, fixed loss geometry, architectural choices, one training seed and unknown forecast vintages constrain generalization. A negative result rejects this tested recipe, not every possible latent-loop architecture. A positive result is a feasibility signal requiring multi-seed confirmation and operational covariate provenance.

H1 compares full observation-space versus frozen-AE latent-loss recipes; it does not isolate the AE from the changed supervision geometry. Sharing also accumulates gradients across repeated applications under the same optimizer, so H3 does not identify a purely representational cause. Any follow-up should separate AE/loss effects and examine learning-rate sensitivity without using this test split for retuning.



## GO / NO-GO



Primary fixed K2 long-horizon threshold checks: {'NP': False, 'PJM': False}. B5-versus-B2 long relative reductions: {'NP': -0.4615517149607442, 'PJM': 0.13768368993511615}; short reductions: {'NP': -0.22364175712091483, 'PJM': -0.04182975976371073}.

NP long horizon: B5 regresses by 46.16% versus B2. Any advantage over B3 cannot establish useful refinement when the single-pass reference is stronger.

PJM long horizon: B5 improves over B2 by 13.77%, but regresses by 1.20% versus compute-matched B3. This is a depth benefit without demonstrated sharing benefit.

The short-horizon non-regression requirement fails: {'NP': -22.364175712091484} percent B5-versus-B2 reductions.

Final direction decision: **NO-GO**. No positive conclusion is forced. K4 is sensitivity evidence and never rescues the primary decision through test-depth selection.

Concrete follow-up, if this direction is revised: first use a matched loss factorial (observation MSE, frozen-AE decoded MSE, frozen-AE latent MSE/cosine) to separate representation from supervision; inspect whether improving latent geometry actually improves decoded errors; and compare shared-block learning-rate scaling against non-shared blocks using validation only. The weak NP conditional predictors also need a stronger single-pass reference before attributing failures solely to recurrence. These are future controls, not changes applied after observing this test set.



References: [DAG official source](https://github.com/decisionintelligence/DAG), [DAG paper](https://arxiv.org/abs/2509.14933), [LatentTSF official source](https://github.com/Muyiiiii/LatentTSF), [WorldTS method reference](https://arxiv.org/abs/2609.31162). This is WorldTS-inspired, not a reproduction.