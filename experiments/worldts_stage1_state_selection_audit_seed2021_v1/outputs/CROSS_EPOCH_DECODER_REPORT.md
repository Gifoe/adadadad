# Cross-epoch Decoder comparison

Six independent Decoder fits use the original untrained task-specific Decoder weights. All three Stage-1 states in each task share model architecture, Train origins, randomized origin order per epoch, batch size, Adam optimizer, type-3 schedule, 20 complete epochs and 22,240 optimizer steps. The same fixed `V_select` scorer chooses checkpoints. `V_diagnostic` is computed only after all six choices are locked.

| Dataset | Stage1 E | Checkpoint | Decoder E | V_select MAE/MSE | V_diagnostic MAE/MSE | Select reconstruction MAE/MSE | Diagnostic reconstruction MAE/MSE | Train MAE at E20 | Seconds | Peak CUDA allocated MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NP | 1 | best | 2 | 0.18997/0.07949 | 0.22356/0.09898 | 0.00608/0.00006 | 0.00586/0.00013 | 0.18338 | 287.1 | 390.4 |
| NP | 1 | last | 20 | 0.19088/0.08240 | 0.23264/0.10188 | 0.00407/0.00006 | 0.00539/0.00012 | 0.18338 | 287.1 | 390.4 |
| NP | 3 | best | 17 | 0.17187/0.06032 | 0.25881/0.11960 | 0.00280/0.00006 | 0.00279/0.00007 | 0.10319 | 275.3 | 390.4 |
| NP | 3 | last | 20 | 0.17218/0.06057 | 0.25831/0.11919 | 0.00228/0.00005 | 0.00239/0.00006 | 0.10319 | 275.3 | 390.4 |
| NP | 6 | best | 5 | 0.19695/0.07636 | 0.23916/0.10662 | 0.00349/0.00008 | 0.00358/0.00014 | 0.08322 | 277.1 | 390.4 |
| NP | 6 | last | 20 | 0.19801/0.07755 | 0.23829/0.10604 | 0.00310/0.00021 | 0.00326/0.00023 | 0.08322 | 277.1 | 390.4 |
| PJM | 1 | best | 19 | 0.14894/0.03720 | 0.18441/0.12054 | 0.00786/0.00016 | 0.00987/0.00033 | 0.19267 | 424.9 | 1269.6 |
| PJM | 1 | last | 20 | 0.15138/0.03830 | 0.18360/0.12080 | 0.00793/0.00017 | 0.01046/0.00043 | 0.19267 | 424.9 | 1269.6 |
| PJM | 3 | best | 17 | 0.15923/0.04362 | 0.18294/0.08231 | 0.01141/0.00040 | 0.01536/0.00154 | 0.08509 | 425.2 | 1269.6 |
| PJM | 3 | last | 20 | 0.16044/0.04410 | 0.18199/0.08216 | 0.01100/0.00045 | 0.01190/0.00153 | 0.08509 | 425.2 | 1269.6 |
| PJM | 6 | best | 17 | 0.16039/0.04294 | 0.20157/0.11945 | 0.01067/0.00038 | 0.01306/0.00078 | 0.05924 | 424.9 | 1269.6 |
| PJM | 6 | last | 20 | 0.16046/0.04304 | 0.20249/0.12106 | 0.00939/0.00037 | 0.01047/0.00082 | 0.05924 | 424.9 | 1269.6 |

The optimizer step count for each last checkpoint is exactly 22,240. The best checkpoint may occur earlier because selection was allowed at step 0 and after every complete epoch; no training was stopped at that point. The E20 rows compare equal update opportunities. The `curves.csv` for each cell and combined `decoder_training_curves.csv` retain all 21 checkpoints, including step 0, training objective, forecast MAE, reconstruction MAE, learning rate, selection MAE/MSE, origin-order SHA and elapsed time. The checkpoint manifest contains file SHA-256, full optimizer and RNG recovery states, and strict model restore. Peak values refer to PyTorch process-owned allocated CUDA memory, not total board occupancy.

The Decoder predictive path used `model.forecast(history, x_future=known_covariates)` without a future-target tensor. Reconstruction used the target latent only as a diagnostic or as the official auxiliary training target. Label foil and repeated-prediction tests passed. Only Decoder parameters received optimizer updates.

Research decision: **NO_GO_NEW_STATE_SELECTION_METHOD**. NP 的较晚状态只在选择集上明显更好，在时间隔离诊断集上方向反转；另一任务没有同向的选择集收益。未满足预设跨任务和区块稳定性门槛，不据此开发新状态选择方法。
