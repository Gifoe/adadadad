# Phase 1 — Why Stage 1 peaks at epoch 1

Both tasks use fixed Train and Validation origins, `eval()`, 32-window batches and the same original latent objective. The Train Probe is 4,883 evenly spaced Train origins; the Validation set is all 4,883 original Validation origins. The gap below is their matched snapshot evaluation difference. The historical online Train loss is displayed separately and never used as that gap.

## NP-H360

| Stage-1 epoch | Online Train loss | Fixed Train Probe | Fixed Validation | Matched gap | 10×latent L1 | 15×trajectory cosine | 0.1×regularizer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.1806 | 4.8169 | 6.8302 | 2.0133 | 2.5321 | 4.2221 | 0.0759 |
| 2 | 4.3625 | 3.9017 | 7.7650 | 3.8633 | 2.6002 | 5.0888 | 0.0760 |
| 3 | 3.7064 | 3.5635 | 8.1821 | 4.6186 | 2.5364 | 5.5699 | 0.0758 |
| 4 | 3.4507 | 3.3885 | 8.4620 | 5.0735 | 2.5342 | 5.8505 | 0.0772 |
| 5 | 3.2777 | 3.2119 | 8.6637 | 5.4518 | 2.5691 | 6.0181 | 0.0766 |
| 6 | 3.1248 | 3.0770 | 8.7022 | 5.6252 | 2.5598 | 6.0656 | 0.0768 |

From epoch 1 to 6, Validation objective rises **1.8720**: L1 contributes 0.0276, trajectory cosine 1.8435, and the batch regularizer 0.0009. The matched Train–Validation gap increases from 2.0133 to 5.6252.

| Epoch | 5 shuffled full-objective mean [min, max] | Fixed chronological | Shuffled no-regularizer mean |
|---:|---:|---:|---:|
| 1 | 6.8006 [6.7993, 6.8024] | 6.8302 | 6.7369 |
| 2 | 7.7286 [7.7268, 7.7307] | 7.7650 | 7.6709 |
| 3 | 8.1524 [8.1503, 8.1550] | 8.1821 | 8.0933 |
| 4 | 8.4337 [8.4321, 8.4355] | 8.4620 | 8.3745 |
| 5 | 8.6345 [8.6327, 8.6364] | 8.6637 | 8.5771 |
| 6 | 8.6723 [8.6704, 8.6744] | 8.7022 | 8.6152 |

Epoch 1 ranks first under all five fixed shuffle partitions and under every no-regularizer ranking. Chronological and shuffled absolute values differ because batch grouping and the 19-window tail change the original unweighted batch-mean estimator; neither change the winner. These are the same Validation origins in every partition.

| Epoch | State | Standard deviation | Effective rank | Mean token L2 |
|---:|---|---:|---:|---:|
| 1 | h_hist | 0.5193 | 12.41 | 5.1708 |
| 1 | h_target | 0.6201 | 12.53 | 5.8037 |
| 1 | h_pred | 0.4254 | 17.88 | 4.6076 |
| 3 | h_hist | 0.5068 | 19.32 | 4.5118 |
| 3 | h_target | 0.6408 | 18.83 | 5.1760 |
| 3 | h_pred | 0.3849 | 27.82 | 3.8669 |
| 6 | h_hist | 0.5168 | 25.19 | 4.3257 |
| 6 | h_target | 0.6673 | 24.26 | 5.0289 |
| 6 | h_pred | 0.3817 | 30.39 | 3.6450 |

| Later epoch | State | Centered CKA on Validation | Uncentered kernel alignment | Train-fitted Procrustes Validation MSE / epoch-1 variance |
|---:|---|---:|---:|---:|
| 3 | h_hist | 0.6930 | 0.7464 | 0.3509 |
| 3 | h_target | 0.6813 | 0.6957 | 0.3613 |
| 3 | h_pred | 0.6654 | 0.7862 | 0.5301 |
| 6 | h_hist | 0.5755 | 0.6335 | 0.5577 |
| 6 | h_target | 0.5834 | 0.5941 | 0.5798 |
| 6 | h_pred | 0.5608 | 0.7167 | 0.7387 |

The geometry uses the same 16 predefined Train and 16 Validation origins at each epoch. CKA is centered linear kernel alignment over token rows; the separate uncentered score is shown explicitly. Effective rank is the exponential entropy of centered singular values. Orthogonal Procrustes is fitted on Train tokens and evaluated only on Validation tokens. Its coordinate error is descriptive and the fitted rotation is never applied at deployment. A geometry change alone does not establish information loss or representation collapse.

## PJM-H360

| Stage-1 epoch | Online Train loss | Fixed Train Probe | Fixed Validation | Matched gap | 10×latent L1 | 15×trajectory cosine | 0.1×regularizer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.0953 | 4.3173 | 4.4474 | 0.1301 | 1.9809 | 2.3926 | 0.0740 |
| 2 | 4.2402 | 3.8460 | 5.1426 | 1.2967 | 2.0791 | 2.9919 | 0.0716 |
| 3 | 3.6051 | 3.2275 | 6.0878 | 2.8603 | 2.2019 | 3.8133 | 0.0726 |
| 4 | 3.1502 | 2.9513 | 6.8652 | 3.9139 | 2.3769 | 4.4147 | 0.0736 |
| 5 | 2.8291 | 2.6948 | 7.4177 | 4.7229 | 2.4569 | 4.8871 | 0.0738 |
| 6 | 2.6438 | 2.6166 | 7.7170 | 5.1004 | 2.4663 | 5.1771 | 0.0736 |

From epoch 1 to 6, Validation objective rises **3.2696**: L1 contributes 0.4854, trajectory cosine 2.7845, and the batch regularizer -0.0004. The matched Train–Validation gap increases from 0.1301 to 5.1004.

| Epoch | 5 shuffled full-objective mean [min, max] | Fixed chronological | Shuffled no-regularizer mean |
|---:|---:|---:|---:|
| 1 | 4.4450 [4.4432, 4.4472] | 4.4474 | 4.3782 |
| 2 | 5.1411 [5.1386, 5.1431] | 5.1426 | 5.0779 |
| 3 | 6.0864 [6.0837, 6.0886] | 6.0878 | 6.0238 |
| 4 | 6.8631 [6.8602, 6.8653] | 6.8652 | 6.8017 |
| 5 | 7.4157 [7.4129, 7.4179] | 7.4177 | 7.3547 |
| 6 | 7.7153 [7.7133, 7.7178] | 7.7170 | 7.6543 |

Epoch 1 ranks first under all five fixed shuffle partitions and under every no-regularizer ranking. Chronological and shuffled absolute values differ because batch grouping and the 19-window tail change the original unweighted batch-mean estimator; neither change the winner. These are the same Validation origins in every partition.

| Epoch | State | Standard deviation | Effective rank | Mean token L2 |
|---:|---|---:|---:|---:|
| 1 | h_hist | 0.4857 | 20.30 | 6.7976 |
| 1 | h_target | 0.5673 | 22.13 | 7.4933 |
| 1 | h_pred | 0.4349 | 19.97 | 6.2824 |
| 3 | h_hist | 0.4362 | 41.13 | 5.8049 |
| 3 | h_target | 0.5631 | 42.13 | 6.7290 |
| 3 | h_pred | 0.3817 | 50.82 | 5.2051 |
| 6 | h_hist | 0.4317 | 59.24 | 5.5585 |
| 6 | h_target | 0.5871 | 58.63 | 6.6089 |
| 6 | h_pred | 0.3724 | 63.38 | 4.9235 |

| Later epoch | State | Centered CKA on Validation | Uncentered kernel alignment | Train-fitted Procrustes Validation MSE / epoch-1 variance |
|---:|---|---:|---:|---:|
| 3 | h_hist | 0.8659 | 0.8859 | 0.2409 |
| 3 | h_target | 0.8069 | 0.8052 | 0.3012 |
| 3 | h_pred | 0.8086 | 0.8722 | 0.4121 |
| 6 | h_hist | 0.7401 | 0.8073 | 0.4443 |
| 6 | h_target | 0.6748 | 0.6950 | 0.5562 |
| 6 | h_pred | 0.6650 | 0.8136 | 0.6789 |

The geometry uses the same 16 predefined Train and 16 Validation origins at each epoch. CKA is centered linear kernel alignment over token rows; the separate uncentered score is shown explicitly. Effective rank is the exponential entropy of centered singular values. Orthogonal Procrustes is fitted on Train tokens and evaluated only on Validation tokens. Its coordinate error is descriptive and the fitted rotation is never applied at deployment. A geometry change alone does not establish information loss or representation collapse.

## Interpretation limits

The consistent rise in matched Validation loss and widening matched gap establishes a genuine train–validation divergence for the original latent objective. The trajectory cosine term dominates the rise, while the regularizer contribution is small. The moving encoder changes target geometry, so the loss itself cannot identify whether forecasting information was lost. The fixed six-cell Decoder matrix tests that downstream consequence. No Test score enters this analysis.
