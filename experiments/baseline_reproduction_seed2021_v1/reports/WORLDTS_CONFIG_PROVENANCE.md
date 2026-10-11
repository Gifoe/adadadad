# WorldTS configuration provenance

Use the released numerical-covariate official source, commit `269426ea6217d7152a244da0e209d922acd46d4f` (2026-10-09), rather than the unavailable-source fallback in the task. Published commands are in `DAG-bench/scripts/w_future/worldts.sh`. No train/validation/test hyperparameter search is introduced.

All executable parameters below are OFFICIAL_SOURCE. The causal architecture, trajectory cosine, VICReg stop-gradient distinction, and two-stage objective also have explicit support in paper Eqs. 6-21. Reproduction environment, logging, output paths, resume checkpoints, and independent metric computations are REPRODUCTION_ASSUMPTION; they do not alter numerical architecture or optimize using the test set.

## NP H=24

| Parameter | Value | Provenance |
|---|---|---|
| allow_missing_future_covariate | false | OFFICIAL_SOURCE |
| batch_size | 32 | OFFICIAL_SOURCE |
| decoder_hidden_multiplier | 2 | OFFICIAL_SOURCE |
| decoder_patch_len | 3 | OFFICIAL_SOURCE |
| predictor_activation | "gelu" | OFFICIAL_SOURCE |
| predictor_d_ff | 256 | OFFICIAL_SOURCE |
| predictor_d_model | 64 | OFFICIAL_SOURCE |
| predictor_dropout | 0.1 | OFFICIAL_SOURCE |
| predictor_e_layers | 1 | OFFICIAL_SOURCE |
| predictor_factor | 1 | OFFICIAL_SOURCE |
| predictor_patch_len | 48 | OFFICIAL_SOURCE |
| predictor_n_heads | 4 | OFFICIAL_SOURCE |
| predictor_patch_stride | 24 | OFFICIAL_SOURCE |
| state_encoder | "causal_patch" | OFFICIAL_SOURCE |
| state_hidden_multiplier | 2 | OFFICIAL_SOURCE |
| state_patch_len | 3 | OFFICIAL_SOURCE |
| condition_dim | 64 | OFFICIAL_SOURCE |
| decoder_type | "causal_patch" | OFFICIAL_SOURCE |
| decoder_warmup_weight | 0.3 | OFFICIAL_SOURCE |
| final_checkpoint_path | "result/NP/worldts_future_symmetric_final24/h24/stages/final.pt" | OFFICIAL_SOURCE |
| horizon | 24 | OFFICIAL_SOURCE |
| latent_alignment_loss | "mae" | OFFICIAL_SOURCE |
| latent_cosine_weight | 15.0 | OFFICIAL_SOURCE |
| latent_detach_target | true | OFFICIAL_SOURCE |
| latent_dim | 128 | OFFICIAL_SOURCE |
| latent_mse_weight | 10.0 | OFFICIAL_SOURCE |
| latent_sigreg_weight | 0.1 | OFFICIAL_SOURCE |
| loss | "MAE" | OFFICIAL_SOURCE |
| lr | 0.0001 | OFFICIAL_SOURCE |
| lradj | "type3" | OFFICIAL_SOURCE |
| norm | true | OFFICIAL_SOURCE |
| num_epochs | 50 | OFFICIAL_SOURCE |
| num_workers | 0 | OFFICIAL_SOURCE |
| patience | 5 | OFFICIAL_SOURCE |
| revin_eps | 1e-05 | OFFICIAL_SOURCE |
| seq_len | 168 | OFFICIAL_SOURCE |
| sigreg_cov_weight | 0.04 | OFFICIAL_SOURCE |
| sigreg_eps | 0.0001 | OFFICIAL_SOURCE |
| sigreg_std_weight | 1.0 | OFFICIAL_SOURCE |
| stage1_checkpoint_path | "result/NP/worldts_future_symmetric_final24/h24/stages/stage1_latent.pt" | OFFICIAL_SOURCE |
| stage1_lr | 0.001 | OFFICIAL_SOURCE |
| stage1_epochs | 20 | OFFICIAL_SOURCE |
| stage2_lr | 0.0001 | OFFICIAL_SOURCE |
| stage2_epochs | 20 | OFFICIAL_SOURCE |
| stage2_pred_epochs | 0 | OFFICIAL_SOURCE |
| stage2_warmup_epochs | 0 | OFFICIAL_SOURCE |
| stage_save_dir | "result/NP/worldts_future_symmetric_final24/h24/stages" | OFFICIAL_SOURCE |
| training_stage | "stage1_latent" | OFFICIAL_SOURCE |
| training_stage_plan | ["stage1_latent", "stage2_decoder_mix"] | OFFICIAL_SOURCE |
| use_amp | 0 | OFFICIAL_SOURCE |
| use_future_x_condition | 1 | OFFICIAL_SOURCE |
| use_modal_condition | 0 | OFFICIAL_SOURCE |
| use_revin | 1 | OFFICIAL_SOURCE |
| use_time_condition | 0 | OFFICIAL_SOURCE |

## NP H=360

| Parameter | Value | Provenance |
|---|---|---|
| allow_missing_future_covariate | false | OFFICIAL_SOURCE |
| batch_size | 32 | OFFICIAL_SOURCE |
| decoder_hidden_multiplier | 2 | OFFICIAL_SOURCE |
| decoder_patch_len | 3 | OFFICIAL_SOURCE |
| predictor_activation | "gelu" | OFFICIAL_SOURCE |
| predictor_d_ff | 256 | OFFICIAL_SOURCE |
| predictor_d_model | 64 | OFFICIAL_SOURCE |
| predictor_dropout | 0.1 | OFFICIAL_SOURCE |
| predictor_e_layers | 1 | OFFICIAL_SOURCE |
| predictor_factor | 1 | OFFICIAL_SOURCE |
| predictor_patch_len | 48 | OFFICIAL_SOURCE |
| predictor_n_heads | 4 | OFFICIAL_SOURCE |
| predictor_patch_stride | 24 | OFFICIAL_SOURCE |
| state_encoder | "causal_patch" | OFFICIAL_SOURCE |
| state_hidden_multiplier | 2 | OFFICIAL_SOURCE |
| state_patch_len | 3 | OFFICIAL_SOURCE |
| condition_dim | 64 | OFFICIAL_SOURCE |
| decoder_type | "causal_patch" | OFFICIAL_SOURCE |
| decoder_warmup_weight | 0.3 | OFFICIAL_SOURCE |
| final_checkpoint_path | "result/NP/worldts_future_symmetric_final24/h360/stages/final.pt" | OFFICIAL_SOURCE |
| horizon | 360 | OFFICIAL_SOURCE |
| latent_alignment_loss | "mae" | OFFICIAL_SOURCE |
| latent_cosine_weight | 15.0 | OFFICIAL_SOURCE |
| latent_detach_target | true | OFFICIAL_SOURCE |
| latent_dim | 128 | OFFICIAL_SOURCE |
| latent_mse_weight | 10.0 | OFFICIAL_SOURCE |
| latent_sigreg_weight | 0.1 | OFFICIAL_SOURCE |
| loss | "MAE" | OFFICIAL_SOURCE |
| lr | 0.0001 | OFFICIAL_SOURCE |
| lradj | "type3" | OFFICIAL_SOURCE |
| norm | true | OFFICIAL_SOURCE |
| num_epochs | 50 | OFFICIAL_SOURCE |
| num_workers | 0 | OFFICIAL_SOURCE |
| patience | 5 | OFFICIAL_SOURCE |
| revin_eps | 1e-05 | OFFICIAL_SOURCE |
| seq_len | 720 | OFFICIAL_SOURCE |
| sigreg_cov_weight | 0.04 | OFFICIAL_SOURCE |
| sigreg_eps | 0.0001 | OFFICIAL_SOURCE |
| sigreg_std_weight | 1.0 | OFFICIAL_SOURCE |
| stage1_checkpoint_path | "result/NP/worldts_future_symmetric_final24/h360/stages/stage1_latent.pt" | OFFICIAL_SOURCE |
| stage1_lr | 0.001 | OFFICIAL_SOURCE |
| stage1_epochs | 20 | OFFICIAL_SOURCE |
| stage2_lr | 0.0001 | OFFICIAL_SOURCE |
| stage2_epochs | 20 | OFFICIAL_SOURCE |
| stage2_pred_epochs | 0 | OFFICIAL_SOURCE |
| stage2_warmup_epochs | 0 | OFFICIAL_SOURCE |
| stage_save_dir | "result/NP/worldts_future_symmetric_final24/h360/stages" | OFFICIAL_SOURCE |
| training_stage | "stage1_latent" | OFFICIAL_SOURCE |
| training_stage_plan | ["stage1_latent", "stage2_decoder_mix"] | OFFICIAL_SOURCE |
| use_amp | 0 | OFFICIAL_SOURCE |
| use_future_x_condition | 1 | OFFICIAL_SOURCE |
| use_modal_condition | 0 | OFFICIAL_SOURCE |
| use_revin | 1 | OFFICIAL_SOURCE |
| use_time_condition | 0 | OFFICIAL_SOURCE |

## PJM H=24

| Parameter | Value | Provenance |
|---|---|---|
| allow_missing_future_covariate | false | OFFICIAL_SOURCE |
| batch_size | 32 | OFFICIAL_SOURCE |
| decoder_hidden_multiplier | 1 | OFFICIAL_SOURCE |
| decoder_patch_len | 5 | OFFICIAL_SOURCE |
| predictor_activation | "gelu" | OFFICIAL_SOURCE |
| predictor_d_ff | 512 | OFFICIAL_SOURCE |
| predictor_d_model | 128 | OFFICIAL_SOURCE |
| predictor_dropout | 0.1 | OFFICIAL_SOURCE |
| predictor_e_layers | 1 | OFFICIAL_SOURCE |
| predictor_factor | 1 | OFFICIAL_SOURCE |
| predictor_patch_len | 48 | OFFICIAL_SOURCE |
| predictor_n_heads | 4 | OFFICIAL_SOURCE |
| predictor_patch_stride | 24 | OFFICIAL_SOURCE |
| state_encoder | "causal_patch" | OFFICIAL_SOURCE |
| state_hidden_multiplier | 1 | OFFICIAL_SOURCE |
| state_patch_len | 5 | OFFICIAL_SOURCE |
| condition_dim | 224 | OFFICIAL_SOURCE |
| decoder_type | "causal_patch" | OFFICIAL_SOURCE |
| decoder_warmup_weight | 0.3 | OFFICIAL_SOURCE |
| final_checkpoint_path | "result/PJM/worldts_future_symmetric_final24/h24/stages/final.pt" | OFFICIAL_SOURCE |
| horizon | 24 | OFFICIAL_SOURCE |
| latent_alignment_loss | "mae" | OFFICIAL_SOURCE |
| latent_cosine_weight | 15.0 | OFFICIAL_SOURCE |
| latent_detach_target | true | OFFICIAL_SOURCE |
| latent_dim | 224 | OFFICIAL_SOURCE |
| latent_mse_weight | 10.0 | OFFICIAL_SOURCE |
| latent_sigreg_weight | 0.1 | OFFICIAL_SOURCE |
| loss | "MAE" | OFFICIAL_SOURCE |
| lr | 0.0001 | OFFICIAL_SOURCE |
| lradj | "type3" | OFFICIAL_SOURCE |
| norm | true | OFFICIAL_SOURCE |
| num_epochs | 50 | OFFICIAL_SOURCE |
| num_workers | 0 | OFFICIAL_SOURCE |
| patience | 5 | OFFICIAL_SOURCE |
| revin_eps | 1e-05 | OFFICIAL_SOURCE |
| seq_len | 168 | OFFICIAL_SOURCE |
| sigreg_cov_weight | 0.04 | OFFICIAL_SOURCE |
| sigreg_eps | 0.0001 | OFFICIAL_SOURCE |
| sigreg_std_weight | 1.0 | OFFICIAL_SOURCE |
| stage1_checkpoint_path | "result/PJM/worldts_future_symmetric_final24/h24/stages/stage1_latent.pt" | OFFICIAL_SOURCE |
| stage1_lr | 0.001 | OFFICIAL_SOURCE |
| stage1_epochs | 20 | OFFICIAL_SOURCE |
| stage2_lr | 0.0002 | OFFICIAL_SOURCE |
| stage2_epochs | 20 | OFFICIAL_SOURCE |
| stage2_pred_epochs | 0 | OFFICIAL_SOURCE |
| stage2_warmup_epochs | 0 | OFFICIAL_SOURCE |
| stage_save_dir | "result/PJM/worldts_future_symmetric_final24/h24/stages" | OFFICIAL_SOURCE |
| training_stage | "stage1_latent" | OFFICIAL_SOURCE |
| training_stage_plan | ["stage1_latent", "stage2_decoder_mix"] | OFFICIAL_SOURCE |
| use_amp | 0 | OFFICIAL_SOURCE |
| use_future_x_condition | 1 | OFFICIAL_SOURCE |
| use_modal_condition | 0 | OFFICIAL_SOURCE |
| use_revin | 1 | OFFICIAL_SOURCE |
| use_time_condition | 0 | OFFICIAL_SOURCE |

## PJM H=360

| Parameter | Value | Provenance |
|---|---|---|
| allow_missing_future_covariate | false | OFFICIAL_SOURCE |
| batch_size | 32 | OFFICIAL_SOURCE |
| decoder_hidden_multiplier | 1 | OFFICIAL_SOURCE |
| decoder_patch_len | 5 | OFFICIAL_SOURCE |
| predictor_activation | "gelu" | OFFICIAL_SOURCE |
| predictor_d_ff | 512 | OFFICIAL_SOURCE |
| predictor_d_model | 128 | OFFICIAL_SOURCE |
| predictor_dropout | 0.1 | OFFICIAL_SOURCE |
| predictor_e_layers | 1 | OFFICIAL_SOURCE |
| predictor_factor | 1 | OFFICIAL_SOURCE |
| predictor_patch_len | 48 | OFFICIAL_SOURCE |
| predictor_n_heads | 4 | OFFICIAL_SOURCE |
| predictor_patch_stride | 24 | OFFICIAL_SOURCE |
| state_encoder | "causal_patch" | OFFICIAL_SOURCE |
| state_hidden_multiplier | 1 | OFFICIAL_SOURCE |
| state_patch_len | 5 | OFFICIAL_SOURCE |
| condition_dim | 224 | OFFICIAL_SOURCE |
| decoder_type | "causal_patch" | OFFICIAL_SOURCE |
| decoder_warmup_weight | 0.3 | OFFICIAL_SOURCE |
| final_checkpoint_path | "result/PJM/worldts_future_symmetric_final24/h360/stages/final.pt" | OFFICIAL_SOURCE |
| horizon | 360 | OFFICIAL_SOURCE |
| latent_alignment_loss | "mae" | OFFICIAL_SOURCE |
| latent_cosine_weight | 15.0 | OFFICIAL_SOURCE |
| latent_detach_target | true | OFFICIAL_SOURCE |
| latent_dim | 224 | OFFICIAL_SOURCE |
| latent_mse_weight | 10.0 | OFFICIAL_SOURCE |
| latent_sigreg_weight | 0.1 | OFFICIAL_SOURCE |
| loss | "MAE" | OFFICIAL_SOURCE |
| lr | 0.0001 | OFFICIAL_SOURCE |
| lradj | "type3" | OFFICIAL_SOURCE |
| norm | true | OFFICIAL_SOURCE |
| num_epochs | 50 | OFFICIAL_SOURCE |
| num_workers | 0 | OFFICIAL_SOURCE |
| patience | 5 | OFFICIAL_SOURCE |
| revin_eps | 1e-05 | OFFICIAL_SOURCE |
| seq_len | 720 | OFFICIAL_SOURCE |
| sigreg_cov_weight | 0.04 | OFFICIAL_SOURCE |
| sigreg_eps | 0.0001 | OFFICIAL_SOURCE |
| sigreg_std_weight | 1.0 | OFFICIAL_SOURCE |
| stage1_checkpoint_path | "result/PJM/worldts_future_symmetric_final24/h360/stages/stage1_latent.pt" | OFFICIAL_SOURCE |
| stage1_lr | 0.001 | OFFICIAL_SOURCE |
| stage1_epochs | 20 | OFFICIAL_SOURCE |
| stage2_lr | 0.0002 | OFFICIAL_SOURCE |
| stage2_epochs | 20 | OFFICIAL_SOURCE |
| stage2_pred_epochs | 0 | OFFICIAL_SOURCE |
| stage2_warmup_epochs | 0 | OFFICIAL_SOURCE |
| stage_save_dir | "result/PJM/worldts_future_symmetric_final24/h360/stages" | OFFICIAL_SOURCE |
| training_stage | "stage1_latent" | OFFICIAL_SOURCE |
| training_stage_plan | ["stage1_latent", "stage2_decoder_mix"] | OFFICIAL_SOURCE |
| use_amp | 0 | OFFICIAL_SOURCE |
| use_future_x_condition | 1 | OFFICIAL_SOURCE |
| use_modal_condition | 0 | OFFICIAL_SOURCE |
| use_revin | 1 | OFFICIAL_SOURCE |
| use_time_condition | 0 | OFFICIAL_SOURCE |

## Code details that supplement the paper

The official configuration specifies PJM latent_dim=224 (rather than assuming 128), symmetric causal patches=5 for PJM and 3 for NP, predictor patch=48/stride=24, Stage 1/2 limits=20/20, and decoder reconstruction coefficient alpha=0.3. The variance/covariance regularizer is selected as legacy_vc; sigreg_cov_weight=0.04. Its variance uses population normalization, covariance uses n-1. It is VICReg-style regularization, not the additional VISReg option present in source.

Official code applies train-only StandardScaler and history-only RevIN. Target/future covariates reuse history statistics. Future fusion final projection is zero-initialized. Validation shuffles windows and averages per-batch losses; original behavior is retained. The official test protocol retains all rolling windows.

Sources: [WorldTS paper](https://arxiv.org/html/2609.31162v1), [released source](https://github.com/decisionintelligence/WorldTS/tree/269426ea6217d7152a244da0e209d922acd46d4f).