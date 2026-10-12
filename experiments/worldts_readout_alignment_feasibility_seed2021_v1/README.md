# WorldTS readout–latent alignment feasibility audit

Completed: six formal fits, six smoke checks and all22 automated contracts. Decision: **NO-GO for a new complex readout-aware loss**; ordinary decoded supervision **INCONCLUSIVE**. Initial full-window and sampled Predictor gradients have no negative cosine. A1/A2 worsen NP MAE versus A0 by3.079%/1.727%, while improving PJM by1.456%/1.278% with time-block intervals including0.

Seed 2021, NP/PJM, T=720/H=360. Reuse the original completed official Stage2 models. Six Predictor-only fits compare A0 official latent objective, A1 decoded MAE, and A2 their train-calibrated combination. No backbone change or baseline retraining.

The research decision and all negative results are in [FINAL_RESEARCH_REPORT](outputs/FINAL_RESEARCH_REPORT.md) and [NEXT_RESEARCH_DECISION](outputs/NEXT_RESEARCH_DECISION.md). Read [PREFLIGHT_REPORT](outputs/PREFLIGHT_REPORT.md), [ALIGNMENT_AUDIT](outputs/ALIGNMENT_AUDIT.md), and [TRAINING_COMPARISON](outputs/TRAINING_COMPARISON.md) for verification, raw diagnostics and all six comparisons. Machine-readable tables, raw execution logs, the fixed protocol, calibration and manifests are included.

Official source: https://github.com/decisionintelligence/WorldTS at `269426ea6217d7152a244da0e209d922acd46d4f`. Source ZIP and 312-file SHA manifest are retained; extraction is deterministic and verified. Third-party source remains under its original license.

Remote experiment: `D:\worldts_readout_alignment_feasibility_seed2021_v1`. Read-only original inputs: `D:\baseline_reproduction_seed2021_v1`. All new full-model best/last checkpoints, optimizer/RNG snapshots, selected validation arrays and horizon diagnostics remain on the server; exact paths and SHA256 are in `ARTIFACT_MANIFEST.json`. Git contains code, reports, small tables, plots, source archive and provenance, not raw data or large tensor arrays.

Validation is used both for audit and checkpoint selection. Candidate test forecasts are deliberately absent; do not describe these results as an unbiased test improvement. Four original-checkpoint test reproductions serve only to verify the historical pipeline.

See [REPRODUCE](REPRODUCE.md) for actual verified Windows commands and data dependencies.
