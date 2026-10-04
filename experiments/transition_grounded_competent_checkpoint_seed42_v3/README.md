# Competent-checkpoint transition mechanism experiment (seed42 V3)

This experiment tests whether transition supervision changes the causal role of the existing recurrent prediction-position state in a competent official recurrent model. Read `EXPERIMENT_SPEC.txt`, `CHECKPOINT_PROVENANCE.md` and `METHOD_DIFF.md` for the fixed scientific protocol. Actual conclusions and endpoints are in `FINAL_REPORT.md` after completion.

Selected official R2 checkpoint: H_train6, original K2; post-training K5. ID2–6; true OOD8/10/12/16/20. No new parameters. Two matched 5,000-update arms differ only in transition-loss coefficient 0 versus1. First run the untrained checkpoint audit, then discarded 200-step smoke runs, then the formal fine-tuning and matched audits.

Run in an environment with PyTorch2.8+CUDA12.8, transformers4.44.2, numpy and matplotlib. Download original artifacts with `python download_official.py`, verify them with `python inspect_artifacts.py`, then execute `python run_experiment.py`. Downloads resume and verify pinned SHA-256 hashes. `PUBLIC_ARTIFACT_PROXY` optionally supplies a standard HTTP proxy; no proxy credentials or machine-specific proxy setting is stored here. The remote execution wrapper activates the existing conda DLL environment before using the existing venv Python.

Outputs include complete CSVs (including per-sample causal paths), fixed analysis plans, metrics and plots. Checkpoints step0/500/1000/2000/5000 are recorded for both arms; only step5000 is a main result. Server resume files additionally store optimizer and all RNG states at 500-step intervals. Retries resume the last committed checkpoint and discard uncommitted history rows. All raw downloaded artifacts and carrier archives remain available on the server with hashes; official data can be retrieved from pinned source IDs.

V1/V2 files remain independent and are preserved. This directory is the sole location for new scientific results. `STATUS.json` and `execution.log` show actual execution state; the presence of code is not evidence of a completed experiment.
