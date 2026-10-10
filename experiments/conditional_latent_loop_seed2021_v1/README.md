# Conditional Latent Loop Feasibility Audit — seed 2021

This experiment tests frozen-target latent forecasting, non-shared depth, shared refinement, and repeated direct future-covariate access. It uses only the official DAG NP/PJM release and the exact official LatentTSF MLP AE class. It is **WorldTS-inspired**, not a WorldTS reproduction or a complete world model.

Protocol is frozen in `configs/protocol.json`. The primary comparison is K=2; K=4 is a prespecified long-horizon sensitivity experiment. Test labels never select model checkpoints, loop depth, or high-change thresholds. A scalar-to-32-dimensional AE is a feature expansion and does not identify physical state.

Sources: [DAG](https://github.com/decisionintelligence/DAG/tree/0758990e2c73bb54138ea3e7b11a35cbc5476bcc), [LatentTSF](https://github.com/Muyiiiii/LatentTSF/tree/7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2), [WorldTS paper](https://arxiv.org/abs/2609.31162). Their original sources are stored unchanged under `upstream/` with licenses and hashes, also bundled in `upstream.zip`. Only NP/PJM and official metadata were extracted locally from the DAG archive by verified HTTP ranges and then uploaded to the execution server with SHA256 verification; no invented data. `scripts/download_data.py` can repeat selective extraction; set `AUDIT_DOWNLOAD_PROXY` to a reachable HTTP proxy or an empty string for direct access.

Server folder: `D:\conditional_latent_loop_seed2021_v1`. Local working copy: `D:\conditional-latent-loop\experiments\conditional_latent_loop_seed2021_v1`.

Run using the isolated `.venv` after activating the existing CUDA-capable conda runtime:

```text
python scripts/common.py
python scripts/run_official_dag.py --dataset NP --horizon 24
python scripts/train_ae.py
python scripts/train_models.py --sanity
python scripts/run_all.py
python scripts/analyze_results.py
```

`run_all.py` runs the complete experiment sequentially and resumes completed models. Each training run stores its exact config, checkpoint, epoch history, predictions and window-level errors. Official DAG commands are parsed from the published shell script and executed through the original benchmark runner; execution-only adjustments are recorded. Full formal results are valid only after `STATUS.json` says `complete` and the report validates every required artifact.

The task allowed protocol changes. The AE timestamp batching, limited K4 sensitivity scope, native Windows execution, and current CUDA dependencies are documented rather than presented as exact environmental reproduction. The published DAG implementation drops incomplete *training* batches despite its README language; its test evaluation keeps all windows. New models retain all training windows.

Released covariates are external forecasts, but their issue times/vintages are not provided. Findings apply to the benchmark's future-covariate interface, not an established real-time deployment protocol. Confidence intervals concern dependent test-window uncertainty under one trained seed, not training-seed robustness.
