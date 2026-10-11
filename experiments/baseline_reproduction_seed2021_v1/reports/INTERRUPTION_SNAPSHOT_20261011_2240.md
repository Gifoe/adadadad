# Baseline reproduction: current evidence and interruption report

Status: **INCOMPLETE — 21/24 configurations completed.** Server 100.94.171.11 is offline in Tailscale; SSH TCP connects time out. Last peer observation was 2026-10-11T14:20:00.1Z. No remote reboot/termination was issued.

## Actual results

|Model|Dataset|Backbone|T/H|Paper MSE/MAE|Measured MSE/MAE|MSE difference|Run type|Status|
|---|---|---|---|---|---|---|---|---|
|DAG|NP|DAG|168/24|0.202/0.237|0.201758/0.236808|0.12%|verified_reuse|MATCHED|
|DAG|NP|DAG|720/360|0.521/0.451|0.527720/0.456922|1.29%|verified_reuse|MATCHED|
|DAG|PJM|DAG|168/24|0.057/0.143|0.057442/0.143281|0.78%|verified_reuse|MATCHED|
|DAG|PJM|DAG|720/360|0.130/0.218|0.129724/0.217574|0.21%|verified_reuse|MATCHED|
|LatentTSF|Electricity|DLinear|720/192|0.168/0.271|0.171538/0.278723|2.11%|fresh_run|MATCHED|
|LatentTSF|Electricity|DLinear|720/336|0.185/0.288|0.196716/0.306010|6.33%|fresh_run|CLOSE|
|LatentTSF|Electricity|DLinear|720/720|0.218/0.317|0.239078/0.344222|9.67%|fresh_run|CLOSE|
|LatentTSF|Electricity|DLinear|720/96|0.157/0.261|0.157547/0.267554|0.35%|fresh_run|MATCHED|
|LatentTSF|Electricity|iTransformer|720/192|0.173/0.279|0.181223/0.292608|4.75%|fresh_run|MATCHED|
|LatentTSF|Electricity|iTransformer|720/336|0.202/0.311|0.195709/0.307831|3.11%|fresh_run|MATCHED|
|LatentTSF|Electricity|iTransformer|720/720|0.250/0.348|0.253541/0.352215|1.42%|fresh_run|MATCHED|
|LatentTSF|Electricity|iTransformer|720/96|0.153/0.260|0.157465/0.267621|2.92%|fresh_run|MATCHED|
|LatentTSF|ETTh1|DLinear|720/192|0.405/0.416|0.403051/0.414384|0.48%|fresh_run|MATCHED|
|LatentTSF|ETTh1|DLinear|720/336|0.435/0.437|0.432642/0.434228|0.54%|fresh_run|MATCHED|
|LatentTSF|ETTh1|DLinear|720/720|0.436/0.468|0.436837/0.467428|0.19%|fresh_run|MATCHED|
|LatentTSF|ETTh1|DLinear|720/96|0.366/0.392|0.364044/0.390882|0.53%|fresh_run|MATCHED|
|LatentTSF|ETTh1|iTransformer|720/192|0.420/0.441|0.429638/0.445351|2.29%|fresh_run|MATCHED|
|LatentTSF|ETTh1|iTransformer|720/336|0.450/0.463|0.451844/0.462907|0.41%|fresh_run|MATCHED|
|LatentTSF|ETTh1|iTransformer|720/720|0.501/0.510|0.506190/0.512377|1.04%|fresh_run|MATCHED|
|LatentTSF|ETTh1|iTransformer|720/96|0.392/0.422|0.394322/0.420839|0.59%|fresh_run|MATCHED|
|WorldTS|PJM|PatchTST-state-transition|168/24|0.056/0.141|0.057516/0.142788|2.71%|fresh_run|MATCHED|

## Corrections based on the papers and released source

- LatentTSF horizon reference values are Appendix D Table 6; Appendix E Table 7 is AE ablation. MAE references were read from Table 6.
- The authors released official WorldTS numerical-covariate code on October 9. Pinned commit 269426ea6217d7152a244da0e209d922acd46d4f is used; no independent model substitution. NP uses latent128/condition64/patch3; PJM uses latent224/condition224/patch5. Both use the published PatchTST transition and Stage1/Stage2 settings.
- DAG uses the exact two-floor split 36690/41932/52416, train drop_last=True and all test windows. Benchmark future-covariate release vintage remains unknown.
- Electricity official LatentTSF stride is 60, including test; all321 target channels are evaluated. ETTh1 stride1 and the official 14400 end boundary are retained.

## Completed verification and limitations

All16 LatentTSF configurations trained; AE tensors/files stayed frozen with identical hashes. 14 MATCHED, two Electricity+DLinear long horizons CLOSE (6.3% and9.7% MSE difference). The residual gap has not been uniquely explained; no test tuning was used.

All4 DAG checkpoints were source/data/hash/scaler/window audited and freshly inferred. Predictions exactly match previous saved predictions. Four official metric archive records agree with float64 independent calculations. Their run_type is verified_reuse. The old NP24 initial stdout is incomplete; a fresh NP24 follow-up is prepared rather than fabricating a log.

WorldTS PJM24 trained Stage1 for6 epochs (best1) and Stage2 for20 (best20). Completed stages successfully reloaded after a wrapper filename assertion failed, then both independent and official evaluations passed (max difference1.45e-9). Exact final weights are stage2_decoder_mix.pt. Its original training GPU peak was not persisted before the failure; 19.78MB is only the resumed fit observation, not its training peak. This result was captured from the last successful SSH stdout; its final server log and complete history have not yet been copied locally.

The standalone raw-CSV target/scaler/hash audit passed all20 earlier results. Trained WorldTS state-collapse diagnostics and final21st raw-target recheck are pending server access. Preflight checked causal context, label invariance, stopgrad/VICReg gradient, two-stage freezing on the official classes.

## Pending work

WorldTS NP24 was last observed at Stage1 epoch5. WorldTS PJM360 and NP360 have not started. Epoch checkpoints and completed-stage snapshots permit resumption after server recovery, but the current server process state is unknown. Fresh DAG NP24 will run after WorldTS to close the old stdout gap. The prepared run_remaining.py then performs trained-checkpoint audits, environment capture, all24 independent checks and generates FINAL_REPORT.md only if all required runs pass.

## Artifacts and environment

Server project: D:\baseline_reproduction_seed2021_v1. Local evidence: D:\baseline-reproduction\experiments\baseline_reproduction_seed2021_v1. Raw data, predictions, scalers and trained weights remain on the server. Git archives metadata, actual available logs, scripts, configurations, official source archive and this incomplete report.

Windows10 build18363, i9-14900KF, 64GiB RAM, RTX5090 driver616.56; Python3.10.20, PyTorch2.8.0+cu128 with verified sm120 CUDA kernels. Private venv overlays NumPy1.26.4 and compatible dependencies while sharing the verified CUDA install. Full captured environment and package versions are included.

Every remaining task is listed in STATUS.json; reproduction commands are in REPRODUCE.md. ISSUE_LOG.md keeps actual errors and corrections. No Loop or new method was added.

Sources: [LatentTSF paper](https://arxiv.org/html/2602.00297v1), [DAG paper](https://arxiv.org/html/2509.14933v1), [WorldTS paper](https://arxiv.org/html/2609.31162v1), [official WorldTS source](https://github.com/decisionintelligence/WorldTS/tree/269426ea6217d7152a244da0e209d922acd46d4f).
