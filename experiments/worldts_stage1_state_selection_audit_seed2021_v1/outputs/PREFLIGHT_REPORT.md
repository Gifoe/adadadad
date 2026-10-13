# Phase 0 — Checkpoint provenance and preflight

The original baseline saves the restored best Stage-1 weights in `stages/stage1_latent.pt`. The raw model at the end of epoch 6 is in `resume_stage1_latent.pt`. The historical epoch-3 file was absent, so both tasks were replayed for six full epochs with the pinned upstream method. The only runtime method changes recorded in `stage1_instrumentation.json` add snapshot/logging hooks and prevent early termination before epoch 6.

| Dataset | Original best epoch | Original raw epoch 6 | Replayed epoch 1/6 tensors identical | Replay validation curve identical | Untrained Decoder frozen |
|---|---:|---:|---|---|---|
| NP-H360 | 1 | 6 | True | True | PASS |
| PJM-H360 | 1 | 6 | True | True | PASS |

The replay preserved Adam, type-3 schedule, batch 32, shuffled training and full validation, the original model and loss, seed 2021, and the original scaler fit on Train only. The six raw snapshots each contain the complete model plus optimizer, scheduler LR state, Python/NumPy/Torch/CUDA RNG states, prior best state and validation history. The saved identity explicitly says raw completed epoch before best restoration. All original source files were checked against 312 SHA-256 manifest entries.

Preformal checks: **48 passed, 0 failed**. They cover strict snapshot restore, nonidentical encoder/predictor weights, raw-versus-restored identity, frozen/common Decoder initialization, one-step Decoder-only gradients, future-label foil, origin equality and 1,079-hour raw-support embargo, RevIN inversion, repeated predictions, identical shuffled window sets, and independent MAE/MSE arithmetic.

Protected original data/config/checkpoint/source SHA-256 hashes are recorded in `checkpoint_hashes.json`; all new checkpoint files remain under the experiment directory. No Test forecast was evaluated.
