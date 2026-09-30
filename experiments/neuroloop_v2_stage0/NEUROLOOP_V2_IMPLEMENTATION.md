# NeuroLoop v2 implementation audit

`models/logpower_stem.py` is the only input stem.  `LogPowerSincStem.power_patches` computes a band-specific spatial projection before squaring and local energy pooling.  Its `signed_mean_patches` helper is diagnostic-only and is never called by `forward`.

`models/neuroloop_v2.py` defines `SharedNeuroLoopV2`.  It owns exactly one `shared_block`; `block_for(k)` returns that object for every k.  The model emits four logits in a single forward pass from the same weights and holds no mutable state from prior batches or trials.  `models/untied_control.py` instead owns four separate blocks solely for the depth-control comparison.

`train/train_neuroloop_v2.py` fixes the loss weights and uses only source train/validation arrays.  `select_then_refit` reinitializes with the same global seed and refits the selected source-validation epoch count on all S1 trials.  Target arrays and target labels are absent from this training API.

`eval/evaluate_loop_trajectory.py` accepts only a frozen model and signal tensor.  Labels are introduced only by `run_stage0_v2.py` after checkpoint serialization and SHA-256 hashing, to calculate frozen metrics and transitions.

The tests under `tests/` check the log-power math, shared-parameter identity, same-checkpoint four-loop outputs, source-only training interface, and causal target preprocessing.  `run_stage0_v2.py` also emits model, checkpoint, stem, selection, trajectory, transition, confidence, and environment audits.
