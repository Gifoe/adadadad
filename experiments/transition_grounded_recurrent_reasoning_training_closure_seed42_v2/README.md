# Transition-Grounded Recurrent Reasoning — Training Protocol Closure Seed42 V2

Read FINAL_REPORT.md for the completed decision and outputs/DECISION.json for machine-readable gates. This experiment first attempts official recurrent composition reproduction; matched baseline/grounded and all OOD/causal evaluation are gated. Empty output CSVs mean NOT RUN, not zero performance. checkpoints/availability.json identifies checkpoints not produced because a gate stopped execution.

The supplied prompt is preserved as EXPERIMENT_SPEC.txt. The user resolved the recurrence conflict explicitly: formal K_train=5, unchanged v1 method. A1 uses the actual official RecurrentGPT2Block and dynamic Poisson recurrence2–8. See OFFICIAL_PROTOCOL_AUDIT.md for exact source behavior and deviations (including the official hidden residual/attention-dropout defaults and explicit user-required zero dropout).

Both formal arms load the identical v1 initial weights; model_extension.py is byte-identical to v1. Every stage resets AdamW and scheduler, uses cumulative atomic+<=D data, and advances only on>=95% current-stage validation for three consecutive evaluations500updates apart. Stage cap50000. First threshold crossing and advancement are distinct. Formal baseline has transition coefficient0, grounded1; all other learning conditions match.

## Reproduction

Place this experiment and the previous transition_grounded_recurrent_reasoning_seed42_v1 folder as siblings under an experiments/ directory or D:/ server workspace. The v1 data/generated/{train,test}.{json,npz}, vocab.json, transition_table.json, integrity_check.json, dataset_stats.json and checkpoints/initial_weights.pt are required. The prior dataset archive and checkpoint are delivered through Git LFS in the existing v1 branch. prepare() copies these unchanged and freezes the disjoint375/375 per-hop validation/test partition.

Environment: Python3.10, torch2.8.0+cu128, transformers4.44.2, NumPy, matplotlib; one RTX5090. `python run_experiment.py` executes A1 then conditional matched closure and unchanged v1 evaluations. run_server.cmd contains the actual server environment invocation. SSH credentials are not stored in the repository.

Training executes in eager BF16 with default AdamW numerical settings. No clipping, dropout, optimizer redesign, extra curriculum budgets, adaptive inference or architecture additions. Safety failure ends that arm. Models execute sequentially.

Interrupted training resumes from local *_resume.pt checkpoints, including optimizer/scheduler, batch permutation/cursor, RNG states and curriculum progress. Resume checkpoints remain on the server; final artifacts are delivered separately. An existing completion_audit.json freezes completed results; rerunning does not retrain or overwrite them. Use a fresh folder for a fresh reproduction.

## Artifacts

CONFIG.json and RUN_METADATA.json freeze settings/environment and source hashes. outputs/ contains learning curves, threshold counts, ID sweep, resource metrics and conditional OOD/state/CF measurements. figures/ contains required plots, with explicit NOT EVALUATED graphics for gated stages. ARTIFACT_MANIFEST.json records SHA256 of actual delivered checkpoints. FINAL_REPORT.md follows the requested11-part order.

GPU hours are measured elapsed GPU-workload wall time including validation and checkpointing, not integrated utilization. Allocated VRAM peaks exclude unrelated server processes; any GPU sharing is disclosed in the report. No synthetic measurements are used.
