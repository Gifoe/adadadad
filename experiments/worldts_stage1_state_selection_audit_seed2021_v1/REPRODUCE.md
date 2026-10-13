# Reproduction on the Windows RTX 5090 server

Use the existing `D:\baseline_reproduction_seed2021_v1\.venv` and activate `E:\Anaconda\envs\persist_stable_251` before Python. Set `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, one GPU, and zero DataLoader workers. The experiment root is `D:\worldts_stage1_state_selection_audit_seed2021_v1` and the original baseline root is `D:\baseline_reproduction_seed2021_v1`. The original NP/PJM CSV files and their scaler/checkpoint files must be present. No dependency installation or dataset download is required.

Run the following commands from the experiment root, replacing `python` with `D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe`. Each command should exit successfully before the dependent next step. Logs from the completed run are in `outputs\logs`.

```powershell
python scripts\bootstrap.py
python scripts\inventory.py
python scripts\stage1_replay.py --dataset NP
python scripts\stage1_replay.py --dataset PJM
python scripts\latent_audit.py --dataset NP
python scripts\latent_audit.py --dataset PJM
python tests\run_checks.py
python scripts\decoder_fit.py --dataset NP --stage1-epoch 1
python scripts\run_matrix.py
python tests\run_checks.py
python scripts\final_diagnostics.py
python scripts\seal_checkpoints.py
python scripts\make_figures.py
```

`run_matrix.py` waits for the first NP/epoch-1 fit to complete, then runs the remaining five cells. It resumes any completed cell without repeating it and verifies identical Decoder initialization, per-epoch Train origin-order hashes, and 22,240 optimizer steps in all six cells. Individual `decoder_fit.py` calls may be used instead; their `latest.pt` checkpoint includes the model, optimizer, RNG and DataLoader-generator state for recovery.

`V_select` origins are 36,690–38,051. `V_diagnostic` origins are 40,210–41,572. With 720 history and 360 forecast hours, raw support intervals are 35,970–38,410 and 39,490–41,931; 1,079 raw hours between them are unused. Train origins are 720–36,330. The standardized series is fitted on Train only and original time ordering is preserved. The source CSV SHA-256 values are checked in `src/ops.py`.

The six Decoder fits use Adam with NP learning rate `1e-4` or PJM `2e-4`, official type-3 decay, batch size 32, and `MAE(prediction,target) + 0.3*MAE(target-state reconstruction,target)`. The original Stage-1 Encoder, Predictor and Condition Encoder remain frozen and in evaluation mode. The original causal patch decoder, history context, RevIN inversion and known future covariates are retained. No Test score enters model selection.

The compact reports are built from raw CSV files after remote outputs are copied to a local checkout:

```powershell
python scripts\build_phase1_report.py
python scripts\build_final_report.py
```

The large `.pt` checkpoints and `.npz` prediction arrays are excluded from Git. Recover them directly on the remote server using paths and SHA-256 values in `outputs/checkpoint_hashes.json`. The pinned source ZIP can be re-extracted and checked against all entries of `configs/source_manifest.json` with `scripts/bootstrap.py`.
