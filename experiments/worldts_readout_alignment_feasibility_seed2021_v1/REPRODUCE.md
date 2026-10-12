# Reproduce on the verified Windows server

Remote host `100.89.214.121`, SSH port22, user `fyl412`; host-key MD5 `cd9979cbe69cef4c72e1b89fc55ea5fa`. Authentication is external to this repository. Do not put passwords or tokens in code/configuration.

Experiment root `D:\worldts_readout_alignment_feasibility_seed2021_v1`; original read-only baseline `D:\baseline_reproduction_seed2021_v1`. Python3.10/PyTorch2.8.0+cu128, RTX5090 32GB, existing environment. No dependency installation was required. CPU4 threads, process affinity16–31 when at least32 logical CPUs are present.

Use Command Prompt (`cmd.exe`), or prefix each command with `cmd /c` from PowerShell. The activation supplies the original DLL paths:

```bat
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\bootstrap.py
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\preflight.py
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\audit.py
```

These commands were actually executed on the server. Preflight uses the complete Stage2 checkpoint named by each original `metrics.json`, strictly checks its SHA, original scalers, source prediction arrays and input data. It does not invent paths. The original folders are `results\worldts\NP_H360_seed2021`, `PJM_H360_seed2021`, `NP_H24_seed2021`, `PJM_H24_seed2021`; each requires `resolved_config.json`, `metrics.json`, `history.csv`, `predictions.npz`, and `scaler.npz` plus the complete checkpoint and `data\NP.csv`, `data\PJM.csv`. `preflight_metrics.json` lists actual checkpoint paths and all input hashes.

NP raw SHA256 `29e142d895fa3254bf138e3bec32051e081794a7393a642db36a5d21859d36e1`; PJM `14feec9daa86e14ba4ed062c690468fa615af6fb18ba56006149b087e26058e5`. Both contain52,416 hourly observations. Original split: train ends36,690; train+validation ends41,932. Train-only StandardScaler; input target first, two original exogenous channels. Long-horizon origins train720–36,330, validation36,690–41,572, test41,932–52,056. H24 follows the original T168/H24 configuration. Labels are supervisory only; original history RevIN, future covariates and causal patch prefix are retained.

Original test reproduction uses batch64 to match its archived numerical execution. Using batch32 first produced a 2.384185791015625e-6 NP max difference; retaining the original batch size restores exact equality. Original validation shuffled windows and averaged batch MAEs without weighting the tail batch. Historical shuffle order was not saved; its exact historical mean is not claimed reproducible. This experiment fixes chronological batch32 partitions, verifies the same batches against the official validator, and applies that same selection criterion to all methods. Full element-weighted MAE/MSE are separately reported.

After passing preflight/audit, the following real command executes audit reporting, train-only calibration, six smoke tests, six formal fits, selected-checkpoint gradient checks and final analysis:

```bat
cmd /c D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\run_pipeline.cmd
```

The wrapper activates the same environment and logs to `outputs\logs\pipeline_attempt1.log`. Exact subprocess argv, start times, exit codes and durations are in `outputs\execution_commands.json`. Formal commands expand to:

```bat
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\analyze.py --audit-only
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\calibrate.py
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\train.py --smoke
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\train.py --dataset NP --method A0
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\train.py --dataset NP --method A1
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\train.py --dataset NP --method A2
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\train.py --dataset PJM --method A0
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\train.py --dataset PJM --method A1
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\train.py --dataset PJM --method A2
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\post_audit.py
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\analyze.py
```

`configs\protocol.json` was written before gradient auditing or candidate training. One lambda per task is calibrated on16 fixed TRAIN-only batches. Adam1e-4/batch32/max1000 steps, validation50/patience5, fixed seed2021 and identical order/dropout RNG. Step0 can win. Twenty-step smoke fits are separate and excluded from the formal count. No candidate test evaluation. Completed fits skip only after checkpoint verification; interrupted fits restore model, optimizer, all RNG states, step/early-stop state and curves from their own `resume.pt`. Source/protocol hashes must match. To conduct an independent fresh rerun, copy the experiment into a new directory without generated `outputs/runs`, `outputs/smoke`, `configs/calibration.json` or training-order arrays; never erase the published results or original baseline. The original baseline location can be overridden with `WORLDTS_BASELINE_ROOT` for a copied environment.

All automated contracts are executed through `preflight.py`; `outputs\tests_report.json` records22 passed checks covering Predictor-only updates for A0/A1/A2, decoded gradient flow, no future-label leakage, official decoder equivalence, RevIN, checkpoint roundtrip, manual cosine and zero-gradient handling, float64 metrics and the no_grad guard. Training additionally checks frozen tensor hashes, finite gradients, shared ordering and fresh selected-checkpoint predictions.

After all processes exit and `STATUS.json` says overall_complete=true, seal the final small/large artifacts and protected inputs:

```bat
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\finish_reports.py
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\seal.py
```

Do not append experiment logs after sealing. The manifest excludes itself to avoid a self-hash cycle. Git publication adds a transport provenance file separately; it is not part of model selection.
