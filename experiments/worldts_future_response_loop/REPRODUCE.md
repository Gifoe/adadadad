# Reproduce WorldTS future-response Loop

The main experiment family is **checkpoint-initialized two-stage continuation**. All methods and continuation seeds use the same task-specific, independently verified original seed2021 checkpoint. These are not three independently pretrained models. The twelve controlled AR fits form a separate from-scratch family.

## Required inputs and environment

Reference baseline commit: `80d44ad134d1aa28349e2cf520b2470a688aa30a` in `Gifoe/adadadad`, directory `experiments/baseline_reproduction_seed2021_v1`. Official numerical source: WorldTS `269426ea6217d7152a244da0e209d922acd46d4f`, `DAG-bench/ts_benchmark/baselines/worldts/`. The source-only archive and per-file SHA256 manifest in this experiment reconstruct the pinned tree; raw data are excluded from Git.

Set `WORLDTS_BASELINE_ROOT` to the preserved original experiment. Its required files are `data/NP.csv`, `data/PJM.csv`, and each `results/worldts/{NP,PJM}_H{24,360}_seed2021/{resolved_config.json,metrics.json,history.csv,predictions.npz,scaler.npz,stages/stage2_decoder_mix.pt}`. Original weights and predictions remain on the server and are identified by SHA256 in Stage0 and the artifact manifest. Original history files supply validation-score provenance for the descriptive current-best-model answer, without changing the matched continuation comparator. If absent, recreate the baseline with its pinned original `scripts/run_worldts.py`; do not substitute different pretrained weights silently.

The executed host is Windows with RTX5090 32GB; the environment audit records the detected device, driver and CUDA build. The actual Python is `D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe`, after activation of `E:\Anaconda\envs\persist_stable_251` to resolve shared PyTorch/CUDA dependencies. Package inventories are in `reports/`. On another host, install the recorded packages with a CUDA build that supports its GPU. Four CPU threads are used; Windows affinity is applied only on hosts with at least32 logical CPUs.

Raw NP/PJM SHA256:

|Dataset|SHA256|
|---|---|
|NP|29e142d895fa3254bf138e3bec32051e081794a7393a642db36a5d21859d36e1|
|PJM|14feec9daa86e14ba4ed062c690468fa615af6fb18ba56006149b087e26058e5|

Each has52416 hourly points. Split indices are train36690 and train+validation41932. Scalers fit training rows only. Test origins advance by one hour, with10461 windows at H24 and10125 at H360. Keep the benchmark future-covariate definition; it does not establish their availability in deployment.

## Exact execution sequence

Use a separate directory containing this experiment's files. On the recorded server:

For an exact activated command shell, run the following in CMD, then the Python commands below from that shell using the recorded full Python path:

```bat
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
cd /d D:\worldts_future_response_loop
set WORLDTS_BASELINE_ROOT=D:\baseline_reproduction_seed2021_v1
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe scripts/bootstrap.py
```

The equivalent script sequence, once activation has populated the shell environment, is:

```powershell
$env:WORLDTS_BASELINE_ROOT = 'D:\baseline_reproduction_seed2021_v1'
$taskPython = 'D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\worldts_future_response_loop'
# Activate the shared CUDA environment in the invoking shell first.
& $taskPython scripts/bootstrap.py
& $taskPython scripts/link_data.py
& $taskPython scripts/stage0.py
& $taskPython scripts/original_validation_reference.py
& $taskPython scripts/preregister.py
& $taskPython scripts/response_model.py
& $taskPython scripts/stage1.py
& $taskPython scripts/synthetic.py
& $taskPython scripts/summarize.py --stage 1
& $taskPython scripts/orchestrate.py
```

The sequence above starts a new experiment. To resume this completed or interrupted directory, run only `scripts/orchestrate.py`, which verifies completed checkpoints and uses full epoch snapshots for interrupted runs. Do not rerun preregistration inside the recorded experiment: its original timestamp and SHA are part of the saved provenance.

After the orchestrator has exited successfully, refresh the synthetic report and final synthesis, then seal the final artifact hashes:

```powershell
& $taskPython scripts/summarize.py --stage 1
& $taskPython scripts/summarize.py --stage 5
& $taskPython scripts/finalize_manifest.py
```

Run the sealing command after all artifact-producing processes have exited. Do not redirect its output into a file inside the experiment after hashes are sealed. The manifest records server-only arrays and checkpoints as well as small published artifacts; later edits require resealing.

The controlled synthetic protocol adds noise levels0.05 and0.5 before any synthetic training, in `configs/synthetic_protocol.json`. This prospective amendment does not alter the real-data preregistration. The initial continued B0 training served as a training-protocol pilot while synthetic diagnostics were pending; no candidate training began before Stage1 closure.

`configs/preregistration.json` fixes all architectures, K, seeds, objectives, budgets, validation expansion gate and engineering thresholds. The orchestrator executes26 mandatory continuation fits, then28 optional fits only if the recorded validation-only gate passes. Every skipped optional fit has an explicit NOT_RUN reason. B1 is the shared-K1 model and C2 control; it is reused, not counted twice. K8 is conditional and excluded unless supported by prior evidence. A third independent dataset is conditional on stable positive signals and legal, comparable inputs.

Individual runs:

```powershell
& $taskPython scripts/train.py --run-id NP_H360_C5_K4_s2021
& $taskPython scripts/post_analysis.py
& $taskPython scripts/summarize.py --stage 5
```

Training calls the exact pinned official benchmark and two-stage loss, with observational hooks and resumable snapshots. Stage1 selects latent validation loss; Stage2 freezes all representation and predictor modules, then selects decoded validation MAE. Maximum20 epochs per stage, patience5, batch32, published task-specific LR and type3 schedule. Configs and executed argv are saved per run. A completed run is skipped only after its checkpoint hash verifies. Interrupted training resumes model, optimizer, early stopping, best model, epoch and Python/NumPy/Torch/CUDA RNG states. Failure logs are retained; a repair creates a new numbered attempt.

## Integrity and analysis

Stage0 repeats all original checkpoint inference and checks predictions, targets, origins, data, scalers and hashes. Forty preflights check shapes, exact original predictions at insertion, finite gradients, effective first/later updates, future-label invariance, `P(0)=0` after learning, and Stage2 freezing. Synthetic checkpoints are strictly reloaded and checked for identical predictions. Final post-analysis strictly reloads every real checkpoint, independently recomputes float64 metrics and checks every raw target against every official test origin.

All loop depth evaluations K1/2/4 use the same saved checkpoint and fixed decoder; a decoder trained on one K may mismatch other depths. Validation diagnostics retain per-iteration correction norms, changes and errors. Masking and0.25 train-standard-deviation noise are evaluation sensitivity tests, not causal effects. Latency uses10 GPU warmups and30 CUDA-event repetitions, batch1/32, with median and p90.

Statistics pair identical test origins and targets, reporting per-seed MSE/MAE. Moving contiguous-origin block bootstrap uses2000 repetitions, block lengths360/720/1680 and seed9281. A single seed cannot establish seed stability. Costs and capacity differences constrain efficiency claims; fixed engineering thresholds are not definitions of statistical significance.

Source, configs, raw scalar tables, logs, six reports and figures are committed. Large weights, resume snapshots, raw arrays and datasets remain server-only; `ARTIFACT_MANIFEST.json` lists their paths and SHA256. The earlier baseline directory is read-only and is not overwritten.
