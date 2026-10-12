# Reproduce the three official baselines

Target server: Windows + RTX 5090. Server directory: `D:\baseline_reproduction_seed2021_v1`. All scripts below run from that directory. Large raw data, predictions and trained weights remain on the server. Git contains reproducibility metadata and summary evidence.

## Environment

```cmd
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
python -m venv --system-site-packages D:\baseline_reproduction_seed2021_v1\.venv
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -m pip install numpy==1.26.4 ray==2.10.0 dash==2.17.0 dash-bootstrap-components==1.6.0 reformer-pytorch==1.4.4 sktime==1.2.0 PyWavelets==1.8.0 datasets==5.1.0 wandb==0.30.0 patool
.venv\Scripts\python.exe scripts\setup_environment.py
```

Do not install the original full requirements files over the shared CUDA environment: their old PyTorch versions do not support sm120. The private venv overlays dependencies and reuses the verified PyTorch 2.8.0+cu128. `pip_freeze.txt` records the executed environment. This setup requires the compatible shared base path above; for another server, first install compatible CUDA PyTorch in a separate base.

## Pinned source

```cmd
git clone https://github.com/Muyiiiii/LatentTSF.git upstream\LatentTSF
git -C upstream\LatentTSF checkout 7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2
git clone https://github.com/decisionintelligence/DAG.git upstream\DAG
git -C upstream\DAG checkout 0758990e2c73bb54138ea3e7b11a35cbc5476bcc
git clone https://github.com/decisionintelligence/WorldTS.git upstream\WorldTS
git -C upstream\WorldTS checkout 269426ea6217d7152a244da0e209d922acd46d4f
```

The official LatentTSF repository includes the pretrained AEs. The wrappers locate the actual shipped paths and never train replacements. WorldTS was newly released on October 9; this task uses official source and commands, superseding the independent implementation fallback.

## Data

```cmd
.venv\Scripts\python.exe scripts\download_and_verify_data.py
.venv\Scripts\python.exe scripts\download_and_verify_data.py --verify-only
```

ETTh1/Electricity download from pinned thuml/Time-Series-Library revision `2b66e59ee19dac8f6f19fb5d4997f289fdfea357`, mirror first. NP/PJM reuse the exact bytes from `D:\conditional_latent_loop_seed2021_v1\data`, reverified by SHA256 and hourly/channel integrity. On a clean machine the script downloads only NP/PJM/FORECAST_META from the authors archive using HTTP ranges, checks ZIP CRC and the pinned SHA256 values, then verifies the full hourly time axis. Set AUDIT_DOWNLOAD_PROXY only if an explicit HTTP proxy is needed. This is benchmark future-covariate availability, with no verified historical release vintage.

## Select or resume jobs

```cmd
.venv\Scripts\python.exe scripts\run_latenttsf.py --dataset ETTh1 --backbone DLinear --horizon 96
.venv\Scripts\python.exe scripts\run_latenttsf.py --dataset Electricity --backbone iTransformer --horizon 96
.venv\Scripts\python.exe scripts\run_all.py --phase latenttsf
.venv\Scripts\python.exe scripts\run_all.py --phase dag
.venv\Scripts\python.exe scripts\worldts_preflight.py --dataset PJM --horizon 24
.venv\Scripts\python.exe scripts\run_worldts.py --dataset PJM --horizon 24
.venv\Scripts\python.exe scripts\run_all.py --phase worldts
.venv\Scripts\python.exe scripts\run_all.py --phase all
.venv\Scripts\python.exe scripts\verify_results.py
```

DAG's default entrypoint strictly verifies/reuses prior official checkpoints; the original full training command is recorded per run. It reexecutes forward prediction on all test windows and matches the old predictions and metrics. This requires the previous checkpoint paths; a clean machine automatically uses the supplied fresh DAG training entrypoint. Explicit fresh command: .venv\Scripts\python.exe scripts\run_dag.py --dataset NP --horizon 24 --fresh. Completed results are never overwritten by this flag; use a new project directory to repeat training. No run is silently labeled newly trained when reused.

LatentTSF and WorldTS epoch snapshots include model, optimizer, early-stopping state and all RNG states. Rerunning the same command resumes incomplete training or verifies the saved completed checkpoint before skipping. Existing logs are retained as numbered attempts. WorldTS completed stages are separately reusable, with stage-boundary RNG restoration. Single-GPU execution is sequential. Windows worker count is 0 for LatentTSF; original batch sizes, sample stride, loss, optimizer and validation remain unchanged.

`patches` records exact runtime instrumentation/API/path differences. No model architecture is substituted. LatentTSF uses seed42 exactly; DAG/WorldTS use seed2021. The project folder name does not override the official seed.


## Complete all remaining work after an interruption

```cmd
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
.venv\Scripts\python.exe -u scripts\run_remaining.py
```

This verifies finished LatentTSF runs, restores incomplete WorldTS epochs/stages, preserves the previously verified DAG NP24 artifacts, freshly trains that configuration to close its old stdout gap, then runs trained-checkpoint validation diagnostics, environment capture, independent verification and FINAL_REPORT generation. It never marks completion before all 24 actual metric files and checks pass. Original numbered attempts remain archived. The completion scripts were deployed after SSH recovered. STATUS.json is written as COMPLETE only after the full independent audit and report generation succeed.
