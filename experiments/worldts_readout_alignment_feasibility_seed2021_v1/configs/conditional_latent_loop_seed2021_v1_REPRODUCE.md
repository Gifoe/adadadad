# Reproduction notes

The full execution folder is `D:\conditional_latent_loop_seed2021_v1` on the Windows GPU server. The Git release contains code, official source bundle, configuration, scalar/window summaries, logs and figures. Full selected model checkpoints and saved forecast arrays remain in that server folder; their SHA256 hashes are in `ARTIFACT_MANIFEST.json`.

## Environment used

Activate the existing CUDA-capable conda runtime before using the private virtual environment:

```bat
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
python -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install numpy==1.26.4 ray==2.10.0 dash==2.17.0 dash-bootstrap-components==1.6.0 reformer-pytorch==1.4.4
```

The shared runtime was not modified. Its PyTorch 2.8.0+cu128 supports the RTX 5090; remaining package versions are recorded in `environment.json`. These commands describe the executed setup, not an exact reproduction of the papers' older environments. Windows processes use CPU affinity mask `0xffff0000` in `scripts/common.py`, four torch/OMP/MKL threads, and no global affinity change. A different machine needs an appropriate process affinity mask or removal of this platform-specific setting.

## Data and pristine sources

From this experiment directory:

```bat
set AUDIT_DOWNLOAD_PROXY=
.venv\Scripts\python.exe scripts\download_data.py
.venv\Scripts\python.exe scripts\prepare_server.py
```

If direct Google Drive access fails, set `AUDIT_DOWNLOAD_PROXY` to an accessible HTTP proxy. The actual run selectively extracted NP/PJM locally through a proxy and transferred them to the server with SHA256 verification. The downloader reads only required ZIP ranges, checks CRC32 and SHA256, and rejects endpoints that ignore HTTP Range. Expected data hashes are in `DATA_SOURCE_MANIFEST.json`. `prepare_server.py` extracts `upstream.zip` and copies these verified CSVs to the original DAG reader's dataset directory. It does not rewrite architecture files.

## Execute and verify

```bat
.venv\Scripts\python.exe -u scripts\run_all.py
.venv\Scripts\python.exe scripts\verify_deliverables.py
.venv\Scripts\python.exe scripts\package_deliverables.py
```

The orchestrator runs 30 forecasting configurations and two AEs sequentially, skips existing completed `result.json` files, and uses validation-selected checkpoints. To conduct a fresh repetition, use a separate new directory with the released sources, protocol and data; retaining existing results intentionally resumes rather than retrains them. `STATUS.json` must report `complete`, and `DELIVERABLE_VERIFICATION.json` must report `passed: true`.

`run_server.cmd` is the exact server launch command and redirects output to `pipeline.log`. Each run also has its own log and epoch history. The final script independently checks all saved predictions, rolling origins, frozen AE weights, source hashes and published tables. Checkpoints must be loaded only from this trusted experiment; original DAG checkpoints preserve its Python configuration objects.

See `FINAL_REPORT.md` for task-specific results and limitations. The future forecast covariates' issuance times are absent from the official release, so deployment-time availability remains unverified. One trained seed and nominal block-bootstrap intervals do not establish multi-seed robustness or simultaneous confidence across all hypotheses.
