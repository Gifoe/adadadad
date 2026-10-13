# Executed server environment

Client date:2026-10-14 Asia/Shanghai; environment snapshot UTC:2026-10-13T16:45:58Z. Server work directory:`D:\neso_worldts_online_v1`. SSH credentials are not stored in this artifact.

OS:`Windows-10-10.0.18363-SP0`. Python:`3.10.20 | packaged by conda-forge | (main, Mar  5 2026, 16:36:49) [MSC v.1944 64 bit (AMD64)]`. Interpreter:`D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe`. Torch:`2.8.0+cu128`; CUDA runtime:12.8; GPU:NVIDIA GeForce RTX 5090; driver616.56; reported board memory32607MiB. Initial free GPU memory~29701MiB, RAMavailable40.73GiB, diskfree28.04GiB. Resource snapshots and package versions are in `outputs/environment.json`.

Activate `E:\Anaconda\envs\persist_stable_251` through the conda batch entrypoint before invoking the existing CUDA-compatible venv launcher. No Torch downgrade or original DAG requirements installation was performed. Only supplied lightweight requirements were checked. CPUthreads4; Windows affinity cores16-31. Single RTX5090; no distributed training or AMP.

Requirements check and original7-test package suite passed; repaired13-test suite passed. Original dummy-model/synthetic tests establish engineering behavior only. The separate `outputs/native_smoke.json` contains a real16-window TRAIN batch, actual imported WorldTS class, nonzero finite E/C/P gradients, Decoder-only Stage2 updates, label foil, shape `[16,48,1]`, and strict checkpoint tensor equality. Formal training and held-out results are separate artifacts.

Official source HEAD:`269426ea6217d7152a244da0e209d922acd46d4f`. A real Git checkout is retained under `upstream/WorldTS`; its tracked source was never patched.
