call E:\Anaconda\Scripts\activate.bat E:\Anaconda\envs\persist_stable_251
if errorlevel 1 exit /b 1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONFAULTHANDLER=1
set CUDA_LAUNCH_BLOCKING=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /d D:\solver_consistent_stage0
D:\solver_consistent_stage0\.venv28\Scripts\python.exe -u prepare_checkpoint_retry.py 1>artifacts\checkpoint_retry_prepare.log 2>&1
if errorlevel 1 exit /b 1
D:\solver_consistent_stage0\.venv28\Scripts\python.exe -u run_pipeline.py 1>artifacts\pipeline.stdout.log 2>artifacts\pipeline.stderr.log
