@echo off
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONFAULTHANDLER=1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /d D:\transition_grounded_recurrent_reasoning_seed42_v1
.venv\Scripts\python.exe run_experiment.py >> execution.log 2>&1
exit /b %errorlevel%
