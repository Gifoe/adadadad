@echo off
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONFAULTHANDLER=1
set PYTHONIOENCODING=utf-8
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set V3_CPU_AFFINITY=FFFF0000
cd /d D:\transition_grounded_competent_checkpoint_seed42_v3
D:\transition_grounded_recurrent_reasoning_seed42_v1\.venv\Scripts\python.exe report.py > report_finalization.log 2>&1
exit /b %errorlevel%
