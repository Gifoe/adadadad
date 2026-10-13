@echo off
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
if errorlevel 1 exit /b 11
cd /d D:\neso_worldts_online_v1
set TASK_PYTHON=D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe
%TASK_PYTHON% -u -m neso_online.prepare --forecast data/raw/forecast_2025.csv data/raw/forecast_2026_jun_dec.csv --demand data/raw/demand_2025.csv data/raw/demand_2026.csv --lookback 336 --horizon 48 --revision-hours 6 --exclude-settlement-mismatch --out outputs/dataset > outputs\logs\prepare_attempt3.log 2>&1
if errorlevel 1 exit /b 12
%TASK_PYTHON% -u scripts/audit_dataset.py > outputs\logs\data_audit_attempt1.log 2>&1
if errorlevel 1 exit /b 13
%TASK_PYTHON% -u scripts/native_smoke.py > outputs\logs\native_smoke_attempt1.log 2>&1
if errorlevel 1 exit /b 14
%TASK_PYTHON% -u -m pytest -q > outputs\logs\tests_patched_attempt3.log 2>&1
if errorlevel 1 exit /b 15
%TASK_PYTHON% -u -m neso_online.native_worldts train --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021 --epochs1 20 --epochs2 20 --patience 5 > outputs\logs\train_attempt1.log 2>&1
if errorlevel 1 exit /b 16
exit /b 0
