@echo off
set "BASELINE_TASK_ROOT=%~dp0.."
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
if errorlevel 1 exit /b 1
if not exist "%BASELINE_TASK_ROOT%\.venv\Scripts\python.exe" python -m venv --system-site-packages "%BASELINE_TASK_ROOT%\.venv"
if errorlevel 1 exit /b 1
"%BASELINE_TASK_ROOT%\.venv\Scripts\python.exe" -m pip install numpy==1.26.4 ray==2.10.0 dash==2.17.0 dash-bootstrap-components==1.6.0 reformer-pytorch==1.4.4 sktime==1.2.0 PyWavelets==1.8.0 datasets==5.1.0 wandb==0.30.0 patool
if errorlevel 1 exit /b 1
"%BASELINE_TASK_ROOT%\.venv\Scripts\python.exe" "%BASELINE_TASK_ROOT%\scripts\setup_environment.py"
exit /b %errorlevel%
