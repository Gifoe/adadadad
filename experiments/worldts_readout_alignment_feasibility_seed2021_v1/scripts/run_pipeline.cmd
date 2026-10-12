@echo off
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
if errorlevel 1 exit /b 1
D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe -u D:\worldts_readout_alignment_feasibility_seed2021_v1\scripts\run_pipeline.py > D:\worldts_readout_alignment_feasibility_seed2021_v1\outputs\logs\pipeline_attempt1.log 2>&1
exit /b %errorlevel%
