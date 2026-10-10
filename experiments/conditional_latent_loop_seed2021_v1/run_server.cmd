@echo off
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
cd /d D:\conditional_latent_loop_seed2021_v1
D:\conditional_latent_loop_seed2021_v1\.venv\Scripts\python.exe -u scripts\run_all.py > pipeline.log 2>&1
