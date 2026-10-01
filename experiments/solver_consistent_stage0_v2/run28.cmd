call E:\Anaconda\Scripts\activate.bat E:\Anaconda\envs\persist_stable_251
if errorlevel 1 exit /b 1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
cd /d D:\solver_consistent_stage0
D:\solver_consistent_stage0\.venv28\Scripts\python.exe prepare28.py
if errorlevel 1 exit /b 1
D:\solver_consistent_stage0\.venv28\Scripts\python.exe -u run_pipeline.py 1>artifacts\pipeline.stdout.log 2>artifacts\pipeline.stderr.log
