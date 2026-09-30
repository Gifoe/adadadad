call E:\Anaconda\Scripts\activate.bat E:\Anaconda\envs\persist_stable_251
if errorlevel 1 exit /b 1
E:\Anaconda\envs\persist_stable_251\python.exe -m venv --system-site-packages D:\solver_consistent_stage0\.venv28
if errorlevel 1 exit /b 1
D:\solver_consistent_stage0\.venv28\Scripts\python.exe -m pip install tokenizers PyYAML scipy matplotlib pytest scikit-learn
if errorlevel 1 exit /b 1
D:\solver_consistent_stage0\.venv28\Scripts\python.exe -c "import torch,numpy,tokenizers; print(torch.__version__,numpy.__version__,torch.cuda.get_device_name(0),tokenizers.__version__)"
