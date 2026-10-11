# Hardware and environment

Windows-10-10.0.18363-SP0

Sun Oct 11 23:09:59 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 616.56                 KMD Version: 616.56        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090      WDDM  |   00000000:01:00.0  On |                  N/A |
| 30%   50C    P0             61W /  575W |    1270MiB /  32607MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2488    C+G   C:\Windows\System32\dwm.exe           N/A      |
|    0   N/A  N/A            5480    C+G   ...na_cw5n1h2txyewy\SearchUI.exe      N/A      |
|    0   N/A  N/A            6944    C+G   ....0.4258.62\msedgewebview2.exe      N/A      |
|    0   N/A  N/A            7640    C+G   ...ewer\bin\GameViewerServer.exe      N/A      |
|    0   N/A  N/A           12396    C+G   ...iences.TextInput.InputApp.exe      N/A      |
|    0   N/A  N/A           12864    C+G   D:\anydesk\AnyDesk.exe                N/A      |
|    0   N/A  N/A           14328    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           14640    C+G   ...湛鸤AweSun\flutter\AweSun.exe      N/A      |
|    0   N/A  N/A           14704    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           15704    C+G   D:\todesk\ToDesk.exe                  N/A      |
|    0   N/A  N/A           21348    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           22688    C+G   ...udDrive\quark_cloud_drive.exe      N/A      |
|    0   N/A  N/A           22720    C+G   ...udDrive\quark_cloud_drive.exe      N/A      |
|    0   N/A  N/A           25544    C+G   ...t\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A           27760    C+G   D:\Microsoft VS Code\Code.exe         N/A      |
|    0   N/A  N/A           32064    C+G   ...xyewy\ShellExperienceHost.exe      N/A      |
+-----------------------------------------------------------------------------------------+


Private venv uses the existing CUDA-compatible PyTorch installation. Process affinity is restricted to CPUs 16-31; OMP/MKL/PyTorch threads=4. Other projects and environments are unchanged.
