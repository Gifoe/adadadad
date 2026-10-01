import json, os, pathlib, platform, shutil, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parent; ART=ROOT/'artifacts';ART.mkdir(exist_ok=True)
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('PYTHONFAULTHANDLER','1')
import torch
def run(args):
    p=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace')
    return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
nvsmi=r'C:\Windows\System32\nvidia-smi.exe'
probe=run([nvsmi]) if pathlib.Path(nvsmi).exists() else {'exit_code':127,'stdout':'','stderr':'nvidia-smi missing'}
query=run([nvsmi,'--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader']) if pathlib.Path(nvsmi).exists() else probe
processes=run([nvsmi,'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']) if pathlib.Path(nvsmi).exists() else probe
usage=shutil.disk_usage('D:/')
out={'stage':'A0','started_unix':time.time(),'source_git_commit':'6071738c90d468f1f2f89552ede44669ac07c3da','python':sys.version,'executable':sys.executable,
     'platform':platform.platform(),'hostname':platform.node(),'torch':torch.__version__,'torch_cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
     'cudnn':torch.backends.cudnn.version() if torch.cuda.is_available() else None,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
     'disk_d':{'total':usage.total,'used':usage.used,'free':usage.free},'thread_env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONFAULTHANDLER')},
     'nvidia_smi':probe,'gpu_query':query,'compute_processes':processes,'ended_unix':time.time(),'exit_code':0}
(ART/'A0_ENVIRONMENT.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
freeze=run([sys.executable,'-m','pip','freeze']);(ART/'pip_freeze.txt').write_text(freeze['stdout']+freeze['stderr'],encoding='utf-8')
print(json.dumps({k:out[k] for k in ('source_git_commit','python','executable','platform','torch','torch_cuda_runtime','cuda_available','gpu','disk_d')},indent=2))
