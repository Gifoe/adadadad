import ctypes,hashlib,importlib.metadata,json,os,pathlib,platform,shutil,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
if os.name=='nt':
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];fn.restype=ctypes.c_int
 assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import torch,numpy as np,psutil
torch.set_num_threads(4)
def save(path,v):path.write_text(json.dumps(v,indent=2,default=str),encoding='utf-8')
env={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'os':platform.platform(),'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torch_file':torch.__file__,'numpy':np.__version__,'numpy_file':np.__file__,'cuda_runtime':torch.version.cuda,'cuda_arch_list':torch.cuda.get_arch_list(),'cpu_count':os.cpu_count(),'ram':dict(psutil.virtual_memory()._asdict()),'disk':dict(psutil.disk_usage(str(ROOT))._asdict()),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'git':shutil.which('git'),'conda_base':'E:/Anaconda/envs/persist_stable_251','private_dependencies':{}}
for name in ['numpy','pandas','scipy','scikit-learn','einops','ray','dash','dash-bootstrap-components','reformer-pytorch','wandb','datasets','sktime','statsmodels','PyWavelets']:
 try:env['private_dependencies'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:env['private_dependencies'][name]=None
assert torch.cuda.is_available()
torch.manual_seed(2021);x=torch.randn(32,64,device='cuda',requires_grad=True);w=torch.randn(64,32,device='cuda',requires_grad=True)
loss=(x@w).square().mean();loss.backward();torch.cuda.synchronize()
assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
sanity={'passed':True,'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'loss':loss.item(),'x_grad_norm':x.grad.norm().item(),'w_grad_norm':w.grad.norm().item(),'free_total_bytes':torch.cuda.mem_get_info(),'arch_list':torch.cuda.get_arch_list()}
ROOT.mkdir(parents=True,exist_ok=True)
save(ROOT/'environment.json',env);save(ROOT/'cuda_sanity.json',sanity)
(ROOT/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
(ROOT/'hardware_report.md').write_text('# Hardware and environment\n\n'+env['os']+'\n\n'+env['nvidia_smi']+'\n\nPrivate venv uses the existing CUDA-compatible PyTorch installation. Process affinity is restricted to CPUs 16-31; OMP/MKL/PyTorch threads=4. Other projects and environments are unchanged.\n',encoding='utf-8')
print(json.dumps({'environment':env['private_dependencies'],'sanity':sanity},indent=2))
