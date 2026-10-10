import os, pathlib, sys, zipfile, shutil, importlib.util
if os.name == 'nt':
    import ctypes
    setter=ctypes.windll.kernel32.SetProcessAffinityMask
    setter.argtypes=[ctypes.c_void_p,ctypes.c_size_t];setter.restype=ctypes.c_int
    assert setter(ctypes.c_void_p(-1),0xFFFF0000)
ROOT=pathlib.Path(__file__).resolve().parents[1]
with zipfile.ZipFile(ROOT/'upstream.zip') as z: z.extractall(ROOT)
dst=ROOT/'upstream/DAG/dataset/forecasting'; dst.mkdir(parents=True,exist_ok=True)
for name in ('NP.csv','PJM.csv','FORECAST_META.csv'): shutil.copyfile(ROOT/'data'/name,dst/name)
import torch, numpy, pandas, sklearn
print('VERSIONS',torch.__version__,numpy.__version__,pandas.__version__,sklearn.__version__,flush=True)
print('MODULES',{m:bool(importlib.util.find_spec(m)) for m in ('einops','ray','scipy','matplotlib','tqdm','statsmodels','dash','reformer_pytorch')},flush=True)
print('CUDA',torch.cuda.get_device_name(0),flush=True)
sys.path.insert(0,str(ROOT/'upstream/DAG'))
from ts_benchmark.data.utils import read_data
for name in ('NP','PJM'):
    df=read_data(str(dst/(name+'.csv'))); print(name,df.shape,list(df.columns),df.index[0],df.index[-1],df.isna().sum().to_dict(),flush=True)
from ts_benchmark.baselines.dag.dag import DAG
print('DAG_IMPORT_OK',flush=True)
