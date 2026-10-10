import os, pathlib, random, sys, json, ast, hashlib, ctypes
ROOT = pathlib.Path(__file__).resolve().parents[1]
if os.name == 'nt':
    setter=ctypes.windll.kernel32.SetProcessAffinityMask
    setter.argtypes=[ctypes.c_void_p,ctypes.c_size_t];setter.restype=ctypes.c_int
    if not setter(ctypes.c_void_p(-1),0xFFFF0000):
        raise RuntimeError('Could not set process-only CPU affinity')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(4)
sys.path.insert(0,str(ROOT/'upstream/DAG'))
sys.path.insert(0,str(ROOT))
from ts_benchmark.data.utils import read_data

def seed_all():
    random.seed(2021); np.random.seed(2021); torch.manual_seed(2021); torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def official_ae():
    # Execute the exact upstream class, excluding unrelated training-only imports.
    path=ROOT/'upstream/LatentTSF/my_AE.py'; source=path.read_text(encoding='utf-8')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='AutoEncoder')
    module=ast.Module(body=[node],type_ignores=[]); env={'nn':torch.nn}
    exec(compile(module,str(path),'exec'),env)
    from types import SimpleNamespace
    return env['AutoEncoder'](SimpleNamespace(seq_len=24,enc_in=1,d_model=32,d_ff=64))

def load_data(name):
    df=read_data(str(ROOT/'data'/(name+'.csv')))
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert len(df)==52416 and df.shape[1]==3
    assert df.columns[-1]=='OT' and all('forecast' in s.lower() or 'foecast' in s.lower() for s in df.columns[:-1])
    a=df.to_numpy(dtype=np.float64); assert np.isfinite(a).all()
    tv=int(len(df)*.8); tr=int(tv*.875)
    sy=StandardScaler().fit(a[:tr,-1:]); sc=StandardScaler().fit(a[:tr,:-1])
    y=torch.from_numpy(sy.transform(a[:,-1:]).astype('float32'))
    c=torch.from_numpy(sc.transform(a[:,:-1]).astype('float32'))
    return df,y,c,tr,tv,sy,sc

class Windows(torch.utils.data.Dataset):
    def __init__(self,y,c,t,h,start,end):
        self.y=y;self.c=c;self.t=t;self.h=h; self.origins=np.arange(max(start,t),end-h+1,dtype=np.int64)
        assert len(self.origins)>0
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        o=int(self.origins[i]);return self.y[o-self.t:o],self.c[o-self.t:o],self.c[o:o+self.h],self.y[o:o+self.h],o

def datasets(name,t,h):
    df,y,c,tr,tv,sy,sc=load_data(name)
    return [Windows(y,c,t,h,a,b) for a,b in ((t,tr),(tr,tv),(tv,len(y)))]

def json_write(path,obj):
    pathlib.Path(path).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def audit_data():
    audits={}; text=['# Data and protocol audit','',
      'Official DAG archive; only NP/PJM extracted with CRC32 and SHA256 verification.',
      'Target is the last channel OT after the official TFB long-format reader; the DAG paper defines NP/PJM OT as market electricity price. The CSV label itself does not spell out price.',
      'Hourly chronological split: tv=floor(N*0.8), train=floor(tv*0.875). Validation/test may use prior history; future labels remain inside their own split.',
      'All windows use stride 1; train windows have complete labels inside training. No test-driven tuning.',
      'MSE_norm=mean(((prediction-target)/training_target_std)^2); MAE_norm=mean(abs(prediction-target)/training_target_std). Equal complete windows, no dropped test windows.',
      'Covariates are released grid/load/wind forecasts. Their issue time/vintage is not encoded in the archive. This is benchmark-available future information, not a verified operational availability claim.',
      'DAG scripts use MAE validation and drop_last=True for multivariate training; preserved in B0. B1-B5 select decoded MSE and share full-window training (drop_last=False), 50 epochs/patience 5.',
      'Frozen-AE representation is a pointwise scalar feature expansion, not evidence of physical latent-state identification.','']
    for name in ('NP','PJM'):
        df,y,c,tr,tv,sy,sc=load_data(name)
        raw=pd.read_csv(ROOT/'data'/(name+'.csv'))
        assert raw['cols'].drop_duplicates().tolist()==list(df.columns)
        for column in df.columns:
            block=raw[raw['cols']==column]
            assert len(block)==len(df)
            assert np.array_equal(pd.to_datetime(block['date']).to_numpy(),df.index.to_numpy()),'Long-format timestamps misaligned'
            assert np.array_equal(block['data'].to_numpy(dtype='float64'),df[column].to_numpy(dtype='float64'))
        assert (df.index[1:]-df.index[:-1]==pd.Timedelta(hours=1)).all()
        row={'shape':list(df.shape),'columns':list(df.columns),'target':df.columns[-1],
        'raw_channel_timestamp_alignment_verified':True,'regular_hourly_spacing_verified':True,
        'boundaries':{'train':[0,tr],'validation':[tr,tv],'test':[tv,len(df)]},
        'boundary_dates':{'train_first':str(df.index[0]),'train_last':str(df.index[tr-1]),'val_first':str(df.index[tr]),'val_last':str(df.index[tv-1]),'test_first':str(df.index[tv]),'test_last':str(df.index[-1])},
        'target_scaler':{'mean':sy.mean_.tolist(),'std':sy.scale_.tolist(),'n_fit':int(sy.n_samples_seen_)},
        'covariate_scaler':{'mean':sc.mean_.tolist(),'std':sc.scale_.tolist(),'n_fit':int(sc.n_samples_seen_)},
        'missing':df.isna().sum().to_dict(),'sample_sha256':hashlib.sha256((ROOT/'data'/(name+'.csv')).read_bytes()).hexdigest(),'tasks':{}}
        for t,h in ((168,24),(720,360)):
            ds=datasets(name,t,h)
            row['tasks'][str(h)]={'history':t,'horizon':h,'windows':[len(d) for d in ds],'first_origins':[int(d.origins[0]) for d in ds],'last_origins':[int(d.origins[-1]) for d in ds],
                'input_shapes':{'y_hist':['B',t,1],'c_hist':['B',t,2],'c_future':['B',h,2]},'loss_only_future_target_shape':['B',h,1],'future_latent_shape':['B',h,32]}
            assert ds[0].origins[-1]+h<=tr and ds[1].origins[0]>=tr and ds[1].origins[-1]+h<=tv
            assert ds[2].origins[0]==tv and ds[2].origins[-1]+h==len(df)
        audits[name]=row;text+=['## '+name,'','```json',json.dumps(row,indent=2,default=str),'```','']
    json_write(ROOT/'data_audit.json',audits);(ROOT/'data_audit.md').write_text('\n'.join(text),encoding='utf-8')
    return audits

if __name__=='__main__':audit_data();print('DATA_AUDIT_PASSED',flush=True)
