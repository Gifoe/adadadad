import os,sys,time,json,csv,hashlib,random,pathlib,ctypes
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=pathlib.Path(os.environ.get('WORLDTS_BASELINE_ROOT','D:/baseline_reproduction_seed2021_v1'))
SOURCE_COMMIT='269426ea6217d7152a244da0e209d922acd46d4f'
DATA_SHA={'NP':'29e142d895fa3254bf138e3bec32051e081794a7393a642db36a5d21859d36e1','PJM':'14feec9daa86e14ba4ed062c690468fa615af6fb18ba56006149b087e26058e5'}
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['WANDB_MODE']='disabled'
import numpy as np,torch,pandas as pd
torch.set_num_threads(4)
if os.name=='nt' and (os.cpu_count() or 0)>=32:
    f=ctypes.windll.kernel32.SetProcessAffinityMask;f.argtypes=[ctypes.c_void_p,ctypes.c_size_t];f.restype=ctypes.c_int
    assert f(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
sys.path.insert(0,str(ROOT/'upstream/WorldTS/DAG-bench'))
def write_json(p,v):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=str),encoding='utf-8')
    for attempt in range(100):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)
def write_csv(p,rows):
    if not rows:return
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()
def tensor_sha(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def rng():return {'py':random.getstate(),'np':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(r):random.setstate(r['py']);np.random.set_state(r['np']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def metrics(p,y):
    assert p.shape==y.shape and np.isfinite(p).all() and np.isfinite(y).all()
    e=p.astype('float64')-y.astype('float64');return float(np.mean(e*e)),float(np.mean(abs(e)))
def baseline_dir(ds,h):return BASE/'results/worldts'/f'{ds}_H{h}_seed2021'
def cfg(ds,h):return json.loads((baseline_dir(ds,h)/'resolved_config.json').read_text(encoding='utf-8'))
def load_data(ds):
    from sklearn.preprocessing import StandardScaler
    p=BASE/'data'/(ds+'.csv');assert sha(p)==DATA_SHA[ds]
    raw=pd.read_csv(p);cols=raw.cols.drop_duplicates().tolist();raw.date=pd.to_datetime(raw.date);df=raw.pivot(index='date',columns='cols',values='data')[cols]
    assert cols[-1]=='OT' and len(df)==52416 and np.isfinite(df.to_numpy()).all()
    assert df.index.is_unique and (df.index[1:]-df.index[:-1]==pd.Timedelta(hours=1)).all()
    tv=int(len(df)*.8);tr=int(tv*.875);sy=StandardScaler().fit(df.iloc[:tr,-1:]);sc=StandardScaler().fit(df.iloc[:tr,:-1]);y=torch.from_numpy(sy.transform(df.iloc[:,-1:]).astype('float32'));c=torch.from_numpy(sc.transform(df.iloc[:,:-1]).astype('float32'))
    return df,y,c,tr,tv,sy,sc
class Windows(torch.utils.data.Dataset):
    def __init__(self,y,c,t,h,start,end):self.y=y;self.c=c;self.t=t;self.h=h;self.origins=np.arange(max(start,t),end-h+1,dtype='int64')
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        o=int(self.origins[i]);return self.y[o-self.t:o],self.c[o-self.t:o],self.c[o:o+self.h],self.y[o:o+self.h],o
