import ctypes,csv,hashlib,json,os,pathlib,random,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['WANDB_MODE']='disabled';os.environ['WANDB_SILENT']='true'
if os.name=='nt':
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];fn.restype=ctypes.c_int
 assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
import numpy as np,torch
torch.set_num_threads(4)
COMMITS={'LatentTSF':'7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2','DAG':'0758990e2c73bb54138ea3e7b11a35cbc5476bcc','WorldTS':'269426ea6217d7152a244da0e209d922acd46d4f'}
DATA_SHA={'NP':'29e142d895fa3254bf138e3bec32051e081794a7393a642db36a5d21859d36e1','PJM':'14feec9daa86e14ba4ed062c690468fa615af6fb18ba56006149b087e26058e5','ETTh1':'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066','Electricity':'7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad'}
def write_json(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=str),encoding='utf-8');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def tensor_sha(module):
 h=hashlib.sha256()
 for k,v in sorted(module.state_dict().items()):
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def rng():return {'py':random.getstate(),'np':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(r):random.setstate(r['py']);np.random.set_state(r['np']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def write_csv(p,rows):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def paper_reference(model,dataset,backbone,h):
 if model=='LatentTSF':
  pairs={('ETTh1','DLinear'):[(.366,.392),(.405,.416),(.435,.437),(.436,.468)],('ETTh1','iTransformer'):[(.392,.422),(.420,.441),(.450,.463),(.501,.510)],('Electricity','DLinear'):[(.157,.261),(.168,.271),(.185,.288),(.218,.317)],('Electricity','iTransformer'):[(.153,.260),(.173,.279),(.202,.311),(.250,.348)]}
  return pairs[dataset,backbone][[96,192,336,720].index(h)]
 if model=='DAG':return {('NP',24):(.202,.237),('NP',360):(.521,.451),('PJM',24):(.057,.143),('PJM',360):(.130,.218)}[dataset,h]
 return {('NP',24):(.171,.217),('NP',360):(.423,.420),('PJM',24):(.056,.141),('PJM',360):(.118,.216)}[dataset,h]
