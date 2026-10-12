import os,sys,pathlib,json,csv,hashlib,random,time,ctypes,inspect
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=pathlib.Path(os.environ.get('WORLDTS_BASELINE_ROOT','D:/baseline_reproduction_seed2021_v1'))
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['WANDB_MODE']='disabled'
import numpy as np,torch,pandas as pd
from torch import nn
from types import SimpleNamespace
torch.set_num_threads(4)
if os.name=='nt' and (os.cpu_count() or 0)>=32:
 f=ctypes.windll.kernel32.SetProcessAffinityMask;f.argtypes=[ctypes.c_void_p,ctypes.c_size_t];f.restype=ctypes.c_int;assert f(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
sys.path.insert(0,str(ROOT/'upstream/WorldTS/DAG-bench'))
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel
from ts_benchmark.baselines.worldts.losses.latent_state_loss import LatentStateLoss
from ts_benchmark.baselines.worldts.worldts import WorldTS
DATA_SHA={'NP':'29e142d895fa3254bf138e3bec32051e081794a7393a642db36a5d21859d36e1','PJM':'14feec9daa86e14ba4ed062c690468fa615af6fb18ba56006149b087e26058e5'}
PROTOCOL=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8'))
def write_json(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
 for n in range(100):
  try:tmp.replace(p);return
  except PermissionError:
   if n==99:raise
   time.sleep(.05)
def write_csv(p,rows):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def state_hash(model,part='all'):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  if part=='frozen' and k.startswith('predictor.'):continue
  if part=='predictor' and not k.startswith('predictor.'):continue
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def seed(value=2021):random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def config(ds,h=360):
 directory=BASE/'results/worldts'/f'{ds}_H{h}_seed2021'
 for name in ['resolved_config.json','metrics.json','history.csv','predictions.npz','scaler.npz']:assert (directory/name).is_file(),str(directory/name)
 cfg=json.loads((directory/'resolved_config.json').read_text(encoding='utf-8'));metadata=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
 checkpoint=pathlib.Path(metadata['checkpoint_path']);assert checkpoint.is_file() and checkpoint.resolve().is_relative_to(BASE.resolve());assert sha(checkpoint)==metadata['checkpoint_sha256']
 assert checkpoint.name=='stage2_decoder_mix.pt','Complete Stage2 checkpoint required'
 return SimpleNamespace(**cfg),directory,metadata,checkpoint
def load_model(ds,h=360,predictor_grad=False):
 seed();cfg,directory,meta,checkpoint=config(ds,h);model=WorldTSModel(cfg);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['Model'],strict=True);model=model.cuda().eval()
 model.requires_grad_(False)
 if predictor_grad:model.predictor.requires_grad_(True)
 return model,cfg,directory,meta
def predictor_mode(model,training):
 model.eval();model.predictor.train(training)
 assert all(not m.training for name,m in model.named_children() if name!='predictor')
 assert all(p.requires_grad==name.startswith('predictor.') for name,p in model.named_parameters())
def criterion(cfg):
 names=set(inspect.signature(LatentStateLoss.__init__).parameters)-{'self'}
 return LatentStateLoss(**{k:v for k,v in vars(cfg).items() if k in names}).cuda()
def data(ds):
 from sklearn.preprocessing import StandardScaler
 path=BASE/'data'/(ds+'.csv');assert path.is_file() and sha(path)==DATA_SHA[ds]
 raw=pd.read_csv(path);cols=raw.cols.drop_duplicates().tolist();raw.date=pd.to_datetime(raw.date);df=raw.pivot(index='date',columns='cols',values='data')[cols]
 assert len(df)==52416 and cols[-1]=='OT' and len(cols)==3 and np.isfinite(df.to_numpy()).all()
 assert df.index.is_unique and (df.index[1:]-df.index[:-1]==pd.Timedelta(hours=1)).all()
 tv=int(len(df)*.8);tr=int(tv*.875);sy=StandardScaler().fit(df.iloc[:tr,-1:]);sc=StandardScaler().fit(df.iloc[:tr,:-1])
 y=torch.from_numpy(sy.transform(df.iloc[:,-1:]).astype('float32'));c=torch.from_numpy(sc.transform(df.iloc[:,:-1]).astype('float32'))
 return {'df':df,'y':y,'c':c,'tr':tr,'tv':tv,'sy':sy,'sc':sc}
def origins(d,split,t=720,h=360):
 begin,end={'train':(t,d['tr']),'val':(d['tr'],d['tv']),'test':(d['tv'],len(d['y']))}[split]
 return np.arange(begin,end-h+1,dtype='int64')
def batch(d,oo,t=720,h=360):
 o=torch.as_tensor(np.asarray(oo),dtype=torch.long);hist=o[:,None]+torch.arange(-t,0)[None,:];future=o[:,None]+torch.arange(h)[None,:]
 return torch.cat([d['y'][hist],d['c'][hist]],-1).cuda(),d['c'][future].cuda(),d['y'][future].cuda()
def forward(model,x,cf,target):return model(x,x_future=cf,future_target=target)
def decode(model,h,out):return WorldTS._decode_model_states(model,h,out)
def losses(model,out,target,crit):
 h=out['h_pred'];z=out['h_target'];latent=crit(h_pred=h,h_target=z,z_visreg_hist=out.get('z_visreg_hist'),z_visreg_target=out.get('z_visreg_target'),z_visreg_pred=out.get('z_visreg_pred'))
 yp=decode(model,h,out);forecast=torch.nn.functional.l1_loss(yp,target)
 l1=torch.nn.functional.l1_loss(h,z.detach());cos=1-torch.nn.functional.cosine_similarity(h.flatten(1),z.detach().flatten(1),dim=-1).mean()
 reg_pred=crit.sigreg_loss(h)['loss'];reg_target=crit.sigreg_loss(z)['loss']
 return {'full':latent['loss'],'l1':l1,'cosine':cos,'regularizer_pred':reg_pred,'regularizer_target':reg_target,'forecast':forecast,'forecast_mse':torch.nn.functional.mse_loss(yp,target),'pred':yp}
def require_gradients():
 if not torch.is_grad_enabled():raise RuntimeError('Gradient audit cannot run under torch.no_grad()')
def cosine_rows(a,b):
 a=a.detach().flatten(1).double();b=b.detach().flatten(1).double();na=a.norm(dim=1);nb=b.norm(dim=1);finite=torch.isfinite(a).all(1)&torch.isfinite(b).all(1);za=na<=1e-12;zb=nb<=1e-12;valid=finite&~za&~zb
 values=torch.full_like(na,float('nan'));values[valid]=(a[valid]*b[valid]).sum(1)/(na[valid]*nb[valid]+1e-12)
 status=['VALID' if v else 'NONFINITE' if not f else 'ZERO_BOTH' if x and y else 'ZERO_L' if x else 'ZERO_Y' for v,f,x,y in zip(valid.cpu().tolist(),finite.cpu().tolist(),za.cpu().tolist(),zb.cpu().tolist())]
 return values.cpu().numpy(),na.cpu().numpy(),nb.cpu().numpy(),status
def flat_gradient(loss,parameters,retain_graph=True):
 require_gradients();grads=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
 return torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(parameters,grads)]),sum(g is None for g in grads)
def metrics(pred,true):
 assert pred.shape==true.shape and np.isfinite(pred).all() and np.isfinite(true).all();e=pred.astype('float64')-true.astype('float64')
 return float(np.mean(e*e)),float(np.mean(np.abs(e)))
def evaluate(model,d,cfg,split='val',save=None,batch_size=32):
 model.eval();oo=origins(d,split,cfg.seq_len,cfg.horizon);pred=[];true=[];means=[]
 with torch.no_grad():
  for start in range(0,len(oo),batch_size):
   x,cf,y=batch(d,oo[start:start+batch_size],cfg.seq_len,cfg.horizon);p=model.forecast(x,x_future=cf);pred.append(p.cpu().numpy());true.append(y.cpu().numpy());means.append(float(torch.nn.functional.l1_loss(p,y)))
 pred=np.concatenate(pred);true=np.concatenate(true);mse,mae=metrics(pred,true)
 if save is not None:np.savez_compressed(save,pred=pred,true=true,origins=oo)
 return {'mse':mse,'mae':mae,'selection_mae_batch_mean':float(np.mean(means)),'n_origins':len(oo)},pred,true,oo
def module_stats(model):return [{'module':name,'parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'training':m.training} for name,m in model.named_children()]
