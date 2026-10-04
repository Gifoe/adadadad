import argparse,csv,hashlib,json,pathlib,random,sys,time,traceback,os
# Process-only CPU affinity workaround for repeated native interpreter faults.
# It does not alter GPU/model/data/optimization settings or other processes.
if sys.platform=='win32' and os.environ.get('V3_CPU_AFFINITY'):
 import ctypes
 api=ctypes.windll.kernel32.SetProcessAffinityMask;api.argtypes=[ctypes.c_void_p,ctypes.c_size_t];api.restype=ctypes.c_int
 if not api(ctypes.c_void_p(-1),int(os.environ['V3_CPU_AFFINITY'],16)):raise ctypes.WinError()
import numpy as np
import torch
from model_carrier import model,objectives,RecurrentGPT2Block
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'outputs';CK=ROOT/'checkpoints';DATA=ROOT/'data'
START=ROOT/'official_artifacts/r2_checkpoint_epoch_2765.pt'
CONFIG=dict(seed=42,H_train=6,H_train_evidence='unique final-stage batch-count divisor plus official Figure5 R2 ID6 label',official_training_recurrence=2,official_validation_K=2,K_train=5,lambda_transition=1.,updates=5000,smoke_updates=200,batch_size=128,lr=1e-5,weight_decay=.01,warmup_steps=0,lr_schedule='constant',precision='bf16',d_model=768,layers=4,heads=12,NoPE=True,dropout=0,pred_pos='last_token',max_len=50,existing_final_LN=True,no_new_parameters=True,train_max_hop=5,D=[2,3,4,5,6,8,10,12,16,20],ID=[2,3,4,5,6],OOD=[8,10,12,16,20],K=[1,2,3,4,5,6,8,10,12,16,20,24],prototype_k=[2,4],prototype_min_count=10,intervention_per_D_k=200,intervention_eligible='D>k, at least one remaining transition',intervention_main_K='max(D,H_train)',intervention_extra_K='min(D+4,24)',validation_threshold=.85,retention_harm_pp=5,mechanism_effect_pp=20,mechanism_primary='equal D,k cell mean at main K; compare to pretrained/final-only/noise; trajectory also +20pp',pretrained_mechanism_absolute_threshold=.5,same_state_correct_preservation_threshold=.5,utility_gain_pp=5,checkpoint_sha256='4e89fde1f592d7afece63a6182f8427beedc85a216adb474777830b6b56495d8',upstream_commit='bb22b192977b1b136fde985c8e7a2392d172d97a')
def dump(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def save(p,v):
 tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,rows,fields=None):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def status(phase,**kw):dump(ROOT/'STATUS.json',dict(phase=phase,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(phase,kw,flush=True)
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def load_data():return tuple({k:v for k,v in np.load(DATA/f'{n}.npz').items()} for n in ['train','test'])
def batch(data,ix):
 x=torch.as_tensor(data['input_ids'][ix],device='cuda');t=torch.as_tensor(data['state_targets'][ix],device='cuda');return x,(x!=0).long(),t
def tensor_sha(sd):
 h=hashlib.sha256()
 for k in sorted(sd):
  v=sd[k].detach().cpu().contiguous();h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()
def starting_state():
 assert sha(START)==CONFIG['checkpoint_sha256']
 c=torch.load(START,map_location='cpu',weights_only=True);s=c['model_state_dict'];return {k.removeprefix('_orig_mod.'):v for k,v in s.items()}
def load_model(name='pretrained',step=5000):
 m=model().cuda()
 s=starting_state() if name=='pretrained' else torch.load(CK/f'{name}_step{step}.pt',map_location='cpu',weights_only=True)['model_state_dict']
 m.load_state_dict(s,strict=True);m.eval();return m
@torch.no_grad()
def accuracy(m,data,D,K=5):
 m.eval();m.num_iterations=K;rows=[]
 for d in D:
  ix=np.flatnonzero(data['hops']==d);good=0
  for s in range(0,len(ix),128):
   x,mask,t=batch(data,ix[s:s+128])
   with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x,mask).logits[:,-1]
   good+=int((z.argmax(-1)==t[:,-1]).sum())
  rows.append(dict(D=d,K=K,correct=good,total=len(ix),accuracy=good/len(ix)))
 return rows
def unit_tests(m,train):
 ix=np.array([np.flatnonzero(train['hops']==d)[0] for d in range(1,6)]);x,mask,t=batch(train,ix);m.num_iterations=5;m.eval()
 with torch.autocast('cuda',dtype=torch.bfloat16):
  a=RecurrentGPT2Block.forward(m,x,mask).logits[:,-1];b=m(x,mask,True);loss,final,inter=objectives(b,t,0)
 assert torch.equal(a,b.final_logits) and float(abs(loss-final))==0
 assert m.lm_head.weight is m.token_embedding.weight
 direct=model();assert set(direct.state_dict())==set(m.state_dict());del direct
 m.zero_grad(set_to_none=True);loss.backward();gradnorm=float(torch.linalg.vector_norm(torch.stack([q.grad.float().norm() for q in m.parameters() if q.grad is not None])));assert np.isfinite(gradnorm) and gradnorm>0;m.zero_grad(set_to_none=True)
 # At zero dropout, a same raw-carrier patch is an exact no-op.
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  normal=m(x,mask,True);patched=m(x,mask,True,patch=(2,normal.states[1]));delta=float((normal.final_logits-patched.final_logits).abs().max())
 assert delta==0
 dump(OUT/'unit_tests.json',dict(status='PASS',original_forward_exact_equal=True,lambda0_equals_finalCE=True,tied_head=True,parameter_keys_unchanged=True,gradient_norm=gradnorm,raw_carrier_noop_max_difference=delta,vocab_size=217,state_token_added=False))
def train_arm(name,train,test,smoke=False):
 summary=OUT/(f'smoke_{name}.json' if smoke else f'training_{name}.json')
 if summary.exists():return json.loads(summary.read_text())
 seed();m=load_model();init_hash=tensor_sha(m.state_dict());plan=np.load(DATA/'batch_plan.npy');opt=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.01);start=0;previous=0.;budget=200 if smoke else 5000;resume=CK/f'{name}_resume.pt'
 if resume.exists() and not smoke:
  r=torch.load(resume,map_location='cpu',weights_only=False);m.load_state_dict(r['model_state_dict']);opt.load_state_dict(r['optimizer']);start=r['step'];previous=r['seconds'];torch.set_rng_state(r['torch_rng']);torch.cuda.set_rng_state_all(r['cuda_rng']);random.setstate(r['python_rng']);np.random.set_state(r['numpy_rng'])
 history=OUT/(f'smoke_history_{name}.csv' if smoke else f'finetune_history_{"baseline" if name=="final_only" else name}.csv');fields=['model','step','loss','final_loss','transition_loss','lr','train_batch_accuracy','elapsed_seconds','peak_vram_mib']
 if start and history.exists():
  committed=[r for r in csv.DictReader(history.open()) if int(r['step'])<=start];csvwrite(history,committed,fields)
 if not smoke and start==0:
  save(CK/f'{name}_step0.pt',dict(model_state_dict=m.state_dict(),step=0,initial_parameter_sha256=init_hash))
  csvwrite(OUT/f'{name}_id_dynamics.csv',[dict(step=0,**r) for r in accuracy(m,test,CONFIG['ID'],5)])
 dynamics=list(csv.DictReader((OUT/f'{name}_id_dynamics.csv').open())) if not smoke and (OUT/f'{name}_id_dynamics.csv').exists() else []
 dynamics=[r for r in dynamics if int(r['step'])<=start]
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();beg=time.perf_counter();first=[];last=[]
 if start and history.exists():
  prior=list(csv.DictReader(history.open()));first=[float(r['loss']) for r in prior[:20]];last=[float(r['loss']) for r in prior[-20:]]
 with history.open('a' if start else 'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  if start==0:w.writeheader()
  for step in range(start,budget):
   m.train();m.num_iterations=5;ix=plan[step];x,mask,t=batch(train,ix);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):o=m(x,mask,True);loss,lf,li=objectives(o,t,float(name=='grounded'))
   assert torch.isfinite(loss),(name,step)
   loss.backward();opt.step();v=float(loss.detach());first.append(v) if step<20 else None;last=(last+[v])[-20:]
   seconds=previous+time.perf_counter()-beg
   w.writerow(dict(model=name,step=step+1,loss=v,final_loss=float(lf.detach()),transition_loss=float(li.detach()),lr=opt.param_groups[0]['lr'],train_batch_accuracy=float((o.final_logits.argmax(-1)==t[:,-1]).float().mean()),elapsed_seconds=seconds,peak_vram_mib=torch.cuda.max_memory_allocated()/2**20))
   del o,loss,lf,li
   if (step+1)%50==0:f.flush();status(('SMOKE_' if smoke else 'FT_')+name,step=step+1,budget=budget,loss=v,seconds=seconds)
   if not smoke and (step+1)%500==0:
    torch.cuda.synchronize();seconds=previous+time.perf_counter()-beg
    save(resume,dict(model_state_dict=m.state_dict(),optimizer=opt.state_dict(),step=step+1,seconds=seconds,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state()))
   if not smoke and step+1 in [500,1000,2000,5000]:
    save(CK/f'{name}_step{step+1}.pt',dict(model_state_dict=m.state_dict(),step=step+1,initial_parameter_sha256=init_hash,lambda_transition=float(name=='grounded')))
    dynamics.extend(dict(step=step+1,**r) for r in accuracy(m,test,CONFIG['ID'],5));csvwrite(OUT/f'{name}_id_dynamics.csv',dynamics)
 result=dict(model=name,smoke=smoke,updates=budget,examples_seen=budget*128,seconds=previous+time.perf_counter()-beg,peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,initial_parameter_sha256=init_hash,first20_loss=float(np.mean(first)) if first else None,last20_loss=float(np.mean(last)),LR=1e-5,recurrence=5)
 dump(summary,result);del m,opt;torch.cuda.empty_cache();return result
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True);CK.mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
 if (OUT/'completion_audit.json').exists():status('ALREADY_COMPLETE');return
 dump(ROOT/'CONFIG.json',CONFIG)
 if not (DATA/'integrity_check.json').exists():
  from data_pipeline import prepare
  prepare(CONFIG['H_train'])
 train,test=load_data();seed()
 import platform,transformers
 meta=ROOT/'RUN_METADATA.json';old=json.loads(meta.read_text()) if meta.exists() else {};now=time.strftime('%Y-%m-%d %H:%M:%S')
 dump(meta,dict(start=old.get('start',now),process_starts=old.get('process_starts',[])+[now],python=sys.version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),host=platform.node(),process_CPU_affinity=os.environ.get('V3_CPU_AFFINITY'),checkpoint_sha256=sha(START),batch_plan_sha256=sha(DATA/'batch_plan.npy')))
 validation=OUT/'official_validation.json'
 if not validation.exists():
  m=load_model();unit_tests(m,train);beg=time.perf_counter();rows=accuracy(m,test,CONFIG['ID'],CONFIG['official_validation_K']);macro=float(np.mean([r['accuracy'] for r in rows]));dump(validation,dict(validated=macro>=.85,macro=macro,by_hop=rows,seconds=time.perf_counter()-beg));status('OFFICIAL_CHECKPOINT_VALIDATED',macro=macro,by_hop=rows);del m;torch.cuda.empty_cache()
 validated=json.loads(validation.read_text())['validated']
 if not validated:
  from report import finalize
  finalize('OFFICIAL CHECKPOINT INVALID');status('COMPLETE',decision='OFFICIAL CHECKPOINT INVALID');return
 from evaluate import audit
 if not (OUT/'pretrained_audit_complete.json').exists():audit('pretrained',train,test)
 for name in ['final_only','grounded']:train_arm(name,train,test,True)
 for name in ['final_only','grounded']:train_arm(name,train,test)
 assert json.loads((OUT/'training_final_only.json').read_text())['initial_parameter_sha256']==json.loads((OUT/'training_grounded.json').read_text())['initial_parameter_sha256']
 for name in ['final_only','grounded']:
  if not (OUT/f'{name}_audit_complete.json').exists():audit(name,train,test)
 from report import finalize,decision
 d=decision();finalize(d);status('COMPLETE',decision=d)
if __name__=='__main__':
 try:main()
 except BaseException as e:status('EXECUTION_ERROR',error=repr(e),traceback=traceback.format_exc());raise
