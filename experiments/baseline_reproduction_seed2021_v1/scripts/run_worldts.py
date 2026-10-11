"""Released official numerical WorldTS runner with observational/resume hooks."""
from common import *
from dag_data import *
from worldts_config import *
import argparse,copy,difflib,inspect,runpy,traceback
def execute(dataset,horizon):
 out=ROOT/'results/worldts'/f'{dataset}_H{horizon}_seed2021';out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').is_file():
  old=json.loads((out/'metrics.json').read_text());assert sha(old['checkpoint_path'])==old['checkpoint_sha256'];print('RESUME_SKIP_VERIFIED',out.name);return
 assert len(list((ROOT/'results/dag').glob('*/metrics.json')))==4,'DAG metric verification must finish first'
 src=ROOT/'upstream/WorldTS/DAG-bench';sys.path.insert(0,str(src));os.chdir(src)
 from worldts_preflight import test
 test(dataset,horizon)
 from ts_benchmark.baselines.worldts.worldts import WorldTS
 from ts_benchmark.baselines.worldts.losses.latent_state_loss import LatentStateLoss
 import ts_benchmark.baselines.worldts.worldts as world_module
 line,argv,hp=published_config(dataset,horizon);published=copy.deepcopy(hp)
 argv[argv.index('--gpus')+1]='0'  # Expand the official shell GPU variable on Windows.
 hp.update({'stage_save_dir':str(out/'stages'),'stage1_checkpoint_path':str(out/'stages/stage1_latent.pt'),'final_checkpoint_path':str(out/'stages/final.pt'),'log_stage1_loss_components':1,'log_stage2_loss_components':1})
 argv[argv.index('--model-hyper-params')+1]=json.dumps(hp);argv[argv.index('--save-path')+1]='baseline_reproduction/'+dataset+'/H'+str(horizon)
 argv+=['--eval-backend','sequential','--save-true-pred','True','--report-method','csv']
 os.environ['WORLDTS_NUMERIC_DATA']=str(ROOT/'data')
 write_json(out/'config.json',{'published_hyperparameters':published,'executed_hyperparameters':hp,'source_commit':COMMITS['WorldTS'],'source_type':'official_source','data_sha256':DATA_SHA[dataset],'architecture_modified':False,'changes':['Windows output paths','loss-component logging','sequential evaluation','epoch resume snapshots','independent metrics']})
 (out/'source_commit.txt').write_text(COMMITS['WorldTS']);(out/'command.txt').write_text('Published:\n'+line+'\nExecuted argv:\n'+json.dumps(argv),encoding='utf-8')
 state={'history':[],'components':{},'start':time.time(),'stage_hash':None,'previous_seconds':0.,'total_train_seconds':0.,'resumed':False,'recorded_training_peak_mb':0.}
 write_json(out/'status.json',{'status':'RUNNING','run_id':out.name,'pid':os.getpid(),'start_time':state['start']})
 # Save actual component diagnostics without changing the differentiable return.
 original_loss=LatentStateLoss.forward
 def loss_observed(self,*a,**kw):
  values=original_loss(self,*a,**kw);mode='train' if torch.is_grad_enabled() else 'validation';bucket=state['components'].setdefault(mode,{})
  for key,value in values.items():
   if value.numel()==1:bucket.setdefault(key,[]).append(float(value.detach()))
  return values
 LatentStateLoss.forward=loss_observed
 original_process=WorldTS._process
 def process_observed(self,*a,**kw):
  result=original_process(self,*a,**kw)
  if str(self.config.training_stage).startswith('stage2'):
   mode='train' if torch.is_grad_enabled() else 'validation';bucket=state['components'].setdefault(mode,{})
   for k in ['decoder_pred_loss','decoder_warm_loss','decoder_warm_weighted_loss']:
    if k in result:bucket.setdefault(k,[]).append(float(result[k].detach()))
  return result
 WorldTS._process=process_observed
 def stage_begin(g):
  stage=g['stage'];model=g['model'];state['current_stage']=stage;state['components']={};state['stage_start']=time.time()
  if stage.startswith('stage2'):
   state['stage_hash']=[tensor_sha(m) for m in model._stage1_modules()];assert all(not p.requires_grad for m in model._stage1_modules() for p in m.parameters())
  else:assert not any(p.requires_grad for p in model.decoder.parameters())
  snapshot=out/('resume_'+stage+'.pt');state['snapshot']=snapshot;state['previous_seconds']=0.
  completed=out/('completed_'+stage+'.json')
  if completed.is_file():
   state['resumed']=True
   stagept=out/'stages'/(stage+'.pt');ck=torch.load(stagept,map_location='cpu',weights_only=False);model.load_state_dict(ck['Model']);meta=json.loads(completed.read_text());state['total_train_seconds']+=meta['train_seconds'];state['history']+=meta['history'];restore_rng(torch.load(out/('boundary_'+stage+'.pt'),map_location='cpu',weights_only=False));print('STAGE_RESUME_COMPLETED',stage,flush=True)
   state['recorded_training_peak_mb']=max(state['recorded_training_peak_mb'],meta.get('peak_gpu_mb',0.));return g['num_epochs'],None,None
  if snapshot.is_file():
   state['resumed']=True
   ck=torch.load(snapshot,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);g['optimizer'].load_state_dict(ck['optimizer']);g['early_stop'].__dict__.update(ck['early_stop']);restore_rng(ck['rng']);state['history']+=ck['history'];state['previous_seconds']=ck['train_seconds'];start=ck['next_epoch']
   if g['early_stop'].early_stop:start=g['num_epochs']
   print('STAGE_EPOCH_RESUME',stage,start,flush=True);return start,ck['best_loss'],ck['best_state']
  return 0,None,None
 def epoch_record(g):
  stage=g['stage'];epoch=g['epoch']+1
  row={'stage':stage,'epoch':epoch,'validation_loss':float(g['valid_loss']),'lr':g['optimizer'].param_groups[0]['lr'],'elapsed_seconds':state['previous_seconds']+time.time()-state['stage_start']}
  for mode,bucket in state['components'].items():
   for k,v in bucket.items():row[mode+'_'+k]=float(np.mean(v))
  state['history'].append(row);write_csv(out/'history.csv',state['history']);state['components']={}
  write_json(out/'status.json',{'run_id':out.name,'status':'RUNNING','stage':stage,'epoch':epoch,'validation_loss':row['validation_loss'],'pid':os.getpid()})
 def epoch_save(g):
  localhistory=[r for r in state['history'] if r['stage']==g['stage']]
  payload={'model':g['model'].state_dict(),'optimizer':g['optimizer'].state_dict(),'early_stop':g['early_stop'].__dict__,'rng':rng(),'history':localhistory,'next_epoch':g['epoch']+1,'best_loss':g['stage_best_loss'],'best_state':g['stage_best_state'],'train_seconds':state['previous_seconds']+time.time()-state['stage_start']}
  tmp=state['snapshot'].with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(state['snapshot'])
 def stage_end(g):
  stage=g['stage'];model=g['model'];meta=out/('completed_'+stage+'.json')
  if stage.startswith('stage2'):
   current=[tensor_sha(m) for m in model._stage1_modules()];assert current==state['stage_hash'];write_json(out/'freeze_audit.json',{'before':state['stage_hash'],'after':current,'unchanged':True,'only_decoder_trainable':all(not p.requires_grad for m in model._stage1_modules() for p in m.parameters())})
  if not meta.is_file():
   cost=state['previous_seconds']+time.time()-state['stage_start'];state['total_train_seconds']+=cost
   peak=torch.cuda.max_memory_allocated()/1024**2;state['recorded_training_peak_mb']=max(state['recorded_training_peak_mb'],peak)
   torch.save(rng(),out/('boundary_'+stage+'.pt'));write_json(meta,{'stage':stage,'train_seconds':cost,'peak_gpu_mb':peak,'history':[r for r in state['history'] if r['stage']==stage]})
  write_csv(out/'history.csv',state['history'])
 original_stage=inspect.getsource(WorldTS._run_stage_train);patched=original_stage
 replacements=[('        for epoch in range(num_epochs):','        start_epoch, stage_best_loss, stage_best_state = _audit_stage_begin(locals())\n        for epoch in range(start_epoch, num_epochs):'),('                    if early_stop is not None:','                    _audit_epoch_record(locals())\n                    if early_stop is not None:'),('                        if early_stop.early_stop:\n                            break','                        if early_stop.early_stop:\n                            _audit_epoch_save(locals())\n                            break'),('                self._adjust_lr(optimizer, epoch + 1, stage_lr_config)','                self._adjust_lr(optimizer, epoch + 1, stage_lr_config)\n            _audit_epoch_save(locals())'),('        self._load_checkpoint_state(stage)','        self._load_checkpoint_state(stage)\n        _audit_stage_end(locals())')]
 for before,after in replacements:assert patched.count(before)==1,before;patched=patched.replace(before,after)
 (ROOT/'patches').mkdir(parents=True,exist_ok=True);(ROOT/'patches/worldts_audit_resume.diff').write_text(''.join(difflib.unified_diff(original_stage.splitlines(True),patched.splitlines(True),fromfile='upstream/WorldTS/worldts.py:_run_stage_train',tofile='instrumented/WorldTS/worldts.py:_run_stage_train')),encoding='utf-8')
 ns=dict(world_module.__dict__);ns.update({'_audit_stage_begin':stage_begin,'_audit_epoch_record':epoch_record,'_audit_epoch_save':epoch_save,'_audit_stage_end':stage_end})
 import textwrap
 exec(compile(textwrap.dedent(patched),str(src/'ts_benchmark/baselines/worldts/worldts.py'),'exec'),ns);WorldTS._run_stage_train=ns['_run_stage_train']
 original_fit=WorldTS.forecast_fit
 def fit_observed(self,*a,**kw):
  torch.cuda.reset_peak_memory_stats();begin=time.time();original_fit(self,*a,**kw);torch.cuda.synchronize()
  model=self._unwrap_model(self.model);model.eval();state['params']=sum(p.numel() for p in model.parameters());state['peak_gpu_mb']=max(state['recorded_training_peak_mb'],torch.cuda.max_memory_allocated()/1024**2)
  state['peak_gpu_scope']='original training' if state['recorded_training_peak_mb'] else 'resumed fit before independent evaluation; initial training peak was not persisted'
  write_json(out/'resource_audit.json',{'peak_gpu_mb':state['peak_gpu_mb'],'peak_gpu_scope':state['peak_gpu_scope'],'resumed':state['resumed'],'train_seconds':state['total_train_seconds']})
  write_json(out/'resolved_config.json',vars(self.config));checkpoint=pathlib.Path(self.config.final_checkpoint_path);assert checkpoint.is_file();state['checkpoint']=checkpoint
  df,y,c,tr,tv,sy,sc=load_data(dataset);assert np.allclose(sy.mean_,self.scaler1.mean_) and np.allclose(sy.scale_,self.scaler1.scale_);assert np.allclose(sc.mean_,self.scaler2.mean_) and np.allclose(sc.scale_,self.scaler2.scale_)
  np.savez(out/'scaler.npz',mean_y=sy.mean_,scale_y=sy.scale_,mean_c=sc.mean_,scale_c=sc.scale_)
  testset=Windows(y,c,hp['seq_len'],horizon,tv,len(y));loader=torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False,drop_last=False)
  preds=[];targets=[];origins=[];start=time.time()
  with torch.no_grad():
   for yh,ch,cf,target,o in loader:
    pred=self._predict_only(torch.cat([yh,ch],-1).cuda(),None,None,exog_future=cf.cuda());preds.append(pred[:,-horizon:,:1].cpu().numpy());targets.append(target.numpy());origins.append(o.numpy())
  torch.cuda.synchronize();state['eval_seconds']=time.time()-start;pred=np.concatenate(preds);true=np.concatenate(targets);origins=np.concatenate(origins);mse,mae=independent_metrics(pred,true)
  np.savez_compressed(out/'predictions.npz',pred=pred,true=true,origins=origins);state['metrics']={'mse':mse,'mae':mae,'windows':len(testset)}
  write_json(out/'independent_metrics.json',state['metrics']);print('WORLDTS_INDEPENDENT_EVALUATION',json.dumps(state['metrics']),flush=True)
  return self
 WorldTS.forecast_fit=fit_observed
 try:
  sys.argv=argv;runpy.run_path(str(src/'scripts/run_benchmark.py'),run_name='__main__');assert 'metrics' in state,'Official runner failed; inspect log'
  from ts_benchmark.recording import read_record_file
  records=[]
  for record in (src/'result/baseline_reproduction'/dataset/('H'+str(horizon))).glob('*.tar.gz'):
   df=read_record_file(str(record))
   for _,r in df.iterrows():
    if np.isfinite(r['mse_norm']):records.append((record,r))
  assert records,'No official evaluation archive produced'
  record,r=records[-1];score=state['metrics'];delta=max(abs(float(r['mse_norm'])-score['mse']),abs(float(r['mae_norm'])-score['mae']));assert delta<2e-5
  write_csv(out/'official_metric_record.csv',[{k:r[k] for k in r.index if k not in ['actual_data','inference_data']}])
  pm,pa=paper_reference('WorldTS',dataset,'PatchTST-state-transition',horizon);diff=abs(score['mse']/pm-1);s2=[r for r in state['history'] if r['stage']=='stage2_decoder_mix'];s1=[r for r in state['history'] if r['stage']=='stage1_latent']
  metrics={'model':'WorldTS','dataset':dataset,'backbone':'PatchTST-state-transition','input_len':hp['seq_len'],'horizon':horizon,'seed':2021,'source_type':'official_source','run_type':'fresh_run','paper_mse':pm,'paper_mae':pa,'reproduced_mse':score['mse'],'reproduced_mae':score['mae'],'relative_mse_diff':diff,'best_epoch':min(s2,key=lambda r:r['validation_loss'])['epoch'],'stage1_best_epoch':min(s1,key=lambda r:r['validation_loss'])['epoch'],'stage1_epochs':len(s1),'stage2_epochs':len(s2),'train_seconds':state['total_train_seconds'],'eval_seconds':state['eval_seconds'],'peak_gpu_mb':state['peak_gpu_mb'],'num_parameters':state['params'],'config_path':str(out/'config.json'),'checkpoint_path':str(state['checkpoint']),'checkpoint_sha256':sha(state['checkpoint']),'predictions_sha256':sha(out/'predictions.npz'),'data_sha256':DATA_SHA[dataset],'source_commit':COMMITS['WorldTS'],'official_mse':float(r['mse_norm']),'official_mae':float(r['mae_norm']),'official_record_path':str(record),'independent_verification':dict(score,max_metric_abs_diff=delta),'status':'MATCHED' if diff<=.05 else 'CLOSE' if diff<=.1 else 'NOT_MATCHED','exit_code':0,'start_time':state['start'],'end_time':time.time()}
  metrics.update({'peak_gpu_scope':state['peak_gpu_scope'],'resumed':state['resumed']});write_json(out/'metrics.json',metrics);write_json(out/'status.json',{'status':metrics['status'],'exit_code':0});print('WORLDTS_RESULT',json.dumps(metrics),flush=True)
 except BaseException:
  write_json(out/'status.json',{'status':'FAILED','exit_code':1,'traceback':traceback.format_exc(),'run_id':out.name});raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NP','PJM'],required=True);p.add_argument('--horizon',type=int,choices=[24,360],required=True);a=p.parse_args();execute(a.dataset,a.horizon)
