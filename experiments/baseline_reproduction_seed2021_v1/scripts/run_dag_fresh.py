"""Original DAG TFB training command with epoch-resume and metric observation."""
from common import *
from dag_data import *
import argparse,copy,difflib,inspect,runpy,shlex,textwrap,traceback
def execute(dataset,horizon,preflight_only=False):
 out=ROOT/('reports/dag_runner_preflight' if preflight_only else 'results/dag')/f'{dataset}_H{horizon}_seed2021';out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').is_file():
  old=json.loads((out/'metrics.json').read_text());assert sha(old['checkpoint_path'])==old['checkpoint_sha256'];print('RESUME_SKIP_VERIFIED',out.name);return
 src=ROOT/'upstream/DAG';sys.path.insert(0,str(src));os.chdir(src)
 from ts_benchmark.baselines.dag.dag import DAG
 import ts_benchmark.baselines.deep_forecasting_model_base as base_module
 from ts_benchmark.recording import read_record_file
 line=next(l for l in (src/'scripts/w_future/DAG.sh').read_text().splitlines() if f'"{dataset}.csv"' in l and f'"horizon": {horizon},' in l)
 argv=shlex.split(line)[1:];hp=json.loads(argv[argv.index('--model-hyper-params')+1])
 argv[argv.index('--save-path')+1]='baseline_reproduction/'+dataset+'/H'+str(horizon)
 argv+=['--eval-backend','sequential','--seed','2021','--deterministic','efficient','--save-true-pred','True','--report-method','csv']
 write_json(out/'config.json',{'published_hyperparameters':hp,'source_commit':COMMITS['DAG'],'run_type':'fresh_run','architecture_modified':False,'data_sha256':DATA_SHA[dataset]})
 (out/'command.txt').write_text('Published:\n'+line+'\nExecuted:\n'+json.dumps(argv),encoding='utf-8');(out/'source_commit.txt').write_text(COMMITS['DAG'])
 state={'history':[],'start':time.time(),'previous_seconds':0.};snapshot=out/'resume.pt'
 def begin(g):
  if not snapshot.is_file():return 0
  ck=torch.load(snapshot,map_location='cpu',weights_only=False);self=g['self'];self.model.load_state_dict(ck['model']);g['optimizer'].load_state_dict(ck['optimizer']);self.early_stopping.__dict__.update(ck['early_stopping']);self.check_point=ck['best_checkpoint'];restore_rng(ck['rng']);state['history']=ck['history'];state['previous_seconds']=ck['train_seconds']
  print('DAG_EPOCH_RESUME',ck['next_epoch'],flush=True);return g['config'].num_epochs if self.early_stopping.early_stop else ck['next_epoch']
 def record(g):
  state['history'].append({'epoch':g['epoch']+1,'valid_mae':float(g['valid_loss']),'lr':g['optimizer'].param_groups[0]['lr']});write_csv(out/'history.csv',state['history'])
  write_json(out/'status.json',{'status':'RUNNING','epoch':g['epoch']+1,'pid':os.getpid()})
 def save(g):
  self=g['self'];payload={'model':self.model.state_dict(),'optimizer':g['optimizer'].state_dict(),'early_stopping':self.early_stopping.__dict__,'best_checkpoint':self.check_point,'rng':rng(),'history':state['history'],'next_epoch':g['epoch']+1,'train_seconds':state['previous_seconds']+time.time()-state['train_start']}
  tmp=snapshot.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(snapshot)
 original=inspect.getsource(DAG.forecast_fit);patched=original
 for before,after in [('        for epoch in range(config.num_epochs):','        start_epoch = _audit_begin(locals())\n        for epoch in range(start_epoch, config.num_epochs):'),('                improved = self.early_stopping(valid_loss, self.model)','                _audit_record(locals())\n                improved = self.early_stopping(valid_loss, self.model)'),('                if self.early_stopping.early_stop:\n                    break','                if self.early_stopping.early_stop:\n                    _audit_save(locals())\n                    break'),('            if self.config.lradj != "TST":\n                self._adjust_lr(optimizer, epoch + 1, config)', '            if self.config.lradj != "TST":\n                self._adjust_lr(optimizer, epoch + 1, config)\n            _audit_save(locals())')]:
  assert patched.count(before)==1,before;patched=patched.replace(before,after)
 (ROOT/'patches/dag_audit_resume.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='official/forecast_fit',tofile='instrumented/forecast_fit')),encoding='utf-8')
 ns=dict(base_module.__dict__);ns.update({'_audit_begin':begin,'_audit_record':record,'_audit_save':save});exec(compile(textwrap.dedent(patched),str(src/'ts_benchmark/baselines/deep_forecasting_model_base.py'),'exec'),ns);original_fit=ns['forecast_fit']
 def observed(self,*a,**kw):
  state['train_start']=time.time();torch.cuda.reset_peak_memory_stats();original_fit(self,*a,**kw);torch.cuda.synchronize();state['train_seconds']=state['previous_seconds']+time.time()-state['train_start'];state['peak_gpu_mb']=torch.cuda.max_memory_allocated()/1024**2
  self.model.load_state_dict(self.check_point['Model']);self.model.eval();torch.save({'model':self.check_point['Model'],'config':vars(self.config)},out/'checkpoint.pt');write_json(out/'resolved_config.json',vars(self.config));state['params']=sum(p.numel() for p in self.model.parameters())
  df,y,c,tr,tv,sy,sc=load_data(dataset);assert np.allclose(sy.mean_,self.scaler1.mean_) and np.allclose(sc.mean_,self.scaler2.mean_)
  np.savez(out/'scaler.npz',mean_y=sy.mean_,scale_y=sy.scale_,mean_c=sc.mean_,scale_c=sc.scale_)
  test=Windows(y,c,hp['seq_len'],horizon,tv,len(y));preds=[];targets=[];origins=[];start=time.time()
  with torch.no_grad():
   for yh,ch,cf,target,o in torch.utils.data.DataLoader(test,batch_size=64,shuffle=False,drop_last=False):
    pred,_=self.model(torch.cat([yh,ch],-1).cuda(),cf.cuda());preds.append(pred[:,-horizon:,:1].cpu().numpy());targets.append(target.numpy());origins.append(o.numpy())
  torch.cuda.synchronize();state['eval_seconds']=time.time()-start;pred=np.concatenate(preds);true=np.concatenate(targets);origins=np.concatenate(origins);mse,mae=independent_metrics(pred,true)
  np.savez_compressed(out/'predictions.npz',pred=pred,true=true,origins=origins);state['score']={'mse':mse,'mae':mae,'windows':len(test)};return self
 DAG.forecast_fit=observed
 if preflight_only:
  from ts_benchmark.baselines.dag.models.dag_model import DAGModel
  adapter=DAG(**hp);adapter._init_criterion();cfg=adapter.config;cfg.series_dim=1;cfg.input_dim=3;cfg.output_dim=1;cfg.enc_in=3;cfg.dec_in=3;cfg.c_out=3;cfg.label_len=cfg.seq_len//2
  model=DAGModel(cfg).float().eval()
  with torch.no_grad():pred,_=model(torch.zeros(2,hp['seq_len'],3),torch.zeros(2,horizon,2))
  assert pred.shape==(2,horizon,1) and torch.isfinite(pred).all()
  write_json(out/'preflight.json',{'passed':True,'published_command_parsed':True,'official_training_hook_compiled':True,'cpu_forward_shape':list(pred.shape),'fresh_training_executed':False});print('DAG_RUNNER_PREFLIGHT_PASSED',dataset,horizon,flush=True);return
 try:
  sys.argv=argv;runpy.run_path(str(src/'scripts/run_benchmark.py'),run_name='__main__');assert 'score' in state,'Official runner failed'
  records=[]
  for p in (src/'result/baseline_reproduction'/dataset/('H'+str(horizon))).glob('*.tar.gz'):
   for _,r in read_record_file(str(p)).iterrows():
    if np.isfinite(r['mse_norm']):records.append((p,r))
  assert records;record,r=records[-1];score=state['score'];delta=max(abs(score['mse']-float(r['mse_norm'])),abs(score['mae']-float(r['mae_norm'])));assert delta<2e-5
  write_csv(out/'official_metric_record.csv',[{k:r[k] for k in r.index if k not in ['actual_data','inference_data']}]);pm,pa=paper_reference('DAG',dataset,'DAG',horizon);diff=abs(score['mse']/pm-1)
  metrics={'model':'DAG','dataset':dataset,'backbone':'DAG','input_len':hp['seq_len'],'horizon':horizon,'seed':2021,'source_type':'official_source','run_type':'fresh_run','paper_mse':pm,'paper_mae':pa,'reproduced_mse':score['mse'],'reproduced_mae':score['mae'],'relative_mse_diff':diff,'best_epoch':min(state['history'],key=lambda r:r['valid_mae'])['epoch'],'epochs':len(state['history']),'train_seconds':state['train_seconds'],'eval_seconds':state['eval_seconds'],'peak_gpu_mb':state['peak_gpu_mb'],'num_parameters':state['params'],'config_path':str(out/'config.json'),'checkpoint_path':str(out/'checkpoint.pt'),'checkpoint_sha256':sha(out/'checkpoint.pt'),'predictions_sha256':sha(out/'predictions.npz'),'data_sha256':DATA_SHA[dataset],'source_commit':COMMITS['DAG'],'official_mse':float(r['mse_norm']),'official_mae':float(r['mae_norm']),'independent_verification':dict(score,max_metric_abs_diff=delta),'exit_code':0,'start_time':state['start'],'end_time':time.time(),'status':'MATCHED' if diff<=.05 else 'CLOSE' if diff<=.1 else 'NOT_MATCHED'}
  write_json(out/'metrics.json',metrics);write_json(out/'status.json',{'status':metrics['status'],'exit_code':0});print('DAG_FRESH_RESULT',json.dumps(metrics),flush=True)
 except BaseException:
  write_json(out/'status.json',{'status':'FAILED','exit_code':1,'traceback':traceback.format_exc()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NP','PJM'],required=True);p.add_argument('--horizon',type=int,choices=[24,360],required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();execute(a.dataset,a.horizon,a.preflight_only)
