import pathlib,sys,argparse,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *
def run(ds,method,smoke=False):
 assert json.loads((ROOT/'outputs/preflight_metrics.json').read_text())['passed'];assert json.loads((ROOT/'outputs/tests_report.json').read_text())['passed']
 calibration=json.loads((ROOT/'configs/calibration.json').read_text());assert calibration['protocol_sha256']==sha(ROOT/'configs/protocol.json');weight=calibration['lambda'][ds]
 rid=f'{ds}_{method}_s2021';directory=ROOT/'outputs'/('smoke' if smoke else 'runs')/rid;directory.mkdir(parents=True,exist_ok=True)
 if (directory/'metrics.json').is_file():
  old=json.loads((directory/'metrics.json').read_text());assert smoke or sha(old['best_checkpoint'])==old['best_checkpoint_sha256'];print('SKIP_VERIFIED_COMPLETED',rid,smoke,flush=True);return
 model,cfg,_,meta=load_model(ds,predictor_grad=True);d=data(ds);crit=criterion(cfg);parameters=list(model.predictor.parameters());optimizer=torch.optim.Adam(parameters,lr=1e-4);frozen=state_hash(model,'frozen');initial=state_hash(model);initial_predictor=state_hash(model,'predictor')
 code={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/ops.py',ROOT/'scripts/train.py',ROOT/'configs/protocol.json',ROOT/'configs/calibration.json']}
 write_json(directory/'source_manifest.json',code)
 available=origins(d,'train');permutation=np.random.default_rng(2021).permutation(len(available));order=available[permutation];assert len(order)>=1000*32
 order_path=ROOT/'configs'/f'training_origin_order_{ds}.npy'
 if order_path.exists():assert np.array_equal(np.load(order_path),order)
 else:np.save(order_path,order)
 order_hash=sha(order_path);initial_rng_hash=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
 maximum=20 if smoke else 1000;curves=[];gradient_rows=[];completed_step=0;bad=0;elapsed=0.;best_step=0;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initial_val=None;best_val=None;latest=directory/'resume.pt';torch.cuda.reset_peak_memory_stats()
 if not smoke and latest.exists():
  snapshot=torch.load(latest,map_location='cpu',weights_only=False);assert snapshot['source_manifest']==code and snapshot['order_hash']==order_hash
  model.load_state_dict(snapshot['model'],strict=True);optimizer.load_state_dict(snapshot['optimizer']);restore_rng(snapshot['rng']);completed_step=snapshot['step'];bad=snapshot['bad'];elapsed=snapshot['elapsed'];best_step=snapshot['best_step'];best_weights=snapshot['best_weights'];best_val=snapshot['best_val'];initial_val=snapshot['initial_val'];curves=snapshot['curves'];gradient_rows=snapshot['gradient_rows'];print('RESUME_FULL_OPTIMIZER_RNG',rid,completed_step,flush=True)
 if not smoke and initial_val is None:
  initial_val,_,_,_=evaluate(model,d,cfg);best_val=dict(initial_val);curves.append({'dataset':ds,'method':method,'step':0,'kind':'validation','lambda':weight,'validation_mse':initial_val['mse'],'validation_mae':initial_val['mae'],'selection_mae':initial_val['selection_mae_batch_mean'],'best_step':0})
 start=time.time();last_val=initial_val
 if completed_step and curves:
  recorded=[r for r in curves if r['kind']=='validation'][-1]
  last_val={'mse':recorded['validation_mse'],'mae':recorded['validation_mae'],'selection_mae_batch_mean':recorded['selection_mae'],'n_origins':len(origins(d,'val'))}
 remaining=[] if bad>=5 else range(completed_step+1,maximum+1)
 for step in remaining:
  predictor_mode(model,True);these=order[(step-1)*32:step*32];x,cf,y=batch(d,these);out=forward(model,x,cf,y);terms=losses(model,out,y,crit)
  objective=terms['full'] if method=='A0' else terms['forecast'] if method=='A1' else terms['full']+weight*terms['forecast']
  if step==1 or step%25==0 or smoke and step==maximum:
   gl,ul=flat_gradient(terms['full'],parameters);gy,uy=flat_gradient(terms['forecast'],parameters);cos,nl,ny,status=cosine_rows(gl[None],gy[None]);assert np.isfinite(nl[0]) and np.isfinite(ny[0]) and ny[0]>0
   gradient_rows.append({'dataset':ds,'method':method,'step':step,'predictor_gradient_cosine':float(cos[0]),'latent_gradient_norm':float(nl[0]),'forecast_gradient_norm':float(ny[0]),'status':status[0],'unused_latent_parameters':ul,'unused_forecast_parameters':uy,'predictor_mode':'train; same forward; no accumulated diagnostic gradients'})
  optimizer.zero_grad(set_to_none=True);objective.backward();assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None);assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('predictor.'));optimizer.step()
  curves.append({'dataset':ds,'method':method,'step':step,'kind':'training','lambda':weight,'objective':float(objective.detach()),'latent_full':float(terms['full'].detach()),'latent_l1':float(terms['l1'].detach()),'latent_cosine':float(terms['cosine'].detach()),'prediction_regularizer':float(terms['regularizer_pred'].detach()),'target_regularizer_constant':float(terms['regularizer_target'].detach()),'forecast_mae':float(terms['forecast'].detach()),'forecast_mse':float(terms['forecast_mse'].detach()),'origin_batch_sha256':hashlib.sha256(these.tobytes()).hexdigest()})
  completed_step=step
  if not smoke and (step%50==0 or step==maximum):
   val,_,_,_=evaluate(model,d,cfg);last_val=val;assert state_hash(model,'frozen')==frozen
   if val['selection_mae_batch_mean']<best_val['selection_mae_batch_mean']:
    best_val=dict(val);best_step=step;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
   else:bad+=1
   curves.append({'dataset':ds,'method':method,'step':step,'kind':'validation','lambda':weight,'validation_mse':val['mse'],'validation_mae':val['mae'],'selection_mae':val['selection_mae_batch_mean'],'best_step':best_step})
   temporary=latest.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng(),'source_manifest':code,'order_hash':order_hash,'step':step,'bad':bad,'elapsed':elapsed+time.time()-start,'best_step':best_step,'best_weights':best_weights,'best_val':best_val,'initial_val':initial_val,'curves':curves,'gradient_rows':gradient_rows},temporary);temporary.replace(latest)
   write_json(directory/'status.json',{'status':'RUNNING','run_id':rid,'step':step,'best_step':best_step,'validation_mae':val['selection_mae_batch_mean'],'early_stopping_bad_checks':bad})
   print('TRAINING_VALIDATION',rid,step,'MAE',val['selection_mae_batch_mean'],'BEST_STEP',best_step,'BAD',bad,flush=True)
   if bad>=5:break
 elapsed+=time.time()-start;peak=torch.cuda.max_memory_allocated()/2**20;assert state_hash(model,'frozen')==frozen;assert state_hash(model,'predictor')!=initial_predictor
 write_csv(directory/'curves.csv',curves);write_csv(directory/'gradient_trajectory.csv',gradient_rows)
 if smoke:
  write_json(directory/'metrics.json',{'dataset':ds,'method':method,'steps':completed_step,'status':'SMOKE_PASS_NOT_FORMAL_RESULT','seconds':elapsed,'peak_gpu_mb':peak,'frozen_unchanged':True,'initial_hash':initial,'frozen_hash':frozen,'final_predictor_hash':state_hash(model,'predictor')});print('SMOKE_PASS',rid,flush=True);return
 assert last_val is not None
 last_path=ROOT/'outputs/checkpoints'/f'{rid}_last.pt';torch.save({'Model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':completed_step,'initial_checkpoint_sha256':meta['checkpoint_sha256']},last_path)
 best_path=ROOT/'outputs/checkpoints'/f'{rid}_best.pt';torch.save({'Model':best_weights,'step':best_step,'initial_checkpoint_sha256':meta['checkpoint_sha256']},best_path);model.load_state_dict(best_weights,strict=True);assert state_hash(model,'frozen')==frozen
 measured,pred,true,oo=evaluate(model,d,cfg,save=directory/'selected_validation.npz');assert abs(measured['selection_mae_batch_mean']-best_val['selection_mae_batch_mean'])<1e-7
 new,_cfg,_,_=load_model(ds,predictor_grad=True);new.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=False)['Model'],strict=True);fresh,fresh_pred,fresh_true,fresh_oo=evaluate(new,d,cfg);assert np.array_equal(fresh_true,true) and np.array_equal(fresh_oo,oo);maxdiff=float(np.abs(fresh_pred-pred).max());assert maxdiff<=1e-6
 result={'run_id':rid,'dataset':ds,'method':method,'seed':2021,'lambda':weight,'steps':completed_step,'maximum_steps':1000,'learning_rate':1e-4,'optimizer':'Adam','batch_size':32,'validation_frequency':50,'patience_checks':5,'best_step':best_step,'initial_val':initial_val,'best_val':best_val,'last_val':last_val,'seconds':elapsed,'peak_gpu_mb':peak,'trainable_predictor_parameters':sum(p.numel() for p in parameters),'total_parameters':sum(p.numel() for p in model.parameters()),'frozen_hash_before':frozen,'frozen_hash_after':state_hash(model,'frozen'),'best_checkpoint':str(best_path),'best_checkpoint_sha256':sha(best_path),'last_checkpoint':str(last_path),'last_checkpoint_sha256':sha(last_path),'initial_checkpoint_sha256':meta['checkpoint_sha256'],'initial_model_hash':initial,'training_order_sha256':order_hash,'initial_cuda_rng_sha256':initial_rng_hash,'fresh_selected_validation_max_diff':maxdiff,'test_set_used':False,'status':'COMPLETE','code_sha256':code}
 write_json(directory/'metrics.json',result);write_json(directory/'status.json',{'status':'COMPLETE','steps':completed_step,'best_step':best_step});print('FORMAL_TRAINING_COMPLETE',rid,json.dumps({'best_step':best_step,'best_mae':best_val['selection_mae_batch_mean'],'steps':completed_step}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NP','PJM']);p.add_argument('--method',choices=['A0','A1','A2']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.dataset:run(a.dataset,a.method,a.smoke)
 elif a.smoke:
  for ds in ['NP','PJM']:
   for method in ['A0','A1','A2']:run(ds,method,True)
  write_json(ROOT/'outputs/smoke_complete.json',{'passed':True,'smoke_cases':6,'steps_each':20,'formal_training_runs':0})
 else:raise ValueError('Specify formal dataset and method, or --smoke')
