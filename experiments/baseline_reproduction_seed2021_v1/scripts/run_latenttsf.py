"""Execute pinned my_train.py with nonarchitectural audit/resume hooks."""
from common import *
import argparse,difflib,inspect,traceback

def execute(dataset,backbone,horizon):
 out=ROOT/'results/latenttsf'/f'{dataset}_{backbone}_H{horizon}_seed42';out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').is_file():
  old=json.loads((out/'metrics.json').read_text());assert sha(old['checkpoint_path'])==old['checkpoint_sha256'];print('RESUME_SKIP_VERIFIED',out.name);return
 src=ROOT/'upstream/LatentTSF';sys.path.insert(0,str(src));os.chdir(out)
 channels,dim,ff,bs,lr,step=(7,32,64,256,.0003,1) if dataset=='ETTh1' else (321,512,1024,32,.001,60)
 pattern='AutoEncoder_MLP_MAE_ETTh1_*' if dataset=='ETTh1' else 'AutoEncoder_MLP_MAE_electricity_*'
 matches=list((src/'checkpoints').glob(pattern+'/checkpoint.pth'));assert len(matches)==1;ae=matches[0]
 data=ROOT/'data'/('ETT-small/ETTh1.csv' if dataset=='ETTh1' else 'electricity/electricity.csv');assert sha(data)==DATA_SHA[dataset]
 args={'task_name':'long_term_forecast','is_training':1,'model_id':out.name,'model':backbone,'data':'ETTh1' if dataset=='ETTh1' else 'custom','root_path':str(data.parent),'data_path':data.name,'features':'M','use_norm':1,'seq_len':720,'label_len':0,'pred_len':horizon,'step':step,'enc_in':channels,'dec_in':channels,'c_out':channels,'d_model':dim,'d_ff':ff,'e_layers':2,'d_layers':1,'n_heads':4,'dropout':.1,'train_epochs':100,'batch_size':bs,'accum_steps':1,'learning_rate':lr,'patience':5,'seed':42,'use_lradj':1,'lradj':'cosine','encoder_type':'AE','ae_type':'MLP','ae_loss':'MAE','autoencoder_path':str(ae),'load_pretrained_ae':1,'mse_weight':10.,'cosine_weight':15.,'perceptual_weight':0.,'reconstruction_weight':0.,'des':'Exp','itr':1,'result_csv':str(out/'official_result.csv'),'result_txt':str(out/'official_result.txt'),'checkpoints':str(out/'checkpoints'),'num_workers':0,'load_checkpoint':1}
 argv=[str(src/'my_train.py')]+[x for k,v in args.items() for x in ['--'+k,str(v)]]+['--use_latent','--freeze_encoder','--freeze_decoder']
 sys.argv=argv
 write_json(out/'config.json',{'arguments':args,'flags':['use_latent','freeze_encoder','freeze_decoder'],'use_latent_norm':False,'delete_checkpoint':False,'source_commit':COMMITS['LatentTSF'],'data_sha256':sha(data),'ae_checkpoint_sha256':sha(ae),'source_type':'official_source','paper_reference_table':'Table 6 (not Table 7; task citation corrected)','environment_changes':['num_workers=0 on Windows','private dependencies; shared verified torch2.8+cu128']})
 (out/'source_commit.txt').write_text(COMMITS['LatentTSF']);(out/'command.txt').write_text(' '.join(argv),encoding='utf-8')
 state={'history':[],'start':time.time(),'train_seconds_previous':0,'initial_ae_sha':None,'resume':out/'resume.pt'}
 write_json(out/'status.json',{'run_id':out.name,'status':'RUNNING','start_time':state['start'],'pid':os.getpid()})
 def setup(g):
  enc=g['encoder'];assert enc is not None and not any(p.requires_grad for p in enc.parameters());assert not g['latent_norm']
  state['initial_ae_sha']=tensor_sha(enc)
  r=rng();val=g['valid_data'];values=val.data_x[g['args'].seq_len:];sumabs=sumsq=0.;count=0
  with torch.no_grad():
   for i in range(0,len(values),2048):
    x=torch.from_numpy(values[i:i+2048]).float().unsqueeze(0).to(g['device']);err=(enc.decode(enc.encode(x))-x).double();sumabs+=err.abs().sum().item();sumsq+=err.square().sum().item();count+=err.numel()
  restore_rng(r)
  write_json(out/'ae_audit.json',{'file_sha256_before':sha(ae),'tensor_sha_before':state['initial_ae_sha'],'all_parameters_frozen':True,'validation_unique_observation_points':len(values),'reconstruction_mae':sumabs/count,'reconstruction_mse':sumsq/count,'scale':'train-only StandardScaler normalized','diagnostic_rng_preserved':True})
  np.savez(out/'scaler.npz',mean=val.scaler.mean_,scale=val.scaler.scale_)
  write_json(out/'data_windows.json',{'train':len(g['train_data']),'validation':len(val),'test':len(g['test_data']),'step':step,'drop_last':False,'validation_shuffle':True,'validation_aggregation':'unweighted mean of per-batch MSE; retained official behavior','features':'M','all_channels':channels,'scaler_train_only':True})
  write_json(out/'resolved_config.json',vars(g['args']))
  if state['resume'].is_file():
   ck=torch.load(state['resume'],map_location='cpu',weights_only=False)
   g['model'].load_state_dict(ck['model']);g['model_optim'].load_state_dict(ck['optimizer']);g['early_stopping'].__dict__.update(ck['early_stopping']);state['history']=ck['history'];state['train_seconds_previous']=ck['train_seconds'];restore_rng(ck['rng']);start=ck['next_epoch']
   if g['early_stopping'].early_stop:start=g['args'].train_epochs
   print('EPOCH_RESUME',start,flush=True)
  else:start=0
  state['train_start']=time.time();state['params']=sum(p.numel() for p in g['model'].parameters());state['ae_params']=sum(p.numel() for p in enc.parameters())
  return start
 def record(g):
  row=dict(g['wandb_log']);row['epoch_train_seconds']=time.time()-g['epoch_time'];state['history'].append(row);write_csv(out/'history.csv',state['history'])
  write_json(out/'status.json',{'run_id':out.name,'status':'RUNNING','epoch':g['epoch']+1,'valid_loss':g['valid_loss'],'pid':os.getpid()})
 def snapshot(g):
  saved={'model':g['model'].state_dict(),'optimizer':g['model_optim'].state_dict(),'early_stopping':g['early_stopping'].__dict__,'history':state['history'],'next_epoch':g['epoch']+1,'rng':rng(),'train_seconds':state['train_seconds_previous']+time.time()-state['train_start']}
  tmp=state['resume'].with_suffix('.tmp');torch.save(saved,tmp);tmp.replace(state['resume'])
 def finish_training(g):
  state['train_seconds']=state['train_seconds_previous']+time.time()-state['train_start'];state['checkpoint']=pathlib.Path(g['path'])/'checkpoint.pth'
  finalhash=tensor_sha(g['encoder']);assert finalhash==state['initial_ae_sha'];assert not any(p.requires_grad for p in g['encoder'].parameters())
  audit=json.loads((out/'ae_audit.json').read_text());audit.update({'file_sha256_after':sha(ae),'tensor_sha_after':finalhash,'unchanged':True});write_json(out/'ae_audit.json',audit);state['eval_start']=time.time()
 def metric_capture(pred,true):
  from utils.metrics import metric
  assert pred.shape==true.shape and pred.shape[1:]==(horizon,channels);assert np.isfinite(pred).all() and np.isfinite(true).all()
  np.savez_compressed(out/'predictions.npz',pred=pred,true=true)
  raw=metric(pred,true);state['official_metrics']=list(map(float,raw))
  # Float64 independent computation is streamed to avoid duplicating large arrays.
  ss=sa=0.;n=0
  for i in range(0,len(pred),32):
   err=pred[i:i+32].astype(np.float64)-true[i:i+32].astype(np.float64);ss+=np.sum(err*err,dtype=np.float64);sa+=np.sum(np.abs(err),dtype=np.float64);n+=err.size
  state['independent']={'mse':ss/n,'mae':sa/n,'elements':n,'windows':len(pred),'max_metric_abs_diff':max(abs(ss/n-raw[1]),abs(sa/n-raw[0]))}
  assert state['independent']['max_metric_abs_diff']<1e-5
  return raw
 original=(src/'my_train.py').read_text(encoding='utf-8');patched=original.replace('wandb.Settings(start_method="thread")','wandb.Settings()').replace('            # Build encoder','            setting = _audit_run_id  # Short filesystem name for Windows MAX_PATH.\n            # Build encoder')
 replacements=[('        for epoch in range(args.train_epochs):','        start_epoch = _audit_setup(locals())\n        for epoch in range(start_epoch, args.train_epochs):'),('            early_stopping(valid_loss, model, path)','            early_stopping(valid_loss, model, path)\n            _audit_record(locals())'),('                print("Early stopping")\n                break','                print("Early stopping")\n                _audit_snapshot(locals())\n                break'),('                adjust_learning_rate(model_optim, epoch+1, args)','                adjust_learning_rate(model_optim, epoch+1, args)\n            _audit_snapshot(locals())'),('        # Total training time','        _audit_finish_training(locals())\n        # Total training time'),('    mae, mse, rmse, mape, mspe = metric(preds, trues)','    mae, mse, rmse, mape, mspe = _audit_metric(preds, trues)')]
 for before,after in replacements:assert patched.count(before)==1,before;patched=patched.replace(before,after)
 patch=''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='upstream/LatentTSF/my_train.py',tofile='instrumented/LatentTSF/my_train.py'))
 (ROOT/'patches').mkdir(parents=True,exist_ok=True);(ROOT/'patches/latenttsf_audit_resume.diff').write_text(patch,encoding='utf-8')
 ns={'__name__':'__main__','__file__':str(src/'my_train.py'),'_audit_run_id':out.name,'_audit_setup':setup,'_audit_record':record,'_audit_snapshot':snapshot,'_audit_finish_training':finish_training,'_audit_metric':metric_capture}
 try:
  exec(compile(patched,str(src/'my_train.py'),'exec'),ns)
  pm,pa=paper_reference('LatentTSF',dataset,backbone,horizon);score=state['independent'];diff=abs(score['mse']/pm-1)
  metrics={'model':'LatentTSF','dataset':dataset,'backbone':backbone,'input_len':720,'horizon':horizon,'seed':42,'source_type':'official_source','run_type':'fresh_run','paper_mse':pm,'paper_mae':pa,'reproduced_mse':score['mse'],'reproduced_mae':score['mae'],'relative_mse_diff':diff,'best_epoch':min(state['history'],key=lambda r:r['valid_loss'])['epoch'],'epochs':len(state['history']),'train_seconds':state['train_seconds'],'eval_seconds':time.time()-state['eval_start'],'peak_gpu_mb':torch.cuda.max_memory_allocated()/1024**2,'num_parameters':state['params'],'frozen_ae_parameters':state['ae_params'],'config_path':str(out/'config.json'),'checkpoint_path':str(state['checkpoint']),'checkpoint_sha256':sha(state['checkpoint']),'predictions_sha256':sha(out/'predictions.npz'),'data_sha256':sha(data),'source_commit':COMMITS['LatentTSF'],'official_metrics':state['official_metrics'],'independent_verification':score,'status':'MATCHED' if diff<=.05 else 'CLOSE' if diff<=.1 else 'NOT_MATCHED','exit_code':0,'start_time':state['start'],'end_time':time.time()}
  write_json(out/'metrics.json',metrics);write_json(out/'status.json',{'run_id':out.name,'status':metrics['status'],'exit_code':0});print('LATENTTSF_RESULT',json.dumps(metrics),flush=True)
 except BaseException:
  write_json(out/'status.json',{'run_id':out.name,'status':'FAILED','exit_code':1,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ETTh1','Electricity'],required=True);p.add_argument('--backbone',choices=['DLinear','iTransformer'],required=True);p.add_argument('--horizon',type=int,choices=[96,192,336,720],required=True);a=p.parse_args();execute(a.dataset,a.backbone,a.horizon)
