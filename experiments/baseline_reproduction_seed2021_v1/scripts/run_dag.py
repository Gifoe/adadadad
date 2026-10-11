"""Reevaluate and verify the four original trained DAG checkpoints."""
from common import *
from dag_data import *
import argparse,shutil
from types import SimpleNamespace
def execute(dataset,horizon):
 src=ROOT/'upstream/DAG';sys.path.insert(0,str(src))
 from ts_benchmark.baselines.dag.models.dag_model import DAGModel
 from ts_benchmark.data.utils import read_data
 from ts_benchmark.recording import read_record_file
 out=ROOT/'results/dag'/f'{dataset}_H{horizon}_seed2021';out.mkdir(parents=True,exist_ok=True)
 old=pathlib.Path('D:/conditional_latent_loop_seed2021_v1/runs')/f'{dataset}_{horizon}_B0';ckpt=old/'checkpoint.pt';assert ckpt.is_file()
 result=json.loads((old/'result.json').read_text());command=json.loads((old/'official_command.json').read_text())
 official_line=command['published_line'];assert official_line in (src/'scripts/w_future/DAG.sh').read_text()
 # Verify every archived source file used by the previous official run.
 source_verification=[]
 for f in (src/'ts_benchmark').rglob('*.py'):
  before=pathlib.Path('D:/conditional_latent_loop_seed2021_v1/upstream/DAG')/f.relative_to(src)
  assert before.is_file(),str(f);current_bytes=f.read_bytes();previous_bytes=before.read_bytes();assert current_bytes.replace(b'\r\n',b'\n')==previous_bytes.replace(b'\r\n',b'\n'),str(f)
  source_verification.append({'file':str(f.relative_to(src)),'current_sha256':sha(f),'previous_sha256':sha(before),'canonical_lf_sha256':hashlib.sha256(current_bytes.replace(b'\r\n',b'\n')).hexdigest(),'only_line_endings_differ':current_bytes!=previous_bytes})
 df,y,c,tr,tv,sy,sc=load_data(dataset);originaldf=read_data(str(ROOT/'data'/(dataset+'.csv')));assert originaldf.equals(df)
 ck=torch.load(ckpt,map_location='cpu',weights_only=False);config=SimpleNamespace(**ck['config']);assert config.seq_len==result['configuration']['seq_len'] and getattr(config,'pred_len',getattr(config,'horizon',None))==horizon
 model=DAGModel(config).float().cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
 copy=out/'checkpoint.pt';shutil.copy2(ckpt,copy);assert sha(copy)==sha(ckpt)
 t=config.seq_len;test=Windows(y,c,t,horizon,tv,len(y));assert len(test)==(10461 if horizon==24 else 10125)
 write_json(out/'config.json',{'source_commit':COMMITS['DAG'],'official_command':command,'resolved_config':result['configuration'],'data_sha256':sha(ROOT/'data'/(dataset+'.csv')),'run_type':'verified_reuse','original_checkpoint':str(ckpt),'original_checkpoint_sha256':sha(ckpt),'all_source_files_verified':source_verification,'train_drop_last':True,'test_drop_last':False})
 (out/'source_commit.txt').write_text(COMMITS['DAG']);(out/'command.txt').write_text('python scripts/run_dag.py --dataset '+dataset+' --horizon '+str(horizon)+'\nOriginal training:\n'+official_line)
 write_json(out/'status.json',{'status':'REEVALUATING','run_type':'verified_reuse','pid':os.getpid()})
 start=time.time();torch.cuda.reset_peak_memory_stats();preds=[];targets=[];origins=[]
 with torch.no_grad():
  for yh,ch,cf,target,o in torch.utils.data.DataLoader(test,batch_size=64,shuffle=False,drop_last=False):
   pred,_=model(torch.cat([yh,ch],-1).cuda(),cf.cuda());preds.append(pred[:,-horizon:,:1].cpu().numpy());targets.append(target.numpy());origins.append(o.numpy())
 torch.cuda.synchronize();evaltime=time.time()-start;pred=np.concatenate(preds);true=np.concatenate(targets);origins=np.concatenate(origins);mse,mae=independent_metrics(pred,true)
 previous=np.load(old/'predictions.npz');assert np.array_equal(origins,previous['origins']);maxdelta=float(np.max(np.abs(pred[:,:,0]-previous['pred'])));assert maxdelta<1e-5
 assert abs(mse-result['mse'])<1e-6 and abs(mae-result['mae'])<1e-6
 official_path=pathlib.Path(result['official_record_path']);assert official_path.is_file();official_record=read_record_file(str(official_path));valid=official_record[np.isfinite(official_record['mse_norm'])];assert len(valid)==1;official_row=valid.iloc[0];official_delta=max(abs(mse-float(official_row['mse_norm'])),abs(mae-float(official_row['mae_norm'])));assert official_delta<2e-5
 write_csv(out/'official_metric_record.csv',[{k:official_row[k] for k in official_row.index if k not in ['actual_data','inference_data']}]);write_json(out/'original_official_record_manifest.json',{'path':str(official_path),'sha256':sha(official_path),'mse_norm':float(official_row['mse_norm']),'mae_norm':float(official_row['mae_norm']),'independent_max_abs_diff':official_delta})
 np.savez_compressed(out/'predictions.npz',pred=pred,true=true,origins=origins);np.savez(out/'scaler.npz',mean_y=sy.mean_,scale_y=sy.scale_,mean_c=sc.mean_,scale_c=sc.scale_)
 history=[{'epoch':i+1,'valid_mae':v} for i,v in enumerate(result['validation_mae'])];write_csv(out/'history.csv',history)
 pm,pa=paper_reference('DAG',dataset,'DAG',horizon);diff=abs(mse/pm-1)
 metrics={'model':'DAG','dataset':dataset,'backbone':'DAG','input_len':t,'horizon':horizon,'seed':2021,'source_type':'official_source','run_type':'verified_reuse','paper_mse':pm,'paper_mae':pa,'reproduced_mse':mse,'reproduced_mae':mae,'relative_mse_diff':diff,'best_epoch':min(history,key=lambda r:r['valid_mae'])['epoch'],'epochs':len(history),'train_seconds':result['train_seconds'],'eval_seconds':evaltime,'peak_gpu_mb':result['peak_cuda_bytes']/1024**2,'reevaluation_peak_gpu_mb':torch.cuda.max_memory_allocated()/1024**2,'num_parameters':result['trainable_parameters'],'config_path':str(out/'config.json'),'checkpoint_path':str(copy),'checkpoint_sha256':sha(copy),'predictions_sha256':sha(out/'predictions.npz'),'data_sha256':DATA_SHA[dataset],'source_commit':COMMITS['DAG'],'status':'MATCHED' if diff<=.05 else 'CLOSE' if diff<=.1 else 'NOT_MATCHED','independent_verification':{'mse':mse,'mae':mae,'windows':len(test),'all_origins_match':True,'max_prediction_difference':maxdelta,'previous_mse':result['mse'],'previous_mae':result['mae']},'exit_code':0,'start_time':start,'end_time':time.time()}
 write_json(out/'metrics.json',metrics);write_json(out/'status.json',{'status':metrics['status'],'exit_code':0});shutil.copy2(old/'result.json',out/'original_result.json');shutil.copy2(old/'official_command.json',out/'original_command.json')
 (out/'train.log').write_text('Verified reuse of prior complete official training. See original_result.json, original_command.json and logs/original_dag_training.log.\nFresh reevaluation: '+json.dumps(metrics),encoding='utf-8')
 print('DAG_VERIFIED_RESULT',json.dumps(metrics),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NP','PJM'],required=True);p.add_argument('--horizon',type=int,choices=[24,360],required=True);p.add_argument('--fresh',action='store_true');a=p.parse_args()
 previous=pathlib.Path('D:/conditional_latent_loop_seed2021_v1/runs')/f'{a.dataset}_{a.horizon}_B0/checkpoint.pt'
 pending_fresh=(ROOT/'results/dag'/f'{a.dataset}_H{a.horizon}_seed2021'/'verified_reuse_metrics.json').is_file()
 if a.fresh or pending_fresh or not previous.is_file():
  from run_dag_fresh import execute as fresh
  fresh(a.dataset,a.horizon)
 else:execute(a.dataset,a.horizon)
