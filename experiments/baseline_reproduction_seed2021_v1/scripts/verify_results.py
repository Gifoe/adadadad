"""Independently check saved predictions, ground truth, train-only scaling and hashes."""
from common import *
from dag_data import load_data
import argparse,pandas as pd
from sklearn.preprocessing import StandardScaler
def verify(allow_partial=False):
 files=sorted((ROOT/'results').glob('*/*/metrics.json'))
 if not allow_partial:assert len(files)==24,f'Expected 24 completed configurations, found {len(files)}'
 checks=[];rows=[];cache={};provenance=[];stage1=[];stage2=[]
 for path in files:
  out=path.parent;m=json.loads(path.read_text());ds=m['dataset'];h=m['horizon'];model=m['model'];run_id=model+'_'+out.name
  assert m['exit_code']==0 and m['source_commit']==COMMITS[model] and m['data_sha256']==DATA_SHA[ds]
  assert sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(out/'predictions.npz')==m['predictions_sha256']
  assert (out/'source_commit.txt').read_text().strip()==COMMITS[model]
  pm,pa=paper_reference(model,ds,m['backbone'],h);assert (m['paper_mse'],m['paper_mae'])==(pm,pa)
  config=json.loads((out/'config.json').read_text());scaler=np.load(out/'scaler.npz');arrays=np.load(out/'predictions.npz');pred=arrays['pred'];true=arrays['true']
  if ds not in cache:
   if model=='LatentTSF':
    rel='ETT-small/ETTh1.csv' if ds=='ETTh1' else 'electricity/electricity.csv';raw=ROOT/'data'/rel;assert sha(raw)==DATA_SHA[ds];df=pd.read_csv(raw)
    cols=[c for c in df.columns if c not in ['date','OT']]+['OT'];values=df[cols].to_numpy();tr=8640 if ds=='ETTh1' else int(len(df)*.7);ss=StandardScaler().fit(values[:tr]);y=ss.transform(values).astype('float32')
    start,end=(11520,14400) if ds=='ETTh1' else (len(df)-int(len(df)*.2),len(df));cache[ds]={'y':y,'mean':ss.mean_,'scale':ss.scale_,'start':start,'end':end}
   else:
    _,y,c,tr,tv,sy,sc=load_data(ds);cache[ds]={'y':y.numpy(),'mean_y':sy.mean_,'scale_y':sy.scale_,'mean_c':sc.mean_,'scale_c':sc.scale_,'start':tv,'end':len(y)}
  data=cache[ds]
  for key in scaler.files:assert np.allclose(scaler[key],data[key],rtol=1e-12,atol=1e-12),(out.name,key)
  step=config['arguments']['step'] if model=='LatentTSF' else 1;origins=np.arange(data['start'],data['end']-h+1,step,dtype='int64')
  if model!='LatentTSF':assert np.array_equal(origins,arrays['origins'])
  channels=7 if ds=='ETTh1' else 321 if ds=='Electricity' else 1
  assert pred.shape==true.shape==(len(origins),h,channels),(out.name,pred.shape)
  ss=sa=0.;count=0;max_target_error=0.
  for start in range(0,len(pred),32):
   stop=start+32;p=pred[start:stop];t=true[start:stop];assert np.isfinite(p).all() and np.isfinite(t).all()
   expected=data['y'][origins[start:stop,None]+np.arange(h)[None,:]];target_error=float(np.max(np.abs(t-expected)));max_target_error=max(max_target_error,target_error);assert target_error==0.,(out.name,target_error)
   e=p.astype('float64')-t.astype('float64');ss+=np.sum(e*e,dtype='float64');sa+=np.sum(np.abs(e),dtype='float64');count+=e.size
  mse,mae=ss/count,sa/count;reported_delta=max(abs(mse-m['reproduced_mse']),abs(mae-m['reproduced_mae']));assert reported_delta<1e-12
  om,oa=(m['official_metrics'][1],m['official_metrics'][0]) if model=='LatentTSF' else (m['official_mse'],m['official_mae']);official_delta=max(abs(mse-om),abs(mae-oa));assert official_delta<2e-5
  diff=abs(mse/pm-1);status='MATCHED' if diff<=.05 else 'CLOSE' if diff<=.1 else 'NOT_MATCHED';assert m['status']==status
  if model=='LatentTSF':
   ae=json.loads((out/'ae_audit.json').read_text());assert ae['unchanged'] and ae['all_parameters_frozen'] and ae['tensor_sha_before']==ae['tensor_sha_after'] and ae['file_sha256_before']==ae['file_sha256_after']
  if model=='WorldTS':
   freeze=json.loads((out/'freeze_audit.json').read_text());assert freeze['unchanged'] and freeze['only_decoder_trainable'];preflight=json.loads((ROOT/'reports'/f'worldts_preflight_{ds}_{h}.json').read_text());assert preflight['passed']
   if not allow_partial:
    trained=json.loads((out/'collapse_diagnostics.json').read_text());assert trained['checkpoint_sha256']==m['checkpoint_sha256'] and trained['trained_future_label_max_abs_diff']==0 and not trained['test_set_used']
    assert not trained['predicted_states']['severe_collapse_flag'] and not trained['target_states']['severe_collapse_flag']
   hist=pd.read_csv(out/'history.csv');assert np.isfinite(hist.validation_loss).all()
   for stage,destination in [('stage1_latent',stage1),('stage2_decoder_mix',stage2)]:
    for record in hist[hist.stage==stage].to_dict('records'):destination.append({'run_id':run_id,'dataset':ds,'horizon':h,**{k:v for k,v in record.items() if pd.notna(v)}})
  check={'run_id':run_id,'model':model,'dataset':ds,'horizon':h,'passed':True,'shape':list(pred.shape),'test_windows':len(origins),'elements':count,'float64_mse':mse,'float64_mae':mae,'reported_max_abs_diff':reported_delta,'official_max_abs_diff':official_delta,'ground_truth_max_abs_diff':max_target_error,'train_only_scaler_verified':True,'checkpoint_sha_verified':True,'predictions_sha_verified':True,'all_test_origins_verified':True}
  checks.append(check);flat={k:v for k,v in m.items() if not isinstance(v,(dict,list))};flat['run_id']=run_id;flat['metric_verification']='PASSED';flat['test_windows']=len(origins);flat['train_log_path']=str(out/'train.log');rows.append(flat)
  cfg=config.get('arguments',config.get('published_hyperparameters',config.get('resolved_config',{})))
  execution_keys=['num_workers','checkpoints','autoencoder_path','result_csv','result_txt','root_path','data_path','model_id'] if model=='LatentTSF' else ['stage1_checkpoint_path','final_checkpoint_path','stage_save_dir','log_stage1_loss_components','log_stage2_loss_components'] if model=='WorldTS' else []
  if model=='WorldTS':cfg={**json.loads((out/'resolved_config.json').read_text()),**config['executed_hyperparameters']}
  for k,v in cfg.items():provenance.append({'run_id':run_id,'model':model,'parameter':k,'value':json.dumps(v,default=str),'source':'REPRODUCTION_ASSUMPTION' if k in execution_keys else 'OFFICIAL_SOURCE','source_commit':COMMITS[model],'execution_note':'Windows execution/path/logging choice; no architecture change' if k in execution_keys else ''})
  print('VERIFIED',model,out.name,mse,mae,flush=True)
 write_csv(ROOT/'results/all_baselines.csv',rows)
 for model,name in [('LatentTSF','latenttsf_results.csv'),('DAG','dag_results.csv'),('WorldTS','worldts_results.csv')]:write_csv(ROOT/'results'/name,[r for r in rows if r['model']==model])
 write_csv(ROOT/'results/dag_official_results.csv',[r for r in rows if r['model']=='DAG']);write_csv(ROOT/'results/worldts_stage1_metrics.csv',stage1);write_csv(ROOT/'results/worldts_stage2_metrics.csv',stage2)
 assert len({r['run_id'] for r in rows})==len(rows)
 write_csv(ROOT/'results/config_provenance.csv',provenance);write_csv(ROOT/'results/run_manifest.csv',rows)
 verification={'all_passed':True,'complete':len(rows)==24,'verified_configs':len(rows),'expected_configs':24,'independent_dtype':'float64','target_validation':'All saved targets exactly equal independently standardized raw CSV targets at every official test origin','results':checks};write_json(ROOT/'results/metric_verification.json',verification)
 print('METRIC_VERIFICATION_PASSED',len(rows),flush=True);return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');a=p.parse_args();verify(a.allow_partial)
