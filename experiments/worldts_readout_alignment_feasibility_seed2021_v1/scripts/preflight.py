import pathlib,sys,platform,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *
sys.path.insert(0,str(ROOT/'tests'))
from test_contracts import run_contracts
def run():
 rows=[];inputs=[];begin=time.time();checks=[]
 assert torch.cuda.is_available();env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'cpu_threads':torch.get_num_threads()}
 for ds,h in [('NP',360),('PJM',360),('NP',24),('PJM',24)]:
  model,cfg,directory,meta=load_model(ds,h);d=data(ds);start_hash=state_hash(model);scaler=np.load(directory/'scaler.npz')
  for key,value in [('mean_y',d['sy'].mean_),('scale_y',d['sy'].scale_),('mean_c',d['sc'].mean_),('scale_c',d['sc'].scale_)]:assert np.array_equal(scaler[key],value)
  measured,pred,target,oo=evaluate(model,d,cfg,'test',batch_size=64);original=np.load(directory/'predictions.npz')
  difference=float(np.abs(pred-original['pred']).max())
  print('PREFLIGHT_COMPARISON_DIAGNOSTIC',ds,h,json.dumps({'origin_equal':bool(np.array_equal(oo,original['origins'])),'target_equal':bool(np.array_equal(target,original['true'])),'target_max_difference':float(np.abs(target-original['true']).max()),'prediction_max_difference':difference,'pred_shape':list(pred.shape),'reference_shape':list(original['pred'].shape)}),flush=True)
  assert np.array_equal(oo,original['origins']) and np.array_equal(target,original['true']);assert difference<=1e-6
  assert abs(measured['mse']-meta['reproduced_mse'])<1e-7 and abs(measured['mae']-meta['reproduced_mae'])<1e-7
  val,vp,vy,vo=evaluate(model,d,cfg,'val');assert state_hash(model)==start_hash
  history=pd.read_csv(directory/'history.csv');s1=history[history.stage=='stage1_latent'];s2=history[history.stage=='stage2_decoder_mix'];best1=int(s1.loc[s1.validation_loss.idxmin(),'epoch']);oldval=float(s2.validation_loss.min())
  print('VALIDATION_REFERENCE_DIAGNOSTIC',ds,h,json.dumps({'computed':val,'historical_best':oldval,'origin_first':int(origins(d,'val',cfg.seq_len,cfg.horizon)[0]),'origin_last':int(origins(d,'val',cfg.seq_len,cfg.horizon)[-1])}),flush=True)
  from ts_benchmark.baselines.utils import forecasting_data_provider
  scaled=np.concatenate([d['sy'].transform(d['df'].iloc[:,-1:]),d['sc'].transform(d['df'].iloc[:,:-1])],axis=1)
  scaled_df=pd.DataFrame(scaled,index=d['df'].index,columns=['OT']+list(d['df'].columns[:-1])).iloc[d['tr']-cfg.seq_len:d['tv']]
  _,loader=forecasting_data_provider(scaled_df,cfg,timeenc=1,batch_size=32,shuffle=False,drop_last=False)
  wrapper=object.__new__(WorldTS);wrapper.model=model;wrapper.config=cfg
  official_fixed_order=wrapper._validate_forecast(loader,1,torch.device('cuda'),torch.nn.L1Loss(),2);model.eval()
  assert abs(official_fixed_order-val['selection_mae_batch_mean'])<1e-7,'Current validator must exactly match official path on identical batches'
  per_window=np.abs(vp.astype('float64')-vy).mean((1,2));tail=len(vo)%32
  low=high=val['selection_mae_batch_mean']
  if tail:
   batches=(len(vo)+31)//32;sorted_error=np.sort(per_window)
   low=(per_window.sum()/32+(1/tail-1/32)*sorted_error[:tail].sum())/batches
   high=(per_window.sum()/32+(1/tail-1/32)*sorted_error[-tail:].sum())/batches
  assert low-1e-6<=oldval<=high+1e-6,'Historical shuffled score outside possible batch-remainder range'
  val['official_fixed_order_selection_mae']=float(official_fixed_order)
  val['historical_shuffled_batch_mean_mae']=oldval
  val['possible_tail_partition_score_range']=[float(low),float(high)]
  rows.append({'dataset':ds,'horizon':h,'initialization':'complete original Stage2 checkpoint reuse; no baseline retraining','test_verification':measured,'max_prediction_abs_diff':difference,'validation':val,'validation_order_note':'Original loader shuffled each validation pass; unequal final batch induces partition-dependent unweighted mean. New comparisons fix chronological batches for all methods and keep the official batch-MAE criterion. Historical shuffle order was not archived; its exact score is not asserted reproducible. Current batches match official validator exactly.','original_best_validation_mae':oldval,'original_stage1_best_epoch':best1,'model_hash_before':start_hash,'model_hash_after':state_hash(model),'module_parameters':module_stats(model),'raw_target_unit':'train-standardized observation; original RevIN de-normalized inside decode','raw_scale_mean':float(d['sy'].mean_[0]),'raw_scale_std':float(d['sy'].scale_[0]),'test_raw_unit_mse':measured['mse']*float(d['sy'].scale_[0])**2,'test_raw_unit_mae':measured['mae']*float(d['sy'].scale_[0])})
  checkpoint=pathlib.Path(meta['checkpoint_path']);inputs.extend({'path':str(p),'sha256':sha(p)} for p in [checkpoint,directory/'resolved_config.json',directory/'scaler.npz',directory/'predictions.npz',directory/'history.csv'])
  print('PREFLIGHT_ORIGINAL_REUSE_PASS',ds,h,difference,val['selection_mae_batch_mean'],flush=True)
 for ds in ['NP','PJM']:checks.extend(run_contracts(ds));inputs.append({'path':str(BASE/'data'/(ds+'.csv')),'sha256':DATA_SHA[ds]})
 write_json(ROOT/'outputs/preflight_metrics.json',{'passed':True,'tasks':rows,'environment':env,'input_artifacts':inputs,'tests':checks,'seconds':time.time()-begin})
 write_json(ROOT/'outputs/tests_report.json',{'passed':True,'checks':checks});write_json(ROOT/'STATUS.json',{'phase':0,'status':'COMPLETE','overall_complete':False})
 table=pd.DataFrame([{'dataset':r['dataset'],'horizon':r['horizon'],'test_mse':r['test_verification']['mse'],'test_mae':r['test_verification']['mae'],'validation_mae':r['validation']['selection_mae_batch_mean'],'max_prediction_difference':r['max_prediction_abs_diff'],'stage1_best_epoch':r['original_stage1_best_epoch']} for r in rows]).to_markdown(index=False)
 (ROOT/'outputs/PREFLIGHT_REPORT.md').write_text('# Phase0 preflight\n\nAll four complete original Stage2 checkpoints strictly loaded and freshly evaluated at every official test origin; no baseline retraining. Models, targets, origins and training-only scalers match archived outputs. Test verification belongs only to these original checkpoints.\n\n'+table+'\n\nAll structural, Predictor-only freezing, decoded-gradient, future-label invariance, original decoder/RevIN, checkpoint roundtrip, artificial cosine, float64 metrics and no_grad guard tests passed. Parameter hashes and per-module counts are in preflight_metrics.json.\n\n`h_hist` is the original history encoder state; `h_target` encodes true future labels with the historical patch prefix and historical RevIN statistics. `h_pred` depends on history and permitted covariates only. The causal patch decoder receives the same historical state prefix for both predicted and target states. `_decode_model_states` detaches frozen history, but does not detach supplied future states. The official deployment `_predict_only` intentionally uses no_grad and detaches h_pred; it is used as a reference and must not be reused as the A1 training path.\n\nFrozen Encoder/Decoder parameters use requires_grad=False and eval(). Predictor training enables only its parameter gradients and train() mode. Decoder autograd remains enabled with respect to predicted states. Target reconstruction is a label-based diagnostic, not a forecast lower bound.\n\nSource pin:269426ea6217d7152a244da0e209d922acd46d4f. Exact frozen objective is10*latent L1+15*trajectory cosine+0.05*prediction legacy VC+0.05*target legacy VC. Frozen target regularization is constant for Predictor; prediction regularization remains batch/time coupled and differentiable.\n',encoding='utf-8')
 print('PREFLIGHT_AND_CONTRACTS_COMPLETE',len(checks),flush=True)
if __name__=='__main__':run()
