import pathlib,sys,argparse,json,math
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O=ROOT/'outputs';F=O/'figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
def md_table(rows):
 if not rows:return ''
 keys=list(rows[0]);fmt=lambda x:f'{x:.6g}' if isinstance(x,float) else str(x)
 return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(fmt(r.get(k,'')) for k in keys)+' |' for r in rows])
def report(name,text):(O/name).write_text(text+'\n',encoding='utf-8')
def fig(name):plt.tight_layout();plt.savefig(F/(name+'.png'));plt.savefig(F/(name+'.svg'));plt.close()
def stats(v):
 a=np.asarray(v,dtype='float64');a=a[np.isfinite(a)]
 if not len(a):return {'valid':0,'mean':None,'median':None,'q10':None,'q25':None,'q75':None,'q90':None,'negative_fraction':None,'near_zero_fraction':None}
 return {'valid':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'q10':float(np.quantile(a,.1)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),'q90':float(np.quantile(a,.9)),'negative_fraction':float(np.mean(a<0)),'near_zero_fraction':float(np.mean(abs(a)<.1))}
def paired_interval(candidate,reference,length):
 # Circular moving blocks resample consecutive origins, preserving window dependence locally.
 a=np.asarray(candidate,dtype='float64');b=np.asarray(reference,dtype='float64');n=len(a);assert len(b)==n and np.isfinite(a).all() and np.isfinite(b).all()
 count=math.ceil(n/length);tail=n-(count-1)*length;rng=np.random.default_rng(PROTOCOL['bootstrap_seed']);starts=rng.integers(0,n,size=(PROTOCOL['bootstrap_repetitions'],count));values=[]
 for v in [a,b]:
  c=np.concatenate([[0.],np.cumsum(np.concatenate([v,v]))]);s=c[starts[:,:-1]+length]-c[starts[:,:-1]];s=s.sum(1)+c[starts[:,-1]+tail]-c[starts[:,-1]];values.append(s/n)
 gain=100*(1-values[0]/values[1]);difference=values[0]-values[1]
 return {'gain_percent':float(100*(1-a.mean()/b.mean())),'gain_ci_low':float(np.quantile(gain,.025)),'gain_ci_high':float(np.quantile(gain,.975)),'difference':float(a.mean()-b.mean()),'difference_ci_low':float(np.quantile(difference,.025)),'difference_ci_high':float(np.quantile(difference,.975)),'block_length':length,'repetitions':PROTOCOL['bootstrap_repetitions']}

def audit_report():
 a=pd.read_csv(O/'gradient_alignment.csv');e=pd.read_csv(O/'error_alignment.csv');p=pd.read_csv(O/'predictor_parameter_alignment.csv');summary=json.loads((O/'alignment_summary.json').read_text());merged=e.merge(a[['dataset','origin','full_cosine','full_status']],on=['dataset','origin'],validate='one_to_one');block_rows=[];corr=[];quartiles=[];norms=[];overview=[]
 for ds in ['NP','PJM']:
  z=merged[merged.dataset==ds].sort_values('origin');az=a[a.dataset==ds];pz=p[(p.dataset==ds)&(p.objective=='full')]
  overview.append({'dataset':ds,'forecast_MAE':float(z.forecast_mae.mean()),'forecast_MSE':float(z.forecast_mse.mean()),'latent_L1':float(z.latent_l1.mean()),'latent_cosine_loss':float(z.latent_trajectory_cosine.mean()),'target_reconstruction_MAE':float(z.target_reconstruction_mae.mean()),'decoded_gap_MAE':float(z.decoded_gap_mae.mean()),'state_cosine':float(az.full_cosine.mean()),'state_negative_fraction':float((az.full_cosine<0).mean()),'parameter_cosine':float(pz.parameter_cosine.mean()),'parameter_negative_fraction':float((pz.parameter_cosine<0).mean())})
  z=z.copy();z['error_quartile']=pd.qcut(z.forecast_mae,4,labels=['Q1 low','Q2','Q3','Q4 high'])
  for q,g in z.groupby('error_quartile',observed=True):quartiles.append({'dataset':ds,'error_quartile':str(q),'n_origins':len(g),'forecast_MAE':float(g.forecast_mae.mean()),**stats(g.full_cosine)})
  for name in ['l1_gradient_norm','cosine_gradient_norm','full_gradient_norm','forecast_gradient_norm']:
   v=az[name];norms.append({'dataset':ds,'gradient':name,'units':'L per latent coordinate; mean objective includes actual batch size',**{k:x for k,x in stats(v).items() if k not in ['negative_fraction','near_zero_fraction']}})
  for length in PROTOCOL['time_block_lengths']:
   z['block']=(z.origin-z.origin.min())//length;b=z.groupby('block')[['latent_l1','latent_trajectory_cosine','forecast_mae','forecast_mse','target_reconstruction_mae','decoded_gap_mae','full_cosine']].mean();b['n_origins']=z.groupby('block').size();b['origin_first']=z.groupby('block').origin.min();b['origin_last']=z.groupby('block').origin.max()
   for i,r in b.iterrows():block_rows.append({'dataset':ds,'block_length':length,'block':int(i),**{k:int(v) if k in ['n_origins','origin_first','origin_last'] else float(v) for k,v in r.items()}})
   for x,y in [('latent_l1','forecast_mae'),('latent_trajectory_cosine','forecast_mae'),('decoded_gap_mae','forecast_mae'),('full_cosine','forecast_mae')]:
    pear=b[x].corr(b[y]);spear=b[x].rank(method='average').corr(b[y].rank(method='average'));corr.append({'dataset':ds,'block_length':length,'n_blocks':len(b),'x':x,'y':y,'Pearson':None if not np.isfinite(pear) else float(pear),'Spearman':None if not np.isfinite(spear) else float(spear),'p_value':'NOT_COMPUTED_overlapping_origins_not_IID','last_block_origins':int(b.iloc[-1].n_origins)})
 write_csv(O/'time_block_alignment.csv',block_rows);write_csv(O/'block_correlations.csv',corr);write_csv(O/'error_quartile_alignment.csv',quartiles);write_csv(O/'gradient_norm_summary.csv',norms);write_json(O/'audit_analysis.json',{'overview':overview,'correlations':corr,'quartiles':quartiles,'gradient_norms':norms,'summary':summary['summaries']})
 # Plot every origin with transparency. Blocks/correlations do not assume these dots are independent.
 plt.figure(figsize=(9,3.6))
 for i,ds in enumerate(['NP','PJM']):
  ax=plt.subplot(1,2,i+1);z=merged[merged.dataset==ds];ax.scatter(z.latent_l1,z.forecast_mae,s=2,alpha=.18);ax.set(xlabel='Latent L1',ylabel='Forecast MAE (train-standardized)',title=ds)
 fig('01_latent_vs_forecast')
 plt.figure(figsize=(9,3.6))
 for i,ds in enumerate(['NP','PJM']):
  ax=plt.subplot(1,2,i+1);z=a[a.dataset==ds]
  for key in ['l1','cosine','full']:ax.hist(z[key+'_cosine'].dropna(),bins=55,histtype='step',density=True,label=key)
  ax.axvline(0,color='gray',lw=1);ax.set(xlabel='State gradient cosine',ylabel='Density',title=ds,xlim=(-1,1));ax.legend()
 fig('02_gradient_cosine_distribution')
 plt.figure(figsize=(9,3.6))
 for i,ds in enumerate(['NP','PJM']):
  ax=plt.subplot(1,2,i+1);z=merged[merged.dataset==ds];ax.scatter(z.forecast_mae,z.full_cosine,s=2,alpha=.18);ax.axhline(0,color='gray',lw=1);ax.set(xlabel='Forecast MAE',ylabel='Full latent / forecast state cosine',title=ds)
 fig('03_cosine_vs_forecast_error')
 plt.figure(figsize=(9,3.6))
 for i,ds in enumerate(['NP','PJM']):
  ax=plt.subplot(1,2,i+1);z=a[a.dataset==ds]
  for key in ['l1_gradient_norm','cosine_gradient_norm','full_gradient_norm','forecast_gradient_norm']:
   v=z[key];v=v[np.isfinite(v)&(v>0)];ax.hist(np.log10(v),bins=45,histtype='step',density=True,label=key.replace('_gradient_norm',''))
  ax.set(xlabel='log10 state gradient norm',ylabel='Density',title=ds);ax.legend()
 fig('04_gradient_norms')
 plt.figure(figsize=(8,3.6));ax=plt.subplot(1,2,1);indices=np.arange(2)
 for j,key in enumerate(['state_cosine','parameter_cosine']):ax.bar(indices+(j-.5)*.3,[r[key] for r in overview],width=.3,label=key.replace('_',' '))
 ax.set(xticks=indices,xticklabels=['NP','PJM'],ylabel='Mean cosine');ax.axhline(0,color='gray',lw=1);ax.legend();ax=plt.subplot(1,2,2)
 for j,key in enumerate(['state_negative_fraction','parameter_negative_fraction']):ax.bar(indices+(j-.5)*.3,[r[key] for r in overview],width=.3,label=key.split('_')[0])
 ax.set(xticks=indices,xticklabels=['NP','PJM'],ylabel='Negative gradient cosine fraction');ax.legend();fig('05_np_pjm_comparison')
 plt.figure(figsize=(9,4))
 for i,ds in enumerate(['NP','PJM']):
  z=pd.DataFrame(block_rows);z=z[(z.dataset==ds)&(z.block_length==360)];ax=plt.subplot(1,2,i+1);ax.plot(z.origin_first,z.forecast_mae,'o-',label='forecast MAE');ax.plot(z.origin_first,z.full_cosine,'s-',label='gradient cosine');ax.set(xlabel='Validation origin (hour)',ylabel='Block mean',title=ds);ax.legend()
 fig('06_time_block_alignment')
 raw=np.load(O/'horizon_alignment_raw.npz');hr=[]
 for ds in ['NP','PJM']:
  for segment in ['1-120','121-240','241-360']:
   for key in ['l1','cosine','full']:
    mask=(raw['dataset']==ds)&(raw['segment']==segment)&(raw['objective']==key);hr.append({'dataset':ds,'segment':segment,'objective':key,'invalid':int((raw['status'][mask]!='VALID').sum()),**stats(raw['cosine'][mask])})
 write_csv(O/'horizon_gradient_summary.csv',hr)
 report('ALIGNMENT_AUDIT.md','# Latent/readout alignment audit\n\n'+md_table(overview)+'\n\n## Scope and interpretation\n\nOriginal Stage2 checkpoints, frozen weights, eval mode, all 4,883 chronological validation origins per task. No optimizer updates. State gradients use a fresh leaf h_pred; Decoder autograd remains enabled. Parameter gradients use the identical eight batches and weights; Predictor requires_grad is temporarily enabled without changing eval mode or accumulating .grad. All model tensor hashes remain unchanged.\n\nOfficial full objective is 10 L1 + 15 trajectory cosine + 0.05 legacy VC(pred) + 0.05 legacy VC(target). Target regularizer is constant with respect to the frozen-Encoder Predictor. Prediction variance/covariance term couples batch and time tokens; rows are batch-objective state-gradient components, not independent full sample losses. L1 and cosine are also analyzed separately. Zero/nonfinite gradients are counted and excluded, not assigned cosine zero. Epsilon=1e-12; near-zero means absolute cosine <0.1. Norms depend on the mean reduction and actual batch size, including the final batch of 19; compare matching batches, not unmatched scales. Positive rescaling changes norms but cannot change the cosine sign.\n\nForecast/reconstruction/gap metrics are in train-standardized observations after the official history RevIN inverse. Raw observation forecast errors are separately recorded. Target reconstruction is label-conditioned and is not a forecasting lower bound. Decoder nonlinearity and history-dependent RevIN can cause geometric mismatch; this observational audit alone does not establish causality.\n\n## Gradient distributions\n\n'+md_table(summary['summaries'])+'\n\n## Forecast-error quartiles\n\n'+md_table(quartiles)+'\n\n## Continuous-time-block correlations\n\n'+md_table(corr)+'\n\nPearson and Spearman use block means at 360/720/1680 hours; partial final blocks are explicitly counted. Window overlap still crosses adjacent block boundaries. No IID-window p-values or causal interpretation. Horizon-slice statistics are in horizon_gradient_summary.csv; gradient-norm distributions in gradient_norm_summary.csv. Figures 01–06 contain all requested views. Correlation of full batch losses is not represented as per-sample latent loss correlation.\n\n## What this resolves\n\nAssess both spaces and high-error quartiles above. Negative cosine is directional conflict for the sampled first-order gradient, whereas cosine near zero alone is weak alignment. Parameter-space checks contain only eight fixed validation batches and do not prove universal conflict. Training intervention and paired time-block effects are required before identifying an exploitable performance bottleneck.\n')
 return overview

def final_report(overview):
 results=[];curves=[];grad=[];per=[];intervals=[];blocks=[];runs={};post=pd.read_csv(O/'post_selected_alignment.csv')
 for ds in ['NP','PJM']:
  runs[ds]={}
  for method in ['A0','A1','A2']:
   directory=O/'runs'/f'{ds}_{method}_s2021';m=json.loads((directory/'metrics.json').read_text());assert m['status']=='COMPLETE' and m['test_set_used'] is False and m['frozen_hash_before']==m['frozen_hash_after'];runs[ds][method]=m;z=np.load(directory/'selected_validation.npz');assert len(z['origins'])==4883;err=z['pred'].astype('float64')-z['true'].astype('float64');mae=np.abs(err).mean((1,2));mse=(err*err).mean((1,2));runs[ds][method]['mae_per_origin']=mae;runs[ds][method]['mse_per_origin']=mse;runs[ds][method]['origins']=z['origins']
   ini=m['initial_val'];b=m['best_val'];last=m['last_val'];results.append({'dataset':ds,'method':method,'seed':2021,'steps':m['steps'],'best_step':m['best_step'],'step0_selection_MAE':ini['selection_mae_batch_mean'],'best_selection_MAE':b['selection_mae_batch_mean'],'last_selection_MAE':last['selection_mae_batch_mean'],'step0_MAE':ini['mae'],'step0_MSE':ini['mse'],'best_MAE':b['mae'],'best_MSE':b['mse'],'last_MAE':last['mae'],'last_MSE':last['mse'],'MAE_gain_vs_step0_percent':100*(1-b['mae']/ini['mae']),'MSE_gain_vs_step0_percent':100*(1-b['mse']/ini['mse']),'lambda':m['lambda'],'seconds_training_and_periodic_validation':m['seconds'],'peak_allocated_GPU_MiB':m['peak_gpu_mb'],'predictor_parameters':m['trainable_predictor_parameters'],'frozen_unchanged':True,'fresh_prediction_max_diff':m['fresh_selected_validation_max_diff']})
   curves.extend(pd.read_csv(directory/'curves.csv').to_dict('records'));grad.extend(pd.read_csv(directory/'gradient_trajectory.csv').to_dict('records'))
  values=list(runs[ds].values())
  for key in ['initial_checkpoint_sha256','initial_model_hash','training_order_sha256','initial_cuda_rng_sha256','frozen_hash_before']:assert len({m[key] for m in values})==1,key
  frames=[pd.read_csv(O/'runs'/f'{ds}_{method}_s2021'/'curves.csv') for method in ['A0','A1','A2']];n=min(m['steps'] for m in values)
  for frame in frames:assert frame[frame.kind=='training'].origin_batch_sha256.iloc[:n].to_list()==frames[0][frames[0].kind=='training'].origin_batch_sha256.iloc[:n].to_list()
  original=pd.read_csv(O/'error_alignment.csv');original=original[original.dataset==ds].sort_values('origin');base_mae=original.forecast_mae.to_numpy();base_mse=original.forecast_mse.to_numpy();assert np.array_equal(original.origin.to_numpy(),runs[ds]['A0']['origins'])
  for i,o in enumerate(original.origin):
   per.append({'dataset':ds,'origin':int(o),'step0_MAE':float(base_mae[i]),'step0_MSE':float(base_mse[i]),**{f'{m}_MAE':float(runs[ds][m]['mae_per_origin'][i]) for m in ['A0','A1','A2']},**{f'{m}_MSE':float(runs[ds][m]['mse_per_origin'][i]) for m in ['A0','A1','A2']}})
  for candidate,reference in [('A0','step0'),('A1','step0'),('A2','step0'),('A1','A0'),('A2','A0'),('A2','A1')]:
   for metric in ['mae','mse']:
    a=runs[ds][candidate][metric+'_per_origin'];b=(base_mae if metric=='mae' else base_mse) if reference=='step0' else runs[ds][reference][metric+'_per_origin']
    for length in PROTOCOL['time_block_lengths']:intervals.append({'dataset':ds,'candidate':candidate,'reference':reference,'metric':metric,**paired_interval(a,b,length),'interpretation':'descriptive validation-selected comparison; not an unbiased test confidence interval'})
   for length in PROTOCOL['time_block_lengths']:
    a=runs[ds][candidate]['mae_per_origin'];b=base_mae if reference=='step0' else runs[ds][reference]['mae_per_origin']
    for st in range(0,len(a),length):
     aa=a[st:st+length];bb=b[st:st+length];blocks.append({'dataset':ds,'candidate':candidate,'reference':reference,'block_length':length,'block':st//length,'n_origins':len(aa),'origin_first':int(original.origin.iloc[st]),'origin_last':int(original.origin.iloc[min(len(a)-1,st+length-1)]),'candidate_MAE':float(aa.mean()),'reference_MAE':float(bb.mean()),'gain_percent':float(100*(1-aa.mean()/bb.mean()))})
 write_csv(O/'training_results.csv',results);write_csv(O/'training_curves.csv',curves);write_csv(O/'training_gradient_trajectory.csv',grad);write_csv(O/'selected_validation_errors.csv',per);write_csv(O/'paired_block_bootstrap.csv',intervals);write_csv(O/'training_time_block_changes.csv',blocks)
 compar=[];concentration=[]
 for ds in ['NP','PJM']:
  for candidate in ['A1','A2']:
   relevant=[r for r in intervals if r['dataset']==ds and r['candidate']==candidate and r['reference']=='A0' and r['metric']=='mae'];r=relevant[0];bb=[x for x in blocks if x['dataset']==ds and x['candidate']==candidate and x['reference']=='A0' and x['block_length']==360];compar.append({'dataset':ds,'candidate':candidate,'reference':'A0','MAE_gain_percent':r['gain_percent'],'CI360_low':r['gain_ci_low'],'CI360_high':r['gain_ci_high'],'CI720_low':relevant[1]['gain_ci_low'],'CI720_high':relevant[1]['gain_ci_high'],'CI1680_low':relevant[2]['gain_ci_low'],'improved_blocks_360':sum(x['gain_percent']>0 for x in bb),'total_blocks_360':len(bb)})
   delta=runs[ds]['A0']['mae_per_origin']-runs[ds][candidate]['mae_per_origin'];order=np.argsort(delta)[::-1];count=math.ceil(len(delta)*.05);positive=np.clip(delta,0,None);top=float(positive[order[:count]].sum()/positive.sum()) if positive.sum()>0 else None;concentration.append({'dataset':ds,'candidate':candidate,'fraction_origins_improved':float((delta>0).mean()),'top5_percent_share_of_positive_MAE_reduction':top,'note':'share of positive reductions, not fraction of net effect; no window filtering'})
 write_csv(O/'improvement_concentration.csv',concentration)
 plt.figure(figsize=(10,4))
 for i,ds in enumerate(['NP','PJM']):
  ax=plt.subplot(1,2,i+1)
  for method in ['A0','A1','A2']:
   z=pd.DataFrame(curves);z=z[(z.dataset==ds)&(z.method==method)&(z.kind=='validation')];ax.plot(z.step,z.selection_mae,'o-',label=method)
  ax.set(xlabel='Optimizer step',ylabel='Official batch-mean validation MAE',title=ds);ax.legend()
 fig('07_validation_training_curves')
 plt.figure(figsize=(10,4))
 for i,ds in enumerate(['NP','PJM']):
  ax=plt.subplot(1,2,i+1)
  for method in ['A0','A1','A2']:
   z=pd.DataFrame(grad);z=z[(z.dataset==ds)&(z.method==method)];ax.plot(z.step,z.predictor_gradient_cosine,label=method,alpha=.8)
  ax.axhline(0,color='gray',lw=1);ax.set(xlabel='Optimizer step',ylabel='Train-batch Predictor gradient cosine',title=ds);ax.legend()
 fig('08_training_gradient_cosine')
 plt.figure(figsize=(10,4))
 for i,ds in enumerate(['NP','PJM']):
  ax=plt.subplot(1,2,i+1)
  for method in ['A1','A2']:
   z=pd.DataFrame(blocks);z=z[(z.dataset==ds)&(z.candidate==method)&(z.reference=='A0')&(z.block_length==360)];ax.plot(z.origin_first,z.gain_percent,'o-',label=method)
  ax.axhline(0,color='gray',lw=1);ax.set(xlabel='Validation origin (hour)',ylabel='MAE improvement vs A0 (%)',title=ds);ax.legend()
 fig('09_training_time_block_gains')
 # Deliberately separate feasibility from new-loss novelty. This experiment introduces no new mechanism.
 both_positive=all(max(r['MAE_gain_percent'] for r in compar if r['dataset']==ds)>=100*PROTOCOL['meaningful_validation_mae_gain'] for ds in ['NP','PJM'])
 some_positive=any(r['MAE_gain_percent']>=100*PROTOCOL['meaningful_validation_mae_gain'] for r in compar)
 simple_status='LIMITED_POSITIVE' if both_positive else 'INCONCLUSIVE' if some_positive else 'NO_GO'
 decision={'overall':'NO-GO','ordinary_decoded_supervision':simple_status,'new_readout_aware_loss':'NO-GO','reason':'This controlled comparison contains ordinary decoded supervision only. No specific residual failure has been shown that ordinary task supervision cannot resolve. Gradient mismatch alone does not establish novelty or the need for Jacobian supervision.','tasks':compar,'concentration':concentration,'seeds':1,'independent_unseen_dataset_validation':False,'candidate_test_evaluation':False,'fairness_checks_passed':True,'formal_training_runs':6,'selected_validation_bias':'same validation used for mechanism audit and checkpoint selection; bootstrap intervals are descriptive, not confirmatory generalization evidence'}
 write_json(O/'research_decision.json',decision)
 calibration=json.loads((ROOT/'configs/calibration.json').read_text());postrows=[]
 for (ds,m),g in post.groupby(['dataset','method']):postrows.append({'dataset':ds,'method':m,'parameter_cosine_mean':float(g.parameter_cosine.mean()),'parameter_negative_fraction':float((g.parameter_cosine<0).mean()),'state_cosine_mean_equal_batch_weight':float(g.state_cosine_mean.mean()),'state_negative_fraction_equal_batch_weight':float(g.state_negative_fraction.mean())})
 common='All six fits start from the same task-specific official Stage2 checkpoint. Only Predictor parameters train; every other parameter/buffer is frozen and every other module remains eval. Adam lr=1e-4, batch32, identical seed/order/initial CUDA RNG, max1000 steps, validation every50, patience5 checks. Step0 eligible. Selection is the official unweighted mean of batch MAEs on a fixed chronological validation partition; element-weighted full MAE/MSE are reported separately. Original validator shuffles validation and gives the short tail batch equal weight, so its historical best value is not exactly reproducible without the missing shuffle order. Current fixed partitions are verified against the official validator. This canonicalization is applied to all methods before comparison; no metric or tolerance was relaxed. Phase0 independent test reproduction uses original batch64 and gives max prediction difference0 for all four checkpoints.\n\nA0=official full latent objective; A1=decoded forecast MAE; A2=full latent + lambda*forecast MAE. Lambda is fixed before fits from the median Predictor gradient-norm ratio over16 TRAIN-only batches: '+str(calibration['lambda'])+'. No validation/test tuning. Target-Encoder regularizer is constant; predicted-state regularizer and L1/cosine retain Predictor gradients. Frozen Decoder autograd is enabled. Six separate20-step smoke checks passed and are not formal results.\n\nReported seconds include optimization and periodic validation; command wall times additionally include startup, Step0 validation, saving and fresh reload validation. Peak GPU is process-owned torch.cuda.max_memory_allocated MiB, not total device occupancy. All frozen tensor hashes, shared data-order prefixes, initialization/RNG hashes and fresh-reload prediction checks pass. No candidate test forecasts were generated. Different actual step counts follow the common early-stop rule.\n'
 report('TRAINING_COMPARISON.md','# Predictor-only supervision comparison\n\n'+common+'\n## Complete results\n\n'+md_table(results)+'\n\n## Paired validation comparisons\n\n'+md_table(compar)+'\n\nCircular moving-block bootstrap2000 repetitions, seed9281, block360/720/1680. Compare matching origins; intervals concern already selected validation checkpoints and cannot justify unbiased generalization claims. Every time block and every origin is retained.\n\n'+md_table(concentration)+'\n\n## Selected-checkpoint alignment on the same fixed8 batches\n\n'+md_table(postrows)+'\n\nTraining latent/forecast component losses and sampled parameter-gradient trajectories are saved in training_curves.csv/training_gradient_trajectory.csv. Best/last full-model checkpoints and full optimizer/RNG resume files remain on the server; SHA paths are in each run metrics.json and the final artifact manifest. No baseline or old experiment was modified.\n')
 q1='Q1 — Objective conflict: '+ '; '.join(f"{r['dataset']} state cosine={r['state_cosine']:.4f}, negative={r['state_negative_fraction']:.1%}; Predictor cosine={r['parameter_cosine']:.4f}, negative={r['parameter_negative_fraction']:.1%}" for r in overview)+'. Interpret together with quartiles, separate L1/cosine/regularization, and eight-batch parameter sampling. Near-zero cosine is not automatically conflict. Parameter gradients diagnose optimizer directions, while state gradients alone cannot. Positive loss scaling cannot explain cosine signs; history RevIN and Decoder geometry remain plausible contributors rather than proven causes.'
 q2='Q2 — Performance consequences: '+ '; '.join(f"{r['dataset']} {r['candidate']} vs A0 full validation MAE gain={r['MAE_gain_percent']:.3f}%, 360-hour descriptive95% interval=[{r['CI360_low']:.3f},{r['CI360_high']:.3f}], positive blocks={r['improved_blocks_360']}/{r['total_blocks_360']}" for r in compar)+'. Also compare every method with Step0 above: beating deteriorated A0 alone does not establish improvement over the original model. This single-seed validation intervention supports only local feasibility; conflict and performance association do not prove a unique causal mechanism.'
 q3='Q3 — Ordinary task supervision: A1 is ordinary MAE through a frozen Decoder; A2 adds only a train-calibrated copy of the same MAE. Thus any observed improvement is already obtained with simple supervision. A2 exceeding A1 would demonstrate a useful tradeoff in this budget, not a need for a new readout-aware loss. No Jacobian, adapters, loop or architecture changes were used.'
 q4='Q4 — NP/PJM difference: the initial latent/decoded/forecast errors and gradient distributions differ as shown in the audit table; paired gains, error quartiles and all time blocks are reported separately. Neither one dataset nor their arithmetic mean establishes a universal mechanism. Decoder-conditioned target reconstruction can remain small while forecasting remains difficult, and it is not a forecasting lower bound. A causal explanation involving distribution shift, Encoder or Decoder requires another intervention.'
 q5='Q5 — Novel mechanism space / Idea A: NO-GO for a new complex readout-aware loss. Ordinary supervision status='+simple_status+'. The experiment has not isolated a concrete failure that simple decoded supervision cannot address. The next defensible action, if ordinary-supervision gains warrant follow-up, is a preregistered independent dataset and additional seeds with A0/A1/A2; retain time-block and Step0 controls. Do not invent a Jacobian mechanism from low cosine or residual forecast error.'
 report('FINAL_RESEARCH_REPORT.md','# WorldTS Readout–Latent Alignment Feasibility Audit\n\nFinal decision: **NO-GO for a new complex readout-aware loss**. Ordinary decoded supervision: **'+simple_status+'**.\n\n'+q1+'\n\n'+q2+'\n\n'+q3+'\n\n'+q4+'\n\n'+q5+'\n\n## Initial diagnostics\n\n'+md_table(overview)+'\n\n## All six formal training outcomes\n\n'+md_table(results)+'\n\n## Protocol, validity and limitations\n\n'+common+'\n\nFixed single seed2021, NP/PJM H360 only for candidate training; H24 original-checkpoint preflight only. Maximum1000 steps is this audit budget, not the paper training protocol. Validation used for audit and selection; no unbiased candidate test/generalization claim. Historical validation-shuffle order is unavailable and disclosed; exact current official validation and original test prediction routes are verified. Best Step0 results are retained when fine-tuning hurts. No failures or negative comparisons are hidden. Preflight attempts1–4 remain in logs (test-batch mismatch and historical randomized validation mean); attempt5 passes without weakening exact-path checks. Prior conditional-latent and future-response-loop negative reports are copied as evidence in configs, and do not become new positive evidence here.\n\nSee ALIGNMENT_AUDIT.md, TRAINING_COMPARISON.md, research_decision.json, block_correlations.csv, paired_block_bootstrap.csv, improvement_concentration.csv and figures01–09. REPRODUCE.md provides actual Windows commands and pinned inputs. Final artifact manifest records file hashes and server-only checkpoint paths. Git publication provenance is recorded after export.\n')
 report('NEXT_RESEARCH_DECISION.md','# Research decision\n\n**NO-GO for Idea A as a new complex readout-aware/Jacobian loss.**\n\n'+q2+'\n\n'+q3+'\n\n'+q5+'\n\nA GO would require a specific remaining error pattern that simple supervision demonstrably fails to fix, stability across seeds/independent data, and a tested mechanism distinguishing Encoder, Decoder and temporal shift. Those facts are absent from this feasibility audit. This conclusion does not deny that gradients can disagree or that task supervision can improve a fixed Predictor; it limits what these observations justify.\n')
 print('FINAL_ANALYSIS_COMPLETE',json.dumps(decision,ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');args=p.parse_args();overview=audit_report()
 if not args.audit_only:final_report(overview)
