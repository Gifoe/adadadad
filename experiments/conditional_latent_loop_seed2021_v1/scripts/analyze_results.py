from common import *
import math,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(df,cols=None):
    if cols is not None:df=df[cols]
    head='| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |'
    lines=[]
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join((f'{v:.3e}' if 0<abs(v)<1e-4 else f'{v:.6f}') if isinstance(v,float) else str(v) for v in row)+' |')
    return head+'\n'+'\n'.join(lines)

def bootstrap(a,b,block):
    # Resample chronological blocks jointly across both models; retain final short block weights.
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    chunks=[slice(i,min(i+block,len(a))) for i in range(0,len(a),block)]
    sa=np.array([a[s].sum() for s in chunks]);sb=np.array([b[s].sum() for s in chunks]);n=np.array([len(a[s]) for s in chunks])
    rng=np.random.default_rng(2021);idx=rng.integers(len(chunks),size=(2000,len(chunks)))
    delta=((sb[idx].sum(1)-sa[idx].sum(1))/n[idx].sum(1))
    relative=1-sa[idx].sum(1)/sb[idx].sum(1)
    return {'improvement_mse':float(b.mean()-a.mean()),'improvement_ci_low':float(np.quantile(delta,.025)),'improvement_ci_high':float(np.quantile(delta,.975)),
        'relative_reduction':float(1-a.mean()/b.mean()),'relative_ci_low':float(np.quantile(relative,.025)),'relative_ci_high':float(np.quantile(relative,.975)),'block_length':block,'blocks':len(chunks),'replicates':2000}

def covariate_scores(c,origins,h):
    delta=np.abs(np.diff(c.numpy(),axis=0)).mean(1);delta=np.r_[0,delta];prefix=np.r_[0,np.cumsum(delta,dtype='float64')]
    return (prefix[origins+h]-prefix[origins])/h

def profile_dag(name,h):
    from types import SimpleNamespace
    from ts_benchmark.baselines.dag.models.dag_model import DAGModel
    out=ROOT/'runs'/f'{name}_{h}_B0';r=json.loads((out/'result.json').read_text())
    cp=torch.load(out/'checkpoint.pt',weights_only=False);cfg=SimpleNamespace(**cp['config']);m=DAGModel(cfg).cuda();m.load_state_dict(cp['model']);m.eval()
    yh,ch,cf,_,_=datasets(name,cfg.seq_len,h)[2][0];args=(torch.cat([yh,ch],-1).unsqueeze(0).cuda(),cf.unsqueeze(0).cuda())
    with torch.no_grad():
        for _ in range(15):m(*args)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):m(*args)
        torch.cuda.synchronize();latency=(time.perf_counter()-start)*10
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as p:m(*args)
        flops=sum(v.flops for v in p.key_averages())
    r.update(inference_ms_batch1=latency,profiled_flops_batch1=int(flops),flop_scope='PyTorch supported matmul/conv only',all_parameters=sum(p.numel() for p in m.parameters()))
    json_write(out/'result.json',r);return r

def main():
    import subprocess
    audit_data()
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_baselines.py')],check=True)
    results=[];arrays={};horizon_rows=[];iteration_rows=[];cov_rows=[];compute=[];comparisons=[]
    for out in sorted((ROOT/'runs').iterdir()):
        if not (out/'result.json').exists():continue
        r=json.loads((out/'result.json').read_text());name=r['dataset'];h=r['horizon'];v=r['variant'];k=r.get('K',0) if v in ('B3','B4','B5') else 0
        if v=='B0' and 'profiled_flops_batch1' not in r:r=profile_dag(name,h)
        results.append({'dataset':name,'horizon':h,'variant':v,'K':k,'mse':r['mse'],'mae':r['mae'],'epochs':r['epochs'],'best_val_mse':r.get('best_val_mse',np.nan),
            'latent_mse':r.get('latent_mse',np.nan),'latent_cosine':r.get('latent_cosine',np.nan),'decoded_true_latent_mse':r.get('decoded_true_latent_mse',np.nan)})
        compute.append({'dataset':name,'horizon':h,'variant':v,'K':k,**{key:r.get(key,np.nan) for key in ['trainable_parameters','all_parameters','profiled_flops_batch1','analytic_dense_attention_flops_batch1','inference_ms_batch1','train_seconds','peak_cuda_bytes','optimizer_updates','epochs']}})
        ar=dict(np.load(out/'predictions.npz'));arrays[name,h,v,k]=ar
        if v in ('B2','B3','B4','B5') and 'iteration_latent_window_mse' not in ar:
            from train_models import build,evaluate
            model=build(name,168 if h==24 else 720,h,v,k)
            model.load_state_dict(torch.load(out/'checkpoint.pt',weights_only=True))
            rescore=evaluate(model,datasets(name,168 if h==24 else 720,h)[2],out)
            assert np.isclose(rescore['mse'],r['mse'],rtol=1e-6,atol=1e-7),'Diagnostic-only rescore changed final score'
            ar=dict(np.load(out/'predictions.npz'));arrays[name,h,v,k]=ar
            del model
        df,y,c,tr,tv,sy,sc=load_data(name);origins=ar['origins'];target=np.stack([y[int(o):int(o)+h,0].numpy() for o in origins])
        # Recompute held-out errors from saved forecasts independently of training/evaluation accumulators.
        error=(ar['pred']-target)**2
        assert abs(float(error.mean())-r['mse'])<1e-5 and np.isfinite(ar['pred']).all()
        assert np.array_equal(origins,np.arange(tv,len(y)-h+1))
        hm=ar.get('horizon_mse',error.mean(0)[None,:]);wm=ar.get('iteration_window_mse',error.mean(1)[:,None])
        for it,row in enumerate(hm):
            for s,part in enumerate(np.array_split(np.arange(h),4)):
                horizon_rows.append({'dataset':name,'horizon':h,'variant':v,'K':k,'iteration':it,'segment':s+1,'start':int(part[0]+1),'end':int(part[-1]+1),'mse':float(row[part].mean())})
        for it in range(wm.shape[1]):
            prev=wm[:,it-1] if it else wm[:,0];current=wm[:,it]
            iteration_rows.append({'dataset':name,'horizon':h,'variant':v,'K':k,'iteration':it,'mse':float(current.mean()),'mean_gain_from_previous':float((prev-current).mean()),
                'fraction_improved':float(np.mean(current<prev-1e-8)) if it else np.nan,'fraction_worsened':float(np.mean(current>prev+1e-8)) if it else np.nan,
                'fraction_tied':float(np.mean(abs(current-prev)<=1e-8)) if it else np.nan,
                'latent_mse':float(ar['iteration_latent_window_mse'][:,it].mean()) if 'iteration_latent_window_mse' in ar else np.nan,
                'latent_cosine_error':float(ar['iteration_latent_window_cosine'][:,it].mean()) if 'iteration_latent_window_cosine' in ar else np.nan})
        t=168 if h==24 else 720
        training_origins=np.arange(t,tr-h+1);threshold=float(np.quantile(covariate_scores(c,training_origins,h),.9));score=covariate_scores(c,origins,h);high=score>=threshold
        json_write(ROOT/'configs'/f'high_change_{name}_{h}.json',{'training_percentile':90,'threshold':threshold,'training_windows':len(training_origins),'test_high_change_windows':int(high.sum())})
        cov_rows.append({'dataset':name,'horizon':h,'variant':v,'K':k,'threshold':threshold,'high_change_windows':int(high.sum()),'high_change_mse':float(wm[high,-1].mean()),
            'normal_change_mse':float(wm[~high,-1].mean()),'zero_future_mse':float(ar['zero_window_mse'].mean()) if 'zero_window_mse' in ar else np.nan,
            'shuffled_future_mse':float(ar['shuffle_window_mse'].mean()) if 'shuffle_window_mse' in ar else np.nan})
    assert len(results)==30, f'Incomplete formal experiment: {len(results)}/30'
    frame=pd.DataFrame(results);budgets=pd.DataFrame(compute);primary=frame[(frame.K<=2)]
    for name in ('NP','PJM'):
        for h in (24,360):
            tests=[('H1','B2','B1',0,0),('H2','B5','B2',2,0),('H3','B5','B3',2,2),('H4','B5','B4',2,2)]
            if h==360:tests += [('H2_K4','B5','B2',4,0),('H3_K4','B5','B3',4,4),('H4_K4','B5','B4',4,4)]
            for label,a,b,ka,kb in tests:
                aa=arrays[name,h,a,ka]['iteration_window_mse'][:,-1];bb=arrays[name,h,b,kb]['iteration_window_mse'][:,-1]
                base_block=max(h,168)
                for block in (base_block,2*base_block):comparisons.append({'dataset':name,'horizon':h,'hypothesis':label,'candidate':a,'candidate_K':ka,'reference':b,'reference_K':kb,'primary_block':block==base_block,**bootstrap(aa,bb,block)})
    ci=pd.DataFrame(comparisons)
    ci['assessment']=np.where(ci.improvement_ci_low>0,'SUPPORTED',np.where(ci.improvement_ci_high<0,'FALSIFIED','UNRESOLVED'))
    # Hard construction controls: B3/B5 execute the same K blocks; B4/B5 have the same parameters.
    matching=[]
    for name in ('NP','PJM'):
        for h,k in [(24,2),(360,2),(360,4)]:
            rows=budgets[(budgets.dataset==name)&(budgets.horizon==h)&(budgets.K==k)]
            b3=rows[rows.variant=='B3'].iloc[0];b4=rows[rows.variant=='B4'].iloc[0];b5=rows[rows.variant=='B5'].iloc[0]
            assert b4.trainable_parameters==b5.trainable_parameters
            ratio=float(b5.profiled_flops_batch1/b3.profiled_flops_batch1);assert abs(ratio-1)<.01
            assert b5.analytic_dense_attention_flops_batch1==b3.analytic_dense_attention_flops_batch1
            matching.append({'dataset':name,'horizon':h,'K':k,'B5_to_B3_profiled_flops_ratio':ratio,'B3_B5_equal_analytical_linear_attention_flops':True,'B4_B5_equal_parameters':True})
    for file,data in [('refinement_results.csv',frame),('compute_budget.csv',budgets),('horizon_error.csv',pd.DataFrame(horizon_rows)),('per_iteration_gain.csv',pd.DataFrame(iteration_rows)),('covariate_diagnostics.csv',pd.DataFrame(cov_rows)),('bootstrap_comparisons.csv',ci)]:data.to_csv(ROOT/file,index=False)
    json_write(ROOT/'compute_matching.json',matching)
    cases=ROOT/'failure_cases';cases.mkdir(exist_ok=True);case_summary=[]
    for name in ('NP','PJM'):
        for h in (24,360):
            a=arrays[name,h,'B5',2];b=arrays[name,h,'B2',0];gain=b['iteration_window_mse'][:,-1]-a['iteration_window_mse'][:,-1]
            df,y,c,tr,tv,sy,sc=load_data(name)
            examples=[]
            if np.any(gain>1e-8):examples.append(('improves',int(np.argmax(gain))))
            if np.any(gain<-1e-8):examples.append(('worsens',int(np.argmin(gain))))
            case_summary.append({'dataset':name,'horizon':h,'improved_windows':int((gain>1e-8).sum()),'worsened_windows':int((gain<-1e-8).sum()),'illustration_selection':'largest positive/largest negative B2-B5 MSE differences after final evaluation only'})
            for kind,idx in examples:
                o=int(a['origins'][idx]);gt=y[o:o+h,0].numpy();fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
                hours=np.arange(1,h+1)
                axes[0].plot(hours,gt,label='truth');axes[0].plot(hours,b['pred'][idx],label='B2');axes[0].plot(hours,a['pred'][idx],label='B5 K2');axes[0].legend();axes[0].set_ylabel('training-standardized target')
                axes[0].set_title(f'{name} H={h}: {kind}; origin {df.index[o]}; B2-B5 MSE={gain[idx]:.4f}')
                axes[1].plot(hours,c[o:o+h].numpy());axes[1].legend([str(v).strip() for v in df.columns[:-1]],fontsize=8);axes[1].set_ylabel('standardized covariates');axes[1].set_xlabel('forecast hour')
                fig.tight_layout();fig.savefig(cases/f'{name}_{h}_{kind}.png',dpi=150);plt.close(fig)
                pd.DataFrame({'hour':np.arange(1,h+1),'truth':gt,'B2':b['pred'][idx],'B5':a['pred'][idx]}).to_csv(cases/f'{name}_{h}_{kind}.csv',index=False)
    json_write(cases/'CASE_SELECTION.json',case_summary)
    def reduction(name,h,ref):
        a=primary[(primary.dataset==name)&(primary.horizon==h)&(primary.variant=='B5')].iloc[0].mse
        b=primary[(primary.dataset==name)&(primary.horizon==h)&(primary.variant==ref)].iloc[0].mse
        return float(1-a/b)
    longpass={n:reduction(n,360,'B2')>=.05 and reduction(n,360,'B3')>0 for n in ('NP','PJM')}
    degradation={n:reduction(n,360,'B2') for n in ('NP','PJM')};short={n:reduction(n,24,'B2') for n in ('NP','PJM')}
    go=any(longpass.values()) and min(degradation.values())>=-.05 and min(short.values())>=-.05
    decision='GO' if go else 'NO-GO'
    hypotheses={}
    for hname in ('H1','H2','H3','H4'):
        rows=ci[(ci.hypothesis==hname)&ci.primary_block]
        hypotheses[hname]={'positive_tasks':int((rows.relative_reduction>0).sum()),'negative_tasks':int((rows.relative_reduction<=0).sum()),
            'supported_tasks':int((rows.improvement_ci_low>0).sum()),'falsified_tasks':int((rows.improvement_ci_high<0).sum()),
            'uncertain_tasks':int(((rows.improvement_ci_low<=0)&(rows.improvement_ci_high>=0)).sum())}
    json_write(ROOT/'decision.json',{'decision':decision,'primary_K':2,'long_task_threshold_pass':longpass,'long_B5_vs_B2_relative_reduction':degradation,'short_B5_vs_B2_relative_reduction':short,'hypotheses':hypotheses})
    base=pd.read_csv(ROOT/'baseline_results.csv');verify=json.loads((ROOT/'baseline_verification.json').read_text());ae=pd.read_csv(ROOT/'ae_reconstruction.csv')
    ae_cost=[]
    for name in ('NP','PJM'):
        r=json.loads((ROOT/'ae'/name/'result.json').read_text())
        ae_cost.append({'dataset':name,'parameters':sum(v.numel() for v in torch.load(ROOT/'ae'/name/'checkpoint.pt',weights_only=True,map_location='cpu').values()),
            'train_seconds':r['train_seconds'],'epochs':r['epochs'],'best_epoch':r['best_epoch'],'latent_forecaster_configurations_using_AE':11})
    pd.DataFrame(ae_cost).to_csv(ROOT/'ae_compute.csv',index=False)
    headline=[]
    for name in ('NP','PJM'):
        for h in (24,360):
            rows=primary[(primary.dataset==name)&(primary.horizon==h)].set_index('variant')
            headline.append({'dataset':name,'horizon':h,'B2_MSE':float(rows.loc['B2','mse']),'B3_K2_MSE':float(rows.loc['B3','mse']),
                'B5_K2_MSE':float(rows.loc['B5','mse']),'B5_vs_B2_reduction_percent':100*reduction(name,h,'B2'),
                'B5_vs_B3_reduction_percent':100*reduction(name,h,'B3')})
    pd.DataFrame(headline).to_csv(ROOT/'primary_decision_summary.csv',index=False)
    interpretation=[]
    for name in ('NP','PJM'):
        gain=reduction(name,360,'B2');depth_gain=reduction(name,360,'B3')
        if gain<=0:
            interpretation.append(f'{name} long horizon: B5 regresses by {-100*gain:.2f}% versus B2. Any advantage over B3 cannot establish useful refinement when the single-pass reference is stronger.')
        elif depth_gain<=0:
            interpretation.append(f'{name} long horizon: B5 improves over B2 by {100*gain:.2f}%, but regresses by {-100*depth_gain:.2f}% versus compute-matched B3. This is a depth benefit without demonstrated sharing benefit.')
        else:
            interpretation.append(f'{name} long horizon: B5 improves over B2 by {100*gain:.2f}% and over compute-matched B3 by {100*depth_gain:.2f}%; the block intervals below qualify this point-estimate signal.')
    if any(v<-.05 for v in short.values()):
        interpretation.append('The short-horizon non-regression requirement fails: '+str({n:100*v for n,v in short.items() if v<-.05})+' percent B5-versus-B2 reductions.')
    report=['# Conditional Latent Loop Feasibility Audit — Seed 2021 V1','',f'**Decision: {decision}.**',
      'Thirty forecasting runs (four official DAG, twenty new K2/single-pass models, six long-horizon K4 models), plus two validation-selected frozen AEs. No test-selected depth or hyperparameters.',
      '',table(pd.DataFrame(headline)),
      'Positive reduction means B5 is better; negative reduction means regression. The continuation threshold is fixed K2 and evaluated on point estimates; task-specific uncertainty is reported below and must qualify the strength of any GO signal.',
      '', '## Implementation and configuration','',
      'Exact source commits: DAG `0758990e2c73bb54138ea3e7b11a35cbc5476bcc`; LatentTSF `7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2`.',
      'Official DAG NP/PJM published script arguments are preserved. The original benchmark runner is executed with observational timing/checkpoint hooks and a sequential backend. Architecture files are unchanged.',
      'New models share a 64-wide, four-head, two-layer patch Transformer with patch length24, dataset-specific feed-forward width128/256, zero dropout, 32-dimensional future states and one-time future conditioning. Refinement adds state and condition patch projections, one pre-norm Transformer layer and a 0.1-scaled residual state readout. B3 uses independently parameterized copies; B4/B5 reuse one block K times. B4 still retains the initial future condition; only direct later access is zeroed.',
      'All new models: Adam1e-3, batch64, max50 epochs, patience5; NP type3/PJM type1 schedules as in DAG. Latent loss=10*MSE+15*mean pointwise cosine loss, exact LatentTSF reduction; observation baseline uses decoded MSE. AEs are fully frozen. No intermediate supervision.',
      'AE uses exact official MLP class via AST extraction without unrelated training imports. Train unique normalized target timestamps, batch512, Adam5e-4, MAE, max500/patience20; validation MAE selects AE. This pointwise expansion does not identify true hidden physical state.',
      'Detailed data boundaries, scalers, column identity, availability caveats and leakage assertions are in data_audit.md/data_audit.json. Runtime packages and source hashes are in environment.json/source_manifest.json.',
      '', '## Official baseline reproduction','',table(base,['dataset','horizon','mse','mae','reference_mse','reference_mae','mse_relative_difference','official_runner_mse']),
      '', 'The independent normalized calculation agrees with the untouched official TFB metric output within2e-5. The paper values are references, not overwritten experimental measurements. Windows and target scaling are identical across models.',
      f'Large reference deviations (>20% relative MSE): {verify["significant_difference_flag"]}. Configuration, exact data bytes, boundary rounding, training-only scaling and rolling-origin equality were inspected before new-model runs. Current CUDA/PyTorch/numerical packages differ from the published environment; this remains a replication caveat rather than grounds for test-driven retuning.',
      '', '## AE reconstruction','',table(ae),
      '',table(pd.DataFrame(ae_cost)),
      'AE preparation is a separate one-time cost per dataset. Forecast-model training seconds below exclude this preparation, setup, downloads, final diagnostic evaluation and profiling.',
      '',table(primary[primary.variant=='B2'],['dataset','horizon','mse','latent_mse','latent_cosine','decoded_true_latent_mse']),
      'Decoded ground-truth latents have much smaller held-out MSE than the forecasting models. AE reconstruction is therefore not the dominant error bottleneck in this run; good reconstruction alone does not imply useful forecasting geometry.',
      '', '## All model comparisons','',table(frame,['dataset','horizon','variant','K','mse','mae','epochs','best_val_mse']),
      '', '## Refinement and horizon diagnostics','',table(pd.DataFrame(iteration_rows)[lambda d:d.variant.isin(['B3','B4','B5'])],['dataset','horizon','variant','K','iteration','mse','mean_gain_from_previous','fraction_improved','fraction_worsened']),
      'Horizon quartiles and reconstruction/latent prediction errors are saved in horizon_error.csv and refinement_results.csv. Intermediate outputs are evaluated from the selected final model, not independently selected checkpoints. Each window has one fixed trained loop depth.',
      '', '## Covariate dependency diagnostics','',table(pd.DataFrame(cov_rows),['dataset','horizon','variant','K','high_change_windows','high_change_mse','zero_future_mse','shuffled_future_mse']),
      'High-change thresholds use training-only90th-percentile mean absolute standardized covariate increments. Zeroing and fixed within-window time shuffling diagnose reliance and distribution shift; neither establishes causality. Forecast issue times/vintages are unverified.',
      '', '## Compute comparison','',table(budgets,['dataset','horizon','variant','K','trainable_parameters','profiled_flops_batch1','inference_ms_batch1','train_seconds','peak_cuda_bytes']),
      'B3/B5 profiled FLOP counts match within1%, and analytical linear/dense-attention FLOPs match exactly; B4/B5 parameter counts are identical. PyTorch profiling uses an unfused equivalent path and counts supported matmul/conv operations, omitting some operations. The additional analytical count includes attention products but excludes normalization/activation/bias/residual elementwise operations. Latencies use the normal optimized path, warmed synchronized CUDA batch1 final-only decoding; diagnostic intermediate decodes are excluded from deployment timing. Actual wall time and own-process allocated peak memory are recorded. Other user jobs may share the GPU. Early stopping is common but realized epoch/update counts can differ.',
      '', '## Dependence-aware uncertainty and hypotheses','',table(ci[ci.primary_block],['dataset','horizon','hypothesis','relative_reduction','relative_ci_low','relative_ci_high','blocks','assessment']),
      'H1=B2 versus B1 (latent recipe versus observation recipe); H2=B5 versus B2 (loop versus single pass); H3=B5 versus B3 (sharing versus matched non-shared depth); H4=B5 versus B4 (repeated direct future-condition access). Relative reductions and confidence limits in this table are fractions, not percentages.',
      'Paired chronological block bootstrap uses2000 replicates and block lengths max(H,168) and twice that length; short-horizon hourly tasks therefore use at least one week to account for additional calendar dependence. Full sensitivity is in bootstrap_comparisons.csv. These intervals reflect dependent evaluation windows under one trained seed. They do not measure training-seed variation. A positive interval supports a task-specific hypothesis; an entirely negative interval falsifies it on that task; an interval crossing zero is unresolved.',
      'The intervals are nominal 95% exploratory comparisons without a multiple-testing correction. They do not provide simultaneous coverage across all tasks and hypotheses.',
      '```json',json.dumps(hypotheses,indent=2),'```',
      '', '## Failures, changes and interpretation','',
      'Setup failures: Google download initially inherited an unusable proxy; corrected with explicit proxy and bounded range reads. Central-directory offsets were corrected before any data CRC verification. Windows affinity integer signatures were corrected and applied process-locally. The original NumPy2 environment is incompatible with upstream np.Inf; isolated NumPy1.26 and missing official imports were installed without modifying shared environments. These setup failures produced no reported scores.',
      'B0 preserves official MAE checkpoint selection and multivariate training drop_last=True, while B1-B5 share decoded-MSE selection and retain all training windows. B0 is an external reference; the mechanism evidence is the matched B1-B5 comparisons. AE batching and K4 long-only scope are declared changes authorized by the task.',
      'Per-iteration latent diagnostics were added after some completed runs. Their selected checkpoints were rescored without retraining, and final decoded MSE was asserted unchanged. Training losses and checkpoint-selection rules were not modified.',
      'Scalar AE expansion, fixed loss geometry, architectural choices, one training seed and unknown forecast vintages constrain generalization. A negative result rejects this tested recipe, not every possible latent-loop architecture. A positive result is a feasibility signal requiring multi-seed confirmation and operational covariate provenance.',
      'H1 compares full observation-space versus frozen-AE latent-loss recipes; it does not isolate the AE from the changed supervision geometry. Sharing also accumulates gradients across repeated applications under the same optimizer, so H3 does not identify a purely representational cause. Any follow-up should separate AE/loss effects and examine learning-rate sensitivity without using this test split for retuning.',
      '', '## GO / NO-GO','',f'Primary fixed K2 long-horizon threshold checks: {longpass}. B5-versus-B2 long relative reductions: {degradation}; short reductions: {short}.',
      *interpretation,
      f'Final direction decision: **{decision}**. No positive conclusion is forced. K4 is sensitivity evidence and never rescues the primary decision through test-depth selection.',
      'Concrete follow-up, if this direction is revised: first use a matched loss factorial (observation MSE, frozen-AE decoded MSE, frozen-AE latent MSE/cosine) to separate representation from supervision; inspect whether improving latent geometry actually improves decoded errors; and compare shared-block learning-rate scaling against non-shared blocks using validation only. The weak NP conditional predictors also need a stronger single-pass reference before attributing failures solely to recurrence. These are future controls, not changes applied after observing this test set.',
      '', 'References: [DAG official source](https://github.com/decisionintelligence/DAG), [DAG paper](https://arxiv.org/abs/2509.14933), [LatentTSF official source](https://github.com/Muyiiiii/LatentTSF), [WorldTS method reference](https://arxiv.org/abs/2609.31162). This is WorldTS-inspired, not a reproduction.']
    (ROOT/'FINAL_REPORT.md').write_text('\n\n'.join(report),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/record_provenance.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_deliverables.py')],check=True)
    print('FINAL_DECISION',decision,longpass,degradation,short,flush=True)

if __name__=='__main__':main()
