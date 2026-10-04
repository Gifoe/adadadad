import csv,hashlib,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import ROOT,OUT,CK,CONFIG,dump,csvwrite
FIG=ROOT/'figures'
def read(n):
    p=OUT/n
    return list(csv.DictReader(p.open(encoding='utf-8'))) if p.exists() else []
def mean(a):
    a=[float(v) for v in a if v not in ('',None)];return float(np.mean(a)) if a else None
def pct(v):return 'NOT RUN' if v is None else f'{100*v:.2f}%'
def savefig(n):
    plt.tight_layout();plt.savefig(FIG/(n+'.png'),dpi=180,bbox_inches='tight');plt.close()
def nafig(n,reason):
    plt.figure(figsize=(8,3));plt.axis('off');plt.text(.5,.65,'NOT EVALUATED',ha='center',fontsize=18);plt.text(.5,.35,reason,ha='center',wrap=True);savefig(n)
def scientific_decision():
    f={n:read(f'final_accuracy_matrix_{n}.csv') for n in ['baseline','grounded']};o=read('overthinking_metrics.csv');i=read('intervention_summary.csv')
    def acc(n,d,k):return float(next(r['accuracy'] for r in f[n] if int(r['D'])==d and int(r['K'])==k))
    def im(n,c,m,stratum='all'):
        return mean([r[m] for r in i if r['model']==n and r['control']==c and r['stratum']==stratum and int(r['K'])==int(r['D'])])
    causal={n:dict(cf_target=im(n,'counterfactual','cf_target_rate'),noise_target=im(n,'noise','cf_target_rate'),trajectory=im(n,'counterfactual','cf_trajectory_accuracy'),noise_trajectory=im(n,'noise','cf_trajectory_accuracy'),same_state_correct_preservation=im(n,'same_state','preservation_rate','original_correct')) for n in f}
    g=causal['grounded'];b=causal['baseline']
    mechanism=g['cf_target']-b['cf_target']>=.2 and g['cf_target']-g['noise_target']>=.2 and g['same_state_correct_preservation'] is not None and g['same_state_correct_preservation']>.5 and g['trajectory']>g['noise_trajectory']
    ood={n:mean([acc(n,d,d) for d in [6,8,10,12,16,20]]) for n in f};stab={n:mean([r['terminal_stability'] for r in o if r['model']==n and int(r['D'])>5]) for n in f};deep={n:max([5]+[d for d in [6,8,10,12,16,20] if acc(n,d,d)>=.9]) for n in f}
    utility=ood['grounded']-ood['baseline']>=.05 or stab['grounded']-stab['baseline']>=.05 or deep['grounded']>=deep['baseline']+4
    decision='GO' if mechanism and utility else 'MECHANISM SUPPORTED — UTILITY NOT SUPPORTED' if mechanism else 'ID CLOSED — MECHANISM NOT SUPPORTED'
    front=read('computational_frontier.csv');stats={}
    for n in f:
        rows=[r for r in front if r['model']==n and int(r['D'])>5];x=np.array([int(r['D']) for r in rows]);y=np.array([int(r['K_star']) for r in rows])
        rank=lambda a:np.array([np.flatnonzero(np.sort(a)==v).mean()+1 for v in a])
        stats[n]=dict(spearman=None if np.std(y)==0 else float(np.corrcoef(rank(x),rank(y))[0,1]),K_star={r['D']:int(r['K_star']) for r in rows})
    dump(OUT/'scientific_metrics.json',dict(decision=decision,causal_gate=bool(mechanism),utility_gate=bool(utility),causal=causal,matched_ood_macro=ood,terminal_stability=stab,deepest_ood_hop_at_90percent=deep,frontier=stats))
    return decision
def finalize(results,decision):
    FIG.mkdir(exist_ok=True);core=all(results.get(n,{}).get('id_closed',False) for n in ['baseline','grounded']);reason='Official A1 reproduction failed' if 'official_baseline' in results and not results['official_baseline']['official_reproduced'] else 'Matched ID closure gate failed'
    headers={'baseline_learning_curve.csv':'model,stage,global_updates,stage_updates,train_loss,current_stage_accuracy,previous_stage_accuracy,train_accuracy_sample,lr,recurrence_statistics','grounded_learning_curve.csv':'model,stage,global_updates,stage_updates,train_loss,current_stage_accuracy,previous_stage_accuracy,train_accuracy_sample,lr,recurrence_statistics'}
    for n in ['baseline','grounded']:
        headers.update({f'id_closure_{n}.csv':'model,D,K,correct,total,accuracy',f'final_accuracy_matrix_{n}.csv':'model,D,K,correct,total,accuracy,compute_regime',f'state_accuracy_{n}.csv':'model,D,K,k,correct,total,accuracy,state_type',f'intervention_results_{n}.csv':'model,sample_id,D,k,K,control,cf_target,original_retention'})
    headers.update({'overthinking_metrics.csv':'model,D,peak_accuracy,K24_accuracy,overthinking_drop,answer_flip_rate,terminal_stability','intervention_summary.csv':'model,D,k,K,control,stratum,n,cf_target_count,cf_target_rate'})
    for n,h in headers.items():
        if not (OUT/n).exists():(OUT/n).write_text(h+'\n',encoding='utf-8')
    thresholds=[]
    for n in ['official_baseline','baseline','grounded']:
        stages={s['D']:s for s in results.get(n,{}).get('stages',[])}
        for d in [2,3,4,5]:
            s=stages.get(d);thresholds.append(dict(model=n,D=d,status='completed' if s and s['criterion_completed'] else 'failed' if s else 'not_run',updates_to_95=s['updates_to_95'] if s else None,advancement_updates=s['updates'] if s and s['criterion_completed'] else None,stage_updates=s['updates'] if s else None))
    csvwrite(OUT/'updates_to_threshold.csv',thresholds)
    for d in [2,3,4,5]:
        plotted=False
        for n in ['baseline','grounded']:
            rows=[r for r in read(f'{n}_learning_curve.csv') if int(r['stage'])==d]
            if rows:plt.plot([int(r['stage_updates']) for r in rows],[float(r['current_stage_accuracy']) for r in rows],'-o',label=n);plotted=True
        if plotted:plt.axhline(.95,ls='--',color='gray');plt.ylim(0,1);plt.xlabel('Stage optimizer updates');plt.ylabel(f'D{d} held-out validation accuracy');plt.legend();savefig(f'd{d}_learning_curve')
        else:plt.close();nafig(f'd{d}_learning_curve',reason)
    # A1 curve remains visible even when it blocks formal comparison.
    official=read('official_baseline_reproduction.csv')
    for d in [2,3,4,5]:
        rows=[r for r in official if int(r['stage'])==d]
        if rows:plt.plot([int(r['stage_updates']) for r in rows],[float(r['current_stage_accuracy']) for r in rows],label=f'D{d}')
    if official:plt.axhline(.95,ls='--',color='gray');plt.ylim(0,1);plt.xlabel('Stage optimizer updates');plt.ylabel('A1 current-stage validation accuracy (K=8)');plt.legend();savefig('official_reproduction_learning_curve')
    if 'baseline' in results:
        for j,n in enumerate(['baseline','grounded']):
            vals=[next(r['updates_to_95'] for r in thresholds if r['model']==n and r['D']==d) for d in [2,3,4,5]]
            for d,v in enumerate(vals):
                if v is not None:plt.bar(d+j*.35,v,width=.35,label=n if d==0 else None)
        plt.xticks(np.arange(4)+.175,[2,3,4,5]);plt.xlabel('Hop D');plt.ylabel('First observed updates to 95%');plt.legend();savefig('updates_to_95')
        for n in ['baseline','grounded']:plt.plot([2,3,4,5],[results[n]['final_id'][str(d)]['accuracy'] for d in [2,3,4,5]],'-o',label=n)
        plt.ylim(0,1);plt.xlabel('Hop D');plt.ylabel('Independent ID accuracy (K=5)');plt.legend();savefig('id_accuracy')
    else:
        for n in ['updates_to_95','id_accuracy']:nafig(n,reason)
    for n in ['baseline','grounded']:
        if core:
            f=read(f'final_accuracy_matrix_{n}.csv');s=read(f'state_accuracy_{n}.csv')
            for label,ks in [('recurrence_hop_heatmap',CONFIG['K']),('state_accuracy_heatmap',list(range(1,25)))]:
                if label.startswith('recurrence'):a=[[float(next(r['accuracy'] for r in f if int(r['D'])==d and int(r['K'])==k)) for k in ks] for d in CONFIG['D']]
                else:a=[[float(next(r['accuracy'] for r in s if int(r['D'])==d and int(r['K'])==24 and int(r['k'])==k)) for k in ks] for d in CONFIG['D']]
                plt.figure(figsize=(10,5));plt.imshow(a,vmin=0,vmax=1,aspect='auto');plt.xticks(range(len(ks)),ks);plt.yticks(range(len(CONFIG['D'])),CONFIG['D']);plt.xlabel('Recurrence');plt.ylabel('Hop D');plt.title(n);plt.colorbar(label='Accuracy');savefig(label+'_'+n)
        else:
            for label in ['recurrence_hop_heatmap','state_accuracy_heatmap']:nafig(label+'_'+n,reason)
    if core:
        for n in ['baseline','grounded']:
            rows=[r for r in read('overthinking_metrics.csv') if r['model']==n];plt.plot([int(r['D']) for r in rows],[float(r['overthinking_drop']) for r in rows],'-o',label=n)
        plt.legend();plt.xlabel('Hop D');plt.ylabel('Peak minus K24 accuracy');savefig('overthinking_curve')
        inter=read('intervention_summary.csv')
        for n,c,label in [('baseline','counterfactual','Baseline CF'),('grounded','counterfactual','Grounded CF'),('grounded','noise','Grounded noise')]:
            ds=[6,8,10,12,16,20];vals=[mean([r['cf_target_rate'] for r in inter if r['model']==n and r['control']==c and r['stratum']=='all' and int(r['D'])==d and int(r['K'])==d]) for d in ds];plt.plot(ds,vals,'-o',label=label)
        plt.ylim(0,1);plt.legend();plt.xlabel('Hop D');plt.ylabel('Counterfactual target rate');savefig('counterfactual_intervention')
    else:
        for n in ['overthinking_curve','counterfactual_intervention']:nafig(n,reason)
    runtime=json.loads((OUT/'runtime_metrics.json').read_text());science=json.loads((OUT/'scientific_metrics.json').read_text()) if core else None
    if (OUT/'evaluation_runtime.json').exists():
        runtime['evaluation_seconds']=json.loads((OUT/'evaluation_runtime.json').read_text())['evaluation_seconds']
        runtime['total_gpu_hours']=(sum(r['seconds'] for r in results.values())+runtime['evaluation_seconds'])/3600
        dump(OUT/'runtime_metrics.json',runtime)
    lines=['# Training Protocol Closure — Seed42 V2','','**Decision: '+decision+'**','','## 1. Why v1 Was Invalid','V1 advanced a fixed-budget curriculum before held-out composition emerged. Baseline training accuracy near 98–100% with held-out ID macro 0.73%, and grounded ID macro 11.97%, did not establish compositional generalization. The prior result was INVALID TRAINING RUN / TRAINING FAILURE, not method-level or hypothesis-level rejection. A protocol mismatch is observed; this run tests whether correcting it closes training.','','## 2. Official Protocol Reproduction','A1 uses the unmodified official RecurrentGPT2Block, original input without <state>, vocabulary 217, padded length 50, last padded position readout, Poisson(4) recurrence clamped 2–8 and evaluation K=8. Each stage recreates AdamW and linear scheduler with 2,000-step warmup and horizon ceil(stage examples/128)*100001. Details and explicit deviations are in OFFICIAL_PROTOCOL_AUDIT.md.','','| Stage | Updates | First observed >=95% | Validation accuracy | Scheduler horizon | Final LR |','|---|---:|---:|---:|---:|---:|']
    for s in results['official_baseline']['stages']:lines.append(f"| D{s['D']} | {s['updates']} | {s['updates_to_95']} | {pct(s['validation'][str(s['D'])]['accuracy'])} | {s['scheduler_horizon']} | {s['final_lr']:.8g} |")
    lines+=['',f"Official reproduction success: **{results['official_baseline']['official_reproduced']}**.", 'Independent A1 endpoint ID (K=8): '+', '.join(f'D{d}={pct(v["accuracy"])} ({v["correct"]}/{v["total"]})' for d,v in results['official_baseline']['final_id'].items()),'','## 3. Training Closure','| Model | D2 | D3 | D4 | D5 | K5 macro | ID_CLOSED |','|---|---:|---:|---:|---:|---:|---|']
    for n in ['baseline','grounded']:
        r=results.get(n);lines.append('| '+n+' | '+' | '.join([pct(r['final_id'][str(d)]['accuracy']) if r else 'NOT RUN' for d in [2,3,4,5]]+[pct(r['id_macro']) if r else 'NOT RUN',str(r['id_closed']) if r else 'NOT RUN'])+' |')
    lines+=['','First observed updates-to-95 and advancement updates (three consecutive passes for formal arms) are separate columns in outputs/updates_to_threshold.csv. Empty values mean never observed or not run, as identified by status.','', '## 4. Learning Dynamics', 'Formal comparison was not run because A1 failed; no acceleration claim is supported.' if 'baseline' not in results else 'Compare first threshold observations and criterion completion separately. Single-seed dynamics do not establish variability; stages can stop at different times under identical criteria.','','## 5. ID Gate',f'Both matched arms closed: **{core}**. '+('OOD and intervention were executed.' if core else 'OOD and intervention were blocked. Unexecuted CSVs contain headers only; figures say NOT EVALUATED.'),'','## 6. OOD Extrapolation',json.dumps(science['matched_ood_macro'],indent=2) if core else 'NOT RUN — closure gate not passed.','','## 7. Transition Semantics','See state_accuracy CSVs and frontier statistics; decoding alone does not establish causal use.' if core else 'NOT RUN — closure gate not passed.','','## 8. Overthinking',json.dumps(science['terminal_stability'],indent=2) if core else 'NOT RUN — closure gate not passed.','','## 9. Counterfactual Intervention',json.dumps(science['causal'],indent=2) if core else 'NOT RUN — closure gate not passed. Causal hypothesis remains untested.','', 'V1 intervention definitions and mechanism/utility thresholds are frozen. Mechanism requires grounded CF target advantage >=20pp over both baseline and noise, majority preservation of originally correct predictions under same-state replacement, and CF trajectory advantage over noise. Utility requires >=5pp matched OOD or terminal-stability gain, or >=4-hop deeper >=90% matched frontier. These thresholds are not retuned from results.','','## 10. Resource Report','| Model | Updates | Examples seen | Measured training hours | Peak allocated VRAM MiB |','|---|---:|---:|---:|---:|']
    for n,r in results.items():lines.append(f"| {n} | {r['global_updates']} | {r['examples_seen']} | {r['seconds']/3600:.4f} | {max(s['peak_vram_mib'] for s in r['stages']):.1f} |")
    lines+=['',f"Total measured training/evaluation GPU-stage wall time: {runtime['total_gpu_hours']:.4f} hours. This is elapsed GPU workload time, not utilization-integrated GPU time. Models ran sequentially on one RTX 5090; unrelated server processes may share it. Checkpoint serialization and validation are included in stage elapsed time. Per-stage resource metrics are in training summaries.",'','## 11. Final Decision',f'**{decision}**','No architecture or loss changes were made after observing results. Failure to reproduce under these fixed current-data settings is an environment/protocol closure failure; it does not falsify grounded transition reasoning.']
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    availability={n:('present' if (CK/n).exists() else 'NOT RUN — '+reason) for n in ['initial_weights.pt','official_baseline_final.pt','matched_baseline_final.pt','grounded_final.pt']}
    dump(CK/'availability.json',availability)
    artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in CK.glob('*_final.pt')]+[dict(path='checkpoints/initial_weights.pt',bytes=(CK/'initial_weights.pt').stat().st_size,sha256=hashlib.sha256((CK/'initial_weights.pt').read_bytes()).hexdigest())]
    dump(ROOT/'ARTIFACT_MANIFEST.json',dict(checkpoints=artifacts,availability=availability))
    dump(OUT/'DECISION.json',dict(decision=decision,official_reproduced=results['official_baseline']['official_reproduced'],core_evaluation_run=core,scientific_metrics=science))
    dump(OUT/'completion_audit.json',dict(completed=True,decision=decision,required_outputs_present=all((OUT/n).exists() for n in headers),method_extension_unchanged=hashlib.sha256((ROOT/'model_extension.py').read_bytes()).hexdigest()==hashlib.sha256((ROOT.parent/'transition_grounded_recurrent_reasoning_seed42_v1/model_extension.py').read_bytes()).hexdigest()))
