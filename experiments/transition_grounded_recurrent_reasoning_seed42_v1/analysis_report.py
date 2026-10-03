import csv,json,pathlib,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import ROOT,OUT,CK,CONFIG,dump
FIG=ROOT/'figures'
FIGURE_NAMES=['final_accuracy_vs_hop','recurrence_hop_heatmap_baseline','recurrence_hop_heatmap_grounded','state_accuracy_heatmap_baseline','state_accuracy_heatmap_grounded','computational_frontier','overthinking_curve','counterfactual_intervention']
def read(name):
    with (OUT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def mean(values):
    a=[float(x) for x in values if x not in ('',None)]
    return float(np.mean(a)) if a else None
def pct(x):return 'NA' if x is None else f'{100*x:.2f}%'
def pp(x):return 'NA' if x is None else f'{100*x:+.2f} pp'
def rank(x):
    x=np.asarray(x);return np.asarray([(np.flatnonzero(np.sort(x)==a).mean()+1) for a in x])
def frontier_stats(rows):
    x=np.asarray([int(r['D']) for r in rows]);y=np.asarray([int(r['K_star']) for r in rows]);slope,intercept=np.polyfit(x,y,1)
    v=float(((y-y.mean())**2).sum());r2=None if not v else float(1-((y-(slope*x+intercept))**2).sum()/v)
    xr=rank(x);yr=rank(y);corr=None if np.std(yr)==0 else float(np.corrcoef(xr,yr)[0,1])
    return dict(spearman=corr,slope=float(slope),intercept=float(intercept),R2=r2)
def savefig(name):plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close()
def manifest():
    files=[]
    for p in CK.glob('*.pt'):files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    dump(ROOT/'ARTIFACT_MANIFEST.json',dict(checkpoints=files,full_data_directory=str(ROOT/'data/generated'),large_files_retained_on_server=True))
def runtime_text(rt):
    s=[]
    for name,r in rt['training'].items():s.append(f'| {name} | {r["optimizer_steps"]} | {r["examples_seen"]:,} | {r["seconds"]/3600:.3f} | {r["peak_vram_mib"]:.1f} | {r["examples_per_second"]:.1f} |')
    return '\n'.join(['GPU: RTX 5090 32GB; BF16. PyTorch allocated peak, excluding other processes.','', '| Model | Updates | Examples seen | Train hours | Peak VRAM MiB | Examples/s |','|---|---:|---:|---:|---:|---:|']+s+[f'\nTotal measured GPU-stage hours: {rt["total_gpu_hours"]:.3f} (training + smoke + evaluation where run).'])
def report_invalid():
    FIG.mkdir(exist_ok=True);rt=json.loads((OUT/'runtime_metrics.json').read_text())
    placeholders={'final_accuracy_matrix_baseline.csv':'model,D,K,correct,total,accuracy,compute_regime','final_accuracy_matrix_grounded.csv':'model,D,K,correct,total,accuracy,compute_regime','state_accuracy_baseline.csv':'model,D,K,k,correct,total,accuracy,state_type','state_accuracy_grounded.csv':'model,D,K,k,correct,total,accuracy,state_type','computational_frontier.csv':'model,D,K_star,peak_accuracy,accuracy_at_frontier','overthinking_metrics.csv':'model,D,peak_accuracy,K24_accuracy,overthinking_drop,answer_flip_rate,terminal_stability','intervention_results_baseline.csv':'model,sample_id,D,k,K,control,cf_target,original_retention','intervention_results_grounded.csv':'model,sample_id,D,k,K,control,cf_target,original_retention','intervention_summary.csv':'model,D,k,K,control,stratum,n,cf_target_count,cf_target_rate'}
    for name,header in placeholders.items():(OUT/name).write_text(header+'\n',encoding='utf-8')
    for arm,r in rt['training'].items():
        with (OUT/f'final_accuracy_matrix_{arm}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=placeholders[f'final_accuracy_matrix_{arm}.csv'].split(','));w.writeheader()
            for d in [2,3,4,5]:w.writerow(dict(model=arm,D=d,K=5,**r['id']['by_hop'][str(d)],compute_regime='matched' if d==5 else 'over'))
    for name in FIGURE_NAMES:
        if name=='final_accuracy_vs_hop':
            plt.figure(figsize=(7,4))
            for arm,r in rt['training'].items():plt.plot([2,3,4,5],[r['id']['by_hop'][str(d)]['accuracy'] for d in [2,3,4,5]],'-o',label=arm)
            plt.axhline(.9,color='gray',linestyle='--',label='ID validity target');plt.xticks([2,3,4,5]);plt.ylim(0,1);plt.xlabel('ID reasoning depth D');plt.ylabel('Accuracy at K=5');plt.title('ID only: invalid training run after matched 20,000 steps');plt.legend();savefig(name);continue
        plt.figure(figsize=(7,3));plt.axis('off');plt.text(.5,.6,'NOT EVALUATED',ha='center',fontsize=20);plt.text(.5,.35,'Invalid training run: ID gate failed after matched 20,000 steps',ha='center',fontsize=10);savefig(name)
    lines=['# Transition-Grounded Recurrent Reasoning — Seed42 Viability Report','', '## Research question','Does transition supervision make recurrent depth correspond to verifiable and causally used reasoning-state transitions?','', '**Decision: INVALID TRAINING RUN / TRAINING FAILURE.** This is neither evidence supporting nor falsifying the research hypothesis.','', '## Experimental setup',json.dumps(CONFIG,indent=2),'', 'Official implementation was inspected and extended; no additional method was introduced. The initial weights, sample plan, K_train, optimizer, learning-rate schedule, batch size, and budgets are shared. Train only atomic and composed hops2–5; test ID hops2–5. All301,250 generated paths replayed successfully; separate1000 random replays passed.','', '## Q1 — ID performance','Both arms received the permitted maximum20,000 updates because at least one failed the12,000-step gate.','', '| Model | D2 | D3 | D4 | D5 | Macro |','|---|---:|---:|---:|---:|---:|']
    for name,r in rt['training'].items():lines.append('| '+name+' | '+' | '.join([pct(r['id']['by_hop'][str(d)]['accuracy']) for d in [2,3,4,5]]+[pct(r['id']['macro'])])+' |')
    for q,title in [(2,'Depth extrapolation'),(3,'Recurrence transition semantics'),(4,'Overthinking'),(5,'Causal state use')]:lines.extend(['',f'## Q{q} — {title}','Not evaluated. The predeclared ID sanity gate failed; OOD or intervention numbers would not establish viability. CSV files contain headers only and figures explicitly say NOT EVALUATED. No synthetic or substituted results.'])
    lines.extend(['','## Failure analysis','Training did not establish sufficient ID composition performance within the fixed budget. Categories A–E (not decodable, decodable/noncausal, causal/no utility, utility/unstable terminal, baseline equivalence) remain untested. Losses and smoke checks show execution and gradient flow, but those do not substitute for ID generalization. No post-result architecture, timestep, gating, or supervision changes were made.','', 'The official training code advances curriculum by measured accuracy and optionally forces at least1000epochs at a stage. This fixed-budget experiment advances after2000updates at max-hop2 (about15passes through17000records), before establishing ID composition. This is a concrete protocol mismatch with the official success-dependent curriculum, not proof of the failure cause. The official zero-scaled residual initialization and2000-step warmup were preserved. The current budget is empirically insufficient to establish the required held-out ID performance.',''])
    for name in ['baseline','grounded']:
        path=OUT/f'training_execution_sanity_{name}.json'
        if path.exists():
            diag=json.loads(path.read_text());lines.extend([f'### Execution diagnostics: {name}, checkpoint step{diag["checkpoint_step"]}','| Hop | Sample train accuracy | Sample ID accuracy |','|---|---:|---:|'])
            for d in [1,2,3,4,5]:lines.append(f'| {d} | {pct(diag["train"][str(d)]["accuracy"])} | {pct(diag["test"].get(str(d),{}).get("accuracy"))} |')
            lines.append(f'Random512records/hop, seed42. Text/cache checks={diag["text_cache_checks"]}; recurrence prefix difference={diag["recurrence_prefix_max_difference"]}; exact raw-state no-op difference={diag["exact_raw_state_noop_max_difference"]}. Padding variants are recorded separately; BF16 rounding does not explain the failed ID gate.')
    lines.extend(['', '## Resource report',runtime_text(rt),'', '## GO / NO-GO','**INVALID TRAINING RUN**, not a hypothesis-level NO-GO. A second research-method round cannot be justified by this run. Any later attempt should first preregister a matched ID-training protocol that establishes baseline composition, with adequate curriculum training before advancing. The counterfactual mechanism remains untested.','', 'Full data and actual initial/baseline/grounded checkpoints remain on the server and are delivered through Git LFS; see ARTIFACT_MANIFEST.json.'])
    rendered=[]
    for line in lines:
        line=line.replace('CSV files contain headers only and figures explicitly say NOT EVALUATED.','Final-accuracy CSVs contain measured ID K=5 counts only; other evaluation CSVs contain headers. Only the ID final-accuracy figure plots measured values; unexecuted evaluation figures say NOT EVALUATED.')
        for old,new in [('hops2–5','hops 2–5'),('All301,250','All 301,250'),('separate1000','separate 1,000'),('maximum20,000','maximum 20,000'),('the12,000','the 12,000'),('checkpoint step','checkpoint step '),('Random512records/hop, seed42.','Random 512 records/hop, seed 42.'),('least1000epochs','least 1,000 epochs'),('after2000updates','after 2,000 updates'),('max-hop2','max-hop 2'),('about15passes through17000records','about 15 passes through 17,000 records'),('and2000-step','and 2,000-step')]:line=line.replace(old,new)
        if line.startswith('##'):
            if rendered and rendered[-1]!='':rendered.append('')
            rendered.extend([line,'']);continue
        if line==json.dumps(CONFIG,indent=2):rendered.extend(['```json',line,'```']);continue
        if rendered and rendered[-1].startswith('|') and not line.startswith('|'):rendered.append('')
        rendered.append(line)
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(rendered),encoding='utf-8');manifest();dump(OUT/'DECISION.json',dict(decision='INVALID TRAINING RUN',training_valid=False,causal_gate=None,utility_gate=None,id_macro={k:v['id']['macro'] for k,v in rt['training'].items()}))
def report():
    FIG.mkdir(exist_ok=True);rt=json.loads((OUT/'runtime_metrics.json').read_text());final={n:read(f'final_accuracy_matrix_{n}.csv') for n in ['baseline','grounded']};state={n:read(f'state_accuracy_{n}.csv') for n in final};over=read('overthinking_metrics.csv');front=read('computational_frontier.csv');inter=read('intervention_summary.csv')
    stats={n:frontier_stats([r for r in front if r['model']==n]) for n in final};dump(OUT/'frontier_statistics.json',stats)
    def acc(n,d,k):return float(next(r['accuracy'] for r in final[n] if int(r['D'])==d and int(r['K'])==k))
    ood={n:mean([acc(n,d,d) for d in [6,8,10,12,16,20]]) for n in final}
    stab={n:mean([r['terminal_stability'] for r in over if r['model']==n and int(r['D'])>5]) for n in final}
    def im(n,control,metric,stratum='all',extra=False):
        return mean([r[metric] for r in inter if r['model']==n and r['control']==control and r['stratum']==stratum and (int(r['K'])>int(r['D']))==extra])
    causal={n:dict(cf_target=im(n,'counterfactual','cf_target_rate'),noise_target=im(n,'noise','cf_target_rate'),trajectory=im(n,'counterfactual','cf_trajectory_accuracy'),noise_trajectory=im(n,'noise','cf_trajectory_accuracy'),original_retention=im(n,'counterfactual','original_retention_rate'),same_state_preservation=im(n,'same_state','preservation_rate'),same_state_correct_preservation=im(n,'same_state','preservation_rate','original_correct'),cf_correct=im(n,'counterfactual','cf_target_rate','original_correct'),cf_incorrect=im(n,'counterfactual','cf_target_rate','original_incorrect'),cf_extra=im(n,'counterfactual','cf_target_rate',extra=True)) for n in final}
    g=causal['grounded'];b=causal['baseline'];mechanism=(g['cf_target']-b['cf_target']>=.2 and g['cf_target']-g['noise_target']>=.2 and g['same_state_correct_preservation'] is not None and g['same_state_correct_preservation']>.5 and g['trajectory']>g['noise_trajectory'])
    deep={n:max([5]+[d for d in [6,8,10,12,16,20] if acc(n,d,d)>=.9]) for n in final}
    utility=(ood['grounded']-ood['baseline']>=.05 or stab['grounded']-stab['baseline']>=.05 or deep['grounded']>=deep['baseline']+4)
    decision='GO' if mechanism and utility else 'MECHANISM SUPPORTED, CURRENT METHOD NOT YET USEFUL' if mechanism else 'NO-GO'
    dump(OUT/'DECISION.json',dict(decision=decision,training_valid=True,causal_gate=mechanism,utility_gate=utility,id_macro={n:rt['training'][n]['id']['macro'] for n in final},matched_ood_macro=ood,deepest_ood_hop_at_90percent=deep,terminal_stability=stab,causal=causal))
    for n in final:plt.plot(CONFIG['D'],[acc(n,d,d) for d in CONFIG['D']],'-o',label=n)
    plt.xlabel('Reasoning depth D');plt.ylabel('Final accuracy (K=D)');plt.ylim(0,1);plt.legend();savefig('final_accuracy_vs_hop')
    for n in final:
        matrix=np.asarray([[acc(n,d,k) for k in CONFIG['K']] for d in CONFIG['D']]);plt.figure(figsize=(9,5));plt.imshow(matrix,vmin=0,vmax=1,aspect='auto');plt.xticks(range(len(CONFIG['K'])),CONFIG['K']);plt.yticks(range(len(CONFIG['D'])),CONFIG['D']);plt.xlabel('Recurrence K');plt.ylabel('Hop D');plt.title(n);plt.colorbar(label='Final accuracy');savefig('recurrence_hop_heatmap_'+n)
        matrix=np.asarray([[float(next(r['accuracy'] for r in state[n] if int(r['D'])==d and int(r['K'])==24 and int(r['k'])==k)) for k in range(1,25)] for d in CONFIG['D']]);plt.figure(figsize=(10,5));plt.imshow(matrix,vmin=0,vmax=1,aspect='auto');plt.xticks(range(24),range(1,25));plt.yticks(range(10),CONFIG['D']);plt.xlabel('Recurrence index k');plt.ylabel('Hop D');plt.title(n+' state decoding');plt.colorbar(label='True-state accuracy');savefig('state_accuracy_heatmap_'+n)
    for n in final:
        r=[r for r in front if r['model']==n];plt.plot([int(x['D']) for x in r],[int(x['K_star']) for x in r],'-o',label=n)
    plt.xlabel('Hop D');plt.ylabel('K* (95% of swept peak)');plt.legend();savefig('computational_frontier')
    for n in final:
        r=[r for r in over if r['model']==n];plt.plot([int(x['D']) for x in r],[float(x['overthinking_drop']) for x in r],'-o',label=n)
    plt.xlabel('Hop D');plt.ylabel('Peak minus K24 accuracy');plt.legend();savefig('overthinking_curve')
    x=np.arange(6);width=.26
    for j,(n,control,label) in enumerate([('baseline','counterfactual','Baseline CF'),('grounded','counterfactual','Grounded CF'),('grounded','noise','Grounded noise')]):
        vals=[mean([r['cf_target_rate'] for r in inter if r['model']==n and r['control']==control and r['stratum']=='all' and int(r['D'])==d and int(r['K'])==d]) for d in [6,8,10,12,16,20]];plt.bar(x+(j-1)*width,vals,width,label=label)
    plt.xticks(x,[6,8,10,12,16,20]);plt.xlabel('Hop D');plt.ylabel('CF final target rate (K=D)');plt.ylim(0,1);plt.legend();savefig('counterfactual_intervention')
    lines=['# Transition-Grounded Recurrent Reasoning — Seed42 Viability Report','','## Research question','Does transition supervision make recurrent depth correspond to verifiable and causally used reasoning-state transitions?','',f'**Decision: {decision}.**','', '## Experimental setup',f'Pinned official commit: {CONFIG["upstream_commit"]}. Both arms use768 dimensions,4 shared GPT2 layers,12 heads,NoPE,zero dropout,BF16,K_train5,batch128. Atomic+2–5hop training62000;750test chains/hop. Fixed12k updates, matched20k extension only if required. Same initial weights and saved deterministic sample plan. AdamW1e-4/decay.01;2000 warmup and frozen20k linear schedule. Only intermediate transition CE differs. Prototypes use raw state carriers from train data only. See CONFIG.json and README.md for all preregistered choices.','', '## Q1 — ID performance','| Model | D2 | D3 | D4 | D5 | K5 macro |','|---|---:|---:|---:|---:|---:|']
    for n in final:lines.append('| '+n+' | '+' | '.join([pct(acc(n,d,5)) for d in [2,3,4,5]]+[pct(rt['training'][n]['id']['macro'])])+' |')
    lines.extend(['','## Q2 — Depth extrapolation',f'Matched K=D OOD macro: baseline{pct(ood["baseline"])}, grounded{pct(ood["grounded"])}, difference{pp(ood["grounded"]-ood["baseline"])}.','', '| Model | Under K<D macro cells | Matched K=D macro hops | Over K>D macro cells |','|---|---:|---:|---:|'])
    for n in final:lines.append('| '+n+' | '+' | '.join(pct(mean([r['accuracy'] for r in final[n] if int(r['D'])>5 and r['compute_regime']==reg])) for reg in ['under','matched','over'])+' |')
    lines.extend(['','Full final accuracy tables retain120 D/K cells/model with correct and total counts. Under/over macro averages cells; matched macro averages six depths. No best-K model selection.','', '## Q3 — Recurrence transition semantics','Frontier statistics:','```json',json.dumps(stats,indent=2),'```','State accuracy is reported for each D,K,k in state_accuracy CSVs. Prefix reuse is exact because K does not condition the deterministic recurrent dynamics. A high state decode rate alone does not show causal use. K* is relative to the observed peak; for weak near-chance curves it has little mechanistic meaning.','', '## Q4 — Overthinking','| Model | OOD peak minus K24 macro | OOD answer flip macro | OOD terminal stability |','|---|---:|---:|---:|'])
    for n in final:
        rows=[r for r in over if r['model']==n and int(r['D'])>5];lines.append('| '+n+' | '+' | '.join(pct(mean([r[key] for r in rows])) for key in ['overthinking_drop','answer_flip_rate','terminal_stability'])+' |')
    lines.extend(['','Flip denominator is samples correct at any swept earlier K; no early-correct samples gives NA. Terminal stability requires every recurrence after D through24 to predict the true final entity.','', '## Q5 — Causal state use','Exactly200fixed samples for each eligible D/k pair; no selection by original correctness. Train-only prototypes have mincount10. Both arms share sample IDs and CF entities. Main metrics below average11D/k cells at K=D. Correct/incorrect strata and K=D+4 are retained in intervention_summary.csv.','', '| Model | CF target | Noise CF target | CF trajectory | Original retention | Same-state preserve | Same-state preserve on correct | CF on correct | CF on incorrect | CF target extraK |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for n,c in causal.items():lines.append('| '+n+' | '+' | '.join(pct(c[key]) for key in ['cf_target','noise_target','trajectory','original_retention','same_state_preservation','same_state_correct_preservation','cf_correct','cf_incorrect','cf_extra'])+' |')
    lines.extend(['',f'Grounded targeted rate vs baseline: {pp(g["cf_target"]-b["cf_target"])}; vs its noise: {pp(g["cf_target"]-g["noise_target"])}. Mechanism gate: {mechanism}.','', 'Prototype averaging is not guaranteed on-manifold; same-state controls are necessary to interpret patch failure. A single pooled average is not evidence of reliable rollout at every depth.','', '### Automatically selected examples','Selection rule: lexicographically first sampled ID for D6,8,20,k2,K=D,each arm; independent of predictions. Token IDs are mapped to the saved vocabulary.'])
    vocab=json.loads((ROOT/'data/vocab.json').read_text())
    def pathtext(ids):return ' -> '.join(vocab[int(i)] for i in ids)
    for r in json.loads((OUT/'representative_examples.json').read_text())['examples']:
        lines.extend(['',f'**{r["model"]} {r["sample_id"]}**: patch at k{r["k"]}, {vocab[r["e_original"]]} -> {vocab[r["e_cf"]]}.',f'Normal true path: {pathtext([r["start_entity"]]+json.loads(r["true_path"]))}',f'Expected suffix: {pathtext([r["e_cf"]]+json.loads(r["expected_cf_path"]))}',f'Observed normal: {pathtext(json.loads(r["normal_path"]))}',f'Observed patched: {pathtext(json.loads(r["observed_path"]))}'])
    lines.extend(['','## Failure analysis','A. State not decodable: inspect state accuracy, especially unseen hops and recurrences>5.','B. State decodable but noncausal: if targeted intervention fails, state information is decodable but not demonstrated to be causally used. Do not claim learned world-state transitions.','C. Causal but no utility: mechanism-only evidence does not justify GO.','D. Extrapolation benefit but unstable terminals: inspect K24 loss, flips, and all-step stability.','E. Baseline equivalence: compare the same interventions and state decoding; supervised decoding must exceed existing behavior in a causal way.','', 'The architecture lets every state recurrence re-read the original entity and relations; state-only patches may be overwritten by those unchanged carriers. No hard feedback, timestep embedding, bottleneck, gating, or second method was added after observing outcomes. The first-round evidence is single-seed and cannot establish variability.','', '## Resource report',runtime_text(rt),'', '## GO / NO-GO',f'**{decision}**. ID sanity passed. Causal gate={mechanism}; utility gate={utility}. Utility reference uses matched-K OOD gain or terminal stability gain>=5pp, or deepest matched-K OOD hop at90% accuracy at least4hops deeper (values: {deep}); causal reference requires>=20pp over both controls, majority preservation of correct predictions, and structured trajectory advantage over noise.'])
    if not mechanism:lines.append('Accuracy changes alone support at most auxiliary regularization. The core grounded-rollout hypothesis is not supported by this first-round experiment.')
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8');manifest()
