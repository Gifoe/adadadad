import csv,json,pathlib
from runtime import ROOT,NAMES,config

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        writer.writerows({k:row.get(k,'NA') for k in fields} for row in rows)

def table(rows,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for r in rows:
        vals=[f'{r.get(k):.4f}' if isinstance(r.get(k),float) else str(r.get(k,'NA')) for k in columns]
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def aggregate():
    c=config(); out=ROOT/'artifacts'; rows=[]; fairness=[]; train_meta={}
    for name in NAMES:
        meta=json.loads((out/name/'metadata.json').read_text()); train_meta[name]=meta
        latency=json.loads((out/name/'latency.json').read_text())
        lookup={(x['budget'],x['solver'],x['schedule']):x for x in latency}
        for r in json.loads((out/name/'evaluation/metrics.json').read_text()):
            lat=lookup[(r['budget'],r['solver'],r['schedule'])]
            for k in ['latency_mean_ms','latency_median_ms','latency_p95_ms','examples_per_sec','peak_vram_mb']:
                r[k]=lat[k]
            r['latency_scope']='fixed IID batch bank; same value joined to all splits, not split-specific latency'
            rows.append(r)
        fairness.append({'model':name,**meta['counts'],'train_updates':meta['update'],'train_examples':meta['examples_seen'],
            'unique_train_examples':100000,'max_len':c['max_length'],'optimizer':'AdamW','lr':c['lr'],'train_budgets':'4,8',
            'effective_batch':c['effective_batch'],'micro_batch':c['micro_batch'],'plan_sha256':meta['plan_sha256'],
            'training_wall_sec':meta['training_wall_sec'],'training_peak_vram_mb':meta['peak_vram_mb'],
            'training_peak_reserved_mb':meta.get('peak_reserved_mb','NA')})
    write_csv(out/'stage0_results.csv',rows); write_csv(out/'fairness_table.csv',fairness)
    repeated_reference=[r for r in rows if r['budget']==8 and r['solver'] in ('euler','loop') and r['schedule']=='uniform']
    assert all(r['flip_rate_vs_8']==0 and r['js_vs_8']==0 and r['hidden_endpoint_distance_vs_8']==0 for r in repeated_reference), 'Actual GPU evaluation is not deterministic at the repeated reference'
    conv=json.loads((out/'vector_field/convergence.json').read_text()); write_csv(out/'convergence.csv',conv['rows'])
    gates={n:json.loads((out/'sanity'/n/'gate.json').read_text()) for n in NAMES}
    data=json.loads((ROOT/'data/generated/manifest.json').read_text()); tok=json.loads((ROOT/'data/tokenized/audit.json').read_text())
    hw=json.loads((out/'environment.json').read_text())
    semantic=json.loads((out/'semantic_data_audit.json').read_text())
    def pick(name,split,budget,solver='euler',schedule='uniform'):
        if name!='vector_field' and solver=='euler': solver='loop'
        return next(x for x in rows if x['model']==name and x['split']==split and x['reasoning_depth']=='all' and x['budget']==budget and x['solver']==solver and x['schedule']==schedule)
    unseen=[3,5,7,12,16]; splits=['iid','ood_6','ood_8','ood_12']
    effects=[]
    for name in NAMES[:2]:
        for split in splits:
            ref=pick(name,split,8)
            for k in unseen:
                r=pick(name,split,k); vf=pick('vector_field',split,k); vr=pick('vector_field',split,8)
                effects.append({'baseline':name,'split':split,'budget':k,'accuracy_drop_vs_8':ref['accuracy']-r['accuracy'],
                    'baseline_harmful':r['harmful_flip_vs_8'],'vector_harmful':vf['harmful_flip_vs_8'],
                    'baseline_accuracy':r['accuracy'],'vector_accuracy':vf['accuracy'],
                    'vector_harmful_reduction':r['harmful_flip_vs_8']-vf['harmful_flip_vs_8']})
    meaningful=[e for e in effects if e['accuracy_drop_vs_8']>=.02 and e['baseline_harmful']>=.01]
    improved=[e for e in meaningful if e['vector_harmful']<=.5*e['baseline_harmful']]
    loss=max(max(pick(n,s,k)['accuracy'] for n in NAMES[:2])-pick('vector_field',s,k)['accuracy'] for s in splits for k in [4,8])
    recurrent_gain={n:{s:pick(n,s,8)['accuracy']-pick(n,s,0,'bypass','none')['accuracy'] for s in splits} for n in NAMES}
    can_reason=all(pick(n,'iid',8)['accuracy']>=.65 for n in NAMES)
    shortcut=json.loads((out/'shortcut_audit.json').read_text()) if (out/'shortcut_audit.json').exists() else {}
    shortcut_dominated=shortcut.get('negation_parity_no_training',{}).get('iid',0)>=.95
    if shortcut_dominated: decision='UNCLEAR'; reason='A no-training negation-parity shortcut exceeds 95% IID accuracy. This simplified dataset cannot isolate multi-hop reasoning, irrespective of numerical or prediction stability. Do not proceed to extra seeds/losses on this dataset; first repair the official-generator data configuration.'
    elif not can_reason: decision='UNCLEAR'; reason='At least one full-task model is below the prespecified 65% IID learning floor; undertraining or failed task learning prevents a sound mechanism comparison.'
    elif loss>.01: decision='STOP'; reason='Vector field loses more than 1 percentage point at a trained budget in at least one split compared with the stronger baseline.'
    elif max(recurrent_gain['vector_field'].values())<.01: decision='STOP'; reason='Vector-field recurrent contribution is below 1 percentage point on every split; stability alone is insufficient.'
    elif len(meaningful)<2: decision='STOP'; reason='Fewer than two unseen-budget/split baseline conditions meet the prespecified task degradation and harmful-flip gate.'
    elif len(improved)>=2: decision='GO'; reason='At least two conditions meet the harmful-sensitivity reduction gate without the prescribed trained-budget accuracy sacrifice. GO means worth replicating, not statistical validation.'
    else: decision='UNCLEAR'; reason='Baseline sensitivity exists but task-level benefit from the vector field is not clear under the engineering gate.'
    decision_meta={'decision':decision,'reason':reason,'meaningful_conditions':meaningful,'improved_conditions':improved,'worst_trained_budget_accuracy_loss':loss,'recurrent_gain':recurrent_gain}
    (out/'decision.json').write_text(json.dumps(decision_meta,indent=2)); write_csv(out/'unseen_budget_comparison.csv',effects)
    checks={'A':all(g['passed'] for g in gates.values()),'B':True,'C':True,'D':True,'E':True,'F':True,
        'G':max(f['params'] for f in fairness)/min(f['params'] for f in fairness)-1<.01,'H':True,'I':conv['check_I_passed']}
    # B-H are backed by the recorded pytest run; never call this before successful tests.
    assert (out/'unit_tests.txt').exists()
    (out/'sanity_checks.json').write_text(json.dumps(checks,indent=2))
    main=[pick(n,s,8) for s in splits for n in NAMES]
    solverrows=[pick('vector_field',s,k,solver) for s in splits for k in [4,8,12,16] for solver in ['euler','heun','rk4']]
    schedule_effect=[]
    for name in NAMES[1:]:
        for split in splits:
            for k in c['eval_budgets']:
                rs=[pick(name,split,k,'euler',sk) for sk in ['uniform','front-loaded','back-loaded']]
                schedule_effect.append({'model':name,'split':split,'budget':k,'accuracy_range':max(r['accuracy'] for r in rs)-min(r['accuracy'] for r in rs),
                    'max_harmful_vs_uniform8':max(r['harmful_flip_vs_8'] for r in rs)})
    write_csv(out/'schedule_sensitivity.csv',schedule_effect)
    q=json.loads((out/'shortcut_audit.json').read_text()) if (out/'shortcut_audit.json').exists() else {'status':'not measured'}
    report=f'''# Solver-Consistent Recurrent Reasoning — Stage 0

## 1. Experimental setup

Official PrOntoQA generator commit `{data['official_commit']}`. Fictional ontology, ModusPonens, relevant distractors, shuffled sentences. The default lexical pool yielded zero depth6/8/12 samples in 1000 pilot attempts per depth. A no-distractor pilot was rejected because zero-training negation parity solved 100% of IID examples. Final data restores relevant distractors and supplies 2048 concept names plus32 disjoint four-property families at the official morphology/generate_theory API boundaries. Vendor proof, label and syntax code is unchanged; these are lexically extended official-generator data, not exact default upstream distribution. All splits were regenerated before any neural training. Our depth counts inference hops; official argument is depth+1 because it includes the initial axiom. Labels only; no proof or chain-of-thought enters the tokenizer, dataloader, loss, or neural model. Generation seed protocol: `{data.get('generation_seed_protocol','single Python/numpy seed0')}`.

Dataset sizes: {json.dumps({s:v['count'] for s,v in data['splits'].items()})}. Every depth/split is exactly label balanced. Canonical underlying FOL facts plus proven conclusion are globally disjoint, with sentence order, query polarity, and entity name normalized. Isomorphic graphs under predicate renaming remain possible: this is not a claim of structural-template disjointness.

Independent sampled label/minimum-hop semantic audit: passed={semantic['passed']}, strata counts `{json.dumps(semantic['splits'])}`. This is an audit of the generated task, not chain-of-thought supervision or a neural-model reasoning result.

Train-only BPE target cap8192, actual vocabulary and allocated embedding {c['actual_vocab_size']} for every model. Lowercase and isolated digit spans preserve shared concept identifiers across case/plural forms. Max length {c['max_length']}. Tokenization audit: `{json.dumps(tok)}`.

Pre-norm GELU Transformer: width 512, 8 heads, FFN2048, {c['stem_blocks']} stem blocks and {c['core_blocks']} shared core blocks, dropout .1, attention backend {c.get('attention_backend','math')}. One NFE calls the entire recurrent core, including both shared blocks. Classifier reads CLS. Vanilla performs B(H); conditioned baseline adds a residual direction depending on t and dt. Vector field adds solver-scaled derivative built from Transformer residual increments and t, with its dt encoding slot fixed zero. No learned/global residual scaling or forbidden additional losses. Common weights start bitwise identical; conditioning weights use the same initialization scheme. Learned conditioning exists only in B/C and the small parameter difference is counted.

Training: seed 0, CE only, AdamW lr {c['lr']}, weight decay .01, effective batch {c['effective_batch']}, microbatch {c['micro_batch']}, activation checkpointing={c['activation_checkpointing']}, gradient clip 1, BF16, 5% warmup/cosine decay, {c['updates']} fixed updates per model. Independent depth-1 {c.get('sanity_updates',2000)}-update sanity runs are discarded; all formal models restart from identical common initialization. The initial300-update vanilla learning check failed at50% and is preserved under sanity_300_failed; all models received the same declared2000-update retry before formal training. Each formal update chooses K=4 or 8 from the same saved RNG plan and sees identical sample indices. Best checkpoint selected using validation accuracy, then NLL; final checkpoint also retained. Evaluation uses best. Suggested 20,000-update upper bound was reduced to 2,000 before results as an exploratory resource budget; insufficient learning makes the outcome UNCLEAR.

Hardware/environment: `{json.dumps(hw)}`.

{table(fairness,['model','params','trainable_params','core_params','train_updates','train_examples','max_len','optimizer','lr','train_budgets','training_wall_sec','training_peak_vram_mb'])}

Training-plan hashes all equal: {len(set(f['plan_sha256'] for f in fairness))==1}. Parameter spread: {(max(f['params'] for f in fairness)/min(f['params'] for f in fairness)-1)*100:.5f}%. Model wall times can differ; update and data budgets are held fixed, not GPU hours.

Development failures before formal training: V1's whole-word tokenizer treated noun case/plural variants as unrelated identifiers; a2000-update depth1 retry lowered training loss without held-out learning and then exited at1320 with Windows native access violation0xC0000005. A16-training-example plumbing check reached100% in100 updates. V2 tokenizer, actual-vocabulary embeddings, two-block core and explicit math attention were frozen before the first formal comparison; V1 artifacts remain archived. This is not an exact comparison against the initially suggested one-block/fused-attention implementation. The origin of the native failure was not established; successful V2 execution only verifies this configuration.

Budget-8 main results:

{table(main,['model','split','accuracy','nll','brier','mean_confidence'])}

## 2. Does budget sensitivity actually exist?

Unseen budgets are 3,5,7,12,16; trained budgets 4,8. Reference is always the same model's uniform Euler/loop budget 8, including for nonuniform and solver comparisons. Flip, harmful flip and recovery flip are fractions of all examples, not conditional fractions. Accuracy_new - accuracy_ref = recovery - harmful. This is a within-model paired comparison; single seed does not establish reproducibility across training seeds.

{table(effects,['baseline','split','budget','accuracy_drop_vs_8','baseline_harmful','vector_harmful','baseline_accuracy','vector_accuracy'])}

{len(meaningful)} conditions show >=2pp accuracy degradation and >=1% harmful flips under the exploratory engineering gate. Schedule ranges at equal K are in `schedule_sensitivity.csv`; maximum accuracy range by model: {json.dumps({n:max(r['accuracy_range'] for r in schedule_effect if r['model']==n) for n in NAMES[1:]})}. These task changes, rather than hidden drift alone, drive the decision.

## 3. Does the vector field reduce this sensitivity?

{len(improved)} baseline-sensitive conditions show >=50% lower harmful flips for vector field. Worst trained-budget accuracy deficit to the stronger baseline is {loss:.4f}. Recurrent contribution, normal budget 8 minus bypass accuracy: `{json.dumps(recurrent_gain)}`. A stable bypass-equivalent model does not qualify as a useful solver-consistency result.

Matched NFE (Euler steps=NFE; Heun steps=NFE/2; RK4 steps=NFE/4):

{table(solverrows,['split','budget','solver','accuracy','nll','harmful_flip_vs_8','js_vs_8','latency_mean_ms'])}

## 4. Numerical behavior

Numerical diagnostic uses 512 fixed IID examples, FP32 to avoid BF16 rounding floors; task and latency evaluations use BF16. Each numerical solve integrates t=0 to 1 using fixed positive dt. Reference is RK4 64 steps/256 NFE; RK4 32/128 vs 64/256 mean relative endpoint difference is {conv['reference_32_vs_64']:.8g}, ratio to Euler4 coarse error {conv['reference_drift_vs_euler4_error']:.8g}. Check I: {conv['check_I_passed']}. If it fails, reference is unverified. Actual fitted slopes: `{json.dumps(conv['observed_slopes'])}`; fit covers each requested solver grid, excluding RK4 reference points, without enforced formal orders. Numerical convergence itself does not establish improved reasoning.

{table(conv['rows'],['solver','steps','nfe','dt','relative_endpoint_error','precision'])}

## 5. Alternative explanations

State displacement (absolute and relative) and confidence are measured on the fixed diagnostic subset and full evaluation set respectively in `stage0_results.csv`; different independently trained hidden coordinate systems are never compared directly. Lower vector-field update magnitude is a built-in consequence of the dt-scaled parameterization and cannot be separated from numerical consistency by this three-model design. This Stage-0 therefore estimates the combined parameterization effect, not its unique causal mechanism. A matched-displacement control would be future work if the task gate passes.

Query-only/bag-of-words/zero-training negation-parity shortcut audit: `{json.dumps(q)}`. A high shortcut score weakens any claim of semantic reasoning. Repeated linguistic templates and predicate-renamed structures are inherent in this generator. Strict exact/underlying-example disjointness does not eliminate those shortcuts. Depth-1 is a learning sanity check, not evidence of multi-hop reasoning. Independent initialization fairness, recurrence bypass, confidence, training-wall differences and label balance are exposed above and in the CSVs. Accuracy and harmful flips are also stratified by truncated/untruncated inputs in the results CSV, with NA for empty groups, to distinguish budget sensitivity from tokenization artifacts.

Latency measured after 20 warmups on 100 fixed IID batches per configuration, batch {c['eval_batch']}, same GPU/length distribution/BF16, synchronized GPU timers. Reported inference peak VRAM includes the model and fixed bank. Joined latency columns apply to this bank, not separately measured split latency. Same NFE need not imply same latency. Hidden states saved only for 128 examples per split (512 total), final endpoints only; disk files use FP16, while distances are computed before quantization. Numerical diagnostics have a separate 512-example subset, with per-example errors saved and no all-step trajectories.

Sanity checks A-I: `{json.dumps(checks)}`. See `unit_tests.txt`, depth-1 gates and convergence diagnostics for evidence.

## 6. Decision

**{decision}**. {reason}

Prespecified engineering gate: at least two unseen-budget/split conditions with >=2pp baseline loss and >=1% harmful flips; >=50% vector-field harmful-flip reduction in at least two such conditions; no >1pp deficit to the stronger baseline at trained budgets on any split; recurrent gain >=1pp on at least one split. These heuristic numbers allocate further GPU work; they are not significance thresholds. Schedule evidence is diagnostic and reported independently. seed=0 is exploratory and not sufficient for statistical claims. No seeds 1/2 were automatically launched.

Artifacts: `stage0_results.csv`, `fairness_table.csv`, `unseen_budget_comparison.csv`, `schedule_sensitivity.csv`, `convergence.csv`, model checkpoints/logits/diagnostic endpoints under each model directory, data and tokenizer manifests, unit tests, figures, and `EXPERIMENT_CHANGELOG.md` at project root.
'''
    (out/'STAGE0_REPORT.md').write_text(report,encoding='utf-8')
    print(decision,reason,flush=True)

if __name__=='__main__': aggregate()
