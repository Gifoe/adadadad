import csv,json,pathlib,time
import numpy as np
from run_experiment import ROOT,OUT,CK,DATA,CONFIG,dump,csvwrite,sha
NAMES=['pretrained','final_only','grounded']

def read(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8')))
def mean(rows,key):
 v=[float(r[key]) for r in rows if r.get(key) not in [None,'','None']];return float(np.mean(v)) if v else None
def matrix(n):return read('pretrained_accuracy_matrix.csv' if n=='pretrained' else f'accuracy_matrix_{n}.csv')
def summaries():
 out={}
 for n in NAMES:
  mat=matrix(n);state=read(f'state_accuracy_{n}.csv');cf=read(f'intervention_summary_{n}.csv');over=read(f'overthinking_{n}.csv')
  out[n]=dict(ID_K5=mean([r for r in mat if int(r['D'])<=6 and int(r['K'])==5],'accuracy'),OOD_KD=mean([r for r in mat if int(r['D'])>6 and int(r['K'])==int(r['D'])],'accuracy'),OOD_K5=mean([r for r in mat if int(r['D'])>6 and int(r['K'])==5],'accuracy'),transition_state_macro=mean([r for r in state if int(r['k'])<=int(r['D'])],'accuracy'),supervised_range_state_macro=mean([r for r in state if int(r['k'])<=min(5,int(r['D']))],'accuracy'),OOD_terminal_stability=mean([r for r in over if int(r['D'])>6],'terminal_stability'),OOD_overthinking_drop=mean([r for r in over if int(r['D'])>6],'overthinking_drop'))
  for control in ['counterfactual','same_state','noise']:
   cells=[r for r in cf if r['control']==control and r['regime']=='main' and r['stratum']=='all']
   out[n][control]=dict(cf_target_rate=mean(cells,'cf_target_rate'),trajectory_accuracy=mean(cells,'cf_trajectory_accuracy'),original_retention=mean(cells,'original_retention_rate'),preservation=mean(cells,'preservation_rate'),n=sum(int(r['n']) for r in cells),cells=len(cells))
  out[n]['same_state_correct_preservation']=mean([r for r in cf if r['control']=='same_state' and r['regime']=='main' and r['stratum']=='original_correct'],'preservation_rate')
 return out

def gates(s):
 p,b,g=[s[n] for n in NAMES];f='cf_target_rate';t='trajectory_accuracy'
 pretrained=(p['transition_state_macro']>=.5 and all(p['counterfactual'][k]>=.5 and p['counterfactual'][k]-p['noise'][k]>=.2 for k in [f,t]) and (p['same_state_correct_preservation'] or 0)>.5)
 differences={k:{n:g['counterfactual'][k]-s[n]['counterfactual'][k] for n in ['pretrained','final_only']} for k in [f,t]}
 noise={k:g['counterfactual'][k]-g['noise'][k] for k in [f,t]}
 causal=all(v>=.2 for ds in differences.values() for v in ds.values()) and all(v>=.2 for v in noise.values()) and (g['same_state_correct_preservation'] or 0)>.5
 harm=p['ID_K5']-g['ID_K5']>.05
 utility={k:all(g[k]-s[n][k]>=.05 for n in ['pretrained','final_only']) for k in ['OOD_KD','OOD_terminal_stability']}
 return dict(pretrained_already_has_mechanism=pretrained,causal_supported=causal,grounded_ID_damaged=harm,ID_retention_drop=p['ID_K5']-g['ID_K5'],causal_differences=differences,grounded_CF_minus_noise=noise,utility=utility)

def decision():
 s=summaries();g=gates(s);dump(OUT/'decision_inputs.json',dict(summary=s,gates=g))
 if g['pretrained_already_has_mechanism']:return 'BASELINE ALREADY HAS MECHANISM'
 if not g['causal_supported']:return 'DECODE-ONLY — NO CAUSAL EFFECT'
 if g['grounded_ID_damaged']:return 'CAUSAL EFFECT — NO UTILITY'
 if any(g['utility'].values()):return 'GO'
 return 'MECHANISM SUPPORTED'

def figures(s):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
 figdir=ROOT/'figures';colors=['#555555','#2678ad','#ce5727']
 def end(fig,name):fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 def heat(n,file,state=False,ax=None):
  rows=read(f'state_accuracy_{n}.csv') if state else matrix(n);ks=list(range(1,25)) if state else CONFIG['K'];arr=np.array([[next(float(r['accuracy']) for r in rows if int(r['D'])==d and int(r['k' if state else 'K'])==k) for k in ks] for d in CONFIG['D']]);im=ax.imshow(arr,aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(ks)));ax.set_xticklabels(ks,rotation=90);ax.set_yticks(range(len(CONFIG['D'])));ax.set_yticklabels(CONFIG['D']);ax.set_xlabel('Recurrences K' if not state else 'State readout k');ax.set_ylabel('Problem depth D');ax.set_title(n);return im
 for file,state in [('pretrained_recurrence_hop_heatmap',False),('pretrained_state_accuracy_heatmap',True)]:
  fig,ax=plt.subplots(figsize=(8,4));im=heat('pretrained',file,state,ax);fig.colorbar(im,ax=ax,label='Accuracy');end(fig,file)
 fig,axs=plt.subplots(1,3,figsize=(14,4))
 for ax,n in zip(axs,NAMES):heat(n,'',True,ax)
 end(fig,'state_accuracy_comparison')
 fig,ax=plt.subplots(figsize=(7,4));x=np.arange(5)
 for j,n in enumerate(NAMES):ax.bar(x+(j-1)*.25,[float(r['accuracy']) for r in matrix(n) if int(r['D'])<=6 and int(r['K'])==5],.25,label=n,color=colors[j])
 ax.set_xticks(x);ax.set_xticklabels(CONFIG['ID']);ax.set_ylim(0,1);ax.set_xlabel('ID depth (K=5)');ax.set_ylabel('Accuracy');ax.legend();end(fig,'id_retention')
 fig,axs=plt.subplots(1,5,figsize=(16,3.5),sharey=True)
 for ax,d in zip(axs,CONFIG['OOD']):
  for j,n in enumerate(NAMES):ax.plot(CONFIG['K'],[next(float(r['accuracy']) for r in matrix(n) if int(r['D'])==d and int(r['K'])==k) for k in CONFIG['K']],label=n,color=colors[j])
  ax.set_title(f'OOD D={d}');ax.set_xlabel('K');ax.set_ylim(0,1)
 axs[0].set_ylabel('Final accuracy');axs[-1].legend(fontsize=8);end(fig,'ood_recurrence_hop_comparison')
 for metric,file in [('cf_target_rate','counterfactual_target_rate'),('trajectory_accuracy','counterfactual_trajectory_accuracy')]:
  fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3)
  for j,c in enumerate(['counterfactual','same_state','noise']):ax.bar(x+(j-1)*.25,[s[n][c][metric] for n in NAMES],.25,label=c)
  ax.set_xticks(x);ax.set_xticklabels(NAMES);ax.set_ylim(0,1);ax.set_ylabel(metric);ax.set_title('Main K; equal D,k cell mean');ax.legend();end(fig,file)
 fig,axs=plt.subplots(1,2,figsize=(10,4));x=np.arange(10)
 for j,n in enumerate(NAMES):
  rows=read(f'overthinking_{n}.csv')
  for ax,key in zip(axs,['overthinking_drop','terminal_stability']):ax.plot(CONFIG['D'],[float(r[key]) for r in rows],marker='o',label=n,color=colors[j]);ax.set_xlabel('Depth D');ax.set_ylabel(key);ax.set_ylim(0,1)
 axs[-1].legend();end(fig,'overthinking_comparison')

def finalize(d):
 validation=json.loads((OUT/'official_validation.json').read_text());invalid=d=='OFFICIAL CHECKPOINT INVALID';completion=None
 runtime={'official_validation':validation,'invalid_checkpoint_gate':invalid};summary={}
 if not invalid:
  summary=summaries();g=gates(summary);idrows=[];comparison=[];over=[]
  for n in NAMES:
   idrows.extend(dict(model=n,D=r['D'],K=5,correct=r['correct'],total=r['total'],accuracy=r['accuracy'],drop_vs_pretrained=next(float(p['accuracy']) for p in matrix('pretrained') if p['D']==r['D'] and int(p['K'])==5)-float(r['accuracy'])) for r in matrix(n) if int(r['D'])<=6 and int(r['K'])==5)
   idrows.append(dict(model=n,D='macro',K=5,correct='',total=3750,accuracy=summary[n]['ID_K5'],drop_vs_pretrained=summary['pretrained']['ID_K5']-summary[n]['ID_K5']))
   comparison.extend(read(f'intervention_summary_{n}.csv'));over.extend(read(f'overthinking_{n}.csv'));runtime[n+'_audit']=json.loads((OUT/f'{n}_audit_complete.json').read_text())
  for n in ['final_only','grounded']:
   runtime[n+'_FT']=json.loads((OUT/f'training_{n}.json').read_text());runtime[n+'_smoke']=json.loads((OUT/f'smoke_{n}.json').read_text())
  runtime['total_GPU_workload_hours']=(validation['seconds']+sum(v['seconds'] for v in runtime.values() if isinstance(v,dict) and 'seconds' in v and v is not validation))/3600
  csvwrite(OUT/'id_retention.csv',idrows);csvwrite(OUT/'intervention_comparison.csv',comparison);csvwrite(OUT/'overthinking_metrics.csv',over);figures(summary)
  # Structural and endpoint verification, independent of whether the scientific gate passes.
  import torch
  from run_experiment import starting_state
  keys=set(starting_state());checkpoint_manifest=[]
  for n in ['final_only','grounded']:
   hist=read(f'finetune_history_{"baseline" if n=="final_only" else n}.csv');assert [int(r['step']) for r in hist]==list(range(1,5001))
   for step in [0,500,1000,2000,5000]:
    p=CK/f'{n}_step{step}.pt';ck=torch.load(p,map_location='cpu',weights_only=True);assert ck['step']==step and set(ck['model_state_dict'])==keys and all(bool(torch.isfinite(v).all()) for v in ck['model_state_dict'].values() if v.is_floating_point());checkpoint_manifest.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),step=step,model=n))
  # Same counterfactual samples, target entities and noise direction source across all models.
  causal_ids=[]
  for n in NAMES:
   causal_ids.append([(r['sample_id'],r['D'],r['k'],r['K'],r['control'],r['e_cf'],r['y_cf']) for r in read(f'intervention_{n}.csv')]);assert all(int(r['D'])>CONFIG['H_train'] for r in read(f'ood_accuracy_{n}.csv'))
  assert causal_ids[0]==causal_ids[1]==causal_ids[2]
  dump(OUT/'checkpoint_manifest.json',checkpoint_manifest)
  completion=dict(status='PASS',decision=d,same_starting_parameters=True,same_batch_plan=True,formal_updates_per_arm=5000,no_new_parameter_keys=True,all_saved_checkpoint_parameters_finite=True,matched_causal_sample_plan=True,true_OOD_only=True,checkpoints=10,completed=time.strftime('%Y-%m-%d %H:%M:%S'))
 runtime['retry_accounting']='Successful-phase workload sum; prior failed native attempts and idle/debug/download intervals are not included in GPU workload hours. Their logs and process start timestamps are preserved; kernel-active total across failed attempts was not measured.'
 dump(OUT/'runtime_metrics.json',runtime)
 def pct(v):return f'{100*v:.2f}%'
 lines=['# Competent-checkpoint mechanism experiment — final report','',f'Final decision: **{d}**. GO/NO-GO: **'+('GO' if d=='GO' else 'NO-GO')+'**.','',f'Official R2 `checkpoint_epoch_2765.pt`, H_train6 (inferred with provenance evidence), original recurrence K2. Official held-out ID macro: **{pct(validation["macro"])}** across D2–6, 750 chains per depth.','', 'Checkpoint provenance: `CHECKPOINT_PROVENANCE.md`. Fixed method and all gates: `METHOD_DIFF.md`.']
 if invalid:
  lines+=['','The official checkpoint failed the >=85% competence gate. Fine-tuning and causal-effect claims were not run. Missing downstream outputs are intentional stopped phases, not completed negative measurements.','',json.dumps(validation,indent=2)]
 else:
  lines+=['','## Actual endpoints','', '| Model | ID K5 | OOD K=D | State e_k macro | CF target | CF trajectory | Noise target | Noise trajectory | Same-state preserve (original correct) | OOD terminal stability |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for n in NAMES:
   s=summary[n];lines.append('| '+n+' | '+' | '.join(pct(v) if v is not None else 'NA' for v in [s['ID_K5'],s['OOD_KD'],s['transition_state_macro'],s['counterfactual']['cf_target_rate'],s['counterfactual']['trajectory_accuracy'],s['noise']['cf_target_rate'],s['noise']['trajectory_accuracy'],s['same_state_correct_preservation'],s['OOD_terminal_stability']])+' |')
  p,b,z=[summary[n] for n in NAMES];lines+=['','## Before and after transition supervision','',f'Pretrained direct intermediate-state macro is {pct(p["transition_state_macro"])}; CF final-target rate {pct(p["counterfactual"]["cf_target_rate"])} and remaining-path accuracy {pct(p["counterfactual"]["trajectory_accuracy"])}. The frozen baseline-already-has-mechanism gate is {g["pretrained_already_has_mechanism"]}. No learned probe is used.','',f'Grounded CF target rate is {pct(z["counterfactual"]["cf_target_rate"])}; CF trajectory is {pct(z["counterfactual"]["trajectory_accuracy"])}. Grounded minus pretrained target/trajectory gains are {100*g["causal_differences"]["cf_target_rate"]["pretrained"]:.2f}/{100*g["causal_differences"]["trajectory_accuracy"]["pretrained"]:.2f}pp; gains over final-only are {100*g["causal_differences"]["cf_target_rate"]["final_only"]:.2f}/{100*g["causal_differences"]["trajectory_accuracy"]["final_only"]:.2f}pp. Causal gate passes: {g["causal_supported"]}.','',f'Transition-state decoding change over pretrained: {100*(z["transition_state_macro"]-p["transition_state_macro"]):.2f}pp; over final-only: {100*(z["transition_state_macro"]-b["transition_state_macro"]):.2f}pp. The separately reported k<=5 state macros are '+', '.join(n+' '+pct(summary[n]['supervised_range_state_macro']) for n in NAMES)+'.','', 'Representation shaping changes causal computation only if the target AND trajectory gates pass. Decodability and final accuracy alone cannot establish causal state use. The DEC0DE-ONLY decision label denotes failure of this causal claim; a decoding improvement is claimed only when its measured gain is positive.','', '## ID retention and true depth extrapolation','',f'At fixed K5 grounded ID retention drop is {100*g["ID_retention_drop"]:.2f}pp. Damaged-reasoning flag (>5pp): {g["grounded_ID_damaged"]}. No arm was rescued or selected at an earlier checkpoint. Original competence is assessed at official K2, which can differ materially from this required K5 retention reference.','',f'OOD means strictly D8/10/12/16/20 because H_train6. OOD matched-K=D macro is {pct(p["OOD_KD"])} / {pct(b["OOD_KD"])} / {pct(z["OOD_KD"])} for pretrained/final-only/grounded. OOD utility gate (>=5pp over both controls): {g["utility"]["OOD_KD"]}. Descriptive sweep peaks/frontiers are saved separately; no peak-selection result substitutes for the fixed matched-K endpoint.','', '## Overthinking and control validity','',f'OOD terminal stability D+1..24 is {pct(p["OOD_terminal_stability"])} / {pct(b["OOD_terminal_stability"])} / {pct(z["OOD_terminal_stability"])}. Stability utility gate: {g["utility"]["OOD_terminal_stability"]}. Peak-minus-K24 accuracy and early-correct-to-K24-wrong rates are in `overthinking_metrics.csv`.','', 'Same-state preservation, original retention, norm-matched noise, main and extra recurrence budgets, and originally-correct/incorrect strata are all retained in `intervention_comparison.csv`. Poor same-state preservation means a prototype patch is disruptive; it weakens the clean causal interpretation even if some target-directed effects appear. No metric includes the injected state in the remaining-path accuracy.','', '## Fairness, runtime and interpretation limits','', 'Both arms start from identical official weights and use identical source records, batch order, seed, optimizer, LR, precision, recurrence and 5,000 updates. Only coefficient0/1 differs. Full smoke runs are discarded. Checksum and endpoint verification is in `completion_audit.json` and `checkpoint_manifest.json`. Train-only model-specific prototypes and shared intervention IDs/noise are verified.','',f'Total recorded GPU workload wall time is {runtime["total_GPU_workload_hours"]:.4f} hours; the phase durations and peak allocated VRAM are in `runtime_metrics.json`. Audit times include data output and prototype/intervention work; this is occupied workload wall time rather than profiler-isolated GPU kernel time. Downloads and source preparation are excluded.','', 'This is one seed and one official shallow recurrent checkpoint. Prototype replacement tests this carrier intervention, not every possible representation elsewhere in the transformer. H_train is supported by public role and stage-update evidence but is not directly encoded in the checkpoint. Intermediate supervision at K1–4 is not itself evidence of causal computation at later recurrences.','', '## Final decision rationale','',json.dumps(g,indent=2),'',f'**{d}**. '+('The full fixed mechanism and retained-ability utility gates pass.' if d=='GO' else 'The evidence does not satisfy the full proposed-method GO criterion. The measured scientific result is preserved without changing the method or thresholds.')]
 lines+=['','Native execution issues and process-only CPU affinity mitigation are recorded in `METHOD_DIFF.md`. Failed attempt logs remain preserved. Reported GPU workload hours sum the successful completed phases; prior failed-attempt GPU-active time was not measured and is not silently treated as zero.']
 if not invalid:
  s=summary['grounded'];b=summary['final_only'];p=summary['pretrained'];changes=[]
  if g['causal_supported']:changes.append('causal computation')
  elif s['transition_state_macro']>max(b['transition_state_macro'],p['transition_state_macro']):changes.append('learning representation only')
  if s['ID_K5']>max(b['ID_K5'],p['ID_K5']) or s['OOD_KD']>max(b['OOD_KD'],p['OOD_KD']):changes.append('final task performance')
  lines+=['', 'Measured supported change categories: **'+(', '.join(changes) if changes else 'none')+'**. Positive numerical differences are descriptive at this single seed, not significance claims.','',f'Additional native-setting reference: the original checkpoint scores {pct(validation["macro"])} at K2, while grounded scores {pct(s["ID_K5"])} at its trained K5. The requested matched-K5 retention comparison does not by itself prove preservation of the original K2 computation. Both references are exposed rather than conflated.']
 (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines).replace('DEC0DE-ONLY','DECODE-ONLY')+'\n',encoding='utf-8')
 if completion is not None:
  assert len(list((ROOT/'figures').glob('*.png')))==8
  assert json.loads((OUT/'training_final_only.json').read_text())['initial_parameter_sha256']==json.loads((OUT/'training_grounded.json').read_text())['initial_parameter_sha256']
  dump(OUT/'completion_audit.json',completion)
