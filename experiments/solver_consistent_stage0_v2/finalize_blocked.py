"""Create the required honest NA artifacts after the frozen A6 gate failed."""
import csv, hashlib, json, pathlib, time
from runtime import ROOT,build
from models.common import counts
ART=ROOT/'artifacts';ART.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'data/generated/manifest.json').read_text());cfg=json.loads((ROOT/'configs/frozen.json').read_text());short=json.loads((ART/'data_shortcut_summary.json').read_text());tok=json.loads((ROOT/'data/tokenized/audit.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
models=['vanilla_loop','step_conditioned_loop','vector_field'];params={m:counts(build(cfg,m))['params'] for m in models}
crashes=[
 {'attempt':1,'exception':'0xc0000005','fault_module':'unknown / StackHash','faulthandler_location':'torch.utils.checkpoint.get_device_states','checkpoint_update':25},
 {'attempt':2,'exception':'0xc0000005','fault_module':'torch_cpu.dll','faulthandler_location':'torch autograd backward','checkpoint_update':25},
 {'attempt':3,'exception':'0x80000003','fault_module':'nvcuda64.dll','faulthandler_location':'checkpoint recompute attention / autograd backward','checkpoint_update':25}]
(ART/'infra_native_crashes.json').write_text(json.dumps({'frozen_config':{'torch':'2.8.0+cu128','cuda_runtime':'12.8','attention_backend':'math','precision':'bf16','activation_checkpointing':True,'micro_batch':16,'effective_batch':128,'max_length':1376,'deterministic':True},'attempts':crashes,'result':'FAIL_THREE_NATIVE_CRASHES','interpretation':'Fault modules/stack locations are evidence, not proven root causes. No setting changed between attempts.'},indent=2),encoding='utf-8')
checks={'A':{'status':'NOT_RUN','evidence':'A6 failed before learning gates'},'B':{'status':'PASS','evidence':'VectorField.forward(self,H,t,padding_mask) signature test'},'C':{'status':'PASS','evidence':'inference-step weight immutability unit test'},'D':{'status':'PASS','evidence':'Euler/Heun/RK4 NFE tests'},'E':{'status':'PASS','evidence':'positive normalized schedule tests'},'F':{'status':'PASS','evidence':'eval repeat determinism/dropout test'},'G':{'status':'PASS','evidence':params},'H':{'status':'PASS','evidence':'135000 unique exact/canonical/graph IDs'},'I':{'status':'NOT_RUN','evidence':'trained Model C unavailable'},'J':{'status':'PASS','evidence':'135000/135000 full semantic audit, zero failures'},'K':{'status':'PASS','evidence':{'max_heldout_accuracy':short['max_heldout_accuracy'],'failure_count':0}},'L':{'status':'FAIL','evidence':'Vanilla A6 run crashed natively three times from the same atomic update-25 checkpoint; 50 updates/validation/round-trip not completed'}}
(ART/'sanity_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
decision={'pipeline_state':'INFRA_BLOCKED','formal_seed0_complete':False,'scientific_decision':'NA','reason':'A6 frozen configuration produced three native crashes; prompt requires stopping before A7/formal training','formal_updates':{m:0 for m in models},'go_stop_unclear':'NOT_APPLICABLE','data_manifest_payload_sha256':manifest['manifest_payload_sha256'],'config_sha256':sha(ROOT/'configs/frozen.json')}
(ART/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
stage_fields=['model','seed','split','reasoning_depth','budget','nfe','solver','schedule','checkpoint_type','checkpoint_update','accuracy','nll','brier','flip_rate_vs_8','harmful_flip_vs_8','recovery_flip_vs_8','js_vs_8','abs_logit_diff_vs_8','confidence_diff_vs_8','mean_confidence','hidden_endpoint_distance','latency_mean_ms','latency_p95_ms','peak_vram_mb','examples','truncation_group','data_hash','config_hash','plan_hash','run_status']
with (ART/'stage0_results.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=stage_fields);w.writeheader()
 for m in models:w.writerow({k:(m if k=='model' else 0 if k=='seed' else manifest['manifest_payload_sha256'] if k=='data_hash' else sha(ROOT/'configs/frozen.json') if k=='config_hash' else 'NOT_RUN_INFRA_BLOCKED' if k=='run_status' else 'NA') for k in stage_fields})
fair_fields=['model','parameters','parameter_difference_vs_min_pct','common_initialization','formal_updates','training_time_sec','peak_vram_mb','status']
with (ART/'fairness_table.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fair_fields);w.writeheader();mn=min(params.values())
 for m in models:w.writerow({'model':m,'parameters':params[m],'parameter_difference_vs_min_pct':(params[m]/mn-1)*100,'common_initialization':'PASS','formal_updates':0,'training_time_sec':'NA','peak_vram_mb':'NA','status':'NOT_RUN_INFRA_BLOCKED'})
for name,fields in {
 'unseen_budget_comparison.csv':['model','split','budget','accuracy','accuracy_drop_vs_8','harmful_flip','status'],
 'schedule_sensitivity.csv':['model','split','budget','solver','schedule','accuracy','harmful_flip','status'],
 'convergence.csv':['solver','steps','nfe','dt','relative_endpoint_error','observed_slope','reference_gate','status']}.items():
 with (ART/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow({k:'NOT_RUN_INFRA_BLOCKED' if k=='status' else 'NA' for k in fields})
report=f'''# Solver-Consistent Recurrent Reasoning — Stage 0 V2

**Pipeline state: INFRA_BLOCKED**

**正式 seed=0 对照实验未完成。**

The unique blocking state is `INFRA_BLOCKED`. No GO / UNCLEAR / STOP scientific decision is permitted because `FORMAL_COMPLETE` was not reached.

## 1. Experimental setup

- Data: 135,000 examples; `prontoqa-symmetric-component-v2`; seed namespace 2000; manifest payload SHA256 `{manifest['manifest_payload_sha256']}`.
- Repair: replace the isolated opposite-goal dummy with a predicate-renamed mirrored proof component. Its queried-entity start fact has opposite polarity, matching unsigned entity/frequency statistics while keeping the signed decoy chain unreachable.
- Full audit: 135,000/135,000 passed parser round-trip, label reachability, opposite non-reachability, minimum depth, contradiction, checksums, and global exact/canonical/predicate-isomorphic deduplication.
- Shortcut gate: PASS. Highest no-training heuristic 55.00%; highest structural classifier 55.44%; highest single surface feature 53.28%; antecedent-frequency 50.00% in every IID/OOD stratum.
- Tokenizer: train-only BPE, actual vocabulary {cfg['actual_vocab_size']}, max length {cfg['max_length']}, SHA256 `{tok['tokenizer_sha256']}`; all split truncation rates <=0.06%.
- Models: parameter counts {params}. Maximum difference {(max(params.values())/min(params.values())-1)*100:.3f}%.
- Environment: Windows 10 build 18363, RTX 5090, Torch 2.8.0+cu128, CUDA runtime 12.8, driver 616.56, BF16, math SDPA, activation checkpointing, deterministic algorithms, microbatch 16/effective batch 128.

## 2. Does budget sensitivity actually exist?

`NOT_RUN_INFRA_BLOCKED`. There are no trained best/final checkpoints and no task-level unseen-budget or harmful-flip results.

## 3. Does vector field reduce sensitivity?

`NOT_RUN_INFRA_BLOCKED`. Any method comparison would be fabricated.

## 4. Numerical behavior

Implementation-only ODE/NFE/schedule tests passed. Trained-Model-C convergence and reference Check I are `NOT_RUN_INFRA_BLOCKED`. Numerical convergence cannot be inferred from the toy implementation test.

## 5. Alternative explanations

The data and registered shortcut suite passed, so the stop is not caused by the old antecedent-frequency leak. The real-shape VRAM probe passed at microbatch 16 with about 4.90 GB allocated and 6.12 GB reserved. The frozen Vanilla A6 job then crashed three times from the same atomic update-25 checkpoint. WER records show `unknown/StackHash` with `0xc0000005`, `torch_cpu.dll` with `0xc0000005`, and `nvcuda64.dll` with `0x80000003`. These are fault locations, not established causes. The required 50-update validation/checkpoint round-trip was never completed.

## 6. Decision

Scientific GO / UNCLEAR / STOP: **NA**. Pipeline state: **INFRA_BLOCKED**. Checks A and I are `NOT_RUN`; Check L is `FAIL`; all formal model updates are 0.

## 7. Execution integrity

Completed: A0 environment/repository audit, A1 full data repair/generation, A2 full semantics/dedup/shortcut gate, A3 tokenizer/length freeze, A4 model interface/fairness tests, A5 solver tests, and A6 real-shape sizing. Failed: A6 sustained stability. Not run: three-model 50-update completion, A6 2,000-update Vanilla stability, A7 learning gates, A8 formal training, A9–A14 evaluations, latency, recurrent contribution, and trained numerical convergence. Unknown metrics are `NA`, not zero. Main figures were **NOT GENERATED** because complete real model results do not exist.
'''
(ART/'STAGE0_REPORT.md').write_text(report,encoding='utf-8')
pub={'formal_seed0_complete':False,'pipeline_state':'INFRA_BLOCKED','scientific_decision':'NA','generator_version':manifest['generator_version'],'data_count':manifest['count'],'manifest_payload_sha256':manifest['manifest_payload_sha256'],'required_figures':'NOT GENERATED','excluded_large_files':['JSONL data','token arrays','checkpoints']}
(ROOT/'PUBLICATION_MANIFEST.json').write_text(json.dumps(pub,indent=2),encoding='utf-8')
print(json.dumps(decision,indent=2))
