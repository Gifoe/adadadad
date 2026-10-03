"""Audit completed artifacts; regenerate report after execution diagnostics."""
import csv,json,time,hashlib,math
import numpy as np
import torch
from PIL import Image
from run_experiment import ROOT,OUT,CK,dump,CONFIG
from analysis_report import report,report_invalid,FIGURE_NAMES
torch.set_num_threads(1)
runtime=json.loads((OUT/'runtime_metrics.json').read_text())
hist={}
for name in ['baseline','grounded']:
    with (OUT/f'train_history_{name}.csv').open() as f:rows=list(csv.DictReader(f))
    assert [int(r['step']) for r in rows]==list(range(1,len(rows)+1))
    assert all(math.isfinite(float(r['loss'])) for r in rows)
    hist[name]=rows
    ck=torch.load(CK/f'{name}_final.pt',weights_only=True,map_location='cpu')
    assert ck['step']==len(rows)==runtime['training'][name]['optimizer_steps']
    assert torch.equal(ck['model_state_dict']['token_embedding.weight'],ck['model_state_dict']['lm_head.weight'])
    assert all(torch.isfinite(t).all() for t in ck['model_state_dict'].values())
    del ck
assert len(hist['baseline'])==len(hist['grounded'])
assert [(r['step'],r['max_hop'],r['lr']) for r in hist['baseline']]==[(r['step'],r['max_hop'],r['lr']) for r in hist['grounded']]
plan=np.load(ROOT/'data/sample_plan.npy');data=np.load(ROOT/'data/generated/train.npz')
assert plan.shape==(20000,128) and int(data['hops'][plan].max())==5
for start,end,maxhop in [(0,2000,2),(2000,4000,3),(4000,7000,4),(7000,20000,5)]:assert data['hops'][plan[start:end]].max()<=maxhop
meta=json.loads((ROOT/'RUN_METADATA.json').read_text())
initial_hash=hashlib.sha256((CK/'initial_weights.pt').read_bytes()).hexdigest();assert meta['initial_sha256']==initial_hash
integrity=json.loads((ROOT/'data/integrity_check.json').read_text());unit=json.loads((OUT/'unit_tests.json').read_text())
assert integrity['status']==unit['status']=='PASS' and integrity['mismatches']==0
if runtime['training_valid']:report()
else:report_invalid()
image_sizes={}
for name in FIGURE_NAMES:
    p=ROOT/'figures'/f'{name}.png'
    with Image.open(p) as im:image_sizes[name]=list(im.size);im.verify()
    assert min(image_sizes[name])>300
audit=dict(status='PASS',completed=True,training_valid=runtime['training_valid'],formal_steps_per_arm=len(hist['baseline']),examples_seen_per_arm=len(hist['baseline'])*128,shared_initial_sha256=initial_hash,shared_sample_plan_sha256=hashlib.sha256((ROOT/'data/sample_plan.npy').read_bytes()).hexdigest(),same_step_hop_lr_records=True,max_training_hop=5,finite_final_weights=True,tied_head_verified=True,figure_sizes=image_sizes,skipped_evaluation_reason=None if runtime['training_valid'] else 'predeclared ID sanity gate failed at matched maximum budget',time=time.strftime('%Y-%m-%d %H:%M:%S'))
dump(OUT/'completion_audit.json',audit)
meta.update(completed=audit['time'],execution_status='COMPLETE' if runtime['training_valid'] else 'COMPLETE_WITH_INVALID_TRAINING',formal_steps_per_arm=audit['formal_steps_per_arm'])
dump(ROOT/'RUN_METADATA.json',meta)
print(json.dumps(audit,indent=2),flush=True)
