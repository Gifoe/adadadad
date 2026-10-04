"""Read-only integrity checks on completed measurements and delivered artifacts."""
import csv,hashlib,json,pathlib
import numpy as np
import torch
from run_experiment import ROOT,OUT,CK,DATA,CONFIG,load_data

def verify():
    decision=json.loads((OUT/'DECISION.json').read_text());runtime=json.loads((OUT/'runtime_metrics.json').read_text())
    assert decision['decision'] in ['OFFICIAL BASELINE REPRODUCTION FAILURE','TRAINING CLOSURE FAILURE','METHOD TRAINING FAILURE','ID CLOSED — MECHANISM NOT SUPPORTED','MECHANISM SUPPORTED — UTILITY NOT SUPPORTED','GO']
    train,test=load_data();split=json.loads((DATA/'closure_split_manifest.json').read_text())
    a=set(split['validation_ids']);b=set(split['test_ids']);assert len(a)==len(b)==1500 and not a&b
    assert not set(train['sample_ids'])&set(test['sample_ids'])
    assert set(train['hops'])=={1,2,3,4,5} and len(train['hops'])==62000
    assert hashlib.sha256((ROOT/'model_extension.py').read_bytes()).hexdigest()==hashlib.sha256((ROOT.parent/'transition_grounded_recurrent_reasoning_seed42_v1/model_extension.py').read_bytes()).hexdigest()
    assert hashlib.sha256((CK/'initial_weights.pt').read_bytes()).hexdigest()=='acfc223cc2d6e99e18c3601f71a2853822a84581c5a726d27fda02f5e768b3f0'
    for n,r in runtime['training'].items():
        stages=r['stages'];assert [s['D'] for s in stages]==list(range(2,2+len(stages)))
        assert sum(s['updates'] for s in stages)==r['global_updates']
        assert sum(s['examples_seen'] for s in stages)==r['examples_seen']
        assert all(s['updates']<=50000 and s['optimizer_reset'] and s['scheduler_horizon']==int(np.ceil(sum(train['hops']<=s['D'])/128))*100001 for s in stages)
        failed=[s for s in stages if not s['criterion_completed']]
        assert not failed or failed==[stages[-1]]
        for d,v in r['final_id'].items():assert v['total']==375 and abs(v['accuracy']-v['correct']/375)<1e-12
        if n!='official_baseline':
            assert abs(r['id_macro']-np.mean([v['accuracy'] for v in r['final_id'].values()]))<1e-12
            curves=list(csv.DictReader((OUT/f'{n}_learning_curve.csv').open()))
            for s in stages:
                if s['criterion_completed']:
                    rows=[q for q in curves if int(q['stage'])==s['D']];assert len(rows)>=3
                    assert all(float(q['current_stage_accuracy'])>=.95 for q in rows[-3:])
                    assert [int(q['stage_updates']) for q in rows[-3:]]==list(range(s['updates']-1000,s['updates']+1,500))
        ck=CK/(('matched_baseline' if n=='baseline' else n)+'_final.pt')
        state=torch.load(ck,weights_only=True,map_location='cpu');assert state['step']==r['global_updates']
        assert all(torch.isfinite(t).all() for t in state['model_state_dict'].values() if torch.is_floating_point(t))
        del state
    if decision['decision']=='OFFICIAL BASELINE REPRODUCTION FAILURE':
        assert set(runtime['training'])=={'official_baseline'} and not decision['core_evaluation_run']
    if not decision['core_evaluation_run']:
        for n in ['baseline','grounded']:
            for prefix in ['final_accuracy_matrix','state_accuracy','intervention_results']:assert not list(csv.DictReader((OUT/f'{prefix}_{n}.csv').open()))
    expected=['d2_learning_curve','d3_learning_curve','d4_learning_curve','d5_learning_curve','updates_to_95','id_accuracy','recurrence_hop_heatmap_baseline','recurrence_hop_heatmap_grounded','state_accuracy_heatmap_baseline','state_accuracy_heatmap_grounded','overthinking_curve','counterfactual_intervention']
    assert all((ROOT/'figures'/f'{n}.png').stat().st_size>1000 for n in expected)
    manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text())
    for r in manifest['checkpoints']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    print(json.dumps(dict(status='PASS',decision=decision['decision'],checkpoint_finite=True,hashes_verified=True,split_disjoint=True,method_unchanged=True,curriculum_stop_verified=True,core_evaluation_run=decision['core_evaluation_run']),indent=2))
if __name__=='__main__':verify()
