import inspect, math, json
import numpy as np
import pytest
import torch
from solvers.fixed_step import integrate,schedule
from models.common import Model,VectorField,counts
from runtime import ROOT,build

@pytest.mark.parametrize('solver,factor',[('euler',1),('heun',2),('rk4',4)])
def test_nfe_and_known_ode(solver,factor):
    errors=[]
    for n in [2,4,8,16]:
        calls=[]
        def f(h,t): calls.append(t); return -h
        y,nfe=integrate(f,torch.tensor(1.,dtype=torch.float64),schedule(n),solver)
        assert nfe==len(calls)==factor*n
        errors.append(abs(y.item()-math.exp(-1)))
    assert all(b<a for a,b in zip(errors,errors[1:]))

@pytest.mark.parametrize('kind',['uniform','front-loaded','back-loaded'])
def test_schedule(kind):
    for n in [1,3,4,5,7,8,12,16,32,64]:
        ds=schedule(n,kind); assert min(ds)>0 and math.isclose(sum(ds),1.)

def test_no_dt_leak_and_weights():
    assert list(inspect.signature(VectorField.forward).parameters)==['self','H','t','padding_mask']
    c={'d_model':32,'n_heads':4,'d_ff':128,'dropout':.1,'core_blocks':1,'stem_blocks':1,'max_length':16,'vocab_size':100,'seed':0}
    m=build(c,'vector_field').eval(); x=torch.tensor([[2,5,6,3,0]])
    before={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad():
        z=m(x,4); z2=m(x,4); m(x,8); m(x,8,'rk4')
    assert torch.equal(z,z2)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    with pytest.raises(TypeError): m.core(torch.zeros(1,5,32),0.,x.ne(0),dt=.25)

def test_parameter_fairness_and_shared_initialization():
    path=ROOT/'configs/frozen.json'
    c=json.loads((path if path.exists() else ROOT/'configs/base.json').read_text())
    ms=[build(c,n) for n in ['vanilla_loop','step_conditioned_loop','vector_field']]
    ns=[counts(m)['params'] for m in ms]; assert max(ns)/min(ns)-1 < .01
    assert min(ns)>=15000000 and max(ns)<=30000000
    for k,v in ms[0].state_dict().items():
        assert torch.equal(v,ms[1].state_dict()[k])
        key='core.core.'+k[5:] if k.startswith('core.') else k
        assert torch.equal(v,ms[2].state_dict()[key])

@pytest.mark.parametrize('solver',['euler','heun','rk4'])
def test_actual_neural_core_calls_at_matched_nfe(solver):
    c={'d_model':32,'n_heads':4,'d_ff':128,'dropout':.1,'core_blocks':1,'stem_blocks':1,'max_length':16,'vocab_size':100,'seed':0}
    m=build(c,'vector_field').eval(); calls=[]
    hook=m.core.core.blocks[0].register_forward_hook(lambda *args:calls.append(1))
    with torch.no_grad(): result=m(torch.tensor([[2,5,6,3]]),8,solver,return_hidden=True)
    hook.remove()
    assert len(calls)==result[-1]==8

def test_disjoint_data():
    paths=list((ROOT/'data/generated').glob('*.jsonl'))
    if not paths: pytest.skip('run generation first')
    import hashlib
    manifest_path=ROOT/'data/generated/manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    seen=set(); texts=set()
    for p in paths:
        count=0
        for line in p.open(encoding='utf-8'):
            row=json.loads(line)
            count+=1
            assert row['underlying_id'] not in seen; seen.add(row['underlying_id'])
            assert row['id'] not in texts; texts.add(row['id'])
            assert row['official_steps']==row['depth']+1
            assert set(row)=={'id','underlying_id','text','label','depth','official_steps'}
        if manifest and p.stem in manifest['splits']:
            assert count==manifest['splits'][p.stem]['count']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['splits'][p.stem]['sha256']

def test_harmful_recovery_accounting():
    from evaluate import metrics
    y=np.array([0,1,0,1]); ref=np.array([[4.,0.],[0.,4.],[0.,4.],[4.,0.]])
    z=np.array([[0.,4.],[0.,4.],[4.,0.],[4.,0.]])
    d=metrics(z,ref,y)
    assert d['harmful_flip_vs_8']==.25 and d['recovery_flip_vs_8']==.25 and d['flip_rate_vs_8']==.5

def test_structural_shortcut_uses_text_only_and_handles_case_and_query_negation():
    import sys
    sys.path.insert(0,str(ROOT/'data'))
    from audit_structural_shortcut import predict
    # The dummy opposite-rule antecedent appears once. No chain is executed.
    facts='Concept680s are attribute70. Every concept680 is a concept2027. Concept1348s are not attribute70. Stella is a concept680. '
    assert predict(facts+'True or false: Stella is attribute70.')==(1,True)
    assert predict(facts+'True or false: Stella is not attribute70.')==(0,True)
