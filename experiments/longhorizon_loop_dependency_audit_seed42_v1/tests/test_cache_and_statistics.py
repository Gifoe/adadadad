import json,pathlib,sys
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from native_backend import TaHCache,capture_cache_snapshot,restore_cache_snapshot,intervene_deep_kv,append_invisible_deep_slot
from core_audit import ancestors,plans,matched_nondependent
from analyze_results import estimate

def fixture_cache():
    c=TaHCache()
    for depth in [0,1]:
        c.current_iter_depth=depth;c.position_ids_to_cache=torch.tensor([[0,1,2]])
        c.valid_mask_to_cache=torch.ones((1,3),dtype=torch.bool)
        for layer in range(28):
            keys=torch.arange(12,dtype=torch.float32).reshape(1,1,3,4)+100*depth
            c.update(keys,keys+1,layer)
    return c

def test_real_cache_snapshot_mask_and_substitution_are_independent():
    c=fixture_cache();snapshot=capture_cache_snapshot(c);left=restore_cache_snapshot(snapshot);right=restore_cache_snapshot(snapshot)
    intervene_deep_kv(left,1,'mask');intervene_deep_kv(right,1,'shallow_substitute')
    for layer in range(28):
        assert left._tah_key_cache[layer][1].shape==snapshot._tah_key_cache[layer][1].shape
        assert torch.equal(left._tah_position_id_cache[layer][1],snapshot._tah_position_id_cache[layer][1])
        assert not left._tah_valid_mask_cache[layer][1][0,1]
        assert snapshot._tah_valid_mask_cache[layer][1].all() and c._tah_valid_mask_cache[layer][1].all()
        assert torch.equal(right._tah_key_cache[layer][1][...,1,:],snapshot._tah_key_cache[layer][0][...,1,:])
        assert not torch.equal(snapshot._tah_key_cache[layer][1][...,1,:],right._tah_key_cache[layer][1][...,1,:])

def test_single_chain_has_no_fabricated_nondependent_control():
    rows=[json.loads(s) for s in (ROOT/'outputs/sample_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    row=next(r for r in rows if r.get('primary') and r['H']==16);p=plans(row)[0]
    assert ancestors(row,15)==set(range(15))
    scores={i:{'ce_nats':1.} for i in range(len(row['input_ids']))}
    assert matched_nondependent(row,p,[1]*len(row['input_ids']),scores) is None

def test_K1_padding_adds_no_visible_or_nonzero_deep_state():
    c=fixture_cache();old=c._tah_key_cache[0][0].clone()
    for layer in range(28):
        for field in ['_tah_key_cache','_tah_value_cache']:
            d=getattr(c,field);d[layer][1]=d[layer][1][...,:2,:]
        for field in ['_tah_position_id_cache','_tah_valid_mask_cache']:
            d=getattr(c,field);d[layer][1]=d[layer][1][:,:2]
    append_invisible_deep_slot(c,2)
    assert torch.equal(c._tah_key_cache[0][0],old)
    for layer in range(28):
        assert c._tah_key_cache[layer][1].shape==old.shape
        assert not c._tah_key_cache[layer][1][...,-1,:].any()
        assert not c._tah_value_cache[layer][1][...,-1,:].any()
        assert c._tah_position_id_cache[layer][1].tolist()==[[0,1,2]]
        assert c._tah_valid_mask_cache[layer][1].tolist()==[[True,True,False]]

def test_problem_cluster_mean_does_not_count_pairs_as_independent():
    rows=[{'parent_id':'A','gain':1.}]*20+[{'parent_id':'B','gain':-1.}]
    v=estimate(rows,'gain');assert v['n_clusters']==2 and v['n_pairs']==21 and v['mean']==0
    assert v['ci95']==[-1.,1.]
