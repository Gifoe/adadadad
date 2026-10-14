"""Read-only audit invariants plus an exact-byte development evidence archive."""
import argparse,collections,hashlib,json,math,pathlib,struct
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(name):return json.loads((OUT/name).read_text(encoding='utf8'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-raw-logits',action='store_true');args=parser.parse_args()
    data=(OUT/'intervention_results.jsonl').read_bytes();raw=data.splitlines(keepends=True);records=[json.loads(s) for s in raw]
    assert len({(r['case_id'],r['condition']) for r in records})==len(records)
    threshold=read('development_thresholds.json');dev_bytes=b''.join(s for s,r in zip(raw,records) if r['phase']=='dev')
    assert hashlib.sha256(dev_bytes).hexdigest()==threshold['development_results_sha256']
    (OUT/'development_records_frozen.jsonl').write_bytes(dev_bytes)
    assert hashlib.sha256((OUT/'sample_manifest.jsonl').read_bytes()).hexdigest()==threshold['manifest_sha256']
    assert hashlib.sha256((ROOT/'config.yaml').read_bytes()).hexdigest()==threshold['config_sha256']
    manifest=[json.loads(s) for s in (OUT/'sample_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    text_hashes={}
    for row in manifest:
        ids=row['input_ids'];assert len(ids)<=4096
        text_hashes[row['sample_id']]=hashlib.sha256(struct.pack('<'+'q'*len(ids),*ids)).hexdigest()
    cases=collections.defaultdict(set);phases=collections.Counter();parents=collections.defaultdict(set)
    for r in records:
        assert r['all_other_depths_frozen'] and math.isfinite(r['U'])
        assert r['text_sha256']==text_hashes[r['sample_id']]
        if args.require_raw_logits:assert (ROOT/r['anchor_logits_npz']).is_file(),r['anchor_logits_npz']
        cases[(r['phase'],r['case_id'])].add(r['condition']);phases[r['phase']]+=1;parents[r['phase']].add(r['parent_id'])
    for phase in ['dev','audit','rename','dev_aligned','audit_aligned','rename_aligned','depths_aligned']:
        plan=read('case_plan_'+phase+'.json');expected=sum(len(r['pairs']) for r in plan)
        chosen={r['case_id'] for r in records if r['phase']==phase and r['kind']=='dependent'}
        assert len(chosen)==expected,(phase,len(chosen),expected)
        for case in chosen:assert {'11','21','12','22','22_mask','22_shallow'}<=cases[(phase,case)]
        runtime=read('core_runtime_'+phase+'.json');assert runtime['weights_hash_before']==runtime['weights_hash_after']
        if phase.endswith('_aligned'):
            a=read('analysis_'+phase+'.json')
            assert a['source_K2_consumer_K1_max_abs_effect']==0
            assert max(abs(r.get('masked_shape_null',0)) for r in a['cases'])==0
    for name,n in [('baseline_pilot_records.jsonl',80),('baseline_length_isolated_records.jsonl',16)]:
        free=[json.loads(s) for s in (OUT/name).read_text(encoding='utf8').splitlines()];assert len(free)==n
        assert len({(r['method'],r['sample_id']) for r in free})==n
        for r in free:
            assert r['generation_policy_uses_reference'] is False
            if name.startswith('baseline_length_isolated'):assert r['weights_hash_before']==r['weights_hash_after']
    report={'status':'PASS','n_branch_records':len(records),'branches_by_phase':dict(phases),'parent_units_by_phase':{p:len(v) for p,v in parents.items()},
            'raw_logit_presence_required':args.require_raw_logits,
            'frozen_development_bytes_sha256':hashlib.sha256(dev_bytes).hexdigest(),
            'checks':['unique case/condition','all planned dependent cells and KV controls complete','finite utilities','manifest/config/development snapshot hashes',
                      'phase weights unchanged','strict aligned masked/source-depth causal nulls zero','free generation has no reference policy']}
    (OUT/'artifact_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
