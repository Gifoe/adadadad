"""Create/verify the same score-independent four-prompt development plan."""
import json,pathlib
def ensure_plan(root):
    root=pathlib.Path(root);rows=[json.loads(s) for s in (root/'outputs/sample_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    ids=json.loads((root/'outputs/pilot_sample_ids.json').read_text(encoding='utf8'));pilot=[r for r in rows if r['sample_id'] in ids]
    nest=[r for r in pilot if r['dataset']=='nestful_v2_math']
    selected=[min(nest,key=lambda r:(r['N'],r['sample_id'])),max(nest,key=lambda r:(r['N'],r['sample_id']))]
    selected += [next(r for r in pilot if r.get('parent_id')=='controlled_dev_000' and r['H']==h) for h in [4,16]]
    plan={'selected_ids':[r['sample_id'] for r in selected],'max_output_tokens':2048,'split':'development_only',
          'selection_rule':'shortest/longest N among pilot NESTFUL, then first controlled group H4/H16; independent of scores',
          'role':'output-length sensitivity, not the primary baseline or formal audit'}
    p=root/'outputs/length_probe_plan.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf8'))==plan
    else:p.write_text(json.dumps(plan,indent=2),encoding='utf8')
    return plan
