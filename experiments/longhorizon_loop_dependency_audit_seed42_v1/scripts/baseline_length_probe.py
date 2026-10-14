"""Separate development-only 2048-token sensitivity; never replaces the pilot."""
import gc,importlib.util,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from native_backend import NativeBackend,weight_hash,torch
from baseline_pilot import score

def main():
    rows=[json.loads(x) for x in (ROOT/'outputs/sample_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    ids=json.loads((ROOT/'outputs/pilot_sample_ids.json').read_text(encoding='utf8'))
    pilot=[r for r in rows if r['sample_id'] in ids];nest=[r for r in pilot if r['dataset']=='nestful_v2_math']
    selected=[min(nest,key=lambda r:(r['N'],r['sample_id'])),max(nest,key=lambda r:(r['N'],r['sample_id']))]
    selected += [next(r for r in pilot if r.get('parent_id')=='controlled_dev_000' and r['H']==h) for h in [4,16]]
    assert len({r['sample_id'] for r in selected})==4
    lock={'selected_ids':[r['sample_id'] for r in selected],'max_output_tokens':2048,'split':'development_only',
          'selection_rule':'shortest/longest N among pilot NESTFUL, then first controlled group H4/H16; independent of scores',
          'role':'output-length sensitivity, not the primary baseline or formal audit'}
    p=ROOT/'outputs/length_probe_plan.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf8'))==lock
    else:p.write_text(json.dumps(lock,indent=2),encoding='utf8')
    src=ROOT/'upstream/NESTFUL/data_v2/executable_functions/basic_functions.py';spec=importlib.util.spec_from_file_location('length_probe_math',src)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=ROOT/'outputs/baseline_length_probe_records.jsonl';done={}
    if path.exists():done={(r['method'],r['sample_id']):r for r in map(json.loads,path.read_text(encoding='utf8').splitlines())}
    seconds=0;provenance=[]
    for standard,methods in [(True,['standard']),(False,['forced1','forced2','adaptive'])]:
        b=NativeBackend(standard=standard);before=weight_hash(b.model)
        for mode in methods:
            for row in selected:
                if (mode,row['sample_id']) in done:continue
                result={'method':mode,'sample_id':row['sample_id'],'parent_id':row['parent_id'],'dataset':row['dataset'],'N':row['N'],'H':row['H'],
                        'role':'development_output_length_sensitivity','max_output_tokens':2048,
                        **b.generate(row['prefix_text'],mode=mode,max_tokens=2048)}
                result.update(score(result['text'],row,module))
                with path.open('a',encoding='utf8') as f:f.write(json.dumps(result)+'\n');f.flush()
                done[(mode,row['sample_id'])]=result
                print('LENGTH_PROBE',json.dumps({k:result[k] for k in ['method','sample_id','format_success','full_task_success','generated_tokens','seconds']}),flush=True)
        after=weight_hash(b.model);assert before==after
        provenance.append({'model':'Standard' if standard else 'max2','weights_hash_before':before,'weights_hash_after':after})
        seconds+=b.gpu_seconds;del b;gc.collect();torch.cuda.empty_cache()
    summary=[]
    for mode in ['standard','forced1','forced2','adaptive']:
        vals=[v for (m,_),v in done.items() if m==mode]
        summary.append({'method':mode,'n':len(vals),'format_successes':sum(v['format_success'] for v in vals),
                        'full_task_successes':sum(v['full_task_success'] for v in vals),'generation_limit_hits':sum(v['hit_generation_limit'] for v in vals),
                        'generated_tokens':sum(v['generated_tokens'] for v in vals),'seconds':sum(v['seconds'] for v in vals)})
    recorded_seconds=sum(v['cuda_forward_seconds'] for v in done.values())
    report={'plan':lock,'summaries':summary,'weight_provenance':provenance,'actual_gpu_forward_seconds_this_run':seconds,
            'recorded_prompt_GPU_seconds_all_runs':recorded_seconds,'resume_accounting':'completed prompt timings retained across interruptions; unrecorded interrupted work remains outside this lower bound'}
    (ROOT/'outputs/baseline_length_probe.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    p=ROOT/'outputs/runtime_profile.json';profile=json.loads(p.read_text(encoding='utf8'));prior=profile.get('length_probe',{}).get('recorded_prompt_GPU_seconds_all_runs',0)
    profile['length_probe']=report;profile['cuda_seconds_so_far']+=recorded_seconds-prior;p.write_text(json.dumps(profile,indent=2),encoding='utf8')
    print('LENGTH_PROBE_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
