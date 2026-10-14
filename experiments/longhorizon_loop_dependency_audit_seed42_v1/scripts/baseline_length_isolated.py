"""One model/prompt per process; validate weights before durable result export."""
import argparse,importlib.util,json,os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from native_backend import NativeBackend,weight_hash
from baseline_pilot import score

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['standard','forced1','forced2','adaptive'],required=True);p.add_argument('--sample-id',required=True);a=p.parse_args()
    plan=json.loads((ROOT/'outputs/length_probe_plan.json').read_text(encoding='utf8'));assert a.sample_id in plan['selected_ids']
    path=ROOT/'outputs/baseline_length_isolated_records.jsonl';done={}
    if path.exists():done={(r['method'],r['sample_id']):r for r in map(json.loads,path.read_text(encoding='utf8').splitlines())}
    key=(a.method,a.sample_id)
    if key not in done:
        row=next(json.loads(s) for s in (ROOT/'outputs/sample_manifest.jsonl').read_text(encoding='utf8').splitlines() if json.loads(s)['sample_id']==a.sample_id)
        src=ROOT/'upstream/NESTFUL/data_v2/executable_functions/basic_functions.py';spec=importlib.util.spec_from_file_location('isolated_math',src)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        b=NativeBackend(standard=a.method=='standard');before=weight_hash(b.model)
        generated=b.generate(row['prefix_text'],mode=a.method,max_tokens=2048)
        after=weight_hash(b.model);assert before==after
        result={'method':a.method,'sample_id':a.sample_id,'parent_id':row['parent_id'],'dataset':row['dataset'],'N':row['N'],'H':row['H'],
                'role':'development_output_length_sensitivity','max_output_tokens':2048,'worker':'one_prompt_per_process',
                'weights_hash_before':before,'weights_hash_after':after,'attention_backend':'unchanged official native SDPA dispatch',
                'process_affinity_control':'inherited from foreground runner when applied; see cpu_affinity_control.json',
                **generated,**score(generated['text'],row,module)}
        with path.open('a',encoding='utf8') as f:f.write(json.dumps(result)+'\n');f.flush();os.fsync(f.fileno())
        done[key]=result;print('ISOLATED_VALIDATED',json.dumps({k:result[k] for k in ['method','sample_id','generated_tokens','format_success','full_task_success']}),flush=True)
    if len(done)==16:
        summaries=[]
        for mode in ['standard','forced1','forced2','adaptive']:
            vals=[v for (m,_),v in done.items() if m==mode];assert len(vals)==4
            summaries.append({'method':mode,'n':4,'format_successes':sum(v['format_success'] for v in vals),'full_task_successes':sum(v['full_task_success'] for v in vals),
                              'generation_limit_hits':sum(v['hit_generation_limit'] for v in vals),'generated_tokens':sum(v['generated_tokens'] for v in vals),'seconds':sum(v['seconds'] for v in vals)})
        seconds=sum(v['cuda_forward_seconds'] for v in done.values());legacy=ROOT/'outputs/baseline_length_probe_records.jsonl'
        legacy_seconds=sum(json.loads(s)['cuda_forward_seconds'] for s in legacy.read_text(encoding='utf8').splitlines()) if legacy.exists() else 0
        report={'plan':plan,'summaries':summaries,'worker':'one_prompt_per_process; each successful prompt has equal before/after parameter hashes',
                'recorded_prompt_GPU_seconds_all_runs':seconds+legacy_seconds,'isolated_validated_prompt_GPU_seconds':seconds,
                'interrupted_legacy_completed_prompt_GPU_seconds':legacy_seconds,'legacy_results_used_for_accuracy':False,
                'native_crash':'multi-prompt runs exited 0xC0000005 without Python traceback; root cause unresolved, partial results retained separately',
                'shutdown':'after durable export, os._exit avoids interpreter/allocator teardown; no backend, precision, mixture or model alteration'}
        (ROOT/'outputs/baseline_length_probe.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        q=ROOT/'outputs/runtime_profile.json';profile=json.loads(q.read_text(encoding='utf8'));prior=profile.get('length_probe',{}).get('recorded_prompt_GPU_seconds_all_runs',0)
        profile['cuda_seconds_so_far']+=seconds+legacy_seconds-prior;profile['length_probe']=report;q.write_text(json.dumps(profile,indent=2),encoding='utf8')
    sys.stdout.flush();sys.stderr.flush()
    # All results and both hashes are flushed first. The worker owns its CUDA
    # context, which the OS releases without running Python object destructors.
    os._exit(0)
if __name__=='__main__':main()
