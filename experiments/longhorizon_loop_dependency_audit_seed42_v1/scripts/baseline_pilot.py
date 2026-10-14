"""Twenty preselected development prompts, four actual frozen-weight policies."""
import collections,gc,importlib.util,json,pathlib,re,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from native_backend import NativeBackend,weight_hash,torch
from prepare_data import execute,typed_equal

def json_arrays(text):
    decoder=json.JSONDecoder();values=[]
    for match in re.finditer(r'\[',text):
        try:
            value,_=decoder.raw_decode(text[match.start():])
            if isinstance(value,list):values.append(value)
        except (json.JSONDecodeError,ValueError):pass
    return values

def score(text,row,module):
    arrays=json_arrays(text);kind=row['dataset']
    if kind=='nestful_v2_math':
        candidates=[a for a in arrays if a and all(isinstance(v,dict) and {'name','label','arguments'}<=v.keys() for v in a)]
        if not candidates:return {'format_success':False,'execution_success':False,'full_task_success':False,'correct_intermediate_rate':None,'first_error_position':0,'error':'no_structured_call_array'}
        calls=candidates[-1];ex=execute(calls,row['tools'],module)
        matches=[typed_equal(a,b) for a,b in zip(ex.get('intermediate',[]),row['executed_intermediates'])]
        first=next((i for i,x in enumerate(matches) if not x),len(matches))
        return {'format_success':True,'execution_success':ex['ok'],'full_task_success':ex['ok'] and typed_equal(ex['value'],row['gold_answer']),
                'correct_intermediate_rate':sum(matches)/max(len(row['executed_intermediates']),1),'first_error_position':first,
                'predicted_call_count':len(calls),'returned_value':ex.get('value'),'error':ex.get('error')}
    candidates=[a for a in arrays if len(a)==16 and all(isinstance(v,str) and len(v)==1 and v in row['alphabet'] for v in a)]
    if not candidates:return {'format_success':False,'execution_success':False,'full_task_success':False,'correct_intermediate_rate':None,'first_error_position':0,'error':'no_16_state_array'}
    states=candidates[-1];matches=[a==b for a,b in zip(states,row['gold_answer'])]
    return {'format_success':True,'execution_success':True,'full_task_success':all(matches),'correct_intermediate_rate':sum(matches)/16,
            'first_error_position':next((i for i,x in enumerate(matches) if not x),16),'predicted_states':states}

def main():
    rows=[json.loads(x) for x in (ROOT/'outputs/sample_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    nest=[r for r in rows if r['split']=='dev' and r['dataset']=='nestful_v2_math']
    # Source-independent ordering, ten NESTFUL plus paired H4/H16 for five groups.
    selected=nest[:10]+[r for r in rows if r['split']=='dev' and r['dataset']!='nestful_v2_math' and not r['renamed'] and r['H'] in [4,16] and int(r['parent_id'].split('_')[-1])<5]
    assert len(selected)==20
    lock=ROOT/'outputs/pilot_sample_ids.json';content=[r['sample_id'] for r in selected]
    if lock.exists():assert json.loads(lock.read_text())==content
    else:lock.write_text(json.dumps(content,indent=2),encoding='utf8')
    source=ROOT/'upstream/NESTFUL/data_v2/executable_functions/basic_functions.py';spec=importlib.util.spec_from_file_location('pilot_official_math',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=ROOT/'outputs/baseline_pilot_records.jsonl';existing={}
    if path.exists():existing={(r['method'],r['sample_id']):r for r in map(json.loads,path.read_text(encoding='utf8').splitlines())}
    provenance=[]
    for standard,methods in [(True,['standard']),(False,['forced1','forced2','adaptive'])]:
        b=NativeBackend(standard=standard);initial=weight_hash(b.model)
        for method in methods:
            for row in selected:
                key=(method,row['sample_id'])
                if key in existing:continue
                generated=b.generate(row['prefix_text'],mode=method,max_tokens=512)
                result={'sample_id':row['sample_id'],'parent_id':row['parent_id'],'dataset':row['dataset'],'H':row['H'],'method':method,
                        'model_weights':'independently_trained_Standard' if standard else 'same_max2_checkpoint',
                        'backend':'official_native_SDPA; minisgl_equivalence_separate','selection_split':'development_only',
                        **generated,**score(generated['text'],row,module)}
                with path.open('a',encoding='utf8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                existing[key]=result
                print('PILOT_SAMPLE',json.dumps({k:result[k] for k in ['sample_id','method','H','format_success','full_task_success','generated_tokens','actual_iterations','seconds']}),flush=True)
        after=weight_hash(b.model);assert initial==after
        provenance.append({'model':'Standard' if standard else 'max2','parameters':b.parameters,'weights_hash_before':initial,'weights_hash_after':after,'actual_gpu_forward_seconds':b.gpu_seconds})
        del b;gc.collect();torch.cuda.empty_cache()
    summaries=[]
    for method in ['standard','forced1','forced2','adaptive']:
        values=[r for (m,_),r in existing.items() if m==method]
        for kind in ['all','nestful_v2_math','controlled_symbol_chains_new_diagnostic']:
            chosen=values if kind=='all' else [v for v in values if v['dataset']==kind]
            if not chosen:continue
            summaries.append({'method':method,'dataset':kind,'n':len(chosen),'format_success_rate':sum(r['format_success'] for r in chosen)/len(chosen),
                              'full_task_success_rate':sum(r['full_task_success'] for r in chosen)/len(chosen),'execution_success_rate':sum(r['execution_success'] for r in chosen)/len(chosen),
                              'mean_actual_depth':sum(r['actual_iterations'] for r in chosen)/sum(r['input_tokens']+len(r['generated_position_depths']) for r in chosen),
                              'generated_tokens':sum(r['generated_tokens'] for r in chosen),'seconds':sum(r['seconds'] for r in chosen),
                              'cuda_forward_seconds':sum(r['cuda_forward_seconds'] for r in chosen),'peak_cuda_allocated_mib':max(r['peak_cuda_allocated_mib'] for r in chosen),
                              'generation_limit_fraction':sum(r['hit_generation_limit'] for r in chosen)/len(chosen)})
    report={'phase':'20_prompt_development_pilot','summaries':summaries,'weight_provenance':provenance,
            'not_matched_weights':'Standard differs from max2; forced1/forced2/adaptive share exact max2 weights',
            'official_serving_engine_equivalence_verified':json.loads((ROOT/'outputs/preflight_native.json').read_text())['official_serving_equivalence_verified'],
            'scope':'native-backend one-shot structured generation; not a deployed Agent or a verified minisgl benchmark'}
    (ROOT/'outputs/baseline_results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (ROOT/'outputs/runtime_profile.json').write_text(json.dumps({'pilot':report,'soft_gpu_budget_hours':8,'cuda_seconds_so_far':sum(x['actual_gpu_forward_seconds'] for x in provenance)},indent=2),encoding='utf8')
    print('BASELINE_PILOT_COMPLETE',json.dumps(summaries),flush=True)
if __name__=='__main__':main()
