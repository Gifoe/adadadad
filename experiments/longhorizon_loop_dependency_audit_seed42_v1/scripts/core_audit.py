"""Fixed-text, cached-prefix 2x2 and semantic/mechanistic controls; resume by case."""
import argparse,collections,datetime,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from native_backend import NativeBackend,capture_cache_snapshot,restore_cache_snapshot,intervene_deep_kv,append_invisible_deep_slot,cache_depth_stats,weight_hash,torch

def ancestors(row,target):
    parents=collections.defaultdict(set)
    for e in row['dag_edges']:parents[e['consumer']].add(e['producer'])
    seen=set();pending=list(parents[target])
    while pending:
        value=pending.pop()
        if value not in seen:seen.add(value);pending.extend(parents[value])
    return seen

def plans(row):
    units={u['index']:u for u in row['units']};pairs=[]
    if row['dataset']=='nestful_v2_math':
        for unit in row['units']:
            for use in unit['uses']:
                source=units[use['producer']];i=source['source_anchor'];j=use['consumer_anchor']
                if not i<j:raise ValueError('Invalid semantic anchor ordering')
                pairs.append({'producer_node':source['index'],'consumer_node':unit['index'],'i':i,'j':j,
                              'source_unit_span':source['unit_token_span'],'consumer_span':use['consumer_token_span'],
                              'relation':'direct_reference_edge','distance_tokens':j-i,'source_anchor_token':source['predicted_anchor_token'],
                              'consumer_anchor_token':use['predicted_anchor_token']})
        pairs=sorted(pairs,key=lambda r:(r['distance_tokens'],r['producer_node'],r['consumer_node']))
        chosen=[pairs[0]]
        if len(pairs)>1 and (pairs[-1]['i'],pairs[-1]['j'])!=(pairs[0]['i'],pairs[0]['j']):chosen.append(pairs[-1])
        return chosen
    source=units[16-row['H']];consumer=units[15]
    return [{'producer_node':source['index'],'consumer_node':15,'i':source['source_anchor'],'j':consumer['source_anchor'],
             'source_unit_span':source['unit_token_span'],'consumer_span':consumer['source_token_span'],
             'relation':'transitive_ancestor_path','path_length':row['H']-1,'distance_tokens':consumer['source_anchor']-source['source_anchor'],
             'source_anchor_token':source['predicted_anchor_token'],'consumer_anchor_token':consumer['predicted_anchor_token']}]

def matched_nondependent(row,pair,policy,base_scores):
    forbidden=ancestors(row,pair['consumer_node']);target=pair['consumer_node'];candidates=[]
    for unit in row['units']:
        i=unit['source_anchor']
        if unit['index'] in forbidden or unit['index']==target or i>=pair['j']:continue
        if policy[i]!=policy[pair['i']]:continue # equal absolute loop budget per2x2cell
        distance=pair['j']-i;ratio=distance/max(pair['distance_tokens'],1)
        if not .5<=ratio<=2:continue
        difficulty=abs(base_scores[i]['ce_nats']-base_scores[pair['i']]['ce_nats'])
        candidates.append((abs(ratio-1)+difficulty,unit,distance,difficulty))
    if not candidates:return None
    _,unit,distance,difficulty=min(candidates,key=lambda x:(x[0],x[1]['index']))
    return {**pair,'producer_node':unit['index'],'i':unit['source_anchor'],'source_unit_span':unit['unit_token_span'],
            'matched_dependent_i':pair['i'],
            'relation':'nondependent_no_DAG_path','distance_tokens':distance,'matching_abs_local_native_ce_difference':difficulty,
            'matching_distance_ratio':distance/max(pair['distance_tokens'],1),'source_anchor_token':unit['predicted_anchor_token']}

def mean_ce(scores,positions):return float(np.mean([scores[pos]['ce_nats'] for pos in positions]))
def mean_correct(scores,positions):return float(np.mean([scores[pos]['correct'] for pos in positions]))
def write_json(path,content):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(content,indent=2),encoding='utf8')

def run_pair(backend,row,pair,policy,base_scores,kind,done,stream,phase):
    ids=row['input_ids'];i=pair['i'];j=pair['j'];source_positions=list(range(pair['source_unit_span'][0]-1,pair['source_unit_span'][1]-1))
    consumer_positions=list(range(pair['consumer_span'][0]-1,pair['consumer_span'][1]-1));stop=max(max(source_positions),max(consumer_positions),j)+1
    assert i<j and stop<=len(ids)
    identity=f'{row["sample_id"]}__{pair["producer_node"]}_{pair["consumer_node"]}__i{i}_j{j}__{kind}'
    if kind=='nondependent':identity+=f'__matched_i{pair["matched_dependent_i"]}'
    aligned=phase.endswith('_aligned')
    if aligned:identity+='__aligned'
    conditions=[('11',1,1,None),('21',2,1,None),('12',1,2,None),('22',2,2,None)]
    if kind=='dependent':conditions += [('22_mask',2,2,'mask'),('22_shallow',2,2,'shallow_substitute')]
    # Preregistered development-only repeat, same cache shapes and token IDs.
    if kind=='dependent' and phase.startswith('dev') and row['sample_id'] in REPEAT_SAMPLE_IDS:
        conditions += [('22_repeat',2,2,None)]
    pending=[c for c in conditions if (identity,c[0]) not in done]
    if not pending:return
    # Prefix computation is shared and immutable, then deeply cloned at the
    # source. Different branches always replay exactly the same token IDs.
    prefix=backend.forward_segment(ids[:i],depths=policy[:i],full_ids=ids,interest=())
    assert prefix['depths']==policy[:i]
    snapshot=capture_cache_snapshot(prefix['cache']);prefix_seconds=prefix['seconds'];del prefix
    for name,ki,kj,ablation in pending:
        cached=restore_cache_snapshot(snapshot);begin=time.perf_counter();before=backend.gpu_seconds
        source=backend.forward_segment([ids[i]],start=i,depths=[ki],cache=cached,full_ids=ids,interest=[i],anchors=[i])
        if aligned and ki==1:append_invisible_deep_slot(source['cache'],i)
        changed=None
        if ablation:changed=intervene_deep_kv(source['cache'],i,ablation)
        source_cache_stats=cache_depth_stats(source['cache'])
        requested=policy[i+1:stop].copy();requested[j-i-1]=kj
        interested=set(source_positions+consumer_positions+[j]);interested={p for p in interested if p>=i+1}
        suffix=backend.forward_segment(ids[i+1:stop],start=i+1,depths=requested,cache=source['cache'],full_ids=ids,interest=interested,anchors=[j])
        scores={p:base_scores[p] for p in source_positions if p<i};scores.update(source['scores']);scores.update(suffix['scores'])
        assert source['depths']==[ki] and suffix['depths']==requested
        raw={**source['anchor_logits'],**suffix['anchor_logits']};directory=ROOT/'outputs/anchor_logits';directory.mkdir(exist_ok=True)
        filename=identity+'__'+name+'.npz';np.savez_compressed(directory/filename,**raw)
        record={'phase':phase,'case_id':identity,'sample_id':row['sample_id'],'parent_id':row['parent_id'],'dataset':row['dataset'],'H':row['H'],'N':row['N'],
                'renamed':row.get('renamed',False),'kind':kind,'condition':name,'source_depth':ki,'consumer_depth':kj,**pair,
                'U':-scores[j]['ce_nats'],'consumer_next_token_correct':scores[j]['correct'],
                'U_semantic_span':-mean_ce(scores,consumer_positions),'consumer_semantic_token_accuracy':mean_correct(scores,consumer_positions),
                'source_next_ce_nats':scores[i]['ce_nats'],'source_semantic_unit_ce_nats':mean_ce(scores,source_positions),
                'source_semantic_unit_accuracy':mean_correct(scores,source_positions),
                'native_source_depth':policy[i],'native_consumer_depth':policy[j],
                'native_source_ce_nats':base_scores[i]['ce_nats'],'prefix_cache_seconds_shared':prefix_seconds,
                'branch_cuda_forward_seconds':backend.gpu_seconds-before,'branch_wall_seconds':time.perf_counter()-begin,
                'executed_branch_iterations':source['executed_iterations']+suffix['executed_iterations'],
                'logical_full_prefix_iterations':sum(policy[:i])+source['executed_iterations']+suffix['executed_iterations'],
                'flop_estimate_parameter_matmuls_only':2*backend.parameters*(source['executed_iterations']+suffix['executed_iterations']),
                'flop_estimate_limit':'excludes attention-score/context-length terms, cache copies and instrumentation; not exact hardware FLOPs',
                'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1048576,'ablation':changed,
                'all_other_depths_frozen':True,'text_sha256':hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),
                'cache_shape_aligned_across_source_depths':aligned,'invisible_padding_executed_extra_loops':0,
                'source_cache_depth_counts':source_cache_stats,'final_cache_depth_counts':cache_depth_stats(suffix['cache']),
                'anchor_logits_npz':'outputs/anchor_logits/'+filename,'per_loop_target_and_top8_logits':source['loop_trace']+suffix['loop_trace'],
                'source_stop_scores':source['stop_scores'].get(i),'consumer_stop_scores':suffix['stop_scores'].get(j),
                'backend_scope':'official native SDPA; external serving-equivalence status remains unverified unless its separate check passed'}
        stream.write(json.dumps(record)+'\n');stream.flush();done.add((identity,name))
        print('CASE_BRANCH',identity,name,'U',record['U'],'seconds',record['branch_wall_seconds'],flush=True)
        del suffix,source,cached,raw
    del snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','audit','rename','dev_aligned','audit_aligned','rename_aligned','depths_aligned'],required=True);p.add_argument('--max-pairs',type=int);a=p.parse_args()
    started=time.perf_counter();base_phase=a.phase.removesuffix('_aligned')
    manifest=ROOT/'outputs/sample_manifest.jsonl';rows=[json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
    global REPEAT_SAMPLE_IDS
    REPEAT_SAMPLE_IDS={next(r['sample_id'] for r in rows if r['split']=='dev' and r['dataset']=='nestful_v2_math')}
    REPEAT_SAMPLE_IDS.update(next(r['sample_id'] for r in rows if r['split']=='dev' and r.get('primary') and r['H']==h) for h in [4,16])
    if base_phase=='dev':rows=[r for r in rows if r['split']=='dev' and (r['dataset']=='nestful_v2_math' or r.get('primary'))]
    elif base_phase=='audit':
        assert (ROOT/'outputs/development_thresholds.json').exists(),'Development thresholds must be frozen first'
        rows=[r for r in rows if r['split']=='audit' and (r['dataset']=='nestful_v2_math' or r.get('primary'))]
    elif base_phase=='depths':
        assert (ROOT/'outputs/development_thresholds.json').exists()
        rows=[r for r in rows if r['split']=='audit' and r['dataset']=='controlled_symbol_chains_new_diagnostic' and not r['renamed'] and r['H'] in [2,8]]
    else:
        assert (ROOT/'outputs/development_thresholds.json').exists()
        rows=[r for r in rows if r['split']=='audit' and r.get('renamed') and r['H'] in [4,16] and int(r['parent_id'].split('_')[-1])<16]
    # Lock every core pair before any audit score is read.
    planned=[{'sample_id':r['sample_id'],'parent_id':r['parent_id'],'pairs':plans(r)} for r in rows]
    plan_path=ROOT/'outputs'/f'case_plan_{a.phase}.json';content=json.dumps(planned,indent=2)
    if plan_path.exists():assert plan_path.read_text(encoding='utf8')==content
    else:plan_path.write_text(content,encoding='utf8')
    path=ROOT/'outputs/intervention_results.jsonl';done=set()
    if path.exists():done={(r['case_id'],r['condition']) for r in map(json.loads,path.read_text(encoding='utf8').splitlines())}
    b=NativeBackend();initial_hash=weight_hash(b.model);finished_pairs=0;missing=[]
    budget_before=json.loads((ROOT/'outputs/runtime_profile.json').read_text(encoding='utf8')).get('cuda_seconds_so_far',0) if (ROOT/'outputs/runtime_profile.json').exists() else 0
    with path.open('a',encoding='utf8') as stream:
        for row in rows:
            if budget_before+b.gpu_seconds>8*3600:
                missing.append({'sample_id':row['sample_id'],'reason':'soft8h_budget_reached_before_policy'});break
            policy_path=ROOT/'outputs/policies'/(row['sample_id']+'.json')
            if policy_path.exists():
                saved=json.loads(policy_path.read_text(encoding='utf8'));assert saved['token_sha256']==hashlib.sha256(np.array(row['input_ids'],dtype=np.int64).tobytes()).hexdigest()
                policy=saved['depths'];base_scores={int(k):v for k,v in saved['native_scores'].items()}
            else:
                policy,base_scores,stops=b.native_policy(row['input_ids']);write_json(policy_path,{'depths':policy,'native_scores':base_scores,'stop_scores':stops,
                    'token_sha256':hashlib.sha256(np.array(row['input_ids'],dtype=np.int64).tobytes()).hexdigest(),'policy':'exact original native adaptive forward, recorded once and frozen'})
            for pair in plans(row):
                if budget_before+b.gpu_seconds>8*3600:missing.append({'sample_id':row['sample_id'],'reason':'soft8h_budget_reached'});break
                run_pair(b,row,pair,policy,base_scores,'dependent',done,stream,a.phase)
                nond=matched_nondependent(row,pair,policy,base_scores)
                if nond:run_pair(b,row,nond,policy,base_scores,'nondependent',done,stream,a.phase)
                else:missing.append({'sample_id':row['sample_id'],'i':pair['i'],'j':pair['j'],'reason':'no_same_role_same_native_depth_nondependent_within_half_to_double_distance'})
                finished_pairs+=1
                if a.max_pairs and finished_pairs>=a.max_pairs:break
            if a.max_pairs and finished_pairs>=a.max_pairs:break
    final_hash=weight_hash(b.model);assert final_hash==initial_hash
    report={'phase':a.phase,'pair_evaluations_visited':finished_pairs,'forward_calls':b.calls,'cuda_forward_wall_seconds':b.gpu_seconds,
            'phase_end_to_end_seconds_including_load_and_file_writes':time.perf_counter()-started,
            'weights_hash_before':initial_hash,'weights_hash_after':final_hash,'unavailable_controls_or_budget_exclusions':missing,
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'baseline_gpu_seconds_prior':budget_before,
            'actual_backend':'official native SDPA; no optimizer/backward/parameter changes'}
    write_json(ROOT/'outputs'/f'core_runtime_{a.phase}.json',report)
    profile_path=ROOT/'outputs/runtime_profile.json';profile=json.loads(profile_path.read_text(encoding='utf8')) if profile_path.exists() else {}
    profile[a.phase]=report;profile['cuda_seconds_so_far']=budget_before+b.gpu_seconds;write_json(profile_path,profile)
    print('CORE_PHASE_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
