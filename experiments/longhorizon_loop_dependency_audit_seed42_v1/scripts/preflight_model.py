import datetime,gc,importlib.metadata,json,pathlib,platform,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from native_backend import *

def main():
    for filename in ['preflight_native.json','official_engine_example.json']:
        p=ROOT/'outputs'/filename;backup=p.with_name(p.stem+'_initial.json')
        if p.exists() and not backup.exists():backup.write_bytes(p.read_bytes())
    env={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'python':sys.version,
         'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','accelerate','numpy','safetensors','huggingface-hub']},
         'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'device_capability':torch.cuda.get_device_capability(),
         'free_gpu_bytes':torch.cuda.mem_get_info()[0],'host_limitation':'Windows; no installed WSL or Docker; Linux minisgl/Triton stack unavailable'}
    (ROOT/'outputs/environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
    try:
        from tah2.minisgl.core import SamplingParams
        from tah2.minisgl.llm import LLM
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/TaH2-1.7B-max2',local_files_only=True)
        prompt=tokenizer.apply_chat_template([{'role':'user','content':'What is 1+1? Give only the answer.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        llm=LLM(model_path=str(ROOT/'models/TaH2-1.7B-max2'))
        try:official={'status':'executed','result':llm.generate([prompt],SamplingParams(max_tokens=64,temperature=0.0))}
        finally:llm.shutdown()
    except Exception as error:
        official={'status':'blocked','exception':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
                  'equivalence_verified':False,'scope_limit':'Native SDPA mechanism results are conditional; no full minisgl reproduction claim'}
    (ROOT/'outputs/official_engine_example.json').write_text(json.dumps(official,indent=2,default=str),encoding='utf8')
    print('OFFICIAL_ENGINE_STATUS',official['status'],official.get('message',''),flush=True)
    b=NativeBackend();hash_before=weight_hash(b.model)
    text='States A B C D. Copy B then C.';ids=b.tokenizer(text,add_special_tokens=False)['input_ids'];x=torch.tensor([ids],device='cuda')
    checks={};torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        wrapper_process=b.model._process_sparse_iteration
        b.model._process_sparse_iteration=b.original_process
        original=b.model(input_ids=x,use_cache=True,labels=None,return_eval_metadata=True)
        b.model._process_sparse_iteration=wrapper_process
        wrapped=b.forward_segment(ids,instrument=False)
        checks['no_intervention_native_wrapper_max_logit_difference']=float((original.logits-wrapped['output'].logits).abs().max())
        assert torch.equal(original.logits,wrapped['output'].logits)
        repeated=b.forward_segment(ids,instrument=False);assert torch.equal(wrapped['output'].logits,repeated['output'].logits)
        checks['repeated_native_bitwise_equal']=True
        max_sequential=0.
        for depth in [1,2]:
            full=b.forward_segment(ids,depths=[depth]*len(ids),instrument=False);cache=None;parts=[]
            for pos,token in enumerate(ids):
                step=b.forward_segment([token],pos,[depth],cache,instrument=False);cache=step['cache'];parts.append(step['output'].logits)
            sequential=torch.cat(parts,dim=1);diff=float((full['output'].logits.float()-sequential.float()).abs().max());max_sequential=max(max_sequential,diff)
            checks[f'full_vs_incremental_depth{depth}_max_logprob_difference']=diff
            targets=x[0,1:]
            full_ce=F.cross_entropy(full['output'].logits[0,:-1].float(),targets,reduction='none')
            step_ce=F.cross_entropy(sequential[0,:-1].float(),targets,reduction='none')
            checks[f'full_vs_incremental_depth{depth}_target_CE_max_abs_difference_nats']=float((full_ce-step_ce).abs().max())
            checks[f'full_vs_incremental_depth{depth}_top1_agreement_fraction']=float((full['output'].logits.argmax(-1)==sequential.argmax(-1)).float().mean())
            assert torch.allclose(full['output'].logits.float(),sequential.float(),atol=.25,rtol=.02),'BF16 causal full/incremental mismatch beyond declared tolerance'
        one=b.forward_segment(ids,depths=[1]*len(ids),instrument=False)
        forced=[1]*len(ids);forced[3]=2;two=b.forward_segment(ids,depths=forced,instrument=False)
        earlier_diff=float((one['output'].logits[0,:3]-two['output'].logits[0,:3]).abs().max());later_shallow_diff=float((one['output'].logits[0,4:]-two['output'].logits[0,4:]).abs().max())
        assert earlier_diff==0 and later_shallow_diff==0
        checks['future_deep_KV_cannot_change_earlier_tokens_max_difference']=earlier_diff
        checks['deep_source_cannot_change_depth1_consumer_max_difference']=later_shallow_diff
        # Earlier K2 queries must also be insulated from later K2 content;
        # changing only future depth while earlier queries are K1 is weaker.
        future_at=min(6,len(ids)-1);changed_ids=ids.copy();changed_ids[future_at]=b.tokenizer.encode('different',add_special_tokens=False)[0]
        original2=b.forward_segment(ids,depths=[2]*len(ids),instrument=False)
        changed2=b.forward_segment(changed_ids,depths=[2]*len(ids),instrument=False)
        future_content_diff=float((original2['output'].logits[0,:future_at]-changed2['output'].logits[0,:future_at]).abs().max())
        assert future_content_diff==0,'Future content affected an earlier depth2 output'
        checks['future_K2_content_cannot_change_earlier_K2_outputs_max_difference']=future_content_diff
        prefix=b.forward_segment(ids[:6],depths=[2]*6,instrument=False);snapshot=capture_cache_snapshot(prefix['cache']);branch=restore_cache_snapshot(snapshot)
        old_key=snapshot._tah_key_cache[0][1].clone();old_mask=snapshot._tah_valid_mask_cache[0][1].clone()
        intervene_deep_kv(branch,3,'mask');branch._tah_key_cache[0][1].zero_()
        assert torch.equal(snapshot._tah_key_cache[0][1],old_key) and torch.equal(snapshot._tah_valid_mask_cache[0][1],old_mask)
        checks['cache_branches_deeply_independent']=True
        substitute=restore_cache_snapshot(snapshot);intervene_deep_kv(substitute,3,'shallow_substitute')
        select=(substitute._tah_position_id_cache[0][1][0]==3).nonzero().flatten();original_index=(substitute._tah_position_id_cache[0][0][0]==3).nonzero().flatten()
        assert torch.equal(substitute._tah_key_cache[0][1][...,select,:],substitute._tah_key_cache[0][0][...,original_index,:])
        checks['shallow_substitution_preserves_shape_position_and_count']=True
        # Native mixture verified against its exact terminal-iteration formula;
        # this does not impersonate a run of the unavailable serving engine.
        anchors=list(range(len(ids)-1));mixed=b.forward_segment(ids,depths=[2]*len(ids),full_ids=ids,interest=anchors,anchors=anchors)
        reconstructed=[]
        for pos in anchors:
            raw1=torch.from_numpy(mixed['anchor_logits'][f'p{pos}_depth1']).cuda().to(torch.bfloat16)
            raw2=torch.from_numpy(mixed['anchor_logits'][f'p{pos}_depth2']).cuda().to(torch.bfloat16)
            g=torch.tensor(mixed['stop_scores'][pos][1]['continue_probability'],device='cuda',dtype=torch.float32)
            expected=torch.logaddexp(torch.log_softmax(raw1,-1)+torch.log(1-g).to(torch.bfloat16),torch.log_softmax(raw2,-1)+torch.log(g).to(torch.bfloat16)).to(torch.bfloat16)
            reconstructed.append(float((mixed['output'].logits[0,pos]-expected).abs().max()))
        checks['stop_prob_mix_manual_reconstruction_max_difference']=max(reconstructed);assert max(reconstructed)<.1
    prompt=b.tokenizer.apply_chat_template([{'role':'user','content':'What is 1+1? Give only the answer.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    example=b.generate(prompt,max_tokens=64)
    if official['status']=='executed':
        native_ids=[i for i in example['generated_token_ids'] if i not in [b.tokenizer.eos_token_id,151645]]
        official['equivalence_verified']=official['result'][0]['token_ids']==native_ids
        official['comparison_scope']='exact greedy output IDs for the same 64-token smoke prompt; not full logit equivalence'
        (ROOT/'outputs/official_engine_example.json').write_text(json.dumps(official,indent=2,default=str),encoding='utf8')
    hash_after=weight_hash(b.model);assert hash_before==hash_after
    report={'status':'native_backend_consistency_pass','evidence_scope':'engineering/native official path only; serving equivalence remains separately reported',
            'model_parameters':b.parameters,'dtype':'BF16','attention_implementation':'official native SDPA','weighted_hidden_method':b.model.weighted_hidden_method,
            'checks':checks,'weights_hash_before':hash_before,'weights_hash_after':hash_after,'actual_native_example':example,
            'gpu_forward_seconds':b.gpu_seconds,'calls':b.calls,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1048576,
            'official_serving_equivalence_verified':official.get('equivalence_verified',False)}
    (ROOT/'outputs/preflight_native.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('NATIVE_PREFLIGHT_PASS',json.dumps(report),flush=True)
if __name__=='__main__':main()
