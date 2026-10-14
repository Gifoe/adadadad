"""Frozen official TaH2 backend; native stop_prob_mix and hierarchical cache."""
import dataclasses,hashlib,json,os,pathlib,random,subprocess,sys,time
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np,torch,torch.nn.functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1];SOURCE=ROOT/'upstream/TaH'
torch.set_num_threads(4)
sys.path.insert(0,str(SOURCE))
from tah2.model.tah_model import TaHForCausalLM
from tah2.model.causal_cache import TaHCache
from transformers import AutoModelForCausalLM,AutoTokenizer

def set_seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

def weight_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def capture_cache_snapshot(cache):
    """Unlike official snapshot_iter, clones data also for in-place ablations."""
    new=cache.snapshot_iter(1)
    for field in ['_tah_key_cache','_tah_value_cache','_tah_position_id_cache','_tah_valid_mask_cache']:
        value=getattr(cache,field);setattr(new,field,{layer:{depth:t.clone() for depth,t in depths.items()} for layer,depths in value.items()})
    new.current_iter_depth=cache.current_iter_depth;new.disable_update=cache.disable_update
    for field in ['_position_ids_to_cache','_valid_mask_to_cache']:
        if hasattr(cache,field):setattr(new,field,getattr(cache,field).clone())
    return new

def restore_cache_snapshot(snapshot):return capture_cache_snapshot(snapshot)

def cache_depth_stats(cache):
    shapes={};valid={}
    for depth,tensor in cache._tah_key_cache[0].items():
        shapes[str(depth+1)]=int(tensor.shape[-2]);valid[str(depth+1)]=int(cache._tah_valid_mask_cache[0][depth].sum())
        assert all(cache._tah_key_cache[layer][depth].shape[-2]==tensor.shape[-2] for layer in range(28))
    return {'layers':28,'physical_rows_per_layer_by_loop':shapes,'valid_rows_per_layer_by_loop':valid}

@torch.inference_mode()
def append_invisible_deep_slot(cache,position):
    """K1 executes no second loop; pad its storage to K2's masked layout.

    No valid KV or future information is added. Position and row ordering match
    the K2 branch, removing BF16 SDPA shape changes from the causal contrast.
    """
    previous=cache.current_iter_depth;cache.current_iter_depth=1
    for layer in range(28):
        pos=cache._tah_position_id_cache[layer][0]
        selected=(pos[0]==position).nonzero().flatten();assert len(selected)==1
        key=cache._tah_key_cache[layer][0][...,selected,:]
        cache.position_ids_to_cache=pos[:,selected]
        cache.valid_mask_to_cache=torch.zeros_like(cache._tah_valid_mask_cache[layer][0][:,selected])
        cache.update(torch.zeros_like(key),torch.zeros_like(key),layer)
    cache.current_iter_depth=previous

@torch.inference_mode()
def intervene_deep_kv(cache,position,mode):
    assert mode in ['mask','shallow_substitute'];count=0
    for layer,depths in cache._tah_key_cache.items():
        if 1 not in depths:continue
        positions=cache._tah_position_id_cache[layer][1];select=positions==position
        if not select.any():continue
        assert int(select.sum())==1 and positions.shape[0]==1
        if mode=='mask':cache._tah_valid_mask_cache[layer][1][select]=0
        else:
            shallow=cache._tah_position_id_cache[layer][0];indices=(shallow[0]==position).nonzero().flatten()
            assert len(indices)==1;target=select[0].nonzero().flatten()
            for field in ['_tah_key_cache','_tah_value_cache']:
                values=getattr(cache,field);values[layer][1][...,target,:]=values[layer][0][...,indices,:]
        count+=1
    assert count==28,'Expected exactly one deep source entry in all28layers'
    return {'mode':mode,'source_position':position,'layers_changed':count,'shape_and_position_ids_preserved':True}

class NativeBackend:
    def __init__(self,standard=False):
        set_seed();assert torch.cuda.is_available()
        commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip();assert commit=='78ddf9a2c25713fced03a736c5746161f6ee590c'
        self.standard=standard;self.path=ROOT/'models'/('TaH2-1.7B-Standard' if standard else 'TaH2-1.7B-max2')
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        if standard:self.model=AutoModelForCausalLM.from_pretrained(self.path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        else:
            saved=TaHForCausalLM._load_saved_tah_config(str(self.path));assert saved.max_iter==2 and saved.weighted_hidden_method=='stop_prob_mix' and saved.iter_attention_mode=='duo'
            adjusted=dataclasses.replace(saved,iter_attention_impl='sdpa')
            self.model=TaHForCausalLM.from_pretrained(str(self.path),tah_config=adjusted,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        self.model.to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
        assert not self.model.training and not any(p.requires_grad for p in self.model.parameters())
        self.parameters=sum(p.numel() for p in self.model.parameters());self.gpu_seconds=0.;self.calls=0;self.trace=[];self.raw_anchor_logits={};self.live=None
        if not standard:
            self.original_process=self.model._process_sparse_iteration
            def wrapped_process(**kwargs):
                output=self.original_process(**kwargs)
                if self.live is not None:
                    positions=kwargs['position_ids'][0];valid=kwargs['valid_mask'][0].bool();depth=int(kwargs['iter_depth'])+1
                    selected=[index for index,pos in enumerate(positions.tolist()) if valid[index] and int(pos) in self.live['interest'] and int(pos)+1<len(self.live['full_ids'])]
                    if selected:
                        index=torch.tensor(selected,device='cuda');logits=output.logits[0].index_select(0,index).float();q=positions.index_select(0,index)
                        targets=self.live['full_ids'][q+1];loss=F.cross_entropy(logits,targets,reduction='none');top_values,top_ids=logits.topk(8,dim=-1)
                        for at,pos in enumerate(q.tolist()):
                            self.trace.append({'position':int(pos),'depth':depth,'raw_ce_nats':float(loss[at]),'target_token_id':int(targets[at]),
                                               'top8_raw_logits':top_values[at].cpu().tolist(),'top8_token_ids':top_ids[at].cpu().tolist()})
                            if int(pos) in self.live['anchors']:self.raw_anchor_logits[f'p{pos}_depth{depth}']=logits[at].cpu().numpy()
                return output
            self.model._process_sparse_iteration=wrapped_process

    @torch.inference_mode()
    def forward_segment(self,ids,start=0,depths=None,cache=None,full_ids=None,interest=None,anchors=(),instrument=True):
        x=torch.tensor(ids,dtype=torch.long,device='cuda').unsqueeze(0);assert x.shape[1]>0
        position=torch.arange(start,start+len(ids),device='cuda').unsqueeze(0)
        counts=torch.tensor(depths,dtype=torch.long,device='cuda').unsqueeze(0) if depths is not None else None
        self.trace=[];self.raw_anchor_logits={}
        full=torch.tensor(full_ids if full_ids is not None else ids,dtype=torch.long,device='cuda')
        chosen=set(interest) if interest is not None else set(range(start,start+len(ids)-1))
        self.live={'full_ids':full,'interest':chosen,'anchors':set(anchors)} if instrument and not self.standard else None
        torch.cuda.synchronize();began=time.perf_counter()
        if self.standard:
            result=self.model(input_ids=x,position_ids=position,past_key_values=cache,use_cache=True)
        else:
            result=self.model(input_ids=x,position_ids=position,iter_count=counts,past_key_values=cache,use_cache=True,labels=None,return_eval_metadata=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-began;self.gpu_seconds+=seconds;self.calls+=1
        self.live=None;scores={}
        wanted=sorted(p for p in chosen if start<=p<start+len(ids) and p+1<len(full))
        for first in range(0,len(wanted),64):
            subset=wanted[first:first+64];index=torch.tensor([p-start for p in subset],device='cuda');q=torch.tensor(subset,device='cuda')
            logits=result.logits[0].index_select(0,index).float();target=full[q+1];ce=F.cross_entropy(logits,target,reduction='none');prediction=logits.argmax(-1)
            for n,pos in enumerate(subset):scores[pos]={'ce_nats':float(ce[n]),'predicted_token_id':int(prediction[n]),'target_token_id':int(target[n]),'correct':bool(prediction[n]==target[n])}
        stop={}
        if not self.standard and result.decider_logits:
            for depth,dense in enumerate(result.decider_logits,1):
                values=dense[0].float()
                for pos in wanted:
                    value=float(values[pos-start])
                    if math_isfinite(value):stop.setdefault(pos,{})[depth]={'continue_logit':value,'continue_probability':float(torch.sigmoid(values[pos-start]))}
        actual=result.iter_count[0].cpu().tolist() if not self.standard else [1]*len(ids)
        return {'output':result,'cache':result.past_key_values,'scores':scores,'loop_trace':self.trace.copy(),'anchor_logits':self.raw_anchor_logits.copy(),
                'stop_scores':stop,'depths':actual,'executed_iterations':sum(actual),'seconds':seconds,
                'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1048576}

    @torch.inference_mode()
    def native_policy(self,ids):
        result=self.forward_segment(ids,full_ids=ids,interest=range(len(ids)-1))
        return result['depths'],result['scores'],result['stop_scores']

    @torch.inference_mode()
    def replay_suffix(self,ids,start,depths,snapshot,interest,anchors=()):
        return self.forward_segment(ids[start:],start,depths[start:],restore_cache_snapshot(snapshot),full_ids=ids,interest=interest,anchors=anchors)

    @torch.inference_mode()
    def generate(self,prompt,mode='adaptive',max_tokens=512):
        ids=self.tokenizer(prompt,add_special_tokens=False)['input_ids'];assert len(ids)<=4096,'No silent truncation'
        k=None if mode=='adaptive' or self.standard else 1 if mode=='forced1' else 2
        torch.cuda.reset_peak_memory_stats();before=self.gpu_seconds;began=time.perf_counter();iterations=0;generated=[];depths_generated=[]
        initial=self.forward_segment(ids,depths=None if k is None else [k]*len(ids),interest=(),instrument=False)
        cache=initial['cache'];iterations+=initial['executed_iterations'];logits=initial['output'].logits[:,-1,:]
        for _ in range(max_tokens):
            token=int(logits.argmax(-1));generated.append(token)
            if token in [self.tokenizer.eos_token_id,151645]:break
            pos=len(ids)+len(generated)-1
            step=self.forward_segment([token],pos,None if k is None else [k],cache,interest=(),instrument=False)
            cache=step['cache'];iterations+=step['executed_iterations'];depths_generated.extend(step['depths']);logits=step['output'].logits[:,-1,:]
        text=self.tokenizer.decode(generated,skip_special_tokens=True)
        return {'text':text,'generated_token_ids':generated,'input_tokens':len(ids),'generated_tokens':len(generated),
                'actual_iterations':iterations,'generated_position_depths':depths_generated,
                'mean_actual_depth_all_executed_input_positions':iterations/(len(ids)+len(depths_generated)),
                'seconds':time.perf_counter()-began,'cuda_forward_seconds':self.gpu_seconds-before,
                'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1048576,
                'hit_generation_limit':len(generated)==max_tokens,'generation_policy_uses_reference':False}

def math_isfinite(x):return bool(np.isfinite(x))
