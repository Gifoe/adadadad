import os,time,json,pathlib,sys,csv,random,hashlib,platform,subprocess,traceback
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from model_extension import make_model,losses,RecurrentGPT2Block,CompositionDataset
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'outputs';CK=ROOT/'checkpoints'
CONFIG=dict(seed=42,d_model=768,num_recurrent_layers=4,num_heads=12,positional_embedding_type='none',dropout=0,precision='bf16',c_scale=0.,input_injection=False,lr=1e-4,weight_decay=.01,adam_betas=[.9,.999],warmup_steps=2000,lr_schedule='official linear warmup/decay, frozen horizon 20000 for both arms including conditional extension',K_train=5,lambda_transition=1.,batch_size=128,curriculum=[[2,2000],[3,2000],[4,3000],[5,5000]],initial_updates=12000,max_updates=20000,intervention_count_per_D=200,prototype_min_count=10,upstream_commit='bb22b192977b1b136fde985c8e7a2392d172d97a',D=[2,3,4,5,6,8,10,12,16,20],K=[1,2,3,4,5,6,8,10,12,16,20,24],same_state_majority_threshold=.5,noise='replace state by isotropic Gaussian direction rescaled to CF prototype L2 norm',frontier='minimum swept K within 95% of swept peak',flip='any swept K<24 correct and K24 wrong, denominator any early correct')
def dump(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def save(obj,path):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)
def status(stage,**kw):dump(ROOT/'STATUS.json',dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(stage,kw,flush=True)
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def load_data():
    train={k:v for k,v in np.load(ROOT/'data/generated/train.npz').items()}
    test={k:v for k,v in np.load(ROOT/'data/generated/test.npz').items()}
    return train,test
def batch(data,indices,length=None):
    hops=data['hops'][indices];length=length or int(hops.max())+2
    x=torch.as_tensor(data['input_ids'][indices,:length],device='cuda')
    mask=(x!=0).long();p=torch.as_tensor(data['state_pos'][indices],device='cuda')
    targets=torch.as_tensor(data['state_targets'][indices],device='cuda')
    assert torch.all(x[torch.arange(len(indices),device='cuda'),p]==217)
    assert torch.equal(mask.sum(1)-1,p)
    return x,mask,p,targets
def plan(train):
    path=ROOT/'data/sample_plan.npy'
    if path.exists():return np.load(path)
    rng=np.random.RandomState(42);rows=[]
    for hop,steps in CONFIG['curriculum']+[[5,8000]]:
        pool=np.flatnonzero(train['hops']<=hop);flat=[]
        while len(flat)<steps*128:flat.extend(rng.permutation(pool).tolist())
        rows.extend(np.asarray(flat[:steps*128]).reshape(steps,128))
    result=np.asarray(rows,dtype=np.int32);np.save(path,result)
    dump(ROOT/'data/sample_plan_manifest.json',dict(steps=len(result),batch_size=128,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_training_hop=5))
    return result
def optimizer(model):
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    sch=get_linear_schedule_with_warmup(opt,2000,20000)
    return opt,sch
@torch.no_grad()
def id_eval(model,test):
    model.eval();model.num_iterations=5;counts={}
    for d in [2,3,4,5]:
        ind=np.flatnonzero(test['hops']==d);correct=0
        for start in range(0,len(ind),128):
            x,mask,p,t=batch(test,ind[start:start+128])
            with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x,mask,p,True)
            correct+=int((o.final_logits.argmax(-1)==t[:,4]).sum())
        counts[str(d)]=dict(correct=correct,total=len(ind),accuracy=correct/len(ind))
    return dict(by_hop=counts,macro=sum(x['accuracy'] for x in counts.values())/4)
def unit(model,train,test):
    ind=np.array([np.flatnonzero(train['hops']==d)[0] for d in range(1,6)])
    x,mask,p,t=batch(train,ind)
    model.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(x,mask,p,True);l0,lf,li=losses(out,t,0.)
        original=RecurrentGPT2Block.forward(model,x,mask).logits
        default=model(x,mask).logits
    baseline_reference=torch.nn.functional.cross_entropy(out.final_logits.float(),t[:,4])
    delta=float(abs(l0-baseline_reference));diff=float((original-default).abs().max())
    assert delta<1e-6 and diff<1e-6
    assert model.lm_head.weight is model.token_embedding.weight
    for h in out.hidden_states_per_recurrence:h.retain_grad()
    l0.backward();norms=[float(h.grad.float().norm()) for h in out.hidden_states_per_recurrence]
    assert all(np.isfinite(n) and n>0 for n in norms)
    model.zero_grad(set_to_none=True)
    # Check official tokenizer and dataset target extraction on actual input.
    vocab=json.loads((ROOT/'data/vocab.json').read_text());records=json.loads((ROOT/'data/generated/train.json').read_text())
    ds=CompositionDataset({v:i for i,v in enumerate(vocab)},records)
    for i in ind:
        a,b=ds[int(i)];assert np.array_equal(a.numpy(),train['input_ids'][i,:train['hops'][i]+2]);assert b[-1].item()==train['state_targets'][i,4]
    # Causal mask: preceding tokens cannot depend on <state>.
    model.num_iterations=2
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        normal=model(x,mask,p,True);patched=model(x,mask,p,True,patch=(1,torch.randn(len(ind),768,device='cuda')))
    max_other=0.
    for row,pos in enumerate(p.tolist()):max_other=max(max_other,float((normal.hidden_states_per_recurrence[1][row,:pos]-patched.hidden_states_per_recurrence[1][row,:pos]).abs().max()))
    assert max_other==0.
    model.num_iterations=5
    dump(OUT/'unit_tests.json',dict(status='PASS',lambda0_difference=delta,default_vs_official_max_difference=diff,grad_norm_per_recurrence=norms,state_pos_checked=True,targets_absorbing_checked=True,pre_state_patch_leakage=max_other))
def train_arm(name,budget,train,test,sample_plan,smoke=False):
    seed();model=make_model(218).cuda();model.load_state_dict(torch.load(CK/'initial_weights.pt',weights_only=True));opt,sch=optimizer(model)
    start=0;elapsed_previous=0.;checkpoint=CK/(name+'_resume.pt')
    if checkpoint.exists() and not smoke:
        ck=torch.load(checkpoint,weights_only=False,map_location='cuda');model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);start=ck['step'];elapsed_previous=ck['seconds']
    history=OUT/(('smoke_' if smoke else 'train_history_')+name+'.csv')
    fields=['step','max_hop','loss','final_loss','transition_loss','lr','step_seconds','elapsed_seconds','peak_vram_mib']
    mode='a' if start>0 else 'w'
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter();first=[];last=[]
    with history.open(mode,newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if start==0:writer.writeheader()
        for step in range(start,budget):
            model.train();model.num_iterations=5;s0=time.perf_counter();ind=sample_plan[step]
            x,mask,p,t=batch(train,ind);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,mask,p,True);loss,lf,li=losses(out,t,float(name=='grounded'))
            assert torch.isfinite(loss),f'Nonfinite loss {step}'
            loss.backward();opt.step();sch.step();torch.cuda.synchronize()
            lv=float(loss);first.append(lv) if step<20 else None
            last.append(lv);last=last[-20:]
            elapsed=elapsed_previous+time.perf_counter()-begin
            writer.writerow(dict(step=step+1,max_hop=int(train['hops'][ind].max()),loss=lv,final_loss=float(lf),transition_loss=float(li),lr=opt.param_groups[0]['lr'],step_seconds=time.perf_counter()-s0,elapsed_seconds=elapsed,peak_vram_mib=torch.cuda.max_memory_allocated()/2**20))
            if (step+1)%50==0:f.flush();status(('SMOKE_' if smoke else 'TRAIN_')+name,step=step+1,budget=budget,loss=lv,elapsed_seconds=elapsed)
            if not smoke and ((step+1)%250==0 or step+1==budget):
                f.flush();save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),step=step+1,seconds=elapsed),checkpoint)
            del out,loss,lf,li
    result=dict(seconds=elapsed_previous+time.perf_counter()-begin,peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,optimizer_steps=budget,examples_seen=budget*128)
    result['examples_per_second']=result['examples_seen']/result['seconds']
    if smoke:
        result.update(first20_loss=float(np.mean(first)),last20_loss=float(np.mean(last)),loss_decreased=bool(np.mean(last)<np.mean(first)),projected_two_model_12000_hours=result['seconds']/budget*24000/3600)
        assert result['loss_decreased']
        dump(OUT/f'smoke_{name}_metrics.json',result)
    else:
        save(dict(model_state_dict=model.state_dict(),step=budget),CK/(name+'_final.pt'))
        result['id']=id_eval(model,test);dump(OUT/f'training_{name}_summary.json',result)
    del model,opt,sch;torch.cuda.empty_cache()
    return result
def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True);CK.mkdir(exist_ok=True)
    completed=OUT/'completion_audit.json'
    if completed.exists() and json.loads(completed.read_text()).get('completed'):
        print('ALREADY_COMPLETED: preserving frozen artifacts; reproduce in a fresh directory.',flush=True);return
    dump(ROOT/'CONFIG.json',CONFIG)
    train,test=load_data();sample_plan=plan(train)
    if not (CK/'initial_weights.pt').exists():
        seed();m=make_model(218);save(m.state_dict(),CK/'initial_weights.pt');del m
    import transformers
    dump(ROOT/'RUN_METADATA.json',dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),platform=platform.platform(),initial_sha256=hashlib.sha256((CK/'initial_weights.pt').read_bytes()).hexdigest(),upstream_commit=CONFIG['upstream_commit'],start=time.strftime('%Y-%m-%d %H:%M:%S'),parameters=sum(p.numel() for p in make_model(218).parameters()),hostname=platform.node()))
    if not (OUT/'unit_tests.json').exists():
        m=make_model(218).cuda();m.load_state_dict(torch.load(CK/'initial_weights.pt',weights_only=True));unit(m,train,test);del m;torch.cuda.empty_cache()
    for name in ['baseline','grounded']:
        if not (OUT/f'smoke_{name}_metrics.json').exists():train_arm(name,200,train,test,sample_plan,True)
    status('SMOKE_PASS')
    results={}
    extension_already_started=(OUT/'extension_trigger.json').exists()
    for name in ['baseline','grounded']:results[name]=train_arm(name,20000 if extension_already_started else 12000,train,test,sample_plan)
    if not extension_already_started and any(x['id']['macro']<.9 for x in results.values()):
        status('MATCHED_EXTENSION_TO_20000',initial_id={k:v['id'] for k,v in results.items()})
        dump(OUT/'extension_trigger.json',{k:v['id'] for k,v in results.items()})
        for name in ['baseline','grounded']:results[name]=train_arm(name,20000,train,test,sample_plan)
    valid=all(x['id']['macro']>=.9 for x in results.values())
    smoke={name:json.loads((OUT/f'smoke_{name}_metrics.json').read_text()) for name in results}
    dump(OUT/'runtime_metrics.json',dict(training=results,smoke=smoke,total_gpu_hours=(sum(x['seconds'] for x in results.values())+sum(x['seconds'] for x in smoke.values()))/3600,training_valid=valid))
    if not valid:
        status('INVALID_TRAINING_RUN',id_macro={k:v['id']['macro'] for k,v in results.items()})
        from analysis_report import report_invalid
        report_invalid();return
    status('TRAINING_VALID')
    from evaluate import evaluate_all
    evaluate_all()
    from analysis_report import report
    report();status('COMPLETE')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('EXECUTION_ERROR',error=repr(e),traceback=traceback.format_exc());raise
