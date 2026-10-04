"""Frozen protocol closure; official A1 gates all matched training."""
import csv, hashlib, json, math, pathlib, random, shutil, sys, time, traceback
import numpy as np
import torch
from transformers import GPT2Config, get_linear_schedule_with_warmup
from model_extension import make_model, losses, RecurrentGPT2Block, CompositionDataset
from gpt_utils_extrapolation import custom_collate

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'outputs'; CK=ROOT/'checkpoints'; DATA=ROOT/'data'
V1=ROOT.parent/'transition_grounded_recurrent_reasoning_seed42_v1'
CONFIG=dict(seed=42,d_model=768,num_recurrent_layers=4,num_heads=12,dropout=0,positional_embedding_type='none',precision='bf16',c_scale=0,input_injection=False,batch_size=128,lr=1e-4,weight_decay=.01,warmup_steps=2000,adam_betas=[.9,.999],num_epochs=100001,max_updates_per_stage=50000,evaluation_interval=500,stages=[2,3,4,5],threshold=.95,consecutive_evaluations=3,K_train=5,lambda_transition=1.,official_dynamic=dict(mean=4,min=2,max=8),official_eval_K=8,official_advance='current validation > .95 once at epoch end',official_sanity_threshold=.90,id_macro_threshold=.90,id_each_hop_preferred=.85,id_K=[1,2,3,4,5,6,8,10,12],D=[2,3,4,5,6,8,10,12,16,20],K=[1,2,3,4,5,6,8,10,12,16,20,24],intervention_count_per_D=200,prototype_min_count=10,same_state_majority_threshold=.5,upstream_commit='bb22b192977b1b136fde985c8e7a2392d172d97a',scheduler_horizon='ceil(cumulative stage examples / 128) * 100001',recurrence_resolution='User explicitly selected fixed K=5 for formal comparison',validation_per_ID_hop=375,independent_test_per_ID_hop=375)

def dump(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def status(stage,**kw):
    dump(ROOT/'STATUS.json',dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(stage,kw,flush=True)
def csvwrite(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def load_data():
    return tuple({k:v for k,v in np.load(DATA/'generated'/f'{n}.npz').items()} for n in ['train','test'])
def batch(data,indices,length=None):
    hops=data['hops'][indices];length=length or int(hops.max())+2
    x=torch.as_tensor(data['input_ids'][indices,:length],device='cuda');mask=(x!=0).long()
    p=torch.as_tensor(data['state_pos'][indices],device='cuda');t=torch.as_tensor(data['state_targets'][indices],device='cuda')
    assert torch.all(x[torch.arange(len(indices),device='cuda'),p]==217)
    return x,mask,p,t
def official_batch(data,indices):
    x=np.zeros((len(indices),50),dtype=np.int64)
    for row,i in enumerate(indices):x[row,:int(data['hops'][i])+1]=data['input_ids'][i,:int(data['hops'][i])+1]
    x=torch.as_tensor(x,device='cuda');return x,(x!=0).long(),torch.as_tensor(data['state_targets'][indices,-1],device='cuda')
def prepare():
    for p in [OUT,CK,DATA/'generated',ROOT/'figures']:p.mkdir(parents=True,exist_ok=True)
    for n in ['train.npz','test.npz','train.json','test.json']:
        if not (DATA/'generated'/n).exists():shutil.copy2(V1/'data/generated'/n,DATA/'generated'/n)
    for n in ['vocab.json','transition_table.json','integrity_check.json','dataset_stats.json']:
        shutil.copy2(V1/'data'/n,DATA/n)
    if not (CK/'initial_weights.pt').exists():shutil.copy2(V1/'checkpoints/initial_weights.pt',CK/'initial_weights.pt')
    train,test=load_data();rng=np.random.RandomState(42);val=[];held=[]
    for d in [2,3,4,5]:
        ix=rng.permutation(np.flatnonzero(test['hops']==d));val.extend(ix[:375]);held.extend(ix[375:])
    assert not set(val)&set(held)
    split=dict(validation_indices=list(map(int,val)),test_indices=list(map(int,held)),validation_ids=test['sample_ids'][val].tolist(),test_ids=test['sample_ids'][held].tolist(),train_sha256=hashlib.sha256((DATA/'generated/train.npz').read_bytes()).hexdigest(),source='unchanged v1 train/test; disjoint partition of existing held-out ID only')
    frozen=DATA/'closure_split_manifest.json'
    if frozen.exists():assert json.loads(frozen.read_text())==split
    else:dump(frozen,split)
    dump(ROOT/'CONFIG.json',CONFIG)
    return train,test,split
def official_model():
    cfg=GPT2Config(vocab_size=217,n_positions=50,n_ctx=50,n_embd=768,n_layer=4,n_head=12,embd_pdrop=0.,attn_pdrop=0.,resid_pdrop=0.,_attn_implementation='eager')
    return RecurrentGPT2Block(cfg,8,positional_embedding_type='none',input_injection=False,c_scale=0.)
@torch.no_grad()
def accuracies(model,data,indices,official=False,K=5,ds=None):
    model.eval();model.num_iterations=K;result={}
    for d in ds or sorted(set(data['hops'][indices].tolist())):
        ix=np.asarray(indices)[data['hops'][indices]==d];correct=0
        for s in range(0,len(ix),128):
            if official:
                x,mask,t=official_batch(data,ix[s:s+128])
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,mask).logits[:,-1].argmax(-1)
            else:
                x,mask,p,t=batch(data,ix[s:s+128]);t=t[:,-1]
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,mask,p,True).final_logits.argmax(-1)
            correct+=int((pred==t).sum())
        result[str(d)]=dict(correct=correct,total=len(ix),accuracy=correct/len(ix))
    return result
def sanity(train,test):
    # Verify cached official data conversion against unmodified official tokenizer/collator.
    raw=json.loads((DATA/'generated/train.json').read_text());vocab=json.loads((DATA/'vocab.json').read_text())[:-1]
    ix=np.array([np.flatnonzero(train['hops']==d)[0] for d in range(1,6)])
    records=[]
    for i in ix:
        r=raw[int(i)].copy();r['input_text']=r['input_text'].replace('<state>','');r['target_text']=r['target_text'].replace('<state>','');records.append(r)
    ds=CompositionDataset({t:i for i,t in enumerate(vocab)},records)
    x,t,mask,lens=custom_collate([ds[i] for i in range(5)],50);cx,cm,ct=official_batch(train,ix)
    assert torch.equal(x,cx.cpu()) and torch.equal(mask,cm.cpu()) and torch.equal(t,ct.cpu())
    m=make_model(218).cuda();m.load_state_dict(torch.load(CK/'initial_weights.pt',weights_only=True))
    x,mask,p,t=batch(train,ix)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        o=m(x,mask,p,True);loss,_,_=losses(o,t,0);ref=torch.nn.functional.cross_entropy(o.final_logits.float(),t[:,4]);a=RecurrentGPT2Block.forward(m,x,mask).logits;b=m(x,mask).logits
    assert float(abs(loss-ref))<1e-6 and float((a-b).abs().max())<1e-6
    assert m.lm_head.weight is m.token_embedding.weight
    dump(OUT/'unit_tests.json',dict(status='PASS',official_cache_equals_collate=True,lambda0_equals_finalCE=True,matched_forward_equals_official=True,tied_head=True,dropout_zero=all(q.p==0 for q in m.modules() if isinstance(q,torch.nn.Dropout))))
    del m;torch.cuda.empty_cache()
def train_arm(name,train,test,split):
    official=name=='official_baseline';summary_path=OUT/f'training_{name}_summary.json'
    if summary_path.exists():return json.loads(summary_path.read_text())
    seed();model=(official_model() if official else make_model(218)).cuda()
    if not official:model.load_state_dict(torch.load(CK/'initial_weights.pt',weights_only=True))
    resume_path=CK/f'{name}_resume.pt';resume=None;stages=[];history=[];global_step=0
    if resume_path.exists():
        resume=torch.load(resume_path,map_location='cpu',weights_only=False);model.load_state_dict(resume['model']);stages=resume['stages'];history=resume['history'];global_step=resume['global_step']
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng']);np.random.set_state(resume['numpy_rng']);random.setstate(resume['python_rng'])
    history_path=OUT/('official_baseline_reproduction.csv' if official else f'{name}_learning_curve.csv')
    for d in [2,3,4,5]:
        if any(not r['criterion_completed'] for r in stages):break
        if any(r['D']==d for r in stages):continue
        pool=np.flatnonzero(train['hops']<=d);nb=math.ceil(len(pool)/128);horizon=nb*100001
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);sch=get_linear_schedule_with_warmup(opt,2000,horizon)
        gen=torch.Generator().manual_seed(42+d);step=0;epoch=0;cursor=0;order=pool[torch.randperm(len(pool),generator=gen).numpy()];streak=0;first95=None;examples=0;elapsed_before=0.;rec_counts={str(k):0 for k in range(2,9)};loss_sum=0.;loss_n=0;lastval={};advanced=False
        if resume is not None and resume['D']==d:
            opt.load_state_dict(resume['optimizer']);sch.load_state_dict(resume['scheduler'])
            s=resume['active'];step=s['step'];epoch=s['epoch'];cursor=s['cursor'];order=s['order'];streak=s['streak'];first95=s['first95'];examples=s['examples'];elapsed_before=s['seconds'];rec_counts=s['rec_counts'];gen.set_state(s['generator']);advanced=s['advanced'];lastval=s['lastval']
            active=s
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
        while step<50000 and not advanced:
            if cursor==len(order):
                epoch+=1;cursor=0;order=pool[torch.randperm(len(pool),generator=gen).numpy()]
            ix=order[cursor:cursor+128];cursor+=len(ix);end_epoch=cursor==len(order)
            k=int(np.clip(np.random.poisson(4),2,8)) if official else 5;rec_counts[str(k)]+=1
            model.train();model.num_iterations=k;opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if official:
                    x,mask,t=official_batch(train,ix);o=model(x,mask);loss=torch.nn.functional.cross_entropy(o.logits[:,-1].float(),t)
                else:
                    x,mask,p,t=batch(train,ix);o=model(x,mask,p,True);loss,_,_=losses(o,t,float(name=='grounded'))
            assert torch.isfinite(loss),f'nonfinite {name} D{d} step{step}'
            loss.backward();opt.step();sch.step();loss_sum+=float(loss);loss_n+=1;examples+=len(ix);step+=1;global_step+=1
            del o,loss
            scheduled=step%500==0 or step==50000;evaluate=scheduled or (official and end_epoch)
            if evaluate:
                lastval=accuracies(model,test,split['validation_indices'],official,8 if official else 5,list(range(2,d+1)));acc=lastval[str(d)]['accuracy']
                if acc>=.95 and first95 is None:first95=step
                if official:advanced=bool(end_epoch and acc>.95)
                elif scheduled:streak=streak+1 if acc>=.95 else 0;advanced=streak>=3
                if scheduled or advanced:
                    train_sample=accuracies(model,train,pool[:min(512,len(pool))].tolist(),official,8 if official else 5)
                    torch.cuda.synchronize();seconds=elapsed_before+time.perf_counter()-started
                    row=dict(model=name,stage=d,global_updates=global_step,stage_updates=step,epoch=epoch,train_loss=loss_sum/max(1,loss_n),current_stage_accuracy=acc,previous_stage_accuracy=json.dumps({z:v['accuracy'] for z,v in lastval.items() if int(z)<d}),train_accuracy_sample=sum(v['correct'] for v in train_sample.values())/sum(v['total'] for v in train_sample.values()),lr=opt.param_groups[0]['lr'],recurrence_statistics=json.dumps(rec_counts),consecutive_passes=streak,advanced=advanced,elapsed_stage_seconds=seconds,peak_vram_mib=torch.cuda.max_memory_allocated()/2**20)
                    history.append(row);csvwrite(history_path,history);loss_sum=0.;loss_n=0
                    status('TRAIN_'+name,D=d,stage_updates=step,global_updates=global_step,val_accuracy=acc,streak=streak,seconds=seconds)
                    active=dict(step=step,epoch=epoch,cursor=cursor,order=order,streak=streak,first95=first95,examples=examples,seconds=seconds,rec_counts=rec_counts,generator=gen.get_state(),advanced=advanced,lastval=lastval)
                    save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),D=d,active=active,stages=stages,history=history,global_step=global_step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),resume_path)
            if end_epoch:
                epoch+=1;cursor=0;order=pool[torch.randperm(len(pool),generator=gen).numpy()]
        seconds=elapsed_before+time.perf_counter()-started
        stages.append(dict(D=d,updates=step,updates_to_95=first95,criterion_completed=advanced,validation=lastval,seconds=seconds,gpu_hours=seconds/3600,examples_seen=examples,peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,optimizer_reset=True,initial_lr=0.,base_lr=1e-4,warmup_steps=2000,scheduler_horizon=horizon,final_lr=opt.param_groups[0]['lr']))
        dump(OUT/f'{name}_stage_metrics.json',stages)
        # Persist stage boundary independently of the active-stage resume.
        save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),D=d,active=active,stages=stages,history=history,global_step=global_step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),resume_path)
        resume=None
        if not advanced:break
    ckname='matched_baseline' if name=='baseline' else name
    save(dict(model_state_dict=model.state_dict(),step=global_step),CK/f'{ckname}_final.pt')
    if name=='baseline':shutil.copy2(CK/'matched_baseline_final.pt',CK/'baseline_final.pt')
    final=accuracies(model,test,split['test_indices'],official,8 if official else 5,[2,3,4,5]);macro=sum(v['accuracy'] for v in final.values())/4
    closed=not official and len(stages)==4 and all(s['criterion_completed'] for s in stages) and macro>=.90
    sanity_pass=official and len(stages)==4 and all(s['criterion_completed'] and s['validation'][str(s['D'])]['accuracy']>=.90 for s in stages)
    sweep=[]
    if not official:
        for K in CONFIG['id_K']:
            scores=accuracies(model,test,split['test_indices'],False,K,[2,3,4,5])
            sweep.extend(dict(model=name,D=int(d),K=K,**v) for d,v in scores.items())
        csvwrite(OUT/f'id_closure_{name}.csv',sweep)
    result=dict(model=name,stages=stages,global_updates=global_step,final_id=final,id_macro=macro,id_closed=closed,official_reproduced=sanity_pass,seconds=sum(s['seconds'] for s in stages),examples_seen=sum(s['examples_seen'] for s in stages))
    dump(summary_path,result);del model;torch.cuda.empty_cache();return result
def main():
    torch.set_num_threads(1)
    if (OUT/'completion_audit.json').exists():status('ALREADY_COMPLETE');return
    train,test,split=prepare()
    import transformers,platform
    dump(ROOT/'RUN_METADATA.json',dict(start=time.strftime('%Y-%m-%d %H:%M:%S'),python=sys.version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),hostname=platform.node(),initial_sha256=hashlib.sha256((CK/'initial_weights.pt').read_bytes()).hexdigest(),model_extension_sha256=hashlib.sha256((ROOT/'model_extension.py').read_bytes()).hexdigest()))
    if not (OUT/'unit_tests.json').exists():sanity(train,test)
    results={'official_baseline':train_arm('official_baseline',train,test,split)}
    if not results['official_baseline']['official_reproduced']:decision='OFFICIAL BASELINE REPRODUCTION FAILURE'
    else:
        for name in ['baseline','grounded']:results[name]=train_arm(name,train,test,split)
        a=results['baseline']['id_closed'];b=results['grounded']['id_closed']
        if a and b:
            dump(OUT/'runtime_metrics.json',dict(training=results,total_gpu_hours=sum(r['seconds'] for r in results.values())/3600))
            from evaluate import evaluate_all
            evaluate_all()
            from closure_report import scientific_decision
            decision=scientific_decision()
        else:decision='METHOD TRAINING FAILURE' if a and not b else 'TRAINING CLOSURE FAILURE'
    evaluation_seconds=json.loads((OUT/'runtime_metrics.json').read_text()).get('evaluation_seconds',0.) if (OUT/'runtime_metrics.json').exists() else 0.
    dump(OUT/'runtime_metrics.json',dict(training=results,evaluation_seconds=evaluation_seconds,total_gpu_hours=(sum(r['seconds'] for r in results.values())+evaluation_seconds)/3600,decision=decision,core_evaluation_run=all(results.get(n,{}).get('id_closed',False) for n in ['baseline','grounded'])))
    from closure_report import finalize
    finalize(results,decision);status('COMPLETE',decision=decision)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('EXECUTION_ERROR',error=repr(e),traceback=traceback.format_exc());raise
