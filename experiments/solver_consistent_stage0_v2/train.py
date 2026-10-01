import argparse, hashlib, json, math, os, pathlib, random, time
import numpy as np
import torch
from torch.nn import functional as F
from runtime import *

@torch.no_grad()
def validate(m,data,c):
    m.eval(); acc=0; loss=0
    for start in range(0,len(data['labels']),c['eval_batch']):
        idx=np.arange(start,min(start+c['eval_batch'],len(data['labels']))); x,y=batch(data,idx,device())
        with amp(): z=m(x,8)
        acc+=(z.argmax(-1)==y).sum().item(); loss+=F.cross_entropy(z.float(),y,reduction='sum').item()
    m.train()
    return {'accuracy':acc/len(data['labels']),'nll':loss/len(data['labels'])}

def atomic_save(obj,path):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def train(name,c=None,sanity=False,out_override=None):
    c=c or config(); seed_all(c['seed']); dev=device()
    if dev.type!='cuda': raise RuntimeError('Formal BF16 experiment requires CUDA')
    data=load_data('train'); val=load_data('validation')
    if sanity:
        data={k:v[data['depths']==1] for k,v in data.items()}
        val={k:v[val['depths']==1] for k,v in val.items()}
        c=dict(c,updates=c.get('sanity_updates',2000),validation_interval=100)
    out=pathlib.Path(out_override) if out_override else ROOT/'artifacts'/('sanity' if sanity else name)
    if sanity and not out_override: out=out/name
    out.mkdir(parents=True,exist_ok=True)
    m=build(c,name).to(dev); print(name,counts(m),flush=True)
    opt=torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    indices,budgets,digest=training_plan(len(data['labels']),c)
    warm=max(1,int(c['warmup_fraction']*c['updates']))
    def multiplier(u):
        if u<warm: return (u+1)/warm
        return .5*(1+math.cos(math.pi*(u-warm)/max(1,c['updates']-warm)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,multiplier)
    best=-1.; best_nll=float('inf'); begin=time.perf_counter(); actual_tokens=0
    start_update=0; elapsed_before=0.
    final=out/'final.pt';latest=out/'latest.pt'
    if latest.exists() and not sanity:
        state=torch.load(latest,map_location=dev,weights_only=False)
        assert state['plan_sha256']==digest and state['config']==c
        m.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_update=state['update']; best=state['best_accuracy']; best_nll=state['best_nll']; elapsed_before=state['training_wall_sec']; actual_tokens=state['tokens_seen']
        torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        if 'numpy_rng' in state:np.random.set_state(state['numpy_rng'])
        if 'python_rng' in state:random.setstate(state['python_rng'])
    torch.cuda.reset_peak_memory_stats(); log=(out/'training.jsonl').open('a',encoding='utf-8')
    m.train()
    for u in range(start_update,c['updates']):
        opt.zero_grad(set_to_none=True); losses=0.
        k=int(budgets[u]); step_start=time.perf_counter()
        for start in range(0,c['effective_batch'],c['micro_batch']):
            idx=indices[u,start:start+c['micro_batch']]; x,y=batch(data,idx,dev)
            with amp(): z=m(x,k); loss=F.cross_entropy(z.float(),y)*len(idx)/c['effective_batch']
            if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss {name} update {u}')
            loss.backward(); losses+=loss.item(); actual_tokens+=int(data['lengths'][idx].sum())
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),c['grad_clip'])
        if not torch.isfinite(norm): raise RuntimeError('nonfinite gradients')
        opt.step(); scheduler.step()
        if (u+1)%20==0:
            record={'update':u+1,'budget':k,'loss':losses,'lr':opt.param_groups[0]['lr'],'elapsed_sec':time.perf_counter()-begin+elapsed_before}
            log.write(json.dumps(record)+'\n'); log.flush(); print(name,record,flush=True)
        def snapshot(validation=None):
            sync();elapsed=time.perf_counter()-begin+elapsed_before
            manifest=ROOT/'data/generated/manifest.json'
            return {'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'update':u+1,'config':c,
                'data_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'plan_sha256':digest,'validation':validation,'best_accuracy':best,'best_nll':best_nll,
                'training_wall_sec':elapsed,'examples_seen':(u+1)*c['effective_batch'],'tokens_seen':actual_tokens,
                'examples_per_sec':(u+1)*c['effective_batch']/elapsed,'tokens_per_sec':actual_tokens/elapsed,
                'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20,'cuda_memory_fraction':.65,'counts':counts(m),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        if (u+1)%25==0:atomic_save(snapshot(),latest)
        if (u+1)%c['validation_interval']==0 or u+1==c['updates']:
            v=validate(m,val,c); print(name,'validation',u+1,v,flush=True)
            is_best=v['accuracy']>best or (v['accuracy']==best and v['nll']<best_nll)
            if is_best: best=v['accuracy']; best_nll=v['nll']
            saved=snapshot(v);atomic_save(saved,latest);atomic_save(saved,final)
            if is_best: atomic_save(saved,out/'best.pt')
            meta={k:v for k,v in saved.items() if k not in ('model','optimizer','scheduler','torch_rng','cuda_rng')}
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    log.close()
    if sanity:
        passed=best>=.65
        (out/'gate.json').write_text(json.dumps({'passed':passed,'threshold':.65,'validation_accuracy':best,'updates':c['updates']},indent=2))
        if not passed: raise RuntimeError(f'Check A failed: {name} depth1 accuracy {best}; formal training gated')
    del m,opt; torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=NAMES,required=True);p.add_argument('--sanity',action='store_true');p.add_argument('--updates',type=int);p.add_argument('--output');a=p.parse_args()
    c=config()
    if a.updates:c=dict(c,updates=a.updates,validation_interval=min(c['validation_interval'],a.updates))
    train(a.model,c=c,sanity=a.sanity,out_override=a.output)
