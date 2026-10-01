"""Gradient/input plumbing diagnostic on 16 TRAIN examples, no inference claims."""
import os,json,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from runtime import *
c=config(); data=load_data('train'); indices=np.flatnonzero(data['depths']==1)
zero=indices[data['labels'][indices]==0][:8]; one=indices[data['labels'][indices]==1][:8]
ix=np.concatenate([zero,one]); x,y=batch(data,ix,device()); m=build(c,'vanilla_loop').to(device()).train()
opt=torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay']); start=time.perf_counter()
result={}
for u in range(1000):
    opt.zero_grad(set_to_none=True)
    with amp(): z=m(x,4); loss=torch.nn.functional.cross_entropy(z.float(),y)
    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    opt.step()
    if (u+1)%100==0:
        m.eval()
        with torch.no_grad(),amp(): pred=m(x,4); acc=(pred.argmax(-1)==y).float().mean().item()
        m.train(); result={'updates':u+1,'train_accuracy':acc,'train_loss':loss.item(),'grad_norm':norm.item(),'elapsed_sec':time.perf_counter()-start,'examples':16,'split':'train_only','purpose':'input/gradient plumbing, not held-out learning evidence'}
        print(result,flush=True); (ROOT/'artifacts/overfit_debug.json').write_text(json.dumps(result,indent=2))
        if acc==1. and loss.item()<.05: break
