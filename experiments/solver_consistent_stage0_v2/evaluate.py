import json, time
import numpy as np
import torch
from runtime import *

def probabilities(z):
    z=z-z.max(-1,keepdims=True); p=np.exp(z); return p/p.sum(-1,keepdims=True)

def metrics(z,ref,y):
    p=probabilities(z.astype(np.float64)); q=probabilities(ref.astype(np.float64)); pred=p.argmax(-1); rpred=q.argmax(-1); mix=(p+q)/2
    js=.5*np.sum(p*np.log(np.maximum(p,1e-300)/mix)+q*np.log(np.maximum(q,1e-300)/mix),axis=-1)
    return {'accuracy':float((pred==y).mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-300)).mean()),
        'brier':float(((p-np.eye(2)[y])**2).sum(-1).mean()),'flip_rate_vs_8':float((pred!=rpred).mean()),
        'harmful_flip_vs_8':float(((rpred==y)&(pred!=y)).mean()),'recovery_flip_vs_8':float(((rpred!=y)&(pred==y)).mean()),
        'js_vs_8':float(js.mean()),'absolute_logit_difference_vs_8':float(np.abs(z-ref).mean()),
        'confidence_difference_vs_8':float(np.abs(p.max(-1)-q.max(-1)).mean()),'mean_confidence':float(p.max(-1).mean())}

def configurations(name,c):
    primary='euler' if name=='vector_field' else 'loop'
    jobs=[(k,primary,'uniform') for k in c['eval_budgets']]
    if name!='vanilla_loop': jobs += [(k,primary,s) for k in c['eval_budgets'] for s in ['front-loaded','back-loaded']]
    if name=='vector_field': jobs += [(k,s,'uniform') for k in [4,8,12,16] for s in ['heun','rk4']]
    return [(0,'bypass','none')]+jobs

@torch.no_grad()
def predict(m,data,c,budget,solver,sched,diagnostic_indices):
    n=len(data['labels']); z=np.empty((n,2),dtype=np.float32); subset=set(diagnostic_indices.tolist()); hidden=[]; hindices=[]; displacement=[]; relative_displacement=[]
    full_length=data['tokens'].shape[1]
    for start in range(0,n,c['eval_batch']):
        ix=np.arange(start,min(start+c['eval_batch'],n)); x,y=batch(data,ix,device())
        with amp(): logits,h,h0,mask,nfe=m(x,budget,solver if budget else 'euler',sched if budget else 'uniform',True)
        assert nfe==budget
        z[ix]=logits.float().cpu().numpy()
        sel=[i for i,v in enumerate(ix) if v in subset]
        if sel:
            v=(h[sel]*mask[sel,:,None]).float(); initial=(h0[sel]*mask[sel,:,None]).float()
            delta=(v-initial).flatten(1).norm(dim=-1); den=initial.flatten(1).norm(dim=-1)
            displacement.extend(delta.cpu().tolist()); relative_displacement.extend((delta/(den+1e-8)).cpu().tolist())
            padded=np.zeros((len(sel),full_length,c['d_model']),np.float32)
            padded[:,:v.shape[1]]=v.cpu().numpy(); hidden.append(padded); hindices.extend(ix[sel].tolist())
    return z,np.concatenate(hidden),np.array(hindices),np.array(displacement),np.array(relative_displacement)

def evaluate(name):
    m,c,saved=checkpoint_model(name)
    output=ROOT/'artifacts'/name/'evaluation'; output.mkdir(parents=True,exist_ok=True)
    results=[]; schedules=[]
    from solvers.fixed_step import schedule
    for k,sol,sk in configurations(name,c):
        factor={'euler':1,'heun':2,'rk4':4}.get(sol,1)
        schedules.append({'model':name,'budget':k,'solver':sol,'schedule':sk,'dts':None if name=='vanilla_loop' else schedule(k//factor,sk) if k else []})
    (output/'schedules.json').write_text(json.dumps(schedules,indent=2))
    for split in ['iid','ood_6','ood_8','ood_12']:
        data=load_data(split); n=len(data['labels'])
        # 128 per split, 512 total, selected before looking at outputs; stratify IID depth.
        rng=np.random.default_rng(8800)
        diagnostic=np.sort(np.concatenate([rng.choice(np.flatnonzero(data['depths']==d),128//len(np.unique(data['depths'])),replace=False) for d in np.unique(data['depths'])]))
        ref,href,hi,_,_=predict(m,data,c,8,'euler','uniform',diagnostic)
        for budget,solver,sk in configurations(name,c):
            tag=f'{split}_b{budget}_{solver}_{sk}'
            path=output/(tag+'.npz')
            if path.exists():
                old=np.load(path); z=old['logits']; hdist=old['hidden_distance']; disp=old['displacement']; rdisp=old['relative_displacement']
            else:
                z,h,ids,disp,rdisp=predict(m,data,c,budget,solver,sk,diagnostic)
                assert np.array_equal(ids,hi)
                hdist=np.linalg.norm((h-href).reshape(len(h),-1),axis=-1)/(np.linalg.norm(href.reshape(len(h),-1),axis=-1)+1e-8)
                np.savez_compressed(path,logits=z,labels=data['labels'],depths=data['depths'],diagnostic_indices=hi,
                    final_hidden=h.astype(np.float16),hidden_distance=hdist,displacement=disp,relative_displacement=rdisp)
                del h
            for depth in ['all']+np.unique(data['depths']).tolist():
                pick=np.ones(n,dtype=bool) if depth=='all' else data['depths']==depth
                dpick=np.ones(len(hi),dtype=bool) if depth=='all' else data['depths'][hi]==depth
                row={'model':name,'seed':0,'split':split,'reasoning_depth':depth,'budget':budget,'nfe':budget,'solver':solver,'schedule':sk,
                    'examples':int(pick.sum()),'checkpoint_update':saved['update'],**metrics(z[pick],ref[pick],data['labels'][pick]),
                    'hidden_endpoint_distance_vs_8':float(hdist[dpick].mean()),'hidden_diagnostic_examples':int(dpick.sum()),
                    'state_displacement_norm':float(disp[dpick].mean()),'state_relative_displacement':float(rdisp[dpick].mean())}
                truncated=pick & data['truncated']; untruncated=pick & ~data['truncated']
                row['truncated_examples']=int(truncated.sum())
                for group,selection in [('truncated',truncated),('untruncated',untruncated)]:
                    d=metrics(z[selection],ref[selection],data['labels'][selection]) if selection.any() else {}
                    row['accuracy_'+group]=d.get('accuracy','NA'); row['harmful_flip_'+group]=d.get('harmful_flip_vs_8','NA')
                results.append(row)
            print('eval',name,tag,'acc',results[-(len(np.unique(data['depths']))+1)]['accuracy'],flush=True)
            (output/'metrics.json').write_text(json.dumps(results,indent=2))
        del href
    del m; torch.cuda.empty_cache()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=NAMES,required=True); evaluate(p.parse_args().model)
