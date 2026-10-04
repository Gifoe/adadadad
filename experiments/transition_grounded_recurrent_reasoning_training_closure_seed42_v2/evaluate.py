import json,csv,time,pathlib
import numpy as np
import torch
from run_experiment import ROOT,OUT,CK,CONFIG,batch,load_data,status,dump
from model_extension import make_model

def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_model(name):
    m=make_model(218).cuda();ck=torch.load(CK/(name+'_final.pt'),weights_only=True,map_location='cuda');m.load_state_dict(ck['model_state_dict']);m.checkpoint_step=ck['step'];m.eval();return m
@torch.no_grad()
def sweep(name,model,test):
    final=[];states=[];over=[];frontier=[]
    model.num_iterations=24
    for d in CONFIG['D']:
        indices=np.flatnonzero(test['hops']==d);pred=[]
        if d<=5:
            independent=set(json.loads((ROOT/'data/closure_split_manifest.json').read_text())['test_indices'])
            indices=np.asarray([i for i in indices if int(i) in independent])
        for s in range(0,len(indices),128):
            x,mask,p,t=batch(test,indices[s:s+128])
            with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x,mask,p,True)
            pred.append(torch.stack([z.argmax(-1) for z in o.state_logits_per_recurrence],1).cpu().numpy())
        pred=np.concatenate(pred);target=test['state_targets'][indices];good=pred==target;final_good=pred==target[:,[-1]]
        np.savez_compressed(OUT/f'rollout_{name}_D{d}.npz',predictions=pred,targets=target,sample_ids=test['sample_ids'][indices])
        acc={k:float(final_good[:,k-1].mean()) for k in CONFIG['K']}
        for K in CONFIG['K']:
            final.append(dict(model=name,D=d,K=K,correct=int(final_good[:,K-1].sum()),total=len(indices),accuracy=acc[K],compute_regime='under' if K<d else 'matched' if K==d else 'over'))
            for k in range(1,K+1):states.append(dict(model=name,D=d,K=K,k=k,correct=int(good[:,k-1].sum()),total=len(indices),accuracy=float(good[:,k-1].mean()),state_type='transition' if k<=d else 'terminal'))
        early=final_good[:,np.asarray([k-1 for k in CONFIG['K'] if k<24])].any(1)
        flips=early&~final_good[:,23];stable=final_good[:,d:].all(1)
        peak=max(acc.values());early_total=int(early.sum())
        over.append(dict(model=name,D=d,peak_accuracy=peak,K24_accuracy=acc[24],overthinking_drop=peak-acc[24],early_correct_count=early_total,flip_count=int(flips.sum()),answer_flip_rate=float(flips.sum()/early_total) if early_total else None,terminal_stable_correct=int(stable.sum()),terminal_stability=float(stable.mean()),total=len(indices)))
        star=min(k for k in CONFIG['K'] if acc[k]>=.95*peak)
        frontier.append(dict(model=name,D=d,K_star=star,peak_accuracy=peak,accuracy_at_frontier=acc[star]))
        status('SWEEP_'+name,D=d,matched_accuracy=acc[d],peak=peak)
    csvwrite(OUT/f'final_accuracy_matrix_{name}.csv',final);csvwrite(OUT/f'state_accuracy_{name}.csv',states)
    return over,frontier
@torch.no_grad()
def prototypes(name,model,train):
    model.num_iterations=4;sums=torch.zeros(2,201,768,device='cuda');counts=torch.zeros(2,201,device='cuda',dtype=torch.long)
    for s in range(0,len(train['hops']),128):
        ind=np.arange(s,min(s+128,len(train['hops'])));x,mask,p,t=batch(train,ind)
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x,mask,p,True)
        for index,k in enumerate([2,4]):
            label=t[:,k-1];sums[index].index_add_(0,label,o.state_hidden_per_recurrence[k-1].float());counts[index].index_add_(0,label,torch.ones_like(label))
    mu=sums/counts.clamp_min(1).unsqueeze(-1)
    torch.save(dict(mu=mu.cpu(),counts=counts.cpu(),carrier='raw hidden before final LayerNorm',source='train only'),OUT/f'prototypes_{name}.pt')
    csvwrite(OUT/f'prototype_counts_{name}.csv',[dict(model=name,k=k,entity=f'<e_{e-1}>',token_id=e,count=int(counts[i,e]),eligible=bool(counts[i,e]>=10)) for i,k in enumerate([2,4]) for e in range(1,201)])
    return mu,counts
def intervention_plan(test,counts):
    rng=np.random.RandomState(42);records=[]
    vocab=json.loads((ROOT/'data/vocab.json').read_text());ids={t:i for i,t in enumerate(vocab)}
    raw=json.loads((ROOT/'data/generated/test.json').read_text());table=json.loads((ROOT/'data/transition_table.json').read_text())
    for d in [6,8,10,12,16,20]:
        for k in ([2] if d==6 else [2,4]):
            pool=np.flatnonzero(test['hops']==d)
            eligible=[int(i) for i in pool if counts[[2,4].index(k),test['state_targets'][i,k-1]]>=10]
            assert len(eligible)>=200
            chosen=sorted(rng.choice(eligible,200,replace=False).tolist(),key=lambda i:str(test['sample_ids'][i]))
            for i in chosen:
                e_orig=int(test['state_targets'][i,k-1]);options=[e for e in range(1,201) if e!=e_orig and counts[[2,4].index(k),e]>=10]
                e_cf=int(rng.choice(options));e=vocab[e_cf];path=[]
                for r in raw[i]['relations'][k:]:e=table[e+':'+r];path.append(ids[e])
                assert path[-1]!=test['state_targets'][i,-1]
                records.append(dict(index=i,sample_id=str(test['sample_ids'][i]),D=d,k=k,e_original=e_orig,e_cf=e_cf,y_original=int(test['state_targets'][i,-1]),y_cf=path[-1],cf_path=path))
    dump(OUT/'intervention_sample_plan.json',records)
    return records
@torch.no_grad()
def intervene(name,model,test,mu,plan):
    results=[];rng=np.random.RandomState(42)
    for d in [6,8,10,12,16,20]:
        for k in ([2] if d==6 else [2,4]):
            group=[r for r in plan if r['D']==d and r['k']==k];extra=min(d+4,24);model.num_iterations=extra
            for s in range(0,len(group),64):
                rows=group[s:s+64];ind=np.asarray([r['index'] for r in rows]);x,mask,p,t=batch(test,ind)
                with torch.autocast('cuda',dtype=torch.bfloat16):normal=model(x,mask,p,True)
                original_predictions=torch.stack([z.argmax(-1) for z in normal.state_logits_per_recurrence],1).cpu().numpy()
                original_q=normal.state_hidden_per_recurrence[k-1].float();qnorm=original_q.norm(dim=-1).cpu().numpy()
                cf=mu[[2,4].index(k),[r['e_cf'] for r in rows]];same=mu[[2,4].index(k),[r['e_original'] for r in rows]]
                noise=torch.as_tensor(rng.normal(size=(len(rows),768)),device='cuda',dtype=torch.float32)
                noise=noise/noise.norm(dim=-1,keepdim=True)*cf.norm(dim=-1,keepdim=True)
                patches={'counterfactual':cf,'same_state':same,'noise':noise}
                for control,q in patches.items():
                    with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x,mask,p,True,patch=(k,q))
                    pred=torch.stack([z.argmax(-1) for z in o.state_logits_per_recurrence],1).cpu().numpy();norm=q.norm(dim=-1).cpu().numpy()
                    for j,r in enumerate(rows):
                        traj=pred[j,k:d];cfcorrect=int((traj==np.asarray(r['cf_path'])).sum())
                        for K in [d,extra]:
                            original=int(original_predictions[j,K-1]);patched=int(pred[j,K-1])
                            results.append(dict(model=name,sample_id=r['sample_id'],D=d,k=k,K=K,control=control,e_original=r['e_original'],e_cf=r['e_cf'],y_original=r['y_original'],y_cf=r['y_cf'],original_prediction=original,unperturbed_correct=int(original==r['y_original']),patched_prediction=patched,cf_target=int(patched==r['y_cf']),original_retention=int(patched==r['y_original']),same_prediction_preserved=int(patched==original),cf_trajectory_correct=cfcorrect,cf_trajectory_total=d-k,cf_trajectory_accuracy=cfcorrect/(d-k),original_state_norm=float(qnorm[j]),patched_state_norm=float(norm[j]),start_entity=int(test['input_ids'][r['index'],0]),relations=json.dumps(test['input_ids'][r['index'],1:d+1].tolist()),true_path=json.dumps(test['state_targets'][r['index'],:d].tolist()),expected_cf_path=json.dumps(r['cf_path']),observed_path=json.dumps(pred[j,:K].tolist()),normal_path=json.dumps(original_predictions[j,:K].tolist())))
            status('INTERVENTION_'+name,D=d,k=k,n=len(group))
    csvwrite(OUT/f'intervention_results_{name}.csv',results)
    return results
def summarize_interventions(results):
    summaries=[]
    for name in ['baseline','grounded']:
        for d in [6,8,10,12,16,20]:
            for k in ([2] if d==6 else [2,4]):
                for K in [d,min(d+4,24)]:
                    for control in ['counterfactual','same_state','noise']:
                        g=[r for r in results if r['model']==name and r['D']==d and r['k']==k and r['K']==K and r['control']==control]
                        for stratum in ['all','original_correct','original_incorrect']:
                            rows=g if stratum=='all' else [r for r in g if bool(r['unperturbed_correct'])==(stratum=='original_correct')]
                            n=len(rows);trajtotal=sum(r['cf_trajectory_total'] for r in rows)
                            summaries.append(dict(model=name,D=d,k=k,K=K,control=control,stratum=stratum,n=n,cf_target_count=sum(r['cf_target'] for r in rows),cf_target_rate=sum(r['cf_target'] for r in rows)/n if n else None,original_retention_count=sum(r['original_retention'] for r in rows),original_retention_rate=sum(r['original_retention'] for r in rows)/n if n else None,preservation_count=sum(r['same_prediction_preserved'] for r in rows),preservation_rate=sum(r['same_prediction_preserved'] for r in rows)/n if n else None,cf_trajectory_correct=sum(r['cf_trajectory_correct'] for r in rows),cf_trajectory_total=trajtotal,cf_trajectory_accuracy=sum(r['cf_trajectory_correct'] for r in rows)/trajtotal if trajtotal else None))
    csvwrite(OUT/'intervention_summary.csv',summaries)
    # Representative examples: lexicographically first sample per D,k,model at K=D.
    chosen=[]
    for name in ['baseline','grounded']:
        for d in [6,8,20]:
            candidates=[r for r in results if r['model']==name and r['D']==d and r['k']==2 and r['K']==d and r['control']=='counterfactual']
            chosen.append(min(candidates,key=lambda r:r['sample_id']))
    dump(OUT/'representative_examples.json',dict(selection_rule='lexicographically first eligible sampled ID for D6,8,20; k2 K=D; both arms; independent of predictions',examples=chosen))
def evaluate_all():
    start=time.perf_counter();train,test=load_data();over=[];frontier=[];results=[];shared_plan=None
    for name in ['baseline','grounded']:
        model=load_model(name);a,b=sweep(name,model,test);over.extend(a);frontier.extend(b)
        mu,counts=prototypes(name,model,train)
        if shared_plan is None:shared_plan=intervention_plan(test,counts.cpu().numpy())
        else:
            for r in shared_plan:assert counts[[2,4].index(r['k']),r['e_cf']]>=10 and counts[[2,4].index(r['k']),r['e_original']]>=10
        results.extend(intervene(name,model,test,mu,shared_plan));del model;torch.cuda.empty_cache()
    csvwrite(OUT/'overthinking_metrics.csv',over);csvwrite(OUT/'computational_frontier.csv',frontier);summarize_interventions(results)
    seconds=time.perf_counter()-start
    dump(OUT/'evaluation_runtime.json',dict(evaluation_seconds=seconds))
    runtime=json.loads((OUT/'runtime_metrics.json').read_text()) if (OUT/'runtime_metrics.json').exists() else dict(total_gpu_hours=0.)
    runtime['evaluation_seconds']=seconds;runtime['total_gpu_hours']+=seconds/3600;dump(OUT/'runtime_metrics.json',runtime)
