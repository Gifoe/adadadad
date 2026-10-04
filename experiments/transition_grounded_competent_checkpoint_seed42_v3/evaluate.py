import json,pathlib,time,shutil
import numpy as np
import torch
from run_experiment import ROOT,OUT,DATA,CONFIG,load_model,batch,dump,csvwrite,status,sha

@torch.no_grad()
def sweep(name,m,test):
 m.eval();m.num_iterations=24;matrix=[];state=[];over=[];front=[];carrier_files=[]
 for d in CONFIG['D']:
  ix=np.flatnonzero(test['hops']==d);pred=[];raw=[];readout=[]
  for s in range(0,len(ix),128):
   x,mask,t=batch(test,ix[s:s+128])
   with torch.autocast('cuda',dtype=torch.bfloat16):o=m(x,mask,True)
   pred.append(torch.stack([z.argmax(-1) for z in o.state_logits],1).cpu().numpy())
   if name=='pretrained':
    raw.append(torch.stack(o.states,1).float().cpu().numpy());readout.append(torch.stack(o.state_logits,1).float().cpu().numpy())
   del o
  pred=np.concatenate(pred);targets=test['state_targets'][ix];good=pred==targets;final_good=pred==targets[:,[-1]]
  np.savez_compressed(OUT/f'rollout_{name}_D{d}.npz',predictions=pred,targets=targets,sample_ids=test['sample_ids'][ix])
  if name=='pretrained':
   folder=ROOT/'carriers';folder.mkdir(exist_ok=True);p=folder/f'pretrained_D{d}.npz'
   np.savez_compressed(p,q_raw=np.concatenate(raw),existing_decoder_logits=np.concatenate(readout),sample_ids=test['sample_ids'][ix]);carrier_files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),q='raw existing prediction-position hidden; decoder includes existing final LN'))
  acc={k:float(final_good[:,k-1].mean()) for k in CONFIG['K']}
  for K in CONFIG['K']:matrix.append(dict(model=name,D=d,K=K,split='ID' if d<=CONFIG['H_train'] else 'OOD',correct=int(final_good[:,K-1].sum()),total=len(ix),accuracy=acc[K]))
  for k in range(1,25):state.append(dict(model=name,D=d,k=k,correct=int(good[:,k-1].sum()),total=len(ix),accuracy=float(good[:,k-1].mean()),target='transition' if k<=d else 'terminal'))
  early=final_good[:,np.array([k-1 for k in CONFIG['K'] if k<24])].any(1);flips=early&~final_good[:,23];peak=max(acc.values());stable=final_good[:,d:].all(1)
  over.append(dict(model=name,D=d,split='ID' if d<=CONFIG['H_train'] else 'OOD',peak_accuracy=peak,K24_accuracy=acc[24],overthinking_drop=peak-acc[24],early_correct=int(early.sum()),correct_to_wrong_count=int(flips.sum()),correct_to_wrong_flip_rate=float(flips.sum()/early.sum()) if early.any() else None,terminal_stability=float(stable.mean()),total=len(ix)))
  star=min(k for k in CONFIG['K'] if acc[k]>=.95*peak);front.append(dict(model=name,D=d,K_star=star,peak_accuracy=peak))
  status('SWEEP_'+name,D=d,K5=acc[5],peak=peak)
 matrix_name='pretrained_accuracy_matrix.csv' if name=='pretrained' else f'accuracy_matrix_{name}.csv'
 csvwrite(OUT/matrix_name,matrix);csvwrite(OUT/f'ood_accuracy_{name}.csv',[r for r in matrix if r['split']=='OOD']);csvwrite(OUT/f'state_accuracy_{name}.csv',state);csvwrite(OUT/f'overthinking_{name}.csv',over);csvwrite(OUT/f'frontier_{name}.csv',front)
 if name=='pretrained':shutil.copy2(OUT/'state_accuracy_pretrained.csv',OUT/'pretrained_state_accuracy.csv');dump(OUT/'pretrained_carrier_manifest.json',carrier_files)
 return matrix,state,over,front
@torch.no_grad()
def prototypes(name,m,train):
 m.eval();m.num_iterations=4;sums=torch.zeros(2,217,768,device='cuda');counts=torch.zeros(2,217,dtype=torch.long,device='cuda')
 for s in range(0,len(train['hops']),128):
  ix=np.arange(s,min(s+128,len(train['hops'])));x,mask,t=batch(train,ix)
  with torch.autocast('cuda',dtype=torch.bfloat16):o=m(x,mask,True)
  for j,k in enumerate([2,4]):
   labels=t[:,k-1];sums[j].index_add_(0,labels,o.states[k-1].float());counts[j].index_add_(0,labels,torch.ones_like(labels))
  if s%(128*128)==0:status('PROTOTYPES_'+name,records=s+len(ix),total=len(train['hops']))
 mu=sums/counts.clamp_min(1).unsqueeze(-1)
 torch.save(dict(mu=mu.cpu(),counts=counts.cpu(),source='same official TRAIN subset only',carrier='raw last-padded-position hidden before existing final LN'),OUT/f'prototypes_{name}.pt')
 csvwrite(OUT/f'prototype_counts_{name}.csv',[dict(model=name,k=k,entity_token_id=e,count=int(counts[j,e]),eligible=bool(counts[j,e]>=10)) for j,k in enumerate([2,4]) for e in range(1,201)])
 return mu,counts
def plan(test,counts):
 p=OUT/'intervention_plan.json'
 if p.exists():return json.loads(p.read_text())
 rng=np.random.RandomState(42);records=json.loads((DATA/'test_records.json').read_text());vocab=json.loads((DATA/'vocab.json').read_text());ids={t:i for i,t in enumerate(vocab)};table=json.loads((DATA/'transition_table.json').read_text());rows=[]
 for d in CONFIG['D']:
  for k in [2,4]:
   if d<=k:continue
   pool=np.flatnonzero(test['hops']==d);eligible=[int(i) for i in pool if counts[[2,4].index(k),test['state_targets'][i,k-1]]>=10]
   assert len(eligible)>=200
   chosen=sorted(rng.choice(eligible,200,replace=False).tolist(),key=lambda i:test['sample_ids'][i])
   for i in chosen:
    original=int(test['state_targets'][i,k-1]);options=[e for e in range(1,201) if e!=original and counts[[2,4].index(k),e]>=10];ecf=int(rng.choice(options));e=vocab[ecf];path=[]
    for rel in records[i]['relations'][k:]:e=table[e+':'+rel];path.append(ids[e])
    assert path and path[-1]!=int(test['state_targets'][i,-1])
    rows.append(dict(index=i,sample_id=str(test['sample_ids'][i]),D=d,k=k,e_original=original,e_cf=ecf,y_original=int(test['state_targets'][i,-1]),y_cf=path[-1],cf_path=path,K_main=max(d,CONFIG['H_train']),K_extra=min(d+4,24)))
 dump(p,rows);noise=np.random.RandomState(4242).normal(size=(len(rows),768)).astype(np.float32);noise/=np.linalg.norm(noise,axis=1,keepdims=True);np.save(OUT/'intervention_noise_directions.npy',noise)
 dump(OUT/'intervention_plan_manifest.json',dict(plan_sha256=sha(p),noise_direction_sha256=sha(OUT/'intervention_noise_directions.npy'),n_samples=len(rows),selection='seed42 without replacement200 per eligible D,k; lexicographic ordering independent of predictions',noise='same fixed Gaussian unit directions across all three models; norm matched to each model CF prototype',carrier_only='position49, no other token hidden modified'))
 return rows
@torch.no_grad()
def intervene(name,m,test,mu,counts,rows):
 directions=np.load(OUT/'intervention_noise_directions.npy');results=[]
 for d in CONFIG['D']:
  for k in [2,4]:
   group=[(i,r) for i,r in enumerate(rows) if r['D']==d and r['k']==k]
   if not group:continue
   m.num_iterations=max(group[0][1]['K_main'],group[0][1]['K_extra'])
   for s in range(0,len(group),64):
    part=group[s:s+64];records=[r for _,r in part];ix=np.array([r['index'] for r in records]);x,mask,t=batch(test,ix)
    assert all(int(counts[[2,4].index(k),r['e_cf']])>=10 and int(counts[[2,4].index(k),r['e_original']])>=10 for r in records)
    with torch.autocast('cuda',dtype=torch.bfloat16):normal=m(x,mask,True)
    normal_pred=torch.stack([z.argmax(-1) for z in normal.state_logits],1).cpu().numpy();qnorm=normal.states[k-1].float().norm(dim=-1).cpu().numpy()
    cf=mu[[2,4].index(k),[r['e_cf'] for r in records]];same=mu[[2,4].index(k),[r['e_original'] for r in records]];noise=torch.as_tensor(directions[[i for i,_ in part]],device='cuda')*cf.norm(dim=-1,keepdim=True)
    del normal
    for control,q in [('counterfactual',cf),('same_state',same),('noise',noise)]:
     with torch.autocast('cuda',dtype=torch.bfloat16):out=m(x,mask,True,patch=(k,q))
     pred=torch.stack([z.argmax(-1) for z in out.state_logits],1).cpu().numpy();norm=q.norm(dim=-1).cpu().numpy();del out
     for j,r in enumerate(records):
      correct=int((pred[j,k:d]==np.array(r['cf_path'])).sum())
      for regime,K in [('main',r['K_main']),('extra',r['K_extra'])]:
       ori=int(normal_pred[j,K-1]);patched=int(pred[j,K-1])
       results.append(dict(model=name,sample_id=r['sample_id'],D=d,k=k,K=K,regime=regime,control=control,e_original=r['e_original'],e_cf=r['e_cf'],y_original=r['y_original'],y_cf=r['y_cf'],unperturbed_prediction=ori,unperturbed_correct=int(ori==r['y_original']),patched_prediction=patched,cf_target=int(patched==r['y_cf']),original_retention=int(patched==r['y_original']),prediction_preserved=int(patched==ori),cf_trajectory_correct=correct,cf_trajectory_total=d-k,cf_trajectory_accuracy=correct/(d-k),carrier_norm=float(qnorm[j]),patch_norm=float(norm[j]),expected_cf_path=json.dumps(r['cf_path']),observed_path=json.dumps(pred[j,:K].tolist()),unperturbed_path=json.dumps(normal_pred[j,:K].tolist())))
   status('INTERVENTION_'+name,D=d,k=k,n=len(group))
 csvwrite(OUT/f'intervention_{name}.csv',results)
 if name=='pretrained':shutil.copy2(OUT/'intervention_pretrained.csv',OUT/'pretrained_intervention_results.csv')
 summary=[]
 for d in CONFIG['D']:
  for k in [2,4]:
   for regime in ['main','extra']:
    for control in ['counterfactual','same_state','noise']:
     base=[r for r in results if r['D']==d and r['k']==k and r['regime']==regime and r['control']==control]
     if not base:continue
     for stratum in ['all','original_correct','original_incorrect']:
      group=base if stratum=='all' else [r for r in base if bool(r['unperturbed_correct'])==(stratum=='original_correct')];n=len(group);traj=sum(r['cf_trajectory_total'] for r in group)
      summary.append(dict(model=name,D=d,k=k,K=base[0]['K'],regime=regime,control=control,stratum=stratum,n=n,cf_target_count=sum(r['cf_target'] for r in group),cf_target_rate=sum(r['cf_target'] for r in group)/n if n else None,original_retention_rate=sum(r['original_retention'] for r in group)/n if n else None,preservation_rate=sum(r['prediction_preserved'] for r in group)/n if n else None,cf_trajectory_correct=sum(r['cf_trajectory_correct'] for r in group),cf_trajectory_total=traj,cf_trajectory_accuracy=sum(r['cf_trajectory_correct'] for r in group)/traj if traj else None))
 csvwrite(OUT/f'intervention_summary_{name}.csv',summary)
 if name=='pretrained':shutil.copy2(OUT/'intervention_summary_pretrained.csv',OUT/'pretrained_intervention_summary.csv')
 return results,summary
def audit(name,train,test):
 torch.cuda.reset_peak_memory_stats();beg=time.perf_counter();m=load_model(name);sweep_start=time.perf_counter();sweep(name,m,test);sweep_seconds=time.perf_counter()-sweep_start
 prototype_start=time.perf_counter();mu,counts=prototypes(name,m,train);prototype_seconds=time.perf_counter()-prototype_start;rows=plan(test,counts.cpu().numpy())
 cf_start=time.perf_counter();intervene(name,m,test,mu,counts,rows);cf_seconds=time.perf_counter()-cf_start
 dump(OUT/f'{name}_audit_complete.json',dict(model=name,seconds=time.perf_counter()-beg,sweep_seconds=sweep_seconds,prototype_seconds=prototype_seconds,intervention_seconds=cf_seconds,peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,starting_checkpoint_sha256=CONFIG['checkpoint_sha256'] if name=='pretrained' else None))
 del m;torch.cuda.empty_cache();status('AUDIT_COMPLETE_'+name)
