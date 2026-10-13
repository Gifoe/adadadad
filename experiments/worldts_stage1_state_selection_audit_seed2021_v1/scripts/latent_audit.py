"""Fixed-origin latent evaluation and cross-epoch geometry, without Test data."""
import argparse, json, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *

COMPONENTS=['loss','mse_loss','cosine_loss','sigreg_loss',
            'sigreg_pred_std_loss','sigreg_target_std_loss',
            'sigreg_pred_cov_loss','sigreg_target_cov_loss',
            'h_pred_std_mean','h_target_std_mean']

def get_model(ds,ep):
    cfg,_,_,_=config(ds)
    saved=torch.load(ROOT/'outputs/checkpoints'/ds/f'replay_stage1_epoch{ep}.pt',
                     map_location='cpu',weights_only=False)
    assert saved['epoch']==ep and saved['dataset']==ds and saved['identity'].startswith('raw completed epoch')
    model=WorldTSModel(cfg);model.load_state_dict(saved['Model'],strict=True)
    model=model.cuda().eval().requires_grad_(False)
    return model,cfg,saved

def one_group(model,crit,d,origin_list):
    rows=[];sample_l1=[];sample_cos=[]
    with torch.no_grad():
        for start in range(0,len(origin_list),32):
            chosen=origin_list[start:start+32]
            x,cf,y=batch(d,chosen)
            out=forward(model,x,cf,y)
            loss=crit(out['h_pred'],out['h_target'],out.get('z_visreg_hist'),
                      out.get('z_visreg_target'),out.get('z_visreg_pred'))
            row={key:float(loss[key].detach().cpu()) for key in COMPONENTS}
            row['no_regularizer']=float(crit.latent_mse_weight*loss['mse_loss']+
                                         crit.latent_cosine_weight*loss['cosine_loss'])
            row['regularizer_contribution']=float(crit.latent_sigreg_weight*loss['sigreg_loss'])
            rows.append(row)
            per_l1=(out['h_pred']-out['h_target']).abs().mean(dim=(1,2))
            per_cos=1-torch.nn.functional.cosine_similarity(out['h_pred'].flatten(1),
                                                            out['h_target'].flatten(1),dim=1)
            sample_l1.extend(per_l1.cpu().numpy().tolist());sample_cos.extend(per_cos.cpu().numpy().tolist())
    result={key:float(np.mean([row[key] for row in rows])) for key in rows[0]}
    result['n_origins']=len(origin_list);result['n_batches']=len(rows)
    result['l1_equal_sample_weight']=float(np.mean(sample_l1))
    result['cosine_equal_sample_weight']=float(np.mean(sample_cos))
    result['n_regrouped_tail']=len(origin_list)%32
    assert abs(result['loss']-(crit.latent_mse_weight*result['mse_loss']+
                               crit.latent_cosine_weight*result['cosine_loss']+
                               crit.latent_sigreg_weight*result['sigreg_loss']))<5e-5
    return result

def latent_metrics(ds):
    d=data(ds);all_val=origins(d,'val');all_train=origins(d,'train')
    assert len(all_val)==4883 and len(all_train)==35611
    probe=all_train[np.linspace(0,len(all_train)-1,4883,dtype=np.int64)]
    assert len(np.unique(probe))==4883
    all_rows=[];sensitivity=[]
    for ep in range(1,7):
        model,cfg,saved=get_model(ds,ep);crit=criterion(cfg)
        measurements={}
        for name,oo in [('validation_chronological',all_val),('train_probe_chronological',probe)]:
            measurements[name]=one_group(model,crit,d,oo)
            all_rows.append({'dataset':ds,'stage1_epoch':ep,'domain':name,
                             'partition_seed':'chronological',**measurements[name],
                             'snapshot_tensor_hash':saved['run_records'][-1]['raw_model_tensor_hash']})
        for seed_value in PROTOCOL['partition_seeds']:
            order=np.random.default_rng(seed_value).permutation(all_val)
            result=one_group(model,crit,d,order)
            sensitivity.append({'dataset':ds,'stage1_epoch':ep,'partition_seed':seed_value,
                                **result})
        all_rows[-2]['matched_train_probe_gap']=measurements['validation_chronological']['loss']-measurements['train_probe_chronological']['loss']
        print('LATENT_DONE',ds,ep,'val',measurements['validation_chronological']['loss'],
              'trainprobe',measurements['train_probe_chronological']['loss'],flush=True)
        write_csv(ROOT/'outputs'/f'stage1_epoch_metrics_{ds}.csv',all_rows)
        write_csv(ROOT/'outputs'/f'batch_partition_sensitivity_{ds}.csv',sensitivity)
    return all_rows,sensitivity

def flatten_tokens(x):return np.asarray(x,dtype=np.float64).reshape(-1,x.shape[-1])

def spectrum_metrics(array):
    matrix=flatten_tokens(array);centered=matrix-matrix.mean(0,keepdims=True)
    singular=np.linalg.svd(centered,full_matrices=False,compute_uv=False)
    probs=singular/(singular.sum()+1e-30)
    effective=float(np.exp(-np.sum(probs*np.log(probs+1e-30))))
    covariance=singular**2
    participation=float(covariance.sum()**2/(np.square(covariance).sum()+1e-30))
    return {'mean':float(matrix.mean()),'std':float(matrix.std()),
            'mean_token_l2':float(np.linalg.norm(matrix,axis=1).mean()),
            'effective_rank_singular_entropy':effective,
            'covariance_participation_ratio':participation,
            'top_singular_values_json':json.dumps(singular[:min(20,len(singular))].tolist()),
            'feature_variances_json':json.dumps(centered.var(0).tolist()),
            'n_tokens':len(matrix),'d':matrix.shape[1]}

def cka(x,y,centered):
    x=flatten_tokens(x);y=flatten_tokens(y)
    if centered:x=x-x.mean(0,keepdims=True);y=y-y.mean(0,keepdims=True)
    xy=x.T@y;xx=x.T@x;yy=y.T@y
    return float(np.square(xy).sum()/(np.linalg.norm(xx)*np.linalg.norm(yy)+1e-30))

def procrustes(train_source,train_reference,val_source,val_reference):
    a=flatten_tokens(train_source);b=flatten_tokens(train_reference)
    av=flatten_tokens(val_source);bv=flatten_tokens(val_reference)
    am=a.mean(0,keepdims=True);bm=b.mean(0,keepdims=True)
    u,_,vt=np.linalg.svd((a-am).T@(b-bm),full_matrices=False)
    rotation=u@vt
    before=np.mean(np.square((av-am)-(bv-bm)))
    after=np.mean(np.square((av-am)@rotation-(bv-bm)))
    reference_variance=np.mean(np.square(bv-bm))
    return {'coordinate_mse_train_registered_val_before':float(before),
            'coordinate_mse_train_registered_val_after':float(after),
            'registered_val_mse_over_epoch1_val_variance':float(after/(reference_variance+1e-30))}

def collect_states(model,d,oo):
    buffer={'h_hist':[],'h_target':[],'h_pred':[]}
    with torch.no_grad():
        for start in range(0,len(oo),4):
            x,cf,y=batch(d,oo[start:start+4]);out=forward(model,x,cf,y)
            for key in buffer:buffer[key].append(out[key].detach().cpu().numpy())
    return {key:np.concatenate(values) for key,values in buffer.items()}

def geometry(ds):
    d=data(ds);train=origins(d,'train');valid=origins(d,'val')
    chosen={'train':train[np.linspace(0,len(train)-1,16,dtype=np.int64)],
            'val':valid[np.linspace(0,len(valid)-1,16,dtype=np.int64)]}
    states={};table=[];alignment=[]
    for ep in [1,3,6]:
        model,_,_=get_model(ds,ep)
        states[ep]={scope:collect_states(model,d,oo) for scope,oo in chosen.items()}
        for scope,blobs in states[ep].items():
            p=blobs['h_pred'].reshape(16,-1);t=blobs['h_target'].reshape(16,-1)
            cosine=np.sum(p*t,axis=1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(t,axis=1)+1e-30)
            for part,values in blobs.items():
                table.append({'dataset':ds,'stage1_epoch':ep,'scope':scope,'state':part,
                              **spectrum_metrics(values),
                              'pred_target_cosine_mean':float(np.mean(cosine)) if part=='h_pred' else None})
        print('GEOMETRY_CAPTURED',ds,ep,flush=True)
    for ep in [3,6]:
        for key in ['h_hist','h_target','h_pred']:
            row={'dataset':ds,'stage1_epoch':ep,'reference_epoch':1,'state':key,
                 'centered_linear_cka_train':cka(states[ep]['train'][key],states[1]['train'][key],True),
                 'centered_linear_cka_val':cka(states[ep]['val'][key],states[1]['val'][key],True),
                 'uncentered_kernel_alignment_train':cka(states[ep]['train'][key],states[1]['train'][key],False),
                 'uncentered_kernel_alignment_val':cka(states[ep]['val'][key],states[1]['val'][key],False)}
            row.update(procrustes(states[ep]['train'][key],states[1]['train'][key],
                                   states[ep]['val'][key],states[1]['val'][key]))
            alignment.append(row)
    write_csv(ROOT/'outputs'/f'latent_geometry_{ds}.csv',table)
    write_csv(ROOT/'outputs'/f'latent_geometry_alignment_{ds}.csv',alignment)
    return table,alignment

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NP','PJM'],required=True)
    a=p.parse_args();latent_metrics(a.dataset);geometry(a.dataset)
