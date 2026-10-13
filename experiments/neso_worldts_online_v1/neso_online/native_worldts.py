"""Train and replay the EXACT official WorldTSModel core on NESO vintage-pair tensors.

This adapter imports native WorldTSModel / LatentStateLoss from the authors' pinned
source, rather than reimplementing their layers. Dataset handling and online
revision are new experimental code; they are not part of the WorldTS paper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
torch.set_num_threads(4)
if os.name=='nt' and (os.cpu_count() or 0)>=32:
    import ctypes
    affinity=ctypes.windll.kernel32.SetProcessAffinityMask
    affinity.argtypes=[ctypes.c_void_p,ctypes.c_size_t];affinity.restype=ctypes.c_int
    assert affinity(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))

def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def tensor_hash(model,part='all'):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        if part=='decoder' and not k.startswith('decoder.'):continue
        if part=='stage1' and k.startswith('decoder.'):continue
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def get_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def atomic_save(payload,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_native(source: str):
    root = Path(source).expanduser().resolve()
    if not (root / 'ts_benchmark/baselines/worldts/models/worldts_model.py').exists():
        raise FileNotFoundError(f'--worldts-source must be .../WorldTS/DAG-bench, got {root}')
    sys.path.insert(0,str(root))
    from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel
    from ts_benchmark.baselines.worldts.losses.latent_state_loss import LatentStateLoss
    return WorldTSModel, LatentStateLoss


def read_arrays(base: Path,splits=('train','val','test')):
    result={}
    for split in splits:
        with np.load(base/f'{split}.npz',allow_pickle=False) as z:result[split]={k:z[k] for k in z.files}
    return result


def train_scaler(data):
    a=data['history']
    if not len(a): raise ValueError('No TRAIN samples; prepare a sufficiently long 2025 archive.')
    # Training HISTORY ONLY: no data from future labels or val/test.
    flat = a.reshape(-1,3).astype('float64')
    return {'mean': flat.mean(axis=0).tolist(),
            'std': np.maximum(flat.std(axis=0),1e-6).tolist()}


def rescale_np(data, scaler):
    mu=np.array(scaler['mean'],np.float32); sd=np.array(scaler['std'],np.float32)
    o={k:v for k,v in data.items()}
    o['history']=((data['history']-mu)/sd).astype('float32')
    o['target']=((data['target']-mu[0])/sd[0]).astype('float32')
    for key in ('cov_a','cov_b'):
        o[key]=((data[key]-mu[1:])/sd[1:]).astype('float32')
    return o


def native_config(model_config: dict, lookback: int, horizon: int):
    x=dict(model_config)
    x.update(seq_len=lookback,pred_len=horizon,horizon=horizon,
             input_dim=3,output_dim=1,series_dim=1,future_x_dim=2)
    return SimpleNamespace(**x)


def as_loader(data, batch_size: int, shuffle:bool, seed:int):
    required=['history','target','cov_a','cov_b','update_mask']
    ds=TensorDataset(*(torch.from_numpy(data[k].astype('float32')) for k in required))
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,drop_last=False,
                      generator=torch.Generator().manual_seed(seed))


def stage1_objective(out, native_loss):
    return native_loss(h_pred=out['h_pred'],h_target=out['h_target'],
                       z_visreg_hist=out.get('z_visreg_hist'),
                       z_visreg_target=out.get('z_visreg_target'),
                       z_visreg_pred=out.get('z_visreg_pred'))['loss']


def target_decode(model,out):
    return model.decode(out['h_target'],target_revin_stats=out['target_revin_stats'],
                        history_tokens=out['h_hist'])


def pred_decode(model,out,latent=None):
    return model.decode(out['h_pred'] if latent is None else latent,
                        target_revin_stats=out['target_revin_stats'],
                        history_tokens=out['h_hist'])


def stage2_objective(model,out,y,rec_weight=.3):
    return F.l1_loss(pred_decode(model,out),y) + rec_weight*F.l1_loss(target_decode(model,out),y)


def evaluate_stage(model,loader,device,stage,native_loss=None):
    model.eval(); errs=[]
    with torch.no_grad():
        for hist,yt,ca,cb,mask in loader:
            hist,yt,ca=hist.to(device),yt.to(device),ca.to(device)
            out=model(input=hist,future_target=yt if stage==1 else None,x_future=ca)
            if stage==1: loss=stage1_objective(out,native_loss)
            else: loss=F.l1_loss(pred_decode(model,out),yt)
            errs.append((float(loss.cpu()),len(hist)))
    return sum(x*n for x,n in errs)/sum(n for x,n in errs)


def fit_two_stage(model,data,loss_obj,*, epochs1,epochs2,patience,lr1,lr2,batch,seed,device,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    ld_train=as_loader(data['train'],batch,True,seed)
    ld_valid=as_loader(data['val'],batch,False,seed)
    if len(ld_valid.dataset)==0: raise ValueError('No validation samples')
    history=[];saved=None
    latest=outdir/'training_resume.pt'
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        model.load_state_dict(saved['model_state'],strict=True);history=saved['history']
    else:
        atomic_save({'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                     'rng':get_rng(),'decoder_tensor_hash':tensor_hash(model,'decoder')},outdir/'untrained_initial.pt')
    for stage,epochs,lr in [(1,epochs1,lr1),(2,epochs2,lr2)]:
        if saved is not None and saved['stage']>stage:continue
        if stage==1:model.configure_for_stage1()
        else:model.configure_for_stage2()
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=lr)
        best=float('inf');best_state=None;bad=0;start_epoch=1;steps=0
        started=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        frozen_part='decoder' if stage==1 else 'stage1'
        frozen_hash=tensor_hash(model,frozen_part)
        if saved is not None and saved['stage']==stage:
            optimizer.load_state_dict(saved['optimizer']);best=saved['best'];best_state=saved['best_state'];bad=saved['bad']
            start_epoch=saved['epoch']+1;steps=saved['optimizer_steps'];started-=saved['stage_seconds']
            frozen_hash=saved['frozen_hash'];ld_train.generator.set_state(saved['train_generator'].cpu())
            ld_valid.generator.set_state(saved['val_generator'].cpu());restore_rng(saved['rng'])
            print('RESUME',stage,start_epoch,steps,flush=True)
        for epoch in range(start_epoch,epochs+1):
            if bad>=patience:break
            model.train()
            if stage==2: model.set_stage1_eval()
            running=0.;cnt=0
            for hist,yt,ca,cb,mask in ld_train:
                hist,yt,ca=hist.to(device),yt.to(device),ca.to(device)
                optimizer.zero_grad(set_to_none=True)
                if stage==1:
                    out=model(input=hist,future_target=yt,x_future=ca)
                    loss=stage1_objective(out,loss_obj)
                else:
                    # Stage 1 frozen, but Decoder needs autograd through the
                    # decoded latent. Target/history states are computed no-grad.
                    with torch.no_grad():
                        out=model(input=hist,future_target=yt,x_future=ca)
                    loss=stage2_objective(model,out,yt)
                if not torch.isfinite(loss): raise ValueError(f'Non-finite loss at stage={stage}, epoch={epoch}')
                loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],max_norm=5.)
                optimizer.step();steps+=1;running+=float(loss.detach().cpu())*len(hist);cnt+=len(hist)
            assert tensor_hash(model,frozen_part)==frozen_hash,'Frozen model tensors changed'
            valid=evaluate_stage(model,ld_valid,device,stage,loss_obj)
            rec={'stage':stage,'epoch':epoch,'train_loss':running/cnt,'val_loss':valid,
                 'optimizer_steps':steps,'learning_rate':lr,'stage_seconds':time.perf_counter()-started,
                 'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated(device)/1048576 if device.type=='cuda' else 0,
                 'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'frozen_part':frozen_part,'frozen_hash':frozen_hash}
            print(json.dumps(rec),flush=True);history.append(rec)
            if valid<best-1e-7:
                best=valid;bad=0
                best_state={k:v.detach().clone().cpu() for k,v in model.state_dict().items()}
                atomic_save({'model_state':best_state,'stage':stage,'epoch':epoch,'val_loss':best,
                             'optimizer':optimizer.state_dict(),'rng':get_rng(),
                             'config':vars(model.config) if hasattr(model,'config') else None},outdir/f'stage{stage}_best.pt')
            else:
                bad+=1
            payload={'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                     'optimizer':optimizer.state_dict(),'rng':get_rng(),'train_generator':ld_train.generator.get_state(),
                     'val_generator':ld_valid.generator.get_state(),'stage':stage,'epoch':epoch,
                     'best':best,'best_state':best_state,'bad':bad,'optimizer_steps':steps,
                     'stage_seconds':rec['stage_seconds'],'history':history,'frozen_hash':frozen_hash,
                     'identity':'raw completed epoch before restoring best'}
            atomic_save(payload,latest);atomic_save(payload,outdir/f'stage{stage}_latest.pt')
            (outdir/'train_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            if bad>=patience:break
        if best_state is None:raise RuntimeError(f'No valid checkpoint in stage {stage}')
        model.load_state_dict(best_state,strict=True)
        saved=None
    return history


class LatentRevisionUpdater(nn.Module):
    """Small, single-step state updater; not a loop, not claimed novel.

    Delta covariates are zero on already-ended settlement periods. Restrict the
    output update to future periods only. Last conv is zero-initialized so the
    initial output is exactly the cached original latent trajectory.
    """
    def __init__(self,latent_dim:int,cov_dim:int=2,hidden:int=64):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(latent_dim+cov_dim+1,hidden,5,padding=2),
                               nn.GELU(),nn.Conv1d(hidden,hidden,5,padding=2),
                               nn.GELU(),nn.Conv1d(hidden,latent_dim,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,z_a,ca,cb,mask):
        if z_a.shape[:2]!=ca.shape[:2] or cb.shape!=ca.shape:
            raise ValueError('Latent/covariate lengths do not match')
        update_mask=mask.float().unsqueeze(-1)
        delta=(cb-ca)*update_mask
        x=torch.cat([z_a,delta,update_mask],dim=-1).transpose(1,2)
        correction=self.net(x).transpose(1,2)*update_mask
        return z_a+correction


def predict_a_b(model,hist,ca,cb,device):
    # At B only the new covariate vintage is made available.
    # The historical target is held fixed at A by dataset design.
    with torch.no_grad():
        oa=model(input=hist.to(device),future_target=None,x_future=ca.to(device))
        old=pred_decode(model,oa)
        ob=model(input=hist.to(device),future_target=None,x_future=cb.to(device))
        recomputed=pred_decode(model,ob)
    return oa,old,recomputed


def train_updater(model, data,*, epochs,patience,learning_rate,batch,seed,device,outdir,hidden=64):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    updater=LatentRevisionUpdater(model.latent_dim,hidden=hidden).to(device)
    optimizer=torch.optim.Adam(updater.parameters(),lr=learning_rate)
    ld_train=as_loader(data['train'],batch,True,seed)
    ld_val=as_loader(data['val'],batch,False,seed)
    best=float('inf');best_state=None;bad=0;hist=[]
    def score(loader,train):
        updater.train(train)
        tot=n=0.
        for x,y,ca,cb,mask in loader:
            x,y,ca,cb,mask=[t.to(device) for t in (x,y,ca,cb,mask)]
            with torch.no_grad():oa=model(input=x,future_target=None,x_future=ca)
            with torch.set_grad_enabled(train):
                z=updater(oa['h_pred'],ca,cb,mask)
                yh=pred_decode(model,oa,z)
                weights=mask.unsqueeze(-1)
                loss=(((yh-y)**2)*weights).sum()/weights.sum().clamp_min(1)
                if train:
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            tot+=float(loss.detach().cpu())*len(x);n+=len(x)
        return tot/max(1,n)
    for e in range(1,epochs+1):
        tr=score(ld_train,True);va=score(ld_val,False)
        rec={'stage':'updater','epoch':e,'train_masked_mse':tr,'val_masked_mse':va}
        hist.append(rec); print(json.dumps(rec))
        if va<best-1e-8:
            best=va;bad=0
            best_state={k:v.detach().clone().cpu() for k,v in updater.state_dict().items()}
            torch.save({'model_state':best_state,'hidden':hidden,'latent_dim':model.latent_dim},outdir/'updater_best.pt')
        else:
            bad+=1
            if bad>=patience:break
    if best_state is None: raise ValueError('Updater never trained')
    updater.load_state_dict(best_state);return updater,hist


def revision_features(ca,cb):
    """Simple non-neural observation-space correction using ONLY revisions.

    No intercept: unchanged covariates necessarily produce zero correction.
    Predictable from input at B, and no future ND is consumed at inference.
    """
    delta=np.asarray(cb)-np.asarray(ca)
    horizon=delta.shape[1]
    lead=np.arange(horizon,dtype=np.float32).reshape(1,horizon,1)/max(1,horizon-1)
    lead=np.broadcast_to(lead,delta.shape[:2]+(1,))
    return np.concatenate([delta,delta*lead,delta*lead**2],axis=-1)


def fit_output_correction(model,train_data,device,batch=64,ridge=1e-2):
    """Fit ridge output-revision baseline on TRAIN windows only."""
    l=as_loader(train_data,batch,False,0)
    Xs=[]; ys=[]
    model.eval()
    for hist,target,ca,cb,mask in l:
        with torch.no_grad():
            out=model(input=hist.to(device),future_target=None,x_future=ca.to(device))
            old=pred_decode(model,out).cpu().numpy()
        features=revision_features(ca.numpy(),cb.numpy())
        selection=mask.numpy().astype(bool)
        Xs.append(features[selection])
        ys.append((target.numpy()-old)[selection].reshape(-1,1))
    if not Xs or not sum(len(x) for x in Xs):
        raise ValueError('No samples for output correction')
    X=np.concatenate(Xs).astype('float64')
    y=np.concatenate(ys).astype('float64')
    w=np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@y)
    return w.reshape(-1).astype(np.float32)


def _masked_metrics(pred,actual,mask):
    d=pred.astype('float64')-actual.astype('float64')
    selected=d[mask.astype(bool)]
    if selected.size==0: return {'mse':None,'mae':None}
    return {'mse':float(np.mean(selected**2)),'mae':float(np.mean(abs(selected)))}


def replay(model,data,scaler,device,updater=None,correction=None,batch=64,save_path=None):
    """Per-update evaluation on SAME target slots after B; no test labels in inference."""
    ld=as_loader(data,batch,False,0)
    scale=float(scaler['std'][0]);mean=float(scaler.get('mean',[0.])[0])
    all_pred={k:[] for k in ('old','recompute','updater','output_correction','seasonal_naive')}
    truth=[];masks=[];measurements={'old_s':0.,'recompute_s':0.,'updater_s':0.,'output_correction_s':0.}
    model.eval()
    if updater is not None:updater.eval()
    for hist,y,ca,cb,mask in ld:
        hist=hist.to(device);ca=ca.to(device);cb=cb.to(device);mask=mask.to(device)
        if hist.shape[1]>=336 and ca.shape[1] <= 336:
            # Same period of the week, 336 half-hour slots earlier.
            left=hist.shape[1]-336
            seasonal=hist[:,left:left+ca.shape[1],0:1].cpu().numpy()
            all_pred['seasonal_naive'].append(seasonal*scale+mean)
        if device.type=='cuda':torch.cuda.synchronize()
        ta=time.perf_counter()
        with torch.no_grad():
            oa=model(input=hist,future_target=None,x_future=ca)
            old=pred_decode(model,oa)
        if device.type=='cuda':torch.cuda.synchronize()
        measurements['old_s']+=time.perf_counter()-ta
        if device.type=='cuda':torch.cuda.synchronize()
        tb=time.perf_counter()
        with torch.no_grad():
            ob=model(input=hist,future_target=None,x_future=cb)
            new=pred_decode(model,ob)
        if device.type=='cuda':torch.cuda.synchronize()
        measurements['recompute_s']+=time.perf_counter()-tb
        if updater is not None:
            if device.type=='cuda':torch.cuda.synchronize()
            tu=time.perf_counter()
            with torch.no_grad():
                z=updater(oa['h_pred'],ca,cb,mask)
                updated=pred_decode(model,oa,z)
            if device.type=='cuda':torch.cuda.synchronize()
            measurements['updater_s']+=time.perf_counter()-tu
            all_pred['updater'].append(updated.cpu().numpy()*scale+mean)
        if correction is not None:
            tc=time.perf_counter()
            features=revision_features(ca.cpu().numpy(),cb.cpu().numpy())
            pred_corrected=old.cpu().numpy()+np.einsum('bhd,d->bh',features,correction)[...,None]
            measurements['output_correction_s']+=time.perf_counter()-tc
            all_pred['output_correction'].append(pred_corrected*scale+mean)
        all_pred['old'].append(old.cpu().numpy()*scale+mean)
        all_pred['recompute'].append(new.cpu().numpy()*scale+mean)
        truth.append(y.numpy()*scale+mean)
        masks.append(mask.cpu().numpy().astype(bool))
    labels=np.concatenate(truth)
    mask=np.concatenate(masks)
    result={'n_windows':int(len(mask)),'n_scored_slots':int(mask.sum()),
            'metric_scope':'only the B-time still-future portion of the A-horizon',
            'normalized_to_training_target_std':False,'metric_units':{'mae':'MW','mse':'MW^2'},
            'methods':{},'elapsed_seconds':measurements}
    predictions={}
    for method,parts in all_pred.items():
        if not parts:continue
        pred=np.concatenate(parts)
        assert np.isfinite(pred).all() and np.isfinite(labels).all()
        predictions[method]=pred
        difference=pred.astype('float64')-labels.astype('float64')
        per_mse=np.sum(difference[:,:,0]**2*mask,axis=1)/np.maximum(mask.sum(axis=1),1)
        per_mae=np.sum(abs(difference[:,:,0])*mask,axis=1)/np.maximum(mask.sum(axis=1),1)
        primary=_masked_metrics(pred,labels,mask)
        result['methods'][method]={**_masked_metrics(pred,labels,mask),
                                   'normalized_mse':primary['mse']/scale**2,
                                   'normalized_mae':primary['mae']/scale,
                                   'mse_per_window':per_mse.tolist(),'mae_per_window':per_mae.tolist()}
    old_error=np.array(result['methods']['old']['mae_per_window'])
    for method,item in result['methods'].items():
        delta=np.array(item['mae_per_window'])-old_error
        item['fraction_windows_mae_improved_vs_old']=float(np.mean(delta< -1e-6))
        item['fraction_windows_mae_worsened_vs_old']=float(np.mean(delta>1e-6))
        item['fraction_windows_mae_tied_vs_old']=float(np.mean(abs(delta)<=1e-6))
        item['mae_gain_vs_old_fraction']=(result['methods']['old']['mae']-item['mae'])/result['methods']['old']['mae']
    if save_path is not None:
        saved={'target_mw':labels,'update_mask':mask,
               **{'pred_'+key+'_mw':value for key,value in predictions.items()}}
        for key in ['issue_a','issue_b','first_target_end','last_target_end','target_end_utc_ns','history_end_utc_ns']:
            if key in data:saved[key]=data[key]
        saved['cov_a_mw']=data['cov_a']*np.asarray(scaler['std'][1:])+np.asarray(scaler['mean'][1:])
        saved['cov_b_mw']=data['cov_b']*np.asarray(scaler['std'][1:])+np.asarray(scaler['mean'][1:])
        np.savez_compressed(save_path,**saved)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','replay','train_updater'])
    p.add_argument('--data',default='outputs/dataset')
    p.add_argument('--out',default='outputs/model')
    p.add_argument('--worldts-source',required=True,help='Path to official WorldTS/DAG-bench directory')
    p.add_argument('--config',default='configs/worldts_neso.json')
    p.add_argument('--seed',type=int,default=2021)
    p.add_argument('--epochs1',type=int,default=20)
    p.add_argument('--epochs2',type=int,default=20)
    p.add_argument('--updater-epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args();set_seed(args.seed)
    root=Path(args.data);outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    if args.action=='replay':
        if (outdir/'test_replay.json').exists():raise RuntimeError('Final Test already completed; never overwrite or retune this locked replay')
        lock_path=outdir/'validation_decision_lock.json'
        if not lock_path.is_file():raise RuntimeError('Final Test is locked until validation decisions and file hashes are frozen')
        lock=json.loads(lock_path.read_text())
        for path,expected in lock['sha256'].items():
            assert file_sha(path)==expected,'Locked file changed: '+path
    if args.action=='train_updater':
        gate=json.loads((outdir/'updater_gate.json').read_text())
        if not gate['allow_updater']:raise RuntimeError('Validation did not justify training the latent updater')
    data=read_arrays(root,('train','val') if args.action!='replay' else ('train','val','test'))
    cfg=json.loads(Path(args.config).read_text())
    lookback=data['train']['history'].shape[1]
    horizon=data['train']['target'].shape[1]
    native_class,loss_class=import_native(args.worldts_source)
    scaler_path=outdir/'train_scaler.json'
    if args.action=='train':
        scaler=train_scaler(data['train'])
        scaler_path.write_text(json.dumps(scaler,indent=2))
    else:
        scaler=json.loads(scaler_path.read_text())
    data={name:rescale_np(x,scaler) for name,x in data.items()}
    dev=torch.device(args.device)
    model=native_class(native_config(cfg,lookback,horizon)).to(dev)
    source_root=Path(args.worldts_source).resolve().parent
    try:
        pinned=subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
    except Exception:
        pinned='UNVERIFIED'
    assert pinned=='269426ea6217d7152a244da0e209d922acd46d4f','Official source commit mismatch'
    provenance={'base_model':'official WorldTSModel (DAG-bench numerical core)',
                'source_directory':str(source_root),'git_commit':pinned,
                'model_config':cfg,'dataset':str(root.resolve()),'seed':args.seed,
                'lookback':lookback,'horizon':horizon,
                'data_assumption':'retrospective ND history availability; forecast generated time treated as available time',
                'model_assessment':'No deployment/live claims from retrospective ND source',
                'action':args.action,'arguments':vars(args),'resolved_native_config':vars(model.config) if hasattr(model,'config') else vars(native_config(cfg,lookback,horizon)),
                'config_sha256':file_sha(args.config),'model_parameters':sum(p.numel() for p in model.parameters()),
                'training_adapter':'NESO adaptation: constant LR, Adam, grad-norm clip5, sample-weighted fixed validation, no drop_last'}
    (outdir/f'run_provenance_{args.action}.json').write_text(json.dumps(provenance,indent=2))
    if args.action=='train':(outdir/'run_provenance.json').write_text(json.dumps(provenance,indent=2))
    if args.action=='train':
        lc={k:cfg[k] for k in ['latent_alignment_loss','latent_mse_weight','latent_cosine_weight',
                               'latent_sigreg_weight','latent_detach_target','sigreg_std_weight',
                               'sigreg_cov_weight','sigreg_eps']}
        criterion=loss_class(**lc).to(dev)
        tr=fit_two_stage(model,data,criterion,epochs1=args.epochs1,epochs2=args.epochs2,
                         patience=args.patience,lr1=cfg['stage1_lr'],lr2=cfg['stage2_lr'],
                         batch=args.batch,seed=args.seed,device=dev,outdir=outdir)
        (outdir/'train_history.json').write_text(json.dumps(tr,indent=2))
        selected=torch.load(outdir/'stage2_best.pt',map_location=dev,weights_only=False)['model_state']
        before=tensor_hash(model);model.load_state_dict(selected,strict=True)
        assert tensor_hash(model)==before,'Fresh best checkpoint reload changed the fitted model'
    else:
        checkpoint=torch.load(outdir/'stage2_best.pt',map_location=dev,weights_only=False)
        model.load_state_dict(checkpoint['model_state'],strict=True)
        model.configure_for_stage2()
    # Observation-space correction is always fitted on TRAIN labels only.
    calibration_file=outdir/'output_correction_weights.json'
    if args.action=='train':
        correction=fit_output_correction(model,data['train'],dev,batch=args.batch)
        calibration_file.write_text(json.dumps({'ridge_weights':correction.tolist(), 'feature_names':['dw','ds','dw_lead','ds_lead','dw_lead2','ds_lead2']},indent=2))
    else:
        correction=np.array(json.loads(calibration_file.read_text())['ridge_weights'],dtype=np.float32)
    updater=None
    if args.action=='train_updater':
        updater, history=train_updater(model,data,epochs=args.updater_epochs,patience=args.patience,
                                        learning_rate=cfg['updater_lr'],batch=args.batch,seed=args.seed,
                                        device=dev,outdir=outdir,hidden=cfg['updater_hidden'])
        (outdir/'updater_history.json').write_text(json.dumps(history,indent=2))
    elif (outdir/'updater_best.pt').exists() and args.action=='replay':
        state=torch.load(outdir/'updater_best.pt',map_location=dev,weights_only=False)
        updater=LatentRevisionUpdater(model.latent_dim,hidden=state['hidden']).to(dev)
        updater.load_state_dict(state['model_state'])
    # Do not treat test as model-selection feedback. Replay is terminal evaluation.
    if args.action in ('train','train_updater'):
        save_path=outdir/('validation_predictions.npz' if args.action=='train' else 'validation_updater_predictions.npz')
        metrics=replay(model,data['val'],scaler,dev,updater,correction=correction,batch=args.batch,save_path=save_path)
        path=outdir/f'{args.action}_validation.json'
    else:
        metrics=replay(model,data['test'],scaler,dev,updater,correction=correction,batch=args.batch,save_path=outdir/'test_predictions.npz')
        path=outdir/'test_replay.json'
    path.write_text(json.dumps(metrics,indent=2))
    print(json.dumps({'written':str(path),'metrics':{m:{k:v for k,v in d.items() if k not in ['mse_per_window','mae_per_window']}
                                                       for m,d in metrics['methods'].items()}},indent=2))

if __name__=='__main__':main()
