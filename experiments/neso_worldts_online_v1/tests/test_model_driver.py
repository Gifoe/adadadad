"""Integration smoke of generic training logic. Does not claim official WorldTS was run."""
import torch
from torch import nn
import numpy as np
from neso_online.native_worldts import (
    fit_two_stage,fit_output_correction,train_updater,replay,revision_features,
)


class DummyCore(nn.Module):
    """Only mirrors the WorldTSModel call signature for testing our driver."""
    def __init__(self,horizon=12):
        super().__init__()
        self.latent_dim=4
        self.encoder=nn.Linear(1,4)
        self.condition=nn.Linear(2,4)
        self.decoder=nn.Linear(4,1)
        self.horizon=horizon
    def configure_for_stage1(self):
        for m in [self.encoder,self.condition]:
            for p in m.parameters():p.requires_grad_(True)
        for p in self.decoder.parameters():p.requires_grad_(False)
    def configure_for_stage2(self):
        for m in [self.encoder,self.condition]:
            for p in m.parameters():p.requires_grad_(False)
        for p in self.decoder.parameters():p.requires_grad_(True)
    def set_stage1_eval(self):
        self.encoder.eval();self.condition.eval()
    def forward(self,input,future_target=None,x_future=None,**kwargs):
        h_hist=self.encoder(input[:,:,:1])
        h_pred=h_hist[:,-1:,:].expand(-1,self.horizon,-1)+self.condition(x_future)
        h_target=self.encoder(future_target) if future_target is not None else None
        return {'h_hist':h_hist,'h_pred':h_pred,'h_target':h_target,
                'target_revin_stats':None,'z_visreg_hist':None,
                'z_visreg_target':None,'z_visreg_pred':None}
    def decode(self,h_tokens,target_revin_stats=None,history_tokens=None):
        return self.decoder(h_tokens)


class DummyLoss(nn.Module):
    def forward(self,h_pred,h_target,**kwargs):
        return {'loss':torch.nn.functional.l1_loss(h_pred,h_target)}


def _arrays(n):
    rng=np.random.default_rng(17+n)
    arr={'history':rng.standard_normal((n,24,3)).astype('float32'),
         'target':rng.standard_normal((n,12,1)).astype('float32'),
         'cov_a':rng.standard_normal((n,12,2)).astype('float32')}
    arr['cov_b']=arr['cov_a'].copy()
    arr['cov_b'][:,3:]+=rng.standard_normal((n,9,2)).astype('float32')*.2
    arr['update_mask']=np.zeros((n,12),dtype=bool)
    arr['update_mask'][:,3:]=True
    return arr


def test_generic_two_stage_updater_and_replay(tmp_path):
    train={'train':_arrays(14),'val':_arrays(6)}
    model=DummyCore()
    hist=fit_two_stage(model,train,DummyLoss(),epochs1=1,epochs2=1,
                       patience=1,lr1=.001,lr2=.001,batch=4,seed=1,
                       device=torch.device('cpu'),outdir=tmp_path)
    assert len(hist)==2
    coef=fit_output_correction(model,train['train'],torch.device('cpu'),batch=4)
    assert len(coef)==6
    updater,u_hist=train_updater(model,train,epochs=1,patience=1,
                                learning_rate=.001,batch=4,seed=1,
                                device=torch.device('cpu'),outdir=tmp_path,hidden=8)
    results=replay(model,train['val'],{'std':[1.,1.,1.]},torch.device('cpu'),
                   updater=updater,correction=coef,batch=4)
    assert len(results['methods'])==4
    assert results['n_scored_slots']==6*9
    for key,value in results['methods'].items():
        assert np.isfinite(value['mse'])
    delta=revision_features(train['train']['cov_a'],train['train']['cov_b'])
    assert np.all(delta[:,:3]==0)
