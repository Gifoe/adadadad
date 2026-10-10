import copy, math
import torch
from torch import nn

class Refiner(nn.Module):
    def __init__(self,patch,width,ff):
        super().__init__();self.patch=patch
        self.state=nn.Linear(patch*32,width);self.condition=nn.Linear(patch*32,width)
        self.block=nn.TransformerEncoderLayer(width,4,ff,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.readout=nn.Linear(width,patch*32)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
    def forward(self,z,c):
        b,h,d=z.shape;u=z.reshape(b,-1,self.patch*32);v=c.reshape(b,-1,self.patch*32)
        return z + .1*self.readout(self.block(self.state(u)+self.condition(v))).reshape(b,h,d)

class ConditionalLoop(nn.Module):
    def __init__(self,ae,t,h,variant,k=2,ff=128):
        super().__init__();self.ae=ae; self.variant=variant;self.k=k;self.patch=24;self.h=h; self.t=t
        self.ff=ff
        assert t%24==0 and h%24==0
        for p in ae.parameters():p.requires_grad_(False)
        # Observation-only modules must not shift initialization of the shared backbone.
        with torch.random.fork_rng(devices=[]):
            self.obs_encoder=nn.Linear(1,32) if variant=='B1' else None
        self.cov_encoder=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,32))
        self.hist_fusion=nn.Linear(64,64);self.patch_embed=nn.Linear(24*64,64)
        self.hist_position=nn.Parameter(torch.zeros(1,t//24,64))
        layer=nn.TransformerEncoderLayer(64,4,ff,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.predictor=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.time_projection=nn.Linear(t//24,h//24)
        self.latent_head=nn.Linear(64,24*32)
        self.future_fusion=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,32))
        self.obs_head=nn.Linear(32,1) if variant=='B1' else None
        if variant=='B3':
            first=Refiner(24,64,ff);self.refiners=nn.ModuleList([copy.deepcopy(first) for _ in range(k)])
        elif variant in ('B4','B5'):self.refiner=Refiner(24,64,ff)
    def train(self,mode=True):
        super().train(mode);self.ae.eval();return self
    def decode(self,z):return self.obs_head(z) if self.variant=='B1' else self.ae.decode(z)
    def forward(self,y_hist,c_hist,c_future,return_all=True):
        assert y_hist.shape[1:]==(self.t,1) and c_future.shape[1:]==(self.h,2)
        if self.variant=='B1':zh=self.obs_encoder(y_hist)
        else:
            with torch.no_grad():zh=self.ae.encode(y_hist)
        ch=self.cov_encoder(c_hist);cf=self.cov_encoder(c_future)
        u=self.hist_fusion(torch.cat([zh,ch],dim=-1))
        b=u.shape[0];u=self.patch_embed(u.reshape(b,self.t//24,24*64))+self.hist_position
        u=self.predictor(u);u=self.time_projection(u.transpose(1,2)).transpose(1,2)
        z=self.latent_head(u).reshape(b,self.h,32)
        z=z+self.future_fusion(torch.cat([z,cf],dim=-1));zs=[z]
        if self.variant=='B3':
            for block in self.refiners:z=block(z,cf);zs.append(z)
        elif self.variant in ('B4','B5'):
            # Same modules/operations/parameters; only direct refinement covariates differ.
            direct=cf*(0. if self.variant=='B4' else 1.)
            for _ in range(self.k):z=self.refiner(z,direct);zs.append(z)
        return zs,[self.decode(v) for v in (zs if return_all else zs[-1:])]

def latent_loss(pred,target):
    mse=torch.nn.functional.mse_loss(pred,target)
    cosine=1-torch.nn.functional.cosine_similarity(pred.reshape(-1,32),target.reshape(-1,32),dim=-1).mean()
    return 10*mse+15*cosine
