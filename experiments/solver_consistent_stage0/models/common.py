import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel,SDPBackend
from solvers.fixed_step import integrate, schedule

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); d=c['d_model']; self.heads=c['n_heads']; self.drop=c['dropout']
        self.backend=c.get('attention_backend','math'); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d); self.out=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.Linear(d,c['d_ff']),nn.GELU(),nn.Dropout(self.drop),nn.Linear(c['d_ff'],d))
        self.dropout=nn.Dropout(self.drop)
    def forward(self,h,mask):
        b,l,d=h.shape
        q,k,v=self.qkv(self.n1(h)).view(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        if self.backend=='math':
            with sdpa_kernel(SDPBackend.MATH):
                a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=self.drop if self.training else 0.)
        else:
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=self.drop if self.training else 0.)
        h=h+self.dropout(self.out(a.transpose(1,2).reshape(b,l,d)))
        return h+self.dropout(self.ff(self.n2(h)))

class Conditioning(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,d))
        self.register_buffer('freq',torch.tensor([1.,2.,4.,8.])*math.pi)
    def forward(self,t,second,device):
        z=torch.tensor([t,second],device=device,dtype=torch.float32)[:,None]*self.freq[None,:]
        return self.net(torch.cat([z.sin(),z.cos()],dim=-1).flatten())

class ResidualCore(nn.Module):
    def __init__(self,c,conditioned=False):
        super().__init__(); self.blocks=nn.ModuleList([Block(c) for _ in range(c['core_blocks'])])
        self.condition=Conditioning(c['d_model']) if conditioned else None
        self.checkpointing=c.get('activation_checkpointing',False)
    def direction(self,h,mask,t=None,second=None):
        # Transformer residual increment, NOT the full B(H) used as a derivative.
        start=h
        if self.condition is not None: h=h+self.condition(t,second,h.device)[None,None,:]
        for block in self.blocks:
            h=checkpoint(block,h,mask,use_reentrant=False) if self.checkpointing and self.training else block(h,mask)
        return (h-start)*mask[:,:,None]

class VectorField(nn.Module):
    """Only H, continuous t, and static token padding are visible to this network."""
    def __init__(self,c): super().__init__(); self.core=ResidualCore(c,True)
    def forward(self,H,t,padding_mask):
        # The second encoding slot is permanently zero, matching conditioning capacity.
        # No solver configuration, dt, budget, cache, or mutable solver state exists here.
        return self.core.direction(H,padding_mask,t,0.)

class Model(nn.Module):
    def __init__(self,c,name):
        super().__init__(); self.name=name; self.c=c
        d=c['d_model']; self.embedding=nn.Embedding(c.get('actual_vocab_size',c['vocab_size']),d,padding_idx=0)
        self.position=nn.Embedding(c['max_length'],d)
        self.stem=nn.ModuleList([Block(c) for _ in range(c['stem_blocks'])])
        self.core=VectorField(c) if name=='vector_field' else ResidualCore(c,name=='step_conditioned_loop')
        self.norm=nn.LayerNorm(d); self.classifier=nn.Linear(d,2)
        self.apply(self.init)
    @staticmethod
    def init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def encode(self,tokens):
        mask=tokens.ne(0); h=self.embedding(tokens)+self.position(torch.arange(tokens.shape[1],device=tokens.device))[None,:,:]
        for block in self.stem:
            h=checkpoint(block,h,mask,use_reentrant=False) if self.c.get('activation_checkpointing',False) and self.training else block(h,mask)
        return h*mask[:,:,None],mask
    def readout(self,h,mask):
        return self.classifier(self.norm(h[:,0,:]))
    def forward(self,tokens,budget=8,solver='euler',schedule_kind='uniform',return_hidden=False):
        h,mask=self.encode(tokens); h0=h; nfe=0
        if budget:
            if self.name=='vector_field':
                factor={'euler':1,'heun':2,'rk4':4}[solver]
                if budget % factor: raise ValueError('budget must be divisible by solver NFE/step')
                h,nfe=integrate(lambda H,t:self.core(H,t,mask),h,schedule(budget//factor,schedule_kind),solver)
            elif self.name=='step_conditioned_loop':
                if solver not in ('euler','loop'): raise ValueError('no solver exchange for discrete baseline')
                t=0.
                for dt in schedule(budget,schedule_kind): h=h+self.core.direction(h,mask,t,dt); t+=dt; nfe+=1
            else:
                if solver not in ('euler','loop') or schedule_kind!='uniform': raise ValueError('vanilla is a discrete recurrence')
                for _ in range(budget): h=h+self.core.direction(h,mask); nfe+=1
        logits=self.readout(h,mask)
        if return_hidden: return logits,h,h0,mask,nfe
        return logits

def counts(m):
    return {'params':sum(p.numel() for p in m.parameters()),'trainable_params':sum(p.numel() for p in m.parameters() if p.requires_grad),'core_params':sum(p.numel() for p in m.core.parameters())}
