"""Pinned WorldTS plus explicit response propagation; no change to the loss/decoder."""
from common import *
from torch import nn
from types import MethodType,SimpleNamespace
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel

class TemporalAttention(nn.Module):
    def __init__(self,d,c,width=64,heads=4):
        super().__init__();self.width=width;self.heads=heads
        self.norm=nn.LayerNorm(2*d+c+2);self.q=nn.Linear(2*d+c+2,width,bias=False);self.k=nn.Linear(2*d+c+2,width,bias=False)
        self.value=nn.Linear(d,width,bias=False);self.out=nn.Linear(width,d,bias=False);nn.init.zeros_(self.out.weight)
    def forward(self,delta,b,c):
        n=delta.shape[1];pos=torch.linspace(0,1,n,device=delta.device,dtype=delta.dtype).view(1,n,1).expand(delta.shape[0],-1,-1)
        x=self.norm(torch.cat([delta,b,c,pos,torch.sin(2*np.pi*pos)],-1));split=lambda z:z.reshape(z.shape[0],n,self.heads,self.width//self.heads).transpose(1,2)
        z=nn.functional.scaled_dot_product_attention(split(self.q(x)),split(self.k(x)),split(self.value(delta)),dropout_p=0.,is_causal=False)
        return self.out(z.transpose(1,2).reshape(delta.shape[0],n,self.width))

class TemporalConv(nn.Module):
    def __init__(self,d,c,width=96,levels=9):
        super().__init__();self.value=nn.Linear(d,width,bias=False);self.gate=nn.Linear(2*d+c,width)
        self.convs=nn.ModuleList([nn.Conv1d(width,width,3,padding=2**j,dilation=2**j,groups=width,bias=False) for j in range(levels)])
        self.out=nn.Linear(width,d,bias=False);nn.init.zeros_(self.out.weight)
    def forward(self,delta,b,c):
        v=self.value(delta)*torch.sigmoid(self.gate(torch.cat([delta,b,c],-1)));v=v.transpose(1,2)
        for conv in self.convs:v=v+nn.functional.gelu(conv(v))
        return self.out(v.transpose(1,2))

class HistoricalLoop(nn.Module):
    def __init__(self,original,k):super().__init__();self.inner=original;self.k=k;self.gate=nn.Parameter(torch.zeros(()))
    def forward(self,x,attn_mask=None):
        x,a=self.inner(x,attn_mask=attn_mask)
        for _ in range(self.k-1):z,a=self.inner(x,attn_mask=attn_mask);x=x+.5*torch.tanh(self.gate)*(z-x)
        return x,a

def build_model(config,spec,initial=None):
    model=WorldTSModel(config)
    if initial is not None:model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['Model'],strict=True)
    variant=spec['variant'];k=spec.get('k',1);p=model.predictor
    if variant=='B0':return model
    if variant=='C1':
        p.patch_predictor.encoder=HistoricalLoop(p.patch_predictor.encoder,k);return model
    d=model.latent_dim;c=model.condition_dim;width=spec.get('width',64)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(spec.get('seed',2021)+903)
        maker=lambda:TemporalConv(d,c,width) if variant=='B2' else TemporalAttention(d,c,width)
        p.response_blocks=nn.ModuleList([maker() for _ in range(k if variant=='C3' else 1)])
        p.response_eta_logit=nn.Parameter(torch.zeros(()))
    p.response_k=k;p.response_spec=dict(spec);p.response_trace=[]
    def response(self,h_hist,c_hist=None,m_hist=None,t_hist=None,c_future=None):
        b=self.autonomous_trajectory(h_hist,c_hist,m_hist,t_hist)
        if c_future is None:return b
        r=self.control_mlp(torch.cat([b,c_future],-1));delta=r;trace=[delta]
        for j in range(self.response_k):
            block=self.response_blocks[j] if variant=='C3' else self.response_blocks[0]
            eta=torch.sigmoid(self.response_eta_logit)
            if variant=='FULL':delta=r+eta*block(b+delta,b,c_future)
            elif variant=='NOANCHOR':delta=delta+eta*block(delta,b,c_future)
            else:delta=r+eta*block(delta,b,c_future)
            trace.append(delta)
        self.response_trace=[x.detach() for x in trace]
        self.response_base=b.detach();self.response_anchor=r.detach()
        result=b+delta;assert torch.isfinite(result).all();return result
    p.forward=MethodType(response,p);return model

def preflight():
    rows=[]
    for ds,h in [('NP',360),('PJM',360),('NP',24),('PJM',24)]:
        config=SimpleNamespace(**cfg(ds,h));initial=baseline_dir(ds,h)/'stages/stage2_decoder_mix.pt'
        for variant,k in [('B0',1),('B1',1),('B2',1),('C1',4),('C3',4),('C4',1),('C5',2),('C5',4),('NOANCHOR',4),('FULL',4)]:
            seed(2021);spec={'variant':variant,'k':k,'width':96 if variant=='B2' else 128 if variant=='C4' else 64,'seed':2021}
            model=build_model(config,spec,initial);model.eval();ref=WorldTSModel(config);ref.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['Model']);ref.eval()
            x=torch.randn(2,config.seq_len,3);f=torch.randn(2,h,2);t=torch.randn(2,h,1)
            with torch.no_grad():
                a=model(x,x_future=f,future_target=t);a2=model(x,x_future=f,future_target=t+19);pred=model.decode(a['h_pred'],a['target_revin_stats'],a['h_hist']);rp=ref.forecast(x,x_future=f)
                assert torch.equal(a['h_pred'],a2['h_pred']);assert torch.equal(pred,rp)
                if hasattr(model.predictor,'response_blocks'):
                    for block in model.predictor.response_blocks:assert torch.count_nonzero(block(torch.zeros_like(a['h_pred']),a['h_pred'],a['z_cond_future']))==0
            model.configure_for_stage1();assert not any(p.requires_grad for p in model.decoder.parameters())
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);gradient_steps=[]
            for step in range(5):
                opt.zero_grad();out=model(x,x_future=f);loss=(out['h_pred']-torch.randn_like(out['h_pred'])).square().mean();loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if hasattr(model.predictor,'response_blocks'):
                    gradient_steps.append({n:float(p.grad.norm()) for n,p in model.predictor.response_blocks.named_parameters() if p.grad is not None})
                opt.step()
            if gradient_steps:
                assert any(v>0 for key,v in gradient_steps[0].items() if '.out.' in key)
                assert any(v>0 for key,v in gradient_steps[-1].items() if '.q.' in key or '.value.' in key)
                with torch.no_grad():
                    for block in model.predictor.response_blocks:assert torch.count_nonzero(block(torch.zeros_like(out['h_pred']),out['h_pred'],out['z_cond_future']))==0
            model.configure_for_stage2();frozen=tensor_sha(model.predictor);opt=torch.optim.Adam(model.decoder.parameters(),lr=.001)
            out=model(x,x_future=f);loss=(model.decode(out['h_pred'],out['target_revin_stats'],out['h_hist'])-t).abs().mean();loss.backward();opt.step();assert frozen==tensor_sha(model.predictor)
            rows.append({'dataset':ds,'horizon':h,'variant':variant,'k':k,'passed':True,'initial_prediction_exact':True,'future_label_invariance':True,'zero_response_preserved':True,'stage2_predictor_frozen':True,'gradient_steps':gradient_steps,'parameters':sum(p.numel() for p in model.parameters())})
            print('RESPONSE_PREFLIGHT_PASS',ds,h,variant,k,flush=True)
    write_json(ROOT/'reports/response_preflight.json',{'passed':True,'source_sha256':sha(__file__),'results':rows})
if __name__=='__main__':preflight()
