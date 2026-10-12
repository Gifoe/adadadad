from common import *
from response_model import build_model,SimpleNamespace
p=ROOT/'results/synthetic/a0.2_noise0.05_B0/checkpoint.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
seed(2021);models=[build_model(SimpleNamespace(**ck['config']),ck['spec']).cuda().eval() for _ in range(2)]
for m in models:m.load_state_dict(ck['Model'],strict=True)
models[0].configure_for_stage2()
x=torch.randn(32,24,2,device='cuda');f=torch.randn(32,48,1,device='cuda')
with torch.no_grad():
 a=models[0].forecast(x,x_future=f);b=models[1].forecast(x,x_future=f);again=models[0].forecast(x,x_future=f)
 print('RELOAD_COMPARISON',json.dumps({'state_exact':tensor_sha(models[0])==tensor_sha(models[1]),'max_abs_diff':float((a-b).abs().max()),'repeat_max_abs_diff':float((a-again).abs().max()),'a_mean':float(a.mean()),'b_mean':float(b.mean())}),flush=True)
