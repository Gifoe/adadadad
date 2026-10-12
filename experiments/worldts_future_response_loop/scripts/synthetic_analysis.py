from common import *
from response_model import build_model,SimpleNamespace

def run():
 rows=[]
 for out in sorted((ROOT/'results/synthetic').iterdir()):
  if not (out/'metrics.json').is_file():continue
  result=json.loads((out/'metrics.json').read_text());a=result['system_a'];noise=result['noise_std'];gen=np.random.default_rng(419);u=gen.normal(size=(4096,1));eps=gen.normal(scale=noise,size=(4096,1));v=np.zeros_like(u)
  for j in range(1,len(v)):v[j]=a*v[j-1]+u[j]+eps[j]
  tr=int(len(v)*.7);tv=int(len(v)*.8);ys=v[:tr].std(0);cs=u[:tr].std(0);y=torch.tensor((v-v[:tr].mean(0))/ys,dtype=torch.float32);c=torch.tensor((u-u[:tr].mean(0))/cs,dtype=torch.float32);oo=np.linspace(tr,tv-48,16,dtype='int64')
  x=torch.stack([torch.cat([y[o-24:o],c[o-24:o]],-1) for o in oo]).cuda();f=torch.stack([c[o:o+48] for o in oo]).cuda().requires_grad_(True);ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);model=build_model(SimpleNamespace(**ck['config']),ck['spec']).cuda().eval();model.load_state_dict(ck['Model'],strict=True);model.configure_for_stage2();pred=model.forecast(x,x_future=f);jac=[]
  for h in range(48):jac.append(torch.autograd.grad(pred[:,h].sum(),f,retain_graph=True)[0].detach().cpu().numpy()[:,:,0])
  jac=np.stack(jac,1);distance=np.arange(48)[:,None]-np.arange(48)[None,:];physical=np.where(distance>=0,(float(cs/ys))*np.power(a,np.maximum(distance,0)),0);outside=(distance>=5)|(distance<0);long=distance>=5
  np.savez_compressed(out/'validation_response_jacobian.npz',jacobian=jac,physical_response=physical,origins=oo)
  rows.append({'system_a':a,'noise_std':noise,'variant':result['variant'],'n_validation_origins':16,'physical_long_response_abs_mean':float(np.abs(physical[long]).mean()),'learned_long_response_abs_mean':float(np.abs(jac[:,long]).mean()),'physical_jacobian_mse':float(((jac-physical[None])**2).mean()),'max_gradient_outside_baseline_support':float(np.abs(jac[:,outside]).max()),'future_input_after_output_abs_mean':float(np.abs(jac[:,distance<0]).mean())})
 write_csv(ROOT/'results/stage1_synthetic_response.csv',rows);assert len(rows)==12
 print('SYNTHETIC_RESPONSE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':run()
