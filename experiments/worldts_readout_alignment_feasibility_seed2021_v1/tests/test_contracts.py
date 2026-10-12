import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *
def run_contracts(ds):
 checks=[];d=data(ds);oo=origins(d,'val')[:4];x,cf,y=batch(d,oo)
 for method in ['A0','A1','A2']:
  model,cfg,_,_=load_model(ds,predictor_grad=True);crit=criterion(cfg);before=state_hash(model,'frozen');p0=state_hash(model,'predictor');predictor_mode(model,True)
  out=forward(model,x,cf,y);terms=losses(model,out,y,crit);params=list(model.predictor.parameters())
  gy,_=flat_gradient(terms['forecast'],params);assert torch.isfinite(gy).all() and gy.norm()>0
  assert not terms['regularizer_target'].requires_grad,'Frozen encoded target regularizer is a constant'
  objective=terms['full'] if method=='A0' else terms['forecast'] if method=='A1' else terms['full']+terms['forecast']
  optimizer=torch.optim.Adam(params,lr=1e-4);optimizer.zero_grad(set_to_none=True);objective.backward()
  assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('predictor.'))
  assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None);optimizer.step()
  assert state_hash(model,'frozen')==before and state_hash(model,'predictor')!=p0
  checks.append({'dataset':ds,'check':'predictor_only_update_'+method,'passed':True,'decoded_gradient_norm':float(gy.norm()),'frozen_hash':before})
 model,cfg,_,_=load_model(ds);model.eval()
 with torch.no_grad():
  a=forward(model,x,cf,y);b=forward(model,x,cf,y+19);none=model(x,x_future=cf,future_target=None)
  assert torch.equal(a['h_pred'],b['h_pred']) and torch.equal(a['h_pred'],none['h_pred'])
  assert torch.equal(a['target_revin_stats']['mean'],b['target_revin_stats']['mean'])
  forecast=model.forecast(x,x_future=cf);decoded=decode(model,a['h_pred'],a);assert torch.equal(forecast,decoded)
  wrapper=object.__new__(WorldTS);wrapper.model=model;wrapper.config=cfg
  assert torch.equal(wrapper._process_forecast(x,y,None,None,exog_future=cf)['output'],wrapper._process_forecast(x,y+29,None,None,exog_future=cf)['output'])
  assert torch.max(torch.abs(wrapper._predict_only(x,None,None,exog_future=cf)-forecast))<=1e-6
  stats=model._compute_revin_stats(x);assert torch.equal(stats['mean'][...,:1],a['target_revin_stats']['mean'])
  assert torch.equal(stats['stdev'][...,:1],a['target_revin_stats']['stdev'])
 saved={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};h0=state_hash(model);model.load_state_dict(saved,strict=True)
 assert state_hash(model)==h0
 with torch.no_grad():assert torch.equal(model.forecast(x,x_future=cf),forecast)
 leaf=a['h_pred'].detach().requires_grad_(True);readout=decode(model,leaf,a)
 grad=torch.autograd.grad(torch.nn.functional.l1_loss(readout,y),leaf)[0];assert torch.isfinite(grad).all() and grad.norm()>0
 with torch.no_grad():
  try:require_gradients()
  except RuntimeError:pass
  else:raise AssertionError('no_grad audit guard must reject disabled gradients')
 u=torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,0.]]);v=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[1.,0.]])
 c,_,_,s=cosine_rows(u,v);assert np.allclose(c[:3],[1.,-1.,0.],atol=1e-11) and np.isnan(c[3]) and s[3]=='ZERO_L'
 p=np.array([[[1.],[2.]],[[3.],[4.]]],dtype='float32');truth=np.zeros_like(p);assert metrics(p,truth)==(7.5,2.5)
 checks.extend({'dataset':ds,'check':name,'passed':True} for name in ['target_label_and_deployment_invariance','official_forward_decode_and_predict_only_equivalence','history_revin_statistics_exact','checkpoint_tensor_and_output_roundtrip','state_gradient_through_frozen_decoder','manual_cosine_and_zero_gradient_handling','float64_forecast_metrics','no_grad_audit_guard'])
 return checks
if __name__=='__main__':
 rows=run_contracts('NP')+run_contracts('PJM');write_json(ROOT/'outputs/tests_report.json',{'passed':True,'checks':rows});print('ALL_CONTRACT_TESTS_PASS',len(rows),flush=True)
