import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *

def run():
 rows=[];start=time.time()
 for ds in ['NP','PJM']:
  d=data(ds);oo=origins(d,'val');nb=(len(oo)+31)//32
  for method in ['A0','A1','A2']:
   directory=ROOT/'outputs/runs'/f'{ds}_{method}_s2021';metrics=json.loads((directory/'metrics.json').read_text());model,cfg,_,_=load_model(ds,predictor_grad=True);model.load_state_dict(torch.load(metrics['best_checkpoint'],map_location='cpu',weights_only=False)['Model'],strict=True);model.eval();before=state_hash(model);crit=criterion(cfg);params=list(model.predictor.parameters())
   for bi in np.linspace(0,nb-1,8,dtype='int64'):
    these=oo[bi*32:bi*32+32];x,cf,y=batch(d,these);out=forward(model,x,cf,y);terms=losses(model,out,y,crit);gl,_=flat_gradient(terms['full'],params);gy,_=flat_gradient(terms['forecast'],params);pc,nl,ny,st=cosine_rows(gl[None],gy[None]);leaf=out['h_pred'].detach().clone().requires_grad_(True);lt=losses(model,dict(out,h_pred=leaf),y,crit);sl=torch.autograd.grad(lt['full'],leaf,retain_graph=True)[0];sy=torch.autograd.grad(lt['forecast'],leaf)[0];sc,sn,yn,ss=cosine_rows(sl,sy);valid=sc[np.isfinite(sc)]
    rows.append({'dataset':ds,'method':method,'best_step':metrics['best_step'],'batch':int(bi),'origin_first':int(these[0]),'origin_last':int(these[-1]),'parameter_cosine':float(pc[0]),'latent_parameter_gradient_norm':float(nl[0]),'forecast_parameter_gradient_norm':float(ny[0]),'parameter_status':st[0],'state_cosine_mean':float(valid.mean()),'state_negative_fraction':float((valid<0).mean()),'state_invalid':sum(s!='VALID' for s in ss),'forecast_mae':float(terms['forecast'].detach()),'latent_full':float(terms['full'].detach())})
   assert state_hash(model)==before and all(p.grad is None for p in model.parameters());print('POST_SELECTED_AUDIT_COMPLETE',ds,method,flush=True)
 write_csv(ROOT/'outputs/post_selected_alignment.csv',rows);write_json(ROOT/'outputs/post_selected_audit.json',{'seconds':time.time()-start,'n_batches':len(rows),'fixed_validation_batches':'same8 as initial parameter audit','no_optimizer_updates':True,'test_used':False})
if __name__=='__main__':run()
