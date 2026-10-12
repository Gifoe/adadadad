import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *
def summary(values):
 a=np.asarray(values);a=a[np.isfinite(a)]
 if not len(a):return {'valid':0,'mean':None,'median':None,'q10':None,'q25':None,'q75':None,'q90':None,'negative_fraction':None,'near_zero_fraction':None}
 return {'valid':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'q10':float(np.quantile(a,.1)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),'q90':float(np.quantile(a,.9)),'negative_fraction':float(np.mean(a<0)),'near_zero_fraction':float(np.mean(abs(a)<.1))}
def run():
 assert json.loads((ROOT/'outputs/preflight_metrics.json').read_text())['passed'];require_gradients();start_time=time.time();errors=[];alignment=[];intervals=[];parameters=[];raw_intervals=[];hashes=[]
 for ds in PROTOCOL['tasks']:
  model,cfg,_,_=load_model(ds);crit=criterion(cfg);assert crit.latent_regularizer_type=='legacy_vc';d=data(ds);oo=origins(d,'val');nb=(len(oo)+31)//32;chosen=set(np.linspace(0,nb-1,8,dtype='int64').tolist());before=state_hash(model)
  for bi,start in enumerate(range(0,len(oo),32)):
   these=oo[start:start+32];x,cf,y=batch(d,these);out=forward(model,x,cf,y);original=out['h_pred'].detach();leaf=original.clone().requires_grad_(True);state=dict(out,h_pred=leaf);terms=losses(model,state,y,crit)
   gy=torch.autograd.grad(terms['forecast'],leaf,retain_graph=True)[0];gradients={key:torch.autograd.grad(terms[key],leaf,retain_graph=True)[0] for key in ['l1','cosine','full']}
   comparisons={key:cosine_rows(g,gy) for key,g in gradients.items()}
   with torch.no_grad():
    reconstruction=decode(model,out['h_target'],out);ep=(terms['pred']-y).double();er=(reconstruction-y).double();eg=(terms['pred']-reconstruction).double();latent_l1=(leaf-out['h_target']).abs().mean((1,2));latent_cos=1-torch.nn.functional.cosine_similarity(leaf.flatten(1),out['h_target'].flatten(1),dim=1)
   for j,o in enumerate(these):
    err={'dataset':ds,'horizon':360,'origin':int(o),'datetime':str(d['df'].index[o]),'batch':bi,'forecast_mse':float(ep[j].square().mean()),'forecast_mae':float(ep[j].abs().mean()),'latent_l1':float(latent_l1[j]),'latent_trajectory_cosine':float(latent_cos[j]),'target_reconstruction_mse':float(er[j].square().mean()),'target_reconstruction_mae':float(er[j].abs().mean()),'decoded_gap_mse':float(eg[j].square().mean()),'decoded_gap_mae':float(eg[j].abs().mean()),'batch_full_latent_objective':float(terms['full']),'batch_regularizer_pred':float(terms['regularizer_pred']),'batch_regularizer_target':float(terms['regularizer_target']),'raw_observation_forecast_mse':float(ep[j].square().mean())*float(d['sy'].scale_[0])**2,'raw_observation_forecast_mae':float(ep[j].abs().mean())*float(d['sy'].scale_[0])}
    row={'dataset':ds,'horizon':360,'origin':int(o),'datetime':err['datetime'],'batch':bi,'forecast_mse':err['forecast_mse'],'forecast_mae':err['forecast_mae']}
    for key,(cs,nl,ny,status) in comparisons.items():row.update({key+'_cosine':float(cs[j]),key+'_gradient_norm':float(nl[j]),key+'_status':status[j]})
    row['forecast_gradient_norm']=float(comparisons['full'][2][j]);errors.append(err);alignment.append(row)
   for a,b in PROTOCOL['horizon_segments']:
    for key,g in gradients.items():
     cs,nl,ny,status=cosine_rows(g[:,a:b],gy[:,a:b]);values=summary(cs);intervals.append({'dataset':ds,'batch':bi,'origin_first':int(these[0]),'origin_last':int(these[-1]),'segment':f'{a+1}-{b}','objective':key,'n_origins':len(these),'invalid_gradients':sum(s!='VALID' for s in status),'latent_gradient_norm_median':float(np.median(nl)),'forecast_gradient_norm_median':float(np.median(ny)),**values})
     raw_intervals.extend({'dataset':ds,'origin':int(o),'segment':f'{a+1}-{b}','objective':key,'cosine':float(c),'latent_norm':float(l),'forecast_norm':float(z),'status':s} for o,c,l,z,s in zip(these,cs,nl,ny,status))
   if bi in chosen:
    model.predictor.requires_grad_(True);param_out=forward(model,x,cf,y);pdiff=float((param_out['h_pred']-original).abs().max());assert pdiff<=1e-6;pl=losses(model,param_out,y,crit);pp=list(model.predictor.parameters());py,uy=flat_gradient(pl['forecast'],pp)
    for key in ['l1','cosine','full']:
     pg,ul=flat_gradient(pl[key],pp);pc,ln,yn,status=cosine_rows(pg[None],py[None]);parameters.append({'dataset':ds,'batch':bi,'origin_first':int(these[0]),'origin_last':int(these[-1]),'objective':key,'parameter_cosine':float(pc[0]),'latent_parameter_gradient_norm':float(ln[0]),'forecast_parameter_gradient_norm':float(yn[0]),'status':status[0],'unused_latent_parameters':ul,'unused_forecast_parameters':uy,'state_cosine_mean_same_batch':summary(comparisons[key][0])['mean'],'same_weight_forward_max_difference':pdiff})
    model.predictor.requires_grad_(False);assert all(p.grad is None for p in model.parameters())
   if bi%40==0:print('ALIGNMENT_AUDIT_PROGRESS',ds,bi,nb,flush=True)
  assert state_hash(model)==before;hashes.append({'dataset':ds,'hash_before':before,'hash_after':state_hash(model),'unchanged':True})
 write_csv(ROOT/'outputs/error_alignment.csv',errors);write_csv(ROOT/'outputs/gradient_alignment.csv',alignment);write_csv(ROOT/'outputs/horizon_alignment.csv',intervals);write_csv(ROOT/'outputs/predictor_parameter_alignment.csv',parameters)
 arrays=pd.DataFrame(raw_intervals);np.savez_compressed(ROOT/'outputs/horizon_alignment_raw.npz',dataset=arrays.dataset.to_numpy(dtype='U3'),origin=arrays.origin.to_numpy(),segment=arrays.segment.to_numpy(dtype='U8'),objective=arrays.objective.to_numpy(dtype='U8'),cosine=arrays.cosine.to_numpy(),latent_norm=arrays.latent_norm.to_numpy(),forecast_norm=arrays.forecast_norm.to_numpy(),status=arrays.status.to_numpy(dtype='U12'))
 summaries=[]
 for ds in ['NP','PJM']:
  ar=pd.DataFrame([r for r in alignment if r['dataset']==ds]);pr=pd.DataFrame([r for r in parameters if r['dataset']==ds])
  for key in ['l1','cosine','full']:
   summaries.append({'dataset':ds,'space':'state','objective':key,'n_total':len(ar),'invalid':int((ar[key+'_status']!='VALID').sum()),**summary(ar[key+'_cosine'])})
   g=pr[pr.objective==key];summaries.append({'dataset':ds,'space':'predictor_parameter','objective':key,'n_total':len(g),'invalid':int((g.status!='VALID').sum()),**summary(g.parameter_cosine)})
 write_json(ROOT/'outputs/alignment_summary.json',{'summaries':summaries,'model_hashes':hashes,'seconds':time.time()-start_time,'all_parameters_frozen_for_state_audit':True,'parameter_audit_exception':'Predictor requires_grad enabled only for fixed same-weight, eval batches; no optimizer or accumulated .grad','full_objective_sample_scope':'batch objective gradient components; legacy VC couples batch and time tokens','gradient_intervals':'slices of complete-objective gradients, not independently redefined segment losses','units':'latent coordinates; forecasts in train-standardized observation units after original history RevIN reversal; raw-space errors also saved'})
 write_json(ROOT/'STATUS.json',{'phase':1,'status':'COMPLETE','overall_complete':False});print('ALIGNMENT_AUDIT_COMPLETE',len(alignment),len(parameters),flush=True)
if __name__=='__main__':run()
