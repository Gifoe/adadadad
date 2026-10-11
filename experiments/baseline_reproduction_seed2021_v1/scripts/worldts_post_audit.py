"""Audit each trained best WorldTS checkpoint on fixed validation origins, on CPU."""
from common import *
from dag_data import *
from worldts_config import *
from types import SimpleNamespace
sys.path.insert(0,str(ROOT/'upstream/WorldTS/DAG-bench'))
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel
def execute():
 rows=[]
 for path in sorted((ROOT/'results/worldts').glob('*/metrics.json')):
  out=path.parent;m=json.loads(path.read_text());ds=m['dataset'];h=m['horizon'];_,_,hp=published_config(ds,h);cfg=json.loads((out/'resolved_config.json').read_text());model=WorldTSModel(SimpleNamespace(**cfg)).float().eval()
  ck=torch.load(m['checkpoint_path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['Model'],strict=True);model.configure_for_stage2();model.eval();df,y,c,tr,tv,sy,sc=load_data(ds)
  origins=np.linspace(tr,tv-h,32,dtype='int64');preds=[];targets=[];hist=[];maxdelta=0.
  with torch.no_grad():
   for start in range(0,len(origins),4):
    oo=origins[start:start+4];yh=torch.stack([y[o-hp['seq_len']:o] for o in oo]);ch=torch.stack([c[o-hp['seq_len']:o] for o in oo]);cf=torch.stack([c[o:o+h] for o in oo]);target=torch.stack([y[o:o+h] for o in oo]);x=torch.cat([yh,ch],-1)
    a=model(x,future_target=target,x_future=cf);changed=model(x,future_target=target+17,x_future=cf);inference=model(x,x_future=cf)
    maxdelta=max(maxdelta,float((a['h_pred']-changed['h_pred']).abs().max()),float((a['h_pred']-inference['h_pred']).abs().max()));assert maxdelta==0
    preds.append(a['h_pred'].numpy());targets.append(a['h_target'].numpy());hist.append(a['h_hist'].numpy())
  def stats(values):
   z=np.concatenate(values).reshape(-1,hp['latent_dim']).astype('float64');std=z.std(0);cov=np.cov(z,rowvar=False);eigen=np.maximum(np.linalg.eigvalsh(cov),0);p=eigen/eigen.sum();nz=p[p>0]
   return {'mean':float(z.mean()),'std_mean':float(std.mean()),'std_min':float(std.min()),'std_max':float(std.max()),'near_constant_dim_fraction':float(np.mean(std<1e-3)),'effective_covariance_rank':float(np.exp(-np.sum(nz*np.log(nz)))),'severe_collapse_flag':bool(np.mean(std<1e-3)>.95)}
  result={'run_id':out.name,'dataset':ds,'horizon':h,'checkpoint_sha256':sha(m['checkpoint_path']),'source_commit':COMMITS['WorldTS'],'validation_origins':origins.tolist(),'test_set_used':False,'device':'CPU','trained_future_label_max_abs_diff':maxdelta,'predicted_states':stats(preds),'target_states':stats(targets),'history_states':stats(hist),'collapse_threshold_note':'Diagnostic flag: more than 95 percent of latent dimensions have std below 1e-3 across sampled validation states; not a statistical acceptance threshold'}
  assert not result['predicted_states']['severe_collapse_flag'] and not result['target_states']['severe_collapse_flag']
  write_json(out/'collapse_diagnostics.json',result);rows.append(result);print('TRAINED_WORLDTS_AUDIT_PASSED',out.name,result['predicted_states'],flush=True)
 assert len(rows)==4;write_json(ROOT/'reports/worldts_trained_checkpoint_audit.json',{'passed':True,'results':rows})
if __name__=='__main__':execute()
