import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *
def run():
 assert json.loads((ROOT/'outputs/alignment_summary.json').read_text())['model_hashes'];rows=[];weights={}
 for ds in ['NP','PJM']:
  model,cfg,_,_=load_model(ds,predictor_grad=True);model.eval();crit=criterion(cfg);d=data(ds);before=state_hash(model);positions=np.linspace(cfg.seq_len,d['tr']-360-32+1,16,dtype='int64');ratios=[]
  for index,o in enumerate(positions):
   these=np.arange(o,o+32,dtype='int64');assert these[-1]+360<=d['tr'];x,cf,y=batch(d,these);out=forward(model,x,cf,y);terms=losses(model,out,y,crit);params=list(model.predictor.parameters());gl,_=flat_gradient(terms['full'],params);gy,_=flat_gradient(terms['forecast'],params);nl=float(gl.double().norm());ny=float(gy.double().norm());assert np.isfinite(nl) and np.isfinite(ny) and nl>0 and ny>0;ratio=nl/ny;ratios.append(ratio)
   rows.append({'dataset':ds,'training_batch':index,'origin_first':int(o),'origin_last':int(these[-1]),'latent_gradient_norm':nl,'forecast_gradient_norm':ny,'ratio':ratio,'model_mode':'eval; initial same original Stage2 weights; no updates'})
  assert state_hash(model)==before;weights[ds]=float(np.median(ratios));print('TRAIN_ONLY_LAMBDA',ds,weights[ds],flush=True)
 value={'lambda':weights,'rule':PROTOCOL['lambda_rule'],'training_batches':rows,'validation_or_test_used':False,'protocol_sha256':sha(ROOT/'configs/protocol.json')}
 p=ROOT/'configs/calibration.json'
 if p.exists():assert json.loads(p.read_text())['lambda']==weights
 else:write_json(p,value)
 write_csv(ROOT/'outputs/gradient_scale_calibration.csv',rows)
if __name__=='__main__':run()
