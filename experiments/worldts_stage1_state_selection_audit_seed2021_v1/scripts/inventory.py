import pathlib,sys,json,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *
def tensors_sha(values):
 h=hashlib.sha256()
 for k,v in sorted(values.items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def run():
 records=[];tasks=[];protected=[]
 for ds in ['NP','PJM']:
  cfg,dr,meta,cp=config(ds);old=dr/'resume_stage1_latent.pt';assert old.is_file();raw=torch.load(old,map_location='cpu',weights_only=False);best=torch.load(dr/'stages/stage1_latent.pt',map_location='cpu',weights_only=False)['Model'];assert raw['next_epoch']==6 and len(raw['history'])==6
  assert tensors_sha(best)==tensors_sha(raw['best_state']);assert tensors_sha(best)!=tensors_sha(raw['model'])
  model=WorldTSModel(cfg);model.load_state_dict(best,strict=True);model.load_state_dict(raw['model'],strict=True)
  decoder={k:v for k,v in best.items() if k.startswith('decoder.')};assert all(torch.equal(v,raw['model'][k]) for k,v in decoder.items())
  destination=ROOT/'outputs/checkpoints'/ds;destination.mkdir(parents=True,exist_ok=True)
  for e,values,identity in [(1,best,'original early-stop best, verified against best_state and epoch history'),(6,raw['model'],'raw epoch6 resume model, before best restoration')]:
   p=destination/f'original_stage1_epoch{e}.pt';torch.save({'Model':values,'epoch':e,'source':str(old if e==6 else dr/'stages/stage1_latent.pt'),'identity':identity},p);records.append({'dataset':ds,'stage':'stage1','epoch':e,'status':'RECOVERED_ORIGINAL','path':str(p),'source_path':str(old if e==6 else dr/'stages/stage1_latent.pt'),'sha256':sha(p),'tensor_hash':tensors_sha(values),'optimizer_available':e==6,'rng_available':e==6,'is_raw_epoch_before_best_restore':e==6,'identity':identity})
  p=destination/'untrained_decoder.pt';torch.save({'Decoder':{k[len('decoder.'):]:v for k,v in decoder.items()},'source':str(dr/'stages/stage1_latent.pt'),'stage1_frozen':True},p)
  tasks.append({'dataset':ds,'epoch3_missing':True,'raw_epoch6_available':True,'best_epoch':raw['history'][0]['epoch'],'historical_loss_epoch1':raw['history'][0]['validation_loss'],'historical_loss_epoch6':raw['history'][-1]['validation_loss'],'untrained_decoder':str(p),'untrained_decoder_tensor_hash':tensors_sha({k[len('decoder.'):]:v for k,v in decoder.items()}),'raw_resume_keys':list(raw),'history':raw['history']})
  for f in [old,dr/'stages/stage1_latent.pt',cp,dr/'resolved_config.json',dr/'history.csv',dr/'scaler.npz',BASE/'scripts/run_worldts.py',BASE/'data'/(ds+'.csv')]:protected.append({'path':str(f),'sha256':sha(f)})
  # Exact raw Stage1 checkpoints saved elsewhere in the baseline result subtree.
  for f in dr.rglob('*.pt'):records.append({'dataset':ds,'stage':'historical_file','epoch':'unknown_unless_payload_proves_identity','status':'INVENTORIED','path':str(f),'sha256':sha(f),'bytes':f.stat().st_size})
 write_csv(ROOT/'outputs/checkpoint_inventory.csv',records);write_json(ROOT/'outputs/checkpoint_hashes.json',{'inventory':records,'tasks':tasks,'protected_inputs':protected,'missing_epoch3_requires_replay':True});print('INVENTORY_COMPLETE_RAW6_BEST1_NO3',flush=True)
if __name__=='__main__':run()
