"""Sequential, restartable execution; never launches all models concurrently."""
from common import *
import argparse,subprocess,shutil
def run(model,dataset,h,backbone=None):
 script={'LatentTSF':'run_latenttsf.py','DAG':'run_dag.py','WorldTS':'run_worldts.py'}[model]
 group={'LatentTSF':'latenttsf','DAG':'dag','WorldTS':'worldts'}[model]
 rid=f'{dataset}_{backbone}_H{h}_seed42' if model=='LatentTSF' else f'{dataset}_H{h}_seed2021'
 out=ROOT/'results'/group/rid;out.mkdir(parents=True,exist_ok=True);log=out/'train.log'
 if (out/'metrics.json').is_file():
  metrics=json.loads((out/'metrics.json').read_text());assert sha(metrics['checkpoint_path'])==metrics['checkpoint_sha256'];print('VERIFIED_COMPLETED_SKIP',rid,metrics['status'],flush=True);return metrics
 argv=[sys.executable,'-u',str(ROOT/'scripts'/script),'--dataset',dataset,'--horizon',str(h)]
 if backbone:argv+=['--backbone',backbone]
 print('START',model,rid,flush=True)
 log_id=model+'_'+rid;attempts=list((ROOT/'logs').glob(log_id+'_attempt*.log'));attempt=ROOT/'logs'/(log_id+'_attempt'+str(len(attempts)+1)+'.log');attempt.parent.mkdir(parents=True,exist_ok=True)
 with attempt.open('wb') as f:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 shutil.copy2(attempt,log)
 if result.returncode:raise RuntimeError('Run failed: '+rid+'; retained log: '+str(attempt))
 metrics=json.loads((out/'metrics.json').read_text());print('FINISH',rid,metrics['status'],metrics['reproduced_mse'],metrics['reproduced_mae'],flush=True)
 rows=[json.loads(p.read_text()) for p in (ROOT/'results').glob('*/*/metrics.json')]
 write_json(ROOT/'STATUS.json',{'phase':model,'status':'RUNNING','last_completed':rid,'completed_configs':len(rows),'total_configs':24})
 return metrics
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','latenttsf','dag','worldts'],default='all');a=p.parse_args()
 if a.phase in ['all','latenttsf']:
  for ds,bb in [('ETTh1','DLinear'),('Electricity','iTransformer')]:
   row=run('LatentTSF',ds,96,bb)
   if row['relative_mse_diff']>.1:raise RuntimeError('Smoke discrepancy >10%; investigate before remaining configurations')
  for ds in ['ETTh1','Electricity']:
   for bb in ['DLinear','iTransformer']:
    for h in [96,192,336,720]:run('LatentTSF',ds,h,bb)
 if a.phase in ['all','dag']:
  for ds in ['NP','PJM']:
   for h in [24,360]:run('DAG',ds,h)
 if a.phase in ['all','worldts']:
  assert len(list((ROOT/'results/dag').glob('*/metrics.json')))==4,'Verify DAG first'
  for ds,h in [('PJM',24),('NP',24),('PJM',360),('NP',360)]:run('WorldTS',ds,h)
 print('PHASE_FINISHED',a.phase,flush=True)
