from common import *

def run():
 assert json.loads((ROOT/'STATUS.json').read_text())['overall_complete']
 files=[]
 for p in ROOT.rglob('*'):
  relative=p.relative_to(ROOT)
  upstream_record='upstream' in relative.parts and 'result' in relative.parts
  if p.is_file() and '__pycache__' not in relative.parts and ('upstream' not in relative.parts or upstream_record) and p.name!='ARTIFACT_MANIFEST.json':files.append({'path':relative.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'server_only':'data' in relative.parts or 'upstream' in relative.parts or p.suffix in ['.pt','.npz','.tmp']})
 external=[]
 for ds,h in [('NP',24),('NP',360),('PJM',24),('PJM',360)]:
  p=baseline_dir(ds,h)/'stages/stage2_decoder_mix.pt';external.append({'path':str(p),'sha256':sha(p),'role':'read-only original initialization and Stage0 reference'})
 for ds in ['NP','PJM']:external.append({'path':str(BASE/'data'/(ds+'.csv')),'sha256':DATA_SHA[ds],'role':'read-only benchmark input'})
 for ds,h in [('NP',24),('NP',360),('PJM',24),('PJM',360)]:
  p=baseline_dir(ds,h)/'history.csv';external.append({'path':str(p),'sha256':sha(p),'role':'read-only original validation-score provenance'})
 write_json(ROOT/'ARTIFACT_MANIFEST.json',{'root':str(ROOT),'baseline_root':str(BASE),'source_commit':SOURCE_COMMIT,'preregistration_sha256':sha(ROOT/'configs/preregistration.json'),'files':files,'external_read_only_inputs':external,'finalized_after_orchestrator_exit':True})
 print('SEALED_ARTIFACT_MANIFEST',len(files),flush=True)

if __name__=='__main__':run()
