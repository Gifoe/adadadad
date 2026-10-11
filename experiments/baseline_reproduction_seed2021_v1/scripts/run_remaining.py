"""Complete/resume all baselines, close the old NP24 log gap, audit and report."""
from common import *
import subprocess,shutil
def call(script,*args,log_prefix):
 argv=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)];logs=ROOT/'logs';logs.mkdir(exist_ok=True);log=logs/(log_prefix+'_attempt'+str(len(list(logs.glob(log_prefix+'_attempt*.log')))+1)+'.log')
 start=time.time()
 with log.open('wb') as f:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 write_json(log.with_suffix('.json'),{'command':argv,'start_time':start,'end_time':time.time(),'exit_code':result.returncode,'log_sha256':sha(log)})
 print('COMMAND_EXIT',script,result.returncode,str(log),flush=True);assert result.returncode==0,str(log)
def main():
 call('download_and_verify_data.py','--verify-only',log_prefix='final_data_verify')
 call('run_all.py','--phase','latenttsf',log_prefix='resume_latenttsf')
 if len(list((ROOT/'results/worldts').glob('*/metrics.json')))!=4:
  call('run_all.py','--phase','dag',log_prefix='resume_dag_verification')
  call('run_all.py','--phase','worldts',log_prefix='resume_worldts')
 # Preserve the already verified old result; fresh training closes its missing stdout.
 out=ROOT/'results/dag/NP_H24_seed2021';metrics=out/'metrics.json'
 if metrics.is_file() and json.loads(metrics.read_text())['run_type']=='verified_reuse':
  archive=ROOT/'reports/dag_verified_reuse_archive/NP_H24_seed2021';archive.parent.mkdir(parents=True,exist_ok=True)
  if not archive.exists():shutil.copytree(out,archive)
  assert sha(archive/'checkpoint.pt')==sha(out/'checkpoint.pt');metrics.replace(out/'verified_reuse_metrics.json')
 if not metrics.is_file():call('run_dag_fresh.py','--dataset','NP','--horizon','24',log_prefix='DAG_fresh_NP_H24_seed2021')
 assert json.loads(metrics.read_text())['run_type']=='fresh_run'
 call('worldts_post_audit.py',log_prefix='worldts_trained_audit')
 call('setup_environment.py',log_prefix='final_environment_audit')
 call('finalize_results.py',log_prefix='final_verification_and_report')
 print('ALL_REQUIRED_EXPERIMENTS_AND_EVIDENCE_COMPLETE',flush=True)
if __name__=='__main__':main()
