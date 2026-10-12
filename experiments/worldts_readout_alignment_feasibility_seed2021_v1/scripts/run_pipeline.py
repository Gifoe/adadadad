import pathlib,sys,json,time,subprocess,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def save(path,value):
 p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
 start=time.time();rows=[];frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/ops.py',ROOT/'scripts/train.py',ROOT/'configs/protocol.json']}
 assert json.loads((ROOT/'outputs/preflight_metrics.json').read_text())['passed']
 assert (ROOT/'outputs/alignment_summary.json').is_file()
 commands=[('audit_report',['analyze.py','--audit-only']),('calibration',['calibrate.py']),('smoke',['train.py','--smoke'])]+[(f'formal_{ds}_{m}',['train.py','--dataset',ds,'--method',m]) for ds in ['NP','PJM'] for m in ['A0','A1','A2']]+[('post_audit',['post_audit.py']),('final_report',['analyze.py'])]
 for name,args in commands:
  for file,digest in frozen.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,'Training protocol/code changed while running'
  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat();log=ROOT/'outputs/logs'/f'{name}_attempt1.log';argv=[sys.executable,'-u',str(ROOT/'scripts'/args[0])]+args[1:];begin=time.time();print('PIPELINE_START',name,timestamp,flush=True);save('STATUS.json',{'phase':'pipeline','status':'RUNNING','active':name,'overall_complete':False})
  with log.open('w',encoding='utf-8') as f:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  rows.append({'name':name,'command':argv,'started_utc':timestamp,'seconds':time.time()-begin,'exit_code':result.returncode,'log':str(log.relative_to(ROOT))});save('outputs/execution_commands.json',{'steps':rows,'seconds':time.time()-start,'training_code_hashes':frozen})
  if result.returncode:
   save('STATUS.json',{'status':'FAILED','active':name,'exit_code':result.returncode,'overall_complete':False});raise RuntimeError('Pipeline step failed: '+name)
  if name=='smoke':assert json.loads((ROOT/'outputs/smoke_complete.json').read_text())['passed']
  print('PIPELINE_DONE',name,rows[-1]['seconds'],flush=True)
 save('STATUS.json',{'status':'COMPLETE','phase':3,'overall_complete':True,'formal_fits':6,'seconds':time.time()-start});print('ALL_PHASES_COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
