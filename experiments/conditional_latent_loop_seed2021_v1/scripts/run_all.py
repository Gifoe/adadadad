from common import *
import subprocess,time,traceback

def run(script,args=()):
    label=script.replace('.py','')+'_'+'_'.join(str(v) for v in args if not str(v).startswith('--'))
    log=ROOT/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)]
    json_write(ROOT/'STATUS.json',{'state':'running','current':label,'command':command,'log':str(log)})
    print('START',label,flush=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\nATTEMPT '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');f.flush()
        completed=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError(f'{label} failed ({completed.returncode}); inspect {log}')
    print('FINISHED',label,flush=True)

def main():
    # If an externally launched official first run is active, wait for it instead of duplicating training.
    import psutil
    while True:
        busy=[]
        for p in psutil.process_iter(['pid','cmdline']):
            try:
                c=' '.join(p.info['cmdline'] or [])
                if p.info['pid']!=os.getpid() and 'python' in c.lower() and 'run_official_dag.py --dataset NP --horizon 24' in c:busy.append(p.info['pid'])
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        if not busy:break
        print('WAIT_FIRST_OFFICIAL_RUN',busy,flush=True);time.sleep(10)
    audit_data()
    for name in ('NP','PJM'):
        for h in (24,360):run('run_official_dag.py',['--dataset',name,'--horizon',h])
    run('verify_baselines.py')
    run('train_ae.py');run('train_models.py',['--sanity'])
    for name in ('NP','PJM'):
        for h in (24,360):
            for v in ('B1','B2','B3','B4','B5'):run('train_models.py',['--dataset',name,'--horizon',h,'--variant',v,'--k',2])
    for name in ('NP','PJM'):
        for v in ('B3','B4','B5'):run('train_models.py',['--dataset',name,'--horizon',360,'--variant',v,'--k',4])
    run('analyze_results.py')
    json_write(ROOT/'STATUS.json',{'state':'complete','formal_runs':30,'AE_runs':2,'report':'FINAL_REPORT.md'})

if __name__=='__main__':
    try:main()
    except Exception as e:
        json_write(ROOT/'STATUS.json',{'state':'failed','error':str(e),'traceback':traceback.format_exc()});raise
