"""Foreground, heartbeat-visible execution with per-case resume and checked exits."""
import datetime,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    from cpu_affinity import apply_task_affinity
    apply_task_affinity()
    from length_probe_plan import ensure_plan
    plan=ensure_plan(ROOT)
    stages=[('length_'+mode+'_'+str(i),['scripts/baseline_length_isolated.py','--method',mode,'--sample-id',sample]) for mode in ['standard','forced1','forced2','adaptive'] for i,sample in enumerate(plan['selected_ids'])]
    stages += [('depth_extension',['scripts/core_audit.py','--phase','depths_aligned'])]
    stages += [('analyze_'+p,['scripts/analyze_results.py','--phase',p]) for p in ['dev','audit','rename','dev_aligned','audit_aligned','rename_aligned','depths_aligned']]
    stages += [('tests',['-m','pytest','-q']),('artifact_checks',['scripts/artifact_checks.py','--require-raw-logits']),('report',['scripts/generate_report.py']),('provenance',['scripts/capture_provenance.py'])]
    logs=ROOT/'outputs/logs';logs.mkdir(exist_ok=True)
    for name,args in stages:
        began=time.perf_counter();path=logs/(name+'_foreground.log')
        with path.open('ab') as out:
            out.write(('\nRESUME_STAGE '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n').encode());out.flush()
            child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            print('STAGE_START',name,'PID',child.pid,flush=True)
            while child.poll() is None:
                time.sleep(20)
                print('STAGE_HEARTBEAT',name,'seconds',round(time.perf_counter()-began,1),flush=True)
        code=child.returncode;print('STAGE_EXIT',name,code,flush=True)
        if code:
            print(path.read_bytes()[-6000:].decode('utf8',errors='replace'),flush=True);raise SystemExit(code)
    print('ALL_REMAINING_STAGES_COMPLETE',flush=True)
if __name__=='__main__':main()
