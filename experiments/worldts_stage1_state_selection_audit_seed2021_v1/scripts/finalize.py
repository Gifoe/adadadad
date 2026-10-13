"""Wait for six locked fits and run the prespecified post-fit audit steps."""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs';STATUS=OUT/'finalization_status.json'

def write(stage,completed):
    tmp=STATUS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'stage':stage,'completed':completed},indent=2),encoding='utf-8')
    tmp.replace(STATUS)

def run():
    completed=[];write('waiting_for_six_fits',completed)
    deadline=time.monotonic()+8*3600
    while True:
        matrix=OUT/'formal_matrix_status.json'
        if matrix.is_file():
            state=json.loads(matrix.read_text())
            if state.get('stage')=='complete' and state.get('all_six_locked'):break
            if str(state.get('stage','')).startswith('failed_'):
                raise RuntimeError('Formal matrix failed: '+json.dumps(state))
        assert time.monotonic()<deadline,'Formal matrix did not finish in eight hours'
        time.sleep(15)
    ordered=[('postformal_checks','tests/run_checks.py'),
             ('final_diagnostics','scripts/final_diagnostics.py'),
             ('seal_checkpoints','scripts/seal_checkpoints.py'),
             ('seven_figures','scripts/make_figures.py'),
             ('final_reports','scripts/build_final_report.py')]
    for name,relative in ordered:
        write('running_'+name,completed)
        log=OUT/'logs'/f'finalize_{name}.log'
        with log.open('w',encoding='utf-8') as f:
            code=subprocess.run([sys.executable,'-u',str(ROOT/relative)],
                                stdout=f,stderr=subprocess.STDOUT,check=False).returncode
        if code:
            write('failed_'+name,completed)
            raise RuntimeError(f'{name} exit={code}; inspect {log}')
        completed.append(name);write('completed_'+name,completed)
        print('FINALIZE_STEP_COMPLETE',name,flush=True)
    write('complete',completed)
    print('FINALIZATION_COMPLETE',flush=True)

if __name__=='__main__':run()
