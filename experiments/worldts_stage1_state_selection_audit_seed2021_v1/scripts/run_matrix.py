"""Run the six fixed Decoder fits once, resuming completed cells."""
import csv, json, pathlib, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
CELLS=[('NP',1),('NP',3),('NP',6),('PJM',1),('PJM',3),('PJM',6)]
STATUS=ROOT/'outputs/formal_matrix_status.json'

def write_status(state):
    temporary=STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
    temporary.replace(STATUS)

def finish(ds,ep):
    p=ROOT/'outputs/checkpoints'/ds/f'decoder_from_stage1_epoch{ep}'/'complete.json'
    if not p.is_file():return None
    result=json.loads(p.read_text())
    assert result['decoder_epochs']==20 and result['optimizer_steps']==22240
    return result

def main():
    check=json.loads((ROOT/'outputs/test_results.json').read_text())
    assert check['failed']==0 and check['passed']>=48
    protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    assert protocol['stage2_epochs']==20 and protocol['stage1_selection_epochs']==[1,3,6]
    state={'stage':'waiting_for_first_formal_fit','completed':[],'all_six_locked':False}
    write_status(state)
    # NP/Epoch-1 was launched directly after the preflight. This wait prevents
    # a second process from touching its checkpoints while it is still active.
    deadline=time.monotonic()+4*3600
    while finish('NP',1) is None:
        assert time.monotonic()<deadline,'First fit did not complete within four hours'
        log=ROOT/'outputs/logs/decoder_NP_stage1ep1_attempt1.log'
        if log.is_file() and 'Traceback (most recent call last)' in log.read_text(errors='replace'):
            raise RuntimeError('First formal fit failed; inspect its log')
        time.sleep(10)
    for ds,ep in CELLS:
        if finish(ds,ep) is not None:
            state['completed'].append(f'{ds}_{ep}')
            write_status(state)
            continue
        state['stage']=f'running_{ds}_{ep}'
        write_status(state)
        log=ROOT/'outputs/logs'/f'decoder_{ds}_stage1ep{ep}_attempt1.log'
        with log.open('w',encoding='utf-8') as f:
            run=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/decoder_fit.py'),
                                '--dataset',ds,'--stage1-epoch',str(ep)],
                               stdout=f,stderr=subprocess.STDOUT,check=False)
        if run.returncode:
            state['stage']=f'failed_{ds}_{ep}';state['exit_code']=run.returncode
            write_status(state)
            raise RuntimeError(f'{ds} Stage1 epoch{ep} Decoder fit failed; inspect {log}')
        assert finish(ds,ep) is not None
        state['completed'].append(f'{ds}_{ep}')
        write_status(state)
    for ds in ['NP','PJM']:
        summaries=[finish(ds,e) for e in [1,3,6]]
        assert len({x['decoder_init_tensor_hash'] for x in summaries})==1
        curves=[list(csv.DictReader((ROOT/'outputs/checkpoints'/ds/
                                     f'decoder_from_stage1_epoch{ep}'/'curves.csv').open(newline='',encoding='utf-8')))
                for ep in [1,3,6]]
        assert all(len(x)==21 for x in curves)
        for epoch in range(1,21):
            assert len({x[epoch]['data_order_sha256'] for x in curves})==1,(ds,epoch)
            assert all(int(x[epoch]['optimizer_steps'])==epoch*1112 for x in curves)
    state['stage']='complete';state['all_six_locked']=True
    write_status(state)
    print('ALL_SIX_FORMAL_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
