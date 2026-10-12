"""Read-only original validation references; do not change primary matched controls."""
import csv,hashlib,json,os,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=pathlib.Path(os.environ.get('WORLDTS_BASELINE_ROOT','D:/baseline_reproduction_seed2021_v1'))
def run():
    rows=[]
    with (ROOT/'results/stage0_baseline_verification.csv').open(newline='') as f:verified=list(csv.DictReader(f))
    for ds,h in [('NP',24),('NP',360),('PJM',24),('PJM',360)]:
        directory=BASE/'results/worldts'/f'{ds}_H{h}_seed2021'
        history=directory/'history.csv';m=json.loads((directory/'metrics.json').read_text())
        with history.open(newline='') as f:values=[r for r in csv.DictReader(f) if r['stage']=='stage2_decoder_mix']
        best=min(values,key=lambda r:float(r['validation_loss']))
        assert int(best['epoch'])==m['best_epoch']
        ref=next(r for r in verified if r['dataset']==ds and int(r['horizon'])==h)
        assert abs(float(ref['mse'])-m['reproduced_mse'])<1e-12 and abs(float(ref['mae'])-m['reproduced_mae'])<1e-12
        rows.append({'run_id':f'original_{ds}_H{h}_s2021','dataset':ds,'horizon':h,'validation_mae':float(best['validation_loss']),'stage2_best_epoch':int(best['epoch']),'test_mse':float(ref['mse']),'test_mae':float(ref['mae']),'history_path':str(history),'history_sha256':hashlib.sha256(history.read_bytes()).hexdigest(),'scope':'read-only original trained checkpoint reference, no added continuation budget; excluded from primary matched control selection'})
    with (ROOT/'results/original_validation_reference.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'read_only_original_references':len(rows)}))
    return rows
if __name__=='__main__':run()
