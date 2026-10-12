"""Summarize observed epoch gradient norms; no new training or model selection."""
import csv,json,math,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]

def run():
    rows=[]
    for metric in sorted((ROOT/'runs').glob('*/metrics.json')):
        m=json.loads(metric.read_text())
        with (metric.parent/'history.csv').open(newline='') as f:
            history=list(csv.DictReader(f))
        keys=sorted({k for r in history for k in r if k.startswith('gradient_mean_norm_')})
        for key in keys:
            values=[float(r[key]) for r in history if r.get(key) and r.get('stage')=='stage1_latent']
            if not values:continue
            assert all(math.isfinite(v) and v>=0 for v in values),(metric,key)
            name=key.removeprefix('gradient_mean_norm_')
            block=name.split('response_blocks.',1)[1].split('.',1)[0] if 'response_blocks.' in name else 'gate_or_scalar'
            rows.append({'run_id':m['run_id'],'dataset':m['dataset'],'horizon':m['horizon'],'variant':m['variant'],'seed':m['seed'],'parameter':name,'response_block':block,'observed_epochs':len(values),'positive_epochs':sum(v>0 for v in values),'min_epoch_mean_gradient_norm':min(values),'median_epoch_mean_gradient_norm':statistics.median(values),'max_epoch_mean_gradient_norm':max(values),'finite':True})
    if rows:
        with (ROOT/'results/response_gradient_audit.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows

if __name__=='__main__':
    rows=run()
    print(json.dumps({'parameter_rows':len(rows),'all_finite':all(r['finite'] for r in rows),'all_observed_parameters_receive_positive_gradient':all(r['positive_epochs']>0 for r in rows)}))
