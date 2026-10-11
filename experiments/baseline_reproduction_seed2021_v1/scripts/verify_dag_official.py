from common import *
sys.path.insert(0,str(ROOT/'upstream/DAG'))
from ts_benchmark.recording import read_record_file
def verify():
 checks=[]
 for dataset in ['NP','PJM']:
  for h in [24,360]:
   out=ROOT/'results/dag'/f'{dataset}_H{h}_seed2021';metrics=json.loads((out/'metrics.json').read_text());old=json.loads((out/'original_result.json').read_text());p=pathlib.Path(old['official_record_path']);assert p.is_file()
   df=read_record_file(str(p));df=df[np.isfinite(df.mse_norm)];assert len(df)==1;r=df.iloc[0]
   assert json.loads(r['strategy_args'])['horizon']==h
   delta=max(abs(metrics['reproduced_mse']-float(r['mse_norm'])),abs(metrics['reproduced_mae']-float(r['mae_norm'])));assert delta<2e-5
   write_csv(out/'official_metric_record.csv',[{k:r[k] for k in r.index if k not in ['actual_data','inference_data']}]);manifest={'official_archive':str(p),'archive_sha256':sha(p),'mse_norm':float(r['mse_norm']),'mae_norm':float(r['mae_norm']),'max_abs_diff':delta};write_json(out/'original_official_record_manifest.json',manifest)
   metrics['official_mse']=float(r['mse_norm']);metrics['official_mae']=float(r['mae_norm']);metrics['independent_verification']['max_metric_abs_diff']=delta;write_json(out/'metrics.json',metrics);checks.append(dict(manifest,dataset=dataset,horizon=h))
 write_json(ROOT/'reports/dag_official_metric_verification.json',{'passed':True,'checks':checks});print('DAG_OFFICIAL_METRICS_VERIFIED',json.dumps(checks),flush=True)
if __name__=='__main__':verify()
