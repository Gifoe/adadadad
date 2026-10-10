from common import *
from ts_benchmark.recording import read_record_file
import time

def main():
    rows=[];agreements=[];official_rows=[]
    for name in ('NP','PJM'):
        records=[]
        for p in (ROOT/'upstream/DAG/result'/name/'DAG').glob('*.tar.gz'):
            df=read_record_file(str(p))
            for _,row in df.iterrows():records.append((p,row))
        for h in (24,360):
            out=ROOT/'runs'/f'{name}_{h}_B0';r=json.loads((out/'result.json').read_text())
            valid=[]
            for p,row in records:
                params=json.loads(row['model_params']);strategy=json.loads(row['strategy_args'])
                if strategy['horizon']==h and np.isfinite(row['mse_norm']):valid.append((p,row))
            assert len(valid)==1, f'Missing or ambiguous original official records: {name}/{h}'
            p,row=valid[0];mse=float(row['mse_norm']);mae=float(row['mae_norm'])
            official_rows.append({key:row[key] for key in row.index if key not in ('actual_data','inference_data')})
            assert abs(mse-r['mse'])<2e-5 and abs(mae-r['mae'])<2e-5, 'Metric protocol mismatch'
            r['official_runner_mse']=mse;r['official_runner_mae']=mae;r['official_record_path']=str(p)
            json_write(out/'result.json',r);rows.append({k:r[k] for k in ['dataset','horizon','variant','mse','mae','reference_mse','reference_mae','mse_relative_difference','epochs','train_seconds','trainable_parameters','peak_cuda_bytes','official_runner_mse','official_runner_mae']})
            agreements.append({'dataset':name,'horizon':h,'mse_difference':mse-r['mse'],'mae_difference':mae-r['mae'],'reference_relative_mse_difference':r['mse_relative_difference']})
    pd.DataFrame(rows).to_csv(ROOT/'baseline_results.csv',index=False)
    pd.DataFrame(official_rows).to_csv(ROOT/'official_metric_records.csv',index=False)
    json_write(ROOT/'baseline_verification.json',{'passed':True,'metric_agreement_tolerance':2e-5,'checks':agreements,
        'significant_difference_flag':[a for a in agreements if abs(a['reference_relative_mse_difference'])>.2],
        'investigation':['Exact published commands and architecture; exact pinned official dataset bytes','Official TFB normalized metrics independently matched using training-only scaler and identical rolling origins','Current sm120-compatible torch2.8/numpy1.26/sklearn1.7/pandas2.3 differ from published environment; single-seed values are not rounded-reference guarantees','No test-selected retuning allowed; large remaining numeric deviations must be flagged in the final report']})
    print('BASELINE_PROTOCOL_VERIFIED',agreements,flush=True)

if __name__=='__main__':main()
