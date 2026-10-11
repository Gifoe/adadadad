from common import *
import argparse,requests,shutil
import pandas as pd
def audit():
 rows=[]
 for name in DATA_SHA:
  rel={'ETTh1':'ETT-small/ETTh1.csv','Electricity':'electricity/electricity.csv'}.get(name,name+'.csv');p=ROOT/'data'/rel
  assert p.is_file(),str(p)
  assert sha(p)==DATA_SHA[name],name+' SHA mismatch'
  df=pd.read_csv(p)
  if name in ['NP','PJM']:
   source_cols=df['cols'].drop_duplicates().tolist();df['date']=pd.to_datetime(df['date']);df=df.pivot(index='date',columns='cols',values='data');cols=source_cols;assert cols[-1]=='OT';df=df[cols];n=len(df);tv=int(n*.8);tr=int(tv*.875);splits=[tr,tv,n]
   times=pd.to_datetime(df.index);channels=3;assert n==52416
  else:
   times=pd.DatetimeIndex(pd.to_datetime(df.date));n=len(df);channels=7 if name=='ETTh1' else 321;assert df.shape[1]==channels+1
   if name=='ETTh1':assert n==17420;splits=[8640,11520,14400]
   else:assert n==26304;splits=[int(n*.7),n-int(n*.2),n]
  assert times.is_unique and times.is_monotonic_increasing
  assert (times[1:].to_numpy()-times[:-1].to_numpy()==np.timedelta64(1,'h')).all()
  values=df.drop(columns=['date']).to_numpy() if 'date' in df.columns else df.to_numpy()
  assert np.isfinite(values).all()
  rows.append({'dataset':name,'path':str(p),'sha256':sha(p),'shape':[n,channels],'columns':list(df.columns),'time_start':str(times[0]),'time_end':str(times[-1]),'split_endpoints':splits,'future_exogenous_vintage':'unknown; official benchmark availability' if name in ['NP','PJM'] else None})
 write_json(ROOT/'reports/data_audit.json',rows);print('DATA_AUDIT_PASSED',[(r['dataset'],r['shape']) for r in rows],flush=True)
 return rows
def download():
 rev='2b66e59ee19dac8f6f19fb5d4997f289fdfea357';session=requests.Session();session.trust_env=False
 for name,rel in [('ETTh1','ETT-small/ETTh1.csv'),('Electricity','electricity/electricity.csv')]:
  p=ROOT/'data'/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if p.is_file() and sha(p)==DATA_SHA[name]:continue
  errors=[]
  for endpoint in ['https://hf-mirror.com','https://huggingface.co']:
   url=endpoint+'/datasets/thuml/Time-Series-Library/resolve/'+rev+'/'+rel
   try:
    with session.get(url,stream=True,timeout=60) as r:
     r.raise_for_status()
     with p.with_suffix('.part').open('wb') as f:
      for b in r.iter_content(1024*1024):f.write(b)
    assert sha(p.with_suffix('.part'))==DATA_SHA[name];p.with_suffix('.part').replace(p)
    print('DOWNLOAD_VERIFIED',name,str(p),flush=True);break
   except Exception as e:errors.append(str(e));print('DOWNLOAD_FAILED',url,repr(e),flush=True)
  else:raise RuntimeError(errors)
 for name in ['NP','PJM','FORECAST_META']:
  dst=ROOT/'data'/(name+'.csv')
  previous=pathlib.Path('D:/conditional_latent_loop_seed2021_v1/data')/(name+'.csv')
  if not dst.exists() and previous.is_file():shutil.copy2(previous,dst)
 if not all((ROOT/'data'/(name+'.csv')).is_file() for name in ['NP','PJM','FORECAST_META']):
  from download_dag_data import download as download_official_dag
  download_official_dag()
 if (ROOT/'data/NP.csv').is_file():
  for upstream in ['DAG','WorldTS/DAG-bench']:
   dst=ROOT/'upstream'/upstream/'dataset/forecasting';dst.mkdir(parents=True,exist_ok=True)
   for name in ['NP','PJM','FORECAST_META']:shutil.copy2(ROOT/'data'/(name+'.csv'),dst/(name+'.csv'))
 audit()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args();audit() if a.verify_only else download()
