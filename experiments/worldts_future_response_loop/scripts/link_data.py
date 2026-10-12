from common import *
import shutil
dst=ROOT/'upstream/WorldTS/DAG-bench/dataset/forecasting';dst.mkdir(parents=True,exist_ok=True)
for ds in ['NP','PJM']:
    p=dst/(ds+'.csv')
    if not p.exists():shutil.copy2(BASE/'data'/(ds+'.csv'),p)
    assert sha(p)==DATA_SHA[ds]
old=BASE/'upstream/WorldTS/DAG-bench/dataset/forecasting/FORECAST_META.csv';shutil.copy2(old,dst/'FORECAST_META.csv')
print('INDEPENDENT_DATA_COPIES_VERIFIED')
