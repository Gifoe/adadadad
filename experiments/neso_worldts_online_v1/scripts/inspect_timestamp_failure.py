import json,pathlib,sys,collections
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.prepare import _normalize_forecast_chunk,_utc_end_from_settlement
result={}
for name in ['forecast_2025','forecast_2026_jun_dec']:
    examples=[];by_date=collections.Counter();deltas=collections.Counter();count=raw=0
    for chunk in pd.read_csv(ROOT/'data/raw'/f'{name}.csv',chunksize=250000,low_memory=False):
        raw+=len(chunk);n=_normalize_forecast_chunk(chunk,strict=False);wrong=n[n.settlement_mismatch]
        if len(wrong):
            count+=len(wrong);by_date.update(wrong.valid_end_utc.dt.strftime('%Y-%m-%d').tolist())
            delta=(wrong.valid_end_utc-_utc_end_from_settlement(chunk.loc[wrong.index,'SETTLEMENT_DATE'],chunk.loc[wrong.index,'SETTLEMENT_PERIOD'])).dt.total_seconds()/60
            deltas.update(delta.tolist())
            if len(examples)<30:
                examples.extend(chunk.loc[wrong.index[:min(5,30-len(examples))]].to_dict(orient='records'))
    result[name]={'raw_rows':raw,'mismatch_rows':count,'delta_minutes_counts':dict(deltas),'by_valid_date':dict(by_date),'examples':examples}
    print(name,json.dumps(result[name],default=str),flush=True)
(ROOT/'outputs/timestamp_failure_audit.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
