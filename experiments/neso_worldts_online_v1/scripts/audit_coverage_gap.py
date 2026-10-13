"""Inspect source vintage availability, without using any held-out target labels."""
import collections,json,pathlib,sys
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.prepare import _normalize_forecast_chunk
parts=[]
for chunk in pd.read_csv(ROOT/'data/raw/forecast_2026_jun_dec.csv',chunksize=250000,low_memory=False):
    n=_normalize_forecast_chunk(chunk,strict=False);n=n.loc[~n.settlement_mismatch]
    parts.append(n[['issue_utc','valid_end_utc']])
f=pd.concat(parts,ignore_index=True).drop_duplicates().sort_values(['issue_utc','valid_end_utc'])
groups={issue:pd.DatetimeIndex(g.valid_end_utc) for issue,g in f.groupby('issue_utc')};issues=pd.DatetimeIndex(sorted(groups))
seen=set();monthly=collections.defaultdict(collections.Counter);examples=[]
for ia in issues:
    key=ia.floor('6h')
    if key in seen:continue
    seen.add(key);month=ia.strftime('%Y-%m');monthly[month]['candidate_A']+=1
    match=issues.searchsorted(ia+pd.Timedelta(hours=6))
    if match>=len(issues) or issues[match]-ia>pd.Timedelta(hours=8):monthly[month]['no_B_in_6_to_8h']+=1;continue
    ib=issues[match];targets=pd.date_range(ia.floor('30min')+pd.Timedelta(minutes=30),periods=48,freq='30min');mask=targets>ib
    missing_a=targets.difference(groups[ia]);missing_b=targets[mask].difference(groups[ib])
    monthly[month]['full_A_24h_present']+=int(len(missing_a)==0)
    monthly[month]['A_and_B_covariates_complete']+=int(len(missing_a)==0 and len(missing_b)==0)
    if (len(missing_a) or len(missing_b)) and '2026-07-15'<=str(ia.date())<='2026-08-31' and len(examples)<25:
        examples.append({'issue_a':str(ia),'issue_b':str(ib),'wanted_first_target':str(targets[0]),'wanted_last_target':str(targets[-1]),
            'source_A_first_target':str(groups[ia].min()),'source_A_last_target':str(groups[ia].max()),
            'source_B_first_target':str(groups[ib].min()),'source_B_last_target':str(groups[ib].max()),
            'n_missing_A':len(missing_a),'n_missing_B_after_B':len(missing_b),'missing_A_examples':[str(x) for x in missing_a[:5]],
            'missing_B_examples':[str(x) for x in missing_b[:5]]})
result={'monthly_source_coverage':{k:dict(v) for k,v in sorted(monthly.items())},'missing_source_examples':examples,
    'scope':'forecast files and production timestamps only; no ND test values or predictions consulted',
    'decision':'retain original immediate24h A horizon and 6h vintage pairing; do not move horizon, invent A vintage, or backfill missing values'}
(ROOT/'outputs/coverage_gap_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
