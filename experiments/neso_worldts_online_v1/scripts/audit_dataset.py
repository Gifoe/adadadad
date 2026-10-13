import hashlib,json,pathlib,sys
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.native_worldts import file_sha,read_arrays,train_scaler
from neso_online.prepare import _utc_end_from_settlement

def main():
    base=ROOT/'outputs/dataset';manifest=json.loads((base/'manifest.json').read_text())
    data=read_arrays(base);coverage={};sets={};revision=[]
    for split,values in data.items():
        n=len(values['history']);assert n>0,split+' has no genuine vintage windows'
        assert values['history'].shape==(n,336,3) and values['target'].shape==(n,48,1)
        assert values['cov_a'].shape==(n,48,2) and values['cov_b'].shape==(n,48,2)
        for key in ['history','target','cov_a','cov_b']:assert np.isfinite(values[key]).all()
        a=pd.to_datetime(values['issue_a'],utc=True);b=pd.to_datetime(values['issue_b'],utc=True)
        targets=values['target_end_utc_ns'];hist=values['history_end_utc_ns'];mask=values['update_mask']
        assert np.array_equal(mask,targets>b.asi8[:,None])
        assert np.array_equal(values['cov_a'][~mask],values['cov_b'][~mask])
        assert np.all(targets[:,0]>a.asi8) and np.all(hist[:,-1]<=a.asi8)
        assert np.all(np.diff(targets,axis=1)==1800*10**9) and np.all(np.diff(hist,axis=1)==1800*10**9)
        assert np.array_equal(hist[:,0],targets[:,0]-336*1800*10**9)
        assert np.all((b-a)>=pd.Timedelta(hours=6)) and np.all((b-a)<=pd.Timedelta(hours=8))
        sets[split]=set(targets.ravel().tolist())
        delta=values['cov_b']-values['cov_a'];rms=np.sqrt(np.sum(delta.astype('float64')**2,axis=(1,2))/(mask.sum(1)*2))
        coverage[split]={'n_windows':n,'n_unique_target_periods':len(sets[split]),
                         'n_distinct_A_issues':len(set(values['issue_a'])),'n_distinct_B_issues':len(set(values['issue_b'])),
                         'first_target_utc':str(pd.Timestamp(int(targets.min()),tz='UTC')),
                         'last_target_utc':str(pd.Timestamp(int(targets.max()),tz='UTC')),
                         'A_first':str(a.min()),'A_last':str(a.max()),
                         'score_slots_min':int(mask.sum(1).min()),'score_slots_max':int(mask.sum(1).max()),
                         'revised_window_fraction':float(np.mean(np.any(delta!=0,axis=(1,2)))),
                         'revision_rms_mw_quantiles':np.quantile(rms,[0,.25,.5,.75,1]).tolist()}
        if split=='train':
            edges=np.quantile(rms,[.25,.5,.75]).tolist()
            (ROOT/'outputs/revision_bin_edges.json').write_text(json.dumps({'fit_split':'train','rms_covariate_revision_mw_edges':edges},indent=2),encoding='utf-8')
    for left,right in [('train','val'),('train','test'),('val','test')]:assert not sets[left].intersection(sets[right])
    parts=[]
    for year in [2025,2026]:
        part=pd.read_csv(ROOT/'data/raw'/f'demand_{year}.csv')
        if 'FORECAST_ACTUAL_INDICATOR' not in part:
            assert year==2025;part['FORECAST_ACTUAL_INDICATOR']='A'
        parts.append(part)
    demands=pd.concat(parts,ignore_index=True)
    actual=demands[demands['FORECAST_ACTUAL_INDICATOR'].astype(str).str.upper()=='A'].copy()
    actual['utc_end']=_utc_end_from_settlement(actual['SETTLEMENT_DATE'],actual['SETTLEMENT_PERIOD'])
    raw_date=actual.SETTLEMENT_DATE.astype(str).str[:10]
    day_counts=actual.groupby(raw_date).SETTLEMENT_PERIOD.agg(['count','min','max'])
    dst_rows=[]
    for date in ['2025-03-30','2025-10-26','2026-03-29','2026-10-25']:
        if date in day_counts.index:
            item=day_counts.loc[date];dst_rows.append({'date':date,**{key:int(item[key]) for key in ['count','min','max']}})
    sources=[]
    for path in sorted((ROOT/'data/raw').glob('*.csv')):
        metadata=json.loads(path.with_suffix('.metadata.json').read_text());assert file_sha(path)==metadata['sha256']
        sources.append(metadata)
    scaler=train_scaler(data['train'])
    report={'manifest':manifest,'coverage':coverage,'disjoint_target_support':True,
            'fixed_A_history':True,'strict_future_mask_and_unchanged_prefix':True,
            'seasonal_naive_alignment':'336 continuous UTC half-hour slots; history index0 matches target lead1 one week earlier',
            'real_dst_actual_day_counts':dst_rows,'actual_day_count_distribution':day_counts['count'].value_counts().sort_index().to_dict(),
            'downloaded_sources':sources,'normalization_fit_split':'TRAIN history only','train_scaler_reference':scaler,
            'removed_source':'d6375700-69c2-4c25-8bde-883a205d742e returns official404 and is absent from current package resources; no substitute CSV fabricated',
            'availability_claim':'retrospective forecast-vintage replay; instantaneous historicND availability unverified; production timestamp is not verified public-release time'}
    (ROOT/'outputs/data_audit.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    lines=['# Real NESO data audit','',report['availability_claim'],'',
           '| Split | Genuine windows | Unique target half-hours | First target UTC | Last target UTC | Revised windows |',
           '|---|---:|---:|---|---|---:|']
    for split,entry in coverage.items():lines.append(f'| {split} | {entry["n_windows"]} | {entry["n_unique_target_periods"]} | {entry["first_target_utc"]} | {entry["last_target_utc"]} | {entry["revised_window_fraction"]:.2%} |')
    lines+=['',report['removed_source'],'',
            'The requested alias `forecast_2026_jun_dec` now points to the official resource named "Embedded Solar and Wind Forecast Archive 2026". Actual per-file coverage is recorded below; it is not assumed from the alias.',
            '',json.dumps(manifest['forecast_stats'],indent=2),'',
            'All retained rows have matching GMT and local-settlement endpoints. The complete raw2025/2026 pass identified1268 still-future2026 rows with conflicting endpoint keys, concentrated in late-April/early-May target dates; those ambiguous rows are excluded without choosing an endpoint. Raw missingness and conflict counts are in the manifest and timestamp_failure_audit.json. All arrays are finite. A/B use exactly the same target timestamps; completed periods are unchanged; all histories end by A. Future target supports are disjoint across splits. Train-only normalization is independently recorded.','',
            'Actual DST settlement-day counts: '+json.dumps(dst_rows),'',
            'Seasonal naive uses a fixed UTC lag of336 half-hour slots. Around DST this differs from matching a local-clock weekday; that is the supplied V1 definition.']
    (ROOT/'DATA_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REAL_DATA_AUDIT_PASS',json.dumps(coverage),flush=True)
if __name__=='__main__':main()
