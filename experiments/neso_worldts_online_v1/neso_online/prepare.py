"""Parse NESO vintage forecasts and demand; build same-target online revision pairs.

All times are UTC. `valid_end` denotes the END of a half-hour delivery period.
The demand label for UK SETTLEMENT_DATE/SETTLEMENT_PERIOD is converted through
Europe/London timezone-aware local midnight; do not assume 48 periods on DST days.

The experiment deliberately fixes demand history at the A issue time, and only
revises future external forecast covariates at B. It is NOT evidence that ND
would have been published at A; this is retrospective-vintage evaluation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FCAST_COLS = [
    'DATE_GMT', 'TIME_GMT', 'SETTLEMENT_DATE', 'SETTLEMENT_PERIOD',
    'EMBEDDED_WIND_FORECAST', 'EMBEDDED_SOLAR_FORECAST', 'Forecast_Datetime',
]
DEMAND_COLS = ['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND', 'FORECAST_ACTUAL_INDICATOR']
SLOT = pd.Timedelta(minutes=30)


def _utc_end_from_settlement(dates: pd.Series, periods: pd.Series) -> pd.Series:
    """Half-hour period-end UTC timestamp using real UK DST (46/48/50 slots)."""
    # NESO files may use ISO YYYY-MM-DD or UK DD/MM/YYYY. Treat slashes as
    # day-first explicitly; pandas' default silently swaps dates such as 03/04.
    raw = dates.astype(str).str.strip()
    iso = raw.str.match(r'^\d{4}-\d{2}-\d{2}')
    # SETTLEMENT_DATE is a civil calendar date, even when CKAN serializes it
    # as midnight with a Z suffix. Do not treat that suffix as London midnight.
    raw=raw.where(~iso,raw.str[:10])
    parsed_iso = pd.to_datetime(raw.where(iso),errors='coerce',format='mixed',dayfirst=False)
    parsed_uk = pd.to_datetime(raw.where(~iso),errors='coerce',format='mixed',dayfirst=True)
    parsed = parsed_iso.fillna(parsed_uk)
    numeric_periods = pd.to_numeric(periods, errors='coerce')
    local_midnight = parsed.dt.tz_localize('Europe/London')
    next_midnight = (parsed + pd.Timedelta(days=1)).dt.tz_localize('Europe/London')
    max_periods = (next_midnight.dt.tz_convert('UTC') - local_midnight.dt.tz_convert('UTC')).dt.total_seconds()/1800
    invalid = parsed.isna() | numeric_periods.isna() | (numeric_periods % 1 != 0) | (numeric_periods < 1) | (numeric_periods > max_periods)
    if invalid.any():
        raise ValueError(f'Invalid settlement date/period, including DST day bounds, at rows {invalid[invalid].index[:5].tolist()}')
    ends = local_midnight.dt.tz_convert('UTC') + pd.to_timedelta(numeric_periods.astype('float64') * 30, unit='m')
    return ends


def _normalize_forecast_chunk(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    missing = set(FCAST_COLS).difference(df.columns)
    if missing:
        raise ValueError(f'Forecast archive is missing fields: {sorted(missing)}')
    out = pd.DataFrame()
    out['issue_utc'] = pd.to_datetime(df['Forecast_Datetime'], utc=True, errors='coerce',format='mixed')
    gmt_day = pd.to_datetime(df['DATE_GMT'], utc=True, errors='coerce',format='mixed').dt.normalize()
    # DATE_GMT + TIME_GMT represent the end of the half-hour period.
    clock = df['TIME_GMT'].astype(str).str.strip()
    clock = clock.where(~clock.str.match(r'^\d{1,2}:\d{2}$'), clock + ':00')
    hour = pd.to_timedelta(clock, errors='coerce')
    out['valid_end_utc'] = gmt_day + hour
    out['wind_mw'] = pd.to_numeric(df['EMBEDDED_WIND_FORECAST'], errors='coerce')
    out['solar_mw'] = pd.to_numeric(df['EMBEDDED_SOLAR_FORECAST'], errors='coerce')
    sp = pd.to_numeric(df['SETTLEMENT_PERIOD'], errors='coerce')
    valid_from_sp = _utc_end_from_settlement(df['SETTLEMENT_DATE'], sp)
    # Some archives have inconsistent TIME_GMT / settlement-period fields;
    # a mismatch is a data-quality failure, not something to silently smooth.
    both = out['valid_end_utc'].notna() & valid_from_sp.notna()
    mismatch = both & ((out['valid_end_utc'] - valid_from_sp).abs() > pd.Timedelta(minutes=1))
    if strict and mismatch.any():
        idx = mismatch[mismatch].index[:3].tolist()
        raise ValueError(f'Forecast DATE_GMT/TIME_GMT does not match UK settlement keys at rows {idx}; inspect DST and endpoint convention')
    out['settlement_mismatch'] = mismatch.to_numpy()
    audit={'raw_settlement_time_mismatches':int(mismatch.sum()),
           'raw_missing_issue':int(out.issue_utc.isna().sum()),
           'raw_missing_gmt_endpoint':int(out.valid_end_utc.isna().sum()),
           'raw_missing_wind_or_solar':int((out.wind_mw.isna()|out.solar_mw.isna()).sum())}
    # We use GMT valid-end when provided; if null, fall back to local key.
    out['valid_end_utc'] = out['valid_end_utc'].fillna(valid_from_sp)
    out = out.dropna(subset=['issue_utc', 'valid_end_utc', 'wind_mw', 'solar_mw'])
    out = out.loc[np.isfinite(out.wind_mw) & np.isfinite(out.solar_mw)]
    out = out.loc[out['valid_end_utc'] > out['issue_utc']].copy()
    out.attrs['raw_audit']=audit
    return out


def load_forecasts(paths: Iterable[str | Path], strict: bool = True,
                   chunksize: int = 250_000,exclude_settlement_mismatch:bool=False) -> tuple[pd.DataFrame, dict]:
    chunks = []
    total_raw = total_mismatch = 0
    file_stats=[]
    for p in map(Path, paths):
        if not p.is_file():
            raise FileNotFoundError(p)
        raw_count=valid_count=excluded_mismatch=0;raw_audit={};issue_min=issue_max=target_min=target_max=None
        for chunk in pd.read_csv(p, chunksize=chunksize, low_memory=False):
            total_raw += len(chunk)
            raw_count += len(chunk)
            n = _normalize_forecast_chunk(chunk, strict=strict and not exclude_settlement_mismatch)
            for key,value in n.attrs['raw_audit'].items():raw_audit[key]=raw_audit.get(key,0)+value
            total_mismatch += int(n.attrs['raw_audit']['raw_settlement_time_mismatches'])
            if exclude_settlement_mismatch:
                excluded_mismatch+=int(n.settlement_mismatch.sum());n=n.loc[~n.settlement_mismatch]
            chunks.append(n.drop(columns=['settlement_mismatch']))
            valid_count += len(n)
            if len(n):
                issue_min=min(issue_min,n.issue_utc.min()) if issue_min is not None else n.issue_utc.min()
                issue_max=max(issue_max,n.issue_utc.max()) if issue_max is not None else n.issue_utc.max()
                target_min=min(target_min,n.valid_end_utc.min()) if target_min is not None else n.valid_end_utc.min()
                target_max=max(target_max,n.valid_end_utc.max()) if target_max is not None else n.valid_end_utc.max()
        file_stats.append({'path':str(p),'raw_rows':raw_count,'retained_future_finite_rows':valid_count,
                           'excluded_future_timestamp_conflicts':excluded_mismatch,**raw_audit,
                           'issue_min':str(issue_min),'issue_max':str(issue_max),
                           'target_min':str(target_min),'target_max':str(target_max)})
    if not chunks:
        raise ValueError('No forecast archives provided')
    data = pd.concat(chunks, ignore_index=True)
    # Duplicates may exist between the 2026 split archives. Fail on conflicts.
    duplicates = data.duplicated(['issue_utc', 'valid_end_utc'], keep=False)
    if duplicates.any():
        grouped = data.loc[duplicates].groupby(['issue_utc', 'valid_end_utc'])[['wind_mw', 'solar_mw']].nunique(dropna=False)
        conflicts = grouped.max(axis=1).gt(1)
        if conflicts.any():
            raise ValueError(f'{int(conflicts.sum())} duplicate issue/target pairs disagree: do not silently choose one archive')
    n_dups = len(data) - len(data.drop_duplicates(['issue_utc', 'valid_end_utc']))
    data = data.drop_duplicates(['issue_utc', 'valid_end_utc']).sort_values(['issue_utc', 'valid_end_utc']).reset_index(drop=True)
    return data, {'forecast_raw_rows': total_raw, 'forecast_valid_rows': len(data),
                  'archive_duplicate_rows': n_dups, 'settlement_time_mismatches': total_mismatch,
                  'timestamp_conflict_policy':'exclude ambiguous rows without choosing either endpoint' if exclude_settlement_mismatch else ('strict failure' if strict else 'GMT preferred, explicitly unverified'),
                  'distinct_forecast_issues': int(data.issue_utc.nunique()),
                  'distinct_valid_times': int(data.valid_end_utc.nunique()),'file_coverage':file_stats}


def load_demand(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, dict]:
    chunks = []; raw = 0; excluded_forecast = 0
    policies=[]
    for p in map(Path, paths):
        if not p.is_file():
            raise FileNotFoundError(p)
        for chunk in pd.read_csv(p, chunksize=250_000, low_memory=False):
            raw += len(chunk)
            missing = set(DEMAND_COLS[:-1]).difference(chunk.columns)
            if missing:
                raise ValueError(f'Historic demand data missing fields: {sorted(missing)}')
            if 'FORECAST_ACTUAL_INDICATOR' in chunk:
                is_actual = chunk['FORECAST_ACTUAL_INDICATOR'].astype(str).str.upper().eq('A')
                policy='explicit A rows only'
            else:
                metadata=json.loads(p.with_suffix('.metadata.json').read_text())
                assert metadata['resource_id']=='b2bde559-3455-4021-b179-dfe60c0337b0'
                assert metadata['resource_metadata']['name']=='Historic Demand Data 2025'
                assert 'outturn data for 2025' in metadata['resource_metadata']['description']
                is_actual=pd.Series(True,index=chunk.index)
                policy='verified official2025 historic outturn archive has no A/F column; all rows are outturn'
            if not any(item['path']==str(p) for item in policies):policies.append({'path':str(p),'actual_policy':policy})
            excluded_forecast += int((~is_actual).sum())
            df = chunk.loc[is_actual].copy()
            sp = pd.to_numeric(df['SETTLEMENT_PERIOD'], errors='coerce')
            n = pd.DataFrame({'valid_end_utc': _utc_end_from_settlement(df['SETTLEMENT_DATE'], sp),
                              'nd_mw': pd.to_numeric(df['ND'], errors='coerce')})
            chunks.append(n.dropna())
    if not chunks:
        raise ValueError('No demand files')
    demand = pd.concat(chunks, ignore_index=True)
    group = demand.groupby('valid_end_utc')['nd_mw']
    if group.nunique().gt(1).any():
        raise ValueError('Conflicting actual ND values for the same half-hour')
    demand = demand.drop_duplicates('valid_end_utc').sort_values('valid_end_utc').reset_index(drop=True)
    return demand, {'demand_raw_rows': raw, 'demand_actual_rows': len(demand),
                    'excluded_non_actual_rows': excluded_forecast,
                    'source_actual_policies':policies,
                    'demand_min_utc': str(demand.valid_end_utc.min()),
                    'demand_max_utc': str(demand.valid_end_utc.max())}


def _history_covariates(forecasts: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Latest forecast available by the end of each historical settlement period.

    This is a historical covariate construction; it does NOT use forecast issues
    after a historical period is over.
    """
    f = forecasts.loc[forecasts.issue_utc <= forecasts.valid_end_utc].copy()
    f = f.sort_values(['valid_end_utc', 'issue_utc']).drop_duplicates('valid_end_utc', keep='last')
    return demand.merge(f[['valid_end_utc', 'wind_mw', 'solar_mw']], how='left', on='valid_end_utc', validate='one_to_one')


@dataclass
class SampleStats:
    issues_examined: int = 0
    issues_paired: int = 0
    rejected_history: int = 0
    rejected_future: int = 0
    rejected_covariates: int = 0
    rejected_revision: int = 0

    def as_dict(self):
        return vars(self).copy()


def build_pairs(forecasts: pd.DataFrame, demand: pd.DataFrame, *,
                lookback: int = 336, horizon: int = 48,
                revision_hours: float = 6, min_eval_slots: int = 12,
                issue_every_hours: float = 6,
                max_samples: int | None = None) -> tuple[dict, dict]:
    """Pairs A/B issue times for the same future target horizon.

    At A: input is past ND and historical covariates, plus forecast vintage A.
    At B: historical input is HELD FIXED; only future slots whose period-end is
    strictly after B are replaced by vintage-B forecast values. Earlier slots
    remain from vintage A. This isolates covariate revision from target arrival.
    """
    if lookback < 2 or horizon < 2:
        raise ValueError('lookback and horizon must be >=2')
    if not (0 < revision_hours < horizon/2):
        raise ValueError('revision_hours must fit in forecast horizon (half-hour units)')
    if max_samples is not None and max_samples < 1:
        raise ValueError('max_samples must be positive')
    f = forecasts.sort_values(['issue_utc', 'valid_end_utc'])
    starts = f.groupby('issue_utc').indices
    issues = sorted(starts.keys())
    if not issues:
        raise ValueError('No forecast issue timestamps')
    # Pandas Timestamp is guaranteed by pd.to_datetime parser, but numpy
    # dictionary group keys can be int in some pandas versions.
    issues = sorted(pd.to_datetime(x, utc=True) for x in issues)
    issues_arr = pd.DatetimeIndex(issues)
    hist = _history_covariates(forecasts, demand)
    hist = hist.set_index('valid_end_utc').sort_index()
    idx = hist.index
    if not idx.is_unique:
        raise AssertionError('duplicate demand times')
    observed = hist[['nd_mw', 'wind_mw', 'solar_mw']].to_numpy(dtype=np.float32)
    y = hist['nd_mw'].to_numpy(dtype=np.float32)
    issue_groups = {k: f.iloc[v].set_index('valid_end_utc')[['wind_mw', 'solar_mw']]
                    for k, v in f.groupby('issue_utc').indices.items()}
    # Coarsen to at most one A time per 6-hour bin, preserving its real issue time.
    selected = []
    seen_bins = set()
    for issue in issues:
        b = issue.floor(f'{int(issue_every_hours * 60)}min')
        if b not in seen_bins:
            selected.append(issue)
            seen_bins.add(b)
    step = pd.Timedelta(hours=revision_hours)
    records=[]
    stats = SampleStats()
    for ia in selected:
        stats.issues_examined += 1
        # Prefer closest subsequent issue to the requested revision gap.
        match = issues_arr.searchsorted(ia + step)
        if match >= len(issues):
            continue
        ib = issues[match]
        if ib - ia > step + pd.Timedelta(hours=2):
            continue
        stats.issues_paired += 1
        # Last completed settlement period by A: its end is <= A.
        hist_end_index = idx.searchsorted(ia, side='right') - 1
        if hist_end_index < lookback - 1:
            stats.rejected_history += 1
            continue
        h_start = hist_end_index - lookback + 1
        x_hist = observed[h_start:hist_end_index+1]
        # Must be exactly contiguous half-hour slots, including across DST.
        times = idx[h_start: hist_end_index+1]
        if pd.isna(x_hist).any() or not (times.to_series().diff().dropna() == SLOT).all():
            stats.rejected_history += 1
            continue
        fst = hist_end_index + 1
        if fst + horizon > len(idx):
            stats.rejected_future += 1
            continue
        target_times = idx[fst:fst+horizon]
        if not (target_times.to_series().diff().dropna() == SLOT).all():
            stats.rejected_future += 1
            continue
        if target_times[0] <= ia or not np.isfinite(y[fst:fst+horizon]).all():
            stats.rejected_future += 1
            continue
        ca = issue_groups[ia].reindex(target_times)
        cb_part = issue_groups[ib].reindex(target_times)
        # B values only allowed for settlement intervals not finished at B.
        update_mask = (target_times > ib)
        cb = ca.copy()
        cb.loc[update_mask] = cb_part.loc[update_mask].to_numpy()
        if ca.isna().any().any() or cb.isna().any().any():
            stats.rejected_covariates += 1
            continue
        if update_mask.sum() < min_eval_slots or not update_mask.any():
            stats.rejected_revision += 1
            continue
        if not np.isfinite(x_hist).all():
            stats.rejected_history += 1
            continue
        records.append((np.asarray(x_hist, dtype=np.float32), y[fst:fst+horizon,None],
                        ca.to_numpy(dtype=np.float32), cb.to_numpy(dtype=np.float32),
                        update_mask.astype(np.bool_), ia, ib, target_times[0], target_times[-1]))
        if max_samples and len(records) >= max_samples:
            break
    if not records:
        raise ValueError('No valid vintage pairs. Inspect time alignment and coverage; do not fill future covariates from hindsight.')
    xhist, labels, ca, cb, masks, issuea, issueb, targetfirst, targetlast = zip(*records)
    # Unicode strings avoid pickle in npz.
    arrays = {'history':np.stack(xhist), 'target':np.stack(labels),
              'cov_a':np.stack(ca), 'cov_b':np.stack(cb),
              'update_mask':np.stack(masks),
              'issue_a':np.array([str(t) for t in issuea]),
              'issue_b':np.array([str(t) for t in issueb]),
              'first_target_end':np.array([str(t) for t in targetfirst]),
              'last_target_end':np.array([str(t) for t in targetlast])}
    arrays['target_end_utc_ns']=np.array([pd.date_range(t,periods=horizon,freq='30min').asi8 for t in targetfirst],dtype=np.int64)
    arrays['history_end_utc_ns']=arrays['target_end_utc_ns'][:,0,None]+np.arange(-lookback,0,dtype=np.int64)[None,:]*SLOT.value
    return arrays, {'pairing':stats.as_dict(),'samples':len(records),
                    'lookback':lookback,'horizon':horizon,
                    'score_slots_min':int(arrays['update_mask'].sum(axis=1).min()),
                    'score_slots_max':int(arrays['update_mask'].sum(axis=1).max()),
                    'revision_changed_fraction':float(np.mean(np.any(abs(arrays['cov_b']-arrays['cov_a'])>0,axis=(1,2))))}


def split_pairs(arrays: dict, *, val_from='2026-01-01', test_from='2026-04-01', test_until='2026-09-01') -> dict:
    """Split by TARGET end, not just issue time. Holdout starts are disjoint.

    2025 -> train, 2026 Q1 -> val, 2026 Q2-Aug -> test.
    """
    starts = pd.to_datetime(arrays['first_target_end'], utc=True)
    ends = pd.to_datetime(arrays['last_target_end'], utc=True)
    val = pd.Timestamp(val_from, tz='UTC')
    test = pd.Timestamp(test_from, tz='UTC')
    stop = pd.Timestamp(test_until, tz='UTC')
    # No shared target timestamps between any two splits. Windows crossing
    # temporal boundaries are dropped instead of assigned by last label only.
    labels = np.where(ends < val,'train',
                      np.where((starts >= val) & (ends < test),'val',
                               np.where((starts >= test) & (ends < stop),'test','ignore')))
    return {k:{c:v[labels==k] for c,v in arrays.items()} for k in ('train','val','test')}


def run(args):
    f, fm = load_forecasts(args.forecast, strict=not args.allow_settlement_mismatch,exclude_settlement_mismatch=args.exclude_settlement_mismatch)
    d, dm = load_demand(args.demand)
    pairs, pm = build_pairs(f,d,lookback=args.lookback,horizon=args.horizon,
                            revision_hours=args.revision_hours,
                            min_eval_slots=args.min_eval_slots,
                            issue_every_hours=args.issue_every_hours,
                            max_samples=args.max_samples)
    groups = split_pairs(pairs, val_from=args.val_from, test_from=args.test_from, test_until=args.test_until)
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for k,v in groups.items():
        np.savez_compressed(out/f'{k}.npz',**v)
    manifest = {'source_files':{'forecast':list(map(str,args.forecast)), 'demand':list(map(str,args.demand))},
                'forecast_stats':fm, 'demand_stats':dm, 'sample_stats':pm,
                'splits':{k:len(v['history']) for k,v in groups.items()},
                'time_policy':{'timezone':'UTC','valid_time':'period_end_utc',
                               'half_hour_slots':'continuous; UK DST handled by Europe/London',
                               'history_label_publication':'NOT VERIFIED: retrospective ND values are treated as observable at period end',
                               'claim':'covariate-vintage backtest, NOT operational online learning'},
                'protocol':{'hold_history_fixed_at_A':True,'update_only_times_after_B':True,
                            'no_test_data_for_training':True}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--forecast',nargs='+',required=True,help='NESO forecast CSV archive(s) 2025/2026')
    ap.add_argument('--demand',nargs='+',required=True,help='NESO actual ND CSV archive(s) 2025/2026')
    ap.add_argument('--out',default='outputs/dataset')
    ap.add_argument('--lookback',type=int,default=336)
    ap.add_argument('--horizon',type=int,default=48)
    ap.add_argument('--revision-hours',type=float,default=6)
    ap.add_argument('--issue-every-hours',type=float,default=6)
    ap.add_argument('--min-eval-slots',type=int,default=12)
    ap.add_argument('--max-samples',type=int)
    ap.add_argument('--val-from',default='2026-01-01')
    ap.add_argument('--test-from',default='2026-04-01')
    ap.add_argument('--test-until',default='2026-09-01')
    ap.add_argument('--allow-settlement-mismatch',action='store_true')
    ap.add_argument('--exclude-settlement-mismatch',action='store_true',help='Explicitly exclude audited ambiguous source rows; never guess their target time')
    run(ap.parse_args())

if __name__=='__main__': main()
