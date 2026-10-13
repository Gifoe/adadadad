"""Matched, dependence-aware analysis. Validation selects; Test only reports."""
import argparse,csv,json,pathlib,sys
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def block_difference(total_diff,counts,dates,block_days,reps=2000):
    # Resample calendar-day aggregates in circular contiguous blocks. Windows
    # and repeated target slots stay together inside their original issue day.
    day=pd.to_datetime(dates,utc=True).floor('D')
    frame=pd.DataFrame({'diff':total_diff,'count':counts},index=day).groupby(level=0).sum()
    frame=frame.reindex(pd.date_range(day.min(),day.max(),freq='D'),fill_value=0)
    diff=frame['diff'].to_numpy();cnt=frame['count'].to_numpy();n=len(frame)
    rng=np.random.default_rng(2021);draws=[]
    blocks=int(np.ceil(n/block_days));offset=np.arange(block_days)
    for _ in range(reps):
        indices=((rng.integers(0,n,size=blocks)[:,None]+offset)%n).ravel()[:n]
        draws.append(diff[indices].sum()/max(cnt[indices].sum(),1))
    disjoint=[]
    for i in range(0,n,block_days):
        if cnt[i:i+block_days].sum():disjoint.append(diff[i:i+block_days].sum()/cnt[i:i+block_days].sum())
    return {'block_days':block_days,'n_calendar_days':n,'point_delta_mae_mw':float(diff.sum()/cnt.sum()),
            'ci95_low_mw':float(np.quantile(draws,.025)),'ci95_high_mw':float(np.quantile(draws,.975)),
            'n_disjoint_blocks':len(disjoint),'fraction_disjoint_blocks_favor_candidate':float(np.mean(np.array(disjoint)<0)),
            'disjoint_block_delta_mae_mw':disjoint,'bootstrap_reps':reps,
            'interpretation':'descriptive dependent-block interval; one training seed, no causal or multi-seed significance claim'}

def main():
    p=argparse.ArgumentParser();p.add_argument('split',choices=['val','test']);p.add_argument('--updater',action='store_true');a=p.parse_args()
    out=ROOT/'outputs/model';name='test_predictions.npz' if a.split=='test' else ('validation_updater_predictions.npz' if a.updater else 'validation_predictions.npz')
    with np.load(out/name) as z:data={k:z[k] for k in z.files}
    with np.load(ROOT/'outputs/dataset'/f'{a.split}.npz') as z:ca=z['cov_a'];cb=z['cov_b']
    scaler=json.loads((out/'train_scaler.json').read_text());scale=scaler['std'][0]
    mask=data['update_mask'];truth=data['target_mw'][:,:,0].astype('float64');count=mask.sum(1)
    assert np.array_equal(mask,data['target_end_utc_ns']>pd.to_datetime(data['issue_b'],utc=True).asi8[:,None])
    predictions={key[5:-3]:data[key][:,:,0].astype('float64') for key in data if key.startswith('pred_') and key.endswith('_mw')}
    errors={method:np.abs(pred-truth) for method,pred in predictions.items()}
    old=errors['old'];window_old=(old*mask).sum(1)/count;rows=[];revisions=[];windows=[]
    delta=cb.astype('float64')-ca.astype('float64');rms=np.sqrt(np.sum(delta**2,axis=(1,2))/(count*2))
    edges=json.loads((ROOT/'outputs/revision_bin_edges.json').read_text())['rms_covariate_revision_mw_edges'];bins=np.searchsorted(edges,rms,side='right')
    for method,pred in predictions.items():
        difference=pred-truth;err=errors[method];mae=float(err[mask].mean());mse=float((difference[mask]**2).mean())
        per=(err*mask).sum(1)/count;change=np.abs(pred-predictions['old']);gain=(float(old[mask].mean())-mae)/float(old[mask].mean())
        rows.append({'split':a.split,'method':method,'n_windows':len(mask),'n_scored_slots':int(mask.sum()),
                     'mae_mw':mae,'mse_mw2':mse,'normalized_mae':mae/scale,'normalized_mse':mse/scale**2,
                     'relative_mae_gain_vs_old':gain,'fraction_windows_improved':float(np.mean(per<window_old-1e-6)),
                     'fraction_windows_worsened':float(np.mean(per>window_old+1e-6)),
                     'fraction_windows_tied':float(np.mean(abs(per-window_old)<=1e-6)),
                     'fraction_windows_prediction_changed_gt_0_001_mw':float(np.mean(np.any((change>.001)&mask,axis=1))),
                     'mean_abs_prediction_revision_mw':float(change[mask].mean())})
        for i in range(len(mask)):
            windows.append({'split':a.split,'window':i,'issue_a':data['issue_a'][i],'issue_b':data['issue_b'][i],
                            'first_target_utc':data['first_target_end'][i],'method':method,'revision_rms_mw':rms[i],
                            'n_scored_slots':int(count[i]),'mae_mw':per[i],'delta_mae_vs_old_mw':per[i]-window_old[i]})
        for lead in range(mask.shape[1]):
            select=mask[:,lead]
            if select.any():revisions.append({'split':a.split,'method':method,'stratum':'lead_from_A_half_hour','bin':lead+1,'n_scored_slots':int(select.sum()),
                'mae_mw':float(err[select,lead].mean()),'mse_mw2':float((difference[select,lead]**2).mean()),
                'delta_mae_vs_old_mw':float((err-old)[select,lead].mean())})
        for category in range(4):
            select=(bins==category)[:,None]&mask
            if select.any():revisions.append({'split':a.split,'method':method,'stratum':'train_revision_rms_quartile','bin':category+1,'n_scored_slots':int(select.sum()),
                'mae_mw':float(err[select].mean()),'mse_mw2':float((difference[select]**2).mean()),
                'delta_mae_vs_old_mw':float((err-old)[select].mean())})
    comparisons={}
    pairs=[('recompute','old'),('output_correction','old'),('output_correction','recompute'),('seasonal_naive','old')]
    if 'updater' in predictions:pairs += [('updater','old'),('updater','recompute'),('updater','output_correction')]
    bootstrap=[]
    for candidate,reference in pairs:
        diff=((errors[candidate]-errors[reference])*mask).sum(1);key=candidate+'_vs_'+reference
        comparisons[key]={}
        for length in [1,7,14]:
            result=block_difference(diff,count,data['first_target_end'],length)
            comparisons[key][str(length)]=result
            bootstrap.append({'split':a.split,'comparison':key,**{k:v for k,v in result.items() if not isinstance(v,list)}})
    # Independently recompute saved driver metrics, never infer units from names.
    metrics_name='test_replay.json' if a.split=='test' else ('train_updater_validation.json' if a.updater else 'train_validation.json')
    driver=json.loads((out/metrics_name).read_text())
    for row in rows:
        method=row['method'];assert abs(row['mae_mw']-driver['methods'][method]['mae'])<1e-8
        assert abs(row['mse_mw2']-driver['methods'][method]['mse'])<1e-6
    prefix='VALIDATION' if a.split=='val' else 'TEST'
    csv_write(ROOT/f'{prefix}_RESULTS.csv',rows);csv_write(ROOT/'outputs'/f'{a.split}_per_window.csv',windows)
    csv_write(ROOT/'outputs'/f'{a.split}_bootstrap.csv',bootstrap)
    other=[];revision_path=ROOT/'REVISION_ANALYSIS.csv'
    if revision_path.exists():
        with open(revision_path,encoding='utf-8') as f:other=[row for row in csv.DictReader(f) if row['split']!=a.split]
    csv_write(revision_path,other+revisions)
    cases=[]
    for method in predictions:
        if method=='old':continue
        diff=((errors[method]-old)*mask).sum(1)/count
        for kind,indices in [('largest_improvements',np.argsort(diff)[:5]),('largest_degradations',np.argsort(diff)[-5:][::-1])]:
            for i in indices:cases.append({'method':method,'kind':kind,'window':int(i),'issue_a':data['issue_a'][i],'issue_b':data['issue_b'][i],
                'delta_mae_vs_old_mw':float(diff[i]),'old_mae_mw':float(window_old[i]),'revision_rms_mw':float(rms[i])})
    result={'split':a.split,'metrics':rows,'block_comparisons':comparisons,'cases':cases,'train_only_revision_edges_mw':edges,
            'selection_role':'validation selection' if a.split=='val' else 'locked final reporting only'}
    (ROOT/'outputs'/f'{a.split}_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if a.split=='val' and not a.updater:
        b1=next(r for r in rows if r['method']=='recompute');b3=next(r for r in rows if r['method']=='output_correction')
        primary=comparisons['recompute_vs_old']['7']
        checks={'gain_ge_1pct':b1['relative_mae_gain_vs_old']>=.01,'seven_day_interval_favors_recompute':primary['ci95_high_mw']<0,
                'two_thirds_blocks_favor_recompute':primary['fraction_disjoint_blocks_favor_candidate']>=2/3,
                'simple_correction_worse_than_recompute_gt_1pct':b3['mae_mw']>b1['mae_mw']*1.01}
        gate={'allow_updater':all(checks.values()),'checks':checks,'rule_file':'configs/execution_protocol.json',
              'decision_data':'Validation only','primary_difference':primary,'validation_metrics':rows}
        (out/'updater_gate.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
        print('UPDATER_GATE',json.dumps(gate),flush=True)
    print('MATCHED_METRICS_CHECK_PASS',json.dumps(rows),flush=True)
if __name__=='__main__':main()
