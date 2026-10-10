"""Independently check saved forecasts, selected checkpoints and source provenance."""
from common import *
import zipfile

def main():
    required=('README.md','data_audit.md','data_audit.json','ae_reconstruction.csv',
        'baseline_results.csv','refinement_results.csv','horizon_error.csv','per_iteration_gain.csv',
        'compute_budget.csv','covariate_diagnostics.csv','bootstrap_comparisons.csv',
        'FINAL_REPORT.md','decision.json','compute_matching.json','source_manifest.json','environment.json')
    for filename in required:
        assert (ROOT/filename).is_file(),filename
    assert json.loads((ROOT/'sanity.json').read_text())['passed']
    expected={f'{n}_{h}_{v}'+('_K2' if v in ('B3','B4','B5') else '')
        for n in ('NP','PJM') for h in (24,360) for v in ('B0','B1','B2','B3','B4','B5')}
    expected|={f'{n}_360_{v}_K4' for n in ('NP','PJM') for v in ('B3','B4','B5')}
    completed={p.parent.name for p in (ROOT/'runs').glob('*/result.json')}
    assert completed==expected,(completed-expected,expected-completed)
    frame=pd.read_csv(ROOT/'refinement_results.csv');assert len(frame)==30
    checks=[]
    for identifier in sorted(expected):
        out=ROOT/'runs'/identifier;r=json.loads((out/'result.json').read_text())
        assert (out/'checkpoint.pt').stat().st_size>0
        df,y,c,tr,tv,sy,sc=load_data(r['dataset']);h=r['horizon']
        with np.load(out/'predictions.npz') as a:
            for key in a.files:assert np.isfinite(a[key]).all(),(identifier,key)
            origins=np.arange(tv,len(y)-h+1)
            assert np.array_equal(a['origins'],origins)
            pred=a['pred'].astype('float64');gt=np.stack([y[o:o+h,0].numpy() for o in origins]).astype('float64')
            assert pred.shape==gt.shape
            mse=float(np.mean((pred-gt)**2));mae=float(np.mean(abs(pred-gt)))
            assert np.isclose(mse,r['mse'],rtol=2e-6,atol=1e-7)
            assert np.isclose(mae,r['mae'],rtol=2e-6,atol=1e-7)
            v=r['variant'];k=r['K'] if v in ('B3','B4','B5') else 0
            row=frame[(frame.dataset==r['dataset'])&(frame.horizon==h)&(frame.variant==v)&(frame.K==k)]
            assert len(row)==1 and np.isclose(row.iloc[0].mse,mse,rtol=2e-6,atol=1e-7)
            if v in ('B2','B3','B4','B5'):
                assert a['iteration_latent_window_mse'].shape==(len(origins),k+1)
                assert a['iteration_latent_window_cosine'].shape==(len(origins),k+1)
            if v=='B0':assert a['window_mse'].shape==(len(origins),)
            else:assert a['iteration_window_mse'].shape==(len(origins),k+1)
        if v!='B0':
            selected=torch.load(out/'checkpoint.pt',weights_only=True,map_location='cpu')
            frozen=torch.load(ROOT/'ae'/r['dataset']/'checkpoint.pt',weights_only=True,map_location='cpu')
            assert all(torch.equal(selected['ae.'+key],value) for key,value in frozen.items()),identifier
            assert r['best_epoch']==min(r['history_log'],key=lambda e:e['val_mse'])['epoch']
        checks.append({'run':identifier,'test_windows':len(origins),'independent_mse':mse,'independent_mae':mae,'passed':True})
    provenance=json.loads((ROOT/'source_manifest.json').read_text())
    for rel,sha in provenance['experiment_file_hashes'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
    with zipfile.ZipFile(ROOT/'upstream.zip') as z:
        for repo,files in provenance['upstream_file_hashes'].items():
            for rel,sha in files.items():
                assert hashlib.sha256(z.read('upstream/'+repo+'/'+rel)).hexdigest()==sha,(repo,rel)
                assert hashlib.sha256((ROOT/'upstream'/repo/rel).read_bytes()).hexdigest()==sha,(repo,rel)
    assert len(pd.read_csv(ROOT/'bootstrap_comparisons.csv'))==44
    assert len(json.loads((ROOT/'compute_matching.json').read_text()))==6
    assert len(list((ROOT/'failure_cases').glob('*.png')))>=2
    json_write(ROOT/'DELIVERABLE_VERIFICATION.json',{'passed':True,'formal_runs':30,'AE_runs':2,
        'checks':['independent float64 MSE/MAE from every saved forecast','all exact rolling test origins',
            'frozen AE weights unchanged in all selected checkpoints','new-model checkpoints selected by validation MSE',
            'source hashes match pristine upstream ZIP and executed sources','required tables and diagnostics complete'],
        'runs':checks})
    print('DELIVERABLES_VERIFIED',len(checks),flush=True)

if __name__=='__main__':main()
