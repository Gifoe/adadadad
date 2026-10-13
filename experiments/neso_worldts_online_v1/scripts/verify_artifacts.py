"""Read-only metric/content verification after the single locked Test replay."""
import datetime,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.native_worldts import file_sha
def main():
    manifest=json.loads((ROOT/'outputs/artifact_manifest.json').read_text())
    for item in manifest['files']:
        path=ROOT/item['relative_path'];assert path.stat().st_size==item['size_bytes'];assert file_sha(path)==item['sha256'],str(path)
    lock=json.loads((ROOT/'outputs/model/validation_decision_lock.json').read_text())
    for path,sha in lock['sha256'].items():assert file_sha(path)==sha,path
    assert (ROOT/'outputs/model/test_replay.json').stat().st_mtime>=datetime.datetime.fromisoformat(lock['frozen_utc']).timestamp()
    assert not (ROOT/'outputs/model/updater_best.pt').exists()
    assert not json.loads((ROOT/'outputs/model/updater_gate.json').read_text())['allow_updater']
    assert '13 passed' in (ROOT/'outputs/logs/tests_final.log').read_text()
    checks={}
    for split,prediction,metrics in [('val','validation_predictions.npz','train_validation.json'),('test','test_predictions.npz','test_replay.json')]:
        with np.load(ROOT/'outputs/model'/prediction) as z:data={k:z[k] for k in z.files}
        stored=json.loads((ROOT/'outputs/model'/metrics).read_text());mask=data['update_mask'];truth=data['target_mw'].astype('float64')
        assert int(mask.sum())==stored['n_scored_slots']
        for name,item in stored['methods'].items():
            residual=data['pred_'+name+'_mw'].astype('float64')-truth;selected=residual[mask]
            assert abs(float(np.mean(abs(selected)))-item['mae'])<1e-8
            assert abs(float(np.mean(selected**2))-item['mse'])<1e-6
        checks[split]={'windows':len(mask),'scored_slots':int(mask.sum()),'independent_MW_MW2_metrics_match':True}
    assert not subprocess.check_output(['git','-C',str(ROOT/'upstream/WorldTS'),'status','--porcelain'],text=True).strip()
    result={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_files_verified':len(manifest['files']),
            'all_manifested_bytes_and_sizes_match':True,'validation_lock_files_unchanged':True,'official_source_clean':True,
            'final_test_after_validation_freeze':True,'conditional_updater_correctly_skipped':True,'engineering_tests_passed':13,
            'independent_metric_checks':checks,'requested_full_april_august_coverage':False,
            'decision':'NO_GO_PERSISTENT_LATENT_UPDATER','note':'This verification report is generated after the content manifest and is not self-hashed inside that manifest.'}
    (ROOT/'outputs/artifacts_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FINAL_CONTENT_AND_METRICS_VERIFIED',json.dumps(result),flush=True)
if __name__=='__main__':main()
