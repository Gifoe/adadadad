import datetime,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.native_worldts import file_sha
out=ROOT/'outputs/model';lock=out/'validation_decision_lock.json'
assert not lock.exists(),'Validation decision already frozen'
assert not (out/'test_replay.json').exists(),'Test already evaluated'
gate=json.loads((out/'updater_gate.json').read_text());compute=json.loads((ROOT/'outputs/compute_results.json').read_text())
assert gate['allow_updater']==(out/'updater_best.pt').exists(),'Conditional updater decision not fully executed'
files=[ROOT/'configs/worldts_neso.json',ROOT/'configs/execution_protocol.json',out/'train_scaler.json',out/'output_correction_weights.json',
       out/'stage1_best.pt',out/'stage2_best.pt',out/'updater_gate.json',ROOT/'outputs/dataset/manifest.json',ROOT/'outputs/revision_bin_edges.json',
       ROOT/'outputs/val_analysis.json',ROOT/'outputs/compute_results.json',ROOT/'neso_online/native_worldts.py',ROOT/'neso_online/prepare.py',
       ROOT/'scripts/analyze_replay.py',ROOT/'scripts/benchmark_compute.py']
files += [ROOT/'outputs/dataset'/f'{split}.npz' for split in ['train','val','test']]
if gate['allow_updater']:files += [out/'updater_best.pt',out/'updater_history.json']
b1=next(r for r in compute['rows'] if r['method']=='recompute' and r['batch']==1 and r['scope']=='host_output')
decision={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'client_date':'2026-10-14',
          'updater_trained':gate['allow_updater'],'updater_gate_checks':gate['checks'],'cheap_recompute':b1['p95_ms_per_batch']<100,
          'test_selection_role':'none; terminal evaluation after this immutable decision',
          'test_settings':'seed2021; batch16; same mask; fixed TRAIN revision edges; 1/7/14-day bootstrap; MAE_MW primary',
          'sha256':{str(path.resolve()):file_sha(path) for path in files}}
lock.write_text(json.dumps(decision,indent=2),encoding='utf-8');print('VALIDATION_DECISIONS_FROZEN',json.dumps(decision),flush=True)
