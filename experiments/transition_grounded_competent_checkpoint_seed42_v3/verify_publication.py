"""Independent recorded-output consistency and publication integrity audit."""
import pathlib,json,csv,hashlib,zipfile,io
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'outputs'
def read(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8')))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('*.json'))+list(OUT.glob('*.csv'))+list(OUT.glob('*.json')):p.read_text(encoding='utf-8')
published_checkpoints=0
for r in json.loads((OUT/'checkpoint_manifest.json').read_text()):
 if r['step']==0:continue
 p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];published_checkpoints+=1
assert published_checkpoints==8
for r in json.loads((OUT/'archive_manifest.json').read_text()):
 p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 with zipfile.ZipFile(p) as z:assert z.testzip() is None
with zipfile.ZipFile(ROOT/'archives/verified_selected_official_dataset.zip') as z:
 for n in ['train','test']:
  actual=np.load(io.BytesIO(z.read(f'data/{n}.npz')));canonical=np.load(ROOT/f'data/{n}.npz');assert all(np.array_equal(actual[k],canonical[k]) for k in canonical.files)
 for n in ['train_records.json','test_records.json','batch_plan.npy','vocab.json','transition_table.json','integrity_check.json','batch_plan_manifest.json']:assert z.read('data/'+n)==(ROOT/'data'/n).read_bytes()
hashes=json.loads((OUT/'source_and_data_hashes.json').read_text())
for rel,h in hashes.items():assert sha(ROOT/rel)==h,rel
vocab=json.loads((ROOT/'data/vocab.json').read_text());ids={e:i for i,e in enumerate(vocab)};table=json.loads((ROOT/'data/transition_table.json').read_text());records=json.loads((ROOT/'data/test_records.json').read_text());plan=json.loads((OUT/'intervention_plan.json').read_text())
for r in plan:
 rec=records[r['index']];assert rec['sample_id']==r['sample_id'];e=vocab[r['e_cf']];path=[]
 for rel in rec['relations'][r['k']:]:e=table[e+':'+rel];path.append(ids[e])
 assert path==r['cf_path'] and path[-1]==r['y_cf'] and r['y_cf']!=r['y_original'];assert r['e_original']==ids[rec['state_path'][r['k']-1]]
with zipfile.ZipFile(ROOT/'archives/recorded_carriers_prototypes_and_controls.zip') as z:
 directions=np.load(io.BytesIO(z.read('outputs/intervention_noise_directions.npy')));assert directions.shape==(len(plan),768) and np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-6)
 assert hashlib.sha256(z.read('outputs/intervention_noise_directions.npy')).hexdigest()==json.loads((OUT/'intervention_plan_manifest.json').read_text())['noise_direction_sha256']
targets=np.load(ROOT/'data/test.npz');intervention_keys=[];counts={}
for name in ['pretrained','final_only','grounded']:
 matrix=read('pretrained_accuracy_matrix.csv' if name=='pretrained' else f'accuracy_matrix_{name}.csv');state=read(f'state_accuracy_{name}.csv')
 for d in sorted(set(targets['hops'])):
  roll=np.load(OUT/f'rollout_{name}_D{d}.npz');sel=targets['hops']==d;assert np.array_equal(roll['targets'],targets['state_targets'][sel]) and np.array_equal(roll['sample_ids'],targets['sample_ids'][sel]);pred=roll['predictions'];gt=roll['targets'];assert len(gt)==750
  for r in [r for r in matrix if int(r['D'])==d]:assert int(r['correct'])==int((pred[:,int(r['K'])-1]==gt[:,-1]).sum())
  for r in [r for r in state if int(r['D'])==d]:assert int(r['correct'])==int((pred[:,int(r['k'])-1]==gt[:,int(r['k'])-1]).sum())
 causal=read(f'intervention_{name}.csv');assert len(causal)==19200;intervention_keys.append([(r['sample_id'],r['D'],r['k'],r['K'],r['control'],r['e_cf'],r['y_cf']) for r in causal])
 for r in causal:
  k,d,K=map(int,[r['k'],r['D'],r['K']]);p=json.loads(r['observed_path']);cf=json.loads(r['expected_cf_path']);assert len(p)==K and len(cf)==d-k;assert int(r['cf_target'])==int(p[-1]==int(r['y_cf']));assert int(r['cf_trajectory_correct'])==sum(a==b for a,b in zip(p[k:d],cf))
 counts[name]=sum(int(r['cf_target']) for r in causal if r['control']=='counterfactual' and r['regime']=='main')
assert intervention_keys[0]==intervention_keys[1]==intervention_keys[2]
for n in ['baseline','grounded']:
 rows=read(f'finetune_history_{n}.csv');assert [int(r['step']) for r in rows]==list(range(1,5001));assert all(np.isfinite(float(r['loss'])) and float(r['lr'])==1e-5 for r in rows)
result=dict(status='PASS',published_checkpoints=8,verified_archives=3,canonical_dataset_arrays_equal=True,canonical_source_and_data_hashes_equal=True,counterfactual_paths_replayed=len(plan),paired_intervention_records_per_model=19200,CF_main_target_counts=counts,CF_main_samples_per_model=3200,recorded_rollout_accuracies_recomputed=True,formal_history_rows_per_arm=5000,UTF8_valid=True)
(OUT/'publication_integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
