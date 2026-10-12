import pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 status=json.loads((ROOT/'STATUS.json').read_text());assert status['overall_complete'] and status['formal_fits']==6
 pf=json.loads((ROOT/'outputs/preflight_metrics.json').read_text());assert pf['passed'];assert json.loads((ROOT/'outputs/tests_report.json').read_text())['passed'];protected=pf['input_artifacts']
 for r in protected:assert sha(pathlib.Path(r['path']))==r['sha256'],'Protected baseline input changed: '+r['path']
 source=json.loads((ROOT/'configs/source_manifest.json').read_text())
 for r in source:assert sha(ROOT/r['path'])==r['sha256'],'Pinned official source changed: '+r['path']
 required=['README.md','REPRODUCE.md','outputs/PREFLIGHT_REPORT.md','outputs/ALIGNMENT_AUDIT.md','outputs/TRAINING_COMPARISON.md','outputs/FINAL_RESEARCH_REPORT.md','outputs/NEXT_RESEARCH_DECISION.md','outputs/preflight_metrics.json','outputs/gradient_alignment.csv','outputs/error_alignment.csv','outputs/training_results.csv','outputs/training_curves.csv']
 assert all((ROOT/p).is_file() for p in required)
 rows=[]
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if not p.is_file() or any(x in rel.parts for x in ['upstream','__pycache__']) or p.name=='ARTIFACT_MANIFEST.json':continue
  rows.append({'path':rel.as_posix(),'absolute_server_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'delivery':'server_only' if p.suffix.lower() in ['.pt','.npz','.npy'] else 'git_and_local'})
 value={'sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'finalized_after_orchestrator_exit':True,'experiment_root':str(ROOT),'source_commit':'269426ea6217d7152a244da0e209d922acd46d4f','repository_parent':'1ddba1c5b81f0045b5c9a884820ef44578e926f9','formal_fits':6,'protected_baseline_inputs_verified_unchanged':protected,'official_source_files_verified':len(source),'files':rows,'self_hash_excluded':True}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps({'sealed_files':len(rows),'server_only':sum(r['delivery']=='server_only' for r in rows),'protected_inputs':len(protected),'pinned_source_files':len(source)}))
if __name__=='__main__':main()
