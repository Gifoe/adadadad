"""Compact byte-preserving export; large logits/weights and private logs stay local."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    from cpu_affinity import apply_task_affinity
    apply_task_affinity()
    chosen=[]
    for folder in ['scripts','tests','outputs']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(ROOT)
            if any(part in ['__pycache__','.pytest_cache','logs','anchor_logits'] for part in rel.parts):continue
            if p.suffix in ['.npz','.npy','.pt','.download','.tmp']:continue
            if p.name=='preflight_manifest_before_identifier_anchor_fix.jsonl':continue
            chosen.append(p)
    chosen += [p for p in ROOT.iterdir() if p.is_file() and (p.suffix in ['.md','.yaml','.txt'] or p.name in ['.gitignore','.gitattributes'])]
    target=ROOT/'artifact_bundle.zip';files=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(chosen):
            rel=p.relative_to(ROOT).as_posix();data=p.read_bytes();z.writestr(rel,data)
            files.append({'path':rel,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    report={'bundle':'artifact_bundle.zip','size_bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'n_files':len(files),'files':files,'excluded':'weights, upstream repos, full-vocabulary NPZ, private logs and preflight obsolete manifest'}
    (ROOT/'artifact_bundle.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
if __name__=='__main__':main()
