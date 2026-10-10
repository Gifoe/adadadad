"""Package reproducible source/summary evidence; keep full checkpoints and forecasts on execution host."""
import pathlib,json,hashlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    state=json.loads((ROOT/'STATUS.json').read_text())
    assert state['state']=='complete'
    rows=[];chosen=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(x in rel.parts for x in ('.venv','__pycache__','upstream')):continue
        if p.name in ('deliverables.zip','ARTIFACT_MANIFEST.json'):continue
        if rel.parts[0]=='data' and p.suffix!='.json':continue
        included=p.suffix not in ('.pt','.npz')
        rows.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':digest(p),'included_in_summary_package':included})
        if included:chosen.append(p)
    (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'full_model_checkpoints_and_predictions_retained_on_server':True,'files':rows},indent=2),encoding='utf-8')
    chosen.append(ROOT/'ARTIFACT_MANIFEST.json')
    with zipfile.ZipFile(ROOT/'deliverables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in chosen:z.write(p,p.relative_to(ROOT).as_posix())
    print('PACKAGED',len(chosen),(ROOT/'deliverables.zip').stat().st_size,digest(ROOT/'deliverables.zip'),flush=True)

if __name__=='__main__':main()
