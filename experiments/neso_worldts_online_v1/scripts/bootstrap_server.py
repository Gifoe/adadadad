import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
archive=ROOT/'original_package.zip'
with zipfile.ZipFile(archive) as z:
    original=[]
    for name in z.namelist():
        if name.endswith('/'):continue
        rel=pathlib.PurePosixPath(name)
        assert rel.parts[0]=='neso_worldts_online_v1' and '..' not in rel.parts
        target=ROOT.joinpath(*rel.parts[1:])
        assert target.resolve().is_relative_to(ROOT.resolve())
        payload=z.read(name)
        original.append({'path':target.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(payload).hexdigest()})
        assert not target.exists(),str(target)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
for directory in ['outputs/logs','outputs/source_metadata','data/raw','upstream']:
    (ROOT/directory).mkdir(parents=True,exist_ok=True)
(ROOT/'outputs/provided_package_manifest.json').write_text(json.dumps({'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':original},indent=2),encoding='utf-8')
print('ORIGINAL_PACKAGE_EXTRACTED',len(original),flush=True)
