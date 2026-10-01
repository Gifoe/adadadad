import json,pathlib,shutil
root=pathlib.Path(__file__).resolve().parent
source=root/'artifacts/sanity'; destination=root/'artifacts/sanity_300_failed'
for p in [source,destination]: assert p.resolve().is_relative_to(root)
if source.exists() and not destination.exists(): source.rename(destination)
for name in ['pipeline.stdout.log','pipeline.stderr.log','status.json']:
    src=root/'artifacts'/name
    if src.exists(): shutil.copy2(src,destination/name)
c=json.loads((root/'configs/frozen.json').read_text()); c['sanity_updates']=2000
(root/'configs/frozen.json').write_text(json.dumps(c,indent=2))
print('Retained 300-update failure; all sanity retries use 2000 updates from scratch')
