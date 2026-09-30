import hashlib,json,pathlib,zipfile
root=pathlib.Path(__file__).parent; target=root/'data/generated'; target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(root/'dataset.zip') as z:
    allowed={'manifest.json','train.jsonl','validation.jsonl','iid.jsonl','ood_6.jsonl','ood_8.jsonl','ood_12.jsonl'}
    assert set(z.namelist())==allowed
    for name in allowed:
        (target/name).write_bytes(z.read(name))
manifest=json.loads((target/'manifest.json').read_text())
assert 'elapsed_sec' in manifest and len(manifest['splits'])==6
for split,meta in manifest['splits'].items():
    assert hashlib.sha256((target/(split+'.jsonl')).read_bytes()).hexdigest()==meta['sha256']
print('All six datasets extracted with verified checksums',flush=True)
