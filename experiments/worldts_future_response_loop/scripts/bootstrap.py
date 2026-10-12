from pathlib import Path
import zipfile,hashlib,json
root=Path(__file__).resolve().parents[1]
for d in ['configs','reports','results','figures','logs','runs','patches']:(root/d).mkdir(exist_ok=True)
with zipfile.ZipFile(root/'official_worldts_source.zip') as z:
    for entry in z.infolist():
        p=(root/entry.filename).resolve();assert p.is_relative_to(root.resolve());z.extract(entry,root)
for r in json.loads((root/'configs/source_manifest.json').read_text()):assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256']
print('PINNED_SOURCE_INTEGRITY_VERIFIED')
