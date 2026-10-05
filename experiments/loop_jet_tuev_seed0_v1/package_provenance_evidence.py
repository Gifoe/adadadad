"""Small public metadata evidence archive; excludes Git repos, signals and caches."""
import hashlib
import json
import pathlib
import zipfile
from process_affinity import configure_process_affinity
configure_process_affinity()

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'data_audit'
path = OUT/'provenance_evidence.zip'
files = []
for folder in ('processed_provenance',):
    files.extend(p for p in (OUT/folder).rglob('*') if p.is_file())
for name in ('raw_tuev_recordings.json', 'raw_tuev_channels.json', 'provenance_search_summary.json'):
    files.append(OUT/name)
inventory = []
with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for p in files:
        relative = p.relative_to(OUT).as_posix()
        data = p.read_bytes()
        archive.writestr(relative, data)
        inventory.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
(OUT/'provenance_evidence_manifest.json').write_text(json.dumps({'archive_bytes': path.stat().st_size,
    'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'files': inventory}, indent=2), encoding='utf-8')
print('EVIDENCE_ARCHIVE', path.stat().st_size, flush=True)
