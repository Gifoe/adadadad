"""Bounded local directory-name search; never reads credentials or unrelated data."""
import ctypes
import json
import os
import pathlib
import sys

from process_affinity import configure_process_affinity
configure_process_affinity()
roots = ['D:/', 'E:/', 'F:/']
skip = {'windows', 'program files', 'program files (x86)', 'anaconda', 'programdata', '$recycle.bin',
        'system volume information', 'node_modules', '.git', '.venv', 'venv', '__pycache__'}
result = {'search': 'Directory names through depth 4 on D/E/F; no claim of exhaustive raw-data absence.',
          'roots': roots, 'hits': [], 'directories_visited': 0, 'permission_errors': []}
def visit(path, depth):
    if depth > 4:
        return
    try:
        entries = list(os.scandir(path))
    except OSError as exc:
        result['permission_errors'].append({'path': path, 'type': type(exc).__name__})
        return
    result['directories_visited'] += 1
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False) or entry.name.lower() in skip:
            continue
        if any(k in entry.name.lower() for k in ('tuev', 'tuh_eeg_events')):
            result['hits'].append(entry.path)
        visit(entry.path, depth+1)
for root in roots:
    if os.path.isdir(root):
        visit(root, 0)
out = pathlib.Path(__file__).resolve().parent/'data_audit'/'local_source_search.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
