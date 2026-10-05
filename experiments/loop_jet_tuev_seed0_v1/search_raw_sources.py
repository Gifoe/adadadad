"""Bounded project/data-disk search, deeper only under TUH/TUEV/event paths."""
import json
import os
import pathlib
import re
from process_affinity import configure_process_affinity
configure_process_affinity()

ROOT = pathlib.Path(__file__).resolve().parent
SKIP = {'windows','program files','program files (x86)','anaconda','programdata','$recycle.bin',
        'system volume information','node_modules','.git','.venv','venv','__pycache__','provenance_git'}
MATCH = re.compile(r'tuev|tuh|(?:^|[\\/])events?(?:[\\/]|$)', re.I)
result = {'policy': 'D/E/F directories to depth 4; to depth 12 only inside TUH/TUEV/event paths. Skip system/software/cache trees. No file contents read.',
          'directories_visited': 0, 'candidate_directories': [], 'matched_raw_files': [],
          'matched_processed_files': [], 'all_seen_edf_rec_files_count': 0, 'errors': []}
def visit(folder, depth):
    allowed_depth = 12 if MATCH.search(folder) else 4
    if depth > allowed_depth:
        return
    try:
        entries = list(os.scandir(folder))
    except OSError as exc:
        result['errors'].append({'path': folder, 'type': type(exc).__name__})
        return
    result['directories_visited'] += 1
    for e in entries:
        if e.is_dir(follow_symlinks=False):
            if e.name.lower() in SKIP:
                continue
            if MATCH.search(e.name):
                result['candidate_directories'].append(e.path)
            visit(e.path, depth+1)
        elif e.is_file(follow_symlinks=False):
            ext = pathlib.Path(e.name).suffix.lower()
            if ext in ('.edf', '.rec'):
                result['all_seen_edf_rec_files_count'] += 1
                if MATCH.search(e.path):
                    result['matched_raw_files'].append(e.path)
            elif ext in ('.h5', '.hdf5', '.pkl') and MATCH.search(e.path):
                result['matched_processed_files'].append(e.path)
for drive in ('D:/', 'E:/', 'F:/'):
    if os.path.isdir(drive):
        visit(drive, 0)
(ROOT/'data_audit'/'raw_source_file_search.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
