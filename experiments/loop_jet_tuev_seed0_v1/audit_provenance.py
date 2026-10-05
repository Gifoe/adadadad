"""Full Git-history text search, plus pinned raw-manifest metadata download."""
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from process_affinity import configure_process_affinity
configure_process_affinity()

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'data_audit'
REPOS = {
    'original_32': ('USuCgex0122e/demo_dataset', 'db96e77affa66398c553427045ee040f06230c2a'),
    'alternate_22': ('USuCgex0122e/demo_dataset_seed20260726', '41602cf9363d83204e6b7145b04efaeafebd9ae4'),
}
EXTENSIONS = {'.py', '.json', '.yaml', '.yml', '.csv', '.txt', '.md'}
KEYWORDS = re.compile(r'channel|ch_names|electrode|montage|bipolar|referen[ct]|\b(?:FP[12]|F[3478]|T[3456]|C[34]|P[34]|O[12])\b|TUEV|TUH|EDF|\bREC\b', re.I)

def git(repo, *args):
    return subprocess.check_output(['git', '--git-dir='+str(repo), *args])

def inspect(label, name, pinned):
    bare = OUT/'provenance_git'/f'{label}.git'
    bare.parent.mkdir(parents=True, exist_ok=True)
    attempts = []
    env = os.environ.copy()
    env['GIT_LFS_SKIP_SMUDGE'] = '1'
    if not bare.exists():
        for endpoint in ('https://hf-mirror.com', 'https://huggingface.co'):
            command = ['git', 'clone', '--bare', endpoint+'/datasets/'+name, str(bare)]
            proc = subprocess.run(command, env=env, capture_output=True, timeout=120)
            attempts.append({'endpoint': endpoint, 'returncode': proc.returncode,
                             'stdout': proc.stdout.decode(errors='replace'), 'stderr': proc.stderr.decode(errors='replace')})
            if proc.returncode == 0:
                break
            # Git leaves an empty destination on some failures. Reuse it by init/fetch,
            # never remove directories outside the experiment.
            if bare.exists():
                raise RuntimeError('Failed partial clone retained for inspection: '+str(bare))
        else:
            raise RuntimeError('Both Git endpoints failed for '+name)
    head = git(bare, 'rev-parse', 'HEAD').decode().strip()
    assert head == pinned, f'Unexpected/missing mirrored revision: {head} != {pinned}'
    commits = git(bare, 'rev-list', '--all').decode().splitlines()
    inventory, blobs = [], {}
    for commit in commits:
        for record in git(bare, 'ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
            if not record:
                continue
            info, path = record.split(b'\t', 1)
            mode, kind, oid = info.decode().split()
            path = path.decode('utf-8')
            if kind != 'blob':
                continue
            inventory.append({'commit': commit, 'path': path, 'blob': oid})
            if pathlib.PurePosixPath(path).suffix.lower() in EXTENSIONS or path == '.gitattributes':
                blobs.setdefault(oid, set()).add(path)
    base = OUT/'processed_provenance'/label
    base.mkdir(parents=True, exist_ok=True)
    matches, saved = [], []
    for oid, paths in blobs.items():
        content = git(bare, 'cat-file', 'blob', oid)
        filename = base/'historical_blobs'/f'{oid}.txt'
        filename.parent.mkdir(exist_ok=True)
        filename.write_bytes(content)
        saved.append({'git_blob': oid, 'sha256': hashlib.sha256(content).hexdigest(),
                      'bytes': len(content), 'historical_paths': sorted(paths)})
        for number, line in enumerate(content.decode('utf-8', errors='replace').splitlines(), 1):
            if KEYWORDS.search(line):
                matches.append({'blob': oid, 'paths': sorted(paths), 'line': number, 'text': line})
    current = []
    for row in inventory:
        if row['commit'] != head:
            continue
        path = row['path']
        if pathlib.PurePosixPath(path).suffix.lower() in EXTENSIONS or path == '.gitattributes':
            destination = base/'current'/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(git(bare, 'cat-file', 'blob', row['blob']))
            current.append(path)
    for filename, data in [('all_commit_file_inventory.json', inventory), ('blob_inventory.json', saved), ('keyword_hits.json', matches)]:
        (base/filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    history = git(bare, 'log', '--all', '--date=iso-strict', '--format=%H%x09%aI%x09%s').decode('utf-8')
    (base/'commit_history.tsv').write_text('commit\tdate\ttitle\n'+history, encoding='utf-8')
    summary = {'repo': name, 'pinned_revision': head, 'commits_examined': len(commits),
               'unique_text_blobs_examined': len(blobs), 'current_text_files': current,
               'all_historical_python_paths': sorted({row['path'] for row in inventory if row['path'].endswith('.py')}),
               'historical_keyword_hit_count': len(matches), 'clone_attempts': attempts,
               'git_archive_is_metadata_and_lfs_pointers_only': True}
    print(label, json.dumps(summary, ensure_ascii=False), flush=True)
    return summary

def main():
    OUT.mkdir(exist_ok=True)
    summaries = {label: inspect(label, *value) for label, value in REPOS.items()}
    (OUT/'provenance_search_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    mapping_history, pointer_history = {}, {}
    for label, output_key in [('original_32', 'hf-original.git'), ('alternate_22', 'hf-alternate.git')]:
        bare = OUT/'provenance_git'/f'{label}.git'
        inventory = json.loads((OUT/'processed_provenance'/label/'all_commit_file_inventory.json').read_text())
        rows, pointers, seen = [], {}, set()
        for row in inventory:
            if not row['path'].startswith('TUEV/'):
                continue
            if row['path'].endswith('.h5'):
                content = git(bare, 'cat-file', 'blob', row['blob']).decode()
                assert content.startswith('version https://git-lfs.github.com/spec/v1\n')
                pointers.setdefault(row['path'], {})[row['blob']] = content
            elif row['blob'] not in seen and pathlib.PurePosixPath(row['path']).suffix in EXTENSIONS:
                seen.add(row['blob'])
                content = git(bare, 'cat-file', 'blob', row['blob']).decode()
                data = json.loads(content) if row['path'].endswith('.json') else {}
                rows.append({'path': row['path'], 'blob': row['blob'], 'commit': row['commit'],
                             'channel_names': data.get('channel_names', data.get('channelNames')),
                             'source_root': data.get('source_root'), 'derived': data.get('prov:wasDerivedFrom'),
                             'n_channels': data.get('channel_count', data.get('channelCount'))})
        mapping_history[output_key], pointer_history[output_key] = rows, pointers
    (OUT/'historical_tuev_mapping_fields.json').write_text(json.dumps(mapping_history, indent=2), encoding='utf-8')
    (OUT/'historical_h5_lfs_pointers.json').write_text(json.dumps(pointer_history, indent=2), encoding='utf-8')
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id='tankalapavankalyan/eeg-corpus-manifest', repo_type='dataset',
                             revision='e4c2d62a9230d053c948a6681a63d1ca0d3f353e',
                             allow_patterns=['*.parquet', 'README.md', 'BUILD_METADATA.json', '.gitattributes'],
                             local_dir=str(OUT/'raw_manifest'), max_workers=2, etag_timeout=10)
    print('RAW_METADATA_ONLY_DOWNLOADED', path, flush=True)

if __name__ == '__main__':
    main()
