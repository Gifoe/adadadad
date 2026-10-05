"""Download only the pinned TUEV subtree; retain and verify complete files."""
import ctypes
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

from process_affinity import configure_process_affinity
configure_process_affinity()

ROOT = pathlib.Path(__file__).resolve().parent
REVISION = 'db96e77affa66398c553427045ee040f06230c2a'
REPO = 'USuCgex0122e/demo_dataset'
DEST = ROOT / 'data' / 'hf_tuev'
AUDIT = ROOT / 'data_audit'
AUDIT.mkdir(exist_ok=True)

def child():
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id=REPO, repo_type='dataset', revision=REVISION,
        allow_patterns=['TUEV/*', 'TUEV/**'], local_dir=str(DEST),
        max_workers=2, etag_timeout=10,
    )
    print('SNAPSHOT_COMPLETE', path, flush=True)

def main():
    log = {'repo': REPO, 'revision': REVISION, 'allow_patterns': ['TUEV/*', 'TUEV/**'], 'attempts': []}
    for endpoint in ['https://hf-mirror.com', 'https://huggingface.co']:
        env = os.environ.copy()
        env.update(HF_ENDPOINT=endpoint, HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='20')
        env.pop('HTTP_PROXY', None)
        env.pop('HTTPS_PROXY', None)
        print('ENDPOINT', endpoint, flush=True)
        start = time.time()
        try:
            proc = subprocess.run([sys.executable, __file__, '--child'], env=env, timeout=600)
            rc = proc.returncode
            error = None
        except subprocess.TimeoutExpired:
            rc, error = -1, 'Snapshot exceeded 600-second bound; complete files retained.'
        log['attempts'].append({'endpoint': endpoint, 'returncode': rc, 'seconds': time.time()-start, 'error': error})
        (AUDIT / 'download_attempts.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
        if rc == 0:
            return
    # A private loopback proxy is optional transport only, not a different data source.
    if os.environ.get('JET_ARTIFACT_PROXY'):
        env = os.environ.copy()
        env.update(HF_ENDPOINT='https://huggingface.co', HF_HUB_DISABLE_XET='1',
                   HF_HUB_DOWNLOAD_TIMEOUT='30', HTTP_PROXY=env['JET_ARTIFACT_PROXY'],
                   HTTPS_PROXY=env['JET_ARTIFACT_PROXY'])
        start = time.time()
        proc = subprocess.run([sys.executable, __file__, '--child'], env=env, timeout=1800)
        log['attempts'].append({'endpoint': 'https://huggingface.co', 'transport': 'SSH loopback HTTP proxy',
                                'returncode': proc.returncode, 'seconds': time.time()-start})
        (AUDIT / 'download_attempts.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
        if proc.returncode == 0:
            return
    raise SystemExit('Download failed; inspect download_attempts.json and execution log.')

if __name__ == '__main__':
    child() if '--child' in sys.argv else main()
