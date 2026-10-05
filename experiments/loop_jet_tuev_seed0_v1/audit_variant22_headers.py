"""Read HDF5 headers/labels via bounded HTTP ranges; never fetch a full signal file."""
import collections
import io
import json
import os
import pathlib
import re
from process_affinity import configure_process_affinity
configure_process_affinity()
import h5py
import numpy as np
import requests

ROOT = pathlib.Path(__file__).resolve().parent
REV = '41602cf9363d83204e6b7145b04efaeafebd9ae4'
REPO = 'USuCgex0122e/demo_dataset_seed20260726'
ENDPOINT = 'https://hf-mirror.com'

class RangeFile(io.RawIOBase):
    def __init__(self, session, url, size, limit=4*1024*1024):
        self.session, self.url, self.size, self.limit = session, url, size, limit
        self.position, self.transferred = 0, 0
        self.cache = {}
        self.blocksize = 65536
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position
    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        result = bytearray()
        while size > 0:
            block = self.position//self.blocksize
            start = block*self.blocksize
            if block not in self.cache:
                end = min(self.size-1, start+self.blocksize-1)
                if self.transferred+end-start+1 > self.limit:
                    raise RuntimeError('H5 header range-read budget exceeded; do not download signals.')
                with self.session.get(self.url, headers={'Range': f'bytes={start}-{end}',
                                                        'Accept-Encoding': 'identity'}, stream=True, timeout=30) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f'Server ignored HTTP range ({response.status_code}); closed without reading body.')
                    value = response.headers.get('Content-Range', '')
                    if value != f'bytes {start}-{end}/{self.size}':
                        raise RuntimeError('Mismatched range: '+value)
                    data = response.content
                    assert len(data) == end-start+1
                self.cache[block] = data
                self.transferred += len(data)
            offset = self.position-start
            part = self.cache[block][offset:offset+size]
            result.extend(part)
            self.position += len(part)
            size -= len(part)
        return bytes(result)

def safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode(errors='replace')
    return v

def main():
    session = requests.Session()
    session.trust_env = False
    entries = session.get(ENDPOINT+'/api/datasets/'+REPO+'/tree/'+REV+'/TUEV', timeout=30).json()
    sizes = {row['path']: row['size'] for row in entries}
    result = {'repo': REPO, 'revision': REV, 'endpoint': ENDPOINT, 'splits': {},
              'policy': 'Only headers and labels, max 4 MiB per file, abort on non-206. No X signal slices.'}
    for split in ('train', 'val', 'test'):
        path = 'TUEV/'+split+'.h5'
        file = RangeFile(session, ENDPOINT+'/datasets/'+REPO+'/resolve/'+REV+'/'+path, sizes[path])
        entry = {'file_bytes': sizes[path]}
        try:
            with h5py.File(file, 'r') as f:
                entry['root_attrs'] = {k: safe(v) for k, v in f.attrs.items()}
                entry['datasets'] = {}
                for key in f:
                    entry['datasets'][key] = {'shape': list(f[key].shape), 'dtype': str(f[key].dtype),
                                             'attrs': {k: safe(v) for k, v in f[key].attrs.items()}}
                labels, counts = np.unique(np.asarray(f['y']), return_counts=True)
                entry['class_counts'] = {str(int(k)): int(v) for k, v in zip(labels, counts)}
                entry['verified_hdf5_header'] = True
        except Exception as exc:
            entry['error'] = str(exc)
            entry['verified_hdf5_header'] = False
        entry['http_bytes_transferred'] = file.transferred
        result['splits'][split] = entry
        print(split, json.dumps(entry), flush=True)
    (ROOT/'data_audit'/'variant22_h5_header_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
