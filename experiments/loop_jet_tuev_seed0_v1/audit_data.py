"""Inspect actual HDF5 contents before any benchmark or training is allowed."""
import ctypes
import hashlib
import json
import pathlib
import sys

from process_affinity import configure_process_affinity
configure_process_affinity()
import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'hf_tuev' / 'TUEV'
OUT = ROOT / 'data_audit'
EXPECTED = {
    'train': (615573204, 'e40adf5321a73e394d0ccae4bc50286cf85248bcaa36ea699e182b42e95c2d56'),
    'val': (165604803, 'bcf0c4664994864b7a4f54e7327e36dbb0cea446b180051a11aaae0711aafd5c'),
    'test': (144788691, '774fb4957dd8148eeccddc4c63f283ccc30e8d29404bc2da788deb7b58ceaac0'),
}

def safe(v):
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')
    if isinstance(v, np.ndarray):
        return [safe(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return safe(v.item())
    return v

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((DATA / 'metadata_summary.json').read_text(encoding='utf-8'))
    result = {'metadata': meta, 'croissant_manifest': (DATA / 'croissant_manifest.csv').read_text(encoding='utf-8'),
              'splits': {}, 'blockers': [], 'split_equivalence': 'unverified', 'montage_identity': 'unverified'}
    for split, (size, digest) in EXPECTED.items():
        path = DATA / (split + '.h5')
        entry = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        assert entry['bytes'] == size and entry['sha256'] == digest, f'{split}: incomplete or corrupt download'
        with h5py.File(path, 'r') as f:
            entry['root_attrs'] = {k: safe(v) for k, v in f.attrs.items()}
            datasets = {}
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = {'shape': list(obj.shape), 'dtype': str(obj.dtype),
                                      'attrs': {k: safe(v) for k, v in obj.attrs.items()},
                                      'compression': obj.compression, 'chunks': list(obj.chunks) if obj.chunks else None}
            f.visititems(visit)
            entry['datasets'] = datasets
            xkey = next((k for k in datasets if k.lower() in ('x', 'eeg', 'signals', 'data')), None)
            ykey = next((k for k in datasets if k.lower() in ('y', 'labels', 'label')), None)
            assert xkey and ykey, f'Unexpected H5 keys: {list(datasets)}'
            x, y = f[xkey], np.asarray(f[ykey]).reshape(-1)
            labels, counts = np.unique(y, return_counts=True)
            entry.update(x_key=xkey, y_key=ykey, n_samples=len(y),
                         class_counts={str(int(k)): int(v) for k, v in zip(labels, counts)})
            assert x.shape[0] == len(y), 'X/y length mismatch'
            entry['finite_first_last'] = bool(np.isfinite(x[0]).all() and np.isfinite(x[-1]).all())
            entry['sample0_min_max_mean_std'] = [float(fn(x[0])) for fn in (np.min, np.max, np.mean, np.std)]
            shape = tuple(x.shape[1:])
            if shape not in ((16, 1000), (1000, 16)):
                result['blockers'].append(f'{split}: actual X.shape={list(x.shape)}; required [N,16,1000] or axis transpose')
            if set(labels.tolist()) != set(range(6)):
                result['blockers'].append(f'{split}: six-class zero-based labels not confirmed')
        result['splits'][split] = entry
        print(split, json.dumps(entry, ensure_ascii=False), flush=True)
    if meta.get('sampling_rate_hz') != 200 or meta.get('window_length_seconds') != 5:
        result['blockers'].append('Sampling rate/window incompatible with 200 Hz / 5 seconds')
    result['disk_bytes'] = sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
    result['compatible'] = not result['blockers']
    result['channel_policy'] = 'No selection, truncation, re-referencing, resize, or resampling permitted without named verified montage.'
    (OUT / 'h5_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for filename in ('metadata_summary.json', 'croissant_manifest.csv', 'croissant.json'):
        (OUT / filename).write_bytes((DATA / filename).read_bytes())
    print('COMPATIBLE', result['compatible'], flush=True)
    print('BLOCKERS', result['blockers'], flush=True)

if __name__ == '__main__':
    main()
