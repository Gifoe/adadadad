"""Unscaled random magnitude diagnostics are observations, never unit inference."""
import json
import pathlib
from process_affinity import configure_process_affinity
configure_process_affinity()
import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
results = {'seed': 20261005, 'unit_status': 'unknown',
           'policy': 'Magnitude does not prove V, mV, uV, normalized, or z-scored units.', 'splits': {}}
for split in ('train', 'val', 'test'):
    with h5py.File(ROOT/'data'/'hf_tuev'/'TUEV'/f'{split}.h5', 'r') as f:
        rng = np.random.default_rng(20261005)
        indices = sorted(rng.choice(len(f['X']), size=10, replace=False).tolist())
        samples = np.stack([f['X'][i] for i in indices])
        results['splits'][split] = {
            'indices': indices, 'finite': bool(np.isfinite(samples).all()),
            'min': float(samples.min()), 'max': float(samples.max()), 'mean': float(samples.mean()),
            'std': float(samples.std()), 'percentile_levels': [.1, 1, 5, 50, 95, 99, 99.9],
            'percentiles': np.percentile(samples, [.1, 1, 5, 50, 95, 99, 99.9]).tolist(),
            'channelwise_std': samples.std(axis=(0, 2)).tolist(),
            'all_zero_sample_channels': int(np.all(samples == 0, axis=-1).sum()),
        }
(ROOT/'data_audit'/'unscaled_magnitude32.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps({k: {'std':v['std'], 'all_zero_sample_channels':v['all_zero_sample_channels']} for k,v in results['splits'].items()}), flush=True)
