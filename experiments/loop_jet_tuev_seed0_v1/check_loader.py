"""Representation refusal, worker-local H5 handles and fixed-flow invariants."""
import ctypes
import gc
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

from process_affinity import configure_process_affinity
configure_process_affinity()
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from lazy_h5 import LazyTUEV, fixed_flow_inputs

ROOT = pathlib.Path(__file__).resolve().parent

def main():
    out = ROOT/'analysis'
    out.mkdir(exist_ok=True)
    result = {'synthetic_only': True}
    with tempfile.TemporaryDirectory(dir=str(out), prefix='loader_smoke_') as directory:
        folder = pathlib.Path(directory).resolve()
        assert folder.is_relative_to(out.resolve())
        expected = np.arange(2*16*1000, dtype=np.float64).reshape(2, 16, 1000)
        for transpose in (False, True):
            path = folder/f'good_{transpose}.h5'
            with h5py.File(path, 'w') as f:
                f['X'] = expected.transpose(0, 2, 1) if transpose else expected
                f['y'] = [0, 5]
            ds = LazyTUEV(path)
            assert ds._handle is None
            x, y, i = ds[0]
            assert x.dtype == torch.float32 and tuple(x.shape) == (16, 5, 200)
            assert torch.equal(x.view(16, 1000), torch.tensor(expected[0], dtype=torch.float32)/100)
            loader = DataLoader(ds, batch_size=2, num_workers=2)
            batch = next(iter(loader))
            assert torch.equal(batch[0][0], x) and batch[1].tolist() == [0, 5]
            ds.close()
            del loader, ds
            gc.collect()
        result.update(lazy_before_first_read=True, transpose_and_float64_conversion=True,
                      windows_two_worker_loading=True, official_divide_by_100=True)
        bad = folder/'bad_32.h5'
        with h5py.File(bad, 'w') as f:
            f['X'] = np.zeros((2, 32, 1000), dtype=np.float32)
            f['y'] = [0, 5]
        try:
            LazyTUEV(bad)
        except ValueError:
            result['rejects_32_channels'] = True
        else:
            raise AssertionError('32-channel input was silently accepted')
        flow_args = SimpleNamespace(P_mean=-.8, P_std=.8, noise_scale=1.)
        xx = torch.zeros(3, 16, 5, 200)
        za, ta = fixed_flow_inputs(xx, [3, 8, 11], flow_args)
        zb, tb = fixed_flow_inputs(xx[1:], [8, 11], flow_args)
        zc, tc = fixed_flow_inputs(xx, [3, 8, 11], flow_args, split='test')
        assert torch.equal(za[1:], zb) and torch.equal(ta[1:], tb)
        assert not torch.equal(za, zc)
        result.update(fixed_noise_batch_order_independent=True, distinct_val_test_streams=True)
    (out/'loader_sanity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
