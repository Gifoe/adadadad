"""Worker-local lazy HDF5; refuse representations outside the frozen protocol."""
import os
import pathlib

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LazyTUEV(Dataset):
    def __init__(self, path, *, x_key='X', y_key='y', amplitude_divisor=100.0):
        self.path = str(pathlib.Path(path).resolve())
        self.x_key, self.y_key = x_key, y_key
        self.amplitude_divisor = float(amplitude_divisor)
        if self.amplitude_divisor <= 0:
            raise ValueError('Amplitude divisor must be positive.')
        self._handle, self._pid = None, None
        with h5py.File(self.path, 'r') as f:
            shape = f[x_key].shape
            if tuple(shape[1:]) not in ((16, 1000), (1000, 16)):
                raise ValueError(f'Incompatible EEG representation: {shape}; required [N,16,1000].')
            self.transpose = tuple(shape[1:]) == (1000, 16)
            self.labels = np.asarray(f[y_key]).reshape(-1).astype(np.int64)
            if len(self.labels) != shape[0] or not np.isin(self.labels, np.arange(6)).all():
                raise ValueError('Invalid labels or mismatched X/y lengths.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            if self._handle is not None:
                self._handle.close()
            self._handle = h5py.File(self.path, 'r')
            self._pid = pid
        x = np.asarray(self._handle[self.x_key][index], dtype=np.float32)
        if self.transpose:
            x = x.T
        x = np.ascontiguousarray(x / self.amplitude_divisor)
        return torch.from_numpy(x).view(16, 5, 200), int(self.labels[index]), int(index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'], state['_pid'] = None, None
        return state

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def fixed_flow_inputs(x, indices, denoiser, seed=20261005, split='val'):
    """Validation/test RNG independent of training and stable across batch/worker order."""
    if split not in ('val', 'test'):
        raise ValueError('Frozen flow draws are for held-out splits only.')
    offset = 0 if split == 'val' else 1000000007
    ts, noises = [], []
    for index in indices:
        generator = torch.Generator(device='cpu').manual_seed((seed+offset+int(index)*100003) % (2**63-1))
        t = torch.sigmoid(torch.randn((), generator=generator)*denoiser.P_std + denoiser.P_mean)
        noise = torch.randn(tuple(x.shape[1:]), generator=generator)*denoiser.noise_scale
        ts.append(t)
        noises.append(noise)
    t = torch.stack(ts).to(x.device).view(-1, 1, 1, 1)
    noise = torch.stack(noises).to(x.device)
    return t*x + (1-t)*noise, t
