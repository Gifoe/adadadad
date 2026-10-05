"""Explicit representation gate. Exit 2 means no benchmark/training may begin."""
import ctypes
import json
import pathlib
import sys

from process_affinity import configure_process_affinity
configure_process_affinity()
import h5py
from lazy_h5 import LazyTUEV

ROOT = pathlib.Path(__file__).resolve().parent

def main():
    audit = json.loads((ROOT/'data_audit'/'h5_audit.json').read_text(encoding='utf-8'))
    blockers = list(audit['blockers'])
    rejected = {}
    for split in ('train', 'val', 'test'):
        path = ROOT/'data'/'hf_tuev'/'TUEV'/f'{split}.h5'
        with h5py.File(path, 'r') as f:
            shape = tuple(f['X'].shape[1:])
            if shape not in ((16, 1000), (1000, 16)):
                blockers.append(f'Live preflight {split}: shape {shape} incompatible')
        try:
            dataset = LazyTUEV(path)
        except ValueError as exc:
            rejected[split] = str(exc)
        else:
            dataset.close()
    result = {'pass': not blockers, 'blockers': blockers, 'actual_loader_rejections': rejected,
              'optimizer_steps_on_TUEV': 0, 'scientific_results_available': False}
    (ROOT/'analysis').mkdir(exist_ok=True)
    (ROOT/'analysis'/'preflight.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if blockers:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
