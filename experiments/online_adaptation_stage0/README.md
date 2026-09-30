# Online adaptation benchmark: Stage 0

This directory is the runnable one-subject gate for the requested benchmark.
It creates a shared, immutable BNCI2014001 A01 cache from the official 2a GDF
files and official competition labels, then evaluates a source-only official
SincAdaptNet, the official BTTA-DG online code path, and the requested
NeuroLoop controls under both full-session EA and strict causal EA.

The command must not be interpreted as a full result: it is intentionally
restricted to `A01`, `S1 -> S2`, seed `0`.  Full 9-subject / 10-seed execution
is prohibited until `causal_audit.json` passes.

```bash
/opt/conda/bin/python run_stage0.py \
  --gdf-dir /root/rivermind-data/datasets/bcic_iv_2a_2b/gdf/2a \
  --labels-dir /root/bciciv2a_labels \
  --official-btta-dir /root/BTTA-DG-official \
  --output /root/rivermind-data/online_adaptation_stage0/bnci2014001_a01_seed0
```

Before running the command, execute `pytest -q test_online_protocol.py`.
All models receive the exact same cached trial arrays.  `run_chronological`
does not accept labels, uses batch size one, and orders every trial as
`predict()` then `update()`.
