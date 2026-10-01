# Solver-Consistent Recurrent Reasoning — Stage 0 V2

This directory is a clean V2 execution of the controlled Stage-0 experiment. It does not reuse V1 data, tokenizers, sanity checkpoints, or formal checkpoints.

## Current gate order

1. Generate the full 135,000 examples with `data/generate_v2.py`.
2. Run `data/audit_data_v2.py` over every row.
3. Run `data/shortcut_suite.py`; no tokenization or neural training is allowed unless it passes.
4. Train a BPE tokenizer only on train, freeze max length, then run model, solver, infrastructure, and learning gates.
5. Formal seed-0 training and evaluation start only after every prior gate passes.

The versioned adapter leaves the pinned vendored PrOntoQA source unchanged. It replaces the isolated opposite-goal dummy with a predicate-renamed mirror component whose signed start literal is unavailable to the queried entity. The manifest records the upstream commit, generator version, checksums, exact balances, and global deduplication policy.

## Commands

```powershell
python data\generate_v2.py --output data\generated --workers 8 --seed-namespace 2000
python data\audit_data_v2.py
python data\shortcut_suite.py
python -m pytest tests -q
```

`artifacts/status.json` is authoritative for the current pipeline state. If data validation fails after the allowed repair rounds, downstream GPU stages remain `NOT_RUN` and no scientific GO/STOP claim is made.
