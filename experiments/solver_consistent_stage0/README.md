# Solver-Consistent Recurrent Reasoning — Stage 0

**Current status: UNCLEAR / BLOCKED_INVALID_DATA. Formal seed-0 comparison is NOT complete.**

A text-only, zero-training antecedent-frequency heuristic solves all 135,000 generated examples, including depths6/8/12. The current dataset cannot isolate multihop reasoning. Initial completed depth1 learning gate stayed at50%; later development retries suffered native Windows/CUDA crashes. The last low-memory retry was stopped explicitly upon discovering the structural shortcut. No formal model has trained. Read `artifacts/STAGE0_REPORT.md` before running anything.

## Files and evidence

- `artifacts/STAGE0_REPORT.md`: six-section report, actual evidence and incomplete deliverables.
- `artifacts/decision.json`: UNCLEAR with `formal_seed0_complete=false`.
- `artifacts/data_shortcut_results.csv`: actual measured no-training shortcut accuracy by split/depth.
- `artifacts/stage0_results.csv`: planned neural evaluation conditions; unmeasured fields NA, status NOT_RUN_INVALID_DATA. This is NOT completed model-result data.
- `artifacts/fairness_table.csv`: parameter counts, planned configuration and zero completed formal updates.
- `artifacts/sanity_checks.json`, `unit_tests.txt`, `semantic_data_audit.json`: recorded gates and tests.
- `artifacts/windows_crash_events.json`: fault-module evidence, not a proven root cause.
- `artifacts/sanity*`: archived real development checkpoints/logs. No sanity weights initialize formal models.
- `configs/frozen.json` and three model YAMLs: final declared configuration. Earlier development changes are recorded in `EXPERIMENT_CHANGELOG.md`.

## Reproduce the data diagnosis

Remote Windows project: `D:\solver_consistent_stage0`. Independent venv `.venv28` inherits PyTorch2.8.0+cu128 from `E:\Anaconda\envs\persist_stable_251`, without modifying that environment. Invoke from an activated matching Conda environment:

```bat
call E:\Anaconda\Scripts\activate.bat E:\Anaconda\envs\persist_stable_251
cd /d D:\solver_consistent_stage0
.venv28\Scripts\python.exe data\audit_structural_shortcut.py
.venv28\Scripts\python.exe -m pytest tests -q
.venv28\Scripts\python.exe abort_report.py
```

`run_pipeline.py` performs tests, semantic audit and shortcut audits before neural training. On the current dataset it writes the incomplete report and ends with status `blocked`, decision UNCLEAR. Do not remove this gate merely to obtain model scores. Repair/version the data and regenerate all splits, tokenizer and learning gates before a formal comparison. Do not reuse a passing gate or old tokenizer after data/config changes.

Persistent execution on this Windows SSH server requires an open SSH connection; ordinary Start-Process descendants are killed upon disconnect. `resume28.cmd` runs in the foreground with logs redirected. Host-specific SSH upload helpers are excluded from this publication. Checkpoints `.pt` and token arrays `.npz` remain on the original remote host. No passwords or SSH connection credentials are published.

## Data protocol and limitation

Official generator https://github.com/asaparov/prontoqa at commit `0a6412b6fddf46324a1cb96e066dd7b3d89b87d6`, vendor unmodified with license. Fictional ontology, ModusPonens, relevant distractors. Finite default property pools could not generate deep OOD; only lexical API inputs were extended to2048 concepts and32 disjoint four-property families. This is a lexically extended official configuration, not the default upstream distribution.

`data/generate_parallel.py` uses8 ordered CPU processes and per-block seed0-derived SeedSequence. Exact sizes100000/10000/10000/5000/5000/5000, label/depth quotas and SHA256 are recorded in `data/generated/manifest.json`. Depth d maps to official steps=d+1. Only input text/final binary labels supervise models. Canonical underlying instances are globally disjoint, including entity/query-polarity normalization; predicate-renamed isomorphic templates can recur.

960 stratified samples pass independent label/minimum-hop verification. That audit does not rule out shortcuts: the current official opposite-goal dummy rule has an isolated antecedent, enabling the100% frequency heuristic. Query-only/bag-of-words/parity controls around50% miss this structural leakage. Rejected default/no-distractor pilots are preserved and excluded from training.

Train-only BPE target cap8192, actual vocabulary2289; lowercase and digit-span tokenization. max_len704 gives0% truncation on every split. Final model configs use actual-vocabulary embedding allocation, width512/heads8/FF2048/stem3/core2, math attention, dropout.1. Parameter counts17296386/17313826/17313826, spread0.10083%. Final engineering configuration microbatch8/accumulation16/effective128, activation checkpointing and64-token padding buckets; all formal models planned for2000 equal updates, BF16 AdamW3e-4 with same examples/budgets. None completed.

## Implemented methods and planned evaluation

Vanilla H<-B(H) receives no time/dt/budget or1/K scale. Step-conditioned H<-H+G(H,t,dt) uses Fourier/MLP conditioning. VectorField.forward(H,t,padding_mask) cannot receive dt/K/NFE/solver; direction is residual increment and the second encoding slot stays0. Solver applies dt externally. Common initialization and three-model capacity are matched; only CE is optimized.

Hand-written Euler/Heun/RK4,1/2/4 NFE per step, [0,1], matched NFE4/8/12/16. Uniform and user-defined front/back-loaded positive schedules sum to1. Ordinary loops never receive solver swaps.

Planned task evaluation includes trained4/8 and unseen3/5/7/12/16, harmful/recovery/total flips versus same-model uniform8, accuracy/NLL/two-class Brier/JS/confidence, within-model hidden endpoints, bypass and matched-solver comparison. Numerical diagnostics use512 IID samples, FP32, RK4 references32/64steps, no imposed convergence slope. Latency uses BF16/fixed batch32/20warmups/100timed IID batches per configuration. These trained-model diagnostics and four requested main figures remain unexecuted. Unit-test ODE convergence does not establish reasoning benefit.

The2000-update exploratory budget and all engineering changes are documented before formal comparisons. Next steps are data repair and runtime diagnosis, then all three learning gates from scratch. Extra seeds/losses on the current shortcut-dominated data do not answer the user's hypothesis.

## GitHub publication snapshot

This directory was added alongside the existing EEG project without changing its source. Extract the complete135000-example data archive before reproducing the audit:

```bash
cd experiments/solver_consistent_stage0
python unpack_dataset.py
python data/audit_structural_shortcut.py
python -m pytest tests -q
```

The50MiB `dataset.zip` is checksum-verified against its manifest and includes all six JSONL splits. Expanded train.jsonl exceeds the GitHub100MiB single-file limit and is intentionally untracked. Final configuration, train-only tokenizer, logs, reports, CSVs and upstream source/license are included. Large neural checkpoints, token arrays, discarded dataset pilots, upstream result archives and machine-specific SSH transfer helpers are excluded; the report preserves their measured metadata. Formal method results are incomplete/NA, not100%: the100% accuracy belongs only to the no-training structural shortcut audit.
