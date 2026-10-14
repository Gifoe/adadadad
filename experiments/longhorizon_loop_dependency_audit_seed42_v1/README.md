# Long-Horizon Loop Dependency Audit — seed 42

This is a frozen-weight viability audit. Its status and measured outcomes are in
[`outputs/FINAL_REPORT.md`](outputs/FINAL_REPORT.md), with per-branch records and
problem-cluster confidence intervals. Native SDPA results are conditional on this
backend. Official mini-SGLang serving equivalence is reported separately.

Run from this experiment directory. Python 3.12 and an RTX 5090-compatible PyTorch
CUDA build are required. Nothing here trains a model or updates its parameters.

```bash
git clone https://github.com/thu-nics/TaH.git upstream/TaH
git -C upstream/TaH checkout 78ddf9a2c25713fced03a736c5746161f6ee590c
git clone https://github.com/IBM/NESTFUL.git upstream/NESTFUL
git -C upstream/NESTFUL checkout fc2c4123e73500a56185a5fb354f05d1c8b4890c
python -m venv .venv
# Activate .venv for your shell, then:
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-native.txt
pip install --no-deps -e upstream/TaH
python scripts/download_models.py
python scripts/prepare_data.py --tokenize
pytest -q
python scripts/preflight_model.py
python scripts/baseline_pilot.py
python scripts/core_audit.py --phase dev
python scripts/analyze_results.py --phase dev
python scripts/core_audit.py --phase audit
python scripts/analyze_results.py --phase audit
python scripts/core_audit.py --phase rename
python scripts/analyze_results.py --phase rename
python scripts/core_audit.py --phase dev_aligned
python scripts/analyze_results.py --phase dev_aligned
python scripts/core_audit.py --phase audit_aligned
python scripts/analyze_results.py --phase audit_aligned
python scripts/core_audit.py --phase rename_aligned
python scripts/analyze_results.py --phase rename_aligned
python scripts/run_remaining.py
# run_remaining includes the isolated 2048-token probe, H2/H8 extension,
# all analyses, tests, report and provenance.
python scripts/generate_report.py
python scripts/capture_provenance.py
python scripts/artifact_checks.py
```

The downloader verifies pinned Hugging Face revisions and records file hashes.
It tries the mirror for connectivity and falls back to the official endpoint.
Full weights, upstream checkouts and full-vocabulary anchor logits are not pushed
to Git. Raw logits remain in `outputs/anchor_logits/` on the execution server.
The compact records contain target CE, top-8 raw logits, actual executed loop
counts, stopper scores, frozen text hashes and counterfactual conditions.

For a full official serving reproduction on Linux, use the official
`pip install -e '.[tah2]'` inside the upstream repository before the serving smoke
test. Windows cannot resolve its Triton wheel; the serving smoke test also records
its actual runtime exception. Installing the native dependencies is not evidence
that the official serving backend was reproduced.

`Standard` has independently trained weights. Only forced-1, forced-2 and native
adaptive share the same max2 weights. Forcing depth preserves the learned
`stop_prob_mix`; it does not use a last-loop-only prediction.

The 736-row fixed manifest contains 96 NESTFUL problems and 640 generated
diagnostic inputs: 80 groups × four depths × base/renamed symbols. The primary
audit uses 64 NESTFUL problems and 64 controlled groups at H4/H16. The latter are
128 inputs but 64 independent groups. H2/H8 are prepared and are not scored as
primary audit data. Renaming is an explicitly separate stability control.

Semantic anchors are prediction positions immediately before the meaningful
identifier/state token. The cache entry being deepened has not consumed that
output token yet. This tests computation that predicts the source output, rather
than an intervention on the emitted token embedding. NESTFUL tests binding
semantics; numeric tool results are not inserted into its reference trace.
Correct teacher-forced intermediate states can screen longer-range effects.

Development results set the local-gain threshold once. Audit scripts require the
frozen threshold file, manifest/config hashes and unchanged case plan. Each branch
replays the same token IDs from an independent cloned cache. Incomplete branches
are resumed by case/condition; already completed branches are skipped. Different
protocols must use a new experiment directory rather than overwriting this audit.

Statistics weight each original problem/group equally; multiple pairs are not
independent observations. The single-anchor DUO identity `I = G_source` is checked
as an architectural null, not treated as a discovery. A whole-span utility is
also reported and is not assumed to obey that identity. An H16 chain has no valid
earlier nondependent same-role state; its unavailable control is recorded.

Source and license attributions are in [`PRIMARY_SOURCES.md`](PRIMARY_SOURCES.md).
The controlled task is newly generated diagnostic data. Prospective unpublished
method ideas are kept outside the public reproduction tree.

The published outputs are immutable audit evidence. For a fresh execution, copy
the code, config, tests and documentation to a new directory with an initially
empty outputs folder. Do not regenerate the locked manifest over downloaded
results. Raw JSON/JSONL artifacts preserve their byte/newline format in Git so
their recorded hashes remain verifiable after checkout.

On the Windows execution host, launch the checked foreground pipeline through
`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_server.ps1`
for a hidden WMI-owned job that survives SSH disconnections. Its PID and logs
are recorded under outputs/. The runner applies process-only affinity using
observed Windows CPU-set efficiency classes. No BIOS or global scheduler setting
is changed. This is an operational workaround, not proof of a hardware diagnosis.
