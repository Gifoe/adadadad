# Server-side Codex execution instructions

You are inside the researcher's RTX 5090 server. Execute the bundled
`neso_worldts_online_v1` project to completion; do not merely summarize or write
more planning documents.

## Objective

Verify data availability and genuine forecast-vintage revision pairs from NESO
2025/2026. Then fit and evaluate the authors' numerical `WorldTSModel`, first
without a novel updater, and measure the accuracy/compute headroom of updating
its forecast with revised wind/solar predictions. Finally train a lightweight
single-step latent updater **only if** strong baselines leave meaningful room.

## Rules

1. Clone the authors' repository, pin the official commit and verify tensor
   shapes and native `LatentStateLoss`; do not swap in an invented Transformer.
2. Keep PyTorch/CUDA compatible with the RTX 5090. Do not let old upstream
   `requirements.txt` silently downgrade a working CUDA build.
3. Download all five official NESO resources through `python -m neso_online.download`.
   Inspect file content, source metadata, hashes, duplicate keys, and UTC/DST.
4. Run unit tests. Build data with `python -m neso_online.prepare`. If sample
   coverage is poor, audit the actual issue times/target periods before making
   code changes. Never use future issue data to repair missing A covariates.
5. Train original WorldTS latent predictor then original Decoder using V1
   adapted hyperparameters. Record all commands, stdout/stderr, seeds, epochs,
   validation losses, GPU timings, memory peaks, and checkpoints.
6. Evaluate on validation: old A forecast, full B recomputation with fixed
   A-history, seasonal naive, and train-only observation-space correction.
7. ONLY if validation has a reproducible non-trivial update problem, fit the
   small `LatentRevisionUpdater` and compare it against full recomputation and
   output correction. It is NOT a Looped Transformer or a verified contribution.
8. Freeze decisions before terminal test replay. Run held-out test once.
9. Write `FINAL_REPORT.md` with full dataset schemas, verified sample counts,
   missingness, 2026 archives overlap, actual published-time assumptions,
   observed metrics, per-update correct/wrong fractions, latency, FLOPs if
   measurable, best checkpoint and evidence for GO / NO-GO.
10. Explicitly state: NESO historic ND is published 21 days in arrears and may
    be revised, and `Forecast_Datetime` is forecast production time, not
    guaranteed public release time. The current data protocol is a retrospective
    forecast-vintage replay with **assumed instantaneous ND availability**.
    A fully causal real-time deployment study needs verified observation
    publication-time metadata. Do not claim it is strictly live-causal.

## Commands

```bash
cd neso_worldts_online_v1
python -m pytest -q
python -m neso_online.download --out data/raw
python -m neso_online.prepare \
  --forecast data/raw/forecast_2025.csv data/raw/forecast_2026_jan_jun.csv data/raw/forecast_2026_jun_dec.csv \
  --demand data/raw/demand_2025.csv data/raw/demand_2026.csv \
  --lookback 336 --horizon 48 --revision-hours 6 --out outputs/dataset
python -m neso_online.native_worldts train --data outputs/dataset \
  --out outputs/model --worldts-source upstream/WorldTS/DAG-bench \
  --config configs/worldts_neso.json --device cuda --batch 16
python -m neso_online.native_worldts train_updater --data outputs/dataset \
  --out outputs/model --worldts-source upstream/WorldTS/DAG-bench \
  --config configs/worldts_neso.json --device cuda --batch 16
python -m neso_online.native_worldts replay --data outputs/dataset \
  --out outputs/model --worldts-source upstream/WorldTS/DAG-bench \
  --config configs/worldts_neso.json --device cuda --batch 16
```

If native dependency/version/API compatibility changes are necessary, preserve
all modifications in a minimal patch and explain their effect on reproduction.

## Final decision

A **GO** requires evidence that the latent updater beats a simple output-space
revision and that it has a meaningful accuracy/cost advantage relative to the
new-covariate **full recomputation** baseline. Otherwise mark NO-GO rather than
continuing indiscriminate architecture search. Evaluation by target time
blocks is essential because windows overlap.
