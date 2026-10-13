# NESO Vintage-Aware WorldTS — reproducible V1

**Scope:** prepare a vintage-aware NESO forecasting dataset for 2025–2026; run
**the native numerical WorldTS core** with its two-stage latent/decoder training;
evaluate whether revised future wind/solar predictions justify a stateful latent
update. The experiment is a **covariate-vintage retrospective online replay**,
**not** an independently validated live operational prediction benchmark.

## Current state

- Implemented: official NESO resource manifest + streamed CSV downloader; GMT / UK
  settlement day alignment; `Forecast_Datetime` issue-time grouping; A/B paired
  covariates for **the same target horizon**; strict chronological target splits;
  original WorldTS model import and stage-1/stage-2 training adapter; fixed history;
  old/fully-recomputed forecasts; train-only output-space calibration; optional
  lightweight persistent-latent correction; masked MSE/MAE and runtime measures.
- Verified here: 7 tests (including DST and same-target alignment), synthetic-data
  prepare pass; generic two-stage training/replay driver pass with a **dummy core**.
- Not completed here: **NESO full-year CSV download**, real sample audit,
  native WorldTS training, RTX 5090 GPU timings, and actual held-out test scores.
  These require a GPU server with network access to NESO and official WorldTS files.
- Do not treat the dummy core/synthetic fixture as a WorldTS or NESO result.

## 1. Native code and environment

On the RTX 5090 machine, create an isolated environment with a recent CUDA-capable
PyTorch (e.g. your verified PyTorch 2.8+cu128 environment). **Do not overwrite it
with DAG-bench's pinned PyTorch 2.4.1**, which predates Blackwell support.

```bash
cd neso_worldts_online_v1
python -m pip install -r requirements.txt
# Additional upstream package imports may require:
python -m pip install scikit-learn scipy einops reformer-pytorch tqdm matplotlib

git clone https://github.com/decisionintelligence/WorldTS.git upstream/WorldTS
git -C upstream/WorldTS checkout 269426ea6217d7152a244da0e209d922acd46d4f
python -m pytest -q
```

**Known compatibility risk:** upstream code may import other TFB components.
Resolve any missing imports by installing minimal compatible dependencies while
recording `pip freeze`. Do not silently change the authors' model. The native
training adapter is *source-interface checked* but **not verified end-to-end
against the upstream package in this CPU-only execution environment**.

## 2. NESO official resources

```bash
python -m neso_online.download --resources \
  forecast_2025 forecast_2026_jan_jun forecast_2026_jun_dec \
  demand_2025 demand_2026 --out data/raw
```

Downloads raw CSVs from the official CKAN resource IDs and produces SHA-256 sidecars.
Full forecast archives contain millions of **issue/target** rows and can be large.
Check each downloaded file and any duplicates in overlapping 2026 snapshots. If
an API dump route is unavailable, use the official data portal's Download CSV
links and place files under the same names (preserve provenance).

Official forecast datasets:
- [2025 embedded wind/solar forecast archive](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts/embedded_solar_and_wind_forecast_archive_2025)
- [2026 embedded wind/solar archive](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts/embedded_solar_and_wind_forecast_archive_2026-0)
- [2026 Jan–Jun forecast archive](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts/embedded_solar_and_wind_forecast_archive_2026_jan-_jun)
- [2025 actual demand](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2025)
- [2026 actual demand](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2026)

## 3. Prepare training/validation/test windows

```bash
python -m neso_online.prepare \
  --forecast data/raw/forecast_2025.csv \
             data/raw/forecast_2026_jan_jun.csv \
             data/raw/forecast_2026_jun_dec.csv \
  --demand data/raw/demand_2025.csv data/raw/demand_2026.csv \
  --lookback 336 --horizon 48 --revision-hours 6 \
  --out outputs/dataset
```

Each example contains:

| Key | Shape / meaning |
| --- | --- |
| `history` | `[L,3]` original known ND + historical wind/solar forecast as of each past interval |
| `cov_a` | `[H,2]` issue-A wind/solar forecast for same future time slots |
| `cov_b` | `[H,2]` issue-B forecast for still-unfinished slots; unchanged prefix from A |
| `target` | `[H,1]` finally measured ND, **not** visible to inference |
| `update_mask` | `[H]` true if settlement interval end is **after issue B** |
| `issue_a`, `issue_b` | model-produced prediction timestamps, UTC (availability latency assumed zero) |
| `first_target_end`, `last_target_end` | UTC timestamps for no-overlap temporal splitting |

The historical target is held **fixed at A** for both paths: only the forecast
vintage is revised. An update to **new target observations** is a different task.
Samples are split by *entire future-label support*: end before 2026-01-01 = train,
2026 Q1 = validation, 2026-04-01 through 2026-08-31 = test.

If there are zero windows or time mismatches, **stop and audit** rather than
filling unavailable covariates from future issues or future labels.

### Critical availability caveat

NESO Historic Demand is described as **21 days in arrears** and may be
retroactively corrected. It does not contain a per-observation first-publication
field. Thus `history` treats completed ND as if known at its settlement period
end — an **unverified retrospective assumption**. This V1 is useful for testing
forecast-vintage mechanics but does **not** yet establish a strictly causal
real-time demand forecasting system. For a live-causal paper experiment, replace
historical ND with a time-stamped operationally available demand feed or introduce
verified release-time logic. `Forecast_Datetime` is a **production** timestamp,
not necessarily the first public availability time; a release-delay audit is also
required for strict live claims.

## 4. Fit native WorldTS and evaluate on validation only

```bash
python -m neso_online.native_worldts train \
  --data outputs/dataset --out outputs/model \
  --worldts-source upstream/WorldTS/DAG-bench \
  --config configs/worldts_neso.json \
  --device cuda --batch 16
```

`stage1_best.pt` selects the smallest held-out Stage-1 latent loss and
`stage2_best.pt` selects decoded **validation MAE**. A train-only linear
observation-correction baseline is fitted after Stage 2. The model uses the
upstream `WorldTSModel` and native `LatentStateLoss`; only the new vintage-specific
sampler and training driver are ours. The config is a **NESO adaptation**, not
one of the authors' original dataset-specific configs.

The initial run can omit any updater: compare `old` versus `recompute` and
`output_correction` on validation. If the revision headroom justifies it, train
one lightweight **non-looped** latent correction module:

```bash
python -m neso_online.native_worldts train_updater \
  --data outputs/dataset --out outputs/model \
  --worldts-source upstream/WorldTS/DAG-bench \
  --config configs/worldts_neso.json \
  --device cuda --batch 16
```

**After freezing the protocol and model selection**, run the held-out test once:

```bash
python -m neso_online.native_worldts replay \
  --data outputs/dataset --out outputs/model \
  --worldts-source upstream/WorldTS/DAG-bench \
  --config configs/worldts_neso.json \
  --device cuda --batch 16
```

Results: `outputs/model/train_validation.json`,
`outputs/model/train_updater_validation.json`,
`outputs/model/test_replay.json`. Replay scores only target timestamps **after
B**, so comparisons refer to *exactly the same final future outcomes*. The full
recompute model uses the **same fixed A-history** as the updater, avoiding a
mismatch in available information.

## 5. Smoke test without NESO data or GPU

```bash
python scripts/make_synthetic_fixture.py --out fixtures/synthetic
python -m neso_online.prepare \
   --forecast fixtures/synthetic/forecast_synthetic.csv \
   --demand fixtures/synthetic/demand_synthetic.csv \
   --out outputs/synthetic_dataset
python -m pytest -q
```

Only the data pipeline and generic training driver are tested on synthetic data.
**This does not run official WorldTS** and does not produce research evidence.

## 6. First-research-round GO/NO-GO

Do not automatically proceed to more loops or complex World Model losses. First:

- Verify that `old` vs `recompute` produces a non-trivial, statistically
  meaningful revision headroom on **validation** (report both improvements and
  regressions by window; use non-overlapping time-block analysis).
- Check whether the train-only **output-space linear correction** already
  delivers equivalent accuracy at lower cost.
- Compare one-step latent updating against **both** output correction and full
  recomputation at matched as-of information. For compute, report *incremental*
  cost at B; cached A computation is shared.
- Reject the proposed updater if it cannot improve the accuracy/compute tradeoff.
- Never use test-label information to choose model weights, update depth, or
  acceptable revisions.

**Status:** The runnable prototype and tests are complete; actual NESO downloads,
model convergence, and numerical performance are pending execution on a networked
GPU server. Please do not cite this V1 as a finished empirical finding.
