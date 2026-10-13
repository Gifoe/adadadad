# Codex Task: NESO Online WorldTS — Real Data and Model Execution

## 0. Objective

You are working on an RTX 5090 research server.

Your task is to **execute the existing `neso_worldts_online_v1` project using real NESO data and the official WorldTS implementation**, not merely review code, summarize papers, or generate an experiment plan.

Our research question is:

**Does WorldTS exhibit a correctable forecasting or computational bottleneck when future covariate forecasts are revised as new information arrives?**

We want to determine whether persistent latent-state updating is worth researching.

Do not assume that an online updater will outperform the original model. The first priority is establishing a reliable benchmark and identifying actual forecasting limitations.

## 1. Existing project and upstream code

The provided engineering package contains:

- `README.md`
- `CODEX_EXECUTION.md`
- `STATUS.md`
- `configs/worldts_neso.json`
- `neso_online/download.py`
- `neso_online/prepare.py`
- `neso_online/native_worldts.py`
- `tests/`

Find the extracted `neso_worldts_online_v1` directory in the workspace.

Read the documentation and source files before execution. If necessary, repair compatibility or implementation problems, but preserve the original experimental definition.

Use the actual official WorldTS code:

https://github.com/decisionintelligence/WorldTS

Pin commit:

`269426ea6217d7152a244da0e209d922acd46d4f`

The model must import the upstream `WorldTSModel` and `LatentStateLoss`.

Do not replace these with a custom Transformer.

Do not introduce Loop, Flow Matching, or new representation objectives in this round.

## 2. Environment preparation

Hardware target: NVIDIA RTX 5090, 32 GB VRAM.

Inspect:

- Operating system, Python, PyTorch, CUDA and GPU driver
- Free GPU memory, RAM and disk space
- Network availability
- Existing Python environments and packages

Prefer an existing CUDA-compatible PyTorch environment, such as PyTorch 2.8+cu128, if it has already been validated on the RTX 5090.

Do not install the original DAG requirements in a way that downgrades CUDA support.

From the project directory, execute:

```bash
python -m pip install -r requirements.txt
python -m pytest -q

git clone https://github.com/decisionintelligence/WorldTS.git upstream/WorldTS
git -C upstream/WorldTS checkout 269426ea6217d7152a244da0e209d922acd46d4f
```

Adapt commands to the actual shell. If the upstream repository already exists, verify its commit rather than overwriting it.

Install only the additional dependencies required by the actual imports.

Before training, conduct a real forward/backward smoke test using the official WorldTS numerical model.

Record complete environment information and any compatibility patches.

## 3. Download real NESO 2025–2026 data

Use the implemented downloader:

```bash
python -m neso_online.download --resources forecast_2025 forecast_2026_jan_jun forecast_2026_jun_dec demand_2025 demand_2026 --out data/raw
```

The official data sources are NESO's embedded wind/solar forecast archives and Historic Demand datasets.

The main fields are:

Forecast archives:
- `Forecast_Datetime`
- `DATE_GMT`
- `TIME_GMT`
- `SETTLEMENT_DATE`
- `SETTLEMENT_PERIOD`
- `EMBEDDED_WIND_FORECAST`
- `EMBEDDED_SOLAR_FORECAST`

Actual observations:
- `SETTLEMENT_DATE`
- `SETTLEMENT_PERIOD`
- `ND`
- `FORECAST_ACTUAL_INDICATOR`

Validate actual downloaded files, resource IDs, SHA256, number of records, timestamps and data coverage.

Check whether the 2026 archives overlap. Deduplicate identical issue-time/target-time records, but stop and investigate conflicting records.

Verify UK daylight-saving transitions, including 46/48/50 settlement periods.

If the API download route fails, use the official NESO CSV download endpoint. Do not substitute synthetic data or unrelated datasets.

## 4. Prepare the vintage-aware benchmark

Use:

```bash
python -m neso_online.prepare --forecast data/raw/forecast_2025.csv data/raw/forecast_2026_jan_jun.csv data/raw/forecast_2026_jun_dec.csv --demand data/raw/demand_2025.csv data/raw/demand_2026.csv --lookback 336 --horizon 48 --revision-hours 6 --out outputs/dataset
```

Initial configuration:

| Parameter | Value |
|---|---|
| Target | National Demand (`ND`) |
| Covariates | Embedded Wind Forecast, Embedded Solar Forecast |
| Data frequency | 30 minutes |
| Lookback | 336 points (7 days) |
| Prediction horizon | 48 points (24 hours) |
| Revision interval | 6 hours |
| Seed | 2021 |

Construct samples containing:

```python
history       # [B, 336, 3]
cov_a         # [B, 48, 2]
cov_b         # [B, 48, 2]
target        # [B, 48, 1]
update_mask   # [B, 48]
issue_a
issue_b
```

A and B must refer to exactly the same target timestamps.

At A, use the available historical input and the earlier forecast version. At B, replace only future covariates whose corresponding target period has not finished.

**For V1, hold the historical ND input fixed at A**, to isolate the contribution of revised future covariates.

Use chronological splits:

- Train: 2025 target periods
- Validation: January–March 2026
- Test: April–August 2026

Drop windows crossing split boundaries. Never let the same future target timestamp appear in more than one split.

Fit all normalization statistics on training data only.

Produce a data audit covering sample counts, forecast-version counts, revision magnitudes, missing values, target alignment and temporal leakage.

Important: the historic ND publication time is not verified to be instantaneous, and forecast production time may differ from public release time. Until those facts are established, label this as a **retrospective forecast-vintage replay**, not a fully verified live operational benchmark.

## 5. Train original WorldTS

Run:

```bash
python -m neso_online.native_worldts train --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021 --epochs1 20 --epochs2 20 --patience 5
```

Use the official two-stage architecture.

Stage 1:
- Train the latent Encoder, covariate Encoder and State Predictor.
- Use the official latent matching, cosine and regularization objectives.
- Select checkpoint using validation latent loss.

Stage 2:
- Freeze Stage 1 modules.
- Train the original Decoder.
- Use predicted-state decoding loss and target-state reconstruction supervision.
- Select checkpoint using validation decoded MAE.

Preserve the official network architecture. Treat the supplied configuration as a NESO-specific adaptation, not as the author's NP/PJM configuration.

Save checkpoints, learning curves, best epoch, training time, GPU memory and full configurations.

Investigate nonfinite losses, state collapse, missing gradients, inconsistent checkpoint reloads or unexpected prediction shapes before running final evaluation.

## 6. Establish online revision baselines

Evaluate the following on the validation split.

| ID | Model | Purpose |
|---|---|---|
| B0 | Old Forecast | Keep the forecast generated at A |
| B1 | Full WorldTS Recompute | Use revised covariates available at B, with identical fixed-A historical information |
| B2 | Seasonal Naive | Simple time-series forecasting baseline |
| B3 | Output-Space Correction | Apply train-only calibrated linear correction to the original forecast |

B1 is the most important control. It represents recomputation with revised covariates, not a model receiving unverified new real-time ND observations.

Compare the same unfinished target intervals after B.

Report:
- MSE and MAE
- Relative improvement versus B0
- Fraction of windows improved and worsened
- Improvement by forecast horizon
- Performance stratified by covariate revision magnitude
- Inference latency and GPU memory
- Incremental compute cost at B

Audit the evaluation code for unit consistency: separate errors in MW/MW² from normalized MSE/MAE. Verify seasonal-baseline time alignment.

Use dependence-aware analysis such as contiguous time-block bootstrap because overlapping windows are not independent.

**Do not evaluate the final test split yet.**

## 7. Determine whether updating creates real headroom

Using validation only, answer:

1. How often do revised wind/solar forecasts change WorldTS predictions?
2. How often does full recomputation improve or worsen the forecast?
3. Are the changes statistically meaningful across distinct time blocks?
4. Does a simple output correction explain most of the benefit?
5. Does full recomputation impose a meaningful incremental computation cost?

Do not automatically train a more complicated network.

If the old and recomputed predictions are nearly identical, or simple output correction is already sufficient, record a negative finding.

## 8. Train a latent-state updater only if justified

If validation demonstrates nontrivial update headroom not already explained by simple baselines, execute:

```bash
python -m neso_online.native_worldts train_updater --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021 --updater-epochs 20 --patience 5
```

Freeze WorldTS.

Train only the supplied lightweight `LatentRevisionUpdater`.

Inputs are the cached latent trajectory from A, A/B covariates and the update mask.

The updater must not access ground-truth future targets during inference.

Compare it fairly against output-space correction and full recomputation at B, including both accuracy and incremental computation.

This is a single-step updater, not a Looped Transformer.

Do not add additional architectures if it fails.

## 9. Freeze models and perform final test

After validation-based decisions are finalized, execute:

```bash
python -m neso_online.native_worldts replay --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021
```

Run test evaluation only after freezing configurations and model selection.

Preserve all test predictions, corresponding target timestamps, update masks and metric calculation settings.

Do not use test results to select updater depth, correction coefficients, checkpoints or additional configurations.

## 10. Required outputs

Create the following files in an independent experiment directory:

- `DATA_AUDIT.md`
- `ENVIRONMENT.md`
- `ISSUE_LOG.md`
- `CONFIG_PROVENANCE.md`
- `TRAINING_REPORT.md`
- `VALIDATION_RESULTS.csv`
- `TEST_RESULTS.csv`
- `REVISION_ANALYSIS.csv`
- `COMPUTE_RESULTS.csv`
- `REPRODUCE.md`
- `FINAL_REPORT.md`

The final report must explicitly provide:

1. Real NESO sample counts, forecast-version coverage and timestamp validation.
2. Evidence that actual WorldTS Stage 1 and Stage 2 were trained.
3. All measured forecasting metrics and computational costs.
4. Cases where newer covariate versions improve or worsen forecasts.
5. Whether the proposed update mechanism offers any advantage over direct recomputation or simple output correction.
6. Remaining problems involving information availability, timestamp handling, data coverage or model implementation.
7. A clear **GO / NO-GO** recommendation.

Do not confuse synthetic tests with real NESO experiments or paper-reported numbers with reproduced results.

## 11. Execution requirements

Execute every feasible step, repairing errors and rerunning tests as needed. Preserve the original error logs and all meaningful implementation changes.

If memory is insufficient, reduce micro-batch size while keeping model architecture and effective training protocol comparable.

If data downloads are blocked, investigate the official alternative endpoints and document the blocker precisely.

If data coverage is insufficient, investigate the actual source records and report the limitation. Never manufacture forecasting vintages or fill missing information using hindsight.

Do not stop after creating scripts or completing smoke tests. Proceed to real training and evaluation when resources allow.

## Final research decision

The experiment must answer:

**Does WorldTS have a correctable forecasting or compute-efficiency limitation when receiving real revisions to future covariates?**

A positive result requires improvement over strong, matched baselines.

If WorldTS recomputation already performs well and is sufficiently cheap, or if simple output correction captures the remaining benefit, return **NO-GO** for a new persistent latent-state updater.

Do not force a positive result.