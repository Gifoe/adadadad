# NESO Online WorldTS real execution

**Decision: NO-GO for a new persistent latent-state updater under this fixed V1 protocol.** The real revisions alter WorldTS forecasts and modestly improve mean accuracy, but a six-coefficient linear correction captures 84.29% of the Validation MAE benefit. Full recomputation is already cheap on the measured RTX 5090: single-window host-output p95 4.872 ms, incremental peak 16.015 MiB. Validation did not justify training the supplied updater. The locked eligible Test gives B1 MAE 1183.252 MW and B3 MAE 1183.327 MW, a 0.075 MW gap. These measurements do not establish a useful accuracy or compute bottleneck for a new mechanism.

Execution status: actual data, model training, Validation, conditional decision, locked Test and reporting completed for available eligible windows. **The requested complete April-August coverage is not achieved:** the strict source-vintage definition yields only 303 Test windows and last target July 19, 2026. No August window qualifies. This is a source/protocol coverage limitation. Conclusions apply to this eligible cohort, one seed and this small supplied native configuration.

## Real data and temporal audit

Forecast archives:9,658,114raw rows,9,643,219retained future finite rows,15,353production issue timestamps.2025CSV contains5,369,698rows;2026full-year resource4,288,416rows. Identical cross-file duplicate issue-target rows:0; conflicting duplicate values:0 (the loader would fail on any). Source SHA256, actual field coverage, resource404 evidence and raw missingness are retained in `DATA_AUDIT.md` and JSON.48raw2026rows lack wind/solar;1268ambiguous timestamp rows were excluded without guessing. No synthetic samples or invented vintages enter the experiment.

Actual historicND:30,238half-hours,2025Jan1 through2026Sep22.2025 official outturn schema has no A/F column;2026 uses actual rows only. Genuine windows:TRAIN1406, Validation354, Test303. Scored unfinished slots:VAL12,494, TEST10,690. All windows have `[336,3]`history, `[48,2]`A/B covariates and `[48,1]`target; finite values, strict period-end>Bmask, identical A/Btarget times, unchanged already-ended prefix, fixed-Ahistory and disjoint future-target support all verified. Real DST demand days verify46/48/50-period behavior; see raw day counts.

The requested obsolete Jan-JunID returns404; the current official resource is named full-year2026 and actually covers issues Jan1-Oct13. It supplies the complete Validation quarter. From June onward many production issues omit the first partly elapsed settlement interval: e.g. A2026July15 00:55:17, required first end01:00 but source first end01:30. Missing immediate covariates cannot be filled from a later vintage. August has124selected source A candidates and0complete immediate24h A horizons. Monthly source evidence and examples are in `outputs/coverage_gap_audit.json`; no Test target values were needed to establish this exclusion.

This is a **retrospective forecast-vintage replay**. [Official Historic Demand documentation](https://www.neso.energy/data-portal/historic-demand-data) states publication21days in arrears and possible retrospective demand edits. Historic ND is therefore not proved observable instantaneously at A. Forecast production time is also not proved public-release time. Neither live operational validity nor causal benefit is claimed.

## Actual training and checkpoint evidence

Pinned official source:`269426ea6217d7152a244da0e209d922acd46d4f`. The495473parameter numerical model and official latent loss were actually imported and trained. Stage1 executed11epochs/968steps, selected6; Stage2 executed20epochs/1760steps, selected20. Stage2 preserves E/C/P exactly; Stage1 keeps D frozen. All loss values are finite; real smoke gradients, selected latent variation, target-label foil, strict restore and full saved Validation prediction reproduction passed. `TRAINING_REPORT.md` includes measured~42.7s stage time, GPU allocations, exact objectives and selected/raw checkpoint evidence. These are reproduced NESO results, not paper NP/PJM numbers. Decoder selection still improved at the20epoch budget boundary; the budget was not changed after observing Validation/Test.

## Matched Validation accuracy

| Method | MAE MW | MSE MW² | Normalized MAE | Normalized MSE | MAE gain vs B0 | Improved windows | Worsened windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | 1439.041334 | 4142646.204109 | 0.23348124 | 0.10905241 | 0.0000% | 0.00% | 0.00% |
| recompute | 1422.043469 | 4019054.551715 | 0.23072337 | 0.10579894 | 1.1812% | 55.65% | 44.35% |
| output_correction | 1424.714552 | 4025264.930988 | 0.23115675 | 0.10596243 | 0.9956% | 53.39% | 46.61% |
| seasonal_naive | 2466.996638 | 11197206.358892 | 0.40026468 | 0.29475902 | -71.4333% | 17.23% | 82.77% |

B1 changes the scored prediction by>0.001MW on100% of windows; mean absolute prediction change145.46MW. It improves55.65% and worsens44.35% of windows. The aggregate MAE gain is1.1812%; B3 gain0.9956% and B3-B1 relative gap0.1878%. The predeclared>1%unexplained-headroom gate fails; no updater is trained. All four gate outcomes and decision timing are saved before Test.

## Locked eligible Test accuracy

| Method | MAE MW | MSE MW² | Normalized MAE | Normalized MSE | MAE gain vs B0 | Improved windows | Worsened windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | 1187.796215 | 2754709.104375 | 0.19271728 | 0.07251589 | 0.0000% | 0.00% | 0.00% |
| recompute | 1183.251635 | 2719237.763963 | 0.19197993 | 0.07158213 | 0.3826% | 52.48% | 47.52% |
| output_correction | 1183.326967 | 2702525.262616 | 0.19199215 | 0.07114218 | 0.3763% | 51.82% | 48.18% |
| seasonal_naive | 2179.127785 | 9303077.401781 | 0.35355861 | 0.24489734 | -83.4597% | 12.21% | 87.79% |

B1 changes100% of windows; mean absolute prediction revision91.32MW. Improvement52.48%, worsening47.52%. Its mean MAE benefit over B0 is0.3826%, not a uniform improvement. B3 gives nearly identical MAE and slightly lower MSE. Test was evaluated once after the SHA lock; no coefficients, model, bins or thresholds were selected from these results. Complete predictions, targets, timestamps and masks are retained server-side with SHA hashes.

## Dependence-aware checks

Circular moving calendar-day block bootstrap,2000draws,seed2021;1/7/14day sensitivity. Daily weighted error sums preserve the original repeated-window scoring; no IID-window t-test is used. Adjacent24h windows overlap, so intervals are descriptive conditional on this fitted seed and eligible cohort.7-day blocks are primary; all lengths are reported.

| Split | Candidate-reference | Block days | Delta MAE MW | Descriptive95% interval MW | Disjoint blocks favor candidate |
|---|---|---:|---:|---|---:|
| Validation | recompute_vs_old | 1 | -16.997866 | [-28.798075, -5.291546] | 62.92% |
| Validation | recompute_vs_old | 7 | -16.997866 | [-34.232207, -0.009276] | 69.23% |
| Validation | recompute_vs_old | 14 | -16.997866 | [-33.603925, -0.048860] | 85.71% |
| Validation | output_correction_vs_old | 1 | -14.326782 | [-24.291557, -4.784745] | 57.30% |
| Validation | output_correction_vs_old | 7 | -14.326782 | [-29.305610, -0.259127] | 61.54% |
| Validation | output_correction_vs_old | 14 | -14.326782 | [-28.577429, 0.525198] | 57.14% |
| Validation | output_correction_vs_recompute | 1 | 2.671083 | [-3.786065, 9.092656] | 48.31% |
| Validation | output_correction_vs_recompute | 7 | 2.671083 | [-4.607094, 10.262589] | 53.85% |
| Validation | output_correction_vs_recompute | 14 | 2.671083 | [-4.463464, 10.320367] | 42.86% |
| Test | recompute_vs_old | 1 | -4.544580 | [-13.939332, 3.689859] | 58.02% |
| Test | recompute_vs_old | 7 | -4.544580 | [-11.664158, 2.650567] | 43.75% |
| Test | recompute_vs_old | 14 | -4.544580 | [-10.729216, 1.709769] | 50.00% |
| Test | output_correction_vs_old | 1 | -4.469248 | [-14.679664, 5.347676] | 50.62% |
| Test | output_correction_vs_old | 7 | -4.469248 | [-13.742885, 6.031113] | 50.00% |
| Test | output_correction_vs_old | 14 | -4.469248 | [-12.635992, 4.745320] | 50.00% |
| Test | output_correction_vs_recompute | 1 | 0.075333 | [-3.968787, 4.152401] | 45.68% |
| Test | output_correction_vs_recompute | 7 | 0.075333 | [-4.999435, 5.554629] | 56.25% |
| Test | output_correction_vs_recompute | 14 | 0.075333 | [-4.837242, 5.906541] | 50.00% |

The Validation7-day B1 interval barely excludes zero; do not interpret its upper endpoint near zero as strong or multi-seed evidence. The full Test comparison is reporting only and never changes the frozen decision.

## Incremental revision-time compute

| Method | Batch | Scope | Mean ms/batch | p95 ms/batch | Peak incremental MiB |
|---|---:|---|---:|---:|---:|
| old | 1 | host_output | 0.011183 | 0.011325 | 0.000000 |
| recompute | 1 | gpu_ready | 4.627318 | 4.821955 | 16.014160 |
| recompute | 1 | host_output | 4.739708 | 4.871655 | 16.014648 |
| seasonal_naive | 1 | host_output | 0.010928 | 0.011020 | 0.000000 |
| output_correction | 1 | host_output | 0.056709 | 0.059505 | 0.000000 |
| old | 16 | host_output | 0.010830 | 0.011120 | 0.000000 |
| recompute | 16 | gpu_ready | 5.323874 | 5.593270 | 262.833984 |
| recompute | 16 | host_output | 5.444043 | 5.652315 | 262.839844 |
| seasonal_naive | 16 | host_output | 0.010818 | 0.011025 | 0.000000 |
| output_correction | 16 | host_output | 0.101274 | 0.107640 | 0.000000 |

Twenty warmups and100timed calls per configuration, batch1/16; CUDA synchronized. A-time history, native latent/statistics and old prediction are cached identically. Host-output B1 includes new-Bnormalization/H2D, complete native model/Decoder, D2H and MW scaling. GPU-ready excludes those transfers. B0/B2copy cached host results; B3 uses only CPUrevision arithmetic and the six TRAIN-fitted coefficients. CPU timings include the common timer/synchronization overhead. No future ND values enter inference. A-time inference, source/network latency, public availability and long-lived cache orchestration are outside this cost scope. GPU allocation columns use live Torch tensors, not total board occupancy.

B3 single-window p950.059505ms. B1 p954.871655ms is far below the100ms cheap-recompute threshold predeclared before measurement, for a6-hour update schedule. A new updater would need an accuracy advantage or a materially different deployment constraint to justify itself; no such advantage was measured here.

## Both improvement and degradation cases

| Split | Case | A UTC | B UTC | B1-B0 MAE MW | Revision RMS MW |
|---|---|---|---|---:|---:|
| Validation | largest_improvements | 2026-01-06 00:12:00+00:00 | 2026-01-06 06:12:00+00:00 | -769.170953 | 496.366996 |
| Validation | largest_degradations | 2026-01-17 00:12:09+00:00 | 2026-01-17 07:12:09+00:00 | 395.854779 | 258.951563 |
| Test | largest_improvements | 2026-05-11 00:12:13+00:00 | 2026-05-11 07:12:09+00:00 | -365.082261 | 523.641323 |
| Test | largest_degradations | 2026-05-17 00:12:13+00:00 | 2026-05-17 06:12:13+00:00 | 250.422743 | 297.662794 |

These are deterministic largest-effect illustrations, not manually filtered examples or training choices. `outputs/val_analysis.json` / `test_analysis.json` retain five examples of each type per method. Per-window CSVs, all lead/Train-quartile strata and case plots provide the full distribution.

## Decision and remaining scope

NO-GO for persistent latent updating in this V1 setup. Prefer the existing native recomputation or the TRAIN-calibrated linear correction according to the measured accuracy/compute tradeoff. No claim that an untrained updater is mathematically incapable of improving; this task deliberately requires evidence of headroom before fitting it.

Limits that affect interpretation: incomplete and nonrandom post-June eligibility; single seed and one small supplied configuration; possibly revised historic ND unavailable at production time; production versus release timestamps; fixed UTC rather than local-DST seasonal lag; Stage2 budget boundary; descriptive dependent-block intervals rather than independent-replication evidence. A future study needing complete summer/live coverage must first obtain defensible current-period vintages and publication logs, then register its changed protocol separately. Existing V1 results remain frozen.

Required code, reports, metrics, logs, source metadata, patch and figures are archived. Large official CSVs, prepared NPZs, prediction NPZs and native PTcheckpoints stay under `D:\neso_worldts_online_v1`; sizes and exactSHA256 are in `outputs/artifact_manifest.json`. `REPRODUCE.md` contains the executed protocol and recovery steps.
