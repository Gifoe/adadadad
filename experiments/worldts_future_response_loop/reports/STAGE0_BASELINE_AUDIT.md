# Stage 0 — Baseline audit

## Research Question

Can the current four official checkpoints support controlled comparisons?

## Pre-experiment Hypothesis

All raw targets, scaler parameters, rolling origins and model predictions match the verified historical artifacts.

## Experiment Design

Read-only baseline artifacts; fresh GPU inference on every official test origin; fixed source 269426ea6217d7152a244da0e209d922acd46d4f.

## Actual Execution

Four configurations freshly reevaluated; no historical baseline file or checkpoint modified.

## Raw Results

| dataset   |   horizon |      mse |      mae |   max_prediction_abs_diff |   windows |
|:----------|----------:|---------:|---------:|--------------------------:|----------:|
| NP        |       360 | 0.430395 | 0.416201 |                         0 |     10125 |
| PJM       |       360 | 0.115082 | 0.208422 |                         0 |     10125 |
| NP        |        24 | 0.167258 | 0.213351 |                         0 |     10461 |
| PJM       |        24 | 0.057516 | 0.142788 |                         0 |     10461 |

## Interpretation

Checkpoint reuse verified by source/data/checkpoint/prediction hashes, exact raw targets and all test origins. This does not establish bitwise reproducibility of retraining.

## Scientific Contribution

This is an audit prerequisite, not a methodological contribution.

## Limitations and Alternative Explanations

Single pretrained seed; efficient GPU determinism may not reproduce training trajectories. Future covariate vintages remain unknown. Original PJM24 training peak was unavailable.

## Decision

Proceed with isolated, matched two-stage continued-training controls; keep historical checkpoint results separate from newly trained controls.

## Reproduction

Activate the verified baseline private venv and execute python scripts/stage0.py. Detailed verified rows: results/stage0_baseline_verification.csv.

