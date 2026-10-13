# NESO Online WorldTS V1 — executed artifact

Real official NESO data and pinned native WorldTS were run on RTX5090. The research decision is **NO-GO for a persistent latent-state updater**; the validation-only gate correctly skipped updater training.

Read [FINAL_REPORT.md](FINAL_REPORT.md), [DATA_AUDIT.md](DATA_AUDIT.md), [TRAINING_REPORT.md](TRAINING_REPORT.md), [REPRODUCE.md](REPRODUCE.md) and the four result CSVs. Execution is complete for eligible data, with TRAIN1406, VAL354 and TEST303windows. The requested complete April-August Test is not available under the original immediate-horizon definition; eligible Test ends July19 and August contributes no window. This limitation is explicit throughout the report.

The immutable supplied engineering ZIP is retained server-side. Its original README, STATUS and execution instructions are copied to `outputs/original_docs`; synthetic/dummy tests in that package are separate from actual native training. Native checkpoints, prediction NPZs and raw CSVs remain server-side, referenced by `outputs/artifact_manifest.json`. Compact source, evidence, CSVs and figures are published.
