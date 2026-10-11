# Baseline reproduction — completed

24/24 configurations completed and independently verified: {'MATCHED': 22, 'CLOSE': 2}. 21 fresh training runs and 3 strictly verified prior DAG runs. WorldTS uses pinned official author source. No Loop was added.

- [Full report](FINAL_REPORT.md)
- [All results](results/all_baselines.csv)
- [Float64 metric and raw-target verification](results/metric_verification.json)
- [Reproduction commands](REPRODUCE.md)
- [Issues and retained failures](ISSUE_LOG.md)
- [Server artifact SHA256 manifest](reports/artifact_manifest.json)

Raw data, trained checkpoints and predictions are retained at `D:\baseline_reproduction_seed2021_v1` on the server. Electricity+DLinear H336/H720 remain CLOSE, with unresolved differences; PJM WorldTS H24 initial training peak memory was not persisted and is explicitly unavailable.

The earlier interruption report remains archived as historical evidence and is superseded by this delivery.
