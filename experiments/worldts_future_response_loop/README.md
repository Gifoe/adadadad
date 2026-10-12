# WorldTS future-response Loop

Final decision: **INCONCLUSIVE**; conclusion category **E**. Completed 26 real-task training executions,12 separately labeled synthetic fits, and fresh inference audits of four original checkpoints.

| dataset   |   seed | nonrecurrent         | shared_loop          |   relative_mse_gain |   relative_mae_gain |   mse_difference_loop_minus_control |
|:----------|-------:|:---------------------|:---------------------|--------------------:|--------------------:|------------------------------------:|
| NP        |   2021 | NP_H360_C4_K1_s2021  | NP_H360_C5_K2_s2021  |         0.0214042   |           0.0178789 |                        -0.0105632   |
| PJM       |   2021 | PJM_H360_B2_K1_s2021 | PJM_H360_C5_K4_s2021 |        -0.000827757 |          -0.0112703 |                         0.000100471 |

Mean two-task MSE gain over the validation-selected nonrecurrent controls: 1.03%. Mean same-checkpoint long-task K4 versus K1 gain: 0.57%. Parameter/latency comparisons all within the frozen tolerances: True.

- [Full research report](FINAL_RESEARCH_REPORT.md)
- [Next research decision and seven explicit answers](NEXT_RESEARCH_DECISION.md)
- [All completed and NOT_RUN configurations](EXPERIMENT_MATRIX.csv)
- [Reproduction instructions](REPRODUCE.md)
- [Paired statistics and time-block bootstrap](STATISTICAL_ANALYSIS.md)
- [Failures and negative results](FAILURE_AND_NEGATIVE_RESULTS.md)
- [Source/integrity/resume limitations](reports/IMPLEMENTATION_AND_RESUME_AUDIT.md)
- [Artifact paths and SHA256 hashes](ARTIFACT_MANIFEST.json)

The real-task family starts every method and continuation seed from the same task-specific verified seed2021 checkpoint. An improvement over continued B0 can remain worse than the original checkpoint; both references are reported. The global one-shot mixer already reaches the full future interval, so extra K cannot be credited with uniquely opening distant paths. Raw predictions, Jacobians, data and weights remain on the server at `D:\worldts_future_response_loop`; Git contains reproducible code, configs, raw score tables, logs, figures and hashes. Original baseline artifacts are read-only.
