# BNCI2014001 Online Adaptation Stage 0

This is a one-subject seed-0 gate only (A01 S1 -> chronological A01 S2). It is not a 9-subject benchmark result.

## Integrity checks

- Shared cache: `(288, 22, 1000)` (expected 288 x 22 x 1000); S1/S2 retain 144 left/right trials each.
- Target labels are separate from the runner interface and are scored only after a full stream is emitted.
- Strict EA consumes x_1...x_t only; paper EA is separately labelled non-causal/full-session.
- Every runner uses batch 1 and `predict()` followed by `update()`; its state resets for S2.
- BTTA-DG is imported from the official source snapshot; its update order is preserved verbatim.

## Metrics

| method | EA | accuracy | balanced accuracy | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| source_only_sinc | paper | 0.8056 | 0.8056 | 0.8009 | 0.6394 | 0.2720 |
| btta_dg_official | paper | 0.8056 | 0.8056 | 0.8009 | 0.6394 | 0.2720 |
| neuroloop_static_k1 | paper | 0.7014 | 0.7014 | 0.7014 | 0.5763 | 0.1288 |
| neuroloop_static_k4 | paper | 0.6944 | 0.6944 | 0.6935 | 0.5724 | 0.0463 |
| neuroloop_shared_k1 | paper | 0.5417 | 0.5417 | 0.4198 | 0.7472 | 0.2075 |
| neuroloop_shared_k2 | paper | 0.7083 | 0.7083 | 0.7014 | 0.6508 | 0.1584 |
| neuroloop_shared_k4 | paper | 0.6458 | 0.6458 | 0.5950 | 0.6436 | 0.0283 |
| neuroloop_shared_k8 | paper | 0.7014 | 0.7014 | 0.7013 | 0.5514 | 0.0390 |
| neuroloop_untied_k4 | paper | 0.7292 | 0.7292 | 0.7290 | 0.5251 | 0.0826 |
| source_only_sinc | strict | 0.8056 | 0.8056 | 0.8009 | 0.6394 | 0.2720 |
| btta_dg_official | strict | 0.8056 | 0.8056 | 0.8009 | 0.6394 | 0.2720 |
| neuroloop_static_k1 | strict | 0.7014 | 0.7014 | 0.7014 | 0.5763 | 0.1288 |
| neuroloop_static_k4 | strict | 0.6944 | 0.6944 | 0.6935 | 0.5724 | 0.0463 |
| neuroloop_shared_k1 | strict | 0.5417 | 0.5417 | 0.4198 | 0.7472 | 0.2075 |
| neuroloop_shared_k2 | strict | 0.7083 | 0.7083 | 0.7014 | 0.6508 | 0.1584 |
| neuroloop_shared_k4 | strict | 0.6458 | 0.6458 | 0.5950 | 0.6436 | 0.0283 |
| neuroloop_shared_k8 | strict | 0.7014 | 0.7014 | 0.7013 | 0.5514 | 0.0390 |
| neuroloop_untied_k4 | strict | 0.7292 | 0.7292 | 0.7290 | 0.5251 | 0.0826 |

The published BTTA mean cannot be used as a quantitative replication check at n=1; the full 9-subject gate remains required before that comparison.
