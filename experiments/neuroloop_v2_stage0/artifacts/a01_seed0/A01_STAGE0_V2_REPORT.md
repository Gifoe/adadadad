# A01 NeuroLoop Stage-0 v2 report

This report is A01 / S1->S2 / seed 0 only. It is a falsification gate, not a multi-subject result.

## Strict-causal same-checkpoint trajectory

| loop | Accuracy | BA | Macro-F1 | NLL | ECE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7986 | 0.7986 | 0.7974 | 0.5325 | 0.1662 |
| 2 | 0.7708 | 0.7708 | 0.7708 | 0.4640 | 0.0959 |
| 3 | 0.7708 | 0.7708 | 0.7708 | 0.4475 | 0.0701 |
| 4 | 0.7639 | 0.7639 | 0.7638 | 0.4627 | 0.0793 |

## Direct answers

1. **Log-power stem / v1 backbone weakness:** synthetic checks passed = `True`. Strict shared loop-4 accuracy is `0.7639` versus source-only Sinc `0.8472`. This does not repair the practical backbone deficit relative to Sinc on A01.
2. **Same checkpoint Acc@1 -> Acc@4:** `0.7986` -> `0.7639`; increase = `-0.0347`.
3. **Correction vs corruption (1->4):** `0.0278` vs `0.0625`.
4. **Ambiguous trials:** low-confidence correction/corruption/gain = `0.0816` / `0.1837` / `-0.1020`; medium = `0.0000` / `0.0000` / `+0.0000`. The low-confidence group does not improve: the extra loops are not correcting ambiguous EEG and instead show overthinking/corruption. Full bins are in `A01_CONFIDENCE_GROUPS.csv`.
5. **Shared vs untied K4:** shared `0.7639`; untied `0.6736`.
6. **Decision:** `NO-GO`.

Selection epochs, source-only: Sinc=21, shared=28, untied=23. Target scoring labels were opened only after all checkpoint hashes were recorded.
