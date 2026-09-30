# NeuroLoop v2 failure analysis

**NO-GO.** The pre-registered A01 gate did not meet both required conditions: Acc@4 > Acc@1 and correction rate > corruption rate.

- Stem synthetic checks passed: `True`.
- Acc@1 -> Acc@4: `0.7986` -> `0.7639`.
- Correction / corruption: `0.0278` / `0.0625`.
- Untied K4 accuracy: `0.6736`; shared K4: `0.7639`.
- Inspect `A01_LOOP_DIAGNOSTICS.csv`: cosine changes close to one indicate identity-like loops; large logit movement together with net accuracy loss indicates overthinking. `model_audit.json` stores learned gamma values.

The v2 protocol forbids increasing width/depth, adding state, changing target preprocessing, or trying other datasets after this result. Do not expand to nine subjects or further seeds.
