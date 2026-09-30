# Stage 0 result — A01 only, seed 0

Artifacts are written outside Git at
`/root/rivermind-data/online_adaptation_stage0/bnci2014001_a01_seed0/`.

| Method | Strict S2 accuracy | BA | Macro-F1 |
|---|---:|---:|---:|
| Source-only SincAdaptNet | 80.56% | 80.56% | 80.09% |
| Official BTTA-DG literal path | 80.56% | 80.56% | 80.09% |
| NeuroLoop static K=1 | 70.14% | 70.14% | 70.14% |
| NeuroLoop static K=4 | 69.44% | 69.44% | 69.35% |
| NeuroLoop shared K=1 + state | 54.17% | 54.17% | 41.98% |
| NeuroLoop shared K=2 + state | 70.83% | 70.83% | 70.14% |
| NeuroLoop shared K=4 + state (final) | 64.58% | 64.58% | 59.50% |
| NeuroLoop shared K=8 + state | 70.14% | 70.14% | 70.13% |
| NeuroLoop untied K=4 + state | 72.92% | 72.92% | 72.90% |

Paper EA and strict causal EA happened to give the same scores in this single
stream. This is an observed result, not a reason to merge the two protocols.

The causal checks passed for BTTA-DG and shared K=4: future S2 samples did not
change a prefix prediction, runners expose no target-label parameter, repeated
streams are deterministic, and session reset restores M0. However the main
hypothesis fails on this single subject: final shared K=4 is below K=1/K=2/K=8,
below static K=4, and below source-only SincAdaptNet. No claim that NeuroLoop
improves online adaptation is justified.

Do not expand to the 10-seed benchmark. The next permitted operation is the
pre-registered 9-subject seed-0 BTTA reproduction gate, followed by a decision
on whether this architecture is worth continuing.
