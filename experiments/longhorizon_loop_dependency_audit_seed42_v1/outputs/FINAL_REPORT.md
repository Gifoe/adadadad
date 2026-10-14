# FINAL REPORT — Long-Horizon Loop Dependency Audit v1

**Final decision: `INCONCLUSIVE`.** Stage 0 native baselines, Stage 1 interfaces, and Stage 2–3 fixed-text interventions were executed. The official serving engine remains unverified on Windows. Stage 4 was not opened because its positive-evidence gate did not pass.

## A. Reproduction & Infrastructure

Official TaH commit: `78ddf9a2c25713fced03a736c5746161f6ee590c`. Max2 HF revision: `e965dff2bd539b5be8087126b7d26bcb47e1b159`; independently trained Standard: `bbf0d82946c8e242ab2dc1f3be20408e317562ec`. NESTFUL commit: `fc2c4123e73500a56185a5fb354f05d1c8b4890c`.

GPU: NVIDIA GeForce RTX 5090; CUDA runtime 12.8; Python 3.12.15; BF16, batch1, seed42. Versions: `{"accelerate": "1.12.0", "huggingface-hub": "0.36.2", "numpy": "2.4.2", "safetensors": "0.7.0", "torch": "2.9.1+cu128", "transformers": "4.57.3"}`. Complete package/model-file hashes are in execution_provenance.json and source_metadata/.

The host is Windows without installed WSL/Docker. The official extra cannot resolve a Windows Triton wheel. After portable imports were installed, the actual official smoke test reached `makeDeviceForHostname(): unsupported gloo device`. No engine source or model weights were patched to bypass this. The mechanism backend uses the pinned official tah2/model path with only its attention implementation changed to native SDPA. This does not meet the required empirical mini-SGLang parity check.

No-intervention native/wrapper difference: 0.0. Repeats, causal isolation, branch independence and native probability-mixture reconstruction are recorded in preflight_native.json. Weight hash before/after: `dc5d4ba20239c81ac1e53b4d63a67290245a85ff848963d62a0c2c215c6a606e`.

Full-prefill/incremental BF16 outputs are numerically close under the declared tolerance, not bitwise equal. Full-vocabulary log-probability max differences were 0.4375/0.5 nats at K1/K2; target-CE and top1 checks are recorded separately. Broad full-vocabulary tolerance alone is not evidence that a 0.01-nat mechanism effect is reliable.

A development-only layout null revealed nontrivial numerical effects: its 95th percentile absolute deviation was 0.06250 nats and maximum 0.07267 nats. The complete matched audit was therefore also run with an invalid, zero deep-cache slot in K1 branches. Original results remain intact. See NUMERIC_REPAIR.md.

Final aligned source-K2/consumer-K1 null maximum: 0 nats; I minus G_source maximum: 0 nats.

Tracked CUDA-synchronized forward/generation wall time: 3975.6s (1.104h). This includes CPU observer work inside forwards and is not a kernel-only GPU utilization measurement. Setup, transfers, model hashing and raw-logit compression are separate. An aborted initial masking call and preflight repetitions are noted as additional unaggregated short runs; this total is a tracked lower bound, not a fictitious exact resource bill. The 8h soft cap was not reached.

Real pilot cost predicted the audit well: original development used 83.06s of measured forwards for 96 dependent pairs plus controls; the original audit used 223.34s for 253 dependent pairs plus controls. Prefixes were reused. Per-branch wall times, actual loop counts, cache depths, peak allocation and a parameter-matmul FLOPs proxy are preserved. The FLOPs proxy excludes attention/context and is not used for fairness claims.

## B. Baseline Results

These are the same 20 development prompts, 512 maximum output tokens, greedy native generation, thinking disabled. Standard uses different weights; the other three use identical max2 weights. Loop averages include all actually executed prompt/decode positions.

| Method | N | Parse | Executable | Full task | Mean depth | Generated tokens | Wall s | Peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard | 20 | 25% | 20% | 0% | 1.000 | 9230 | 183.7 | 3690.9 |
| forced1 | 20 | 0% | 0% | 0% | 1.000 | 10240 | 218.5 | 5318.1 |
| forced2 | 20 | 0% | 0% | 0% | 2.000 | 10240 | 432.0 | 5409.5 |
| adaptive | 20 | 35% | 35% | 0% | 1.244 | 9510 | 216.2 | 5318.1 |

A parsed controlled state array counts as executable output, even when its values are wrong. NESTFUL execution uses actual official functions and typed final-value equality. Unquoted variable references are rejected as invalid JSON rather than silently repaired. These results describe this prompt/output-budget protocol; they do not prove that the checkpoint cannot reason with longer outputs or a different agent interface. Correct-intermediate rates and first errors are available in baseline_pilot_records.jsonl.



### Separate development output-length sensitivity

Same prompts/weights/native SDPA dispatch, max2048 outputs; four deterministically selected development prompts, not the primary 20-prompt comparison. Native workers crashed with 0xC0000005 and 0xC0000409 (Python/UCRT event-log modules). Partial multi-prompt records are retained but excluded from accuracy. Successful isolated prompts have immediate before/after weight validation and durable export before OS process teardown. The remaining worker processes were constrained to the observed lower-efficiency CPU-set class (logical processors 16–31), with no BIOS/voltage/global scheduler or model changes. Earlier validated Standard prompts used the original affinity. Thus latency here is operational diagnostic data, not a fair output-length speed comparison. The crash root cause remains unresolved.

| Method | N | Parse successes | Full successes | Limit hits | Generated tokens | Wall s |
| --- | --- | --- | --- | --- | --- | --- |
| standard | 4 | 2 | 0 | 2 | 4828 | 95.8 |
| forced1 | 4 | 0 | 0 | 4 | 8192 | 348.5 |
| forced2 | 4 | 2 | 0 | 2 | 7704 | 657.4 |
| adaptive | 4 | 2 | 0 | 4 | 8192 | 351.5 |
## C. Local vs Delayed Benefit

Local-small threshold: 0.01000 nats, frozen on development only. Primary numbers below use cache-aligned replay. Confidence intervals use 5000 seed42 resamples of original problems/matched groups, equal weight per parent. They are descriptive and are not multiplicity-adjusted.

| Dataset | G_local anchor | G_local full source unit | G_source delayed | G_source complete consumer span |
| --- | --- | --- | --- | --- |
| NESTFUL binding | -0.00162 [-0.00243, -0.00078]; pairs=125, groups=64 | -0.00004 [-0.00040, 0.00033]; pairs=125, groups=64 | -0.01147 [-0.02415, -0.00007]; pairs=125, groups=64 | -0.00269 [-0.00506, -0.00057]; pairs=125, groups=64 |
| Controlled states | -0.03190 [-0.10464, 0.04725]; pairs=128, groups=64 | -0.02639 [-0.05229, 0.00142]; pairs=128, groups=64 | 0.00506 [-0.00109, 0.01139]; pairs=128, groups=64 | 0.00506 [-0.00109, 0.01139]; pairs=128, groups=64 |

| Dataset | Pairs with gain >0.01 | Pairs with gain <-0.01 | Native-K1/low-local pairs | Positive eligible pairs |
| --- | --- | --- | --- | --- |
| NESTFUL binding | 17 | 26 | 106 | 14 |
| Controlled states | 62 | 44 | 0 | 0 |

There are positive individual counterfactuals. Their presence is not hidden by the negative/uncertain mean. Selecting those audit cases retrospectively is an oracle, not evidence that an online policy can predict their benefit or that they have semantic dependency specificity.

| Dataset / stratum | Delayed source gain, 95% CI and actual N |
| --- | --- |
| NESTFUL binding / native_source_K1 | -0.00313 [-0.01134, 0.00516]; pairs=107, groups=63 |
| NESTFUL binding / low_local_abs_gain | -0.00817 [-0.02046, 0.00295]; pairs=116, groups=63 |
| NESTFUL binding / native_source_K1_and_low_local | -0.00340 [-0.01157, 0.00481]; pairs=106, groups=63 |
| Controlled states / native_source_K1 | NA (no eligible clusters) |
| Controlled states / low_local_abs_gain | 0.00324 [-0.04495, 0.05144]; pairs=5, groups=5 |
| Controlled states / native_source_K1_and_low_local | NA (no eligible clusters) |

Continuous problem-mean associations: `{"controlled_symbol_chains_new_diagnostic": {"OLS_slope_descriptive_only": -0.002468788180062003, "n_clusters": 64, "scope": "problem-mean association; not causal mediation or independence proof", "spearman_rho": 0.04418498168498168}, "nestful_v2_math": {"OLS_slope_descriptive_only": 0.7120784553475913, "n_clusters": 64, "scope": "problem-mean association; not causal mediation or independence proof", "spearman_rho": -0.15146520146520145}}`. Correlation is descriptive; no independence or mediation conclusion follows from it. Raw per-loop source CE and the full source-unit metric are also provided. A low mixture-CE gain is not automatically evidence of an unchanged deep representation.

![Local versus delayed gain](figures/audit_aligned/local_vs_delayed.png)

## D. Producer–Consumer Interaction

| Dataset | U11 | U21 | U12 | U22 |
| --- | --- | --- | --- | --- |
| NESTFUL binding | -1.06891 [-1.31078, -0.83516]; pairs=125, groups=64 | -1.06891 [-1.31078, -0.83516]; pairs=125, groups=64 | -1.02239 [-1.26859, -0.79188]; pairs=125, groups=64 | -1.03386 [-1.28251, -0.80011]; pairs=125, groups=64 |
| Controlled states | -1.91856 [-2.04262, -1.79714]; pairs=128, groups=64 | -1.91856 [-2.04262, -1.79714]; pairs=128, groups=64 | -1.91660 [-2.03253, -1.80046]; pairs=128, groups=64 | -1.91154 [-2.02848, -1.79513]; pairs=128, groups=64 |

| Dataset | G_source | G_consumer | I |
| --- | --- | --- | --- |
| NESTFUL binding | -0.01147 [-0.02415, -0.00007]; pairs=125, groups=64 | 0.04652 [-0.00325, 0.10217]; pairs=125, groups=64 | -0.01147 [-0.02415, -0.00007]; pairs=125, groups=64 |
| Controlled states | 0.00506 [-0.00109, 0.01139]; pairs=128, groups=64 | 0.00196 [-0.03389, 0.03816]; pairs=128, groups=64 | 0.00506 [-0.00109, 0.01139]; pairs=128, groups=64 |

At the single target anchor, DUO forbids the K1 consumer from seeing a K2 producer. Thus U21=U11 and I=G_source. This observed identity is an architecture check, not independent evidence for coordinated allocation. Complete consumer-span utilities are supplementary and may include other frozen K2 positions; the identity is not extended to them.

## E. Mechanism Controls

| Dataset | Dependent minus matched nondependent | Deep-KV masking attenuation | Deep minus shallow substitution | Masked-layout null |
| --- | --- | --- | --- | --- |
| NESTFUL binding | -0.00623 [-0.02858, 0.01547]; pairs=12, groups=12 | -0.01147 [-0.02415, -0.00007]; pairs=125, groups=64 | -0.00733 [-0.01939, 0.00303]; pairs=125, groups=64 | 0.00000 [0.00000, 0.00000]; pairs=125, groups=64 |
| Controlled states | 0.00962 [-0.00416, 0.02234]; pairs=64, groups=64 | 0.00506 [-0.00109, 0.01139]; pairs=128, groups=64 | 0.00228 [-0.00427, 0.00889]; pairs=128, groups=64 | 0.00000 [0.00000, 0.00000]; pairs=128, groups=64 |

Controls match output-token event role, native source depth and a half-to-double token-distance bound, then rank distance/local native-CE differences. They do not enforce identical operator type or identical graph-position role; matching is coarse and its distances/difficulty differences are saved. Their absolute loop budgets are asserted equal. Sparse NESTFUL control availability limits specificity claims; no control is fabricated for H16, where every earlier state is an ancestor. All missing-control reasons are recorded in core_runtime_*.json.

A positive deep-versus-shallow gap alone can mean that duplicated shallow memory is harmful. It does not show that adding deep memory is better than K1. Likewise, KV masking can establish access dependence only when it attenuates a real positive semantic benefit. Neither result alone establishes a long-horizon allocation advantage.

In the aligned layout, masked K2 source memory and the K1 padded source are observationally equivalent. Therefore the masking attenuation reduces to G_source when that strict null passes. Together with I=G_source, these are overlapping algebraic contrasts, not three independent confirmations. Dependency specificity and actual task success remain the discriminating evidence.

![Mechanism controls](figures/audit_aligned/mechanism_controls.png)

## F. Long-Horizon Effects

| Controlled depth | Delayed source gain | Consumer gain |
| --- | --- | --- |
| H=4 | 0.00314 [-0.00537, 0.01139]; pairs=64, groups=64 | -0.00565 [-0.05196, 0.04282]; pairs=64, groups=64 |
| H=16 | 0.00698 [-0.00326, 0.01756]; pairs=64, groups=64 | 0.00957 [-0.04044, 0.05749]; pairs=64, groups=64 |

Paired H16 minus H4 delayed-source contrast: 0.00384 [-0.01065, 0.01829]; pairs=64, groups=64. This remains a depth/position/distance contrast, not an isolated graph-depth causal effect.

The primary audit contains 64 independent NESTFUL problems and 64 controlled matched groups (128 H4/H16 inputs). It has 253 available dependent pairs: 125 NESTFUL and 128 controlled, hence 1012 primary 2x2 branches per layout. Three NESTFUL problems have only one selected valid pair. H2/H8 use a separate cost-led extension on the already locked 64 groups. They are not promoted to the primary test. Symbol renaming is scored separately on the first 16 locked groups, at both primary depths.

| Renamed controlled stratum | Delayed source gain |
| --- | --- |
| all | -0.01886 [-0.03345, -0.00486]; pairs=32, groups=16 |
| H4 | -0.00778 [-0.02668, 0.01021]; pairs=16, groups=16 |
| H16 | -0.02994 [-0.05402, -0.00613]; pairs=16, groups=16 |

Matched renamed-minus-base delayed-source contrast on the same 16 groups: -0.02888 [-0.04352, -0.01529]; pairs=32, groups=16. Original and renamed inputs each retain their own recorded native depth policy, so this is a robustness comparison rather than a pure fixed-policy lexical intervention.

NESTFUL anchors predict the first digit of a variable identifier, with its full reference span also scored. Its labels do not carry numeric tool results; this is a binding probe, not a test of hidden arithmetic-result retention. Controlled anchors predict actual state symbols, but correct intermediate states are teacher-forced. They can screen earlier source information. H changes source location/token distance as well as chain organization, so a depth difference cannot be attributed solely to latent horizon. The source cache entry being deepened is the prediction position before the output value, rather than the position after consuming that value.

![Dependency-depth effect](figures/audit_aligned/depth_effect.png)



### Locked-manifest H2/H8 extension

Measured primary replay cost was below the cap, so all 64 existing audit groups were also evaluated at H2 and H8 with cache alignment. This adds 128 dependent pairs / 512 core cells plus controls, without adding independent groups or changing the frozen primary thresholds. These exploratory results do not reopen the primary Stage4 gate.

| Depth | Delayed source gain | Native K1 source gain |
| --- | --- | --- |
| 2 | -0.00012 [-0.01373, 0.01291]; pairs=64, groups=64 | NA (no eligible clusters) |
| 8 | 0.00403 [-0.00366, 0.01167]; pairs=64, groups=64 | NA (no eligible clusters) |
## G. Compute–Performance

Positive-evidence gate: `{"controlled_symbol_chains_new_diagnostic": {"dependency_specific": false, "native_K1_low_local_positive": false, "positive_mask_attenuation": false, "positive_source_gt_0p01_nats": false}, "nestful_v2_math": {"dependency_specific": false, "native_K1_low_local_positive": false, "positive_mask_attenuation": false, "positive_source_gt_0p01_nats": false}}`. Stage4 random/local-oracle/joint-oracle allocation runs were not executed. No equal-budget task-performance comparison or oracle advantage is claimed. The 512-token baselines have unequal executed compute, and are baseline diagnostics rather than matched-budget evidence.

## H. Negative Evidence & Remaining Limits

- Native original/wrapper equality and cache causal checks do not replace missing serving-engine equivalence.
- No reliable positive dependency-specific delayed benefit supports a GO decision in the target low-local/native-K1 stratum.
- Controlled-state sources may have no native-K1 examples; their actual subgroup N is reported above rather than filled in.
- A 512-token one-shot task protocol yields zero full-task successes in the pilot. Longer-output sensitivity is reported separately if executed.
- NESTFUL probes bindings and identifier digits; controlled teacher forcing reveals intermediate values. Neither alone identifies general hidden long-range reasoning.
- The original cache layouts generate numerical effects comparable to small purported benefits. Both layouts are reported and the strict null is checked.
- H16 lacks a legal nondependent same-role control, while NESTFUL matching coverage is sparse.
- One seed, one checkpoint family, small independent-group counts and descriptive multiple CIs limit generalization.
- Deployed agent runs and Stage4 privileged/equal-budget allocations are not executed. No missing result is imputed.

## I. Final Research Decision

**INCONCLUSIVE.** The requested mechanism audit is implemented and actually run, with useful negative evidence for these probes. It does not justify GO_H2 or GO_H1_ONLY. It also does not establish H0 for the broader research question: the official serving parity check, free task capability, semantic source-carrier choice and the main native-K1 state stratum remain limiting. Building a coordinated-recurrence model now would outrun the evidence.

The next decisive experiment should make the source a value-bearing, causally observable state, separate token distance from graph depth, and avoid revealing every relevant mediator through teacher forcing. It must first pass a development capability/eligible-source check. More checkpoints or a new router are not substitutes for that identification step. Prospective method candidates are delivered separately from these public audit records.

Reproduction commands are in README.md; per-phase outputs include locked plans, policies, thresholds, raw branch records, model hashes, aligned and unaligned statistics and figures. See PRIMARY_SOURCES.md for the verified primary literature and upstream licenses.
