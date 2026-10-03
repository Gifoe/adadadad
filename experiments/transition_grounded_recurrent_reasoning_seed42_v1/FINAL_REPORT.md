# Transition-Grounded Recurrent Reasoning — Seed42 Viability Report

## Research question

Does transition supervision make recurrent depth correspond to verifiable and causally used reasoning-state transitions?

**Decision: INVALID TRAINING RUN / TRAINING FAILURE.** This is neither evidence supporting nor falsifying the research hypothesis.

## Experimental setup

```json
{
  "seed": 42,
  "d_model": 768,
  "num_recurrent_layers": 4,
  "num_heads": 12,
  "positional_embedding_type": "none",
  "dropout": 0,
  "precision": "bf16",
  "c_scale": 0.0,
  "input_injection": false,
  "lr": 0.0001,
  "weight_decay": 0.01,
  "adam_betas": [
    0.9,
    0.999
  ],
  "warmup_steps": 2000,
  "lr_schedule": "official linear warmup/decay, frozen horizon 20000 for both arms including conditional extension",
  "K_train": 5,
  "lambda_transition": 1.0,
  "batch_size": 128,
  "curriculum": [
    [
      2,
      2000
    ],
    [
      3,
      2000
    ],
    [
      4,
      3000
    ],
    [
      5,
      5000
    ]
  ],
  "initial_updates": 12000,
  "max_updates": 20000,
  "intervention_count_per_D": 200,
  "prototype_min_count": 10,
  "upstream_commit": "bb22b192977b1b136fde985c8e7a2392d172d97a",
  "D": [
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    16,
    20
  ],
  "K": [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    16,
    20,
    24
  ],
  "same_state_majority_threshold": 0.5,
  "noise": "replace state by isotropic Gaussian direction rescaled to CF prototype L2 norm",
  "frontier": "minimum swept K within 95% of swept peak",
  "flip": "any swept K<24 correct and K24 wrong, denominator any early correct"
}
```

Official implementation was inspected and extended; no additional method was introduced. The initial weights, sample plan, K_train, optimizer, learning-rate schedule, batch size, and budgets are shared. Train only atomic and composed hops 2–5; test ID hops 2–5. All 301,250 generated paths replayed successfully; separate 1,000 random replays passed.

## Q1 — ID performance

Both arms received the permitted maximum 20,000 updates because at least one failed the 12,000-step gate.

| Model | D2 | D3 | D4 | D5 | Macro |
|---|---:|---:|---:|---:|---:|
| baseline | 0.93% | 0.67% | 0.80% | 0.53% | 0.73% |
| grounded | 38.93% | 7.73% | 0.80% | 0.40% | 11.97% |


## Q2 — Depth extrapolation

Not evaluated. The predeclared ID sanity gate failed; OOD or intervention numbers would not establish viability. Final-accuracy CSVs contain measured ID K=5 counts only; other evaluation CSVs contain headers. Only the ID final-accuracy figure plots measured values; unexecuted evaluation figures say NOT EVALUATED. No synthetic or substituted results.

## Q3 — Recurrence transition semantics

Not evaluated. The predeclared ID sanity gate failed; OOD or intervention numbers would not establish viability. Final-accuracy CSVs contain measured ID K=5 counts only; other evaluation CSVs contain headers. Only the ID final-accuracy figure plots measured values; unexecuted evaluation figures say NOT EVALUATED. No synthetic or substituted results.

## Q4 — Overthinking

Not evaluated. The predeclared ID sanity gate failed; OOD or intervention numbers would not establish viability. Final-accuracy CSVs contain measured ID K=5 counts only; other evaluation CSVs contain headers. Only the ID final-accuracy figure plots measured values; unexecuted evaluation figures say NOT EVALUATED. No synthetic or substituted results.

## Q5 — Causal state use

Not evaluated. The predeclared ID sanity gate failed; OOD or intervention numbers would not establish viability. Final-accuracy CSVs contain measured ID K=5 counts only; other evaluation CSVs contain headers. Only the ID final-accuracy figure plots measured values; unexecuted evaluation figures say NOT EVALUATED. No synthetic or substituted results.

## Failure analysis

Training did not establish sufficient ID composition performance within the fixed budget. Categories A–E (not decodable, decodable/noncausal, causal/no utility, utility/unstable terminal, baseline equivalence) remain untested. Losses and smoke checks show execution and gradient flow, but those do not substitute for ID generalization. No post-result architecture, timestep, gating, or supervision changes were made.

The official training code advances curriculum by measured accuracy and optionally forces at least 1,000 epochs at a stage. This fixed-budget experiment advances after 2,000 updates at max-hop 2 (about 15 passes through 17,000 records), before establishing ID composition. This is a concrete protocol mismatch with the official success-dependent curriculum, not proof of the failure cause. The official zero-scaled residual initialization and 2,000-step warmup were preserved. The current budget is empirically insufficient to establish the required held-out ID performance.

### Execution diagnostics: baseline, checkpoint step 20000

| Hop | Sample train accuracy | Sample ID accuracy |
|---|---:|---:|
| 1 | 91.41% | NA |
| 2 | 97.46% | 0.59% |
| 3 | 97.85% | 0.78% |
| 4 | 100.00% | 0.59% |
| 5 | 100.00% | 0.59% |

Random 512 records/hop, seed 42. Text/cache checks=2000; recurrence prefix difference=0.0; exact raw-state no-op difference=0.0. Padding variants are recorded separately; BF16 rounding does not explain the failed ID gate.

### Execution diagnostics: grounded, checkpoint step 20000

| Hop | Sample train accuracy | Sample ID accuracy |
|---|---:|---:|
| 1 | 89.65% | NA |
| 2 | 98.05% | 38.87% |
| 3 | 99.02% | 8.20% |
| 4 | 99.61% | 0.59% |
| 5 | 99.61% | 0.59% |

Random 512 records/hop, seed 42. Text/cache checks=2000; recurrence prefix difference=0.0; exact raw-state no-op difference=0.0. Padding variants are recorded separately; BF16 rounding does not explain the failed ID gate.

## Resource report

GPU: RTX 5090 32GB; BF16. PyTorch allocated peak, excluding other processes.

| Model | Updates | Examples seen | Train hours | Peak VRAM MiB | Examples/s |
|---|---:|---:|---:|---:|---:|
| baseline | 20000 | 2,560,000 | 0.283 | 2088.1 | 2512.2 |
| grounded | 20000 | 2,560,000 | 0.285 | 2416.4 | 2496.8 |

Total measured GPU-stage hours: 0.573 (training + smoke + evaluation where run).

## GO / NO-GO

**INVALID TRAINING RUN**, not a hypothesis-level NO-GO. A second research-method round cannot be justified by this run. Any later attempt should first preregister a matched ID-training protocol that establishes baseline composition, with adequate curriculum training before advancing. The counterfactual mechanism remains untested.

Full data and actual initial/baseline/grounded checkpoints remain on the server and are delivered through Git LFS; see ARTIFACT_MANIFEST.json.