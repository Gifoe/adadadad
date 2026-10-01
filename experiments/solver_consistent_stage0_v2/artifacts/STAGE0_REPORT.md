# Solver-Consistent Recurrent Reasoning — Stage 0 V2

**Pipeline state: INFRA_BLOCKED**

**正式 seed=0 对照实验未完成。**

The unique blocking state is `INFRA_BLOCKED`. No GO / UNCLEAR / STOP scientific decision is permitted because `FORMAL_COMPLETE` was not reached.

## 1. Experimental setup

- Data: 135,000 examples; `prontoqa-symmetric-component-v2`; seed namespace 2000; manifest payload SHA256 `e363c4bebe28946f5f621834f7c80e9f52bb8f1930a3278caec872af3349553c`.
- Repair: replace the isolated opposite-goal dummy with a predicate-renamed mirrored proof component. Its queried-entity start fact has opposite polarity, matching unsigned entity/frequency statistics while keeping the signed decoy chain unreachable.
- Full audit: 135,000/135,000 passed parser round-trip, label reachability, opposite non-reachability, minimum depth, contradiction, checksums, and global exact/canonical/predicate-isomorphic deduplication.
- Shortcut gate: PASS. Highest no-training heuristic 55.00%; highest structural classifier 55.44%; highest single surface feature 53.28%; antecedent-frequency 50.00% in every IID/OOD stratum.
- Tokenizer: train-only BPE, actual vocabulary 2288, max length 1376, SHA256 `2116bfb62e88e61c2aea4c868d1caae9e50f5037ee325d852d3a84a64e84510c`; all split truncation rates <=0.06%.
- Models: parameter counts {'vanilla_loop': 17639938, 'step_conditioned_loop': 17657378, 'vector_field': 17657378}. Maximum difference 0.099%.
- Environment: Windows 10 build 18363, RTX 5090, Torch 2.8.0+cu128, CUDA runtime 12.8, driver 616.56, BF16, math SDPA, activation checkpointing, deterministic algorithms, microbatch 16/effective batch 128.

## 2. Does budget sensitivity actually exist?

`NOT_RUN_INFRA_BLOCKED`. There are no trained best/final checkpoints and no task-level unseen-budget or harmful-flip results.

## 3. Does vector field reduce sensitivity?

`NOT_RUN_INFRA_BLOCKED`. Any method comparison would be fabricated.

## 4. Numerical behavior

Implementation-only ODE/NFE/schedule tests passed. Trained-Model-C convergence and reference Check I are `NOT_RUN_INFRA_BLOCKED`. Numerical convergence cannot be inferred from the toy implementation test.

## 5. Alternative explanations

The data and registered shortcut suite passed, so the stop is not caused by the old antecedent-frequency leak. The real-shape VRAM probe passed at microbatch 16 with about 4.90 GB allocated and 6.12 GB reserved. The frozen Vanilla A6 job then crashed three times from the same atomic update-25 checkpoint. WER records show `unknown/StackHash` with `0xc0000005`, `torch_cpu.dll` with `0xc0000005`, and `nvcuda64.dll` with `0x80000003`. These are fault locations, not established causes. The required 50-update validation/checkpoint round-trip was never completed.

## 6. Decision

Scientific GO / UNCLEAR / STOP: **NA**. Pipeline state: **INFRA_BLOCKED**. Checks A and I are `NOT_RUN`; Check L is `FAIL`; all formal model updates are 0.

## 7. Execution integrity

Completed: A0 environment/repository audit, A1 full data repair/generation, A2 full semantics/dedup/shortcut gate, A3 tokenizer/length freeze, A4 model interface/fairness tests, A5 solver tests, and A6 real-shape sizing. Failed: A6 sustained stability. Not run: three-model 50-update completion, A6 2,000-update Vanilla stability, A7 learning gates, A8 formal training, A9–A14 evaluations, latency, recurrent contribution, and trained numerical convergence. Unknown metrics are `NA`, not zero. Main figures were **NOT GENERATED** because complete real model results do not exist.
