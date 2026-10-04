# Official protocol audit (frozen before observing closure results)

Upstream: OSU-NLP-Group/Loop-Think-Generalize, commit `bb22b192977b1b136fde985c8e7a2392d172d97a`. Fresh fetch confirmed HEAD and origin/main at this commit. The README, train_extrapolation.py, gpt_utils_extrapolation.py and inference_extrapolation.py were read from the actual checkout. Verbatim snapshots are retained in official/. No checkpoint metadata was available to override the source defaults; this is a source-config reproduction on the unchanged v1 generated dataset, not a claim to reproduce an undocumented published checkpoint bit-for-bit.

| Behavior | Actual source evidence | A1 | Formal matched comparison |
|---|---|---|---|
| Curriculum | train_extrapolation.py lines 390–437: cumulative records through stage k; train_model called per stage | Atomic + <=D, D2 to D5 | Same |
| Advancement | lines 231, 256–261: current split >0.95 once at epoch end; force_grok optionally requires epoch >=1000 | Official >0.95 once at epoch end, force_grok=False | User-required >=0.95 for three consecutive 500-update evaluations |
| AdamW reset | lines 124–129: new optimizer and scheduler in each train_model | Yes, LR1e-4, weight decay .01, betas(.9,.999), eps1e-8 | Identical |
| Scheduler | lines 127–129: len(dataloader)*num_epochs, default num_epochs100001 and warmup2000 | Per-stage horizon, no global20k decay | Identical |
| Dynamic recurrence | lines 148–152, 201–203: Poisson(4), clipped2–8; validation K8 | Exact policy | Fixed K5 by explicit user resolution |
| Original input/readout | CompositionDataset/custom_collate; train lines166–174 | Remove added <state>; pad to50, last padded position logits[:, -1]; vocab217 | Unchanged v1 <state> readout; vocab218; same intermediate targets |
| Architecture | RecurrentGPT2Block lines322–410 | Unmodified class, four shared GPT2 blocks,768dim,12heads,NoPE, no injection,c_scale0,tiedhead | Verbatim v1 model_extension.py; same class layers and initialization |
| Dropout | train lines352–353 passes only embd_pdrop; GPT2Config4.44.2 defaults resid/attn dropout .1 | Explicit all three=0 per user requirement | All three=0, unchanged v1 |
| Train/eval flag | train line138 calls train once; evaluate_model_test calls eval without restoring train | Wrapper restores train each update; with all dropout0 this has no stochastic effect | Same restoration |
| DataLoader | shuffle=True, default drop_last=False | Full shuffled passes including short final batch | Same; stage-specific seed42+D yields identical order for both formal arms |
| Safety ceiling | Official epochs are a ceiling, not prescribed duration | Added user cap50,000 updates/stage | Same cap; stop failed arm, no later stages |
| Compilation | Default train_mode=max-autotune is only a runtime option | Eager official blocks under BF16, avoiding a new numerical implementation | Same |

## Recurrence conflict resolution

The supplied prompt's Rule0 fixes K_train=5 while PartE prefers dynamic2–8. The user explicitly answered: “正式对比固定 K=5，保持 v1 方法不变（推荐）”. Thus only A1 uses official dynamic recurrence; formal baseline and grounded both remain K5. No recurrence-dependent supervision or method changes were introduced.

## Per-stage scheduler horizons

| Stage | Train examples | Batches/epoch | Scheduler horizon | Reset |
|---|---:|---:|---:|---|
| D2 | 17000 | 133 | 13300133 | AdamW + scheduler |
| D3 | 32000 | 250 | 25000250 | AdamW + scheduler |
| D4 | 47000 | 368 | 36800368 | AdamW + scheduler |
| D5 | 62000 | 485 | 48500485 | AdamW + scheduler |

Scheduler creation sets actual initial LR0 because of warmup; base LR is1e-4. At update2000 LR reaches1e-4. The50k update cap does not replace the official scheduler horizon. No stage LR decay, gradient clipping, force-grok extension or optimizer improvement is added.

## Data and held-out use

Reuse v1 training records verbatim: 2000 atomic facts +15000 records each for D2–5, total62000. No new dataset is generated. Existing750 held-out records per ID hop are deterministically partitioned with RandomState42 into375 curriculum-validation and375 independent endpoint-test records. The manifest freezes IDs before training. Neither partition overlaps training; endpoint labels cannot advance curriculum. OOD remains750 records each atD6,8,10,12,16,20. Only A1 removes the v1-added state token, recovering original-format inputs with the same entities/relations/targets.

The cached A1 inputs/targets/masks are checked against the unmodified official CompositionDataset and custom_collate(max_len=50). Formal lambda0 loss and default logits are checked against actual official forward. Existing v1 initial weights are reused by both formal arms. Loss and extension source are unchanged, coefficient0 vs1 is the only arm difference.

## Measurement definitions

First observed updates_to_95 is evaluated at epoch ends for A1 and at500-update intervals for formal arms. Advancement updates are recorded separately. A1 has extra epoch-end evaluations for its official advancement rule. Both formal arms have the same500-update cadence and three-pass criterion. Full curves include all scheduled evaluations; validation accuracy is not independent final-test accuracy.

The logged train_accuracy_sample uses the first512 records of the cumulative training pool, all atomic facts under the unchanged record ordering. It measures atomic retrieval fit, not whole-pool training accuracy. This fixed diagnostic subset is identical between arms; it cannot advance the curriculum. Training loss averages actual cumulative-pool updates.

Independent ID closure requires completion of Stage5 plus K5 macro>=.90; >=.85 at each hop is reported as preferred rather than silently added as another hard gate. A1 must complete D2–5 stages at the official threshold to establish current-stage reproduction; endpoint retention is additionally measured and reported. Failure stops downstream experiments. Missing gated outputs/checkpoints are explicitly marked NOT RUN.

The original v1 prototype/intervention code is copied, with path/import adaptation only and independent-ID filtering in the accuracy sweep. The original special case D6 uses k2 only; k4 on D6 has only two remaining transitions and was excluded in the v1 preregistration. All deeper OOD hops use k2 andk4. This is preserved rather than changed to improve outcomes.
