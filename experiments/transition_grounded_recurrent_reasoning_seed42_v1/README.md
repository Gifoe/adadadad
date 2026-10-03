# Transition-Grounded Recurrent Reasoning: seed42 viability experiment

Question: does supervising intermediate states induce causally used recurrent
state transitions, and does that improve extrapolation or terminal stability?

Official source: https://github.com/OSU-NLP-Group/Loop-Think-Generalize
Pinned commit: bb22b192977b1b136fde985c8e7a2392d172d97a.
The six inspected source files are preserved verbatim in `official/`.
The actual official generation functions are executed by AST extraction, avoiding
their hardcoded top-level 40-hop writes. `model_extension.py` subclasses the
official RecurrentGPT2Block, retaining GPT2Block layers, initialization (including
official training default c_scale=0), causal attention and tied output head.
The official CompositionDataset tokenizer/target behavior is verified against
the cached training arrays. A fixed-update driver extends the official AdamW,
BF16, warmup scheduler, curriculum and evaluation mechanisms for this protocol.

Both arms: 768 dimensions, four shared recurrent layers, twelve heads, NoPE,
all embedding/residual/attention dropouts zero, K_train=5, batch128, seed42.
Final prediction uses the last real input token `<state>` (then padding).
The raw carrier is normalized only when decoded, matching the official final
LayerNorm/head. Prototypes and patches use the raw carrier, never decoded tokens.
Only intermediate CE differs (coefficient zero vs one).

Frozen choices before formal outcomes:
- Global AdamW, LR1e-4, decay.01, standard betas, no clipping, 2000 warmup.
- Official linear scheduler frozen at the permitted maximum 20000 updates;
  no LR restart or different schedule for an arm. This differs from official
  per-curriculum optimizer recreation, needed here for an exact shared budget.
- 2000/2000/3000/5000 updates at max hops2/3/4/5. If either K5 ID macro<.90,
  both extend by8000 max-hop5 updates. If still invalid, no OOD interpretation.
- Smoke200 updates/arm from the saved initialization; discard these weights.
- Full deterministic sample plan is saved and shared by both arms.
- Official generated 6+ training splits are retained only as unused artifacts.
- OOD interventions: fixed200 eligible test IDs per D and k, seed42; identical
  IDs/CF entities/noise directions across arms. Train-only prototypes, min10.
- Patch raw state only after recurrence2, or4 for D>=8; K=D and min(D+4,24).
- Noise: isotropic Gaussian replacement, norm equal to the CF prototype.
- Same-state preservation must exceed50% in the originally-correct subgroup;
  report actual rates rather than calling this threshold strong evidence.
- Mechanism reference gate: grounded CF rate at least20pp above baseline and
  its own noise at K=D, plus majority same-state preservation and a structured
  CF trajectory advantage over noise. Utility reference: matched-K OOD>=5pp,
  or terminal stability>=5pp, or demonstrably deeper useful performance.
- No method changes after outcomes. A negative result is reported as such.

Run on the server (activated Torch2.8/cu128 environment):
`python data_pipeline.py` then `python run_experiment.py`.
Dependencies: torch2.8.0+cu128, transformers4.44.2, numpy, tqdm, matplotlib,
scipy. `run_server.cmd` sets one CPU thread and writes `execution.log`.
An SSH connection must remain alive; the driver saves resumable checkpoints
every250 formal updates. Interrupted updates are replayed from the checkpoint
with the identical sample plan; resumed CSV suffixes must be reconciled first.

Accuracy uses the full vocabulary argmax, without restricting entity candidates.
Evaluation may reuse the prefix of one 24-recurrence deterministic forward for
the K sweep: there is no dropout, halting or K conditioning, so this is exactly
the same computation. Raw correct/total counts and rollout IDs are saved.
No validation-based sample selection, model selection, or hypothesis rescue.

Prototype means are empirical train-state averages; averaging does not guarantee
they lie on the nonlinear representation manifold. Same-state/noise controls
measure this limitation, and causal failure must not be replaced by probe claims.

Checkpoints and full data remain in the isolated server D: directory. Large
files are excluded from ordinary Git; delivery manifest records hashes/sizes.
`FINAL_REPORT.md` contains the actual outcome once the run completes.
