# Competent-checkpoint mechanism experiment — final report

Final decision: **DECODE-ONLY — NO CAUSAL EFFECT**. GO/NO-GO: **NO-GO**.

Official R2 `checkpoint_epoch_2765.pt`, H_train6 (inferred with provenance evidence), original recurrence K2. Official held-out ID macro: **98.88%** across D2–6, 750 chains per depth.

Checkpoint provenance: `CHECKPOINT_PROVENANCE.md`. Fixed method and all gates: `METHOD_DIFF.md`.

## Actual endpoints

| Model | ID K5 | OOD K=D | State e_k macro | CF target | CF trajectory | Noise target | Noise trajectory | Same-state preserve (original correct) | OOD terminal stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained | 36.56% | 0.00% | 3.24% | 0.03% | 0.30% | 0.38% | 0.37% | 0.00% | 0.00% |
| final_only | 98.19% | 0.00% | 5.82% | 0.31% | 0.39% | 0.47% | 0.36% | 0.45% | 0.00% |
| grounded | 78.91% | 0.48% | 37.87% | 0.56% | 0.54% | 0.34% | 0.38% | 0.35% | 0.00% |

## Before and after transition supervision

Pretrained direct intermediate-state macro is 3.24%; CF final-target rate 0.03% and remaining-path accuracy 0.30%. The frozen baseline-already-has-mechanism gate is False. No learned probe is used.

Grounded CF target rate is 0.56%; CF trajectory is 0.54%. Grounded minus pretrained target/trajectory gains are 0.53/0.24pp; gains over final-only are 0.25/0.15pp. Causal gate passes: False.

Transition-state decoding change over pretrained: 34.63pp; over final-only: 32.06pp. The separately reported k<=5 state macros are pretrained 6.26%, final_only 9.39%, grounded 70.95%.

Representation shaping changes causal computation only if the target AND trajectory gates pass. Decodability and final accuracy alone cannot establish causal state use. The DECODE-ONLY decision label denotes failure of this causal claim; a decoding improvement is claimed only when its measured gain is positive.

## ID retention and true depth extrapolation

At fixed K5 grounded ID retention drop is -42.35pp. Damaged-reasoning flag (>5pp): False. No arm was rescued or selected at an earlier checkpoint. Original competence is assessed at official K2, which can differ materially from this required K5 retention reference.

OOD means strictly D8/10/12/16/20 because H_train6. OOD matched-K=D macro is 0.00% / 0.00% / 0.48% for pretrained/final-only/grounded. OOD utility gate (>=5pp over both controls): False. Descriptive sweep peaks/frontiers are saved separately; no peak-selection result substitutes for the fixed matched-K endpoint.

## Overthinking and control validity

OOD terminal stability D+1..24 is 0.00% / 0.00% / 0.00%. Stability utility gate: False. Peak-minus-K24 accuracy and early-correct-to-K24-wrong rates are in `overthinking_metrics.csv`.

Same-state preservation, original retention, norm-matched noise, main and extra recurrence budgets, and originally-correct/incorrect strata are all retained in `intervention_comparison.csv`. Poor same-state preservation means a prototype patch is disruptive; it weakens the clean causal interpretation even if some target-directed effects appear. No metric includes the injected state in the remaining-path accuracy.

## Fairness, runtime and interpretation limits

Both arms start from identical official weights and use identical source records, batch order, seed, optimizer, LR, precision, recurrence and 5,000 updates. Only coefficient0/1 differs. Full smoke runs are discarded. Checksum and endpoint verification is in `completion_audit.json` and `checkpoint_manifest.json`. Train-only model-specific prototypes and shared intervention IDs/noise are verified.

Total recorded GPU workload wall time is 0.3219 hours; the phase durations and peak allocated VRAM are in `runtime_metrics.json`. Audit times include data output and prototype/intervention work; this is occupied workload wall time rather than profiler-isolated GPU kernel time. Downloads and source preparation are excluded.

This is one seed and one official shallow recurrent checkpoint. Prototype replacement tests this carrier intervention, not every possible representation elsewhere in the transformer. H_train is supported by public role and stage-update evidence but is not directly encoded in the checkpoint. Intermediate supervision at K1–4 is not itself evidence of causal computation at later recurrences.

## Final decision rationale

{
  "pretrained_already_has_mechanism": false,
  "causal_supported": false,
  "grounded_ID_damaged": false,
  "ID_retention_drop": -0.42346666666666666,
  "causal_differences": {
    "cf_target_rate": {
      "pretrained": 0.0053125,
      "final_only": 0.0025000000000000005
    },
    "trajectory_accuracy": {
      "pretrained": 0.002356336805555555,
      "final_only": 0.0014903893849206344
    }
  },
  "grounded_CF_minus_noise": {
    "cf_target_rate": 0.0021875000000000006,
    "trajectory_accuracy": 0.0015797371031746024
  },
  "utility": {
    "OOD_KD": false,
    "OOD_terminal_stability": false
  }
}

**DECODE-ONLY — NO CAUSAL EFFECT**. The evidence does not satisfy the full proposed-method GO criterion. The measured scientific result is preserved without changing the method or thresholds.

Native execution issues and process-only CPU affinity mitigation are recorded in `METHOD_DIFF.md`. Failed attempt logs remain preserved. Reported GPU workload hours sum the successful completed phases; prior failed-attempt GPU-active time was not measured and is not silently treated as zero.

Measured supported change categories: **learning representation only**. Positive numerical differences are descriptive at this single seed, not significance claims.

Additional native-setting reference: the original checkpoint scores 98.88% at K2, while grounded scores 78.91% at its trained K5. The requested matched-K5 retention comparison does not by itself prove preservation of the original K2 computation. Both references are exposed rather than conflated.

Grounded ID accuracy is 19.28pp below the matched final-only arm and 19.97pp below the original checkpoint at its native K2. Thus the formal matched-K5 retention flag is not evidence that the original competence was preserved. OOD 0.48% is near the nominal 1/200=0.5% entity chance rate; it is not a useful extrapolation gain.

Same-state replacement on originally correct chains preserves only 0.35% in grounded. This positive-control failure makes the prototype intervention strongly disruptive. The experiment supplies no positive causal evidence; it cannot distinguish absence of a usable carrier from failure of mean-prototype interchangeability, and it cannot rule out mechanisms in other token positions. A separate future audit would need to establish same-state interchangeability before interpreting counterfactual replacement as a clean mechanism test. No such rescue experiment was substituted into this fixed run.

Relative-to-peak K* is descriptive; near-chance OOD peaks do not establish a useful computational frontier.

Publication integrity checks also compare recorded dataset arrays with the canonical selected official data and fingerprint the executed training/instrumentation/intervention bytecode, including the used official constructor/forward. `executed_bytecode_audit.json` records the scope and matches. A damaged README in an earlier transport/archive snapshot was restored from canonical source; core source/data checks matched. Final artifacts are published only after checksum and UTF-8 checks.
