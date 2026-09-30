# EEG online-adaptation benchmark protocol — Stage 0 freeze

## Scope and gate

This freeze runs exactly one complete stream: BNCI2014001 / BCI Competition IV
2a subject `A01`, source `A01T` (S1) and chronological target `A01E` (S2),
seed `0`. It is a plumbing and reproducibility gate, not a 9-subject result.
No other subject, dataset, or seed may be launched from this stage.

## Shared input and labels

- Retain only original left/right trials and the original 22 EEG channels in
  GDF order. The three EOG channels are excluded.
- Apply a common FIR 1–48 Hz filter before 0–4 s epoch extraction at 250 Hz.
- Both methods consume the same immutable `(144, 22, 1000)` target tensor in
  original cue order, with batch size one.
- `source_labels.npy` is opened for S1 training/selection only.
  `target_labels_scoring_only.npy` is a separate file and is not opened until
  every model checkpoint has been written.

## Training and selection

S1 is split by the final marked run: 120 source-train and 24 source-validation
left/right trials. Selection uses source-validation balanced accuracy, choosing
the earliest tied epoch. Each selected configuration is then refit on all 144
S1 trials before any S2 scoring label is opened. SincAdaptNet preserves the
official pretraining optimizer (`Adam`, 1e-3, weight decay 1e-4); NeuroLoop uses
AdamW 3e-4, weight decay 5e-4, gradient clip 5.0, and chronological 16-trial
TBPTT chunks. Labels occur only in cross-entropy.

## Online semantics

Every runner is called as `predict(trial_t)` followed by `update()`. The state
resets to learnable source M0 at a new session. No target label, predicted class
ID, or gradient is available to either update. Two EA modes are reported:

- **paper**: target-session EA uses all target signals and is explicitly
  non-causal/full-session;
- **strict**: the EA transform for trial t uses only x1 through xt, including
  the current unlabeled signal but never a future signal.

The primary metric is accuracy; BA, macro-F1, NLL, and ECE are secondary.
