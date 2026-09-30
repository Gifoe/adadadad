# NeuroLoop Stage-0 v2 protocol

## Scope

This is one pre-registered falsification experiment: BNCI2014001 / BCIC-IV-2a subject A01, S1 (`A01T`) to S2 (`A01E`), left versus right hand, seed 0.  It is not an online-adaptation benchmark and it makes no claim about BTTA-DG efficacy.  It does not run other subjects, seeds, datasets, state mechanisms, or hyperparameter searches.

## Data and information boundary

The shared cache contains 288 filtered `(22, 1000)` trials: 144 S1 source trials and 144 chronological S2 target trials.  The cache preprocessing is fixed at 1--48 Hz, 22 EEG channels, 250 Hz, and 0--4 s.  A run-level split of S1 provides 120 source-training and 24 source-validation trials.  Model selection uses only that source validation split; the selected epoch is then refit from seed 0 on all 144 source trials.  Checkpoint hashes are written before `target_labels_scoring_only.npy` is opened.

S2 is evaluated under two explicitly separate EA routes: `paper` (all-target covariance) and `strict` (causal running covariance in the given target order).  Strict EA is primary.  Neither route has persistent model state: every target trial is passed independently, with no cross-trial, session, or adaptive memory.

## Stem and model

The input is transformed as learnable Sinc band filters -> band-specific spatial filters (`B=24`, spatial rank `8`) -> square energy -> non-overlapping 50-sample / 0.2-s local average -> `log(power + 1e-6)` -> 128-dimensional tokens.  It never averages signed raw waveforms as its representation.

The classifier uses a shared parameter-tied transformer refinement block across four within-trial loops.  Each loop takes the current latent and H0 anchor plus an iteration embedding, emits a delta, and updates `H(k+1)=H(k)+gamma(k)*delta`, where each learned gamma starts at 0.05.  Temporal attention is factorized within each band and frequency attention is factorized within each patch.  The model has 4 heads, 128-dimensional tokens, FFN width 512, dropout 0.1, a shared pool/classifier, and no recurrence across trials.

Training uses AdamW (`3e-4`, weight decay `5e-4`), CE, gradient clipping 5, 40 epochs, and pre-registered deep-supervision weights `[0.125, 0.125, 0.25, 0.5]`.  The tied Kmax=4 checkpoint alone supplies the output at loops 1--4.  No standalone K1/K2/K3/K4 checkpoints are trained.

## Comparators and gate

Controls are source-only official SincAdaptNet and an untied four-block control with the same stem/token size.  The only trajectory claim is supported if, on strict S2 from the same tied checkpoint, both `Acc@4 > Acc@1` and `P(wrong@1, correct@4) > P(correct@1, wrong@4)` hold.  Otherwise this stage is a NO-GO and further expansion is prohibited by this protocol.
