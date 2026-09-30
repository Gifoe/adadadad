# BNCI2014001 A01 data audit

| Check | Observed Stage 0 value |
|---|---|
| Raw source | `A01T.gdf` |
| Raw target | `A01E.gdf` |
| Official scoring labels | competition-IV `results/ds2a/true_labels.zip` |
| Cache tensor | 288 x 22 x 1000 float32 |
| S1/S2 selected trials | 144 / 144 |
| Per-session class counts | 72 left, 72 right |
| Sampling rate | 250 Hz |
| Trial duration | 4 seconds |
| Preprocessing | common FIR 1–48 Hz, then [cue, cue+4s) |
| EEG channels | 22 original GDF EEG channels, unchanged order |
| Excluded channels | EOG-left, EOG-central, EOG-right |

The cache carries original session, cue sample, run marker, event code, and
original trial index in `manifest.csv`. Target rows state `WITHHELD` in their
public label column. Target scoring labels live in a physically separate NumPy
file whose opening occurs only after model checkpoints are written.

The generated cache audit records SHA-256 hashes for both GDF files, both
official label files, signals, and source/scoring-label arrays.
