# BTTA-DG provenance and Stage 0 limitation

The evaluator imports the original classes directly from the official
`luo-huan-123/BTTA-DG` master snapshot
`5932d026bbd8a7de106d31a6d264f4f4924537e4`:

- `SincAdaptNet.py` (`32205021…d5db1a`)
- `BTTA_DG.py` (`ce03a1d4…931af5`)
- `pretrain_SincAdaptNet.py` (`1948ce92…0db42c5`)

The source-default BNCI2014001 adaptation parameters are retained: GMM
components `6`, confidence threshold `0.596`, and entropy threshold `0.673`.
The source SincAdaptNet architecture is instantiated with 22 channels, 2
classes, 16 spatial components, 24 Sinc filters, kernel 51, and 250 Hz.

There is a material behavior in the official online loop: it invokes
`OnlineClustererGMM.update(...)` but does **not** invoke `add_sample(...)`.
Consequently its buffers are never populated and the calibrated prediction
remains the source probability. Stage 0 preserves this exact source order
rather than silently repairing or replacing BTTA-DG. The equality of
source-only and BTTA-DG metrics is expected from that literal code path, not
evidence that adaptation succeeded.

The A01 paper-compatible accuracy is not comparable to the published 9-subject
86.50 ± 2.49% mean. A full seed-0, 9-subject reproduction check is required
before any BTTA-DG versus NeuroLoop efficacy claim.
