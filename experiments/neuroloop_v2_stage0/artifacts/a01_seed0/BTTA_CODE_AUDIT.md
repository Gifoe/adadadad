# BTTA-DG code audit

## Examined source

The examined official snapshot is `/root/BTTA-DG-official` at the revision recorded by the surrounding repository metadata.  Its online route calls `clusterer.update(alpha, pred, prob_map)` but does not call `clusterer.add_sample(alpha, pred, prob_map)` before that update.

## Consequence

The clustering/GMM buffer that the paper algorithm requires is not populated along this code path.  In addition, the local prior runner did not invoke a calibrated `predict_class()` posterior route.  Therefore the previous 80.56% number is only a source-only SincAdaptNet sanity result; it is not evidence for functioning BTTA-DG online adaptation.

## Scope decision

Stage-0 v2 excludes BTTA-DG from efficacy comparisons.  It keeps Source-only SincAdaptNet as a fixed sanity baseline and evaluates only within-trial shared refinement.  A future repair must be explicitly separated into `BTTA-DG-official-code` and `BTTA-DG-paper-faithful`; silently adding the missing buffer write must not be called an official-code reproduction.
