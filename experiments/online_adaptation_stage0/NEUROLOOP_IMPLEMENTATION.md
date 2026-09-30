# NeuroLoop implementation audit

The final Stage 0 model is `NeuroLoopSpec(steps=4, stateful=True, untied=False)`
and has **439,490** trainable parameters. It is below the one-million-parameter
ceiling.

| Required component | Implementation |
|---|---|
| Frequency stem | 24 learnable Sinc band-pass filters |
| Spatial structure | Separate 22-to-128 projection for each band |
| Patches | 50 samples/patch, yielding 20 patches at 250 Hz / 4 s |
| Attention | Per-band temporal attention, then per-patch cross-frequency attention, then state cross-attention |
| State | 25 x 128 tokens: global plus 24 band tokens; initialized from learnable source M0 |
| Loop | One shared F-theta used K times, with an 8-way iteration embedding |
| Anti-drift | Learned alpha/beta evidence reinjection from H0 each loop |
| Head | Mean pool then Linear(128, classes) |
| Update | Entropy-quality-gated interpolation between M_(t-1) and global/band proposals |

The update receives final features and predicted probabilities only to compute
entropy; it never receives a predicted class ID or target label. `predict()` is
read-only and `update()` is the sole state mutation point.

Controls are static K=1, static K=4, shared-state K=1/2/4/8, and an untied K=4
control. The untied control uses a 16-wide FFN per independent block to stay
below 1M parameters (922,242 parameters); it is a capacity-constrained control,
not a parameter-identical replacement for the 439,490-parameter shared model.
