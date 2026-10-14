# Sources and evidence boundaries

Checked 2026-10-15. These sources describe prior methods; their reported metrics
are not results of this audit. No claim of literature exhaustiveness is made.

- [Official TaH code](https://github.com/thu-nics/TaH/tree/78ddf9a2c25713fced03a736c5746161f6ee590c): native TaH2 implementation, DUO cache access and probability-mixture readout. Exact checkpoint settings take precedence over defaults.
- [TaH2 paper](https://arxiv.org/abs/2609.35748): token-wise extra depth supervised by lookahead prediction improvement. Generic adaptive looping is already an established baseline.
- [NESTFUL](https://github.com/IBM/NESTFUL/tree/fc2c4123e73500a56185a5fb354f05d1c8b4890c): structured reference calls and executable functions. This audit uses the publicly available math subset, with original tools and verified execution. The reference contains variable bindings rather than a transcript of numeric tool returns.
- [RecurTrace](https://arxiv.org/abs/2609.03379): loop-time memory and a loss-reduction-supervised halting head. Adding memory plus local stopping alone would substantially overlap this work.
- [Trajectory/readout diagnosis](https://arxiv.org/abs/2607.20519): separates recurrent trajectory formation from exit readout and compares learned gates with post-hoc readouts. A claim about gate/readout separation needs a more specific contribution than this existing distinction.
- [Continuous depth batching](https://arxiv.org/abs/2608.09444): token exit scheduling and recurrent KV management. Hardware latency is a separate question from counting loops.
- [FlashLoop](https://arxiv.org/abs/2609.29812): token-sparse recurrent updates, sparse attention and quantized cross-loop KV residuals. Generic lazy updates or KV savings are not an unoccupied research direction.
- [MELT](https://arxiv.org/abs/2605.07721): shared KV storage updated by a learned gate, adapted through post-training. Memory/compute decoupling alone is already studied; this is distinct from asking whether local prediction utility predicts future semantic use.

Upstream TaH and NESTFUL repositories provide Apache-2.0 licenses. The generated
controlled-symbol task is a new diagnostic dataset, not an established benchmark.
Future-method candidates are kept separately from the public reproduction files.
