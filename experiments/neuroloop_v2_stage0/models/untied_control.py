"""Untied ordinary-depth control for the v2 loop experiment."""

from __future__ import annotations

from torch import nn

from models.neuroloop_v2 import EEGRefinementBlock, NeuroLoopV2Spec, _BaseNeuroLoopV2


class UntiedDepthControl(_BaseNeuroLoopV2):
    """F1 -> F2 -> F3 -> F4 with same stem/head, but no cross-loop tying."""

    def __init__(self, spec: NeuroLoopV2Spec = NeuroLoopV2Spec()) -> None:
        super().__init__(spec)
        self.blocks = nn.ModuleList([
            EEGRefinementBlock(spec.d_model, spec.heads, spec.ffn_dim, spec.dropout)
            for _ in range(spec.max_loops)
        ])

    def block_for(self, iteration: int) -> EEGRefinementBlock:
        return self.blocks[iteration]
