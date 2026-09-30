"""State-free, parameter-tied within-trial NeuroLoop v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from models.logpower_stem import LogPowerSincStem


@dataclass(frozen=True)
class NeuroLoopV2Spec:
    channels: int = 22
    classes: int = 2
    sample_rate: int = 250
    bands: int = 24
    spatial_components: int = 8
    d_model: int = 128
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    max_loops: int = 4
    patch_size: int = 50
    gamma_init: float = 0.05


class EEGRefinementBlock(nn.Module):
    """Shared F_theta: temporal, cross-frequency, H0-anchor, then FFN."""

    def __init__(self, d_model: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.temporal = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.frequency = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.anchor = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_temporal = nn.LayerNorm(d_model)
        self.norm_frequency = nn.LayerNorm(d_model)
        self.norm_anchor = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, h0: Tensor, iteration_embedding: Tensor) -> Tensor:
        """Emit a refinement delta without directly overwriting the representation."""
        batch, bands, patches, d_model = h.shape
        x = self.input_norm(h)
        temporal_input = x.reshape(batch * bands, patches, d_model)
        temporal, _ = self.temporal(temporal_input, temporal_input, temporal_input, need_weights=False)
        x = self.norm_temporal(x + self.dropout(temporal.reshape(batch, bands, patches, d_model)))
        frequency_input = x.permute(0, 2, 1, 3).reshape(batch * patches, bands, d_model)
        frequency, _ = self.frequency(frequency_input, frequency_input, frequency_input, need_weights=False)
        x = self.norm_frequency(x + self.dropout(frequency.reshape(batch, patches, bands, d_model).permute(0, 2, 1, 3)))
        query = x.reshape(batch, bands * patches, d_model)
        anchor = h0.reshape(batch, bands * patches, d_model)
        anchored, _ = self.anchor(query, anchor, anchor, need_weights=False)
        x = self.norm_anchor(x + self.dropout(anchored.reshape(batch, bands, patches, d_model)) + iteration_embedding[:, None, None, :])
        return self.norm_ffn(x + self.dropout(self.ffn(x)))


class _BaseNeuroLoopV2(nn.Module):
    def __init__(self, spec: NeuroLoopV2Spec) -> None:
        super().__init__()
        self.spec = spec
        self.stem = LogPowerSincStem(
            channels=spec.channels,
            bands=spec.bands,
            spatial_components=spec.spatial_components,
            d_model=spec.d_model,
            sample_rate=spec.sample_rate,
            patch_size=spec.patch_size,
        )
        self.iteration_embedding = nn.Embedding(spec.max_loops, spec.d_model)
        self.gamma = nn.Parameter(torch.full((spec.max_loops,), spec.gamma_init))
        self.classifier = nn.Linear(spec.d_model, spec.classes)
        self.dropout = nn.Dropout(spec.dropout)

    def block_for(self, iteration: int) -> EEGRefinementBlock:
        raise NotImplementedError

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def architecture_audit(self) -> dict:
        return {"spec": asdict(self.spec), "trainable_parameters": self.parameter_count(), "state_free": True}

    def forward(self, x: Tensor, return_features: bool = False):
        h0 = self.stem(x)
        h = h0
        logits: list[Tensor] = []
        features: list[Tensor] = []
        for iteration in range(self.spec.max_loops):
            embedding = self.iteration_embedding(torch.full((x.shape[0],), iteration, dtype=torch.long, device=x.device))
            delta = self.block_for(iteration)(h, h0, embedding)
            # gamma starts at 0.05: each loop is a small residual correction.
            h = h + self.gamma[iteration] * delta
            features.append(h)
            logits.append(self.classifier(self.dropout(h.mean(dim=(1, 2)))))
        return (logits, features) if return_features else logits


class SharedNeuroLoopV2(_BaseNeuroLoopV2):
    """Final v2: exactly one F_theta reused at every refinement step."""

    def __init__(self, spec: NeuroLoopV2Spec = NeuroLoopV2Spec()) -> None:
        super().__init__(spec)
        self.shared_block = EEGRefinementBlock(spec.d_model, spec.heads, spec.ffn_dim, spec.dropout)

    def block_for(self, iteration: int) -> EEGRefinementBlock:
        return self.shared_block
