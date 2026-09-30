"""Specified NeuroLoop architecture for the online-adaptation benchmark.

The model separates a read-only prediction of trial ``t`` from the state update
performed after its prediction. It has no predicted-label input anywhere in the
memory update path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SincBandStem(nn.Module):
    """B learnable band-pass filters producing band x channel x patch features."""

    def __init__(self, channels: int, bands: int = 24, kernel_size: int = 129, patch_size: int = 50):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.channels, self.bands, self.kernel_size, self.patch_size = channels, bands, kernel_size, patch_size
        self.low_hz = nn.Parameter(torch.linspace(1.0, 34.0, bands))
        self.bandwidth_hz = nn.Parameter(torch.full((bands,), 6.0))
        self.register_buffer("time", torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32))
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def _filters(self, sample_rate: float) -> Tensor:
        low = F.softplus(self.low_hz).clamp(0.5, 45.0)
        high = (low + F.softplus(self.bandwidth_hz) + 1.0).clamp(max=49.0)
        time = self.time.to(device=low.device, dtype=low.dtype) / sample_rate
        band = 2.0 * high[:, None] * torch.sinc(2.0 * high[:, None] * time)
        band -= 2.0 * low[:, None] * torch.sinc(2.0 * low[:, None] * time)
        return (band * self.window.to(device=low.device, dtype=low.dtype))[:, None, :]

    def forward(self, x: Tensor, sample_rate: int) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"expected (batch,{self.channels},time), got {tuple(x.shape)}")
        batch, channels, length = x.shape
        z = F.conv1d(x.reshape(batch * channels, 1, length), self._filters(sample_rate), padding=self.kernel_size // 2)
        z = z.reshape(batch, channels, self.bands, length).permute(0, 2, 1, 3)
        # P=20 at 250 Hz / 4 s: do not feed all raw samples to attention.
        return F.avg_pool1d(z.reshape(batch * self.bands * channels, 1, length), self.patch_size, self.patch_size).reshape(batch, self.bands, channels, -1)


class FactorizedLoopBlock(nn.Module):
    """One shared F_theta: temporal -> cross-frequency -> state cross-attention."""

    def __init__(self, d_model: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.temporal = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_frequency = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.state_cross_attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_temporal = nn.LayerNorm(d_model)
        self.norm_frequency = nn.LayerNorm(d_model)
        self.norm_state = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, memory: Tensor, iteration: Tensor) -> Tensor:
        batch, bands, patches, d_model = h.shape
        temporal_input = h.reshape(batch * bands, patches, d_model)
        temporal, _ = self.temporal(temporal_input, temporal_input, temporal_input, need_weights=False)
        h = self.norm_temporal(h + self.dropout(temporal.reshape(batch, bands, patches, d_model)))
        frequency_input = h.permute(0, 2, 1, 3).reshape(batch * patches, bands, d_model)
        frequency, _ = self.cross_frequency(frequency_input, frequency_input, frequency_input, need_weights=False)
        h = self.norm_frequency(h + self.dropout(frequency.reshape(batch, patches, bands, d_model).permute(0, 2, 1, 3)))
        flat = h.reshape(batch, bands * patches, d_model)
        context, _ = self.state_cross_attention(flat, memory, memory, need_weights=False)
        h = self.norm_state(h + self.dropout(context.reshape(batch, bands, patches, d_model)) + iteration[:, None, None, :])
        return self.norm_ffn(h + self.dropout(self.ffn(h)))


@dataclass(frozen=True)
class NeuroLoopSpec:
    channels: int = 22
    classes: int = 2
    sample_rate: int = 250
    bands: int = 24
    d_model: int = 128
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    steps: int = 4
    stateful: bool = True
    untied: bool = False


class NeuroLoop(nn.Module):
    """Final compact EEG-specific loop architecture (default shared K=4)."""

    def __init__(self, spec: NeuroLoopSpec):
        super().__init__()
        if not 1 <= spec.steps <= 8:
            raise ValueError("only K in [1,8] is supported")
        self.spec = spec
        self.stem = SincBandStem(spec.channels, spec.bands)
        # Each frequency band owns a C -> d spatial projection.
        self.band_spatial = nn.Parameter(torch.empty(spec.bands, spec.channels, spec.d_model))
        nn.init.xavier_uniform_(self.band_spatial)
        self.spatial_bias = nn.Parameter(torch.zeros(spec.bands, spec.d_model))
        self.input_norm = nn.LayerNorm(spec.d_model)
        self.iteration_embedding = nn.Embedding(8, spec.d_model)
        self.source_memory_initialization = nn.Parameter(torch.zeros(spec.bands + 1, spec.d_model))
        nn.init.trunc_normal_(self.source_memory_initialization, std=0.02)
        if spec.untied:
            # Capacity control for the explicitly non-final untied ablation.
            self.loop_blocks = nn.ModuleList([FactorizedLoopBlock(spec.d_model, spec.heads, 16, spec.dropout) for _ in range(spec.steps)])
        else:
            self.shared_loop_block = FactorizedLoopBlock(spec.d_model, spec.heads, spec.ffn_dim, spec.dropout)
        self.alpha = nn.Parameter(torch.zeros(8))
        self.beta = nn.Parameter(torch.zeros(8))
        self.reinjection_norm = nn.LayerNorm(spec.d_model)
        self.state_gate = nn.Linear(2 * spec.d_model, spec.d_model)
        self.classifier = nn.Linear(spec.d_model, spec.classes)
        self.dropout = nn.Dropout(spec.dropout)

    @property
    def stateful(self) -> bool:
        return self.spec.stateful

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        state = self.source_memory_initialization.unsqueeze(0).expand(batch_size, -1, -1)
        return state.to(device=device or self.source_memory_initialization.device).clone()

    def encode_evidence(self, x: Tensor) -> Tensor:
        band_channel_patch = self.stem(x, self.spec.sample_rate)  # N,B,C,P
        # N,B,C,P x B,C,D -> N,B,P,D, explicit band-specific spatial organization.
        h0 = torch.einsum("nbcp,bcd->nbpd", band_channel_patch, self.band_spatial)
        return self.input_norm(h0 + self.spatial_bias[None, :, None, :])

    def _block(self, iteration: int) -> FactorizedLoopBlock:
        return self.loop_blocks[iteration] if self.spec.untied else self.shared_loop_block

    def forward_with_state(self, x: Tensor, persistent_state: Tensor | None = None, return_intermediates: bool = False):
        """Read M_(t-1), compute logits for t, and leave M unchanged."""
        h0 = self.encode_evidence(x)
        memory = persistent_state if self.spec.stateful and persistent_state is not None else self.initial_state(x.shape[0], x.device)
        h = h0
        per_loop_logits: list[Tensor] = []
        for iteration in range(self.spec.steps):
            iteration_vector = self.iteration_embedding(torch.full((x.shape[0],), iteration, device=x.device, dtype=torch.long))
            refined = self._block(iteration)(h, memory, iteration_vector)
            h = self.reinjection_norm(h + torch.sigmoid(self.alpha[iteration]) * refined + torch.sigmoid(self.beta[iteration]) * h0)
            per_loop_logits.append(self.classifier(self.dropout(h.mean(dim=(1, 2)))))
        logits = per_loop_logits[-1]
        return (logits, h, per_loop_logits) if return_intermediates else (logits, h)

    def update_state(self, final_h: Tensor, persistent_state: Tensor, probabilities: Tensor) -> Tensor:
        """Gated unlabeled state update after prediction; no class ID enters it."""
        if not self.spec.stateful:
            return persistent_state
        global_token = final_h.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        band_tokens = final_h.mean(dim=2)
        proposal = torch.cat((global_token, band_tokens), dim=1)
        gate = torch.sigmoid(self.state_gate(torch.cat((persistent_state, proposal), dim=-1)))
        entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        quality = (1.0 - entropy / torch.log(torch.tensor(float(self.spec.classes), device=entropy.device))).detach().clamp(0.0, 1.0)
        rho = quality[:, None, :] * gate
        return (1.0 - rho) * persistent_state + rho * proposal
