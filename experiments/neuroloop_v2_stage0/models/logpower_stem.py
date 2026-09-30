"""Log-power EEG stem used by the state-free NeuroLoop v2 Stage-0 test."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _inverse_softplus(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value))


class LogPowerSincStem(nn.Module):
    """Sinc -> band spatial filters -> power -> local pooling -> log-power."""

    def __init__(
        self,
        channels: int = 22,
        bands: int = 24,
        spatial_components: int = 8,
        d_model: int = 128,
        sample_rate: int = 250,
        filter_length: int = 129,
        patch_size: int = 50,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if filter_length % 2 == 0:
            raise ValueError("filter_length must be odd")
        self.channels = channels
        self.bands = bands
        self.spatial_components = spatial_components
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.patch_size = patch_size
        self.epsilon = epsilon
        initial_low = torch.linspace(2.0, 35.0, bands)
        initial_bandwidth = torch.full((bands,), 6.0)
        self.low_parameter = nn.Parameter(_inverse_softplus(initial_low - 0.5))
        self.bandwidth_parameter = nn.Parameter(_inverse_softplus(initial_bandwidth - 1.0))
        self.band_spatial = nn.Parameter(torch.empty(bands, channels, spatial_components))
        self.projection = nn.Parameter(torch.empty(bands, spatial_components, d_model))
        self.projection_bias = nn.Parameter(torch.zeros(bands, d_model))
        nn.init.xavier_uniform_(self.band_spatial)
        nn.init.xavier_uniform_(self.projection)
        self.token_norm = nn.LayerNorm(d_model)
        self.register_buffer("time", torch.arange(-(filter_length // 2), filter_length // 2 + 1, dtype=torch.float32))
        self.register_buffer("window", torch.hamming_window(filter_length, periodic=False))

    def band_edges_hz(self) -> tuple[Tensor, Tensor]:
        low = (F.softplus(self.low_parameter) + 0.5).clamp(0.5, 45.0)
        high = (low + F.softplus(self.bandwidth_parameter) + 1.0).clamp(max=49.0)
        return low, high

    def _filters(self) -> Tensor:
        low, high = self.band_edges_hz()
        time = self.time.to(device=low.device, dtype=low.dtype) / self.sample_rate
        response = 2.0 * high[:, None] * torch.sinc(2.0 * high[:, None] * time)
        response -= 2.0 * low[:, None] * torch.sinc(2.0 * low[:, None] * time)
        return (response * self.window.to(device=low.device, dtype=low.dtype))[:, None, :]

    def filtered_waveforms(self, x: Tensor) -> Tensor:
        """Return band x channel x time waveforms, before spatial/power stages."""
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"expected (batch,{self.channels},time), got {tuple(x.shape)}")
        batch, channels, length = x.shape
        filtered = F.conv1d(x.reshape(batch * channels, 1, length), self._filters(), padding=self.filter_length // 2)
        return filtered.reshape(batch, channels, self.bands, length).permute(0, 2, 1, 3)

    def power_patches(self, x: Tensor) -> Tensor:
        """Return non-negative B x bands x spatial-components x patches power."""
        filtered = self.filtered_waveforms(x)
        spatial = torch.einsum("nbct,bcr->nbrt", filtered, self.band_spatial)
        power = spatial.square()
        batch, bands, spatial_components, length = power.shape
        return F.avg_pool1d(power.reshape(batch * bands * spatial_components, 1, length), self.patch_size, self.patch_size).reshape(batch, bands, spatial_components, -1)

    def signed_mean_patches(self, x: Tensor) -> Tensor:
        """Diagnostic-only old behavior: direct averaging of signed band signals."""
        filtered = self.filtered_waveforms(x)
        batch, bands, channels, length = filtered.shape
        return F.avg_pool1d(filtered.reshape(batch * bands * channels, 1, length), self.patch_size, self.patch_size).reshape(batch, bands, channels, -1)

    def forward(self, x: Tensor) -> Tensor:
        power = self.power_patches(x)
        log_power = torch.log(power + self.epsilon).permute(0, 1, 3, 2)
        tokens = torch.einsum("nbpr,brd->nbpd", log_power, self.projection) + self.projection_bias[None, :, None, :]
        return self.token_norm(tokens)
