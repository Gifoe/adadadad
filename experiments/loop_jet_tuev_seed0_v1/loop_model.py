"""Loop-JET: unchanged official parameters and endpoint/loss semantics."""
import pathlib
import sys

import torch
from torch.utils.checkpoint import checkpoint

OFFICIAL = pathlib.Path(__file__).resolve().parent / 'official'
sys.path.insert(0, str(OFFICIAL))
from denoiser import Denoiser
from models.raw_vit import RawViTDiffusion


class LoopRawViTDiffusion(RawViTDiffusion):
    def __init__(self, *args, loop_count=2, activation_checkpointing=False, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.blocks) != 12:
            raise ValueError('This experiment is frozen to the official 12-block JiT-B/16.')
        if loop_count not in (1, 2):
            raise ValueError('Only K=1 equivalence and formal K=2 are permitted.')
        self.loop_count = loop_count
        self.activation_checkpointing = activation_checkpointing

    def _blocks(self, feats, cond, start, stop):
        for i in range(start, stop):
            block = self.blocks[i]  # Same module/parameters at every recurrence.
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                feats = checkpoint(block, feats, cond, use_reentrant=False)
            else:
                feats = block(feats, cond)
        return feats

    def forward_endpoints(self, x, t, y, decode_intermediate=True):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.output_shape:
            raise ValueError(f'Expected [B,{self.output_shape}], received {tuple(x.shape)}')
        b, c, p, d = x.shape
        cond = (self.t_embedder(t) + self.y_embedder(y)).repeat_interleave(c, dim=0)
        feats = self.patch_embed(x).view(b*c, p, self.hidden_size) + self.pos_embed
        h = self._blocks(feats, cond, 0, 4)
        endpoints = []
        for k in range(self.loop_count):
            h = self._blocks(h, cond, 4, 8)
            if decode_intermediate or k == self.loop_count-1:
                post = self._blocks(h, cond, 8, 12)
                endpoint = self.final_layer(post, cond).view(b, c, p, d)
                endpoints.append(endpoint)
        return tuple(endpoints)

    def forward(self, x, t, y):
        return self.forward_endpoints(x, t, y, decode_intermediate=False)[-1]


class LoopDenoiser(Denoiser):
    def __init__(self, args, *, deep_supervised=False, activation_checkpointing=False):
        super().__init__(args)
        # Preserve official seeded initialization AND subsequent RNG state exactly.
        original = self.net
        with torch.random.fork_rng(devices=[]):
            loop = LoopRawViTDiffusion(
                model_name=args.model, num_channels=args.num_eeg_channels,
                patch_size=args.eeg_patch_size, target_length=args.target_length,
                num_classes=args.class_num, attn_dropout=args.attn_dropout,
                proj_dropout=args.proj_dropout, loop_count=2,
                activation_checkpointing=activation_checkpointing,
            )
        loop.load_state_dict(original.state_dict(), strict=True)
        self.net = loop
        self.deep_supervised = deep_supervised
        self.ds_lambda = 1.0 / 3.0

    def endpoint_loss(self, z, t, target, endpoint):
        denominator = (1-t).clamp_min(self.t_eps)
        return self._compute_loss((target-z)/denominator, (endpoint-z)/denominator, target, endpoint)

    def training_losses(self, x, labels):
        x = self.to_training_space(x)
        labels = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(len(x), x.device).view(-1, 1, 1, 1)
        e = self._noise_like(x)
        z = t*x + (1-t)*e
        endpoints = self.net.forward_endpoints(z, t.flatten(), labels, decode_intermediate=self.deep_supervised)
        final = self.endpoint_loss(z, t, x, endpoints[-1])
        intermediate = self.endpoint_loss(z, t, x, endpoints[0]) if self.deep_supervised else None
        objective = (final + self.ds_lambda*intermediate)/(1+self.ds_lambda) if self.deep_supervised else final
        return objective, final, intermediate

    def forward(self, x, labels):
        return self.training_losses(x, labels)[0]

    def loss_components(self, target, endpoint):
        # Call the official implementation; do not substitute whole-trace TV/std.
        return {
            'endpoint_l1': (target-endpoint).abs().mean(),
            'statistics': self._statistics_loss(target, endpoint),
            'tv': self._total_variation_loss(endpoint),
            'correlation_loss': self._pearson_corr_loss(target, endpoint),
        }
