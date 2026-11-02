# SPDX-License-Identifier: MIT
"""Masked segment convolutions and recurrent context updates."""

import torch
from torch import nn
from einops.layers.torch import Rearrange
from gateloop_transformer import SimpleGateLoopLayer


class MaskedChannelGate(nn.Module):
    """Gate feature channels using only the valid segment positions."""

    def __init__(self, dim, reduction_factor=4, min_dim=16):
        super().__init__()
        hidden = max(dim // reduction_factor, min_dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1"),
        )

    def forward(self, features, mask=None):
        if mask is None:
            summary = features.mean(dim=-1)
        else:
            features = features.masked_fill(~mask, 0.0)
            count = mask.float().sum(dim=-1).clamp(min=1e-5)
            summary = features.sum(dim=-1) / count
        return features * self.net(summary)


class SegmentConvolution(nn.Module):
    """Project a segment sequence with padding masked before and after projection."""

    def __init__(self, dim, dim_out=None, groups=8, dropout=0.0):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, features, mask=None):
        if mask is not None:
            features = features.masked_fill(~mask, 0.0)
        projected = self.proj(features)
        if mask is not None:
            projected = projected.masked_fill(~mask, 0.0)
        return self.dropout(self.act(self.norm(projected)))


class SegmentResidualBlock(nn.Module):
    """Refine endpoint features through two convolutions and a channel gate."""

    def __init__(self, dim, dim_out=None, *, groups=8, dropout=0.0):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.block1 = SegmentConvolution(dim, dim_out, groups=groups, dropout=dropout)
        self.block2 = SegmentConvolution(dim_out, dim_out, groups=groups, dropout=dropout)
        self.excite = MaskedChannelGate(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, features, mask=None):
        residual = self.residual_conv(features)
        refined = self.block1(features, mask=mask)
        refined = self.block2(refined, mask=mask)
        return self.excite(refined, mask=mask) + residual


class RecurrentSegmentContext(nn.Module):
    """Update sequence context, reusing the recurrent state when sampling."""

    def __init__(self, dim, *, depth, use_heinsen=True):
        super().__init__()
        self.gateloops = nn.ModuleList(
            SimpleGateLoopLayer(dim=dim, use_heinsen=use_heinsen) for _ in range(depth)
        )

    def forward(self, features, cache=None):
        if features.numel() == 0:
            return features, None
        prefix = None
        if cache is not None:
            prefix, features = features[:, :-1], features[:, -1:]
        previous_states = iter(cache if cache is not None else [])
        next_states = []
        for layer in self.gateloops:
            update, state = layer(
                features, cache=next(previous_states, None), return_cache=True
            )
            next_states.append(state)
            features = features + update
        if prefix is not None:
            features = torch.cat((prefix, features), dim=-2)
        return features, next_states
