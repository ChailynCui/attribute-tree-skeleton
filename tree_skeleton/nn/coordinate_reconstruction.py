# SPDX-License-Identifier: MIT
"""Decode quantized branch endpoints into coordinate distributions."""

from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from local_attention import LocalMHA
from x_transformers.x_transformers import FeedForward, RMSNorm

from tree_skeleton._tensor_ops import is_odd
from .layers import SegmentResidualBlock


class CoordinateDecoder(nn.Module):
    def __init__(
        self, *, dim_codebook, attn_decoder_depth, local_attn_kwargs,
        local_attn_window_size, attn_dropout, ff_dropout,
        decoder_dims_through_depth, init_decoder_conv_kernel,
        resnet_dropout, num_discrete_coors,
    ):
        super().__init__()
        input_dim = dim_codebook * 2
        self.attention = nn.ModuleList(
            nn.ModuleList([
                LocalMHA(
                    dim=input_dim, causal=False, prenorm=True,
                    dropout=attn_dropout, window_size=local_attn_window_size,
                    **local_attn_kwargs,
                ),
                nn.Sequential(
                    RMSNorm(input_dim), FeedForward(input_dim, glu=True, dropout=ff_dropout)
                ),
            ])
            for _ in range(attn_decoder_depth)
        )
        first_dim, *conv_dims = decoder_dims_through_depth
        assert is_odd(init_decoder_conv_kernel)
        self.input_convolution = nn.Sequential(
            nn.Conv1d(input_dim, first_dim, kernel_size=init_decoder_conv_kernel,
                      padding=init_decoder_conv_kernel // 2),
            nn.SiLU(),
            Rearrange("b c n -> b n c"),
            nn.LayerNorm(first_dim),
            Rearrange("b n c -> b c n"),
        )
        self.residual_layers = nn.ModuleList()
        current_dim = first_dim
        for next_dim in conv_dims:
            self.residual_layers.append(
                SegmentResidualBlock(current_dim, next_dim, dropout=resnet_dropout)
            )
            current_dim = next_dim
        self.readout = nn.Sequential(
            nn.Linear(current_dim, num_discrete_coors * 6),
            Rearrange("... (v c) -> ... v c", v=6),
        )

    def forward(self, quantized, segment_mask):
        conv_mask = rearrange(segment_mask, "b n -> b 1 n")
        features = quantized
        for attention, feed_forward in self.attention:
            features = attention(features, mask=segment_mask) + features
            features = feed_forward(features) + features
        features = rearrange(features, "b n d -> b d n")
        features = self.input_convolution(features.masked_fill(~conv_mask, 0.0))
        for layer in self.residual_layers:
            features = layer(features, mask=conv_mask)
        return rearrange(features, "b d n -> b n d")
