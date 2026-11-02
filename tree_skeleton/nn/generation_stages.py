# SPDX-License-Identifier: MIT
"""Segment context and endpoint prediction with explicit incremental state."""

from typing import Any, NamedTuple

from torch import Tensor, nn
from einops import pack, rearrange
from x_transformers import Decoder

from tree_skeleton._tensor_ops import pad_to_length, safe_cat
from .layers import RecurrentSegmentContext


class GenerationCache(NamedTuple):
    segment_embeddings: Tensor | None = None
    segment_attention: Any = None
    endpoint_attention: Any = None
    segment_recurrence: Any = None
    endpoint_recurrence: Any = None


class SegmentContext(nn.Module):
    """Summarize completed segments and attend to the morphology prefix."""

    def __init__(
        self, *, dim, tokens_per_segment, recurrent_depth, use_heinsen,
        attn_depth, attn_dim_head, attn_heads, flash_attn, dropout, attn_kwargs,
    ):
        super().__init__()
        self.summarize = nn.Sequential(
            nn.Linear(tokens_per_segment * dim, dim), nn.LayerNorm(dim)
        )
        self.recurrence = (
            RecurrentSegmentContext(dim, depth=recurrent_depth, use_heinsen=use_heinsen)
            if recurrent_depth > 0 else None
        )
        self.attention = Decoder(
            dim=dim, depth=attn_depth, attn_dim_head=attn_dim_head,
            heads=attn_heads, attn_flash=flash_attn,
            attn_dropout=dropout, ff_dropout=dropout, **attn_kwargs,
        )

    def forward(
        self, segments, *, prefix=None, condition=None, modulation=None,
        cached_embeddings=None, attention_cache=None, recurrent_cache=None,
    ):
        segment_count = segments.shape[1]
        if cached_embeddings is not None and segment_count <= cached_embeddings.shape[1]:
            return cached_embeddings, attention_cache, recurrent_cache

        features = self.summarize(rearrange(segments, "b n t d -> b n (t d)"))
        prefix_length = 0 if prefix is None else prefix.shape[1]
        if prefix is not None:
            features, _ = pack([prefix, features], "b * d")
        if self.recurrence is not None:
            features, recurrent_cache = self.recurrence(features, cache=recurrent_cache)
        if modulation is not None:
            features = modulation(features, condition)
        attended, attention_cache = self.attention(
            features, cache=attention_cache, return_hiddens=True
        )
        # Only the first call includes prefix outputs. Later calls return the
        # uncached suffix; the stored segment embeddings already omit prefixes.
        if prefix_length and cached_embeddings is None:
            attended = attended[:, prefix_length:]
        attended = safe_cat((cached_embeddings, attended), dim=-2)
        return attended, attention_cache, recurrent_cache


class EndpointDecoder(nn.Module):
    """Predict endpoint codes inside each segment from its causal context."""

    def __init__(
        self, *, dim, dim_fine, recurrent_depth, use_heinsen, attn_depth,
        attn_dim_head, attn_heads, flash_attn, dropout, attn_kwargs, vocabulary_size,
    ):
        super().__init__()
        self.context_projection = nn.Linear(dim, dim_fine) if dim != dim_fine else nn.Identity()
        self.recurrence = (
            RecurrentSegmentContext(dim, depth=recurrent_depth, use_heinsen=use_heinsen)
            if recurrent_depth > 0 else None
        )
        self.attention = Decoder(
            dim=dim_fine, depth=attn_depth, attn_dim_head=attn_dim_head,
            heads=attn_heads, attn_flash=flash_attn,
            attn_dropout=dropout, ff_dropout=dropout, **attn_kwargs,
        )
        self.readout = nn.Linear(dim_fine, vocabulary_size)

    def forward(
        self, segment_context, grouped_codes, *, start, code_length,
        tokens_per_segment, condition=None, modulation=None, incremental=False,
        attention_cache=None, recurrent_cache=None,
    ):
        batch = grouped_codes.shape[0]
        position = code_length % tokens_per_segment
        context = self.context_projection(segment_context)
        context, _ = pack([start, context], "b * d")
        grouped_codes = pad_to_length(grouped_codes, context.shape[1], dim=1)
        features, _ = pack([context, grouped_codes], "b n * d")
        features = features[..., :-1, :]

        if self.recurrence is not None:
            features = rearrange(features, "b n t d -> b (n t) d")
            padded_length = features.shape[1]
            features, recurrent_cache = self.recurrence(
                features[:, :code_length + 1], cache=recurrent_cache
            )
            features = pad_to_length(features, padded_length, dim=-2)
            features = rearrange(
                features, "b (n t) d -> b n t d", t=tokens_per_segment
            )

        if incremental:
            features = features[:, -1:]
            if attention_cache is not None:
                for intermediate in attention_cache.attn_intermediates:
                    key, value = intermediate.cached_kv
                    key, value = (
                        rearrange(t, "(b n) ... -> b n ...", b=batch)[:, -1, :, :position]
                        for t in (key, value)
                    )
                    intermediate.cached_kv = (key, value)

        single_segment = features.shape[1] == 1
        if modulation is not None:
            features = modulation(features, condition)
        features = rearrange(features, "b n t d -> (b n) t d")
        if single_segment:
            features = features[:, :position + 1]
        features, attention_cache = self.attention(
            features, cache=attention_cache, return_hiddens=True
        )
        if (position + 1) % tokens_per_segment == 0:
            attention_cache = None
        if not single_segment:
            features = rearrange(features, "(b n) t d -> b (n t) d", b=batch)
            features = features[:, :code_length + 1]
        return self.readout(features), attention_cache, recurrent_cache
