# SPDX-License-Identifier: MIT
"""Pool shared branch endpoints before quantization, then restore segment order."""

import torch
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from tree_skeleton.representation.segments import scatter_mean


def quantize_endpoints(
    quantizer, endpoint_features, *, segments, segment_mask, pad_id,
    num_quantizers, checkpoint_quantizer=False,
):
    batch, device = segments.shape[0], segments.device
    num_vertices = int(segments.amax().item() + 1)
    endpoint_features = rearrange(endpoint_features, "b nl (nv d) -> b (nl nv) d", nv=2)
    feature_dim = endpoint_features.shape[-1]
    vertex_features = torch.zeros((batch, num_vertices + 1, feature_dim), device=device)
    indices = segments.masked_fill(~segment_mask[..., None], num_vertices)
    feature_indices = repeat(indices, "b nl nv -> b (nl nv) d", d=feature_dim)
    pooled = scatter_mean(vertex_features, feature_indices, endpoint_features, dim=-2)

    vertex_mask = torch.ones((batch, num_vertices + 1), device=device, dtype=torch.bool)
    vertex_mask[:, -1] = False
    if checkpoint_quantizer:
        quantized, codes, loss = checkpoint(
            quantizer, pooled, mask=vertex_mask, use_reentrant=False
        )
    else:
        quantized, codes, loss = quantizer(pooled, mask=vertex_mask)

    features = quantized.gather(-2, feature_indices)
    features = rearrange(features, "b (nl nv) d -> b nl (nv d)", nv=2)
    code_indices = repeat(indices, "b nl nv -> b (nl nv) q", q=num_quantizers)
    segment_codes = codes.gather(-2, code_indices)
    endpoint_mask = repeat(segment_mask, "b nl -> b (nl nv) 1", nv=2)
    return features, segment_codes.masked_fill(~endpoint_mask, pad_id), loss
