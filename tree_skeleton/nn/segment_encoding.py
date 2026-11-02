# SPDX-License-Identifier: MIT
"""Graph encoding of branch geometry with direct endpoint indexing."""

from functools import partial
from math import pi

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
from local_attention import LocalMHA
from torch_geometric.nn.conv import SAGEConv
from x_transformers.x_transformers import FeedForward, RMSNorm

from tree_skeleton.representation.segments import (
    discretize,
    get_derived_segment_features,
    get_derived_segment_edge_features,
    scatter_mean,
)


class SegmentGraphEncoder(nn.Module):
    """Encode segment geometry and adjacency without owning a tokenizer."""

    def __init__(
        self, *, num_discrete_coors, coor_continuous_range, dim_coor_embed,
        num_discrete_length, dim_length_embed, num_discrete_directions,
        dim_direction_embed, num_discrete_midps, dim_midp_embed,
        num_discrete_angle, dim_angle_embed, encoder_dims_through_depth,
        dim_codebook, sageconv_kwargs, attn_encoder_depth, local_attn_kwargs,
        local_attn_window_size, attn_dropout, ff_dropout,
    ):
        super().__init__()
        self.discretize_segment_coords = partial(
            discretize, num_discrete=num_discrete_coors, continuous_range=coor_continuous_range
        )
        self.coordinates = nn.Embedding(num_discrete_coors, dim_coor_embed)
        self.discretize_length = partial(
            discretize, num_discrete=num_discrete_length, continuous_range=(0.0, 2.0)
        )
        self.length = nn.Embedding(num_discrete_length, dim_length_embed)
        self.discretize_midps = partial(
            discretize, num_discrete=num_discrete_midps, continuous_range=coor_continuous_range
        )
        self.midpoint = nn.Embedding(num_discrete_midps, dim_midp_embed)
        self.discretize_directions = partial(
            discretize, num_discrete=num_discrete_directions, continuous_range=coor_continuous_range
        )
        self.direction = nn.Embedding(num_discrete_directions, dim_direction_embed)
        self.discretize_angle = partial(
            discretize, num_discrete=num_discrete_angle, continuous_range=(0.0, pi / 2.0)
        )
        self.angle = nn.Embedding(num_discrete_angle, dim_angle_embed)

        feature_dim = (
            dim_coor_embed * 6 + dim_length_embed + dim_direction_embed * 3
            + dim_angle_embed + dim_midp_embed * 3
        )
        self.input_projection = nn.Linear(feature_dim, dim_codebook)
        first_dim, *graph_dims = encoder_dims_through_depth
        self.graph_input = SAGEConv(dim_codebook, first_dim, **sageconv_kwargs)
        self.input_activation = nn.Sequential(nn.SiLU(), nn.LayerNorm(first_dim))
        self.graph_layers = nn.ModuleList()
        current_dim = first_dim
        for next_dim in graph_dims:
            self.graph_layers.append(SAGEConv(current_dim, next_dim, **sageconv_kwargs))
            current_dim = next_dim

        self.attention = nn.ModuleList(
            nn.ModuleList([
                LocalMHA(
                    dim=current_dim, causal=False, prenorm=True,
                    dropout=attn_dropout, window_size=local_attn_window_size,
                    **local_attn_kwargs,
                ),
                nn.Sequential(
                    RMSNorm(current_dim), FeedForward(current_dim, glu=True, dropout=ff_dropout)
                ),
            ])
            for _ in range(attn_encoder_depth)
        )
        self.endpoint_projection = nn.Linear(current_dim, dim_codebook * 2)

    def forward(
        self,
        *,
        vertices,
        segments,
        segment_edges,
        segment_mask,
        segment_edges_mask,
        return_segment_coordinates=False,
    ):
        batch, device = vertices.shape[0], vertices.device
        num_segments = segments.shape[1]

        segment_without_pad = segments.masked_fill(~rearrange(segment_mask, "b nl -> b nl 1"), 0)
        batch_indices = torch.arange(batch, device=device)[:, None, None]
        segment_coords = vertices[batch_indices, segment_without_pad]

        segment_edges_wo_pad = segment_edges.masked_fill(
            ~rearrange(segment_edges_mask, "b e -> b e 1"), 0
        )

        segment_edge_coords = segment_coords[batch_indices, segment_edges_wo_pad]

        derived_segment_features = get_derived_segment_features(segment_coords)

        discrete_midps = self.discretize_midps(derived_segment_features["midps"])
        midp_embed = self.midpoint(discrete_midps)

        discrete_length = self.discretize_length(derived_segment_features["length"])
        length_embed = self.length(discrete_length)

        discrete_directions = self.discretize_directions(derived_segment_features["directions"])
        direction_embed = self.direction(discrete_directions)

        derived_segment_edge_features = get_derived_segment_edge_features(
            segment_edge_coords, segment_edges_mask
        )

        discrete_angle = self.discretize_angle(derived_segment_edge_features)
        segment_edge_angle_embed = self.angle(discrete_angle)

        dim_angle = segment_edge_angle_embed.shape[-1]

        segment_angle_embed = torch.zeros((batch, num_segments + 1, dim_angle), device=device)
        segment_edge_angle_embed = repeat(
            segment_edge_angle_embed, "b e d -> b (e repeat) d", repeat=2
        )

        segment_edges_wo_pad = segment_edges_wo_pad.masked_fill(
            ~rearrange(segment_edges_mask, "b e -> b e 1"), num_segments
        )

        segment_edges_index = rearrange(segment_edges_wo_pad, "b e nel -> b (e nel) 1")
        segment_edges_index = repeat(
            segment_edges_index, "b enel 1 -> b enel repeat", repeat=dim_angle
        )

        averaged_segment_angle_embed = scatter_mean(
            segment_angle_embed, segment_edges_index, segment_edge_angle_embed, dim=-2
        )

        angle_embed = averaged_segment_angle_embed[:, :num_segments, :]

        discrete_segment_coords = self.discretize_segment_coords(segment_coords)
        discrete_segment_coords = rearrange(discrete_segment_coords, "b nl nv c -> b nl (nv c)")

        segment_coor_embed = self.coordinates(discrete_segment_coords)
        segment_coor_embed = rearrange(segment_coor_embed, "b nl c d -> b nl (c d)")

        segment_embed, _ = pack(
            [segment_coor_embed, length_embed, direction_embed, angle_embed, midp_embed], "b nl *"
        )
        segment_embed = self.input_projection(segment_embed)

        segment_index_offsets = reduce(segment_mask.long(), "b nl -> b", "sum")
        segment_index_offsets = F.pad(segment_index_offsets.cumsum(dim=0), (1, -1), value=0)
        segment_index_offsets = rearrange(segment_index_offsets, "b -> b 1 1")

        segment_edges = segment_edges + segment_index_offsets
        segment_edges = segment_edges[segment_edges_mask]
        segment_edges = rearrange(segment_edges, "be ij -> ij be")

        orig_segment_embed_shape = segment_embed.shape[:2]

        segment_embed = segment_embed[segment_mask]

        segment_embed = self.graph_input(segment_embed, segment_edges)

        segment_embed = self.input_activation(segment_embed)

        for conv in self.graph_layers:
            segment_embed = conv(segment_embed, segment_edges)

        shape = (*orig_segment_embed_shape, segment_embed.shape[-1])

        segment_embed = segment_embed.new_zeros(shape).masked_scatter(
            rearrange(segment_mask, "... -> ... 1"), segment_embed
        )

        for attn, ff in self.attention:
            segment_embed = attn(segment_embed, mask=segment_mask) + segment_embed
            segment_embed = ff(segment_embed) + segment_embed

        if not return_segment_coordinates:
            return segment_embed

        return segment_embed, discrete_segment_coords
