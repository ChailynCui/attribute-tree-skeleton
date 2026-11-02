# SPDX-License-Identifier: MIT
"""Shared skeleton tokenization assembled from graph and coordinate components."""

import torch
from torch.nn import Module
from torch.amp import autocast
from einops import rearrange, repeat, reduce

from tree_skeleton._tensor_ops import exists, first
from tree_skeleton.representation.segments import (
    undiscretize, derive_segment_edges_from_segments, gaussian_blur_1d,
)
from .quantization import ResidualLFQ
from .segment_encoding import SegmentGraphEncoder
from .coordinate_reconstruction import CoordinateDecoder
from .endpoint_quantization import quantize_endpoints


class TreeSegmentTokenizer(Module):
    def __init__(
        self,
        num_discrete_coors=128,
        coor_continuous_range=(-1.0, 1.0),
        dim_coor_embed=64,
        num_discrete_length=128,
        dim_length_embed=16,
        num_discrete_directions=128,
        dim_direction_embed=64,
        num_discrete_midps=128,
        dim_midp_embed=16,
        num_discrete_angle=128,
        dim_angle_embed=16,
        encoder_dims_through_depth=(64, 128, 256, 256, 384),
        init_decoder_conv_kernel=7,
        decoder_dims_through_depth=(
            128,
            128,
            128,
            128,
            192,
            192,
            192,
            192,
            256,
            256,
            256,
            256,
            256,
            256,
            384,
            384,
            384,
        ),
        dim_codebook=192,
        num_quantizers=2,
        codebook_size=8192,
        rlfq_kwargs=dict(),
        sageconv_kwargs=dict(normalize=True, project=True),
        commit_loss_weight=0.1,
        bin_smooth_blur_sigma=0.4,
        attn_encoder_depth=0,
        attn_decoder_depth=0,
        local_attn_kwargs=dict(dim_head=32, heads=8),
        local_attn_window_size=64,
        pad_id=-1,
        attn_dropout=0.0,
        ff_dropout=0.0,
        resnet_dropout=0,
        checkpoint_quantizer=False,
    ):
        super().__init__()
        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range
        self.segment_encoder = SegmentGraphEncoder(
            num_discrete_coors=num_discrete_coors,
            coor_continuous_range=coor_continuous_range,
            dim_coor_embed=dim_coor_embed,
            num_discrete_length=num_discrete_length,
            dim_length_embed=dim_length_embed,
            num_discrete_directions=num_discrete_directions,
            dim_direction_embed=dim_direction_embed,
            num_discrete_midps=num_discrete_midps,
            dim_midp_embed=dim_midp_embed,
            num_discrete_angle=num_discrete_angle,
            dim_angle_embed=dim_angle_embed,
            encoder_dims_through_depth=encoder_dims_through_depth,
            dim_codebook=dim_codebook,
            sageconv_kwargs=sageconv_kwargs,
            attn_encoder_depth=attn_encoder_depth,
            local_attn_kwargs=local_attn_kwargs,
            local_attn_window_size=local_attn_window_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        # Keep decoder parameters before the quantizer in optimizer checkpoints.
        self.add_module("coordinate_decoder", None)
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.quantizer = ResidualLFQ(
            dim=dim_codebook, num_quantizers=num_quantizers,
            codebook_size=codebook_size, commitment_loss_weight=1.0,
            **rlfq_kwargs,
        )
        self.checkpoint_quantizer = checkpoint_quantizer
        self.pad_id = pad_id
        self.coordinate_decoder = CoordinateDecoder(
            dim_codebook=dim_codebook,
            attn_decoder_depth=attn_decoder_depth,
            local_attn_kwargs=local_attn_kwargs,
            local_attn_window_size=local_attn_window_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            decoder_dims_through_depth=decoder_dims_through_depth,
            init_decoder_conv_kernel=init_decoder_conv_kernel,
            resnet_dropout=resnet_dropout,
            num_discrete_coors=num_discrete_coors,
        )
        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(
        self, *, vertices, segments, segment_edges, segment_mask,
        segment_edges_mask, return_segment_coordinates=False,
    ):
        return self.segment_encoder(
            vertices=vertices, segments=segments, segment_edges=segment_edges,
            segment_mask=segment_mask, segment_edges_mask=segment_edges_mask,
            return_segment_coordinates=return_segment_coordinates,
        )

    def quantize(self, *, segments, segment_mask, segment_embed, pad_id=None):
        return quantize_endpoints(
            self.quantizer, self.segment_encoder.endpoint_projection(segment_embed),
            segments=segments, segment_mask=segment_mask,
            pad_id=self.pad_id if pad_id is None else pad_id,
            num_quantizers=self.num_quantizers,
            checkpoint_quantizer=self.checkpoint_quantizer,
        )

    def decode(self, quantized, segment_mask):
        return self.coordinate_decoder(quantized, segment_mask)

    @torch.no_grad()
    def codes_to_segments(self, codes, segment_mask=None, return_discrete_codes=False):
        codes = rearrange(codes, "b ... -> b (...)")

        if not exists(segment_mask):
            segment_mask = reduce(
                codes != self.pad_id, "b (nl nv q) -> b nl", "all", nv=2, q=self.num_quantizers
            )

        codes = rearrange(codes, "b (n q) -> b n q", q=self.num_quantizers)

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, "b (nl nv) d -> b nl (nv d)", nv=2)

        decoded = self.decode(quantized, segment_mask=segment_mask)

        decoded = decoded.masked_fill(~segment_mask[..., None], 0.0)
        pred_segment_coords = self.coordinate_decoder.readout(decoded)

        pred_segment_coords = pred_segment_coords.argmax(dim=-1)

        pred_segment_coords = rearrange(pred_segment_coords, "... (v c) -> ... v c", v=2)

        continuous_coors = undiscretize(
            pred_segment_coords,
            num_discrete=self.num_discrete_coors,
            continuous_range=self.coor_continuous_range,
        )

        continuous_coors = continuous_coors.masked_fill(
            ~rearrange(segment_mask, "b nf -> b nf 1 1"), float("nan")
        )

        if not return_discrete_codes:
            return continuous_coors, segment_mask

        return continuous_coors, pred_segment_coords, segment_mask

    @torch.no_grad()
    def segment_to_codes(self, vertices, segments, segment_edges=None, *args, **kwargs):

        inputs = [vertices, segments, segment_edges]
        inputs = [*filter(exists, inputs)]
        ndims = {i.ndim for i in inputs}

        assert len(ndims) == 1
        batch_less = first(list(ndims)) == 2

        if batch_less:
            inputs = [rearrange(i, "... -> 1 ...") for i in inputs]

        input_kwargs = dict(zip(["vertices", "segments", "segment_edges"], inputs))

        self.eval()

        codes = self.forward(**input_kwargs, return_codes=True, **kwargs)

        if batch_less:
            codes = rearrange(codes, "1 ... -> ...")

        return codes

    def forward(
        self,
        *,
        vertices,
        segments,
        segment_edges=None,
        return_codes=False,
        return_loss_breakdown=False,
        return_recon_segments=False,
        only_return_recon_segments=False,
        rvq_sample_codebook_temp=1.0,
    ):
        if not exists(segment_edges):
            segment_edges = derive_segment_edges_from_segments(segments, pad_id=self.pad_id)

        segment_mask = reduce(segments != self.pad_id, "b nl c -> b nl", "all")
        segment_edges_mask = reduce(segment_edges != self.pad_id, "b e ij -> b e", "all")

        encoded, segment_coordinates = self.encode(
            vertices=vertices,
            segments=segments,
            segment_edges=segment_edges,
            segment_edges_mask=segment_edges_mask,
            segment_mask=segment_mask,
            return_segment_coordinates=True,
        )

        quantized, codes, commit_loss = self.quantize(
            segment_embed=encoded,
            segments=segments,
            segment_mask=segment_mask,
        )

        if return_codes:
            assert not return_recon_segments, (
                "cannot return reconstructed segments when just returning raw codes"
            )

            codes = codes.masked_fill(~repeat(segment_mask, "b nf -> b (nf 2) 1"), self.pad_id)
            return codes

        decode = self.decode(quantized, segment_mask=segment_mask)

        pred_segment_coords = self.coordinate_decoder.readout(decode)

        if return_recon_segments or only_return_recon_segments:
            recon_segments = undiscretize(
                pred_segment_coords.argmax(dim=-1),
                num_discrete=self.num_discrete_coors,
                continuous_range=self.coor_continuous_range,
            )

            recon_segments = rearrange(recon_segments, "b nl (nv c) -> b nl nv c", nv=2)
            recon_mask = rearrange(segment_mask, "b nl -> b nl 1 1")
            recon_segments = recon_segments.masked_fill(~recon_mask, float("nan"))

        if only_return_recon_segments:
            return recon_segments

        pred_segment_coords = rearrange(pred_segment_coords, "b ... c -> b c (...)")
        segment_coordinates = rearrange(segment_coordinates, "b ... -> b 1 (...)")

        with autocast("cuda", enabled=False):
            pred_log_prob = pred_segment_coords.log_softmax(dim=1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, segment_coordinates, 1.0)

            if self.bin_smooth_blur_sigma >= 0.0:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma=self.bin_smooth_blur_sigma)

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim=1)

            segment_mask = repeat(segment_mask, "b nl -> b (nl r)", r=6)
            recon_loss = recon_losses[segment_mask].mean()

        total_loss = recon_loss + commit_loss.sum() * self.commit_loss_weight

        loss_breakdown = {"recon_loss": recon_loss, "commit_loss": commit_loss.sum()}

        if not return_loss_breakdown:
            if not return_recon_segments:
                return total_loss

            return recon_segments, total_loss

        if not return_recon_segments:
            return total_loss, loss_breakdown

        return recon_segments, total_loss, loss_breakdown


# Keep existing Python imports and serialized model references compatible.
SkeletonTokenizer = TreeSegmentTokenizer
