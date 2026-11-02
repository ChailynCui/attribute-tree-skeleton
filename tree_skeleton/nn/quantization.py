# SPDX-License-Identifier: MIT
# Adapted research implementation; see README.md for license notices.
from contextlib import nullcontext
from math import log2
from functools import partial

import torch
import torch.distributed as dist
from torch import einsum, nn
from torch.amp import autocast
from torch.distributed import nn as dist_nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack

from vector_quantize_pytorch.lookup_free_quantization import (
    LFQ as BaseLFQ,
    LossBreakdown,
    Return,
    default,
    entropy,
    exists,
)
from vector_quantize_pytorch.residual_lfq import ResidualLFQ as BaseResidualLFQ


def global_sum_and_count(values, count):
    value_sum = values.sum(dim=0)
    token_count = values.new_tensor(count)

    if dist.is_initialized() and dist.get_world_size() > 1:
        value_sum = dist_nn.all_reduce(value_sum)
        dist.all_reduce(token_count)

    return value_sum, token_count


class LFQ(BaseLFQ):
    def forward(
        self,
        x,
        inv_temperature=100.0,
        return_loss_breakdown=False,
        mask=None,
    ):
        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        if should_transpose:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack([x], "b * d")

        assert x.shape[-1] == self.dim, (
            f"expected dimension of {self.dim} but received {x.shape[-1]}"
        )

        x = self.project_in(x)

        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)
        x = self.maybe_l2norm(x)

        force_f32 = self.force_quantization_f32
        quantization_context = (
            partial(autocast, x.device.type, enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():
            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            original_input = x
            codebook_value = torch.ones_like(x) * self.codebook_scale
            quantized = torch.where(x > 0, codebook_value, -codebook_value)
            indices = reduce((quantized > 0).int() * self.mask.int(), "b n c d -> b n c", "sum")
            quantized = self.maybe_l2norm(quantized)

            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized

            if self.training:
                codebook = self.codebook.float() if force_f32 else self.codebook
                codebook = self.maybe_l2norm(codebook)

                if exists(mask):
                    input_for_entropy = original_input[mask]
                else:
                    input_for_entropy = rearrange(original_input, "b n ... -> (b n) ...")

                if self.frac_per_sample_entropy < 1.0:
                    num_tokens = input_for_entropy.size(0)
                    num_sampled_tokens = (
                        max(1, int(num_tokens * self.frac_per_sample_entropy)) if num_tokens else 0
                    )
                    rand_mask = (
                        torch.randn(num_tokens, device=x.device).argsort(dim=-1)
                        < num_sampled_tokens
                    )
                    input_for_entropy = input_for_entropy[rand_mask]

                distance = -2 * einsum("... i d, j d -> ... i j", input_for_entropy, codebook)
                per_sample_probs = (-distance * inv_temperature).softmax(dim=-1)
                token_count = per_sample_probs.shape[0]

                entropy_sum, global_count = global_sum_and_count(
                    entropy(per_sample_probs),
                    token_count,
                )
                probability_sum, _ = global_sum_and_count(
                    per_sample_probs,
                    token_count,
                )

                denominator = global_count.clamp_min(1)
                per_sample_entropy = entropy_sum.sum() / (denominator * self.num_codebooks)
                avg_prob = probability_sum / denominator
                codebook_entropy = entropy(avg_prob).mean()
                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            else:
                entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(entropy_aux_loss + self.entropy_loss_offset)

            if self.training and self.commitment_loss_weight > 0.0:
                commit_loss = F.mse_loss(original_input, quantized.detach(), reduction="none")
                if exists(mask):
                    commit_loss = commit_loss[mask]
                commit_sum, commit_count = global_sum_and_count(
                    commit_loss.reshape(-1),
                    commit_loss.numel(),
                )
                commit_loss = commit_sum / commit_count.clamp_min(1)
            else:
                commit_loss = self.zero

            if force_f32:
                x = x.type(orig_dtype)

        x = rearrange(x, "b n c d -> b n (c d)")
        x = self.project_out(x)

        if should_transpose:
            x = unpack([x], ps, "b * d")[0]
            x = rearrange(x, "b ... d -> b d ...")
            indices = unpack([indices], ps, "b * c")[0]

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        aux_loss = (
            entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight
        )
        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)


class ResidualLFQ(BaseResidualLFQ):
    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_size,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        soft_clamp_input_value=None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        codebook_dim = int(log2(codebook_size))
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([])

        for index in range(num_quantizers):
            codebook_scale = 2**-index
            layer = LFQ(
                dim=codebook_dim,
                codebook_scale=codebook_scale,
                soft_clamp_input_value=soft_clamp_input_value,
                **kwargs,
            )
            self.layers.append(layer)
            if exists(soft_clamp_input_value):
                soft_clamp_input_value *= 0.5

        assert all(not layer.has_projections for layer in self.layers)
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of
