# SPDX-License-Identifier: MIT
"""Compose morphology conditioning, segment context, and endpoint prediction."""

from math import ceil
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from einops import rearrange, repeat

from tree_skeleton._tensor_ops import pad_to_length, divisible_by, exists, set_module_requires_grad_
from .tokenizer import TreeSegmentTokenizer
from .condition_inputs import MorphologyConditioner
from .length_planning import LengthPlanner
from .generation_stages import GenerationCache, SegmentContext, EndpointDecoder
from .sampling import SkeletonSampler


class AttributeTreeGenerator(Module):
    def __init__(
        self,
        tokenizer: TreeSegmentTokenizer,
        dim=512,
        max_seq_len=8192,
        flash_attn=True,
        attn_depth=12,
        attn_dim_head=64,
        attn_heads=16,
        attn_kwargs=dict(ff_glu=True, attn_num_mem_kv=4),
        dropout=0.0,
        coarse_pre_gateloop_depth=2,
        fine_pre_gateloop_depth=2,
        gateloop_use_heinsen=False,
        fine_attn_depth=2,
        pad_id=-1,
        num_species=0,
        num_attributes=0,
        num_attribute_buckets=3,
        attribute_input_type="bucket",
        num_attribute_fourier_features=8,
        attribute_conditioning="additive",
        attribute_film_pooling="mean",
        text_embedding_dim=0,
        text_condition_tokens=3,
        isolate_condition_init_rng=False,
        length_progress_conditioning=False,
        length_prediction_loss_weight=0.1,
    ):
        super().__init__()
        # Reserve module order so saved optimizer states keep parameter order.
        for name in ("morphology", "planning", "segment_stage", "endpoint_stage"):
            self.add_module(name, None)
        dim, dim_fine = (dim, dim) if isinstance(dim, int) else dim
        if length_prediction_loss_weight < 0:
            raise ValueError("length_prediction_loss_weight must be non-negative")
        self.tokenizer = tokenizer
        set_module_requires_grad_(tokenizer, False)
        self.codebook_size = tokenizer.codebook_size
        self.num_quantizers = tokenizer.num_quantizers
        self.start_embedding = nn.Parameter(torch.randn(dim_fine))
        self.eos_token_id = self.codebook_size
        assert divisible_by(max_seq_len, 2 * self.num_quantizers), (
            f"max_seq_len ({max_seq_len}) must be divisible by "
            f"(2 x {self.num_quantizers}) = {2 * self.num_quantizers}"
        )
        self.code_embedding = nn.Embedding(self.codebook_size + 1, dim)
        self.max_seq_len = max_seq_len
        self.max_num_segments = max_seq_len // (2 * self.num_quantizers)
        self.length_progress_conditioning = bool(length_progress_conditioning)
        self.length_prediction_loss_weight = float(length_prediction_loss_weight)
        initialization = torch.random.fork_rng(devices=[]) if isolate_condition_init_rng else nullcontext()
        with initialization:
            self.morphology = MorphologyConditioner(
                dim=dim, dim_fine=dim_fine, num_species=num_species,
                num_attributes=num_attributes, num_attribute_buckets=num_attribute_buckets,
                attribute_input_type=attribute_input_type,
                num_attribute_fourier_features=num_attribute_fourier_features,
                attribute_conditioning=attribute_conditioning,
                attribute_film_pooling=attribute_film_pooling,
                text_embedding_dim=text_embedding_dim,
                text_condition_tokens=text_condition_tokens,
            )
            if self.length_progress_conditioning and self.condition_token_count == 0:
                raise ValueError(
                    "length/progress conditioning requires structured or text conditioning"
                )
            self.planning = LengthPlanner(
                dim=dim, enabled=self.length_progress_conditioning,
                max_num_segments=self.max_num_segments, num_quantizers=self.num_quantizers,
            )
        self.code_level_embedding = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.endpoint_embedding = nn.Parameter(torch.randn(2, dim))
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.segment_stage = SegmentContext(
            dim=dim, tokens_per_segment=2 * self.num_quantizers,
            recurrent_depth=coarse_pre_gateloop_depth, use_heinsen=gateloop_use_heinsen,
            attn_depth=attn_depth, attn_dim_head=attn_dim_head,
            attn_heads=attn_heads, flash_attn=flash_attn, dropout=dropout,
            attn_kwargs=attn_kwargs,
        )
        self.endpoint_stage = EndpointDecoder(
            dim=dim, dim_fine=dim_fine, recurrent_depth=fine_pre_gateloop_depth,
            use_heinsen=gateloop_use_heinsen, attn_depth=fine_attn_depth,
            attn_dim_head=attn_dim_head, attn_heads=attn_heads,
            flash_attn=flash_attn, dropout=dropout, attn_kwargs=attn_kwargs,
            vocabulary_size=self.codebook_size + 1,
        )
        self.pad_id = pad_id
        tokenizer.pad_id = pad_id

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_species(self):
        return self.morphology.num_species

    @property
    def num_attributes(self):
        return self.morphology.num_attributes

    @property
    def attribute_input_type(self):
        return self.morphology.attribute_input_type

    @property
    def attribute_conditioning(self):
        return self.morphology.attribute_conditioning

    @property
    def attribute_film_pooling(self):
        return self.morphology.attribute_film_pooling

    @property
    def condition_token_count(self):
        return self.morphology.condition_token_count

    @property
    def attribute_uses_prefix(self):
        return self.morphology.attribute_uses_prefix

    @property
    def attribute_uses_film(self):
        return self.morphology.attribute_uses_film

    @property
    def text_embedding_dim(self):
        return self.morphology.text_embedding_dim

    @property
    def text_condition_tokens(self):
        return self.morphology.text_condition_tokens

    @property
    def uses_text_condition(self):
        return self.morphology.uses_text_condition

    def embed_species(self, species_id):
        return self.morphology.embed_species(species_id)

    def embed_attributes(self, attribute_ids=None, attribute_values=None):
        return self.morphology.embed_attributes(attribute_ids, attribute_values)

    def pooled_attribute_condition(self, tokens):
        return self.morphology.pooled_attribute_condition(tokens)

    def prepare_condition_tokens(self, **conditions):
        return self.morphology(**conditions)

    def prepare_planned_num_segments(self, planned, batch, device):
        return self.planning.prepare_planned_num_segments(planned, batch, device)

    @torch.no_grad()
    def predict_num_segments(self, *, batch_size=1, **conditions):
        tokens = self.morphology(batch=int(batch_size), device=self.device, **conditions)
        return self.planning.predict_num_segments(tokens)

    def generate(self, *args, **kwargs):
        return SkeletonSampler(self).generate(*args, **kwargs)

    def forward(
        self,
        *,
        vertices=None,
        segments=None,
        segment_edges=None,
        codes=None,
        species_id=None,
        attribute_ids=None,
        attribute_values=None,
        text_embeddings=None,
        planned_num_segments=None,
        cache=None,
        **kwargs,
    ):
        if not exists(codes):
            codes = self.tokenizer.segment_to_codes(
                vertices=vertices, segments=segments, segment_edges=segment_edges
            )

        return self.forward_on_codes(
            codes,
            species_id=species_id,
            attribute_ids=attribute_ids,
            attribute_values=attribute_values,
            text_embeddings=text_embeddings,
            planned_num_segments=planned_num_segments,
            cache=cache,
            **kwargs,
        )

    def forward_on_codes(
        self,
        codes=None,
        return_loss=True,
        return_cache=False,
        append_eos=True,
        species_id=None,
        attribute_ids=None,
        attribute_values=None,
        text_embeddings=None,
        planned_num_segments=None,
        cache=None,
    ):

        if codes.ndim > 2:
            codes = rearrange(codes, "b ... -> b (...)")

        batch, seq_len, device = *codes.shape, codes.device

        assert seq_len <= self.max_seq_len, (
            f"received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}"
        )

        original_code_lens = ((codes == self.pad_id).cumsum(dim=-1) == 0).sum(dim=-1)
        if self.length_progress_conditioning and return_loss:
            tokens_per_segment = 2 * self.num_quantizers
            if bool((original_code_lens % tokens_per_segment != 0).any()):
                raise ValueError("non-padding training code lengths must contain complete segments")
            actual_num_segments = original_code_lens.float() / tokens_per_segment
        else:
            actual_num_segments = None

        if append_eos:
            assert exists(codes)

            code_lens = original_code_lens

            pad_length = 1

            codes = F.pad(codes, (0, pad_length), value=self.pad_id)

            batch_arange = torch.arange(batch, device=device)

            batch_arange = rearrange(batch_arange, "... -> ... 1")
            code_lens = rearrange(code_lens, "... -> ... 1")

            codes[batch_arange, code_lens] = self.eos_token_id

        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes

        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.code_embedding(codes)

        condition_tokens = self.prepare_condition_tokens(
            batch=batch,
            device=device,
            species_id=species_id,
            attribute_ids=attribute_ids,
            attribute_values=attribute_values,
            text_embeddings=text_embeddings,
        )
        attribute_cond = self.pooled_attribute_condition(condition_tokens)

        predicted_length_fraction = None
        planned = None
        if self.length_progress_conditioning:
            predicted_length_fraction = self.planning(condition_tokens)
            if exists(planned_num_segments):
                planned = self.prepare_planned_num_segments(
                    planned_num_segments,
                    batch,
                    device,
                )
            elif return_loss:
                planned = actual_num_segments
            else:
                planned = (predicted_length_fraction * self.max_num_segments).clamp(
                    1.0, float(self.max_num_segments)
                )

            progress_features = self.planning.length_progress_features(
                planned,
                codes.shape[-2],
                device,
            )
            codes = codes + self.planning.progress(progress_features)

        if exists(condition_tokens) and self.attribute_conditioning == "additive":
            additive_cond = condition_tokens.sum(dim=1)
            codes = codes + rearrange(additive_cond, "b d -> b 1 d")

        seq_arange = torch.arange(codes.shape[-2], device=device)

        codes = codes + self.position_embedding(seq_arange)

        code_len = codes.shape[-2]

        second_part_len = codes.shape[-2]

        level_embed = repeat(
            self.code_level_embedding,
            "q d -> (r q) d",
            r=ceil(second_part_len / self.num_quantizers),
        )
        codes = codes + level_embed[:second_part_len]

        vertex_embed = repeat(
            self.endpoint_embedding,
            "nlv d -> (r nlv q) d",
            r=ceil(second_part_len / (2 * self.num_quantizers)),
            q=self.num_quantizers,
        )
        codes = codes + vertex_embed[:second_part_len]

        num_tokens_per_segment = self.num_quantizers * 2

        code_len_is_multiple_of_line = divisible_by(code_len, num_tokens_per_segment)

        next_multiple_code_len = ceil(code_len / num_tokens_per_segment) * num_tokens_per_segment

        codes = pad_to_length(codes, next_multiple_code_len, dim=-2)

        grouped_codes = rearrange(codes, "b (nl n) d -> b nl n d", n=num_tokens_per_segment)

        complete_segments = grouped_codes if code_len_is_multiple_of_line else grouped_codes[:, :-1]
        previous = GenerationCache(*cache) if cache is not None else GenerationCache()
        prefix = condition_tokens if self.attribute_uses_prefix else None
        segment_context, segment_attention, segment_recurrence = self.segment_stage(
            complete_segments, prefix=prefix, condition=attribute_cond,
            modulation=self.morphology.segment_film,
            cached_embeddings=previous.segment_embeddings,
            attention_cache=previous.segment_attention,
            recurrent_cache=previous.segment_recurrence,
        )

        sos = repeat(self.start_embedding, "d -> b d", b=batch)
        if self.length_progress_conditioning:
            sos_features = torch.stack(
                (
                    (planned / float(self.max_num_segments)).clamp(0.0, 1.0),
                    torch.zeros_like(planned),
                ),
                dim=-1,
            )
            sos = sos + self.planning.progress(sos_features)

        logits, endpoint_attention, endpoint_recurrence = self.endpoint_stage(
            segment_context, grouped_codes, start=sos, code_length=code_len,
            tokens_per_segment=num_tokens_per_segment, condition=attribute_cond,
            modulation=self.morphology.endpoint_film, incremental=cache is not None,
            attention_cache=previous.endpoint_attention,
            recurrent_cache=previous.endpoint_recurrence,
        )
        if not return_loss:
            if return_cache:
                return logits, GenerationCache(
                    segment_context, segment_attention, endpoint_attention,
                    segment_recurrence, endpoint_recurrence,
                )
            return logits

        ce_loss = F.cross_entropy(
            rearrange(logits, "b n c -> b c n"), labels, ignore_index=self.pad_id
        )

        if not self.length_progress_conditioning:
            return ce_loss, None

        target_length_fraction = (actual_num_segments / float(self.max_num_segments)).clamp(
            0.0, 1.0
        )
        per_sample_length_loss = F.smooth_l1_loss(
            predicted_length_fraction,
            target_length_fraction,
            reduction="none",
        )
        # Preserve the full-validation aggregation contract by weighting the
        # auxiliary objective with non-pad autoregressive target counts too.
        target_weights = (original_code_lens + int(append_eos)).float()
        length_loss = (
            per_sample_length_loss * target_weights
        ).sum() / target_weights.sum().clamp_min(1.0)
        total_loss = ce_loss + self.length_prediction_loss_weight * length_loss
        return total_loss, {
            "ce_loss": ce_loss.detach(),
            "length_loss": length_loss.detach(),
            "predicted_length_segments_mean": (
                predicted_length_fraction.detach() * self.max_num_segments
            ).mean(),
            "target_length_segments_mean": actual_num_segments.detach().mean(),
        }


# Keep existing Python imports and serialized model references compatible.
SkeletonGenerator = AttributeTreeGenerator
