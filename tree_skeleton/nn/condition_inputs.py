# SPDX-License-Identifier: MIT
"""Species and morphology conditioning as an independently callable module."""

import torch
from torch import nn
from einops import rearrange

from tree_skeleton._tensor_ops import safe_cat, exists
from .conditioning import AttributeFiLM, ContinuousAttributeEmbedder


class MorphologyConditioner(nn.Module):
    """Own condition validation, embeddings, pooling, and feature modulation."""

    def __init__(
        self, *, dim, dim_fine, num_species=0, num_attributes=0,
        num_attribute_buckets=3, attribute_input_type="bucket",
        num_attribute_fourier_features=8, attribute_conditioning="additive",
        attribute_film_pooling="mean", text_embedding_dim=0, text_condition_tokens=3,
    ):
        super().__init__()
        assert attribute_conditioning in {"additive", "prefix", "film", "prefix_film"}, (
            f"unknown attribute conditioning mode: {attribute_conditioning}"
        )
        assert attribute_input_type in {"bucket", "continuous"}, (
            f"unknown attribute input type: {attribute_input_type}"
        )
        assert attribute_film_pooling in {"mean", "gated_sum", "flatten_project"}, (
            f"unknown attribute FiLM pooling mode: {attribute_film_pooling}"
        )
        num_species = int(num_species)
        if num_species < 0:
            raise ValueError("num_species must be non-negative")
        if (num_attributes > 0 or num_species > 0) and text_embedding_dim > 0:
            raise ValueError(
                "structured species/attributes and direct text embeddings are mutually exclusive"
            )
        if text_embedding_dim < 0:
            raise ValueError("text_embedding_dim must be non-negative")
        if text_embedding_dim > 0 and text_condition_tokens < 1:
            raise ValueError("text_condition_tokens must be positive in direct-text mode")
        self.num_species = num_species
        self.num_attributes = num_attributes
        self.attribute_input_type = attribute_input_type
        self.attribute_conditioning = attribute_conditioning
        self.attribute_film_pooling = attribute_film_pooling
        self.text_embedding_dim = int(text_embedding_dim)
        self.text_condition_tokens = int(text_condition_tokens)
        self.uses_text_condition = self.text_embedding_dim > 0
        self.condition_token_count = (
            self.text_condition_tokens
            if self.uses_text_condition
            else num_attributes + int(num_species > 0)
        )
        self.attribute_uses_prefix = attribute_conditioning in {"prefix", "prefix_film"}
        self.attribute_uses_film = attribute_conditioning in {"film", "prefix_film"}
        self.species = nn.Embedding(num_species, dim) if num_species > 0 else None
        self.buckets = (
            nn.ModuleList(
                [nn.Embedding(num_attribute_buckets, dim) for _ in range(num_attributes)]
            )
            if attribute_input_type == "bucket"
            else None
        )
        self.continuous = (
            ContinuousAttributeEmbedder(
                num_attributes, dim, num_fourier_features=num_attribute_fourier_features
            )
            if num_attributes > 0 and attribute_input_type == "continuous"
            else None
        )
        self.attribute_slots = (
            nn.Parameter(torch.randn(num_attributes, dim))
            if num_attributes > 0 and self.attribute_uses_prefix
            else None
        )
        self.text_projection = (
            nn.Sequential(
                nn.LayerNorm(self.text_embedding_dim), nn.Linear(self.text_embedding_dim, dim)
            )
            if self.uses_text_condition
            else None
        )
        self.text_slots = (
            nn.Parameter(torch.randn(self.text_condition_tokens, dim))
            if self.uses_text_condition
            else None
        )
        self.pool = (
            nn.Sequential(nn.Linear(self.condition_token_count * dim, dim), nn.LayerNorm(dim))
            if self.condition_token_count > 0
            and self.attribute_uses_film
            and attribute_film_pooling == "flatten_project"
            else None
        )
        self.gates = (
            nn.Parameter(torch.randn(self.condition_token_count, dim) * 0.02)
            if self.condition_token_count > 0
            and self.attribute_uses_film
            and attribute_film_pooling == "gated_sum"
            else None
        )
        self.segment_film = (
            AttributeFiLM(dim, dim)
            if self.condition_token_count > 0 and self.attribute_uses_film
            else None
        )
        self.endpoint_film = (
            AttributeFiLM(dim, dim_fine)
            if self.condition_token_count > 0 and self.attribute_uses_film
            else None
        )

    def prepare_attribute_ids(self, attribute_ids, batch, device):
        if self.num_attributes == 0:
            return None

        assert self.attribute_input_type == "bucket", (
            "attribute_ids are only valid for bucket attribute_input_type"
        )

        if not exists(attribute_ids):
            attribute_ids = torch.zeros(
                (batch, self.num_attributes), device=device, dtype=torch.long
            )

        attribute_ids = attribute_ids.to(device).long()

        if attribute_ids.ndim == 1:
            attribute_ids = rearrange(attribute_ids, "b -> b 1")

        assert attribute_ids.shape[-1] == self.num_attributes, (
            f"expected {self.num_attributes} attribute ids, got {attribute_ids.shape[-1]}"
        )
        if attribute_ids.shape[0] != batch:
            raise ValueError(
                f"attribute id batch ({attribute_ids.shape[0]}) does not match code batch ({batch})"
            )

        return attribute_ids

    def prepare_species_id(self, species_id, batch, device):
        if self.num_species == 0:
            return None
        if not exists(species_id):
            raise ValueError("species_id is required when num_species > 0")

        species_id = torch.as_tensor(species_id, device=device, dtype=torch.long)
        if species_id.ndim == 0:
            species_id = rearrange(species_id, "-> 1")
        elif species_id.ndim == 2 and species_id.shape[-1] == 1:
            species_id = rearrange(species_id, "b 1 -> b")
        elif species_id.ndim != 1:
            raise ValueError(
                f"species_id must have shape [batch] or [batch, 1], got {tuple(species_id.shape)}"
            )
        if species_id.shape[0] != batch:
            raise ValueError(
                f"species_id batch ({species_id.shape[0]}) does not match code batch ({batch})"
            )
        if bool(((species_id < 0) | (species_id >= self.num_species)).any()):
            raise ValueError(f"species_id values must be in [0, {self.num_species})")
        return species_id

    def prepare_attribute_values(self, attribute_values, batch, device):
        if self.num_attributes == 0:
            return None

        assert self.attribute_input_type == "continuous", (
            "attribute_values are only valid for continuous attribute_input_type"
        )

        if not exists(attribute_values):
            attribute_values = torch.zeros(
                (batch, self.num_attributes), device=device, dtype=torch.float
            )

        attribute_values = attribute_values.to(device).float()

        if attribute_values.ndim == 1:
            attribute_values = rearrange(attribute_values, "b -> b 1")

        assert attribute_values.shape[-1] == self.num_attributes, (
            f"expected {self.num_attributes} attribute values, got {attribute_values.shape[-1]}"
        )
        if attribute_values.shape[0] != batch:
            raise ValueError(
                f"attribute value batch ({attribute_values.shape[0]}) does not match "
                f"code batch ({batch})"
            )
        if not bool(torch.isfinite(attribute_values).all()):
            raise ValueError("attribute_values must be finite")
        if bool(((attribute_values < 0.0) | (attribute_values > 1.0)).any()):
            raise ValueError("attribute_values must be in [0, 1]")

        return attribute_values

    def prepare_text_embeddings(self, text_embeddings, batch, device):
        if not self.uses_text_condition:
            return None
        if not exists(text_embeddings):
            raise ValueError("text_embeddings are required in direct-text mode")

        text_embeddings = text_embeddings.to(device).float()
        if text_embeddings.ndim == 1:
            text_embeddings = rearrange(text_embeddings, "d -> 1 d")
        if text_embeddings.ndim != 2:
            raise ValueError(
                f"text_embeddings must have shape [batch, {self.text_embedding_dim}], "
                f"got {tuple(text_embeddings.shape)}"
            )
        if text_embeddings.shape[0] != batch:
            raise ValueError(
                f"text embedding batch ({text_embeddings.shape[0]}) does not match "
                f"code batch ({batch})"
            )
        if text_embeddings.shape[-1] != self.text_embedding_dim:
            raise ValueError(
                f"expected text embedding dim {self.text_embedding_dim}, "
                f"got {text_embeddings.shape[-1]}"
            )
        return text_embeddings

    def validate_condition_inputs(
        self,
        *,
        species_id=None,
        attribute_ids=None,
        attribute_values=None,
        text_embeddings=None,
    ):
        has_ids = exists(attribute_ids)
        has_values = exists(attribute_values)
        has_text = exists(text_embeddings)
        has_species = exists(species_id)

        if self.uses_text_condition:
            if not has_text:
                raise ValueError("text_embeddings are required in direct-text mode")
            if has_species or has_ids or has_values:
                raise ValueError(
                    "text_embeddings cannot be combined with structured species/attributes"
                )
            return

        if self.num_species > 0 and not has_species:
            raise ValueError("species_id is required when num_species > 0")
        if self.num_species == 0 and has_species:
            raise ValueError("this model was constructed without species conditioning")

        if self.num_attributes > 0:
            if has_text:
                raise ValueError("text_embeddings cannot be combined with structured attributes")
            if self.attribute_input_type == "bucket":
                if has_values:
                    raise ValueError("attribute_values are invalid for bucket conditioning")
            else:
                if has_ids:
                    raise ValueError("attribute_ids are invalid for continuous conditioning")
            return

        if self.num_species > 0:
            if has_ids or has_values or has_text:
                raise ValueError(
                    "this model was constructed with species conditioning but "
                    "without attribute conditioning"
                )
            return

        if has_ids or has_values or has_text or has_species:
            raise ValueError("this model was constructed without a conditioning input")

    def embed_species(self, species_id):
        if self.species is None:
            return None
        return rearrange(self.species(species_id), "b d -> b 1 d")

    def embed_attributes(self, attribute_ids=None, attribute_values=None):
        if self.num_attributes == 0:
            return None

        if self.attribute_input_type == "bucket":
            attribute_tokens = torch.stack(
                [
                    embed(attribute_ids[:, attr_idx])
                    for attr_idx, embed in enumerate(self.buckets)
                ],
                dim=1,
            )
        else:
            attribute_tokens = self.continuous(attribute_values)

        if self.attribute_uses_prefix:
            attribute_tokens = attribute_tokens + self.attribute_slots

        return attribute_tokens

    def embed_text_condition(self, text_embeddings):
        if not self.uses_text_condition:
            return None
        projected = self.text_projection(text_embeddings)
        return rearrange(projected, "b d -> b 1 d") + self.text_slots

    def pooled_attribute_condition(self, attribute_tokens):
        if not exists(attribute_tokens) or not self.attribute_uses_film:
            return None

        if self.attribute_film_pooling == "mean":
            return attribute_tokens.mean(dim=1)

        if self.attribute_film_pooling == "gated_sum":
            attribute_weights = self.gates.softmax(dim=0)
            return (attribute_tokens * attribute_weights).sum(dim=1)

        return self.pool(rearrange(attribute_tokens, "b n d -> b (n d)"))

    def forward(
        self,
        *,
        batch,
        device,
        species_id=None,
        attribute_ids=None,
        attribute_values=None,
        text_embeddings=None,
    ):
        """Validate and embed the frozen condition-token contract once."""

        self.validate_condition_inputs(
            species_id=species_id,
            attribute_ids=attribute_ids,
            attribute_values=attribute_values,
            text_embeddings=text_embeddings,
        )

        if self.uses_text_condition:
            text_embeddings = self.prepare_text_embeddings(
                text_embeddings,
                batch,
                device,
            )
            return self.embed_text_condition(text_embeddings)

        species_token = None
        attribute_tokens = None
        if self.num_species > 0:
            species_id = self.prepare_species_id(species_id, batch, device)
            species_token = self.embed_species(species_id)
        if self.num_attributes > 0:
            if self.attribute_input_type == "bucket":
                attribute_ids = self.prepare_attribute_ids(attribute_ids, batch, device)
            else:
                attribute_values = self.prepare_attribute_values(
                    attribute_values,
                    batch,
                    device,
                )
            attribute_tokens = self.embed_attributes(
                attribute_ids=attribute_ids,
                attribute_values=attribute_values,
            )

        # Frozen structured token order: species, height, crown width, density.
        return safe_cat((species_token, attribute_tokens), dim=1)
