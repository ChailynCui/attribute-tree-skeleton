import torch
from torch import nn
from torch.nn import Module
from einops import rearrange
from math import pi


class AttributeFiLM(Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.LayerNorm(cond_dim), nn.SiLU(), nn.Linear(cond_dim, hidden_dim * 2)
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, tokens, cond):
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)

        if tokens.ndim == 3:
            scale = rearrange(scale, "b d -> b 1 d")
            shift = rearrange(shift, "b d -> b 1 d")
        elif tokens.ndim == 4:
            scale = rearrange(scale, "b d -> b 1 1 d")
            shift = rearrange(shift, "b d -> b 1 1 d")
        else:
            raise ValueError(f"AttributeFiLM expects 3D or 4D tokens, got {tokens.ndim}D")

        return tokens * (1.0 + scale) + shift


class ContinuousAttributeEmbedder(Module):
    def __init__(self, num_attributes, dim, num_fourier_features=8):
        super().__init__()
        self.num_attributes = num_attributes
        self.type_embed = nn.Parameter(torch.randn(num_attributes, dim))
        self.register_buffer(
            "freqs",
            (2 ** torch.arange(num_fourier_features, dtype=torch.float)) * pi,
            persistent=False,
        )

        feature_dim = 1 + num_fourier_features * 2
        self.to_token = nn.Sequential(
            nn.Linear(feature_dim, dim), nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )

    def forward(self, attribute_values):
        attribute_values = attribute_values.clamp(0.0, 1.0)
        values = rearrange(attribute_values, "b n -> b n 1")
        angles = values * self.freqs
        features = torch.cat((values, angles.sin(), angles.cos()), dim=-1)
        return self.to_token(features) + self.type_embed
