# SPDX-License-Identifier: MIT
"""Optional segment-budget prediction and progress features."""

import torch
from torch import nn
from einops import rearrange, repeat
from tree_skeleton._tensor_ops import exists


class LengthPlanner(nn.Module):
    def __init__(self, *, dim, enabled, max_num_segments, num_quantizers):
        super().__init__()
        self.length_progress_conditioning = bool(enabled)
        self.max_num_segments = max_num_segments
        self.num_quantizers = num_quantizers
        self.predictor = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, 1),
            )
            if self.length_progress_conditioning
            else None
        )
        self.progress = nn.Linear(2, dim) if self.length_progress_conditioning else None

        # A warm-started model begins at exactly the old token distribution.
        # The planner still receives gradients immediately, while the zeroed
        # final projection lets progress conditioning enter smoothly.
        if self.length_progress_conditioning:
            nn.init.zeros_(self.predictor[-1].weight)
            nn.init.zeros_(self.predictor[-1].bias)
            nn.init.zeros_(self.progress.weight)
            nn.init.zeros_(self.progress.bias)

    def predict_length_fraction(self, condition_tokens):
        if not self.length_progress_conditioning:
            raise RuntimeError("length/progress conditioning is disabled")
        if not exists(condition_tokens):
            raise ValueError("condition tokens are required for length prediction")
        condition_summary = condition_tokens.mean(dim=1)
        return self.predictor(condition_summary).squeeze(-1).sigmoid()

    def prepare_planned_num_segments(self, planned_num_segments, batch, device):
        planned = torch.as_tensor(
            planned_num_segments,
            device=device,
            dtype=torch.float,
        )
        if planned.ndim == 0:
            planned = planned.repeat(batch)
        elif planned.ndim == 2 and planned.shape[-1] == 1:
            planned = rearrange(planned, "b 1 -> b")
        elif planned.ndim != 1:
            raise ValueError("planned_num_segments must be a scalar or a [batch] tensor")
        if planned.shape[0] != batch:
            raise ValueError(
                f"planned length batch ({planned.shape[0]}) does not match code batch ({batch})"
            )
        if not bool(torch.isfinite(planned).all()):
            raise ValueError("planned_num_segments must be finite")
        return planned.clamp(1.0, float(self.max_num_segments))

    @torch.no_grad()
    def predict_num_segments(self, condition_tokens):
        fraction = self.predict_length_fraction(condition_tokens)
        return (fraction * self.max_num_segments).round().long().clamp(
            1, self.max_num_segments
        )

    def length_progress_features(self, planned_num_segments, token_count, device):
        """Return [planned length fraction, relative segment progress]."""

        length_fraction = (planned_num_segments / float(self.max_num_segments)).clamp(0.0, 1.0)
        token_positions = torch.arange(token_count, device=device)
        segment_positions = torch.div(
            token_positions,
            2 * self.num_quantizers,
            rounding_mode="floor",
        ).float()
        denominators = (planned_num_segments - 1.0).clamp_min(1.0)
        progress = (
            rearrange(segment_positions, "n -> 1 n") / rearrange(denominators, "b -> b 1")
        ).clamp(0.0, 1.0)
        length_fraction = repeat(
            length_fraction,
            "b -> b n",
            n=token_count,
        )
        return torch.stack((length_fraction, progress), dim=-1)

    def forward(self, condition_tokens):
        return self.predict_length_fraction(condition_tokens)
