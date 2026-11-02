# SPDX-License-Identifier: MIT
# Adapted research implementation; see README.md for license notices.
import json
from pathlib import Path

import torch


def to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        raise ValueError("metrics_logger only supports scalar tensors")

    if isinstance(value, (int, float)):
        return float(value)

    return value


def append_metrics(metrics_path: Path, *, step: int, epoch: float, split: str, metrics: dict):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": int(step),
        "epoch": float(epoch),
        "split": split,
    }

    for key, value in metrics.items():
        payload[key] = to_python_scalar(value)

    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
