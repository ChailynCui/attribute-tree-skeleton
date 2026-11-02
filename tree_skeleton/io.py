"""Portable artifact IO. Data and checkpoints are always supplied by the user."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pickle
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temp, path)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load_skeletons(path):
    # Pickles and torch checkpoints must come from a trusted source.
    with Path(path).open("rb") as f:
        rows = pickle.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of skeleton dictionaries")
    return rows


def read_geometry(path):
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["vertices"], dtype=np.float64), np.asarray(z["lines"], dtype=np.int64)


def write_geometry(path, vertices, lines, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, vertices=np.asarray(vertices), lines=np.asarray(lines), **extra)
    os.replace(temp, path)
