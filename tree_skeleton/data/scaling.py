"""Fit once on training geometry and reuse the frozen category coefficients."""

from __future__ import annotations
import hashlib
import numpy as np
from tree_skeleton.config import SPECIES


def indices_fingerprint(indices):
    text = "".join(f"{int(i)}\n" for i in sorted(indices))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def fit_category_scales(vertices, species, train_indices, *, target=0.99, scope="train"):
    """Held-out coordinate values are not visited while fitting in train mode."""
    if scope not in ("train", "all"):
        raise ValueError("Scale scope must be train or explicit legacy all")
    if not 0 < target < 1:
        raise ValueError("Scale target must be between zero and one")
    if len(vertices) != len(species):
        raise ValueError("Geometry and species counts differ")
    train = list(map(int, train_indices))
    if len(train) != len(set(train)) or any(i < 0 or i >= len(vertices) for i in train):
        raise ValueError("Invalid training indices")
    allowed = sorted(train) if scope == "train" else list(range(len(vertices)))
    fitted = {}
    for name in SPECIES:
        indices = [i for i in allowed if species[i] == name]
        if not indices:
            raise ValueError(f"No {scope} geometry for {name}")
        extent = 0.0
        for i in indices:
            points = np.asarray(vertices[i], dtype=np.float64)
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or not len(points)
                or not np.isfinite(points).all()
            ):
                raise ValueError(f"Invalid scale-fitting geometry at {i}")
            extent = max(extent, float(np.abs(points).max()))
        if extent <= 0:
            raise ValueError(f"Zero extent for {name}")
        fitted[name] = {
            "scale": float(target / extent),
            "fit_extent": extent,
            "fit_count": len(indices),
            "fit_indices_sha256": indices_fingerprint(indices),
        }
    return {
        "version": "tree_skeleton_coordinates_v2",
        "scale_fit": scope,
        "target": target,
        "train_indices_sha256": indices_fingerprint(train),
        "categories": fitted,
        "source_upstream_shape_calibrations_retained": True,
        "attribute_cdf_fit": "train only",
        "historical_pickle_byte_identity_claimed": False,
    }


def coordinate_bounds(records, groups, *, limit=1.0):
    """Report out-of-range coordinates without changing scales or clipping."""
    result = {}
    for split, indices in groups.items():
        maximum = 0.0
        outside = []
        nonfinite = []
        for i in indices:
            points = np.asarray(records[int(i)]["vertices"])
            if not np.isfinite(points).all():
                nonfinite.append(int(i))
                continue
            extent = float(np.abs(points).max())
            maximum = max(maximum, extent)
            if extent > limit:
                outside.append(int(i))
        result[split] = {
            "max_abs_coordinate": maximum,
            "outside_model_range_count": len(outside),
            "outside_model_range_indices": outside,
            "nonfinite_indices": nonfinite,
        }
    return {"model_abs_limit": limit, "clipping_applied": False, "splits": result}
