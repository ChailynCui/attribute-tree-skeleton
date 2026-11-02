"""Deterministic retrieval and fixed-tree scaling reference experiments."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from tree_skeleton.config import SPECIES, ATTRIBUTES
from tree_skeleton.io import read_json, load_skeletons, write_json, write_geometry
from tree_skeleton.geometry.metrics import (
    measure_geometry,
    build_train_attribute_references,
    train_cdf_percentile,
)


def evaluate_references(config, requests, output):
    attributes = read_json(config["data"]["attributes"])
    data = load_skeletons(config["data"]["geometry"])
    refs = build_train_attribute_references(attributes, species_order=SPECIES)
    rows = []
    seen = set()
    for plan in read_json(requests)["rows"]:
        key = (plan["species"], plan["condition_id"])
        if key in seen:
            continue
        seen.add(key)
        species = plan["species"]
        target = np.asarray(plan.get("target_percentiles", plan["attribute_percentiles"]))
        candidates = [
            r for r in attributes["samples"] if r["species"] == species and r["split"] == "train"
        ]
        candidates.sort(key=lambda r: r["global_index"])
        matrix = np.array(
            [[r["attributes"][a]["percentile"] for a in ATTRIBUTES] for r in candidates]
        )
        nearest = int(np.argmin(np.square(matrix - target).sum(1)))
        center = int(np.argmin(np.square(matrix - 0.5).sum(1)))
        for method, choice in [("retrieval", nearest), ("fixed_tree_scaling", center)]:
            index = int(candidates[choice]["global_index"])
            tree = data[index]
            v = np.asarray(tree["vertices"], dtype=np.float64).copy()
            e = np.asarray(tree["lines"], dtype=np.int64)
            row = {
                "species": species,
                "condition_id": plan["condition_id"],
                "method": method,
                "training_source_index": index,
                "status": "failed",
                "attribute_hit": False,
                "composite_hit": False,
            }
            try:
                if method == "fixed_tree_scaling":
                    base = measure_geometry(v, e, attribute_definition=attributes["definition"])[
                        "attributes"
                    ]
                    physical = {
                        a: float(np.quantile(refs[species][a], target[j]))
                        for j, a in enumerate(ATTRIBUTES)
                    }
                    if min(base["height"], base["crown_width"]) <= 0:
                        raise ValueError("Degenerate fixed reference dimensions")
                    xy = np.median(v[:, :2], axis=0)
                    z = float(v[:, 2].min())
                    v[:, :2] = xy + (v[:, :2] - xy) * (
                        physical["crown_width"] / base["crown_width"]
                    )
                    v[:, 2] = z + (v[:, 2] - z) * (physical["height"] / base["height"])
                measured = measure_geometry(v, e, attribute_definition=attributes["definition"])
                realized = np.array(
                    [
                        train_cdf_percentile(measured["attributes"][a], refs[species][a])
                        for a in ATTRIBUTES
                    ]
                )
                error = np.abs(realized - target)
                hit = bool(np.isfinite(error).all() and max(error) <= 0.15 + 1e-12)
                row.update(
                    status="completed",
                    target_percentiles=target.tolist(),
                    realized_percentiles=realized.tolist(),
                    mean_absolute_error=float(error.mean()),
                    attribute_hit=hit,
                    composite_hit=bool(hit and measured["structure"]["strict_tree"]),
                )
                write_geometry(
                    Path(output) / method / species / f"{plan['condition_id']}.npz", v, e
                )
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
    result = {
        "rows": rows,
        "deterministic_cases_per_method": len(seen),
        "no_generated_seed_replication": True,
    }
    result["methods"] = {}
    for method in ("retrieval", "fixed_tree_scaling"):
        group = [r for r in rows if r["method"] == method]
        valid = [r for r in group if r["status"] == "completed"]
        result["methods"][method] = {
            "planned": len(group),
            "completed": len(valid),
            "mean_absolute_error_valid_only": float(
                np.mean([r["mean_absolute_error"] for r in valid])
            )
            if valid
            else None,
            "attribute_hit_rate": sum(r["attribute_hit"] for r in group) / max(len(group), 1),
            "composite_hit_rate": sum(r["composite_hit"] for r in group) / max(len(group), 1),
        }
    write_json(Path(output) / "summary.json", result)
    return result["methods"]
