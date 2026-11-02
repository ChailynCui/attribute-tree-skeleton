"""Geometry and control evaluation with explicit planned failure denominators."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from tree_skeleton.config import SPECIES, ATTRIBUTES
from tree_skeleton.io import read_json, read_geometry, load_skeletons, write_json
from tree_skeleton.geometry.metrics import (
    sample_skeleton_points,
    measure_geometry,
    build_train_attribute_references,
    train_cdf_percentile,
)
from .fidelity import evaluate_species


def directory(root, row):
    # Keep CPU evaluation independent of torch and its model dependencies.
    import re

    condition = row.get("physical_condition_id", row["condition_id"])
    if row["species"] not in SPECIES or not re.fullmatch(r"[A-Za-z0-9_-]+", condition):
        raise ValueError("Invalid species or condition path")
    return (
        Path(root) / "samples" / row["species"] / condition / f"seed{int(row['generation_seed'])}"
    )


def evaluate(config, requests, generated, output, *, variant="direct", workers=4):
    if variant not in ("raw", "direct"):
        raise ValueError("variant must be raw or direct")
    manifest = read_json(requests)
    attributes = read_json(config["data"]["attributes"])
    refs = build_train_attribute_references(attributes, species_order=SPECIES)
    dataset = load_skeletons(config["data"]["geometry"]) if manifest["protocol"] == "q4" else None
    rows = []
    reference = {s: [] for s in SPECIES}
    predictions = {s: [] for s in SPECIES}
    for plan in manifest["rows"]:
        row = {
            **plan,
            "variant": variant,
            "status": "failed",
            "attribute_hit": False,
            "composite_hit": False,
        }
        species = plan["species"]
        folder = directory(generated, plan)
        if dataset is not None:
            index = int(plan.get("reference_index", plan.get("target_global_index", -1)))
            truth = dataset[index]
            reference[species].append(
                {
                    "points": sample_skeleton_points(
                        truth["vertices"], truth["lines"], point_count=1024
                    ),
                    "structure": measure_geometry(truth["vertices"], truth["lines"])["structure"],
                }
            )
        try:
            sample = read_json(folder / "sample.json")
            if sample.get("status") != "completed":
                raise ValueError("Generation did not complete naturally")
            v, e = read_geometry(folder / f"{variant}_tree.npz")
            measured = measure_geometry(v, e, attribute_definition=attributes["definition"])
            realized = {
                a: train_cdf_percentile(measured["attributes"][a], refs[species][a])
                for a in ATTRIBUTES
            }
            target = plan.get("target_percentiles", plan["attribute_percentiles"])
            errors = {a: abs(realized[a] - float(target[j])) for j, a in enumerate(ATTRIBUTES)}
            finite = all(np.isfinite(x) for x in measured["attributes"].values())
            hit = finite and max(errors.values()) <= 0.15
            row.update(
                status="completed",
                attributes=measured["attributes"],
                percentiles=realized,
                errors=errors,
                structure=measured["structure"],
                attribute_hit=bool(hit),
                composite_hit=bool(hit and measured["structure"]["strict_tree"]),
            )
            if dataset is not None:
                predictions[species].append(
                    {
                        "points": sample_skeleton_points(v, e, point_count=1024),
                        "structure": measured["structure"],
                    }
                )
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)
    planned = len(manifest["rows"])
    complete = [r for r in rows if r["status"] == "completed"]
    result = {
        "variant": variant,
        "protocol": manifest["protocol"],
        "planned_count": planned,
        "completed": len(complete),
        "failure_count": planned - len(complete),
        "failure_rate": (planned - len(complete)) / max(planned, 1),
        "attribute_hit_rate": sum(x["attribute_hit"] for x in rows) / max(planned, 1),
        "composite_hit_rate": sum(x["composite_hit"] for x in rows) / max(planned, 1),
        "mean_absolute_percentile_error_valid_only": {
            a: float(np.mean([x["errors"][a] for x in complete])) if complete else None
            for a in ATTRIBUTES
        },
        "metric_contract": {
            "points_per_tree": 1024,
            "sampling": "edge_arc_length_midpoints",
            "alignment": "none",
            "chamfer": "sum of two mean squared nearest-point distances",
            "hit_denominator": "all planned requests",
            "set_distance_rows": "measurable completed outputs; failure rate reported separately",
        },
        "rows": rows,
    }
    if dataset is not None:
        result["quality_by_species"] = [
            evaluate_species(s, reference[s], predictions[s], workers=workers)
            for s in SPECIES
            if reference[s]
        ]
    else:
        result["marginal_effects"] = marginal_effects(rows)
    write_json(output, result)
    return {
        k: v
        for k, v in result.items()
        if k not in ("rows", "quality_by_species", "marginal_effects")
    }


def marginal_effects(rows, *, bootstrap_samples=10000, seed=1875194580):
    """Paired high-low effects; bootstrap generation seeds, never condition rows."""
    out = []
    rng = np.random.default_rng(seed)
    for species in SPECIES:
        for attribute in ATTRIBUTES:
            values = {}
            for r in rows:
                if (
                    r["species"] == species
                    and r["status"] == "completed"
                    and r["condition_id"] in (attribute + "_low", attribute + "_high")
                ):
                    values.setdefault(r["generation_seed"], {})[r["condition_id"]] = r[
                        "percentiles"
                    ][attribute]
            effects = [
                v[attribute + "_high"] - v[attribute + "_low"]
                for v in values.values()
                if attribute + "_high" in v and attribute + "_low" in v
            ]
            effect = np.asarray(effects)
            ci = None
            if len(effect):
                boot = effect[rng.integers(0, len(effect), (bootstrap_samples, len(effect)))].mean(
                    1
                )
                ci = np.quantile(boot, [0.025, 0.975]).tolist()
            out.append(
                {
                    "species": species,
                    "attribute": attribute,
                    "complete_seed_pairs": len(effects),
                    "mean_high_minus_low": float(effect.mean()) if len(effect) else None,
                    "percentile_bootstrap_95_ci": ci,
                    "incomplete_pairs_excluded_from_effect_but_retained_in_failure_counts": True,
                }
            )
    return out
