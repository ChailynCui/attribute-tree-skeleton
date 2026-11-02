"""Fixed condition manifests separate generation input from reference geometry."""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from tree_skeleton.config import SPECIES, ATTRIBUTES
from tree_skeleton.io import read_json, write_json


def build_requests(attribute_file, split_file, output, *, protocol="q4", seeds=32, arm="main"):
    attributes = read_json(attribute_file)
    split = read_json(split_file)
    rows = []
    if protocol == "q4":
        for index in sorted(split["splits"]["test"]):
            x = attributes["samples"][index]
            # Matches the established deterministic Q4 seed contract.
            token = f"four_species_q4_condition_matched_v1:{index}".encode("ascii")
            seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**31 - 1) + 1
            rows.append(
                {
                    "species": x["species"],
                    "species_id": SPECIES.index(x["species"]),
                    "condition_id": f"test_{index:05d}",
                    "reference_index": index,
                    "generation_seed": seed,
                    "attribute_percentiles": [x["attributes"][a]["percentile"] for a in ATTRIBUTES],
                }
            )
    elif protocol == "marginal7":
        conditions = {"center": [0.5, 0.5, 0.5]}
        for j, a in enumerate(ATTRIBUTES):
            for label, value in [("low", 0.25), ("high", 0.75)]:
                u = [0.5, 0.5, 0.5]
                u[j] = value
                conditions[f"{a}_{label}"] = u
        seed_plan = read_json(Path(__file__).parents[1] / "evaluation_seeds.json")["generation"]
        if not 1 <= seeds <= len(seed_plan):
            raise ValueError("Choose 1 to 32 published generation seeds")
        for sid, species in enumerate(SPECIES):
            for name, u in conditions.items():
                for seed in seed_plan[:seeds]:
                    rows.append(
                        {
                            "species": species,
                            "species_id": sid,
                            "condition_id": name,
                            "generation_seed": seed,
                            "attribute_percentiles": u,
                        }
                    )
    else:
        raise ValueError("protocol must be q4 or marginal7")
    from tree_skeleton.geometry.metrics import (
        build_train_attribute_references,
        train_cdf_percentile,
    )

    refs = build_train_attribute_references(attributes, species_order=SPECIES)
    pooled = {
        a: np.sort(
            [x["attributes"][a]["value"] for x in attributes["samples"] if x["split"] == "train"]
        )
        for a in ATTRIBUTES
    }
    for row in rows:
        target = list(row["attribute_percentiles"])
        row["target_percentiles"] = target
        if protocol == "q4":
            source = attributes["samples"][row["reference_index"]]
            physical = [source["attributes"][a]["value"] for a in ATTRIBUTES]
        else:
            physical = [
                float(np.quantile(refs[row["species"]][a], target[j]))
                for j, a in enumerate(ATTRIBUTES)
            ]
        row["target_physical_values"] = dict(zip(ATTRIBUTES, map(float, physical)))
        if arm == "global_cdf":
            row["attribute_percentiles"] = [
                train_cdf_percentile(physical[j], pooled[a]) for j, a in enumerate(ATTRIBUTES)
            ]
        elif arm == "species_only":
            row["attribute_percentiles"] = [0.5, 0.5, 0.5]
            if protocol == "marginal7":
                row["physical_condition_id"] = "center"
        elif arm not in ("main", "joint_shuffled"):
            raise ValueError("Unknown control arm")
    result = {
        "format": "tree_skeleton_requests_v1",
        "protocol": protocol,
        "arm": arm,
        "planned_count": len(rows),
        "planned_physical_samples": len(
            {
                (
                    r["species"],
                    r.get("physical_condition_id", r["condition_id"]),
                    r["generation_seed"],
                )
                for r in rows
            }
        ),
        "rows": rows,
    }
    write_json(output, result)
    return result
