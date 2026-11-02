"""Exhaustive generated/test-to-training Chamfer proximity, without alignment."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from tree_skeleton.config import SPECIES
from tree_skeleton.io import read_json, read_geometry, load_skeletons, write_json
from tree_skeleton.geometry.metrics import sample_skeleton_points
from .fidelity import _pair_distance
from .evaluate import directory


def evaluate_proximity(config, requests, generated, output):
    data = load_skeletons(config["data"]["geometry"])
    split = read_json(config["data"]["split"])
    plans = read_json(requests)
    if plans["protocol"] != "q4":
        raise ValueError("Proximity requires held-out Q4 references")
    test = set(split["splits"]["test"])
    rows = []
    for species in SPECIES:
        indices = [
            i for i in sorted(split["splits"]["train"]) if data[i]["meta"]["species"] == species
        ]
        points = [
            sample_skeleton_points(data[i]["vertices"], data[i]["lines"], point_count=1024)
            for i in indices
        ]
        trees = [cKDTree(p) for p in points]
        for plan in plans["rows"]:
            if plan["species"] != species:
                continue
            index = int(plan["reference_index"])
            if index not in test:
                raise ValueError("Reference is outside the test split")
            for group in ("test", "generated"):
                row = {
                    "species": species,
                    "condition_id": plan["condition_id"],
                    "source_set": group,
                    "status": "failed",
                }
                try:
                    if group == "test":
                        v, e = data[index]["vertices"], data[index]["lines"]
                    else:
                        folder = directory(generated, plan)
                        if read_json(folder / "sample.json")["status"] != "completed":
                            raise ValueError("Incomplete generation")
                        v, e = read_geometry(folder / "direct_tree.npz")
                    query = sample_skeleton_points(v, e, point_count=1024)
                    tree = cKDTree(query)
                    distances = [_pair_distance(query, p, tree, t) for p, t in zip(points, trees)]
                    best = int(np.argmin(distances))
                    row.update(
                        status="completed",
                        nearest_training_index=indices[best],
                        symmetric_squared_chamfer=distances[best],
                        training_candidates=len(indices),
                    )
                except Exception as error:
                    row["error"] = str(error)
                rows.append(row)
            write_json(
                output,
                {
                    "planned_count": 2 * len(plans["rows"]),
                    "rows": rows,
                    "search": "exhaustive same-category training set",
                    "point_count": 1024,
                    "alignment": "none",
                    "finished": False,
                },
            )
    result = {
        "planned_count": 2 * len(plans["rows"]),
        "rows": rows,
        "search": "exhaustive same-category training set",
        "point_count": 1024,
        "alignment": "none",
        "finished": True,
    }
    write_json(output, result)
    return {
        "planned_count": result["planned_count"],
        "completed": sum(r["status"] == "completed" for r in rows),
    }
