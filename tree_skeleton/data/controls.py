"""Materialize the fixed-input, shuffled-correspondence and pooled-CDF arms."""

from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from tree_skeleton.config import SPECIES, ATTRIBUTES
from tree_skeleton.io import read_json, write_json, fingerprint
from .canonical import map_through_train_cdf
from .derangement import deterministic_derangement


def build_controls(attribute_file, split_file, output, *, seed=1351053704):
    base = read_json(attribute_file)
    split = read_json(split_file)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new control artifact directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = base["samples"]
    train = np.array(split["splits"]["train"], dtype=int)
    values = np.array([[x["attributes"][a]["value"] for a in ATTRIBUTES] for x in rows])
    matched = np.array([[x["attributes"][a]["percentile"] for a in ATTRIBUTES] for x in rows])
    pooled = np.column_stack([map_through_train_cdf(values[:, j], train)[0] for j in range(3)])
    shuffled = matched.copy()
    mapping = []
    for species in SPECIES:
        group = [int(i) for i in train if rows[i]["species"] == species]
        for recipient, donor in deterministic_derangement(
            group, species=species, seed=seed
        ).items():
            shuffled[recipient] = matched[donor]
            mapping.append({"recipient": recipient, "donor": donor, "species": species})
    for arm, matrix in [
        ("species_only", np.full_like(matched, 0.5)),
        ("joint_shuffled", shuffled),
        ("global_cdf", pooled),
    ]:
        payload = copy.deepcopy(base)
        for i, row in enumerate(payload["samples"]):
            for j, a in enumerate(ATTRIBUTES):
                row["attributes"][a]["model_input_percentile"] = float(matrix[i, j])
        payload["model_input_artifact"] = {
            "arm": arm,
            "seed": seed,
            "source_attribute_sha256": fingerprint(attribute_file),
            "joint_shuffle_scope": "within species, train only",
            "fixed_scope": "all splits",
            "cdf_scope": "pooled train only",
        }
        write_json(output / f"{arm}_attributes.json", payload)
        adapted = copy.deepcopy(split)
        adapted.setdefault("attributes", {})["file_sha256"] = fingerprint(
            output / f"{arm}_attributes.json"
        )
        write_json(output / f"{arm}_split.json", adapted)
    write_json(output / "derangement.json", {"seed": seed, "mapping": mapping})
    result = {
        "status": "built",
        "arms": ["species_only", "joint_shuffled", "global_cdf"],
        "train_count": len(train),
        "source_attribute_sha256": fingerprint(attribute_file),
        "source_split_sha256": fingerprint(split_file),
    }
    write_json(output / "manifest.json", result)
    return result
