"""Build external experiment assets, or verify an existing frozen release."""

from __future__ import annotations
import copy
from pathlib import Path
import pickle
import numpy as np
from tree_skeleton.config import SPECIES, ATTRIBUTES
from tree_skeleton.io import read_json, write_json, load_skeletons, fingerprint
from tree_skeleton.geometry.metrics import measure_geometry
from .canonical import (
    inverse_normalization,
    repair_strict_tree,
    stratified_species_split,
    map_through_train_cdf,
)
from .scaling import fit_category_scales, indices_fingerprint, coordinate_bounds

GEOMETRY = "four_species_12000_species_global_strict.pkl"
ATTR_FILE = "four_species_12000_attributes_v2p2.json"
SPLIT_FILE = "four_species_12000_split_seed20260805.json"


def verify_assets(config):
    d = config["data"]
    expected_scope = d.get("coordinate_scale_fit", "train")
    protocol_path = Path(
        d.get("coordinate_protocol") or Path(d["geometry"]).parent / "coordinate_protocol.json"
    )
    protocol = read_json(protocol_path) if protocol_path.is_file() else None
    if expected_scope == "train" and not protocol:
        raise ValueError(
            "Training-only coordinates require coordinate_protocol.json. Prepare matching external assets with --scale-fit train before using this configuration."
        )
    if protocol and protocol.get("scale_fit") != expected_scope:
        raise ValueError("Coordinate scale fitting scope differs from this configuration")
    rows = load_skeletons(d["geometry"])
    attrs = read_json(d["attributes"])
    split = read_json(d["split"])
    if len(attrs["samples"]) != len(rows):
        raise ValueError("Attribute and geometry row counts differ")
    groups = {k: list(map(int, split["splits"][k])) for k in ("train", "validation", "test")}
    flat = sum(groups.values(), [])
    if len(flat) != len(rows) or set(flat) != set(range(len(rows))) or len(set(flat)) != len(flat):
        raise ValueError("Split must cover every row exactly once with no overlap")
    import hashlib

    name_hash = hashlib.sha256(("\n".join(x["name"] for x in rows) + "\n").encode()).hexdigest()
    expected = split.get("dataset", {}).get("ordered_name_sha256")
    if expected and expected != name_hash:
        raise ValueError("Geometry ordering differs from the frozen split")
    for key, section in [("geometry", "dataset"), ("attributes", "attributes")]:
        expected = split.get(section, {}).get("file_sha256")
        if expected and fingerprint(d[key]) != expected:
            raise ValueError(f"{section} file hash differs from frozen manifest")
    membership = {i: k for k, indices in groups.items() for i in indices}
    for i, a in enumerate(attrs["samples"]):
        if int(a.get("global_index", i)) != i or a["split"] != membership[i]:
            raise ValueError(f"Attribute identity or split mismatch at {i}")
        if a["species"] != rows[i]["meta"]["species"]:
            raise ValueError(f"Species mismatch at {i}")
    if protocol:
        if protocol.get("geometry_sha256") != fingerprint(d["geometry"]):
            raise ValueError("Coordinate protocol belongs to a different geometry artifact")
        if protocol.get("train_indices_sha256") != indices_fingerprint(groups["train"]):
            raise ValueError("Scale fitting used a different training partition")
        for name, fitted in protocol["categories"].items():
            fit_indices = [
                i
                for i in (groups["train"] if expected_scope == "train" else range(len(rows)))
                if rows[i]["meta"]["species"] == name
            ]
            if fitted["fit_count"] != len(fit_indices) or fitted[
                "fit_indices_sha256"
            ] != indices_fingerprint(fit_indices):
                raise ValueError("Category scale provenance disagrees with the fitting partition")
        for i, row in enumerate(rows):
            stored = row.get("preprocess", {})
            fitted = protocol["categories"][row["meta"]["species"]]
            if (
                stored.get("scale_fit") != expected_scope
                or stored.get("normalization_scale") != fitted["scale"]
            ):
                raise ValueError(
                    f"Per-tree coordinate scale differs from frozen category scale at {i}"
                )
    bounds = coordinate_bounds(rows, groups)
    if any(
        x["outside_model_range_count"] or x["nonfinite_indices"] for x in bounds["splits"].values()
    ):
        raise ValueError(
            "Coordinates exceed the model range [-1,1] or are nonfinite. No clipping was applied. See the preparation bounds report; choose any margin/range explicitly and refit only on training data."
        )
    return {
        "status": "verified",
        "samples": len(rows),
        "split_counts": {k: len(v) for k, v in groups.items()},
        "sha256": {k: fingerprint(d[k]) for k in ("geometry", "attributes", "split")},
        "geometry_scale_fit": expected_scope,
        "coordinate_protocol_verified": protocol is not None,
        "coordinate_bounds": bounds,
        "attribute_cdf_fit": "training split only",
    }


def prepare_sources(
    source_manifest, output, definition, *, split_path=None, scale_fit="train", scale_target=0.99
):
    """Inputs are external source pickles; this is not a raw Blender generator.

    The split is frozen before fitting one scale per category on train only.
    Explicit scope all is retained solely for historical compatibility.
    """
    source_manifest = Path(source_manifest).resolve()
    spec = read_json(source_manifest)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prepare into a new, empty external data directory")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    raw = []
    for sid, species in enumerate(SPECIES):
        source = Path(spec["sources"][species])
        if not source.is_absolute():
            source = source_manifest.parent / source
        samples = load_skeletons(source)
        if len(samples) != 3000:
            raise ValueError(f"{species}: historical preparation requires 3000 source trees")
        for local, sample in enumerate(samples):
            row = copy.deepcopy(sample)
            row["id"] = len(records)
            row["name"] = f"four_species_3000_v1_{species}_{local:05d}"
            row["meta"] = {
                **row.get("meta", {}),
                "species": species,
                "species_id": sid,
                "species_local_index": local,
            }
            restored, _ = inverse_normalization(sample)
            records.append(row)
            raw.append(restored)
    if split_path:
        groups = read_json(split_path)["splits"]
    else:
        rng = np.random.default_rng(20260805)
        groups = {"train": [], "validation": [], "test": []}
        for sid, _ in enumerate(SPECIES):
            parts, _ = stratified_species_split(
                records, np.arange(sid * 3000, (sid + 1) * 3000), rng
            )
            for key, value in parts.items():
                groups[key].extend(map(int, value))
    groups = {k: sorted(map(int, v)) for k, v in groups.items()}
    flat = sum(groups.values(), [])
    if len(flat) != len(records) or set(flat) != set(range(len(records))):
        raise ValueError("The supplied split is not a disjoint full partition")
    train = set(groups["train"])
    protocol = fit_category_scales(
        raw,
        [x["meta"]["species"] for x in records],
        groups["train"],
        target=scale_target,
        scope=scale_fit,
    )
    for sid, species in enumerate(SPECIES):
        ids = range(sid * 3000, (sid + 1) * 3000)
        scale = protocol["categories"][species]["scale"]
        for i in ids:
            records[i]["vertices"] = (raw[i] * scale).astype(np.float32)
            lines, repair = repair_strict_tree(
                records[i]["vertices"], records[i].get("lines", records[i].get("edges"))
            )
            records[i]["lines"] = lines
            records[i]["edges"] = lines.copy()
            records[i]["meta"]["topology_repair"] = repair
            records[i]["preprocess"] = {
                "normalization_scale": scale,
                "normalization_center": np.zeros(3),
                "normalization_basis": "species_global_max_abs_raw",
                "scale_fit": scale_fit,
                "normalize_target": scale_target,
            }
    measured = [
        measure_geometry(x["vertices"], x["lines"], attribute_definition=definition)["attributes"]
        for x in records
    ]
    values = np.asarray([[x[a] for a in ATTRIBUTES] for x in measured])
    percentiles = np.empty_like(values)
    fits = {}
    for sid, species in enumerate(SPECIES):
        ids = np.arange(sid * 3000, (sid + 1) * 3000)
        ti = np.array([i for i in ids if i in train])
        fits[species] = {}
        for j, a in enumerate(ATTRIBUTES):
            mapped, fit = map_through_train_cdf(values[:, j], ti)
            percentiles[ids, j] = mapped[ids]
            fits[species][a] = fit
    membership = {i: k for k, ids in groups.items() for i in ids}
    attr_rows = []
    for i, x in enumerate(records):
        attr_rows.append(
            {
                "global_index": i,
                "tree_id": f"{x['meta']['species']}/{x['meta']['species_local_index']:05d}",
                "source_name": x["name"],
                "species": x["meta"]["species"],
                "species_id": x["meta"]["species_id"],
                "split": membership[i],
                "attributes": {
                    a: {"value": float(values[i, j]), "percentile": float(percentiles[i, j])}
                    for j, a in enumerate(ATTRIBUTES)
                },
            }
        )
    with (output / GEOMETRY).open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    attributes = {
        "sample_count": len(records),
        "definition": definition,
        "species_vocabulary": dict(zip(SPECIES, range(4))),
        "attribute_names": list(ATTRIBUTES),
        "percentile_fit": fits,
        "samples": attr_rows,
    }
    write_json(output / ATTR_FILE, attributes)
    import hashlib

    names = hashlib.sha256(("\n".join(x["name"] for x in records) + "\n").encode()).hexdigest()
    write_json(
        output / SPLIT_FILE,
        {
            "sample_count": len(records),
            "splits": groups,
            "dataset": {
                "sample_count": len(records),
                "file_sha256": fingerprint(output / GEOMETRY),
                "ordered_name_sha256": names,
            },
            "attributes": {"file_sha256": fingerprint(output / ATTR_FILE)},
        },
    )
    protocol["geometry_sha256"] = fingerprint(output / GEOMETRY)
    bounds = coordinate_bounds(records, groups)
    protocol["coordinate_bounds"] = bounds
    write_json(output / "coordinate_protocol.json", protocol)
    return {
        "status": "prepared",
        "samples": len(records),
        "output": str(output.resolve()),
        "scale_fit": scale_fit,
        "model_range_compatible": not any(
            x["outside_model_range_count"] or x["nonfinite_indices"]
            for x in bounds["splits"].values()
        ),
        "coordinates_clipped": False,
    }
