"""Flat, explicit experiment specifications without an inheritance tree."""

from __future__ import annotations
import copy
from pathlib import Path
import yaml

SPECIES = ("callistemon", "japanese_maple", "weeping_willow", "white_birch")
ATTRIBUTES = ("height", "crown_width", "branch_density")


def load_config(path, *, data_root=None):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("format") != "tree_skeleton_experiment_v1":
        raise ValueError("Expected format: tree_skeleton_experiment_v1")
    if "inherit_from" in config:
        raise ValueError("Release configurations must be explicit, without inheritance")
    config = copy.deepcopy(config)
    root = Path(data_root or config["data"].get("root", "external_data")).expanduser().resolve()
    for key in ("geometry", "attributes", "split"):
        p = Path(config["data"][key])
        config["data"][key] = str(p if p.is_absolute() else root / p)
    if config["data"].get("coordinate_protocol"):
        p = Path(config["data"]["coordinate_protocol"])
        config["data"]["coordinate_protocol"] = str(p if p.is_absolute() else root / p)
    if config["data"].get("coordinate_scale_fit", "train") not in ("train", "all"):
        raise ValueError("coordinate_scale_fit must be train or explicit legacy all")
    config["data"]["root"] = str(root)
    if tuple(config["data"]["species"]) != SPECIES:
        raise ValueError("Species ID order must match the published protocol")
    if tuple(config["data"]["attribute_names"]) != ATTRIBUTES:
        raise ValueError("Attribute order must be height, crown_width, branch_density")
    if config["data"].get("augmentation") != "xy_dihedral":
        raise ValueError("The published experiments use XY dihedral augmentation")
    return config
