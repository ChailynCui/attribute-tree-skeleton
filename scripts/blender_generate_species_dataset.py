"""Generate a 3000-tree white-birch skeleton dataset in Blender.

Enable the Sapling Tree Gen add-on before running this script. The white-birch
preset is embedded here; no project modules or configuration scripts are
required. This script consumes an existing parameter-manifest CSV.

Blender GUI:
    Open this .py file in the Scripting workspace. Place
    white_birch_3000_manifest.csv beside it, or edit the relative INPUT_CSV.
    GUI_MODE selects the first 500 rows (pilot), the remaining 2500 rows
    (remainder), or all 3000 rows (full). Output defaults to white_birch_raw_3000
    beside the script. If pasted into an unsaved text block, paths are relative to
    the saved .blend folder, or the working directory for an unsaved .blend.

Command line, from the project directory:
    blender --background --python scripts/blender_generate_species_dataset.py -- \
      --input-csv scripts/white_birch_3000_manifest.csv \
      --output-dir outputs/white_birch_raw --start 0 --count 3000

Command-line paths are relative to the current working directory. The CSV
must contain only white_birch rows. Exported edge indices are 1-based. Target levels
are parameter-design strata, not measured morphology labels.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

import bpy


# GUI settings: paths below are relative to this script's directory.
GUI_MODE = "pilot"  # pilot, remainder, or full
INPUT_CSV = "white_birch_3000_manifest.csv"
LOCAL_OUTPUT_DIR = "white_birch_raw_3000"
PILOT_COUNT = 500
TOTAL_COUNT = 3000
FORCE_NO_LEAVES = True


# Embedded white-birch preset; per-tree parameters come from the manifest CSV.
PER_TREE_FIELDS = (
    "scale",
    "length",
    "branches",
    "downAngle",
    "downAngleV",
    "attractUp",
    "attractOut",
    "segSplits",
    "splitAngle",
    "splitAngleV",
    "levels",
)


TUPLE_FIELDS = {
    "length",
    "branches",
    "downAngle",
    "downAngleV",
    "attractUp",
    "attractOut",
    "segSplits",
    "splitAngle",
    "splitAngleV",
}


INTEGER_TUPLE_FIELDS = {"branches"}


PRESET = {
    "seed": 0,
    "showLeaves": True,
    "armLevels": 0,
    "leafDist": "6",
    "baseSize": 0.3499999940395355,
    "loopFrames": 0,
    "af3": 4.0,
    "previewArm": False,
    "leafangle": -45.0,
    "useParentAngle": True,
    "handleType": "0",
    "branches": (0, 60, 30, 10),
    "autoTaper": True,
    "splitAngle": (12.0, 18.0, 16.0, 0.0),
    "baseSize_s": 0.800000011920929,
    "closeTip": False,
    "af2": 1.0,
    "prune": False,
    "scale0": 1.0,
    "rMode": "rotate",
    "useOldDownAngle": False,
    "scaleV0": 0.10000000149011612,
    "splitBias": 0.0,
    "resU": 2,
    "curveBack": (0.0, -5.0, 0.0, 0.0),
    "scale": 12.0,
    "shape": "8",
    "leafDownAngle": 45.0,
    "af1": 1.0,
    "ratio": 0.019999999552965164,
    "horzLeaves": True,
    "leafRotate": 137.5,
    "minRadius": 0.0020000000949949026,
    "bevelRes": 2,
    "splitByLen": True,
    "rootFlare": 1.149999976158142,
    "makeMesh": False,
    "downAngleV": (0.0, 25.0, 30.0, 10.0),
    "levels": 2,
    "scaleV": 2.0,
    "armAnim": False,
    "lengthV": (0.05000000074505806, 0.20000000298023224, 0.3499999940395355, 0.0),
    "pruneWidth": 0.3100000023841858,
    "gustF": 0.07500000298023224,
    "taper": (1.0, 1.0, 1.0, 1.0),
    "splitAngleV": (2.0, 2.0, 0.0, 0.0),
    "prunePowerLow": 0.0010000000474974513,
    "leafScaleT": 0.20000000298023224,
    "leafScaleX": 0.5,
    "leafRotateV": 0.0,
    "ratioPower": 1.399999976158142,
    "segSplits": (0.3499999940395355, 0.3499999940395355, 0.3499999940395355, 0.0),
    "downAngle": (90.0, 60.0, 50.0, 45.0),
    "rotateV": (0.0, 0.0, 0.0, 0.0),
    "gust": 1.0,
    "attractUp": (0.0, -1.0, -0.6499999761581421, 0.0),
    "leafScaleV": 0.25,
    "frameRate": 1.0,
    "curveV": (100.0, 80.0, 80.0, 0.0),
    "boneStep": (1, 1, 1, 1),
    "customShape": (0.699999988079071, 1.0, 0.30000001192092896, 0.5900000333786011),
    "pruneBase": 0.30000001192092896,
    "leafAnim": False,
    "curveRes": (10, 8, 3, 1),
    "nrings": 0,
    "bevel": True,
    "taperCrown": 0.0,
    "baseSplits": 2,
    "leafShape": "hex",
    "splitHeight": 0.550000011920929,
    "wind": 1.0,
    "curve": (0.0, -30.0, -25.0, 0.0),
    "rotate": (137.5, 137.5, 137.5, 137.5),
    "length": (1.0, 0.33000001311302185, 0.375, 0.44999998807907104),
    "leafScale": 0.20000000298023224,
    "attractOut": (0.0, 0.20000000298023224, 0.25, 0.0),
    "prunePowerHigh": 0.10000000149011612,
    "branchDist": 1.5,
    "useArm": False,
    "pruneRatio": 1.0,
    "shapeS": "7",
    "leafDownAngleV": 10.0,
    "pruneWidthPeak": 0.5,
    "radiusTweak": (1.0, 1.0, 1.0, 1.0),
    "leaves": 16,
}


SPECIES = "white_birch"
BRANCH_LEVELS = 2


def script_arguments() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def local_script_directory() -> Path:
    """Locate the script without machine-specific paths or import searches."""
    raw_path = globals().get("__file__")
    if raw_path:
        resolved = Path(bpy.path.abspath(str(raw_path))).expanduser().resolve()
        if resolved.is_file():
            return resolved.parent

    space_data = getattr(bpy.context, "space_data", None)
    text_block = getattr(space_data, "text", None)
    text_path = getattr(text_block, "filepath", "")
    if text_path:
        resolved = Path(bpy.path.abspath(text_path)).expanduser().resolve()
        if resolved.is_file():
            return resolved.parent

    blend_path = getattr(bpy.data, "filepath", "")
    if blend_path:
        return Path(bpy.path.abspath(blend_path)).expanduser().resolve().parent
    return Path.cwd()


SCRIPT_DIR = local_script_directory()


def parse_args() -> argparse.Namespace:
    arguments = script_arguments()
    if not arguments:
        manifest = Path(INPUT_CSV).expanduser()
        if not manifest.is_absolute():
            manifest = SCRIPT_DIR / manifest
        output_dir = SCRIPT_DIR / LOCAL_OUTPUT_DIR
        if GUI_MODE == "pilot":
            return argparse.Namespace(
                input_csv=manifest,
                output_dir=output_dir,
                start=0,
                count=PILOT_COUNT,
                overwrite=False,
            )
        if GUI_MODE == "remainder":
            return argparse.Namespace(
                input_csv=manifest,
                output_dir=output_dir,
                start=PILOT_COUNT,
                count=TOTAL_COUNT - PILOT_COUNT,
                overwrite=False,
            )
        if GUI_MODE == "full":
            return argparse.Namespace(
                input_csv=manifest,
                output_dir=output_dir,
                start=0,
                count=TOTAL_COUNT,
                overwrite=False,
            )
        raise ValueError(
            f"invalid GUI_MODE={GUI_MODE!r}; use pilot, remainder, or full"
        )

    parser = argparse.ArgumentParser(
        description="Generate white-birch skeletons using Blender Sapling."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def parse_tuple(raw_value: str, *, integer: bool) -> tuple:
    value = ast.literal_eval(raw_value)
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"expected tuple/list, got {raw_value!r}")
    if integer:
        return tuple(int(round(float(item))) for item in value)
    return tuple(float(item) for item in value)


def parse_manifest_value(field: str, raw_value: str):
    if field in TUPLE_FIELDS:
        return parse_tuple(raw_value, integer=field in INTEGER_TUPLE_FIELDS)
    if field == "levels":
        return int(raw_value)
    return float(raw_value)


def load_manifest(path: Path) -> tuple[str, list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {
        "idx",
        "seed",
        "species",
        "stage",
        "height_level",
        "crown_width_level",
        "branch_density_level",
        *PER_TREE_FIELDS,
    }
    missing = required - set(source_rows[0] if source_rows else ())
    if missing:
        raise KeyError(f"manifest missing columns: {sorted(missing)}")
    species_values = {row["species"] for row in source_rows}
    if len(species_values) != 1:
        raise ValueError("one manifest must contain exactly one species")
    species = next(iter(species_values))
    if species != SPECIES:
        raise ValueError(f"this script requires species={SPECIES}, got {species}")

    seen_indices = set()
    rows = []
    for row_number, source in enumerate(source_rows, start=2):
        try:
            index = int(source["idx"])
            seed = int(source["seed"])
            if index in seen_indices:
                raise ValueError(f"duplicate idx {index}")
            seen_indices.add(index)
            overrides = {
                field: parse_manifest_value(field, source[field])
                for field in PER_TREE_FIELDS
            }
            if overrides["levels"] != BRANCH_LEVELS:
                raise ValueError(
                    f"{species} must use levels={BRANCH_LEVELS}, "
                    f"got {overrides['levels']}"
                )
        except Exception as error:
            raise ValueError(
                f"invalid manifest row {row_number}: {error}"
            ) from error
        rows.append(
            {
                "idx": index,
                "seed": seed,
                "overrides": overrides,
                "source": source,
            }
        )
    return species, rows


def view3d_override():
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "WINDOW":
                    return {
                        "window": window,
                        "screen": screen,
                        "area": area,
                        "region": region,
                    }
    return None


def operator_properties():
    if not hasattr(bpy.ops.curve, "tree_add"):
        raise RuntimeError(
            "bpy.ops.curve.tree_add is unavailable. Enable Sapling Tree Gen."
        )
    properties = bpy.ops.curve.tree_add.get_rna_type().properties
    return properties, set(properties.keys())


def coerce_to_operator_types(kwargs: dict, properties, valid_keys: set) -> dict:
    fixed = {}
    for key, value in kwargs.items():
        if key not in valid_keys:
            continue
        prop = properties.get(key)
        array_length = getattr(prop, "array_length", 0)
        if array_length > 0:
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            values = values[:array_length]
            while len(values) < array_length:
                values.append(values[-1] if values else 0.0)
            fixed[key] = tuple(values)
        else:
            fixed[key] = value
    return fixed


def add_tree(kwargs: dict, override):
    before = set(bpy.data.objects)
    if override is None:
        bpy.ops.curve.tree_add(**kwargs)
    else:
        with bpy.context.temp_override(**override):
            bpy.ops.curve.tree_add(**kwargs)
    new_objects = list(set(bpy.data.objects) - before)
    tree = next((item for item in new_objects if item.type == "CURVE"), None)
    if tree is None and new_objects:
        tree = new_objects[0]
    if tree is None:
        raise RuntimeError("Sapling did not create an object")
    return tree, new_objects


def export_evaluated_skeleton(
    tree,
    node_path: Path,
    connection_path: Path,
) -> tuple[int, int, int]:
    if tree.type == "CURVE":
        tree.data.dimensions = "3D"
        tree.data.bevel_depth = 0.0
        tree.data.bevel_resolution = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = tree.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None:
        raise RuntimeError("could not evaluate the Sapling curve as a mesh")
    world_matrix = tree.matrix_world.copy()
    try:
        with node_path.open("w", encoding="utf-8") as handle:
            for vertex in mesh.vertices:
                coordinate = world_matrix @ vertex.co
                handle.write(f"{coordinate.x},{coordinate.y},{coordinate.z}\n")
        with connection_path.open("w", encoding="utf-8") as handle:
            for edge in mesh.edges:
                first, second = edge.vertices
                handle.write(f"{first + 1},{second + 1}\n")

        parent = list(range(len(mesh.vertices)))

        def find(vertex):
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex

        for edge in mesh.edges:
            first, second = (int(value) for value in edge.vertices)
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[second_root] = first_root
        components = len({find(index) for index in range(len(mesh.vertices))})
        return len(mesh.vertices), len(mesh.edges), components
    finally:
        evaluated.to_mesh_clear()


def remove_objects(objects) -> None:
    for item in objects:
        if item.name in bpy.data.objects:
            bpy.data.objects.remove(item, do_unlink=True)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    species, rows = load_manifest(input_csv)
    if args.count is not None and args.count < 0:
        raise ValueError("--count must be non-negative")
    if args.start < 0 or args.start > len(rows):
        raise ValueError("--start is outside the manifest")
    stop = len(rows) if args.count is None else min(
        len(rows), args.start + args.count
    )
    selected = rows[args.start:stop]
    properties, valid_keys = operator_properties()
    override = view3d_override()
    if override is None:
        print("[WARN] no VIEW_3D context; attempting direct operator calls")

    log_path = output_dir / "generation_log.csv"
    write_header = not log_path.exists()
    failures = 0
    with log_path.open("a", encoding="utf-8", newline="") as log_handle:
        fieldnames = (
            "idx",
            "seed",
            "species",
            "stage",
            "height_level",
            "crown_width_level",
            "branch_density_level",
            "levels",
            "node_count",
            "edge_count",
            "component_count",
            "status",
            "message",
        )
        writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        print(
            f"[GEN] species={species} levels={BRANCH_LEVELS} "
            f"rows={args.start}:{stop} -> {output_dir}"
        )
        for position, item in enumerate(selected, start=args.start):
            index = item["idx"]
            source = item["source"]
            node_path = output_dir / f"SkeletonNode_{index:05d}.txt"
            connection_path = output_dir / f"SkeletonConnection_{index:05d}.txt"
            if (
                not args.overwrite
                and node_path.is_file()
                and connection_path.is_file()
            ):
                print(f"[{position + 1}/{stop}] idx={index} cached")
                continue

            kwargs = dict(PRESET)
            kwargs.update(item["overrides"])
            kwargs["seed"] = item["seed"]
            kwargs["levels"] = BRANCH_LEVELS
            kwargs["makeMesh"] = False
            if FORCE_NO_LEAVES:
                kwargs["showLeaves"] = False
                kwargs["leaves"] = 0
            if "do_update" in valid_keys:
                kwargs["do_update"] = True
            kwargs = coerce_to_operator_types(kwargs, properties, valid_keys)

            new_objects = []
            row = {
                "idx": index,
                "seed": item["seed"],
                "species": species,
                "stage": source["stage"],
                "height_level": source["height_level"],
                "crown_width_level": source["crown_width_level"],
                "branch_density_level": source["branch_density_level"],
                "levels": BRANCH_LEVELS,
                "node_count": "",
                "edge_count": "",
                "component_count": "",
                "status": "",
                "message": "",
            }
            try:
                tree, new_objects = add_tree(kwargs, override)
                node_count, edge_count, components = export_evaluated_skeleton(
                    tree, node_path, connection_path
                )
                row.update(
                    {
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "component_count": components,
                        "status": "completed",
                    }
                )
                print(
                    f"[{position + 1}/{stop}] idx={index} seed={item['seed']} "
                    f"nodes={node_count} edges={edge_count} "
                    f"components={components}"
                )
            except Exception as error:
                failures += 1
                row["status"] = "failed"
                row["message"] = f"{type(error).__name__}: {error}"
                node_path.unlink(missing_ok=True)
                connection_path.unlink(missing_ok=True)
                print(f"[ERROR] idx={index}: {row['message']}")
            finally:
                remove_objects(new_objects)
            writer.writerow(row)
            log_handle.flush()

    print(
        f"[DONE] species={species} attempted={len(selected)} "
        f"failures={failures} output={output_dir}"
    )
    if failures:
        raise RuntimeError(
            f"{failures} rows failed; inspect {log_path} and rerun to fill them"
        )


if __name__ == "__main__":
    main()
