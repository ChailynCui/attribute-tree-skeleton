from __future__ import annotations
import hashlib
import json
import os
import pickle
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Iterable
import numpy as np

SPECIES_ORDER = (
    "callistemon",
    "japanese_maple",
    "weeping_willow",
    "white_birch",
)
METHOD = "strict_tree_root_oriented_branch_paths"
VERSION = "four_species_direct_morphology_prior_v1"
EPSILON = 1e-12
NUMERIC_KEYS = (
    "count",
    "min",
    "q01",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "q99",
    "max",
    "mean",
    "std",
)
ALGORITHM_DEFINITION = {
    "strict_tree_validation": (
        "Require finite Vx3 vertices, valid unique non-self E x 2 edges, "
        "E=V-1, and one connected acyclic component for every fitted tree."
    ),
    "root_selection": (
        "Choose a vertex np.isclose to the minimum z; on a tie choose the "
        "candidate nearest the median XY position, with lowest vertex index "
        "as the final tie-break."
    ),
    "orientation": (
        "Orient every edge away from the root with deterministic BFS over sorted vertex neighbors."
    ),
    "branch_paths": (
        "Collapse each maximal root-oriented degree-2 chain into one branch "
        "path. The root is always a path boundary; every non-root vertex with "
        "undirected degree != 2 is a path boundary."
    ),
    "branch_angle": (
        "For each child branch path with a parent path, measure arccos of the "
        "normalized dot product between the parent's terminal directed edge "
        "and the child's initial directed edge. Root-originating paths have no "
        "parent angle. Zero-length tangent pairs are excluded and counted."
    ),
    "junction_degree": (
        "For every oriented branch path, record 1 plus the number of child "
        "branch paths leaving its downstream endpoint. Thus a terminal path "
        "records 1 and a binary fork records 3, matching Direct's branch-path "
        "prior convention."
    ),
    "tree_edge_median": (
        "For each tree, take the median of its finite edge lengths greater "
        "than 1e-12, then summarize those per-tree medians."
    ),
    "pooled_edge_median": (
        "Take the median after pooling all finite edge lengths greater than "
        "1e-12 across that species' fitted training trees."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_indices(indices: Iterable[int]) -> str:
    encoded = "".join(f"{int(index)}\n" for index in indices).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    """Write JSON through a same-directory temporary file and atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def enable_numpy_pickle_compatibility() -> None:
    """Allow NumPy-2-created pickles to load under a local NumPy 1.x."""

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_dataset(path: Path) -> list[dict]:
    enable_numpy_pickle_compatibility()
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, list):
        raise TypeError(f"{path} must contain a list, found {type(dataset)!r}")
    return dataset


def numeric_prior(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("cannot fit an empty morphology prior")
    if not np.all(np.isfinite(array)):
        raise ValueError("morphology prior input contains non-finite values")
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "q01": float(np.quantile(array, 0.01)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def choose_lowest_root(vertices: np.ndarray) -> int:
    minimum_z = float(vertices[:, 2].min())
    candidates = np.flatnonzero(np.isclose(vertices[:, 2], minimum_z))
    if len(candidates) == 1:
        return int(candidates[0])
    center_xy = np.median(vertices[:, :2], axis=0)
    distances = np.linalg.norm(vertices[candidates, :2] - center_xy[None, :], axis=1)
    minimum_distance = float(distances.min())
    tied = candidates[np.isclose(distances, minimum_distance)]
    return int(tied.min())


def validate_and_orient_tree(
    sample: dict, *, sample_index: int
) -> tuple[np.ndarray, np.ndarray, int, list[list[int]], list[int]]:
    """Validate a strict tree and return a deterministic root orientation."""

    vertices = np.asarray(sample.get("vertices"), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2:
        raise ValueError(f"sample {sample_index}: vertices must have shape (V, 3), V >= 2")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"sample {sample_index}: non-finite vertices")
    line_key = "lines" if "lines" in sample else "edges"
    if line_key not in sample:
        raise KeyError(f"sample {sample_index}: missing lines/edges")
    lines = np.asarray(sample[line_key], dtype=np.int64)
    if lines.ndim != 2 or lines.shape[1] != 2:
        raise ValueError(f"sample {sample_index}: lines must have shape (E, 2)")
    if len(lines) != len(vertices) - 1:
        raise ValueError(f"sample {sample_index}: not strict, V={len(vertices)} E={len(lines)}")
    if len(lines) and (int(lines.min()) < 0 or int(lines.max()) >= len(vertices)):
        raise ValueError(f"sample {sample_index}: edge index out of range")

    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    edge_keys: set[tuple[int, int]] = set()
    for first_value, second_value in lines:
        first, second = int(first_value), int(second_value)
        if first == second:
            raise ValueError(f"sample {sample_index}: self-loop at {first}")
        key = (min(first, second), max(first, second))
        if key in edge_keys:
            raise ValueError(f"sample {sample_index}: duplicate edge {key}")
        edge_keys.add(key)
        adjacency[first].append(second)
        adjacency[second].append(first)
    for neighbors in adjacency:
        neighbors.sort()

    root = choose_lowest_root(vertices)
    parent = [-1] * len(vertices)
    parent[root] = root
    queue = deque([root])
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for neighbor in adjacency[current]:
            if parent[neighbor] >= 0:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if visited != len(vertices):
        raise ValueError(f"sample {sample_index}: disconnected, visited {visited}/{len(vertices)}")
    return vertices, lines, root, adjacency, parent


def root_oriented_branch_paths(
    adjacency: list[list[int]], parent: list[int], root: int
) -> list[dict]:
    """Collapse a rooted strict tree into directed degree-2 branch paths."""

    children = [[] for _ in adjacency]
    for vertex, parent_vertex in enumerate(parent):
        if vertex != root:
            children[parent_vertex].append(vertex)
    for values in children:
        values.sort()

    paths: list[dict] = []
    pending = deque((root, child, None) for child in children[root])
    while pending:
        start, first, parent_path = pending.popleft()
        vertices = [int(start), int(first)]
        current = int(first)
        while len(adjacency[current]) == 2 and len(children[current]) == 1:
            current = int(children[current][0])
            vertices.append(current)
        path_id = len(paths)
        paths.append(
            {
                "path_id": path_id,
                "parent_path": parent_path,
                "vertices": vertices,
            }
        )
        for child in children[current]:
            pending.append((current, child, path_id))
    return paths


def angle_between(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= EPSILON or second_norm <= EPSILON:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def tree_morphology(sample: dict, *, sample_index: int) -> dict:
    vertices, lines, root, adjacency, parent = validate_and_orient_tree(
        sample, sample_index=sample_index
    )
    paths = root_oriented_branch_paths(adjacency, parent, root)
    if not paths:
        raise ValueError(f"sample {sample_index}: tree contains no branch path")

    branch_angles: list[float] = []
    zero_tangent_pairs = 0
    child_path_count = np.zeros(len(paths), dtype=np.int64)
    for path in paths:
        parent_path = path["parent_path"]
        if parent_path is None:
            continue
        child_path_count[int(parent_path)] += 1
        parent_vertices = paths[int(parent_path)]["vertices"]
        child_vertices = path["vertices"]
        if parent_vertices[-1] != child_vertices[0]:
            raise RuntimeError(f"sample {sample_index}: branch-path hierarchy is inconsistent")
        angle = angle_between(
            vertices[parent_vertices[-1]] - vertices[parent_vertices[-2]],
            vertices[child_vertices[1]] - vertices[child_vertices[0]],
        )
        if angle is None:
            zero_tangent_pairs += 1
        else:
            branch_angles.append(angle)

    junction_degrees = (1 + child_path_count).astype(np.float64)
    edge_lengths = np.linalg.norm(vertices[lines[:, 1]] - vertices[lines[:, 0]], axis=1)
    positive_edge_lengths = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > EPSILON)]
    if not len(positive_edge_lengths):
        raise ValueError(f"sample {sample_index}: no positive finite edge lengths")
    return {
        "branch_angles": branch_angles,
        "junction_degrees": junction_degrees,
        "edge_lengths": positive_edge_lengths,
        "tree_edge_median": float(np.median(positive_edge_lengths)),
        "vertex_count": int(len(vertices)),
        "edge_count": int(len(lines)),
        "branch_path_count": int(len(paths)),
        "parent_child_path_pairs": int(np.sum(child_path_count)),
        "zero_tangent_pairs_skipped": int(zero_tangent_pairs),
        "nonpositive_edges_excluded": int(len(edge_lengths) - len(positive_edge_lengths)),
    }


def sample_species(sample: dict, *, sample_index: int) -> str:
    meta = sample.get("meta")
    species = meta.get("species") if isinstance(meta, dict) else None
    if species not in SPECIES_ORDER:
        raise ValueError(f"training sample {sample_index}: invalid species {species!r}")
    return str(species)


def fit_species_prior(
    dataset: list[dict],
    train_indices: list[int],
    species: str,
    *,
    dataset_path: Path,
    split_manifest_path: Path,
    source_hashes: dict[str, str],
) -> dict:
    species_indices = [
        index
        for index in train_indices
        if sample_species(dataset[index], sample_index=index) == species
    ]
    if not species_indices:
        raise ValueError(f"no training samples found for species {species}")

    branch_angles: list[float] = []
    junction_degrees: list[np.ndarray] = []
    tree_edge_medians: list[float] = []
    pooled_edge_lengths: list[np.ndarray] = []
    diagnostics = {
        "vertex_count": 0,
        "edge_count": 0,
        "branch_path_count": 0,
        "parent_child_path_pairs": 0,
        "zero_tangent_pairs_skipped": 0,
        "nonpositive_edges_excluded": 0,
    }
    for index in species_indices:
        result = tree_morphology(dataset[index], sample_index=index)
        branch_angles.extend(result["branch_angles"])
        junction_degrees.append(result["junction_degrees"])
        tree_edge_medians.append(result["tree_edge_median"])
        pooled_edge_lengths.append(result["edge_lengths"])
        for key in diagnostics:
            diagnostics[key] += int(result[key])

    if not branch_angles:
        raise ValueError(f"species {species}: no valid parent-child branch angles")
    pooled = np.concatenate(pooled_edge_lengths)
    degrees = np.concatenate(junction_degrees)
    branch_angle = numeric_prior(branch_angles)
    branch_angle["unit"] = "radian"
    branch_angle["degrees"] = {
        key: float(np.degrees(value))
        for key, value in branch_angle.items()
        if key not in {"count", "unit"}
    }
    fit_indices_hash = sha256_indices(species_indices)
    return {
        "schema_version": 1,
        "method": METHOD,
        "version": VERSION,
        "species": species,
        "fit_split": "train",
        "tree_count": int(len(species_indices)),
        "source_dataset": str(dataset_path.resolve()),
        "source_split_manifest": str(split_manifest_path.resolve()),
        "dataset_sha256": source_hashes["dataset_sha256"],
        "split_manifest_sha256": source_hashes["split_manifest_sha256"],
        "fit_indices_sha256": fit_indices_hash,
        "source_hashes": {
            **source_hashes,
            "fit_indices_sha256": fit_indices_hash,
        },
        "algorithm_definition": dict(ALGORITHM_DEFINITION),
        "branch_angle": branch_angle,
        "junction_degree": numeric_prior(degrees),
        "tree_edge_median": numeric_prior(tree_edge_medians),
        "pooled_edge_median": float(np.median(pooled)),
        "fit_diagnostics": diagnostics,
    }


def validate_split_manifest(payload: dict, dataset_size: int | None = None) -> list[int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("splits"), dict):
        raise ValueError("split manifest must contain a splits object")
    splits = payload["splits"]
    for name in ("train", "validation", "test"):
        if not isinstance(splits.get(name), list):
            raise ValueError(f"split manifest is missing splits.{name}")
    parsed = {
        name: [int(value) for value in splits[name]] for name in ("train", "validation", "test")
    }
    for name, indices in parsed.items():
        if len(indices) != len(set(indices)):
            raise ValueError(f"split {name} contains duplicate indices")
        if any(index < 0 for index in indices):
            raise ValueError(f"split {name} contains negative indices")
        if dataset_size is not None and any(index >= dataset_size for index in indices):
            raise ValueError(f"split {name} contains an out-of-range index")
    train = set(parsed["train"])
    validation = set(parsed["validation"])
    test = set(parsed["test"])
    if train & validation or train & test or validation & test:
        raise ValueError("train/validation/test splits overlap")
    expected_size = payload.get("sample_count")
    if dataset_size is not None and expected_size is not None:
        if int(expected_size) != dataset_size:
            raise ValueError(f"split sample_count={expected_size}, dataset size={dataset_size}")
    if dataset_size is not None and len(train | validation | test) != dataset_size:
        raise ValueError("train/validation/test do not partition the dataset")
    return parsed["train"]


def prior_path(output_dir: Path, species: str) -> Path:
    return output_dir / f"{species}_training_morphology_prior.json"


def valid_numeric_prior(value: object, *, require_count: bool = True) -> bool:
    if not isinstance(value, dict):
        return False
    keys = NUMERIC_KEYS if require_count else NUMERIC_KEYS[1:]
    try:
        return all(np.isfinite(float(value[key])) for key in keys)
    except (KeyError, TypeError, ValueError):
        return False


def load_valid_prior(path: Path, species: str, source_hashes: dict[str, str]) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("method") != METHOD
        or payload.get("version") != VERSION
        or payload.get("species") != species
        or payload.get("fit_split") != "train"
        or int(payload.get("tree_count", 0)) <= 0
        or payload.get("dataset_sha256") != source_hashes["dataset_sha256"]
        or payload.get("split_manifest_sha256") != source_hashes["split_manifest_sha256"]
        or not valid_numeric_prior(payload.get("branch_angle"))
        or not valid_numeric_prior(payload.get("junction_degree"))
        or not valid_numeric_prior(payload.get("tree_edge_median"))
    ):
        return None
    try:
        pooled = float(payload["pooled_edge_median"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(pooled) or pooled <= 0:
        return None
    degrees = payload["branch_angle"].get("degrees")
    if not valid_numeric_prior(degrees, require_count=False):
        return None
    return payload


def make_output_manifest(
    *,
    dataset_path: Path,
    split_manifest_path: Path,
    split_payload: dict,
    source_hashes: dict[str, str],
    priors: dict[str, dict],
    output_dir: Path,
) -> dict:
    train_indices = validate_split_manifest(split_payload)
    return {
        "schema_version": 1,
        "status": "complete",
        "method": METHOD,
        "version": VERSION,
        "fit_split": "train",
        "leakage_guard": {
            "fit_indices_source": "split_manifest.splits.train only",
            "validation_samples_used": 0,
            "test_samples_used": 0,
            "train_indices_sha256": sha256_indices(train_indices),
        },
        "source_dataset": str(dataset_path.resolve()),
        "source_split_manifest": str(split_manifest_path.resolve()),
        "source_hashes": dict(source_hashes),
        "species_order": list(SPECIES_ORDER),
        "tree_count": {species: int(priors[species]["tree_count"]) for species in SPECIES_ORDER},
        "outputs": {
            species: {
                "path": str(prior_path(output_dir, species).resolve()),
                "sha256": sha256_file(prior_path(output_dir, species)),
            }
            for species in SPECIES_ORDER
        },
    }


def valid_output_manifest(path: Path, expected: dict, source_hashes: dict[str, str]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        payload.get("status") != "complete"
        or payload.get("method") != METHOD
        or payload.get("version") != VERSION
        or payload.get("fit_split") != "train"
        or payload.get("source_hashes") != source_hashes
    ):
        return False
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        return False
    return all(
        isinstance(outputs.get(species), dict)
        and outputs[species].get("sha256") == expected[species]
        for species in SPECIES_ORDER
    )


def build_priors(
    dataset_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict:
    dataset_path = Path(dataset_path)
    split_manifest_path = Path(split_manifest_path)
    output_dir = Path(output_dir)
    split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    validate_split_manifest(split_payload)
    source_hashes = {
        "dataset_sha256": sha256_file(dataset_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
    }
    declared_dataset_hash = (
        split_payload.get("dataset", {}).get("file_sha256")
        if isinstance(split_payload.get("dataset"), dict)
        else None
    )
    if declared_dataset_hash and declared_dataset_hash != source_hashes["dataset_sha256"]:
        raise ValueError("split manifest dataset hash does not match --dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    priors: dict[str, dict] = {}
    missing_species: list[str] = []
    if force:
        missing_species = list(SPECIES_ORDER)
    else:
        for species in SPECIES_ORDER:
            path = prior_path(output_dir, species)
            if not path.exists():
                missing_species.append(species)
                continue
            prior = load_valid_prior(path, species, source_hashes)
            if prior is None:
                raise FileExistsError(
                    f"invalid or stale output exists: {path}; pass --force to replace"
                )
            priors[species] = prior

        if not missing_species and manifest_path.exists():
            hashes = {
                species: sha256_file(prior_path(output_dir, species)) for species in SPECIES_ORDER
            }
            if valid_output_manifest(manifest_path, hashes, source_hashes):
                return {
                    "status": "skipped_valid",
                    "manifest": str(manifest_path),
                    "outputs": {
                        species: str(prior_path(output_dir, species)) for species in SPECIES_ORDER
                    },
                }
            raise FileExistsError(
                f"invalid or stale output exists: {manifest_path}; pass --force to replace"
            )
        if manifest_path.exists():
            raise FileExistsError(
                f"incomplete output set conflicts with {manifest_path}; pass --force to replace"
            )

    if missing_species:
        dataset = load_dataset(dataset_path)
        train_indices = validate_split_manifest(split_payload, len(dataset))
        for species in missing_species:
            fitted = fit_species_prior(
                dataset,
                train_indices,
                species,
                dataset_path=dataset_path,
                split_manifest_path=split_manifest_path,
                source_hashes=source_hashes,
            )
            species_counts = split_payload.get("species_counts")
            if isinstance(species_counts, dict):
                declared = species_counts.get(species, {})
                expected_train = declared.get("train") if isinstance(declared, dict) else None
                if expected_train is not None and fitted["tree_count"] != int(expected_train):
                    raise ValueError(
                        f"{species}: fitted {fitted['tree_count']} train trees, "
                        f"manifest declares {expected_train}"
                    )
            atomic_json(prior_path(output_dir, species), fitted)
            priors[species] = fitted

    manifest = make_output_manifest(
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        split_payload=split_payload,
        source_hashes=source_hashes,
        priors=priors,
        output_dir=output_dir,
    )
    atomic_json(manifest_path, manifest)
    return {
        "status": "built",
        "manifest": str(manifest_path),
        "outputs": {species: str(prior_path(output_dir, species)) for species in SPECIES_ORDER},
    }
