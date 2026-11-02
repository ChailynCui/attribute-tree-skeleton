from __future__ import annotations
import itertools
import numpy as np
from scipy.spatial import cKDTree

ATTRIBUTE_NAMES = ("height", "crown_width", "branch_density")
LEVEL_NAMES = ("low", "medium", "high")


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)

    def find(self, value: int) -> int:
        while int(self.parent[value]) != value:
            self.parent[value] = self.parent[int(self.parent[value])]
            value = int(self.parent[value])
        return value

    def union(self, first: int, second: int) -> bool:
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return False
        if self.size[first] < self.size[second]:
            first, second = second, first
        self.parent[second] = first
        self.size[first] += self.size[second]
        return True


def inverse_normalization(sample: dict) -> tuple[np.ndarray, dict]:
    vertices = np.asarray(sample["vertices"], dtype=np.float64)
    preprocess = sample.get("preprocess", {})
    if not isinstance(preprocess, dict):
        preprocess = {}
    scale = float(preprocess.get("normalization_scale", 1.0))
    center = np.asarray(
        preprocess.get("normalization_center", np.zeros(3)), dtype=np.float64
    ).reshape(3)
    if not np.isfinite(scale) or scale <= 0 or not np.all(np.isfinite(center)):
        raise ValueError("invalid stored normalization transform")
    summary = {
        "global_normalized": bool(preprocess.get("global_normalized")),
        "normalize_target": preprocess.get("normalize_target"),
        "normalization_center": center.tolist(),
        "normalization_scale": scale,
        "normalization_basis": preprocess.get("normalization_basis"),
    }
    return vertices / scale + center, summary


def clean_edges(lines: np.ndarray) -> tuple[np.ndarray, dict]:
    seen = set()
    output = []
    loops = 0
    duplicates = 0
    for first, second in np.asarray(lines, dtype=np.int64):
        first, second = int(first), int(second)
        if first == second:
            loops += 1
            continue
        key = (first, second) if first < second else (second, first)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        output.append((first, second))
    return np.asarray(output, dtype=np.int32).reshape(-1, 2), {
        "self_loops_removed": loops,
        "duplicate_edges_removed": duplicates,
    }


def repair_strict_tree(vertices: np.ndarray, raw_lines: np.ndarray) -> tuple[np.ndarray, dict]:
    lines, cleanup = clean_edges(raw_lines)
    if len(lines) and (lines.min() < 0 or lines.max() >= len(vertices)):
        raise ValueError("edge index outside vertex range")
    lengths = (
        np.linalg.norm(vertices[lines[:, 1]] - vertices[lines[:, 0]], axis=1)
        if len(lines)
        else np.empty(0, dtype=np.float64)
    )
    order = np.lexsort((np.arange(len(lines)), lengths))
    union_find = UnionFind(len(vertices))
    keep = np.zeros(len(lines), dtype=bool)
    for edge_index in order:
        first, second = lines[int(edge_index)]
        if union_find.union(int(first), int(second)):
            keep[int(edge_index)] = True
    forest = lines[keep]
    cycle_edges_removed = int(np.sum(~keep))

    roots = np.asarray([union_find.find(index) for index in range(len(vertices))])
    _, labels = np.unique(roots, return_inverse=True)
    component_count = int(labels.max()) + 1
    component_sizes = np.bincount(labels, minlength=component_count)
    main_component = int(np.argmax(component_sizes))
    main_indices = np.flatnonzero(labels == main_component)
    main_tree = cKDTree(vertices[main_indices])
    bridges = []
    bridge_lengths = []
    for component in range(component_count):
        if component == main_component:
            continue
        component_indices = np.flatnonzero(labels == component)
        distances, nearest = main_tree.query(vertices[component_indices], k=1)
        local = int(np.argmin(distances))
        first = int(component_indices[local])
        second = int(main_indices[int(nearest[local])])
        if not union_find.union(first, second):
            raise RuntimeError("bridge unexpectedly formed a cycle")
        bridges.append((first, second))
        bridge_lengths.append(float(distances[local]))
    repaired = (
        np.concatenate((forest, np.asarray(bridges, dtype=np.int32)), axis=0) if bridges else forest
    ).astype(np.int32, copy=False)
    if len(repaired) != len(vertices) - 1:
        raise RuntimeError(f"strict-tree repair failed: V={len(vertices)} E={len(repaired)}")
    positive = lengths[lengths > 0]
    median_edge = float(np.median(positive)) if len(positive) else 0.0
    return repaired, {
        "method": "short_edge_kruskal_then_nearest_to_largest_component_v1",
        "vertices_moved": False,
        "input_edge_count": int(len(raw_lines)),
        "output_edge_count": int(len(repaired)),
        "component_count_before": component_count,
        "cycle_edges_removed": cycle_edges_removed,
        "bridge_edges_added": int(len(bridges)),
        "bridge_length_max": float(max(bridge_lengths, default=0.0)),
        "bridge_max_over_original_median_edge": (
            float(max(bridge_lengths, default=0.0) / median_edge) if median_edge > 0 else 0.0
        ),
        **cleanup,
        "strict_tree_after": True,
    }


def map_through_train_cdf(values: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, dict]:
    values = np.asarray(values, dtype=np.float64)
    sorted_train = np.sort(values[train_indices])
    denominator = max(len(sorted_train) - 1, 1)
    left = np.searchsorted(sorted_train, values, side="left")
    right = np.searchsorted(sorted_train, values, side="right")
    ranks = np.where(right > left, 0.5 * (left + right - 1), left - 0.5)
    percentiles = np.clip(ranks / denominator, 0.0, 1.0)
    return percentiles, {
        "fit_count": int(len(train_indices)),
        "tie_method": "average rank",
        "unseen_value_method": "midpoint between adjacent train ranks",
        "fit_min": float(sorted_train[0]),
        "fit_max": float(sorted_train[-1]),
    }


def target_cell(sample: dict) -> tuple[str, str, str]:
    meta = sample.get("meta", {})
    direct = tuple(meta.get(f"{attribute}_level") for attribute in ATTRIBUTE_NAMES)
    if all(value in LEVEL_NAMES for value in direct):
        return direct
    nested = meta.get("target_levels", {})
    if isinstance(nested, dict):
        values = tuple(nested.get(attribute) for attribute in ATTRIBUTE_NAMES)
        if all(value in LEVEL_NAMES for value in values):
            return values
    raise ValueError(f"sample {sample.get('name')} has no valid frozen three-attribute target cell")


def allocate_stratified_counts(
    sizes: np.ndarray,
    total: int,
    fraction: float,
    capacities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    ideals = sizes.astype(np.float64) * float(fraction)
    counts = np.minimum(np.floor(ideals).astype(np.int64), capacities)
    remaining = int(total - counts.sum())
    tie_break = rng.random(len(sizes))
    order = np.lexsort((tie_break, -(ideals - np.floor(ideals))))
    while remaining > 0:
        changed = False
        for index in order:
            index = int(index)
            if counts[index] >= capacities[index]:
                continue
            counts[index] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            raise RuntimeError("stratified allocation has insufficient capacity")
    if remaining < 0:
        raise RuntimeError("stratified allocation base counts exceed requested total")
    return counts


def stratified_species_split(
    dataset: list[dict],
    indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index in indices:
        groups.setdefault(target_cell(dataset[int(index)]), []).append(int(index))
    expected_cells = set(itertools.product(LEVEL_NAMES, repeat=3))
    if set(groups) != expected_cells:
        missing = sorted(expected_cells - set(groups))
        raise ValueError(f"target-cell stratification is missing cells: {missing}")
    cells = sorted(groups)
    shuffled = []
    for cell in cells:
        shuffled.append(rng.permutation(np.asarray(groups[cell], dtype=np.int64)))
    sizes = np.asarray([len(values) for values in shuffled], dtype=np.int64)
    train_counts = allocate_stratified_counts(sizes, 2400, 0.8, sizes, rng)
    validation_counts = allocate_stratified_counts(
        sizes,
        300,
        0.1,
        sizes - train_counts,
        rng,
    )
    result = {"train": [], "validation": [], "test": []}
    cell_report = {}
    for cell_index, (cell, values) in enumerate(zip(cells, shuffled)):
        train_end = int(train_counts[cell_index])
        validation_end = train_end + int(validation_counts[cell_index])
        partitions = {
            "train": values[:train_end],
            "validation": values[train_end:validation_end],
            "test": values[validation_end:],
        }
        for split_name, split_values in partitions.items():
            result[split_name].extend(int(value) for value in split_values)
        cell_report["/".join(cell)] = {
            "total": int(len(values)),
            **{
                split_name: int(len(split_values))
                for split_name, split_values in partitions.items()
            },
        }
    return {
        name: np.asarray(values, dtype=np.int64) for name, values in result.items()
    }, cell_report
