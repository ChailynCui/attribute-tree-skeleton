from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable
import numpy as np
from scipy.spatial import cKDTree

NULL_SEED = 20260824
NULL_PERMUTATIONS = 1000
STRUCTURE_METRICS = (
    "edge_count",
    "total_branch_length",
    "bifurcation_count",
    "leaf_count",
    "graph_depth_max",
)


def _pair_distance(
    left: np.ndarray, right: np.ndarray, left_tree: cKDTree, right_tree: cKDTree
) -> float:
    left_to_right = right_tree.query(left, k=1, workers=1)[0]
    right_to_left = left_tree.query(right, k=1, workers=1)[0]
    return float(np.mean(np.square(left_to_right)) + np.mean(np.square(right_to_left)))


def cross_chamfer_matrix(
    left_points: list[np.ndarray],
    right_points: list[np.ndarray],
    *,
    workers: int,
) -> np.ndarray:
    if not left_points or not right_points:
        raise ValueError("Chamfer matrix requires two non-empty sets")
    left_trees = [cKDTree(points) for points in left_points]
    right_trees = [cKDTree(points) for points in right_points]

    def one_row(index: int) -> tuple[int, np.ndarray]:
        values = np.empty(len(right_points), dtype=np.float64)
        for right_index, right_tree in enumerate(right_trees):
            values[right_index] = _pair_distance(
                left_points[index],
                right_points[right_index],
                left_trees[index],
                right_tree,
            )
        return index, values

    matrix = np.empty((len(left_points), len(right_points)), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for index, values in pool.map(one_row, range(len(left_points))):
            matrix[index] = values
    return matrix


def symmetric_chamfer_matrix(points: list[np.ndarray], *, workers: int) -> np.ndarray:
    matrix = np.zeros((len(points), len(points)), dtype=np.float64)
    if len(points) < 2:
        return matrix
    trees = [cKDTree(item) for item in points]

    def one_row(index: int) -> tuple[int, np.ndarray]:
        values = np.empty(len(points), dtype=np.float64)
        values[index] = 0.0
        for other in range(index + 1, len(points)):
            values[other] = _pair_distance(points[index], points[other], trees[index], trees[other])
        return index, values

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for index, values in pool.map(one_row, range(len(points))):
            matrix[index, index + 1 :] = values[index + 1 :]
    upper = np.triu(matrix, 1)
    matrix = upper + upper.T
    return matrix


def wasserstein1(left: Iterable[float], right: Iterable[float]) -> float:
    left = np.asarray(list(left), dtype=np.float64)
    right = np.asarray(list(right), dtype=np.float64)
    grid = np.linspace(0.0, 1.0, 101)
    return float(np.mean(np.abs(np.quantile(left, grid) - np.quantile(right, grid))))


def one_nna_with_null(
    reference_points: list[np.ndarray],
    generated_points: list[np.ndarray],
    *,
    workers: int,
) -> dict[str, Any]:
    if len(reference_points) != len(generated_points):
        return {
            "value": None,
            "status": "not_computable_unbalanced",
            "reference_count": len(reference_points),
            "generated_count": len(generated_points),
        }
    n = len(reference_points)
    pooled = reference_points + generated_points
    distances = symmetric_chamfer_matrix(pooled, workers=workers)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    labels = np.concatenate((np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)))
    observed = float(np.mean(labels[nearest] == labels))
    rng = np.random.default_rng(NULL_SEED)
    null_values = np.empty(NULL_PERMUTATIONS, dtype=np.float64)
    for index in range(NULL_PERMUTATIONS):
        permuted = rng.permutation(labels)
        null_values[index] = float(np.mean(permuted[nearest] == permuted))
    return {
        "value": observed,
        "status": "completed",
        "reference_count": n,
        "generated_count": n,
        "nearest_neighbor_tie": "np.argmin selects the first pooled index",
        "label_permutation_null": {
            "seed": NULL_SEED,
            "permutations": NULL_PERMUTATIONS,
            "mean_accuracy": float(null_values.mean()),
            "sd_accuracy": float(null_values.std(ddof=1)),
            "expected_accuracy": 0.5,
        },
    }


POINT_COUNT = 1024
WELD_TOLERANCE = 1e-6


def evaluate_species(
    species: str,
    reference: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    *,
    workers: int,
) -> dict[str, Any]:
    reference_points = [row["points"] for row in reference]
    generated_points = [row["points"] for row in generated]
    result: dict[str, Any] = {
        "species": species,
        "planned_reference_count": len(reference),
        "planned_generated_count": len(reference),
        "reference_count": len(reference),
        "generated_valid_count": len(generated),
        "generation_failure_count": len(reference) - len(generated),
        "generation_failure_rate": float((len(reference) - len(generated)) / len(reference)),
        "point_count": POINT_COUNT,
        "weld_tolerance": WELD_TOLERANCE,
        "status": "completed" if generated else "insufficient_data",
    }
    if not generated:
        result.update(
            {
                "mmd_cd": None,
                "cov_cd": None,
                "one_nna_cd": one_nna_with_null(
                    reference_points, generated_points, workers=workers
                ),
                "structure_w1": {metric: None for metric in STRUCTURE_METRICS},
            }
        )
        return result

    cross = cross_chamfer_matrix(reference_points, generated_points, workers=workers)
    nearest_generated_for_reference = np.argmin(cross, axis=1)
    nearest_reference_for_generated = np.argmin(cross, axis=0)
    result["mmd_cd"] = {
        "value": float(np.mean(np.min(cross, axis=1))),
        "direction": "mean over reference trees of nearest generated CD",
        "distance": "squared-l2 symmetric Chamfer",
    }
    result["cov_cd"] = {
        "value": float(len(np.unique(nearest_reference_for_generated)) / len(reference)),
        "direction": "fraction of reference trees selected by generated nearest matches",
    }
    result["one_nna_cd"] = one_nna_with_null(
        reference_points,
        generated_points,
        workers=workers,
    )
    structure_w1: dict[str, float | None] = {}
    for metric in STRUCTURE_METRICS:
        reference_values = [float(row["structure"][metric]) for row in reference]
        generated_values = [float(row["structure"][metric]) for row in generated]
        structure_w1[metric] = wasserstein1(reference_values, generated_values)
    result["structure_w1"] = structure_w1
    result["matching_audit"] = {
        "reference_nearest_generated_unique": int(len(np.unique(nearest_generated_for_reference))),
        "generated_nearest_reference_unique": int(len(np.unique(nearest_reference_for_generated))),
    }
    return result
