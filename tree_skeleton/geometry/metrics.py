# SPDX-License-Identifier: MIT
# Adapted research implementation; see README.md for license notices.
"""Deterministic geometry metrics for the formal tree-skeleton evaluations.

The module deliberately keeps the geometry contract in one place.  It does not
select checkpoints, filter samples, or repair generated outputs.  Endpoint
welding is an analysis operation used to measure the spatial graph implied by
an edge stream; the input arrays are never modified in place.

The attribute implementation mirrors the frozen V2.2 definition stored in the
four-species attribute JSON: H98 height, robust PCA crown width, and crown
skeleton length divided by the robust crown-box volume.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .graph import weld_skeleton_vertices


ATTRIBUTE_NAMES = ("height", "crown_width", "branch_density")
EPS = 1e-12
DEFAULT_WELD_TOLERANCE = 1e-6
DEFAULT_POINT_COUNT = 1024
DEFAULT_DISTANCE_CHUNK = 256


def _skeleton_arrays(
    vertices: np.ndarray,
    lines: np.ndarray,
    *,
    context: str = "skeleton",
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    lines = np.asarray(lines, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{context}: vertices must have shape (V, 3)")
    if lines.size == 0:
        lines = np.empty((0, 2), dtype=np.int64)
    elif lines.ndim != 2 or lines.shape[1] != 2:
        raise ValueError(f"{context}: lines must have shape (E, 2)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"{context}: vertices contain non-finite values")
    if len(lines) and (int(lines.min()) < 0 or int(lines.max()) >= len(vertices)):
        raise ValueError(f"{context}: line index is outside the vertex array")
    return vertices, lines


def _adjacency(vertex_count: int, lines: np.ndarray) -> list[list[int]]:
    adjacency = [[] for _ in range(vertex_count)]
    for first, second in lines:
        first, second = int(first), int(second)
        adjacency[first].append(second)
        adjacency[second].append(first)
    return adjacency


def _component_labels(adjacency: list[list[int]]) -> tuple[np.ndarray, int]:
    labels = np.full(len(adjacency), -1, dtype=np.int64)
    component_count = 0
    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue
        labels[start] = component_count
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component_count
                    queue.append(neighbor)
        component_count += 1
    return labels, component_count


def _lowest_root(vertices: np.ndarray) -> int:
    if len(vertices) == 0:
        return -1
    minimum_z = float(vertices[:, 2].min())
    candidates = np.flatnonzero(np.isclose(vertices[:, 2], minimum_z))
    if len(candidates) == 1:
        return int(candidates[0])
    center = np.median(vertices[:, :2], axis=0)
    distances = np.linalg.norm(vertices[candidates, :2] - center, axis=1)
    return int(candidates[np.lexsort((candidates, distances))[0]])


def _bfs_depth(adjacency: list[list[int]], start: int) -> int:
    if start < 0 or start >= len(adjacency):
        return 0
    depths = {int(start): 0}
    queue = deque([int(start)])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in depths:
                depths[neighbor] = depths[current] + 1
                queue.append(neighbor)
    return int(max(depths.values())) if depths else 0


def graph_metrics(
    vertices: np.ndarray,
    lines: np.ndarray,
    *,
    weld_tolerance: float = DEFAULT_WELD_TOLERANCE,
) -> dict:
    """Return spatial graph and structural metrics without repairing a graph.

    ``edge_count`` and topology fields refer to the welded spatial graph;
    ``raw_edge_count`` preserves the input edge-stream count.  A strict tree is
    true only when the welded graph has one component, zero cycle rank, and
    ``E = V - 1``.
    """

    vertices, lines = _skeleton_arrays(vertices, lines)
    if weld_tolerance < 0:
        raise ValueError("weld_tolerance must be non-negative")
    welded_vertices, welded_lines, weld = weld_skeleton_vertices(
        vertices,
        lines,
        snap_tolerance=float(weld_tolerance),
    )
    welded_vertices = np.asarray(welded_vertices, dtype=np.float64)
    welded_lines = np.asarray(welded_lines, dtype=np.int64).reshape(-1, 2)

    adjacency = _adjacency(len(welded_vertices), welded_lines)
    labels, component_count = _component_labels(adjacency)
    cycle_rank = int(len(welded_lines) - len(welded_vertices) + component_count)
    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.int64)
    if len(welded_lines):
        lengths = np.linalg.norm(
            welded_vertices[welded_lines[:, 1]] - welded_vertices[welded_lines[:, 0]],
            axis=1,
        )
    else:
        lengths = np.empty(0, dtype=np.float64)

    component_edge_counts = np.zeros(component_count, dtype=np.int64)
    for first, _second in welded_lines:
        component_edge_counts[labels[int(first)]] += 1

    component_depths = []
    for component_id in range(component_count):
        members = np.flatnonzero(labels == component_id)
        if len(members):
            component_depths.append(_bfs_depth(adjacency, int(members[0])))

    root = _lowest_root(welded_vertices)
    tree_depth = _bfs_depth(adjacency, root) if component_count == 1 else None
    return {
        "raw_vertex_count": int(len(vertices)),
        "raw_edge_count": int(len(lines)),
        "vertex_count": int(len(welded_vertices)),
        "edge_count": int(len(welded_lines)),
        "component_count": int(component_count),
        "cycle_rank": cycle_rank,
        "strict_tree": bool(
            component_count == 1
            and cycle_rank == 0
            and len(welded_lines) == len(welded_vertices) - 1
        ),
        "connected": bool(component_count == 1),
        "leaf_count": int(np.sum(degrees == 1)),
        "bifurcation_count": int(np.sum(degrees >= 3)),
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "tree_depth": tree_depth,
        "graph_depth_max": int(max(component_depths)) if component_depths else 0,
        "largest_component_edge_ratio": (
            float(component_edge_counts.max() / len(welded_lines))
            if len(welded_lines) and component_count
            else 0.0
        ),
        "total_branch_length": float(lengths.sum()),
        "edge_length_mean": float(lengths.mean()) if len(lengths) else None,
        "edge_length_median": float(np.median(lengths)) if len(lengths) else None,
        "edge_length_p95": float(np.quantile(lengths, 0.95)) if len(lengths) else None,
        "edge_length_max": float(lengths.max()) if len(lengths) else None,
        "coordinate_abs_max": (
            float(np.abs(welded_vertices).max()) if len(welded_vertices) else None
        ),
        "weld_tolerance": float(weld_tolerance),
        "weld": weld,
    }


def _attribute_v2(
    vertices: np.ndarray,
    lines: np.ndarray,
    definition: Mapping[str, object],
    *,
    weld_tolerance: float,
) -> tuple[dict[str, float], dict]:
    """Compute the frozen V2.2 attribute values and their audit summary."""

    vertices, lines = _skeleton_arrays(vertices, lines, context="attributes")
    welded_vertices, welded_lines, weld = weld_skeleton_vertices(
        vertices,
        lines,
        snap_tolerance=float(weld_tolerance),
    )
    vertices = np.asarray(welded_vertices, dtype=np.float64)
    lines = np.asarray(welded_lines, dtype=np.int64).reshape(-1, 2)
    if not len(vertices) or not len(lines):
        raise ValueError("V2.2 attributes require a nonempty skeleton")

    height_definition = definition["height"]
    crown_definition = definition["crown_width"]
    density_definition = definition["branch_density"]
    height_top_quantile = float(height_definition["top_quantile"])
    base_quantile = float(crown_definition["crown_base_quantile"])
    crown_top_quantile = float(crown_definition["crown_top_quantile"])
    span_low_quantile = float(crown_definition["span_low_quantile"])
    span_high_quantile = float(crown_definition["span_high_quantile"])
    for key in (
        "crown_base_quantile",
        "crown_top_quantile",
        "span_low_quantile",
        "span_high_quantile",
    ):
        if not np.isclose(float(crown_definition[key]), float(density_definition[key])):
            raise ValueError("V2.2 crown and density regions must match")

    z = vertices[:, 2]
    minimum_z = float(z.min())
    height_top_z = float(np.quantile(z, height_top_quantile))
    crown_top_z = float(np.quantile(z, crown_top_quantile))
    crown_base_z = float(np.quantile(z, base_quantile))
    crown_mask = (z >= crown_base_z) & (z <= crown_top_z)
    if int(crown_mask.sum()) < 8:
        crown_mask = np.ones(len(vertices), dtype=bool)
        crown_base_z = minimum_z

    crown_xy = vertices[crown_mask, :2]
    centered = crown_xy - crown_xy.mean(axis=0)
    if len(crown_xy) >= 3:
        _u, _s, axes = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ axes.T
    else:
        projected = centered
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    spans = np.quantile(projected[:, :2], span_high_quantile, axis=0) - np.quantile(
        projected[:, :2], span_low_quantile, axis=0
    )
    width_primary = float(max(spans[0], 0.0))
    width_secondary = float(max(spans[1], 0.0))
    crown_width = float(0.5 * (width_primary + width_secondary))

    edge_vectors = vertices[lines[:, 1]] - vertices[lines[:, 0]]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    mid_z = 0.5 * (vertices[lines[:, 0], 2] + vertices[lines[:, 1], 2])
    crown_edge_mask = (mid_z >= crown_base_z) & (mid_z <= crown_top_z)
    crown_skeleton_length = float(lengths[crown_edge_mask].sum())
    crown_height = float(max(crown_top_z - crown_base_z, 0.0))
    crown_box_volume = float(crown_height * width_primary * width_secondary)
    adjacency = _adjacency(len(vertices), lines)
    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.int64)
    crown_fork_count = int(np.sum((degrees >= 3) & crown_mask))

    return (
        {
            "height": float(height_top_z - minimum_z),
            "crown_width": crown_width,
            "branch_density": float(crown_skeleton_length / max(crown_box_volume, EPS)),
        },
        {
            "definition_version": definition.get("definition_version"),
            "selected_candidate_id": definition.get("selected_candidate_id"),
            "crown_base_z": crown_base_z,
            "crown_top_z": crown_top_z,
            "crown_height": crown_height,
            "crown_vertex_count": int(crown_mask.sum()),
            "crown_edge_count": int(crown_edge_mask.sum()),
            "crown_width_primary": width_primary,
            "crown_width_secondary": width_secondary,
            "crown_box_volume": crown_box_volume,
            "crown_skeleton_length": crown_skeleton_length,
            "crown_fork_count": crown_fork_count,
            "weld": weld,
        },
    )


def measure_geometry(
    vertices: np.ndarray,
    lines: np.ndarray,
    *,
    attribute_definition: Mapping[str, object] | None = None,
    weld_tolerance: float = DEFAULT_WELD_TOLERANCE,
) -> dict:
    """Measure topology, structure, and optionally the frozen attributes."""

    result = {
        "structure": graph_metrics(
            vertices,
            lines,
            weld_tolerance=weld_tolerance,
        ),
        "attributes": None,
        "attribute_summary": None,
    }
    if attribute_definition is not None:
        values, summary = _attribute_v2(
            vertices,
            lines,
            attribute_definition,
            weld_tolerance=weld_tolerance,
        )
        result["attributes"] = values
        result["attribute_summary"] = summary
    return result


def build_train_attribute_references(
    payload: Mapping[str, object],
    *,
    species_order: Iterable[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Build train-only native-unit references from the frozen attribute JSON."""

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("attribute payload has no samples list")
    order = tuple(species_order or sorted({str(row["species"]) for row in samples}))
    references: dict[str, dict[str, np.ndarray]] = {}
    for species in order:
        rows = [
            row for row in samples if row.get("species") == species and row.get("split") == "train"
        ]
        if not rows:
            raise ValueError(f"no train attribute rows for species {species!r}")
        references[species] = {
            name: np.sort(
                np.asarray(
                    [float(row["attributes"][name]["value"]) for row in rows],
                    dtype=np.float64,
                )
            )
            for name in ATTRIBUTE_NAMES
        }
    return references


def train_cdf_percentile(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1 or not len(reference):
        raise ValueError("CDF reference must be a nonempty one-dimensional array")
    left = int(np.searchsorted(reference, float(value), side="left"))
    right = int(np.searchsorted(reference, float(value), side="right"))
    rank = 0.5 * (left + right - 1) if right > left else left - 0.5
    return float(np.clip(rank / max(len(reference) - 1, 1), 0.0, 1.0))


def compare_attribute_metrics(
    original: Mapping[str, float],
    reconstruction: Mapping[str, float],
    references: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Return native-unit and shared train-CDF errors for one pair of trees."""

    comparison = {}
    for name in ATTRIBUTE_NAMES:
        original_value = float(original[name])
        reconstruction_value = float(reconstruction[name])
        original_percentile = train_cdf_percentile(original_value, references[name])
        reconstruction_percentile = train_cdf_percentile(
            reconstruction_value,
            references[name],
        )
        comparison[name] = {
            "original_value": original_value,
            "reconstruction_value": reconstruction_value,
            "absolute_error": abs(reconstruction_value - original_value),
            "signed_error": reconstruction_value - original_value,
            "original_percentile": original_percentile,
            "reconstruction_percentile": reconstruction_percentile,
            "absolute_percentile_error": abs(reconstruction_percentile - original_percentile),
            "signed_percentile_error": reconstruction_percentile - original_percentile,
        }
    return comparison


def sample_skeleton_points(
    vertices: np.ndarray,
    lines: np.ndarray,
    *,
    point_count: int = DEFAULT_POINT_COUNT,
) -> np.ndarray:
    """Deterministically sample midpoints along the edge-arc-length measure."""

    if point_count < 1:
        raise ValueError("point_count must be positive")
    vertices, lines = _skeleton_arrays(vertices, lines, context="sampling")
    if len(lines):
        starts = vertices[lines[:, 0]]
        ends = vertices[lines[:, 1]]
        vectors = ends - starts
        lengths = np.linalg.norm(vectors, axis=1)
        positive = lengths > EPS
        starts = starts[positive]
        vectors = vectors[positive]
        lengths = lengths[positive]
    else:
        starts = np.empty((0, 3), dtype=np.float64)
        vectors = np.empty((0, 3), dtype=np.float64)
        lengths = np.empty(0, dtype=np.float64)

    if not len(lengths) or float(lengths.sum()) <= EPS:
        if not len(vertices):
            raise ValueError("cannot sample an empty skeleton")
        indices = np.linspace(0, len(vertices) - 1, point_count).round().astype(int)
        return vertices[indices].copy()

    cumulative = np.cumsum(lengths)
    total = float(cumulative[-1])
    targets = (np.arange(point_count, dtype=np.float64) + 0.5) * total / point_count
    edge_indices = np.searchsorted(cumulative, targets, side="right")
    edge_indices = np.minimum(edge_indices, len(lengths) - 1)
    previous = np.zeros_like(targets)
    mask = edge_indices > 0
    previous[mask] = cumulative[edge_indices[mask] - 1]
    fractions = (targets - previous) / lengths[edge_indices]
    return starts[edge_indices] + vectors[edge_indices] * fractions[:, None]


def _nearest_squared_distances(
    queries: np.ndarray,
    targets: np.ndarray,
    *,
    chunk_size: int = DEFAULT_DISTANCE_CHUNK,
) -> np.ndarray:
    if not len(queries) or not len(targets):
        raise ValueError("Chamfer inputs must be nonempty")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        delta = queries[start:stop, None, :] - targets[None, :, :]
        result[start:stop] = np.min(np.sum(delta * delta, axis=2), axis=1)
    return result


def squared_chamfer(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    chunk_size: int = DEFAULT_DISTANCE_CHUNK,
) -> dict[str, float]:
    """Compute the protocol's symmetric squared-L2 Chamfer distance."""

    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if points_a.ndim != 2 or points_a.shape[1] != 3:
        raise ValueError("points_a must have shape (N, 3)")
    if points_b.ndim != 2 or points_b.shape[1] != 3:
        raise ValueError("points_b must have shape (M, 3)")
    a_to_b = _nearest_squared_distances(points_a, points_b, chunk_size=chunk_size)
    b_to_a = _nearest_squared_distances(points_b, points_a, chunk_size=chunk_size)
    a_mean = float(a_to_b.mean())
    b_mean = float(b_to_a.mean())
    return {
        "symmetric_squared_l2": a_mean + b_mean,
        "a_to_b": a_mean,
        "b_to_a": b_mean,
        "point_count_a": int(len(points_a)),
        "point_count_b": int(len(points_b)),
        "distance_chunk_size": int(chunk_size),
    }


def skeleton_chamfer(
    vertices_a: np.ndarray,
    lines_a: np.ndarray,
    vertices_b: np.ndarray,
    lines_b: np.ndarray,
    *,
    point_count: int = DEFAULT_POINT_COUNT,
    chunk_size: int = DEFAULT_DISTANCE_CHUNK,
) -> dict[str, float]:
    """Sample two skeletons identically and return squared Chamfer metrics."""

    points_a = sample_skeleton_points(vertices_a, lines_a, point_count=point_count)
    points_b = sample_skeleton_points(vertices_b, lines_b, point_count=point_count)
    result = squared_chamfer(points_a, points_b, chunk_size=chunk_size)
    result.update(
        {
            "point_sampling": "edge_arc_length_midpoints",
            "point_count_requested": int(point_count),
            "distance": "squared_l2_symmetric_sum",
        }
    )
    return result


def load_attribute_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
