from __future__ import annotations
from collections import deque
from pathlib import Path
import numpy as np
from .graph import compute_attributes
from .assembly_utils import (
    adjacency_for,
    edge_lengths,
    edge_statistics,
    spatial_topology,
    validate_tree,
)


def angle_between(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def numeric_prior(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot fit an empty morphology prior")
    return {
        "count": int(len(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
        "mean": float(array.mean()),
    }


def robust_angle_score(angle: float, prior: dict) -> float:
    scale = max(
        float(prior["q75"] - prior["q25"]),
        np.radians(5.0),
    )
    return abs(float(angle) - float(prior["median"])) / scale


def stable_causal_forest(
    segments: np.ndarray,
    weld_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Stable-weld raw segments and reject duplicates and causal cycles."""

    raw_vertices = np.asarray(segments, dtype=np.float64).reshape(-1, 3)
    if not len(raw_vertices):
        raise ValueError("raw segment stream is empty")
    keys = np.round(raw_vertices / weld_tolerance).astype(np.int64)
    vertex_by_key: dict[tuple[int, int, int], int] = {}
    welded_vertices: list[np.ndarray] = []
    inverse = np.empty(len(raw_vertices), dtype=np.int64)
    for raw_index, (key, point) in enumerate(zip(keys, raw_vertices)):
        tuple_key = tuple(int(value) for value in key)
        vertex = vertex_by_key.get(tuple_key)
        if vertex is None:
            vertex = len(welded_vertices)
            vertex_by_key[tuple_key] = vertex
            welded_vertices.append(point.copy())
        inverse[raw_index] = vertex

    parent = np.arange(len(welded_vertices), dtype=np.int64)
    rank = np.zeros(len(welded_vertices), dtype=np.int8)

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return vertex

    def union(first: int, second: int) -> bool:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return False
        if rank[first_root] < rank[second_root]:
            parent[first_root] = second_root
        elif rank[first_root] > rank[second_root]:
            parent[second_root] = first_root
        else:
            parent[second_root] = first_root
            rank[first_root] += 1
        return True

    kept_lines: list[tuple[int, int]] = []
    source_edge_ids: list[int] = []
    seen_edges: set[tuple[int, int]] = set()
    degenerate_count = 0
    duplicate_count = 0
    cycle_count = 0
    for source_edge, (first, second) in enumerate(inverse.reshape(-1, 2)):
        first = int(first)
        second = int(second)
        if first == second:
            degenerate_count += 1
            continue
        edge_key = (min(first, second), max(first, second))
        if edge_key in seen_edges:
            duplicate_count += 1
            continue
        seen_edges.add(edge_key)
        if not union(first, second):
            cycle_count += 1
            continue
        kept_lines.append((first, second))
        source_edge_ids.append(source_edge)

    return (
        np.asarray(welded_vertices, dtype=np.float64),
        np.asarray(kept_lines, dtype=np.int64).reshape(-1, 2),
        np.asarray(source_edge_ids, dtype=np.int64),
        {
            "raw_vertex_count": int(len(raw_vertices)),
            "raw_edge_count": int(len(segments)),
            "welded_vertex_count": int(len(welded_vertices)),
            "causal_forest_edge_count": int(len(kept_lines)),
            "welded_vertices_removed": int(len(raw_vertices) - len(welded_vertices)),
            "degenerate_edges_removed": int(degenerate_count),
            "duplicate_edges_removed": int(duplicate_count),
            "cycle_edges_removed": int(cycle_count),
        },
    )


def forest_components(
    vertex_count: int,
    lines: np.ndarray,
    source_edge_ids: np.ndarray,
) -> list[dict]:
    """Return non-isolated tree components in causal order."""

    adjacency = adjacency_for(vertex_count, lines)
    visited = np.zeros(vertex_count, dtype=bool)
    components: list[dict] = []
    for start in range(vertex_count):
        if visited[start] or not adjacency[start]:
            continue
        vertices: list[int] = []
        edge_indices: set[int] = set()
        queue = deque([start])
        visited[start] = True
        while queue:
            current = queue.popleft()
            vertices.append(current)
            for neighbor, edge_index in adjacency[current]:
                edge_indices.add(int(edge_index))
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        edges = np.asarray(sorted(edge_indices), dtype=np.int64)
        earliest_local = int(edges[np.argmin(source_edge_ids[edges])])
        root = int(lines[earliest_local, 0])
        components.append(
            {
                "vertices": np.asarray(sorted(vertices), dtype=np.int64),
                "edge_indices": edges,
                "root": root,
                "first_source_edge": int(source_edge_ids[earliest_local]),
                "last_source_edge": int(source_edge_ids[edges].max()),
            }
        )
    components.sort(
        key=lambda item: (
            item["first_source_edge"],
            item["root"],
        )
    )
    return components


def rooted_parent(
    vertex_count: int,
    lines: np.ndarray,
    root: int,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    adjacency = adjacency_for(vertex_count, lines)
    parent = np.full(vertex_count, -1, dtype=np.int64)
    parent[root] = root
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor, _edge_index in adjacency[current]:
            if parent[neighbor] >= 0:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if np.any(parent < 0):
        raise RuntimeError("assembled tree orientation failed")
    return parent, adjacency


def fit_forest_continuation_prior(
    vertices: np.ndarray,
    lines: np.ndarray,
    components: list[dict],
) -> dict:
    adjacency = adjacency_for(len(vertices), lines)
    angles: list[float] = []
    for component in components:
        component_vertices = set(int(value) for value in component["vertices"])
        root = int(component["root"])
        parent = {root: root}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor, _edge_index in adjacency[current]:
                if neighbor not in component_vertices or neighbor in parent:
                    continue
                parent[neighbor] = current
                queue.append(neighbor)
        for vertex in component_vertices:
            if vertex == root or len(adjacency[vertex]) != 2:
                continue
            children = [
                neighbor
                for neighbor, _edge_index in adjacency[vertex]
                if parent.get(neighbor) == vertex
            ]
            if len(children) != 1:
                continue
            angle = angle_between(
                vertices[vertex] - vertices[parent[vertex]],
                vertices[children[0]] - vertices[vertex],
            )
            if angle is not None:
                angles.append(angle)
    if not angles:
        angles = [0.0]
    prior = numeric_prior(angles)
    prior["unit"] = "radian"
    prior["degrees"] = {
        key: float(np.degrees(value))
        for key, value in prior.items()
        if key not in {"count", "unit"}
    }
    return prior


def component_child_vectors(
    vertices: np.ndarray,
    lines: np.ndarray,
    component: dict,
) -> list[np.ndarray]:
    root = int(component["root"])
    vectors = []
    for edge_index in component["edge_indices"]:
        first, second = (int(value) for value in lines[int(edge_index)])
        if first == root:
            vectors.append(vertices[second] - vertices[root])
        elif second == root:
            vectors.append(vertices[first] - vertices[root])
    return vectors


def collision_audit(
    moved_points: np.ndarray,
    assembled_vertices: np.ndarray,
    assembled_adjacency: list[list[tuple[int, int]]],
    target: int,
    *,
    clearance: float,
    weld_tolerance: float,
) -> tuple[float, float | None, int, float | None]:
    if not len(moved_points):
        return 0.0, None, 0, None
    excluded = {int(target)}
    frontier = {int(target)}
    for _hop in range(2):
        next_frontier: set[int] = set()
        for vertex in frontier:
            for neighbor, _edge_index in assembled_adjacency[vertex]:
                if neighbor not in excluded:
                    excluded.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    order = np.argsort(assembled_vertices[:, 0], kind="stable")
    sorted_x = assembled_vertices[order, 0]
    clearance_squared = float(clearance * clearance)
    tolerance_squared = float(weld_tolerance * weld_tolerance)
    penalty = 0.0
    minimum_squared: float | None = None
    exact_minimum_squared: float | None = None
    exact_count = 0
    for point in moved_points:
        first = int(np.searchsorted(sorted_x, point[0] - clearance, side="left"))
        last = int(np.searchsorted(sorted_x, point[0] + clearance, side="right"))
        if first == last:
            continue
        indices = order[first:last]
        squared = np.einsum(
            "ij,ij->i",
            assembled_vertices[indices] - point,
            assembled_vertices[indices] - point,
        )
        exact_mask = squared <= tolerance_squared
        if np.any(exact_mask):
            exact_indices = indices[exact_mask]
            if len(exact_indices):
                exact_count += 1
                exact_value = float(
                    np.min(
                        np.sum(
                            (assembled_vertices[exact_indices] - point) ** 2,
                            axis=1,
                        )
                    )
                )
                if exact_minimum_squared is None or exact_value < exact_minimum_squared:
                    exact_minimum_squared = exact_value

        soft_mask = np.asarray(
            [int(index) not in excluded for index in indices],
            dtype=bool,
        )
        if not np.any(soft_mask):
            continue
        candidate_minimum = float(squared[soft_mask].min())
        if minimum_squared is None or candidate_minimum < minimum_squared:
            minimum_squared = candidate_minimum
        if candidate_minimum < clearance_squared:
            penalty += (clearance - float(np.sqrt(candidate_minimum))) / clearance
    return (
        float(penalty / len(moved_points)),
        None if minimum_squared is None else float(np.sqrt(minimum_squared)),
        int(exact_count),
        (None if exact_minimum_squared is None else float(np.sqrt(exact_minimum_squared))),
    )


def score_component_candidate(
    *,
    target: int,
    component_root_point: np.ndarray,
    component_other_points: np.ndarray,
    child_vectors: list[np.ndarray],
    assembled_vertices: np.ndarray,
    parent: np.ndarray,
    adjacency: list[list[tuple[int, int]]],
    assembled_root: int,
    root_xy: np.ndarray,
    threshold: float,
    clearance: float,
    weld_tolerance: float,
    training_prior: dict,
    continuation_prior: dict,
    weights: dict,
) -> dict:
    translation = assembled_vertices[target] - component_root_point
    translation_norm = float(np.linalg.norm(translation))
    new_degree = int(len(adjacency[target]) + len(child_vectors))
    angle_prior = continuation_prior if new_degree == 2 else training_prior["branch_angle"]
    attachment_angles: list[float] = []
    angle_scores: list[float] = []
    if target == assembled_root:
        angle_score = 2.0
    else:
        incoming = assembled_vertices[target] - assembled_vertices[parent[target]]
        for child_vector in child_vectors:
            angle = angle_between(incoming, child_vector)
            if angle is None:
                continue
            attachment_angles.append(angle)
            angle_scores.append(robust_angle_score(angle, angle_prior))
        angle_score = float(np.mean(angle_scores)) if angle_scores else 2.0

    degree_soft_limit = int(round(training_prior["junction_degree"]["q99"]))
    degree_excess = max(0, new_degree - degree_soft_limit)
    degree_score = float(degree_excess * degree_excess)

    radial = assembled_vertices[target, :2] - root_xy
    radial_norm = float(np.linalg.norm(radial))
    outward_penalties: list[float] = []
    if radial_norm > 1e-12:
        for child_vector in child_vectors:
            horizontal = child_vector[:2]
            horizontal_norm = float(np.linalg.norm(horizontal))
            if horizontal_norm <= 1e-12:
                continue
            cosine = float(np.dot(radial, horizontal) / (radial_norm * horizontal_norm))
            outward_penalties.append(max(0.0, -cosine))
    outward_score = float(np.mean(outward_penalties)) if outward_penalties else 0.0

    moved_points = component_other_points + translation
    (
        overlap_score,
        minimum_clearance,
        exact_collision_count,
        exact_minimum_clearance,
    ) = collision_audit(
        moved_points,
        assembled_vertices,
        adjacency,
        target,
        clearance=clearance,
        weld_tolerance=weld_tolerance,
    )
    distance_score = translation_norm / threshold
    total = (
        weights["distance"] * distance_score
        + weights["angle"] * angle_score
        + weights["degree"] * degree_score
        + weights["outward"] * outward_score
        + weights["collision"] * overlap_score
    )
    valid = exact_collision_count == 0
    if not valid:
        total += 1_000_000.0 + exact_collision_count
    return {
        "target_vertex": int(target),
        "valid": bool(valid),
        "total_score": float(total),
        "translation": translation.tolist(),
        "translation_norm": translation_norm,
        "distance_score": float(distance_score),
        "attachment_angles_degrees": [float(np.degrees(value)) for value in attachment_angles],
        "angle_score": float(angle_score),
        "angle_prior": (
            "raw_forest_local_continuation" if new_degree == 2 else "training_reference_branch"
        ),
        "new_degree": new_degree,
        "degree_soft_limit": degree_soft_limit,
        "degree_score": degree_score,
        "outward_score": outward_score,
        "collision_score": overlap_score,
        "minimum_nonlocal_clearance": minimum_clearance,
        "exact_collision_count": exact_collision_count,
        "exact_minimum_clearance": exact_minimum_clearance,
    }


def assemble_components(
    forest_vertices: np.ndarray,
    forest_lines: np.ndarray,
    forest_source_edge_ids: np.ndarray,
    components: list[dict],
    *,
    threshold: float,
    candidate_count: int,
    weld_tolerance: float,
    training_prior: dict,
    continuation_prior: dict,
    weights: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    if not components:
        raise ValueError("no non-isolated short-edge components")
    clearance = max(
        0.6 * float(training_prior["pooled_edge_median"]),
        1e-6,
    )

    anchor = components[0]
    anchor_vertices = anchor["vertices"]
    anchor_map = {int(old): new for new, old in enumerate(anchor_vertices)}
    assembled_vertices = forest_vertices[anchor_vertices].copy()
    assembled_lines = np.asarray(
        [
            (
                anchor_map[int(forest_lines[edge_index, 0])],
                anchor_map[int(forest_lines[edge_index, 1])],
            )
            for edge_index in anchor["edge_indices"]
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    assembled_source_ids = forest_source_edge_ids[anchor["edge_indices"]].copy()
    edge_component_ids = np.zeros(len(assembled_lines), dtype=np.int32)
    assembled_root = int(anchor_map[int(anchor["root"])])
    root_xy = assembled_vertices[assembled_root, :2].copy()
    actions: list[dict] = []

    for action_id, component in enumerate(components[1:], start=1):
        parent, adjacency = rooted_parent(
            len(assembled_vertices),
            assembled_lines,
            assembled_root,
        )
        component_vertices = component["vertices"]
        component_root = int(component["root"])
        other_vertices = component_vertices[component_vertices != component_root]
        child_vectors = component_child_vectors(forest_vertices, forest_lines, component)
        distances = np.linalg.norm(
            assembled_vertices - forest_vertices[component_root],
            axis=1,
        )
        order = np.argsort(distances, kind="stable")
        shortlist = order[: min(candidate_count, len(order))]

        def evaluate(targets: np.ndarray) -> list[dict]:
            return [
                score_component_candidate(
                    target=int(target),
                    component_root_point=forest_vertices[component_root],
                    component_other_points=forest_vertices[other_vertices],
                    child_vectors=child_vectors,
                    assembled_vertices=assembled_vertices,
                    parent=parent,
                    adjacency=adjacency,
                    assembled_root=assembled_root,
                    root_xy=root_xy,
                    threshold=threshold,
                    clearance=clearance,
                    weld_tolerance=weld_tolerance,
                    training_prior=training_prior,
                    continuation_prior=continuation_prior,
                    weights=weights,
                )
                for target in targets
            ]

        candidate_audit = evaluate(shortlist)
        valid_candidates = [item for item in candidate_audit if item["valid"]]
        if not valid_candidates and len(shortlist) < len(order):
            remaining = order[len(shortlist) :]
            candidate_audit.extend(evaluate(remaining))
            valid_candidates = [item for item in candidate_audit if item["valid"]]
        if not valid_candidates:
            raise RuntimeError(
                f"no direct BioAttach candidate avoids an unintended weld for component {action_id}"
            )
        candidate_audit.sort(
            key=lambda item: (
                item["total_score"],
                item["translation_norm"],
                item["target_vertex"],
            )
        )
        selected = next(item for item in candidate_audit if item["valid"])
        target = int(selected["target_vertex"])
        translation = np.asarray(selected["translation"], dtype=np.float64)

        local_to_global = {component_root: target}
        new_points = forest_vertices[other_vertices] + translation
        new_start = len(assembled_vertices)
        for offset, old_vertex in enumerate(other_vertices):
            local_to_global[int(old_vertex)] = new_start + offset
        assembled_vertices = np.concatenate((assembled_vertices, new_points), axis=0)
        new_lines = np.asarray(
            [
                (
                    local_to_global[int(forest_lines[edge_index, 0])],
                    local_to_global[int(forest_lines[edge_index, 1])],
                )
                for edge_index in component["edge_indices"]
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        assembled_lines = np.concatenate((assembled_lines, new_lines), axis=0)
        new_source_ids = forest_source_edge_ids[component["edge_indices"]]
        assembled_source_ids = np.concatenate((assembled_source_ids, new_source_ids), axis=0)
        edge_component_ids = np.concatenate(
            (
                edge_component_ids,
                np.full(len(new_lines), action_id, dtype=np.int32),
            ),
            axis=0,
        )
        actions.append(
            {
                "action_id": action_id,
                "component_root_vertex": component_root,
                "component_vertex_count_before_merge": int(len(component_vertices)),
                "component_edge_count": int(len(component["edge_indices"])),
                "first_source_edge": int(component["first_source_edge"]),
                "last_source_edge": int(component["last_source_edge"]),
                "selected_candidate": selected,
                "top_candidates": candidate_audit[:5],
            }
        )

    return (
        assembled_vertices,
        assembled_lines,
        edge_component_ids,
        assembled_source_ids,
        actions,
    )


def build_direct_geometry(
    raw_segments: np.ndarray,
    training_prior: dict,
    *,
    absolute_threshold: float = 0.03,
    quantile: float = 0.99,
    quantile_multiplier: float = 4.0,
    candidate_count: int = 64,
    weld_tolerance: float = 1e-7,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run Direct in memory and return geometry plus its audit report."""

    raw_segments = np.asarray(raw_segments, dtype=np.float64)
    if raw_segments.ndim != 3 or raw_segments.shape[1:] != (2, 3):
        raise ValueError("raw_segments must have shape (E, 2, 3)")
    if not len(raw_segments):
        raise ValueError("raw_segments is empty")
    if (
        absolute_threshold <= 0
        or not 0 < quantile < 1
        or quantile_multiplier <= 0
        or candidate_count < 1
        or weld_tolerance <= 0
    ):
        raise ValueError("invalid Direct BioAttach arguments")
    weights = dict(
        weights
        or {
            "distance": 1.0,
            "angle": 0.75,
            "degree": 0.65,
            "outward": 0.35,
            "collision": 1.1,
        }
    )
    if any(value < 0 for value in weights.values()):
        raise ValueError("Direct BioAttach weights must be non-negative")

    raw_lengths = np.linalg.norm(raw_segments[:, 1] - raw_segments[:, 0], axis=1)
    quantile_value = float(np.quantile(raw_lengths, quantile))
    threshold = max(float(absolute_threshold), quantile_value * float(quantile_multiplier))
    (
        forest_vertices,
        causal_lines,
        causal_source_ids,
        preprocessing,
    ) = stable_causal_forest(raw_segments, weld_tolerance)
    causal_lengths = edge_lengths(forest_vertices, causal_lines)
    long_mask = causal_lengths > threshold
    short_lines = causal_lines[~long_mask]
    short_source_ids = causal_source_ids[~long_mask]
    preprocessing["long_edges_removed"] = int(long_mask.sum())
    preprocessing["short_forest_edge_count"] = int(len(short_lines))
    components = forest_components(len(forest_vertices), short_lines, short_source_ids)
    preprocessing["short_forest_component_count"] = int(len(components))
    preprocessing["isolated_vertices_discarded"] = int(
        len(forest_vertices)
        - len(
            np.unique(short_lines.reshape(-1)) if len(short_lines) else np.empty(0, dtype=np.int64)
        )
    )

    continuation_prior = fit_forest_continuation_prior(forest_vertices, short_lines, components)
    (
        vertices,
        lines,
        edge_component_ids,
        source_edge_ids,
        actions,
    ) = assemble_components(
        forest_vertices,
        short_lines,
        short_source_ids,
        components,
        threshold=threshold,
        candidate_count=candidate_count,
        weld_tolerance=weld_tolerance,
        training_prior=training_prior,
        continuation_prior=continuation_prior,
        weights=weights,
    )

    topology = validate_tree(vertices, lines, context="Direct BioAttach output")
    welded_topology = spatial_topology(vertices, lines, weld_tolerance)
    if not welded_topology["strict_tree"]:
        raise RuntimeError("Direct BioAttach output failed spatial topology validation")
    output_lengths = edge_lengths(vertices, lines)
    output_vectors = vertices[lines[:, 1]] - vertices[lines[:, 0]]
    source_vectors = raw_segments[source_edge_ids, 1] - raw_segments[source_edge_ids, 0]
    vector_errors = np.linalg.norm(output_vectors - source_vectors, axis=1)

    raw_vertices = raw_segments.reshape(-1, 3)
    raw_lines = np.arange(len(raw_vertices), dtype=np.int64).reshape(-1, 2)
    raw_attributes, raw_quality, raw_summary = compute_attributes(
        raw_vertices, raw_lines, component_mode="all"
    )
    attributes, quality, summary = compute_attributes(vertices, lines, component_mode="all")
    report = {
        "schema_version": 1,
        "status": "completed",
        "method": "direct_morphology_aware_component_bioattach_v1",
        "threshold": {
            "effective": threshold,
            "absolute_floor": float(absolute_threshold),
            "quantile": float(quantile),
            "quantile_value": quantile_value,
            "quantile_multiplier": float(quantile_multiplier),
        },
        "candidate_count": int(candidate_count),
        "weld_tolerance": float(weld_tolerance),
        "weights": weights,
        "training_morphology_prior": training_prior,
        "local_continuation_prior": continuation_prior,
        "preprocessing": preprocessing,
        "output": {
            "indexed_topology": topology,
            "spatially_welded_topology": welded_topology,
            "edge_lengths": edge_statistics(output_lengths),
            "attributes": {name: float(value) for name, value in attributes.items()},
            "attribute_quality": quality,
            "attribute_summary": summary,
        },
        "raw": {
            "edge_lengths": edge_statistics(raw_lengths),
            "attributes": {name: float(value) for name, value in raw_attributes.items()},
            "attribute_quality": raw_quality,
            "attribute_summary": raw_summary,
        },
        "preserved_edge_vectors": {
            "count": int(len(vector_errors)),
            "maximum_error": float(vector_errors.max()) if len(vector_errors) else 0.0,
            "mean_error": float(vector_errors.mean()) if len(vector_errors) else 0.0,
        },
        "retained_source_edge_count": int(len(source_edge_ids)),
        "retained_source_edge_ratio": float(len(source_edge_ids) / max(len(raw_segments), 1)),
        "component_attachment_count": int(len(actions)),
        "actions": actions,
        "lineage": {
            "edge_component_ids": edge_component_ids.astype(np.int32).tolist(),
            "source_edge_indices": source_edge_ids.astype(np.int32).tolist(),
        },
    }
    return vertices.astype(np.float64), lines.astype(np.int64), report
