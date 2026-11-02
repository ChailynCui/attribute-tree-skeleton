from collections import Counter, defaultdict, deque
import math
import numpy as np

DEFAULT_WELD_TOLERANCE = 1e-6
EPS = 1e-8


def build_adjacency(lines: np.ndarray, num_vertices: int):
    adjacency = [[] for _ in range(num_vertices)]
    for edge_index, (a, b) in enumerate(lines):
        a = int(a)
        b = int(b)
        adjacency[a].append((b, edge_index))
        adjacency[b].append((a, edge_index))
    return adjacency


def weld_skeleton_vertices(
    vertices: np.ndarray,
    lines: np.ndarray,
    snap_tolerance: float = DEFAULT_WELD_TOLERANCE,
):
    vertices = np.asarray(vertices, dtype=float)
    lines = np.asarray(lines, dtype=np.int64)
    if lines.size == 0:
        lines = lines.reshape(0, 2)
    else:
        lines = lines.reshape(-1, 2)
    raw_num_vertices = int(len(vertices))
    raw_num_edges = int(len(lines))

    if raw_num_vertices == 0:
        return (
            vertices,
            lines,
            {
                "raw_num_vertices": raw_num_vertices,
                "raw_num_edges": raw_num_edges,
                "welded_num_vertices": 0,
                "welded_num_edges": 0,
                "welded_vertices_removed": 0,
                "degenerate_edges_removed": 0,
                "duplicate_edges_removed": 0,
            },
        )

    if snap_tolerance > 0:
        keys = np.round(vertices / float(snap_tolerance)).astype(np.int64)
    else:
        keys = vertices

    _, first_indices, inverse = np.unique(
        keys,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    order = np.argsort(first_indices)
    old_unique_to_new = np.empty(len(first_indices), dtype=np.int64)
    old_unique_to_new[order] = np.arange(len(first_indices), dtype=np.int64)
    welded_vertices = vertices[first_indices[order]]
    remapped_lines = old_unique_to_new[inverse[lines]]

    nondegenerate_mask = remapped_lines[:, 0] != remapped_lines[:, 1]
    nondegenerate_lines = remapped_lines[nondegenerate_mask]

    if len(nondegenerate_lines) > 0:
        edge_keys = np.sort(nondegenerate_lines, axis=1)
        _, edge_first_indices = np.unique(edge_keys, axis=0, return_index=True)
        welded_lines = nondegenerate_lines[np.sort(edge_first_indices)]
    else:
        welded_lines = nondegenerate_lines

    stats = {
        "raw_num_vertices": raw_num_vertices,
        "raw_num_edges": raw_num_edges,
        "welded_num_vertices": int(len(welded_vertices)),
        "welded_num_edges": int(len(welded_lines)),
        "welded_vertices_removed": int(raw_num_vertices - len(welded_vertices)),
        "degenerate_edges_removed": int(raw_num_edges - int(nondegenerate_mask.sum())),
        "duplicate_edges_removed": int(len(nondegenerate_lines) - len(welded_lines)),
    }
    return welded_vertices, welded_lines.astype(np.int32, copy=False), stats


def graph_quality(adjacency, num_vertices: int, num_edges: int):
    visited = np.zeros(num_vertices, dtype=bool)
    component_count = 0
    isolated_count = 0
    for vertex in range(num_vertices):
        if visited[vertex]:
            continue
        component_count += 1
        if not adjacency[vertex]:
            visited[vertex] = True
            isolated_count += 1
            continue
        queue = deque([vertex])
        visited[vertex] = True
        while queue:
            cur = queue.popleft()
            for nxt, _ in adjacency[cur]:
                if visited[nxt]:
                    continue
                visited[nxt] = True
                queue.append(nxt)

    cycle_rank = max(0, int(num_edges - num_vertices + component_count))
    return {
        "num_components": int(component_count),
        "num_isolated_vertices": int(isolated_count),
        "cycle_rank": int(cycle_rank),
    }


def connected_components(adjacency):
    visited = np.zeros(len(adjacency), dtype=bool)
    components = []
    for vertex in range(len(adjacency)):
        if visited[vertex]:
            continue
        queue = deque([vertex])
        visited[vertex] = True
        component = []
        while queue:
            cur = queue.popleft()
            component.append(cur)
            for nxt, _ in adjacency[cur]:
                if visited[nxt]:
                    continue
                visited[nxt] = True
                queue.append(nxt)
        components.append(component)
    return components


def subset_skeleton(vertices: np.ndarray, lines: np.ndarray, keep_vertices):
    keep_vertices = np.asarray(sorted(set(int(item) for item in keep_vertices)), dtype=np.int64)
    if len(keep_vertices) == 0:
        raise ValueError("cannot subset skeleton to an empty vertex set")

    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[keep_vertices] = np.arange(len(keep_vertices), dtype=np.int64)

    line_mask = np.isin(lines[:, 0], keep_vertices) & np.isin(lines[:, 1], keep_vertices)
    subset_lines = old_to_new[lines[line_mask]]
    return vertices[keep_vertices], subset_lines.astype(np.int32, copy=False), keep_vertices


def choose_lowest_root(vertices: np.ndarray):
    z_values = vertices[:, 2]
    min_z = float(z_values.min())
    lowest_candidates = np.where(np.isclose(z_values, min_z))[0]

    if len(lowest_candidates) == 1:
        return int(lowest_candidates[0])

    center_xy = np.median(vertices[:, :2], axis=0)
    distances = np.linalg.norm(vertices[lowest_candidates, :2] - center_xy[None, :], axis=1)
    return int(lowest_candidates[int(np.argmin(distances))])


def choose_trunk_anchor(vertices: np.ndarray, adjacency):
    z_values = vertices[:, 2]
    center_xy = np.median(vertices[:, :2], axis=0)
    z_threshold = np.quantile(z_values, 0.05)
    low_candidates = np.where(z_values <= z_threshold)[0]

    if len(low_candidates) == 0:
        return choose_lowest_root(vertices)

    degrees = np.asarray([len(items) for items in adjacency], dtype=np.int32)

    # Pick the lowest branching node (deg>=3) near the base. If no branching
    # node exists in the bottom 5%, fall back to the lowest non-degree-2 node,
    # then to the lowest vertex overall.
    preferred_pools = [
        low_candidates[degrees[low_candidates] >= 3],
        low_candidates[degrees[low_candidates] != 2],
        low_candidates,
    ]

    for pool in preferred_pools:
        if len(pool) == 0:
            continue

        # Prefer low Z, but still bias toward the tree's central column.
        # Weeping forms can have very low dangling branches far from the trunk;
        # giving XY distance a bit more weight keeps the anchor on the main stem.
        z_min = z_values[pool].min()
        z_rank = z_values[pool] - z_min
        xy_dist = np.linalg.norm(vertices[pool, :2] - center_xy[None, :], axis=1)
        score = z_rank * 2.0 + xy_dist * 1.2
        return int(pool[int(np.argmin(score))])

    return int(np.argmin(z_values))


def choose_trunk_base(vertices: np.ndarray, adjacency, trunk_anchor: int | None = None):
    if trunk_anchor is None:
        trunk_anchor = choose_trunk_anchor(vertices, adjacency)

    center_xy = np.median(vertices[:, :2], axis=0)
    current = int(trunk_anchor)
    previous = -1

    while True:
        downward = []
        current_xy = vertices[current, :2]
        current_z = float(vertices[current, 2])
        for nxt, _ in adjacency[current]:
            nxt = int(nxt)
            if nxt == previous:
                continue
            if float(vertices[nxt, 2]) >= current_z - EPS:
                continue
            step_xy = float(np.linalg.norm(vertices[nxt, :2] - current_xy))
            center_dist = float(np.linalg.norm(vertices[nxt, :2] - center_xy))
            downward.append((step_xy, center_dist, float(vertices[nxt, 2]), nxt))

        if not downward:
            break

        downward.sort(key=lambda item: (item[0], item[1], item[2]))
        previous, current = current, int(downward[0][3])

    return current


def choose_root(vertices: np.ndarray, adjacency=None):
    # Backward-compatible alias: "root" now means the geometric lowest point.
    return choose_lowest_root(vertices)


def bfs_parent(adjacency, root: int):
    parent = [-1] * len(adjacency)
    depth = [-1] * len(adjacency)
    queue = deque([root])
    depth[root] = 0
    order = [root]

    while queue:
        cur = queue.popleft()
        for nxt, _ in adjacency[cur]:
            if depth[nxt] != -1:
                continue
            depth[nxt] = depth[cur] + 1
            parent[nxt] = cur
            queue.append(nxt)
            order.append(nxt)
    return parent, depth, order


def path_to_root(parent, node: int):
    path = []
    cur = int(node)
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def build_branch_graph(adjacency, lines: np.ndarray, root: int):
    degrees = np.array([len(items) for items in adjacency], dtype=np.int32)
    special_nodes = {int(index) for index in np.where(degrees != 2)[0].tolist()}
    special_nodes.add(int(root))

    visited_edges = np.zeros(len(lines), dtype=bool)
    branch_records = []
    branch_adjacency = defaultdict(list)

    for start in sorted(special_nodes):
        for neighbor, edge_index in adjacency[start]:
            if visited_edges[edge_index]:
                continue

            path_vertices = [start]
            path_edges = []
            prev = start
            cur = int(neighbor)
            cur_edge = int(edge_index)

            while True:
                visited_edges[cur_edge] = True
                path_edges.append(cur_edge)
                path_vertices.append(cur)

                if cur in special_nodes:
                    end = cur
                    break

                next_candidates = [(nxt, eid) for nxt, eid in adjacency[cur] if nxt != prev]
                if not next_candidates:
                    end = cur
                    special_nodes.add(cur)
                    break

                prev = cur
                cur, cur_edge = int(next_candidates[0][0]), int(next_candidates[0][1])

            branch_id = len(branch_records)
            branch_records.append(
                {
                    "branch_id": branch_id,
                    "start": int(start),
                    "end": int(end),
                    "num_segments": len(path_edges),
                    "vertex_path": path_vertices,
                    "edge_path": path_edges,
                }
            )
            branch_adjacency[start].append((end, branch_id))
            branch_adjacency[end].append((start, branch_id))

    return branch_records, branch_adjacency, degrees


def compute_branch_depth(branch_adjacency, root: int):
    if root not in branch_adjacency:
        return 0
    queue = deque([root])
    depth = {root: 0}
    while queue:
        cur = queue.popleft()
        for nxt, _ in branch_adjacency[cur]:
            if nxt in depth:
                continue
            depth[nxt] = depth[cur] + 1
            queue.append(nxt)
    return int(max(depth.values())) if depth else 0


def compute_attributes(vertices: np.ndarray, lines: np.ndarray, component_mode: str = "all"):
    if component_mode not in {"all", "trunk", "largest"}:
        raise ValueError(f"unknown component_mode: {component_mode}")

    vertices, lines, weld_stats = weld_skeleton_vertices(vertices, lines)
    num_vertices = int(len(vertices))
    num_edges = int(len(lines))

    if num_vertices == 0:
        raise ValueError("empty vertex array")

    adjacency = build_adjacency(lines, num_vertices)
    quality = graph_quality(adjacency, num_vertices, num_edges)

    component_stats = {
        "component_mode": component_mode,
        "selected_component_vertices": num_vertices,
        "selected_component_edges": num_edges,
    }
    if component_mode != "all" and quality["num_components"] > 1:
        if component_mode == "trunk":
            selected_anchor = choose_trunk_anchor(vertices, adjacency)
            _parent, anchor_depth, _order = bfs_parent(adjacency, selected_anchor)
            keep_vertices = np.where(np.asarray(anchor_depth) >= 0)[0]
        else:
            components = connected_components(adjacency)
            keep_vertices = max(
                components,
                key=lambda component: (
                    sum(len(adjacency[int(vertex)]) for vertex in component),
                    len(component),
                ),
            )

        original_vertices = num_vertices
        original_edges = num_edges
        vertices, lines, kept_old_indices = subset_skeleton(vertices, lines, keep_vertices)
        num_vertices = int(len(vertices))
        num_edges = int(len(lines))
        adjacency = build_adjacency(lines, num_vertices)
        component_stats.update(
            {
                "selected_component_vertices": num_vertices,
                "selected_component_edges": num_edges,
                "selected_component_vertex_ratio": float(num_vertices / max(original_vertices, 1)),
                "selected_component_edge_ratio": float(num_edges / max(original_edges, 1)),
                "selected_component_old_min_vertex": int(kept_old_indices.min()),
                "selected_component_old_max_vertex": int(kept_old_indices.max()),
            }
        )

    root_vertex = choose_lowest_root(vertices)
    trunk_anchor = choose_trunk_anchor(vertices, adjacency)
    parent, depth, reachable_order = bfs_parent(adjacency, trunk_anchor)
    reachable_mask = np.array([item >= 0 for item in depth], dtype=bool)

    z_values = vertices[:, 2]
    height = float(np.quantile(z_values, 0.98) - z_values[root_vertex])

    # Crown base: first fork along the trunk path (degree >= 3).
    # This is where the trunk first splits, the biological equivalent of "first live branch".
    reachable_indices_cb = np.where(reachable_mask)[0]
    if len(reachable_indices_cb) == 0:
        highest_vertex = trunk_anchor
    else:
        highest_vertex = int(reachable_indices_cb[int(np.argmax(z_values[reachable_indices_cb]))])
    trunk_path = path_to_root(parent, highest_vertex)
    degrees = np.array([len(items) for items in adjacency], dtype=np.int32)
    crown_base_z = root_crown_z = float(z_values[root_vertex])
    for vid in trunk_path:
        if degrees[int(vid)] >= 3:
            crown_base_z = float(z_values[int(vid)])
            break
    crown_base_height = float(crown_base_z - z_values[root_vertex])
    crown_height = float(np.quantile(z_values, 0.98) - crown_base_z)
    crown_mask = z_values >= crown_base_z
    if int(crown_mask.sum()) < 8:
        crown_mask = np.ones(len(vertices), dtype=bool)

    # Crown width: widest horizontal diameter via PCA.
    # Project crown vertices onto XY, find the two principal axes, measure
    # max-min along each, average them (forestry standard: two perpendicular directions).
    crown_xy = vertices[crown_mask, :2]
    if len(crown_xy) >= 3:
        centroid = crown_xy.mean(axis=0)
        _, _, vh = np.linalg.svd(crown_xy - centroid, full_matrices=False)
        proj_principal = (crown_xy - centroid) @ vh[0]
        proj_secondary = (crown_xy - centroid) @ vh[1]
        width_principal = float(proj_principal.max() - proj_principal.min())
        width_secondary = float(proj_secondary.max() - proj_secondary.min())
        crown_width = float((width_principal + width_secondary) / 2.0)
    else:
        # Degenerate: fall back to simple XY spread.
        width_principal = float(
            np.quantile(vertices[crown_mask, 0], 0.95) - np.quantile(vertices[crown_mask, 0], 0.05)
        )
        width_secondary = float(
            np.quantile(vertices[crown_mask, 1], 0.95) - np.quantile(vertices[crown_mask, 1], 0.05)
        )
        crown_width = float(max(width_principal, width_secondary))

    unstable_attribute_names = []

    branch_records, branch_adjacency, degrees = build_branch_graph(adjacency, lines, trunk_anchor)
    num_branches = int(len(branch_records))

    # Use a crown-only height and the two crown spans to approximate volume.
    density_denom = crown_height * width_principal * width_secondary
    if density_denom <= EPS:
        branch_density = 0.0
        unstable_attribute_names.append("branch_density")
    else:
        branch_density = float(num_branches / density_denom)

    branch_order_depth = compute_branch_depth(branch_adjacency, trunk_anchor)

    attributes = {
        "height": height,
        "crown_width": crown_width,
        "branch_density": branch_density,
        "branch_order_depth": float(branch_order_depth),
    }

    missing_attribute_names = [
        name for name, value in attributes.items() if not math.isfinite(float(value))
    ]
    if missing_attribute_names:
        unstable_attribute_names.extend(missing_attribute_names)

    summary = {
        "root_vertex": int(root_vertex),
        "trunk_anchor_vertex": int(trunk_anchor),
        "highest_vertex": int(highest_vertex),
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "raw_num_vertices": weld_stats["raw_num_vertices"],
        "raw_num_edges": weld_stats["raw_num_edges"],
        "welded_num_vertices": weld_stats["welded_num_vertices"],
        "welded_num_edges": weld_stats["welded_num_edges"],
        "welded_vertices_removed": weld_stats["welded_vertices_removed"],
        "degenerate_edges_removed": weld_stats["degenerate_edges_removed"],
        "duplicate_edges_removed": weld_stats["duplicate_edges_removed"],
        "num_branches": num_branches,
        "num_leaf_vertices": int((degrees == 1).sum()),
        "num_fork_vertices": int((degrees >= 3).sum()),
        "max_degree": int(degrees.max()) if len(degrees) > 0 else 0,
        **component_stats,
    }

    quality_flags = {
        "is_connected": quality["num_components"] == 1,
        "has_cycle": quality["cycle_rank"] > 0,
        "num_components": quality["num_components"],
        "cycle_rank": quality["cycle_rank"],
        "num_isolated_vertices": quality["num_isolated_vertices"],
        "raw_num_vertices": weld_stats["raw_num_vertices"],
        "welded_num_vertices": weld_stats["welded_num_vertices"],
        "welded_vertices_removed": weld_stats["welded_vertices_removed"],
        "degenerate_edges_removed": weld_stats["degenerate_edges_removed"],
        "duplicate_edges_removed": weld_stats["duplicate_edges_removed"],
        "missing_attribute_count": len(missing_attribute_names),
        "unstable_attribute_names": sorted(set(unstable_attribute_names)),
    }

    return attributes, quality_flags, summary


def component_labels(num_vertices: int, lines: np.ndarray):
    labels = np.full(num_vertices, -1, dtype=np.int32)
    adjacency = [[] for _ in range(num_vertices)]
    for a, b in lines:
        adjacency[a].append(b)
        adjacency[b].append(a)

    component_id = 0
    for start in range(num_vertices):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = component_id
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if labels[neighbor] == -1:
                    labels[neighbor] = component_id
                    stack.append(neighbor)
        component_id += 1

    return labels, component_id
