from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from .graph import component_labels, weld_skeleton_vertices


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def edge_lengths(vertices: np.ndarray, lines: np.ndarray) -> np.ndarray:
    if not len(lines):
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(vertices[lines[:, 1]] - vertices[lines[:, 0]], axis=1)


def edge_statistics(lengths: np.ndarray) -> dict:
    if not len(lengths):
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "total": 0.0,
        }
    return {
        "count": int(len(lengths)),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95": float(np.quantile(lengths, 0.95)),
        "p99": float(np.quantile(lengths, 0.99)),
        "max": float(lengths.max()),
        "total": float(lengths.sum()),
    }


def adjacency_for(vertex_count: int, lines: np.ndarray) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for edge_index, (first, second) in enumerate(lines):
        first = int(first)
        second = int(second)
        adjacency[first].append((second, edge_index))
        adjacency[second].append((first, edge_index))
    return adjacency


def validate_tree(vertices: np.ndarray, lines: np.ndarray, *, context: str) -> dict:
    if len(vertices) == 0:
        raise ValueError(f"{context}: empty vertex array")
    labels, component_count = component_labels(len(vertices), lines)
    cycle_rank = int(len(lines) - len(vertices) + component_count)
    if component_count != 1 or cycle_rank != 0 or len(lines) != len(vertices) - 1:
        raise RuntimeError(
            f"{context} is not a strict tree: V={len(vertices)} "
            f"E={len(lines)} C={component_count} cycle_rank={cycle_rank}"
        )
    return {
        "vertex_count": int(len(vertices)),
        "edge_count": int(len(lines)),
        "component_count": int(component_count),
        "cycle_rank": cycle_rank,
        "strict_tree": True,
        "label_count": int(len(np.unique(labels))),
    }


def spatial_topology(
    vertices: np.ndarray,
    lines: np.ndarray,
    weld_tolerance: float,
) -> dict:
    welded_vertices, welded_lines, weld = weld_skeleton_vertices(
        vertices, lines, snap_tolerance=weld_tolerance
    )
    labels, component_count = component_labels(len(welded_vertices), welded_lines)
    cycle_rank = int(len(welded_lines) - len(welded_vertices) + component_count)
    return {
        "welded_vertex_count": int(len(welded_vertices)),
        "welded_edge_count": int(len(welded_lines)),
        "component_count": int(component_count),
        "cycle_rank": cycle_rank,
        "strict_tree": bool(
            component_count == 1
            and cycle_rank == 0
            and len(welded_lines) == len(welded_vertices) - 1
        ),
        "label_count": int(len(np.unique(labels))),
        "weld": weld,
    }
