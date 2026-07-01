"""Single-variable variants of the rectangle-connectivity stage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable


ALGORITHMS_DIR = Path(__file__).resolve().parents[1]
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import build_graph as bg


SUPPORTED_CONNECTIVITY_STRATEGIES = (
    "none",
    "candidate_bridge",
)


@dataclass(frozen=True)
class ConnectivityResult:
    """Rectangles and diagnostics produced by one connectivity strategy."""

    rectangles: list[bg.Rect]
    bridge_rectangles: list[bg.Rect]
    stats: dict


def _coordinates(rect) -> tuple[float, float, float, float]:
    return (float(rect.x1), float(rect.y1), float(rect.x2), float(rect.y2))


def rectangle_set_sha256(rectangles: Iterable[bg.Rect]) -> str:
    """Hash a rectangle multiset independently of its in-memory object type."""
    coordinates = sorted(_coordinates(rect) for rect in rectangles)
    payload = json.dumps(
        coordinates,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def connected_components(
    rectangles: list[bg.Rect],
    min_edge_length: float,
) -> list[list[int]]:
    """Return the connected components of the rectangle-adjacency graph."""
    adjacency = bg.getAdjacencyGraph(rectangles, min_edge_length)
    visited = [False] * len(rectangles)
    components: list[list[int]] = []
    for start in range(len(rectangles)):
        if visited[start]:
            continue
        component: list[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor, is_adjacent in enumerate(adjacency[node]):
                if is_adjacent and not visited[neighbor]:
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def apply_connectivity_strategy(
    *,
    strategy: str,
    selected_rectangles: list[bg.Rect],
    connectivity_candidates: list[bg.Rect],
    min_edge_length: float,
    grid_map,
    pixel_size: float,
    grid_size: int,
) -> ConnectivityResult:
    """Apply bridge repair or a controlled no-op to identical rectangles."""
    if strategy not in SUPPORTED_CONNECTIVITY_STRATEGIES:
        choices = ", ".join(SUPPORTED_CONNECTIVITY_STRATEGIES)
        raise ValueError(f"未知连通性策略 {strategy!r}；可选值：{choices}")

    before = list(selected_rectangles)
    components_before = connected_components(before, min_edge_length)
    if strategy == "candidate_bridge":
        after = bg.repairConnectivity(
            before,
            connectivity_candidates,
            min_edge_length,
            grid_map=grid_map,
            pixel_size=pixel_size,
            grid_size=grid_size,
        )
    else:
        after = list(before)

    before_coordinates = [_coordinates(rect) for rect in before]
    after_prefix = [_coordinates(rect) for rect in after[:len(before)]]
    if after_prefix != before_coordinates:
        raise ValueError("连通性策略不得修改或重排补全前矩形。")

    bridge_rectangles = list(after[len(before):])
    components_after = connected_components(after, min_edge_length)
    bridge_area = sum(
        max(0.0, rect.x2 - rect.x1) * max(0.0, rect.y2 - rect.y1)
        for rect in bridge_rectangles
    )
    return ConnectivityResult(
        rectangles=list(after),
        bridge_rectangles=bridge_rectangles,
        stats={
            "connectivity_strategy": strategy,
            "pre_repair_rectangles_sha256": rectangle_set_sha256(before),
            "component_count_before": len(components_before),
            "component_count_after": len(components_after),
            "component_sizes_before": [len(component) for component in components_before],
            "component_sizes_after": [len(component) for component in components_after],
            "fully_connected_before": bool(before) and len(components_before) == 1,
            "fully_connected_after": bool(after) and len(components_after) == 1,
            "connectivity_bridge_count": len(bridge_rectangles),
            "connectivity_bridge_area_m2": bridge_area,
            "bridge_rectangles": [
                {
                    "x1": rect.x1,
                    "y1": rect.y1,
                    "x2": rect.x2,
                    "y2": rect.y2,
                }
                for rect in bridge_rectangles
            ],
        },
    )
