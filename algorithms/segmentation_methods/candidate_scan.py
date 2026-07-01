"""Controlled variants of row/column candidate generation.

Only the candidate scan strategy changes between variants.  Greedy selection,
random gap filling, connectivity repair, traversal, coverage and rendering stay
in the shared pipeline so the ablation remains a one-variable comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import segmentation as seg
except ModuleNotFoundError:  # Package import used by test discovery.
    from algorithms import segmentation as seg


SUPPORTED_CANDIDATE_SCAN_STRATEGIES = (
    "row_only",
    "col_only",
    "row_then_col",
)


@dataclass
class CandidateSelection:
    """Greedy rectangles and metadata produced by one scan strategy."""

    selected_rectangles: list[seg.Rect]
    connectivity_candidates: list[seg.Rect]
    stats: dict


def _row_candidates(
    grid_map: np.ndarray,
    min_edge_length: float,
    pixel_size: float,
    grid_size: int,
    candidate_top_k: int | None,
) -> list[seg.Rect]:
    return seg.generateCandidates(
        grid_map,
        min_edge_length,
        pixel_size,
        grid_size,
        candidate_top_k,
    )


def _col_candidates(
    grid_map: np.ndarray,
    min_edge_length: float,
    pixel_size: float,
    grid_size: int,
    candidate_top_k: int | None,
) -> list[seg.Rect]:
    return seg.generateColCandidates(
        grid_map,
        min_edge_length,
        pixel_size,
        grid_size,
        candidate_top_k,
    )


def select_candidates(
    strategy: str,
    grid_map: np.ndarray,
    raw_map: np.ndarray,
    min_edge_length: float,
    pixel_size: float,
    grid_size: int,
    candidate_top_k: int,
    alpha: float,
    beta: float,
) -> CandidateSelection:
    """Generate and greedily select candidates for one controlled strategy."""
    if strategy not in SUPPORTED_CANDIDATE_SCAN_STRATEGIES:
        choices = ", ".join(SUPPORTED_CANDIDATE_SCAN_STRATEGIES)
        raise ValueError(f"未知候选扫描策略 {strategy!r}；可选值：{choices}")

    row_candidates: list[seg.Rect] = []
    col_candidates: list[seg.Rect] = []
    row_connectivity_candidates: list[seg.Rect] = []
    col_connectivity_candidates: list[seg.Rect] = []
    row_selected: list[seg.Rect] = []
    col_selected: list[seg.Rect] = []

    if strategy in {"row_only", "row_then_col"}:
        row_candidates = _row_candidates(
            grid_map,
            min_edge_length,
            pixel_size,
            grid_size,
            candidate_top_k,
        )
        row_connectivity_candidates = _row_candidates(
            grid_map,
            min_edge_length,
            pixel_size,
            grid_size,
            None,
        )
        row_selected = seg.greedySelection(
            row_candidates,
            raw_map,
            pixel_size,
            alpha,
            beta,
        )

    if strategy == "col_only":
        col_candidates = _col_candidates(
            grid_map,
            min_edge_length,
            pixel_size,
            grid_size,
            candidate_top_k,
        )
        col_connectivity_candidates = _col_candidates(
            grid_map,
            min_edge_length,
            pixel_size,
            grid_size,
            None,
        )
        col_selected = seg.greedySelection(
            col_candidates,
            raw_map,
            pixel_size,
            alpha,
            beta,
        )

    if strategy == "row_then_col":
        # The second phase sees only free space not already covered by row rectangles.
        masked_grid = seg.maskRects(grid_map, row_selected, pixel_size, grid_size)
        col_candidates = _col_candidates(
            masked_grid,
            min_edge_length,
            pixel_size,
            grid_size,
            candidate_top_k,
        )
        initial_cover_count = seg.buildCoverCount(row_selected, raw_map.shape, pixel_size)
        col_selected = seg.greedySelection(
            col_candidates,
            raw_map,
            pixel_size,
            alpha,
            beta,
            initial_cover_count=initial_cover_count,
        )

    selected = row_selected + col_selected

    # Connectivity repair is allowed to use only candidate orientations belonging
    # to the strategy.  For the two-stage method, col candidates are regenerated on
    # the original grid so a bridge may cross a row-selected region, matching the
    # original pipeline behaviour.
    if strategy == "row_only":
        connectivity_candidates = row_connectivity_candidates
    elif strategy == "col_only":
        connectivity_candidates = col_connectivity_candidates
    else:
        unmasked_col_candidates = _col_candidates(
            grid_map,
            min_edge_length,
            pixel_size,
            grid_size,
            None,
        )
        connectivity_candidates = seg.deduplicateCandidates(
            row_connectivity_candidates + unmasked_col_candidates
        )

    return CandidateSelection(
        selected_rectangles=selected,
        connectivity_candidates=connectivity_candidates,
        stats={
            "candidate_scan_strategy": strategy,
            "candidate_top_k_per_scanline": candidate_top_k,
            "scan_stage_order": {
                "row_only": ["row"],
                "col_only": ["col"],
                "row_then_col": ["row", "col"],
            }[strategy],
            "row_candidate_count": len(row_candidates),
            "row_selected_count": len(row_selected),
            "column_candidate_count": len(col_candidates),
            "column_selected_count": len(col_selected),
            "candidate_count_total": len(row_candidates) + len(col_candidates),
            "greedy_rectangle_count": len(selected),
            "connectivity_candidate_count": len(connectivity_candidates),
        },
    )
