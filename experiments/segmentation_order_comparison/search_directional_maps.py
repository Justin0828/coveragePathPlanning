"""Search reproducible directional maps for the candidate-scan ablation.

The search deliberately keeps every planning parameter fixed and varies only a
connected union of horizontal halls and vertical shafts.  It scores the actual
continuous swept-disc trajectory metrics, not a rectangle-only proxy.

The output is a JSON leaderboard.  Each retained candidate is also emitted as
a PNG under the requested output directory so it can be inspected and promoted
to an experiment map without rerunning the search.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.configuration import load_experiment_config
from src.main_pipeline import (
    _motion_duration_metrics,
    build_traversal_order,
    compute_coverage_metrics,
    create_robot,
    generate_pipeline_poses,
    segment_map_into_rectangles,
)


STRATEGIES = ("row_only", "col_only", "row_then_col")
HEIGHT = 110
WIDTH = 150


def _paint(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Paint a free axis-aligned rectangle after clipping to the canvas."""
    mask[max(0, y1):min(HEIGHT, y2), max(0, x1):min(WIDTH, x2)] = True


def make_directional_map(seed: int) -> tuple[np.ndarray, dict]:
    """Build a connected, asymmetric hall-and-shaft map from one seed.

    Horizontal halls are intentionally dominant, while shafts continue above
    and below them.  The staggered endpoints prevent either single direction
    from reducing the complete free space to one family of large rectangles.
    """
    rng = random.Random(seed)
    free = np.zeros((HEIGHT, WIDTH), dtype=bool)
    descriptors: list[dict[str, int | str]] = []

    # A broad central hall gives the row stage its high-value first rectangles.
    hall_height = rng.choice((10, 12, 14, 16, 18))
    hall_y = rng.randint(42, 60)
    hall_x1 = rng.randint(5, 18)
    hall_x2 = rng.randint(128, 145)
    _paint(free, hall_x1, hall_y, hall_x2, hall_y + hall_height)
    descriptors.append({"kind": "central_hall", "x1": hall_x1, "y1": hall_y, "x2": hall_x2, "y2": hall_y + hall_height})

    # Two offset side halls make column-only candidates pay for interrupted
    # horizontal reach, but remain connected to the central hall through shafts.
    side_halls: dict[str, tuple[int, int, int, int]] = {}
    for side, y_low in (("lower", rng.randint(12, 30)), ("upper", rng.randint(76, 90))):
        height = rng.choice((7, 8, 10, 12))
        span = rng.randint(52, 92)
        if rng.random() < 0.5:
            x1 = rng.randint(4, 20)
            x2 = min(WIDTH - 4, x1 + span)
        else:
            x2 = rng.randint(130, 146)
            x1 = max(4, x2 - span)
        _paint(free, x1, y_low, x2, y_low + height)
        side_halls[side] = (x1, y_low, x2, y_low + height)
        descriptors.append({"kind": f"{side}_hall", "x1": x1, "y1": y_low, "x2": x2, "y2": y_low + height})

    # One shaft is reserved for each side hall, guaranteeing a single connected
    # free-space component.  Other shafts cross the central hall and create the
    # directional residuals required by the second scan stage.
    shafts: list[tuple[int, int, int, int]] = []
    for side, (side_x1, side_y1, side_x2, side_y2) in side_halls.items():
        width = rng.choice((6, 7, 8, 10))
        x = rng.randint(side_x1, max(side_x1, side_x2 - width))
        y1 = side_y1 if side == "lower" else hall_y
        y2 = hall_y + hall_height if side == "lower" else side_y2
        _paint(free, x, y1, x + width, y2)
        shafts.append((x, y1, x + width, y2))
        descriptors.append({"kind": "connector_shaft", "x1": x, "y1": y1, "x2": x + width, "y2": y2})

    # Shafts intersect the central hall and selectively extend away from it.
    # Their width is safely above the 2.5 m minimum edge threshold (5 pixels).
    shaft_count = rng.randint(4, 7)
    shaft_xs = sorted(rng.sample(range(12, 138), shaft_count))
    for index, x in enumerate(shaft_xs):
        width = rng.choice((6, 7, 8, 10, 12))
        reaches_low = rng.random() < 0.70
        reaches_high = rng.random() < 0.70
        y1 = rng.randint(1, max(2, hall_y - 18)) if reaches_low else hall_y
        y2 = rng.randint(hall_y + hall_height + 18, HEIGHT - 1) if reaches_high else hall_y + hall_height
        _paint(free, x, y1, x + width, y2)
        shafts.append((x, y1, x + width, y2))
        descriptors.append({"kind": "shaft", "x1": x, "y1": y1, "x2": x + width, "y2": y2})

    # A few short horizontal teeth create residual, directionally mixed pockets.
    for index in range(rng.randint(2, 5)):
        shaft_x1, shaft_y1, shaft_x2, shaft_y2 = rng.choice(shafts)
        x = shaft_x1
        length = rng.randint(12, 35)
        height = rng.choice((5, 6, 7))
        # Place every tooth inside its parent shaft's y span so it cannot create
        # an isolated island.  If the span is short, its only valid placement is
        # still connected at the hall/shaft junction.
        y = rng.randint(shaft_y1, max(shaft_y1, shaft_y2 - height))
        if rng.random() < 0.5:
            x1, x2 = x - length, x + 4
        else:
            x1, x2 = x + 2, x + length
        _paint(free, x1, y, x2, y + height)
        descriptors.append({"kind": "tooth", "x1": x1, "y1": y, "x2": x2, "y2": y + height})

    return free, {"seed": seed, "primitives": descriptors}


def _save_map(free: np.ndarray, path: Path) -> None:
    # Pipeline loadMap flips PNG rows so keep the visual image conventional here.
    Image.fromarray(np.flipud(np.where(free, 255, 0).astype(np.uint8))).save(path)


def _is_connected(free: np.ndarray) -> bool:
    """Require one 4-connected free-space component for a valid coverage map."""
    points = np.argwhere(free)
    if len(points) == 0:
        return False
    start_y, start_x = (int(value) for value in points[0])
    frontier = deque([(start_y, start_x)])
    visited = np.zeros_like(free, dtype=bool)
    visited[start_y, start_x] = True
    reached = 0
    while frontier:
        y, x = frontier.popleft()
        reached += 1
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < HEIGHT
                and 0 <= next_x < WIDTH
                and free[next_y, next_x]
                and not visited[next_y, next_x]
            ):
                visited[next_y, next_x] = True
                frontier.append((next_y, next_x))
    return reached == int(free.sum())


def _evaluate(map_path: Path, base_config, strategy: str, scratch_dir: Path) -> dict:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        base_config,
        experiment_name="search",
        map_path=str(map_path),
        candidate_scan_strategy=strategy,
        output_directory=str(scratch_dir),
        segmentation_preview_path=str(scratch_dir / "segmentation.png"),
        traversal_preview_path=str(scratch_dir / "traversal.png"),
        coverage_image_path=str(scratch_dir / "coverage.png"),
        output_video_path=str(scratch_dir / "coverage.mp4"),
        metrics_path=str(scratch_dir / "metrics.json"),
        config_snapshot_path=str(scratch_dir / "config_snapshot.json"),
    )
    raw_map, _preview, rectangles, segmentation = segment_map_into_rectangles(config)
    graph, order, _start = build_traversal_order(rectangles, config)
    segments = []
    poses, _robot_after = generate_pipeline_poses(
        rectangles,
        order,
        create_robot(config.robot, config.work_speed_policy),
        config.sample_rate,
        motion_segments=segments,
    )
    coverage = compute_coverage_metrics(
        poses,
        raw_map,
        create_robot(config.robot, config.work_speed_policy),
        config.pixel_size,
        motion_segments=segments,
    )
    duration = _motion_duration_metrics(segments)
    return {
        "full_trajectory_coverage": coverage["trajectory_coverage_ratio"],
        "work_coverage": coverage["work_coverage_ratio"],
        "duration_seconds": duration["duration_seconds"],
        "rectangle_count": segmentation["final_rectangle_count"],
        "transition_count": max(0, len(order) - 1),
        "segmentation_coverage": segmentation["final_partition"]["free_coverage_ratio"],
        "segmentation_overlap": segmentation["final_partition"]["free_overlap_ratio"],
        "adjacency_edges": sum(sum(row) for row in graph) // 2,
    }


def _score(results: dict[str, dict]) -> tuple[float, dict]:
    rtc = results["row_then_col"]
    singles = [results["row_only"], results["col_only"]]
    max_single_coverage = max(item["full_trajectory_coverage"] for item in singles)
    min_single_duration = min(item["duration_seconds"] for item in singles)
    coverage_advantage = rtc["full_trajectory_coverage"] - max_single_coverage
    duration_advantage = (min_single_duration - rtc["duration_seconds"]) / min_single_duration
    # Coverage has priority; the duration term breaks ties and rewards a true
    # Pareto improvement.  A strong penalty rejects empty/degenerate maps.
    score = 1000 * coverage_advantage + 100 * duration_advantage
    if rtc["segmentation_coverage"] < 0.90:
        score -= 100
    return score, {
        "coverage_advantage_vs_best_single": coverage_advantage,
        "duration_advantage_vs_fastest_single": duration_advantage,
        "pareto_win": coverage_advantage > 0 and duration_advantage > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--keep", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "map_search",
    )
    args = parser.parse_args()
    if args.trials <= 0 or args.keep <= 0:
        raise ValueError("--trials and --keep must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_experiment_config(
        Path(__file__).resolve().parent / "configs/row_then_col/experiment_scan_mixed_weave.json"
    )
    leaderboard = []
    with tempfile.TemporaryDirectory(prefix="directional-map-search-") as temporary:
        map_path = Path(temporary) / "candidate.png"
        for trial in range(args.trials):
            seed = args.seed + trial
            free, design = make_directional_map(seed)
            free_ratio = float(free.mean())
            if not 0.18 <= free_ratio <= 0.58 or not _is_connected(free):
                continue
            _save_map(free, map_path)
            try:
                results = {
                    strategy: _evaluate(map_path, base_config, strategy, Path(temporary) / strategy)
                    for strategy in STRATEGIES
                }
            except (ValueError, RuntimeError):
                continue
            score, comparison = _score(results)
            leaderboard.append({
                "score": score,
                "free_ratio": free_ratio,
                "design": design,
                "results": results,
                "comparison": comparison,
                "map": free,
            })

    leaderboard.sort(key=lambda item: item["score"], reverse=True)
    retained = leaderboard[:args.keep]
    report_candidates = []
    for rank, candidate in enumerate(retained, start=1):
        image_name = f"rank_{rank:02d}_seed_{candidate['design']['seed']}.png"
        _save_map(candidate.pop("map"), args.output_dir / image_name)
        report_candidates.append({
            "rank": rank,
            "map_file": image_name,
            **candidate,
        })
    report = {
        "search_seed": args.seed,
        "trials_requested": args.trials,
        "candidates_evaluated": len(leaderboard),
        "ranking_rule": "1000 * full-trajectory-coverage advantage + 100 * duration advantage",
        "candidates": report_candidates,
    }
    with (args.output_dir / "leaderboard.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
