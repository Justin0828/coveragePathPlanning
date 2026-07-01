from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import copy
import math
import os
import random
import sys
import time

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithm_api import bg, conn_methods, rc, seg, seg_methods
from src.configuration import DEFAULT_CONFIG_PATH, PipelineConfig, REPOSITORY_ROOT, RobotConfig, load_experiment_config
from src.visualization import (
    build_coverage_mask,
    render_coverage_image,
    save_coverage_png,
    save_coverage_video,
    save_traversal_order,
)


def create_robot(
    config: RobotConfig,
    work_speed_policy: str = "coverage_safe",
) -> rc.Robot:
    return rc.Robot(
        disc_radius=config.disc_radius,
        arm_length=config.arm_length,
        car_width=config.car_width,
        car_half_length=config.car_half_length,
        pivot_to_car_center=config.pivot_to_car_center,
        speed_limit=config.speed_limit,
        angular_velocity_limit=config.angular_velocity_limit,
        arm_angle_limit=config.arm_angle_limit,
        arm_angular_velocity_limit=config.arm_angular_velocity_limit,
        work_speed_policy=work_speed_policy,
        heading=config.initial_heading,
        arm_angle=config.initial_arm_angle,
        arm_angular_velocity=0.0,
        car_speed=0.0,
        car_angular_velocity=0.0,
        car_position=config.initial_car_position,
    )


def create_default_robot() -> rc.Robot:
    """Compatibility helper using the repository's default robot config."""
    return create_robot(load_experiment_config().robot)


def _rect_area(rect: bg.Rect) -> float:
    return max(0.0, rect.x2 - rect.x1) * max(0.0, rect.y2 - rect.y1)


def _center(rect: bg.Rect | rc.Rect) -> tuple[float, float]:
    return ((rect.x1 + rect.x2) / 2.0, (rect.y1 + rect.y2) / 2.0)


def _copy_robot(robot: rc.Robot) -> rc.Robot:
    return copy.deepcopy(robot)


def _to_bg_rect(rect: seg.Rect) -> bg.Rect:
    return bg.Rect(rect.x1, rect.y1, rect.x2, rect.y2)


def _to_rc_rect(rect: bg.Rect) -> rc.Rect:
    return rc.Rect(rect.x1, rect.y1, rect.x2, rect.y2)


def _rectangle_partition_metrics(
    rects: list[seg.Rect] | list[bg.Rect],
    raw_map: np.ndarray,
    pixel_size: float,
) -> dict:
    """Measure rectangle-union coverage independently of robot trajectory coverage."""
    free_mask = raw_map == 255
    total_free_px = int(np.sum(free_mask))
    cover_count = seg.buildCoverCount(rects, raw_map.shape, pixel_size)
    covered_free_px = int(np.sum((cover_count > 0) & free_mask))
    overlap_free_px = int(np.sum((cover_count > 1) & free_mask))
    covered_obstacle_px = int(np.sum((cover_count > 0) & ~free_mask))
    return {
        "total_free_px": total_free_px,
        "covered_free_px": covered_free_px,
        "uncovered_free_px": total_free_px - covered_free_px,
        "overlap_free_px": overlap_free_px,
        "covered_obstacle_px": covered_obstacle_px,
        "free_coverage_ratio": covered_free_px / total_free_px if total_free_px else 0.0,
        "free_overlap_ratio": overlap_free_px / total_free_px if total_free_px else 0.0,
    }


def _map_output_path(path_str: str) -> Path:
    path = Path(path_str)
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def _choose_start_rect(rect_list: list[bg.Rect], start_rect: int | None) -> int:
    if not rect_list:
        raise ValueError("分割结果为空，无法选择起始矩形。")
    if start_rect is not None:
        if not 0 <= start_rect < len(rect_list):
            raise ValueError(f"start_rect={start_rect} 超出范围 [0, {len(rect_list) - 1}]")
        return start_rect
    return max(range(len(rect_list)), key=lambda idx: _rect_area(rect_list[idx]))


def _boundary_point_towards(rect: bg.Rect, target: tuple[float, float]) -> tuple[float, float]:
    cx, cy = _center(rect)
    dx = target[0] - cx
    dy = target[1] - cy
    if abs(dx) >= abs(dy):
        x = rect.x2 if dx >= 0 else rect.x1
        y = min(max(cy, rect.y1), rect.y2)
        return (x, y)
    x = min(max(cx, rect.x1), rect.x2)
    y = rect.y2 if dy >= 0 else rect.y1
    return (x, y)


def _boundary_point_away_from(rect: bg.Rect, target: tuple[float, float]) -> tuple[float, float]:
    cx, cy = _center(rect)
    opposite = (2 * cx - target[0], 2 * cy - target[1])
    return _boundary_point_towards(rect, opposite)


def _default_entry_exit_points(rect: bg.Rect) -> tuple[tuple[float, float], tuple[float, float]]:
    width = rect.x2 - rect.x1
    height = rect.y2 - rect.y1
    if width >= height:
        return (rect.x1, (rect.y1 + rect.y2) / 2.0), (rect.x2, (rect.y1 + rect.y2) / 2.0)
    return ((rect.x1 + rect.x2) / 2.0, rect.y1), ((rect.x1 + rect.x2) / 2.0, rect.y2)


def _transition_points(
    rect_a: bg.Rect,
    rect_b: bg.Rect,
    ref_point: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute exit point of rect_a and entry point of rect_b at their shared boundary.
    When ref_point is provided, the free coordinate on the shared edge is clamped to the
    overlap range using ref_point's value instead of always using the midpoint. This
    minimises lateral travel within rect_a when the entry position is known.
    """
    eps = 1e-9
    overlap_y1 = max(rect_a.y1, rect_b.y1)
    overlap_y2 = min(rect_a.y2, rect_b.y2)
    if abs(rect_a.x2 - rect_b.x1) <= eps and overlap_y2 >= overlap_y1:
        y = (overlap_y1 + overlap_y2) / 2.0
        if ref_point is not None:
            y = max(overlap_y1, min(overlap_y2, ref_point[1]))
        p = (rect_a.x2, y)
        return p, p
    if abs(rect_b.x2 - rect_a.x1) <= eps and overlap_y2 >= overlap_y1:
        y = (overlap_y1 + overlap_y2) / 2.0
        if ref_point is not None:
            y = max(overlap_y1, min(overlap_y2, ref_point[1]))
        p = (rect_a.x1, y)
        return p, p

    overlap_x1 = max(rect_a.x1, rect_b.x1)
    overlap_x2 = min(rect_a.x2, rect_b.x2)
    if abs(rect_a.y2 - rect_b.y1) <= eps and overlap_x2 >= overlap_x1:
        x = (overlap_x1 + overlap_x2) / 2.0
        if ref_point is not None:
            x = max(overlap_x1, min(overlap_x2, ref_point[0]))
        p = (x, rect_a.y2)
        return p, p
    if abs(rect_b.y2 - rect_a.y1) <= eps and overlap_x2 >= overlap_x1:
        x = (overlap_x1 + overlap_x2) / 2.0
        if ref_point is not None:
            x = max(overlap_x1, min(overlap_x2, ref_point[0]))
        p = (x, rect_a.y1)
        return p, p

    center_a = _center(rect_a)
    center_b = _center(rect_b)
    return _boundary_point_towards(rect_a, center_b), _boundary_point_towards(rect_b, center_a)


def _build_rect_entry_exit_points(order: list[int], rect_list: list[bg.Rect]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if not order:
        return []

    points: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for idx, rect_idx in enumerate(order):
        rect = rect_list[rect_idx]

        if len(order) == 1:
            points.append(_default_entry_exit_points(rect))
            continue

        prev_idx = order[idx - 1] if idx > 0 else None
        next_idx = order[idx + 1] if idx < len(order) - 1 else None

        if prev_idx is None:
            out_point, _ = _transition_points(rect, rect_list[next_idx])
            in_point = _boundary_point_away_from(rect, _center(rect_list[next_idx]))
            points.append((in_point, out_point))
            continue

        in_point = _transition_points(rect_list[prev_idx], rect)[1]  # entry point on current rect's boundary

        if next_idx is None:
            out_point = _boundary_point_away_from(rect, _center(rect_list[prev_idx]))
            points.append((in_point, out_point))
            continue

        out_point, _ = _transition_points(rect, rect_list[next_idx], ref_point=in_point)
        if math.dist(in_point, out_point) <= 1e-9:
            out_point = _boundary_point_away_from(rect, _center(rect_list[prev_idx]))
        points.append((in_point, out_point))

    return points


def _pose_from_robot(robot: rc.Robot) -> dict:
    return {
        "heading": robot.heading,
        "arm_angle": robot.arm_angle,
        "car_speed": robot.car_speed,
        "car_angular_velocity": robot.car_angular_velocity,
        "car_position": robot.car_position,
    }


def _append_stationary_pose(poses: list[dict], robot: rc.Robot) -> None:
    poses.append(_pose_from_robot(robot))


def _append_turn_poses(
    poses: list[dict],
    robot: rc.Robot,
    target_heading: float,
    sample_rate: float,
    motion_segments: list[rc.MotionSegment] | None = None,
) -> None:
    current_heading = robot.heading % 360
    target_heading = target_heading % 360
    delta = (target_heading - current_heading + 540) % 360 - 180
    if abs(delta) <= 1e-9:
        robot.heading = target_heading
        return

    angular_speed = max(abs(robot.angular_velocity_limit), 1e-6)
    total_time = abs(delta) / angular_speed
    if motion_segments is not None:
        motion_segments.append(rc.MotionSegment(
            phase="vehicle_turn",
            coverage_active=False,
            duration=total_time,
            car_position_start=robot.car_position,
            car_position_end=robot.car_position,
            heading_start=current_heading,
            heading_end=current_heading + delta,
            arm_angle_start=robot.arm_angle,
            arm_angle_end=robot.arm_angle,
        ))
    steps = max(1, int(math.ceil(total_time * sample_rate)))
    signed_velocity = angular_speed if delta >= 0 else -angular_speed

    for step in range(1, steps + 1):
        ratio = step / steps
        poses.append(
            {
                "heading": (current_heading + delta * ratio) % 360,
                "arm_angle": robot.arm_angle,
                "car_speed": 0.0,
                "car_angular_velocity": signed_velocity,
                "car_position": robot.car_position,
            }
        )

    robot.heading = target_heading
    robot.car_speed = 0.0
    robot.car_angular_velocity = 0.0


def _append_move_poses(
    poses: list[dict],
    robot: rc.Robot,
    target_position: tuple[float, float],
    sample_rate: float,
    motion_segments: list[rc.MotionSegment] | None = None,
) -> None:
    start = robot.car_position
    distance = math.dist(start, target_position)
    if distance <= 1e-9:
        robot.car_position = target_position
        return

    speed = max(abs(robot.speed_limit), 1e-6)
    total_time = distance / speed
    if motion_segments is not None:
        motion_segments.append(rc.MotionSegment(
            phase="transition",
            coverage_active=False,
            duration=total_time,
            car_position_start=start,
            car_position_end=target_position,
            heading_start=robot.heading,
            heading_end=robot.heading,
            arm_angle_start=robot.arm_angle,
            arm_angle_end=robot.arm_angle,
        ))
    steps = max(1, int(math.ceil(total_time * sample_rate)))

    for step in range(1, steps + 1):
        ratio = step / steps
        x = start[0] + (target_position[0] - start[0]) * ratio
        y = start[1] + (target_position[1] - start[1]) * ratio
        poses.append(
            {
                "heading": robot.heading,
                "arm_angle": robot.arm_angle,
                "car_speed": speed,
                "car_angular_velocity": 0.0,
                "car_position": (x, y),
            }
        )

    robot.car_position = target_position
    robot.car_speed = 0.0
    robot.car_angular_velocity = 0.0


def _transition_robot_to_entry_pose(
    robot: rc.Robot,
    target_pose: rc.Robot,
    sample_rate: float,
    motion_segments: list[rc.MotionSegment] | None = None,
) -> list[dict]:
    poses: list[dict] = []
    if abs(robot.arm_angle) > 1e-9:
        poses.extend(rc._generateArmTurnPoses(
            robot,
            robot.car_position,
            robot.heading,
            robot.arm_angle,
            0.0,
            robot.arm_angular_velocity_limit,
            sample_rate,
            motion_segments,
            "arm_retract",
        ))
    if math.dist(robot.car_position, target_pose.car_position) > 1e-9:
        heading_to_target = rc.computeHeading(robot.car_position, target_pose.car_position)
        _append_turn_poses(poses, robot, heading_to_target, sample_rate, motion_segments)
        _append_move_poses(poses, robot, target_pose.car_position, sample_rate, motion_segments)
    _append_turn_poses(poses, robot, target_pose.heading, sample_rate, motion_segments)
    robot.arm_angle = 0.0
    robot.arm_angular_velocity = 0.0
    robot.car_speed = 0.0
    robot.car_angular_velocity = 0.0
    return poses


def segment_map_into_rectangles(
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, list[bg.Rect], dict]:
    map_path = _map_output_path(config.map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"未找到地图文件：{map_path}")

    raw_map = seg.loadMap(str(map_path))
    grid_map = seg.resterizeMap(raw_map, config.grid_size)
    candidate_selection = seg_methods.select_candidates(
        strategy=config.candidate_scan_strategy,
        grid_map=grid_map,
        raw_map=raw_map,
        min_edge_length=config.min_edge_length,
        pixel_size=config.pixel_size,
        grid_size=config.grid_size,
        candidate_top_k=config.candidate_top_k,
        alpha=config.alpha,
        beta=config.beta,
    )
    selected_rects = candidate_selection.selected_rectangles
    if not selected_rects:
        raise ValueError("未分割出任何可覆盖矩形，请检查 map.png 或调整分割参数。")
    greedy_partition = _rectangle_partition_metrics(
        selected_rects,
        raw_map,
        config.pixel_size,
    )

    # Random expansion session: fill uncovered free-space gaps
    selected_rects = seg.randomExpansionSession(
        grid_map,
        selected_rects,
        config.pixel_size,
        config.grid_size,
        config.min_edge_length,
        seed=config.random_seed,
    )
    expanded_total = len(selected_rects)
    expanded_partition = _rectangle_partition_metrics(
        selected_rects,
        raw_map,
        config.pixel_size,
    )

    all_candidates_bg = [_to_bg_rect(r) for r in candidate_selection.connectivity_candidates]
    selected_bg = [_to_bg_rect(r) for r in selected_rects]
    start_rect_before_connectivity = _choose_start_rect(
        selected_bg,
        config.start_rect,
    )
    connectivity_result = conn_methods.apply_connectivity_strategy(
        strategy=config.connectivity_strategy,
        selected_rectangles=selected_bg,
        connectivity_candidates=all_candidates_bg,
        min_edge_length=config.min_edge_length,
        grid_map=grid_map,
        pixel_size=config.pixel_size,
        grid_size=config.grid_size,
    )
    repaired_bg = connectivity_result.rectangles

    repaired_seg = [seg.Rect(r.x1, r.y1, r.x2, r.y2) for r in repaired_bg]
    preview = seg.drawSegmentation(raw_map.copy(), repaired_seg, config.pixel_size)
    seg.saveMap(preview, str(_map_output_path(config.segmentation_preview_path)))
    stats = {
        **candidate_selection.stats,
        "expanded_rectangle_count": expanded_total,
        "start_rectangle_before_connectivity": start_rect_before_connectivity,
        "start_rectangle_coordinates_before_connectivity": {
            "x1": selected_bg[start_rect_before_connectivity].x1,
            "y1": selected_bg[start_rect_before_connectivity].y1,
            "x2": selected_bg[start_rect_before_connectivity].x2,
            "y2": selected_bg[start_rect_before_connectivity].y2,
        },
        **connectivity_result.stats,
        "final_rectangle_count": len(repaired_bg),
        "greedy_partition": greedy_partition,
        "expanded_partition": expanded_partition,
        "final_partition": _rectangle_partition_metrics(
            repaired_bg,
            raw_map,
            config.pixel_size,
        ),
    }
    return raw_map, preview, repaired_bg, stats


def build_traversal_order(
    rect_list: list[bg.Rect],
    config: PipelineConfig,
    fixed_start_rect: int | None = None,
) -> tuple[list[list[int]], list[int], int]:
    adjacency_graph = bg.getAdjacencyGraph(rect_list, config.min_edge_length)
    start_rect = (
        _choose_start_rect(rect_list, fixed_start_rect)
        if fixed_start_rect is not None
        else _choose_start_rect(rect_list, config.start_rect)
    )
    if config.traversal_order is None:
        order = bg.findMinTimeTraversalOrder(
            rect_list,
            adjacency_graph,
            start_rect,
            config.robot,
        )
    else:
        order = list(config.traversal_order)
        if not order or order[0] != start_rect:
            raise ValueError("固定 traversal.order 必须从选定的起始矩形开始。")
        if any(index < 0 or index >= len(rect_list) for index in order):
            raise ValueError("固定 traversal.order 包含越界矩形编号。")
        if any(
            not adjacency_graph[current][following]
            for current, following in zip(order, order[1:])
        ):
            raise ValueError("固定 traversal.order 包含不相邻的矩形转移。")
        reachable = set(bg.findReachableFromStart(start_rect, adjacency_graph))
        if set(order) != reachable:
            raise ValueError("固定 traversal.order 必须覆盖起点可达的全部矩形。")
    if not order:
        raise ValueError("未能从建图模块得到有效的矩形遍历顺序。")
    return adjacency_graph, order, start_rect


def generate_pipeline_poses(
    rect_list: list[bg.Rect],
    order: list[int],
    robot: rc.Robot,
    sample_rate: float,
    motion_segments: list[rc.MotionSegment] | None = None,
) -> tuple[list[dict], rc.Robot]:
    if not rect_list or not order:
        return [], robot

    working_robot = _copy_robot(robot)
    all_poses: list[dict] = []
    visited_rects: set[int] = set()
    rect_points = _build_rect_entry_exit_points(order, rect_list)

    for seq_idx, rect_idx in enumerate(order):
        rect_bg = rect_list[rect_idx]
        rect_rc = _to_rc_rect(rect_bg)
        in_point, out_point = rect_points[seq_idx]

        entry_robot = rc.findRobotPoseByInPoint(rect_rc, in_point, _copy_robot(working_robot))

        if not all_poses:
            working_robot = _copy_robot(entry_robot)
            _append_stationary_pose(all_poses, working_robot)
        else:
            all_poses.extend(_transition_robot_to_entry_pose(
                working_robot,
                entry_robot,
                sample_rate,
                motion_segments,
            ))
            working_robot.heading = entry_robot.heading
            working_robot.arm_angle = entry_robot.arm_angle
            working_robot.car_position = entry_robot.car_position
            working_robot.car_speed = 0.0
            working_robot.car_angular_velocity = 0.0

        if rect_idx not in visited_rects:
            working_robot = rc.findRobotPoseByInPoint(rect_rc, in_point, working_robot)
            all_poses.extend(
                rc.boustrophedonCoverage(
                    rect_rc,
                    in_point,
                    out_point,
                    working_robot,
                    sample_rate=sample_rate,
                    motion_segments=motion_segments,
                )
            )
            visited_rects.add(rect_idx)
            continue

        exit_robot = rc.findRobotPoseByInPoint(rect_rc, out_point, _copy_robot(working_robot))
        all_poses.extend(_transition_robot_to_entry_pose(
            working_robot,
            exit_robot,
            sample_rate,
            motion_segments,
        ))
        working_robot.heading = exit_robot.heading
        working_robot.arm_angle = exit_robot.arm_angle
        working_robot.car_position = exit_robot.car_position
        working_robot.car_speed = 0.0
        working_robot.car_angular_velocity = 0.0

    return all_poses, working_robot


def _motion_disc_center(
    segment: rc.MotionSegment,
    ratio: float,
    robot: rc.Robot,
) -> tuple[float, float]:
    """Evaluate the exact parametric disc-center curve for one motion segment."""
    car_x = segment.car_position_start[0] + (
        segment.car_position_end[0] - segment.car_position_start[0]
    ) * ratio
    car_y = segment.car_position_start[1] + (
        segment.car_position_end[1] - segment.car_position_start[1]
    ) * ratio
    heading = segment.heading_start + (
        segment.heading_end - segment.heading_start
    ) * ratio
    if segment.arm_motion == "oscillating":
        arm_angle, _ = rc._oscillatingArmAngle(
            ratio * segment.duration,
            segment.arm_angle_start,
            segment.arm_angular_velocity,
            segment.arm_angle_lower,
            segment.arm_angle_upper,
        )
    else:
        arm_angle = segment.arm_angle_start + (
            segment.arm_angle_end - segment.arm_angle_start
        ) * ratio
    heading_rad = math.radians(heading)
    pivot_x = car_x - robot.pivot_to_car_center * math.cos(heading_rad)
    pivot_y = car_y - robot.pivot_to_car_center * math.sin(heading_rad)
    arm_heading = math.radians(heading + 180.0 + arm_angle)
    return (
        pivot_x + robot.arm_length * math.cos(arm_heading),
        pivot_y + robot.arm_length * math.sin(arm_heading),
    )


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return math.dist(point, start)
    ratio = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator
    ratio = max(0.0, min(1.0, ratio))
    projection = (start[0] + ratio * dx, start[1] + ratio * dy)
    return math.dist(point, projection)


def _flatten_disc_center_curve(
    segment: rc.MotionSegment,
    robot: rc.Robot,
    tolerance: float,
) -> list[tuple[float, float]]:
    """Approximate C(t) geometrically; the tolerance is independent of pose Hz."""
    start = _motion_disc_center(segment, 0.0, robot)
    end = _motion_disc_center(segment, 1.0, robot)
    points = [start]
    max_angle_step = 5.0

    def subdivide(
        ratio_start: float,
        point_start: tuple[float, float],
        ratio_end: float,
        point_end: tuple[float, float],
        depth: int,
    ) -> None:
        ratio_mid = (ratio_start + ratio_end) / 2.0
        point_mid = _motion_disc_center(segment, ratio_mid, robot)
        chord_error = _point_segment_distance(point_mid, point_start, point_end)
        fraction = ratio_end - ratio_start
        heading_change = abs(segment.heading_end - segment.heading_start) * fraction
        if segment.arm_motion == "oscillating":
            disc_arm_change = (
                abs(segment.heading_end - segment.heading_start) * fraction
                + abs(segment.arm_angular_velocity) * segment.duration * fraction
            )
        else:
            disc_arm_change = abs(
                (segment.heading_end + segment.arm_angle_end)
                - (segment.heading_start + segment.arm_angle_start)
            ) * fraction
        if (
            depth < 20
            and (
                chord_error > tolerance
                or heading_change > max_angle_step
                or disc_arm_change > max_angle_step
            )
        ):
            subdivide(
                ratio_start,
                point_start,
                ratio_mid,
                point_mid,
                depth + 1,
            )
            subdivide(
                ratio_mid,
                point_mid,
                ratio_end,
                point_end,
                depth + 1,
            )
            return
        points.append(point_end)

    subdivide(0.0, start, 1.0, end, 0)
    return points


def _rasterize_capsule(
    mask: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
    pixel_size: float,
) -> None:
    """Rasterize {x: distance(x, segment) <= radius} using pixel centers."""
    height, width = mask.shape
    min_x, max_x = min(start[0], end[0]) - radius, max(start[0], end[0]) + radius
    min_y, max_y = min(start[1], end[1]) - radius, max(start[1], end[1]) + radius
    col1 = max(0, int(math.floor(min_x / pixel_size)) - 1)
    col2 = min(width, int(math.ceil(max_x / pixel_size)) + 1)
    row1 = max(0, int(math.floor(min_y / pixel_size)) - 1)
    row2 = min(height, int(math.ceil(max_y / pixel_size)) + 1)
    if row1 >= row2 or col1 >= col2:
        return

    rows, cols = np.mgrid[row1:row2, col1:col2]
    xs = (cols + 0.5) * pixel_size
    ys = (rows + 0.5) * pixel_size
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        distance_squared = (xs - start[0]) ** 2 + (ys - start[1]) ** 2
    else:
        ratios = np.clip(
            ((xs - start[0]) * dx + (ys - start[1]) * dy) / denominator,
            0.0,
            1.0,
        )
        nearest_x = start[0] + ratios * dx
        nearest_y = start[1] + ratios * dy
        distance_squared = (xs - nearest_x) ** 2 + (ys - nearest_y) ** 2
    mask[row1:row2, col1:col2] |= distance_squared <= radius ** 2 + 1e-12


def build_parametric_coverage_masks(
    motion_segments: list[rc.MotionSegment],
    map_shape: tuple[int, int],
    robot: rc.Robot,
    pixel_size: float,
    geometry_tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return work and full-trajectory masks from parametric swept discs."""
    tolerance = geometry_tolerance or min(
        pixel_size / 8.0,
        robot.disc_radius / 20.0,
    )
    if tolerance <= 0:
        raise ValueError("geometry_tolerance 必须为正数。")
    work_mask = np.zeros(map_shape, dtype=bool)
    trajectory_mask = np.zeros(map_shape, dtype=bool)
    for segment in motion_segments:
        centers = _flatten_disc_center_curve(segment, robot, tolerance)
        pairs = list(zip(centers, centers[1:]))
        if not pairs:
            pairs = [(centers[0], centers[0])]
        for start, end in pairs:
            _rasterize_capsule(
                trajectory_mask,
                start,
                end,
                robot.disc_radius,
                pixel_size,
            )
            if segment.coverage_active:
                _rasterize_capsule(
                    work_mask,
                    start,
                    end,
                    robot.disc_radius,
                    pixel_size,
                )
    return work_mask, trajectory_mask, tolerance


def compute_coverage_metrics(
    poses: list[dict],
    raw_map: np.ndarray,
    robot: rc.Robot,
    pixel_size: float,
    motion_segments: list[rc.MotionSegment] | None = None,
    rect_list: list[bg.Rect] | None = None,
    order: list[int] | None = None,
) -> dict:
    """Compute physical swept-disc metrics independently of pose sample rate."""
    free_mask = raw_map == 255
    total_free_px = int(np.sum(free_mask))
    if motion_segments is not None:
        work_covered, trajectory_covered, tolerance = build_parametric_coverage_masks(
            motion_segments,
            raw_map.shape,
            robot,
            pixel_size,
        )
        source = "parametric_swept_disc"
    else:
        work_covered = build_coverage_mask(poses, raw_map.shape, robot, pixel_size)
        trajectory_covered = work_covered.copy()
        tolerance = None
        source = "sampled_disc_poses"
    work_covered_free_px = int(np.sum(work_covered & free_mask))
    trajectory_covered_free_px = int(np.sum(trajectory_covered & free_mask))
    return {
        "total_free_px": total_free_px,
        "covered_free_px": work_covered_free_px,
        "uncovered_free_px": total_free_px - work_covered_free_px,
        "coverage_ratio": work_covered_free_px / total_free_px if total_free_px else 0.0,
        "work_covered_free_px": work_covered_free_px,
        "work_uncovered_free_px": total_free_px - work_covered_free_px,
        "work_coverage_ratio": work_covered_free_px / total_free_px if total_free_px else 0.0,
        "trajectory_covered_free_px": trajectory_covered_free_px,
        "trajectory_uncovered_free_px": total_free_px - trajectory_covered_free_px,
        "trajectory_coverage_ratio": (
            trajectory_covered_free_px / total_free_px if total_free_px else 0.0
        ),
        "coverage_source": source,
        "geometry_tolerance_m": tolerance,
        "covered_map": work_covered,
        "work_covered_map": work_covered,
        "trajectory_covered_map": trajectory_covered,
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(payload: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)


def _artifact(path: str | Path) -> dict:
    target = Path(path)
    return {
        "path": _repository_path(target),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }


def _motion_duration_metrics(
    motion_segments: list[rc.MotionSegment],
) -> dict:
    """Summarize exact continuous durations rather than sampled pose counts."""
    phase_durations: dict[str, float] = {}
    work_duration = 0.0
    for segment in motion_segments:
        phase_durations[segment.phase] = (
            phase_durations.get(segment.phase, 0.0) + segment.duration
        )
        if segment.coverage_active:
            work_duration += segment.duration
    total_duration = sum(segment.duration for segment in motion_segments)
    work_distances = [
        math.dist(segment.car_position_start, segment.car_position_end)
        for segment in motion_segments
        if segment.coverage_active
    ]
    work_speeds = [
        distance / segment.duration
        for segment, distance in zip(
            (segment for segment in motion_segments if segment.coverage_active),
            work_distances,
        )
        if segment.duration > 0 and distance > 0
    ]
    work_distance = sum(work_distances)
    return {
        "duration_source": "continuous_motion_segments",
        "duration_seconds": total_duration,
        "work_duration_seconds": work_duration,
        "non_work_duration_seconds": total_duration - work_duration,
        "phase_duration_seconds": phase_durations,
        "work_distance_m": work_distance,
        "actual_average_work_speed_mps": (
            work_distance / work_duration if work_duration > 0 else 0.0
        ),
        "actual_max_work_speed_mps": max(work_speeds, default=0.0),
    }


def _traversal_reachability_metrics(
    rect_list: list[bg.Rect],
    adjacency_graph: list[list[int]],
    start_rect: int,
    raw_map: np.ndarray,
    pixel_size: float,
) -> dict:
    """Measure how much of the partition can be visited from one fixed start."""
    reachable_rectangles = bg.findReachableFromStart(start_rect, adjacency_graph)
    reachable_set = set(reachable_rectangles)
    unreachable_rectangles = [
        index for index in range(len(rect_list)) if index not in reachable_set
    ]
    return {
        "total_rectangle_count": len(rect_list),
        "reachable_rectangles": reachable_rectangles,
        "unreachable_rectangles": unreachable_rectangles,
        "reachable_rectangle_count": len(reachable_rectangles),
        "unreachable_rectangle_count": len(unreachable_rectangles),
        "reachable_rectangle_ratio": (
            len(reachable_rectangles) / len(rect_list) if rect_list else 0.0
        ),
        "all_rectangles_reachable": not unreachable_rectangles,
        "reachable_partition": _rectangle_partition_metrics(
            [rect_list[index] for index in reachable_rectangles],
            raw_map,
            pixel_size,
        ),
    }


def run_experiment(config: PipelineConfig) -> dict:
    """Run one configured experiment and persist all research artifacts."""
    started_at = datetime.now(timezone.utc)
    total_start = time.perf_counter()
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    robot = create_robot(config.robot, config.work_speed_policy)

    _write_json(config.snapshot(), config.config_snapshot_path)

    stage_start = time.perf_counter()
    raw_map, segmentation_map, rect_list, segmentation_stats = segment_map_into_rectangles(config)
    segmentation_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    adjacency_graph, order, start_rect = build_traversal_order(
        rect_list,
        config,
        fixed_start_rect=segmentation_stats["start_rectangle_before_connectivity"],
    )
    traversal_seconds = time.perf_counter() - stage_start
    reachability_metrics = _traversal_reachability_metrics(
        rect_list,
        adjacency_graph,
        start_rect,
        raw_map,
        config.pixel_size,
    )
    save_traversal_order(
        segmentation_map,
        rect_list,
        order,
        config.pixel_size,
        config.traversal_preview_path,
    )

    stage_start = time.perf_counter()
    motion_segments: list[rc.MotionSegment] = []
    poses, _ = generate_pipeline_poses(
        rect_list,
        order,
        robot,
        config.sample_rate,
        motion_segments=motion_segments,
    )
    pose_generation_seconds = time.perf_counter() - stage_start

    metrics = compute_coverage_metrics(
        poses,
        raw_map,
        robot,
        config.pixel_size,
        motion_segments=motion_segments,
        rect_list=rect_list,
        order=order,
    )
    duration_metrics = _motion_duration_metrics(motion_segments)
    covered_area_m2 = (
        metrics["work_covered_free_px"] * config.pixel_size ** 2
    )
    duration_seconds = duration_metrics["duration_seconds"]
    coverage_image = render_coverage_image(
        segmentation_map,
        raw_map,
        metrics["covered_map"],
        config.coverage_color,
    )
    save_coverage_png(coverage_image, config.coverage_image_path)

    stage_start = time.perf_counter()
    save_coverage_video(
        raw_map=raw_map,
        segmentation_map=segmentation_map,
        poses=poses,
        robot=_copy_robot(robot),
        pixel_size=config.pixel_size,
        sample_rate=config.sample_rate,
        output_path=config.output_video_path,
        coverage_color=config.coverage_color,
        trail_color=config.trail_color,
        robot_color=config.robot_color,
    )
    rendering_seconds = time.perf_counter() - stage_start

    finished_at = datetime.now(timezone.utc)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "experiment": {
            "name": config.experiment_name,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "config": _repository_path(config.config_path),
            "robot_config": _repository_path(config.robot_config_path),
            "random_seed": config.random_seed,
        },
        "input": {
            "map": _repository_path(config.map_path),
            "map_sha256": _sha256(config.map_path),
            "map_shape_px": {"height": int(raw_map.shape[0]), "width": int(raw_map.shape[1])},
            "pixel_size_m": config.pixel_size,
        },
        "segmentation": segmentation_stats,
        "traversal": {
            "start_rectangle": start_rect,
            "start_rectangle_coordinates": {
                "x1": rect_list[start_rect].x1,
                "y1": rect_list[start_rect].y1,
                "x2": rect_list[start_rect].x2,
                "y2": rect_list[start_rect].y2,
            },
            "order": order,
            "transition_count": max(0, len(order) - 1),
            "unique_rectangle_count": len(set(order)),
            "revisit_count": len(order) - len(set(order)),
            "adjacency_edge_count": sum(sum(row) for row in adjacency_graph) // 2,
            **reachability_metrics,
        },
        "trajectory": {
            "pose_count": len(poses),
            "sample_rate_hz": config.sample_rate,
            "sampled_pose_duration_seconds": max(0, len(poses) - 1) / config.sample_rate,
            **duration_metrics,
            "configured_speed_limit_mps": config.robot.speed_limit,
            "work_speed_policy": config.work_speed_policy,
            "arm_angular_velocity_limit_deg_s": config.robot.arm_angular_velocity_limit,
            "work_speed_limited": (
                duration_metrics["actual_max_work_speed_mps"]
                < config.robot.speed_limit - 1e-9
            ),
        },
        "coverage": {
            "source": metrics["coverage_source"],
            "partition_coverage_ratio": segmentation_stats["final_partition"]["free_coverage_ratio"],
            "total_free_px": metrics["total_free_px"],
            "covered_free_px": metrics["covered_free_px"],
            "uncovered_free_px": metrics["uncovered_free_px"],
            "coverage_ratio": metrics["coverage_ratio"],
            "covered_free_area_m2": covered_area_m2,
            "execution_seconds_per_covered_m2": (
                duration_seconds / covered_area_m2
                if covered_area_m2 > 0
                else None
            ),
            "covered_m2_per_execution_second": (
                covered_area_m2 / duration_seconds
                if duration_seconds > 0
                else None
            ),
            "work_disc": {
                "covered_free_px": metrics["work_covered_free_px"],
                "uncovered_free_px": metrics["work_uncovered_free_px"],
                "coverage_ratio": metrics["work_coverage_ratio"],
            },
            "full_trajectory_disc": {
                "covered_free_px": metrics["trajectory_covered_free_px"],
                "uncovered_free_px": metrics["trajectory_uncovered_free_px"],
                "coverage_ratio": metrics["trajectory_coverage_ratio"],
            },
            "geometry": {
                "method": metrics["coverage_source"],
                "disc_radius_m": robot.disc_radius,
                "arm_length_m": robot.arm_length,
                "pixel_criterion": "pixel_center",
                "geometry_tolerance_m": metrics["geometry_tolerance_m"],
                "sampling_rate_independent": metrics["coverage_source"] == "parametric_swept_disc",
                "motion_segment_count": len(motion_segments),
            },
        },
        "rectangles": [
            {"id": idx, "x1": rect.x1, "y1": rect.y1, "x2": rect.x2, "y2": rect.y2}
            for idx, rect in enumerate(rect_list)
        ],
        "runtime_seconds": {
            "segmentation": segmentation_seconds,
            "traversal": traversal_seconds,
            "pose_generation": pose_generation_seconds,
            "rendering": rendering_seconds,
            "total": time.perf_counter() - total_start,
        },
        "artifacts": {
            "segmentation_image": _artifact(config.segmentation_preview_path),
            "traversal_image": _artifact(config.traversal_preview_path),
            "coverage_video": _artifact(config.output_video_path),
            "coverage_image": _artifact(config.coverage_image_path),
            "config_snapshot": _artifact(config.config_snapshot_path),
            "metrics_json": {"path": _repository_path(config.metrics_path)},
        },
    }
    _write_json(report, config.metrics_path)
    return report


def main(config: PipelineConfig | None = None) -> dict:
    return run_experiment(config or load_experiment_config())


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible path-coverage experiment.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="实验 JSON 配置路径",
    )
    args = parser.parse_args(argv)
    try:
        report = run_experiment(load_experiment_config(args.config))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
