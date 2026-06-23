from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import build_graph as bg
import rectangle_coverage as rc
import segmentation as seg


@dataclass
class PipelineConfig:
    map_path: str = "map_test1.png"
    output_video_path: str = "pipeline.mp4"
    segmentation_preview_path: str = "segmentation_preview.png"

    pixel_size: float = 0.5
    grid_size: int = 1
    min_edge_length: float = 2.5
    alpha: float = 0.5
    beta: float = 2.0
    candidate_top_k: int = 10

    start_rect: int | None = None
    # sample_rate: float = 20.0
    sample_rate: float = 2.0


def create_default_robot() -> rc.Robot:
    return rc.Robot(
        # disc_radius=0.25,
        disc_radius=0.75,
        arm_length=1.2,
        car_width=0.8,
        car_half_length=1.0,
        pivot_to_car_center=0.3,
        # speed_limit=1.0,
        speed_limit=2.0,
        angular_velocity_limit=30.0,
        arm_angle_limit=(0, 90),
        arm_angular_velocity_limit=90.0,
        heading=0.0,
        arm_angle=0.0,
        arm_angular_velocity=0.0,
        car_speed=0.0,
        car_angular_velocity=0.0,
        car_position=(0.0, 0.0),
    )


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


def _map_output_path(path_str: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    return (base_dir / path_str).resolve()


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
) -> None:
    current_heading = robot.heading % 360
    target_heading = target_heading % 360
    delta = (target_heading - current_heading + 540) % 360 - 180
    if abs(delta) <= 1e-9:
        robot.heading = target_heading
        return

    angular_speed = max(abs(robot.angular_velocity_limit), 1e-6)
    total_time = abs(delta) / angular_speed
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
) -> None:
    start = robot.car_position
    distance = math.dist(start, target_position)
    if distance <= 1e-9:
        robot.car_position = target_position
        return

    speed = max(abs(robot.speed_limit), 1e-6)
    total_time = distance / speed
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
) -> list[dict]:
    poses: list[dict] = []
    if math.dist(robot.car_position, target_pose.car_position) > 1e-9:
        heading_to_target = rc.computeHeading(robot.car_position, target_pose.car_position)
        _append_turn_poses(poses, robot, heading_to_target, sample_rate)
        _append_move_poses(poses, robot, target_pose.car_position, sample_rate)
    _append_turn_poses(poses, robot, target_pose.heading, sample_rate)
    robot.arm_angle = target_pose.arm_angle
    robot.arm_angular_velocity = 0.0
    robot.car_speed = 0.0
    robot.car_angular_velocity = 0.0
    return poses


def segment_map_into_rectangles(config: PipelineConfig) -> tuple[np.ndarray, list[bg.Rect]]:
    map_path = _map_output_path(config.map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"未找到地图文件：{map_path}")

    raw_map = seg.loadMap(str(map_path))
    grid_map = seg.resterizeMap(raw_map, config.grid_size)
    # Phase 1: row-based candidates → greedy selection
    candidates_row = seg.generateCandidates(
        grid_map,
        config.min_edge_length,
        config.pixel_size,
        config.grid_size,
        config.candidate_top_k,
    )
    selected_row = seg.greedySelection(
        candidates_row,
        raw_map,
        config.pixel_size,
        config.alpha,
        config.beta,
    )

    # Phase 2: col-based candidates on remaining free space
    # Treat row-selected areas as obstacles so col rects fill the gaps
    masked_grid = seg.maskRects(grid_map, selected_row, config.pixel_size, config.grid_size)
    candidates_col = seg.generateColCandidates(
        masked_grid,
        config.min_edge_length,
        config.pixel_size,
        config.grid_size,
        config.candidate_top_k,
    )
    # Pre-fill cover_count so col greedy gain is relative to what row already covered
    init_cover = seg.buildCoverCount(selected_row, raw_map.shape, config.pixel_size)
    selected_col = seg.greedySelection(
        candidates_col,
        raw_map,
        config.pixel_size,
        config.alpha,
        config.beta,
        initial_cover_count=init_cover,
    )

    selected_rects = selected_row + selected_col
    if not selected_rects:
        raise ValueError("未分割出任何可覆盖矩形，请检查 map.png 或调整分割参数。")
    print(f"[segmentation] row={len(selected_row)} rects, col={len(selected_col)} rects, total={len(selected_rects)}")

    # Random expansion session: fill uncovered free-space gaps
    selected_rects = seg.randomExpansionSession(
        grid_map, selected_rects, config.pixel_size, config.grid_size, config.min_edge_length
    )
    print(f"[random expansion] total={len(selected_rects)} rects after expansion")

    # Connectivity repair: bridge pool must use UNMASKED col candidates so it can span
    # across row-selected regions and bridge disconnected components
    candidates_col_unmasked = seg.generateColCandidates(
        grid_map,
        config.min_edge_length,
        config.pixel_size,
        config.grid_size,
        config.candidate_top_k,
    )
    all_candidates_bg = [_to_bg_rect(r) for r in candidates_row + candidates_col_unmasked]
    selected_bg = [_to_bg_rect(r) for r in selected_rects]
    repaired_bg = bg.repairConnectivity(selected_bg, all_candidates_bg, config.min_edge_length)
    if len(repaired_bg) > len(selected_bg):
        print(f"[connectivity] inserted {len(repaired_bg) - len(selected_bg)} bridge rect(s)")

    preview = seg.drawSegmentation(raw_map.copy(), selected_rects, config.pixel_size)
    seg.saveMap(preview, str(_map_output_path(config.segmentation_preview_path)))
    return raw_map, repaired_bg


def build_traversal_order(rect_list: list[bg.Rect], config: PipelineConfig) -> tuple[list[list[int]], list[int], int]:
    adjacency_graph = bg.getAdjacencyGraph(rect_list, config.min_edge_length)
    start_rect = _choose_start_rect(rect_list, config.start_rect)
    order = bg.findMinTimeTraversalOrder(rect_list, adjacency_graph, start_rect)
    if not order:
        raise ValueError("未能从建图模块得到有效的矩形遍历顺序。")
    return adjacency_graph, order, start_rect


def generate_pipeline_poses(
    rect_list: list[bg.Rect],
    order: list[int],
    robot: rc.Robot,
    sample_rate: float,
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
            all_poses.extend(_transition_robot_to_entry_pose(working_robot, entry_robot, sample_rate))
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
                )
            )
            visited_rects.add(rect_idx)
            continue

        exit_robot = rc.findRobotPoseByInPoint(rect_rc, out_point, _copy_robot(working_robot))
        all_poses.extend(_transition_robot_to_entry_pose(working_robot, exit_robot, sample_rate))
        working_robot.heading = exit_robot.heading
        working_robot.arm_angle = exit_robot.arm_angle
        working_robot.car_position = exit_robot.car_position
        working_robot.car_speed = 0.0
        working_robot.car_angular_velocity = 0.0

    return all_poses, working_robot


def _render_frames(
    raw_map: np.ndarray,
    rect_list: list[bg.Rect],
    poses: list[dict],
    robot: rc.Robot,
    pixel_size: float,
) -> list[np.ndarray]:
    if not poses:
        return []

    base_map = raw_map.copy()
    for rect in rect_list:
        rc._drawRect(base_map, _to_rc_rect(rect), pixel_size, color=216)

    frames: list[np.ndarray] = []
    trail_map = base_map.copy()
    prev_position: tuple[float, float] | None = None
    for pose in poses:
        if prev_position is not None:
            rc._drawLine(trail_map, prev_position, pose["car_position"], pixel_size, color=96)
        prev_position = pose["car_position"]

        frame = trail_map.copy()
        robot.heading = pose["heading"]
        robot.arm_angle = pose["arm_angle"]
        robot.car_speed = pose["car_speed"]
        robot.car_angular_velocity = pose["car_angular_velocity"]
        robot.car_position = pose["car_position"]
        rc._drawRobot(frame, robot, pixel_size)
        frames.append(frame)

    return frames


def compute_coverage_metrics(
    poses: list[dict],
    raw_map: np.ndarray,
    robot: rc.Robot,
    pixel_size: float,
) -> dict:
    """Compute disc-swept coverage ratio over the free-space map.

    For each pose, reconstructs the disc center using the same geometry as
    _drawRobot, then rasterises the disc footprint onto a boolean coverage map.
    Returns total free pixels, covered free pixels, coverage ratio, and the map.
    """
    free_mask = raw_map == 255  # white pixels = free space
    total_free_px = int(np.sum(free_mask))
    if not poses or total_free_px == 0:
        return {
            "total_free_px": total_free_px,
            "covered_free_px": 0,
            "coverage_ratio": 0.0,
            "covered_map": np.zeros_like(raw_map, dtype=bool),
        }

    covered = np.zeros_like(raw_map, dtype=bool)
    H, W = raw_map.shape
    disc_radius_px = max(1, int(robot.disc_radius / pixel_size))

    for pose in poses:
        heading_rad = math.radians(pose["heading"])
        cx, cy = pose["car_position"]
        pivot_x = cx - robot.pivot_to_car_center * math.cos(heading_rad)
        pivot_y = cy - robot.pivot_to_car_center * math.sin(heading_rad)
        arm_heading_rad = math.radians(pose["heading"] + 180 + pose["arm_angle"])
        disc_x = pivot_x + robot.arm_length * math.cos(arm_heading_rad)
        disc_y = pivot_y + robot.arm_length * math.sin(arm_heading_rad)

        cx_px, cy_px = rc.coordinateWorldToMap((disc_x, disc_y), pixel_size)
        r = disc_radius_px
        y0 = max(0, cy_px - r)
        y1 = min(H, cy_px + r + 1)
        x0 = max(0, cx_px - r)
        x1 = min(W, cx_px + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        disc_mask = (xs - cx_px) ** 2 + (ys - cy_px) ** 2 <= r ** 2
        covered[y0:y1, x0:x1] |= disc_mask

    covered_free_px = int(np.sum(covered & free_mask))
    return {
        "total_free_px": total_free_px,
        "covered_free_px": covered_free_px,
        "coverage_ratio": covered_free_px / total_free_px,
        "covered_map": covered,
    }


def _save_video(frames: list[np.ndarray], output_path: Path, sample_rate: float) -> None:
    if not frames:
        raise ValueError("没有可写入视频的帧。")

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("导出视频需要安装 imageio。") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [np.repeat(frame[:, :, None], 3, axis=2) for frame in frames]
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        imageio.mimsave(output_path, rgb_frames, duration=1.0 / sample_rate)
        return

    try:
        with imageio.get_writer(output_path, fps=sample_rate, macro_block_size=1) as writer:
            for frame in rgb_frames:
                writer.append_data(frame)
    except Exception:
        fallback_path = output_path.with_suffix(".gif")
        imageio.mimsave(fallback_path, rgb_frames, duration=1.0 / sample_rate)
        raise RuntimeError(f"视频写入失败，已回退输出为 GIF：{fallback_path}")


def main(config: PipelineConfig | None = None) -> tuple[list[bg.Rect], list[int], list[dict]]:
    config = config or PipelineConfig()
    robot = create_default_robot()

    raw_map, rect_list = segment_map_into_rectangles(config)
    _, order, start_rect = build_traversal_order(rect_list, config)
    poses, _ = generate_pipeline_poses(rect_list, order, robot, config.sample_rate)

    metrics = compute_coverage_metrics(poses, raw_map, robot, config.pixel_size)

    frames = _render_frames(raw_map, rect_list, poses, _copy_robot(robot), config.pixel_size)
    _save_video(frames, _map_output_path(config.output_video_path), config.sample_rate)

    print(f"分割得到 {len(rect_list)} 个矩形")
    print(f"起始矩形索引: {start_rect}")
    print(f"矩形遍历顺序: {order}")
    print(f"总位姿数: {len(poses)}")
    print(f"分割预览图: {_map_output_path(config.segmentation_preview_path)}")
    print(f"输出视频: {_map_output_path(config.output_video_path)}")
    print(f"覆盖率: {metrics['coverage_ratio']:.1%}  ({metrics['covered_free_px']}/{metrics['total_free_px']} px)")

    return rect_list, order, poses


if __name__ == "__main__":
    main()
