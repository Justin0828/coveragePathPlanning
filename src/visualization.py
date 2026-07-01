"""Research visualizations derived from pipeline outputs."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import textwrap
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle as PlotRectangle
import numpy as np
from PIL import Image

from src.algorithm_api import bg, rc


def disc_center(pose: dict, robot: rc.Robot) -> tuple[float, float]:
    heading = math.radians(pose["heading"])
    pivot_x = pose["car_position"][0] - robot.pivot_to_car_center * math.cos(heading)
    pivot_y = pose["car_position"][1] - robot.pivot_to_car_center * math.sin(heading)
    arm_heading = math.radians(pose["heading"] + 180.0 + pose["arm_angle"])
    return (
        pivot_x + robot.arm_length * math.cos(arm_heading),
        pivot_y + robot.arm_length * math.sin(arm_heading),
    )


def update_coverage_mask(
    covered: np.ndarray,
    pose: dict,
    robot: rc.Robot,
    pixel_size: float,
) -> None:
    disc_x, disc_y = disc_center(pose, robot)
    center_x, center_y = rc.coordinateWorldToMap((disc_x, disc_y), pixel_size)
    radius = max(1, int(robot.disc_radius / pixel_size))
    height, width = covered.shape
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    covered[y0:y1, x0:x1] |= (xs - center_x) ** 2 + (ys - center_y) ** 2 <= radius ** 2


def build_coverage_mask(
    poses: Iterable[dict],
    map_shape: tuple[int, int],
    robot: rc.Robot,
    pixel_size: float,
) -> np.ndarray:
    covered = np.zeros(map_shape, dtype=bool)
    for pose in poses:
        update_coverage_mask(covered, pose, robot, pixel_size)
    return covered


def _base_rgb(segmentation_map: np.ndarray) -> np.ndarray:
    return np.repeat(segmentation_map[:, :, None], 3, axis=2).astype(np.uint8)


def render_coverage_image(
    segmentation_map: np.ndarray,
    raw_map: np.ndarray,
    covered: np.ndarray,
    coverage_color: tuple[int, int, int],
) -> np.ndarray:
    frame = _base_rgb(segmentation_map)
    frame[covered & (raw_map == 255)] = coverage_color
    return frame


def _robot_mask(shape: tuple[int, int], pose: dict, robot: rc.Robot, pixel_size: float) -> np.ndarray:
    canvas = np.full(shape, 255, dtype=np.uint8)
    current = copy.deepcopy(robot)
    current.heading = pose["heading"]
    current.arm_angle = pose["arm_angle"]
    current.car_speed = pose["car_speed"]
    current.car_angular_velocity = pose["car_angular_velocity"]
    current.car_position = pose["car_position"]
    rc._drawRobot(canvas, current, pixel_size)
    return canvas != 255


def _frame_sequence(
    raw_map: np.ndarray,
    segmentation_map: np.ndarray,
    poses: list[dict],
    robot: rc.Robot,
    pixel_size: float,
    coverage_color: tuple[int, int, int],
    trail_color: tuple[int, int, int],
    robot_color: tuple[int, int, int],
):
    covered = np.zeros(raw_map.shape, dtype=bool)
    trail = np.zeros(raw_map.shape, dtype=np.uint8)
    previous: tuple[float, float] | None = None
    for pose in poses:
        update_coverage_mask(covered, pose, robot, pixel_size)
        if previous is not None:
            rc._drawLine(trail, previous, pose["car_position"], pixel_size, color=1)
        previous = pose["car_position"]
        frame = render_coverage_image(segmentation_map, raw_map, covered, coverage_color)
        frame[trail.astype(bool)] = trail_color
        frame[_robot_mask(raw_map.shape, pose, robot, pixel_size)] = robot_color
        # Array row 0 is world y=0; image/video row 0 is displayed at the top.
        yield np.flipud(frame)


def _pad_frame_to_even_dimensions(frame: np.ndarray) -> np.ndarray:
    """Pad RGB video frames for H.264's yuv420p even-dimension requirement.

    The planning raster may legitimately have an odd width or height (for
    example, a 65.4 m field sampled at 0.5 m/px is 131 pixels high).  Padding
    only affects the MP4 canvas; the map, metrics, and still images retain
    their original dimensions.
    """

    height, width = frame.shape[:2]
    pad_height, pad_width = height % 2, width % 2
    if not pad_height and not pad_width:
        return frame
    padding = [(0, pad_height), (0, pad_width)] + [(0, 0)] * (frame.ndim - 2)
    return np.pad(frame, padding, mode="constant", constant_values=0)


def save_coverage_video(
    raw_map: np.ndarray,
    segmentation_map: np.ndarray,
    poses: list[dict],
    robot: rc.Robot,
    pixel_size: float,
    sample_rate: float,
    output_path: str | Path,
    coverage_color: tuple[int, int, int],
    trail_color: tuple[int, int, int],
    robot_color: tuple[int, int, int],
) -> None:
    if not poses:
        raise ValueError("没有可写入视频的位姿。")
    import imageio.v2 as imageio

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _frame_sequence(
        raw_map, segmentation_map, poses, robot, pixel_size,
        coverage_color, trail_color, robot_color,
    )
    if path.suffix.lower() == ".gif":
        imageio.mimsave(path, list(frames), duration=1.0 / sample_rate)
        return
    with imageio.get_writer(path, fps=sample_rate, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(_pad_frame_to_even_dimensions(frame))


def save_coverage_png(image: np.ndarray, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(image), mode="RGB").save(path)


def save_traversal_order(
    segmentation_map: np.ndarray,
    rect_list: list[bg.Rect],
    order: list[int],
    pixel_size: float,
    output_path: str | Path,
) -> None:
    """Draw every ordered transition as an arrow over the segmentation map."""
    width = segmentation_map.shape[1] * pixel_size
    height = segmentation_map.shape[0] * pixel_size
    figure_width = max(7.0, min(14.0, width / max(height, 1e-9) * 9.0))
    fig, ax = plt.subplots(figsize=(figure_width, 9.0), constrained_layout=True)
    ax.imshow(
        segmentation_map,
        cmap="gray",
        origin="lower",
        extent=(0, width, 0, height),
        vmin=0,
        vmax=255,
    )

    centers = [((rect.x1 + rect.x2) / 2.0, (rect.y1 + rect.y2) / 2.0) for rect in rect_list]
    rectangle_colors = [plt.get_cmap("tab20")(idx % 20) for idx in range(len(rect_list))]
    for rect_idx, rect in enumerate(rect_list):
        ax.add_patch(
            PlotRectangle(
                (rect.x1, rect.y1),
                rect.x2 - rect.x1,
                rect.y2 - rect.y1,
                fill=False,
                linewidth=1.6,
                edgecolor=rectangle_colors[rect_idx],
                alpha=0.95,
                zorder=2,
            )
        )

    for step, (source_idx, target_idx) in enumerate(zip(order, order[1:]), start=1):
        source, target = centers[source_idx], centers[target_idx]
        arrow = FancyArrowPatch(
            source,
            target,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color="#ff5a36",
            alpha=0.88,
            shrinkA=7,
            shrinkB=7,
            connectionstyle=f"arc3,rad={0.08 if step % 2 else -0.08}",
        )
        ax.add_patch(arrow)
        midpoint = ((source[0] + target[0]) / 2.0, (source[1] + target[1]) / 2.0)
        ax.annotate(
            str(step),
            midpoint,
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.18", "fc": "#c8371d", "ec": "none", "alpha": 0.9},
        )

    first_visit: dict[int, int] = {}
    for step, rect_idx in enumerate(order, start=1):
        first_visit.setdefault(rect_idx, step)
    placed_labels: list[tuple[float, float]] = []
    collision_distance = max(width, height) * 0.035

    def contains(rect: bg.Rect, point: tuple[float, float]) -> bool:
        return rect.x1 <= point[0] <= rect.x2 and rect.y1 <= point[1] <= rect.y2

    def choose_label_position(rect_idx: int) -> tuple[float, float]:
        """Choose a readable position that is guaranteed to stay in its rectangle."""
        rect = rect_list[rect_idx]
        center = centers[rect_idx]
        fractions = (0.2, 0.35, 0.5, 0.65, 0.8)
        candidates = [
            (
                rect.x1 + (rect.x2 - rect.x1) * x_fraction,
                rect.y1 + (rect.y2 - rect.y1) * y_fraction,
            )
            for y_fraction in fractions
            for x_fraction in fractions
        ]

        def score(point: tuple[float, float]) -> tuple[float, float, float]:
            # Prefer a part of this rectangle that is not hidden inside another
            # selected rectangle.  Then avoid existing labels, while retaining
            # the center as the natural fallback.
            overlap_count = sum(
                contains(other, point)
                for other_idx, other in enumerate(rect_list)
                if other_idx != rect_idx
            )
            nearest_label = min(
                (math.dist(point, previous) for previous in placed_labels),
                default=collision_distance,
            )
            separation = min(nearest_label, collision_distance)
            return (-float(overlap_count), separation, -math.dist(point, center))

        position = max(candidates, key=score)
        # All candidates are convex combinations of the rectangle bounds; this
        # assertion prevents future placement changes from reintroducing the bug.
        if not contains(rect, position):
            raise AssertionError(f"矩形 R{rect_idx} 的标签位置落在矩形外：{position}")
        return position

    for rect_idx, visit_step in first_visit.items():
        center = centers[rect_idx]
        label_position = choose_label_position(rect_idx)
        placed_labels.append(label_position)
        ax.annotate(
            f"R{rect_idx}\n#{visit_step}",
            center,
            xytext=label_position,
            textcoords="data",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#102a43",
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": rectangle_colors[rect_idx],
                "linewidth": 1.8,
                "alpha": 0.9,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": rectangle_colors[rect_idx],
                "linewidth": 1.2,
                "alpha": 0.8,
            } if math.dist(center, label_position) > 1e-9 else None,
            zorder=5,
        )

    order_caption = " → ".join(f"R{rect_idx}" for rect_idx in order)
    wrapped_order = "\n".join(textwrap.wrap(order_caption, width=55, break_long_words=False))
    ax.set_title(
        "Rectangle traversal order (arrow labels are transition steps)\n" + wrapped_order,
        fontsize=12,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
