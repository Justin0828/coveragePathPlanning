"""Validated JSON configuration for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments/fields2cover_comparison/configs/experiment_map_test1.json"
)
SUPPORTED_CANDIDATE_SCAN_STRATEGIES = {"row_only", "col_only", "row_then_col"}
SUPPORTED_CONNECTIVITY_STRATEGIES = {"none", "candidate_bridge"}
SUPPORTED_WORK_SPEED_POLICIES = {"coverage_safe", "commanded"}


def _resolve_path(value: str, *, base: Path = REPOSITORY_ROOT) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _require_keys(data: dict[str, Any], keys: set[str], section: str) -> None:
    missing = sorted(keys - data.keys())
    if missing:
        raise ValueError(f"配置段 {section!r} 缺少字段：{', '.join(missing)}")


@dataclass(frozen=True)
class RobotConfig:
    disc_radius: float
    arm_length: float
    car_width: float
    car_half_length: float
    pivot_to_car_center: float
    speed_limit: float
    angular_velocity_limit: float
    arm_angle_limit: tuple[float, float]
    arm_angular_velocity_limit: float
    initial_heading: float = 0.0
    initial_arm_angle: float = 0.0
    initial_car_position: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotConfig":
        required = {
            "disc_radius", "arm_length", "car_width", "car_half_length",
            "pivot_to_car_center", "speed_limit", "angular_velocity_limit",
            "arm_angle_limit", "arm_angular_velocity_limit",
        }
        _require_keys(data, required, "robot")
        values = dict(data)
        values["arm_angle_limit"] = tuple(values["arm_angle_limit"])
        values["initial_car_position"] = tuple(values.get("initial_car_position", (0.0, 0.0)))
        config = cls(**values)
        positive = {
            "disc_radius": config.disc_radius,
            "arm_length": config.arm_length,
            "car_width": config.car_width,
            "car_half_length": config.car_half_length,
            "speed_limit": config.speed_limit,
            "angular_velocity_limit": config.angular_velocity_limit,
            "arm_angular_velocity_limit": config.arm_angular_velocity_limit,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"机器人配置必须为正数：{', '.join(invalid)}")
        if len(config.arm_angle_limit) != 2:
            raise ValueError("robot.arm_angle_limit 必须恰好包含两个角度。")
        if config.arm_angle_limit[0] > config.arm_angle_limit[1]:
            raise ValueError("robot.arm_angle_limit 下限不能大于上限。")
        if config.arm_angle_limit[0] > 0 or config.arm_angle_limit[1] < 0:
            raise ValueError("robot.arm_angle_limit 必须包含 0 度。")
        return config


@dataclass(frozen=True)
class PipelineConfig:
    experiment_name: str
    config_path: str
    robot_config_path: str
    map_path: str
    output_directory: str
    output_video_path: str
    segmentation_preview_path: str
    traversal_preview_path: str
    coverage_image_path: str
    metrics_path: str
    config_snapshot_path: str
    pixel_size: float
    grid_size: int
    min_edge_length: float
    alpha: float
    beta: float
    candidate_top_k: int
    candidate_scan_strategy: str
    connectivity_strategy: str
    work_speed_policy: str
    start_rect: int | None
    traversal_order: tuple[int, ...] | None
    sample_rate: float
    coverage_color: tuple[int, int, int]
    trail_color: tuple[int, int, int]
    robot_color: tuple[int, int, int]
    random_seed: int
    robot: RobotConfig

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "config_path", "robot_config_path", "map_path", "output_directory",
            "output_video_path", "segmentation_preview_path", "traversal_preview_path",
            "coverage_image_path", "metrics_path", "config_snapshot_path",
        ):
            path = Path(payload[key])
            try:
                payload[key] = str(path.relative_to(REPOSITORY_ROOT))
            except ValueError:
                payload[key] = str(path)
        return payload


def load_experiment_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    _require_keys(
        document,
        {"experiment_name", "map", "segmentation", "traversal", "rendering", "outputs", "robot_config"},
        "root",
    )
    map_config = document["map"]
    segmentation = document["segmentation"]
    connectivity = document.get("connectivity", {})
    motion = document.get("motion", {})
    traversal = document["traversal"]
    rendering = document["rendering"]
    outputs = document["outputs"]

    robot_config_path = Path(_resolve_path(document["robot_config"]))
    with robot_config_path.open("r", encoding="utf-8") as handle:
        robot = RobotConfig.from_dict(json.load(handle))

    output_directory = Path(_resolve_path(outputs["directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)

    def output(name: str) -> str:
        return str((output_directory / outputs[name]).resolve())

    colors = rendering.get("colors", {})
    config = PipelineConfig(
        experiment_name=str(document["experiment_name"]),
        config_path=str(config_path),
        robot_config_path=str(robot_config_path),
        map_path=_resolve_path(map_config["path"]),
        output_directory=str(output_directory),
        output_video_path=output("coverage_video"),
        segmentation_preview_path=output("segmentation_image"),
        traversal_preview_path=output("traversal_image"),
        coverage_image_path=output("coverage_image"),
        metrics_path=output("metrics_json"),
        config_snapshot_path=output("config_snapshot"),
        pixel_size=float(map_config["pixel_size"]),
        grid_size=int(segmentation["grid_size"]),
        min_edge_length=float(segmentation["min_edge_length"]),
        alpha=float(segmentation["alpha"]),
        beta=float(segmentation["beta"]),
        candidate_top_k=int(segmentation["candidate_top_k"]),
        candidate_scan_strategy=str(segmentation.get("candidate_scan_strategy", "row_then_col")),
        connectivity_strategy=str(connectivity.get("strategy", "candidate_bridge")),
        work_speed_policy=str(motion.get("work_speed_policy", "coverage_safe")),
        start_rect=traversal.get("start_rect"),
        traversal_order=(
            tuple(int(index) for index in traversal["order"])
            if traversal.get("order") is not None
            else None
        ),
        sample_rate=float(rendering["sample_rate"]),
        coverage_color=tuple(colors.get("coverage", (30, 144, 255))),
        trail_color=tuple(colors.get("trail", (255, 196, 0))),
        robot_color=tuple(colors.get("robot", (220, 40, 40))),
        random_seed=int(document.get("random_seed", 42)),
        robot=robot,
    )
    if (
        config.pixel_size <= 0
        or config.grid_size <= 0
        or config.sample_rate <= 0
        or config.candidate_top_k <= 0
    ):
        raise ValueError(
            "pixel_size、grid_size、sample_rate 和 candidate_top_k 必须为正数。"
        )
    if config.candidate_scan_strategy not in SUPPORTED_CANDIDATE_SCAN_STRATEGIES:
        choices = ", ".join(sorted(SUPPORTED_CANDIDATE_SCAN_STRATEGIES))
        raise ValueError(
            "segmentation.candidate_scan_strategy "
            f"必须是以下值之一：{choices}"
        )
    if config.connectivity_strategy not in SUPPORTED_CONNECTIVITY_STRATEGIES:
        choices = ", ".join(sorted(SUPPORTED_CONNECTIVITY_STRATEGIES))
        raise ValueError(
            f"connectivity.strategy 必须是以下值之一：{choices}"
        )
    if config.work_speed_policy not in SUPPORTED_WORK_SPEED_POLICIES:
        choices = ", ".join(sorted(SUPPORTED_WORK_SPEED_POLICIES))
        raise ValueError(
            f"motion.work_speed_policy 必须是以下值之一：{choices}"
        )
    for name, color in {
        "coverage": config.coverage_color,
        "trail": config.trail_color,
        "robot": config.robot_color,
    }.items():
        if len(color) != 3 or any(not 0 <= channel <= 255 for channel in color):
            raise ValueError(f"rendering.colors.{name} 必须是三个 0..255 整数。")
    return config
