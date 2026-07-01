"""Run the controlled vehicle-speed coverage experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
ROBOT_CONFIG_ROOT = CONFIG_ROOT / "robots"
RESULT_ROOT = EXPERIMENT_ROOT / "results"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from plot_results import plot_comparison  # noqa: E402
from src.configuration import PipelineConfig, RobotConfig, load_experiment_config  # noqa: E402
from src.main_pipeline import (  # noqa: E402
    build_traversal_order,
    run_experiment,
    segment_map_into_rectangles,
)


def _speed_id(speed: float) -> str:
    return f"speed_{speed:.1f}".replace(".", "p")


def available_maps() -> list[str]:
    return sorted(
        path.stem.removeprefix("experiment_")
        for path in CONFIG_ROOT.glob("experiment_*.json")
    )


def available_robot_configs() -> list[tuple[float, Path, RobotConfig]]:
    variants = []
    for path in ROBOT_CONFIG_ROOT.glob("robot_speed_*.json"):
        with path.open("r", encoding="utf-8") as handle:
            robot = RobotConfig.from_dict(json.load(handle))
        variants.append((robot.speed_limit, path.resolve(), robot))
    return sorted(variants, key=lambda item: item[0])


def _variant_config(
    base: PipelineConfig,
    speed: float,
    robot_path: Path,
    robot: RobotConfig,
    traversal_order: tuple[int, ...] | None = None,
) -> PipelineConfig:
    output = (RESULT_ROOT / base.experiment_name / _speed_id(speed)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return replace(
        base,
        robot=robot,
        robot_config_path=str(robot_path),
        output_directory=str(output),
        output_video_path=str(output / Path(base.output_video_path).name),
        segmentation_preview_path=str(output / Path(base.segmentation_preview_path).name),
        traversal_preview_path=str(output / Path(base.traversal_preview_path).name),
        coverage_image_path=str(output / Path(base.coverage_image_path).name),
        metrics_path=str(output / Path(base.metrics_path).name),
        config_snapshot_path=str(output / Path(base.config_snapshot_path).name),
        traversal_order=traversal_order,
    )


def _robot_control_signature(robot: RobotConfig) -> dict:
    payload = asdict(robot)
    payload.pop("speed_limit")
    return payload


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)


def _write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def _summary_row(report: dict) -> dict:
    trajectory = report["trajectory"]
    coverage = report["coverage"]
    speed = trajectory["configured_speed_limit_mps"]
    return {
        "speed_id": _speed_id(speed),
        "configured_speed_mps": speed,
        "actual_average_work_speed_mps": trajectory["actual_average_work_speed_mps"],
        "actual_max_work_speed_mps": trajectory["actual_max_work_speed_mps"],
        "arm_angular_velocity_deg_s": trajectory["arm_angular_velocity_limit_deg_s"],
        "work_speed_policy": trajectory["work_speed_policy"],
        "work_speed_limited": trajectory["work_speed_limited"],
        "work_disc_coverage_ratio": coverage["work_disc"]["coverage_ratio"],
        "covered_free_area_m2": coverage["covered_free_area_m2"],
        "duration_seconds": trajectory["duration_seconds"],
        "work_duration_seconds": trajectory["work_duration_seconds"],
        "non_work_duration_seconds": trajectory["non_work_duration_seconds"],
        "execution_seconds_per_covered_m2": coverage["execution_seconds_per_covered_m2"],
        "result_directory": str(
            Path(report["artifacts"]["metrics_json"]["path"]).parent
        ),
        "rectangle_order_sha256": _stable_hash(
            {"rectangles": report["rectangles"], "order": report["traversal"]["order"]}
        ),
    }


def _fixed_order(
    base: PipelineConfig,
    robot_variant: tuple[float, Path, RobotConfig],
) -> tuple[int, ...]:
    speed, path, robot = robot_variant
    reference = _variant_config(base, speed, path, robot)
    _, _, rectangles, stats = segment_map_into_rectangles(reference)
    _, order, _ = build_traversal_order(
        rectangles,
        reference,
        fixed_start_rect=stats["start_rectangle_before_connectivity"],
    )
    return tuple(order)


def _run_map(map_name: str, variants: list[tuple[float, Path, RobotConfig]]) -> dict:
    base = load_experiment_config(CONFIG_ROOT / f"experiment_{map_name}.json")
    if base.work_speed_policy != "commanded":
        raise ValueError("车速实验必须配置 motion.work_speed_policy='commanded'。")
    signatures = {_stable_hash(_robot_control_signature(robot)) for _, _, robot in variants}
    if len(signatures) != 1:
        raise ValueError("机器人配置除 speed_limit 外存在差异，无法进行单变量比较。")

    fixed_order = _fixed_order(base, variants[0])
    reports = []
    for speed, robot_path, robot in variants:
        config = _variant_config(base, speed, robot_path, robot, fixed_order)
        reports.append(run_experiment(config))

    rows = [_summary_row(report) for report in reports]
    order_hashes = {row["rectangle_order_sha256"] for row in rows}
    if len(order_hashes) != 1:
        raise ValueError("不同速度组的矩形集合或遍历顺序不一致。")
    if any(row["work_speed_limited"] for row in rows):
        raise ValueError("commanded 模式下存在未达到配置速度的作业段。")

    map_result_root = RESULT_ROOT / map_name
    csv_path = map_result_root / "comparison_summary.csv"
    _write_csv(rows, csv_path)
    plot_paths = plot_comparison(csv_path, map_result_root)
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "comparison": "vehicle_speed_coverage_duration",
        "experiment_name": map_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "controlled_variables": {
            "map": base.snapshot()["map_path"],
            "random_seed": base.random_seed,
            "arm_angular_velocity_deg_s": variants[0][2].arm_angular_velocity_limit,
            "work_speed_policy": base.work_speed_policy,
            "fixed_traversal_order": list(fixed_order),
            "rectangle_order_sha256": next(iter(order_hashes)),
            "note": "Only robot.speed_limit and output paths differ.",
        },
        "results": rows,
        "artifacts": {
            "summary_csv": str(csv_path.relative_to(REPOSITORY_ROOT)),
            "coverage_duration_curve": str(plot_paths[0].relative_to(REPOSITORY_ROOT)),
            "speed_effect_curve": str(plot_paths[1].relative_to(REPOSITORY_ROOT)),
        },
    }
    _write_json(summary, map_result_root / "comparison_summary.json")
    return summary


def run_comparison(
    map_names: list[str] | None = None,
    speeds: list[float] | None = None,
) -> dict:
    variants = available_robot_configs()
    if speeds:
        requested = set(speeds)
        variants = [variant for variant in variants if variant[0] in requested]
        missing = requested - {variant[0] for variant in variants}
        if missing:
            raise ValueError(f"缺少车速配置：{sorted(missing)}")
    if not variants:
        raise ValueError("没有可用的车速机器人配置。")
    maps = map_names or available_maps()
    summaries = [_run_map(map_name, variants) for map_name in maps]
    return {
        "schema_version": "1.0",
        "status": "completed",
        "comparison": "vehicle_speed_coverage_duration",
        "maps": summaries,
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare coverage and duration at fixed arm speed and varying vehicle speeds."
    )
    parser.add_argument("--map", action="append", choices=available_maps(), dest="map_names")
    parser.add_argument("--speed", action="append", type=float, dest="speeds")
    args = parser.parse_args(argv)
    try:
        payload = run_comparison(args.map_names, args.speeds)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
