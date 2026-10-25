#!/usr/bin/env python3

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import fields2cover as f2c

from path_validation import validate_path
from robot_config import load_robot_config
from trajectory_coverage import evaluate_full_trajectory_coverage


EXPERIMENTS_DIR = Path(__file__).resolve().parent
MAPS_DIR = EXPERIMENTS_DIR / "maps"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
PLANNERS_PATH = EXPERIMENTS_DIR / "configs" / "planners.json"


def load_planner_config(profile: str) -> dict:
    with PLANNERS_PATH.open(encoding="utf-8") as config_file:
        document = json.load(config_file)
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported planner configuration schema_version")
    try:
        return document["planners"][profile]
    except KeyError as error:
        available = ", ".join(sorted(document.get("planners", {})))
        raise KeyError(
            f"Unknown planner profile '{profile}'. Available profiles: {available}"
        ) from error


def create_options(config: dict):
    options = f2c.Options()
    options.hg_alg = {
        "NONE": f2c.HGAlg_NONE,
        "CONST": f2c.HGAlg_CONST,
    }[config["headland_algorithm"]]
    options.hg_swaths = int(config["headland_swaths"])
    options.sg_alg = {
        "BRUTE_FORCE": f2c.SGAlg_BRUTE_FORCE,
        "GIVEN_ANGLE": f2c.SGAlg_GIVEN_ANGLE,
    }[config["swath_algorithm"]]
    options.sg_obj = {
        "N_SWATH_MOD": f2c.SGObjFunc_N_SWATH_MOD,
        "N_SWATH": f2c.SGObjFunc_N_SWATH,
        "SWATH_LENGTH": f2c.SGObjFunc_SWATH_LENGTH,
        "FIELD_COV": f2c.SGObjFunc_FIELD_COV,
        "OVERLAPS": f2c.SGObjFunc_OVERLAPS,
    }[config["swath_objective"]]
    options.sg_angle = math.radians(float(config["swath_angle_deg"]))
    options.rp_alg = {
        "SHORTEST_ROUTE": f2c.RPAlg_SHORTEST_ROUTE,
    }[config["route_algorithm"]]
    options.pp_alg = {
        "DUBINS": f2c.PPAlg_DUBINS,
    }[config["path_algorithm"]]
    return options


def resolve_map_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() and path.parent == Path("."):
        path = MAPS_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Map does not exist: {path}")
    return path.resolve()


def plot_result(field, result, output_path: Path) -> None:
    # Gnuplot 6 emits one compatibility warning per path state through
    # Matplot++. Suppress that child-process stderr while retaining the image.
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            f2c.Visualizer.figure()
            f2c.Visualizer.plot(field)
            f2c.Visualizer.plot(result)
            f2c.Visualizer.save(str(output_path))
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if output_path.exists() and output_path.stat().st_size > 0:
            return
        time.sleep(0.05)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Plot was not created: {output_path}")


def flatten_route_swaths(route):
    swaths = f2c.Swaths()
    for index in range(route.sizeVectorSwaths()):
        swaths.append(route.getSwaths(index))
    return swaths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one coverage-only Fields2Cover experiment"
    )
    parser.add_argument("--robot", required=True, help="Robot profile name")
    parser.add_argument("--map", required=True, help="GeoJSON filename or path")
    parser.add_argument(
        "--planner", default="f2c_full_field", help="Planner profile name"
    )
    parser.add_argument("--run-name", required=True, help="Result directory name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_path = resolve_map_path(args.map)
    output_dir = RESULTS_DIR / args.run_name
    if output_dir.exists():
        raise FileExistsError(
            f"Result directory already exists: {output_dir}. Use a new run name."
        )
    output_dir.mkdir(parents=True)

    robot_config = load_robot_config(args.robot)
    planner_config = load_planner_config(args.planner)
    robot = robot_config.create_fields2cover_robot()
    options = create_options(planner_config)
    raw_field = f2c.Parser().importCellJson(str(map_path))
    clearance = robot_config.obstacle_clearance_m
    field = f2c.Cell.buffer(raw_field, -clearance)
    if field.area() <= 0.0:
        raise ValueError(
            "Safety buffer removed all workable area. Reduce the robot width "
            "or safety margin."
        )

    started = time.perf_counter()
    route = f2c.planCovRoute(robot, field, options)
    path = f2c.planCovPath(robot, field, options)
    elapsed = time.perf_counter() - started

    swaths = flatten_route_swaths(route)
    coverage_objective = f2c.OBJ_FieldCoverage()
    coverage_ratio = coverage_objective.computeCost(field, swaths)
    overlap_ratio = f2c.OBJ_Overlaps().computeCost(field, swaths)
    swath_length = sum(swaths.at(index).length() for index in range(swaths.size()))
    execution_time = swath_length / robot_config.cruise_speed_mps
    covered_area = coverage_ratio * field.area()

    path.saveToFile(str(output_dir / "path.csv"))
    validation = validate_path(
        output_dir / "path.csv", map_path, robot_config.body_width_m
    )
    full_trajectory_coverage = evaluate_full_trajectory_coverage(
        path,
        raw_field,
        field,
        robot_config.coverage_width_m,
        clearance,
    )
    with (output_dir / "validation.json").open("w", encoding="utf-8") as output_file:
        json.dump(validation, output_file, indent=2)

    metrics = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "map": str(map_path.relative_to(EXPERIMENTS_DIR.parent)),
        "field_area_m2": field.area(),
        "raw_field_area_m2": raw_field.area(),
        "safety_excluded_area_m2": raw_field.area() - field.area(),
        "robot_profile": args.robot,
        "robot": {
            **robot_config.__dict__,
            "max_curvature_per_m": robot_config.max_curvature_per_m,
            "obstacle_clearance_m": clearance,
        },
        "planner_profile": args.planner,
        "planner": planner_config,
        # The top-level coverage fields are the comparison metric: the tool
        # sweep generated by all path states, including turns and transfers.
        "coverage_ratio": full_trajectory_coverage["coverage_ratio"],
        "coverage_percent": full_trajectory_coverage["coverage_percent"],
        "covered_area_m2": full_trajectory_coverage["covered_area_m2"],
        "uncovered_area_m2": full_trajectory_coverage["uncovered_area_m2"],
        "full_trajectory_coverage": full_trajectory_coverage,
        "swath_only_coverage": {
            "model": "Fields2Cover OBJ_FieldCoverage over productive swaths only",
            "evaluation_area_m2": field.area(),
            "coverage_ratio": coverage_ratio,
            "coverage_percent": coverage_ratio * 100.0,
            "covered_area_m2": covered_area,
            "uncovered_area_m2": field.area() - covered_area,
        },
        "overlap_ratio": overlap_ratio,
        "swath_count": swaths.size(),
        "swath_length_m": swath_length,
        "productive_swath_time_s": execution_time,
        "path_length_m": path.length(),
        "estimated_task_time_s": path.getTaskTime(),
        "turn_and_transit_time_s": path.getTaskTime() - execution_time,
        "path_state_count": path.size(),
        "planning_wall_time_s": elapsed,
        "path_validation": validation,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)

    plot_result(field, route, output_dir / "route.png")
    plot_result(field, path, output_dir / "path.png")

    if not validation["collision_free"]:
        raise RuntimeError(
            "Path validation failed: the robot footprint leaves the free field. "
            f"See {output_dir / 'metrics.json'} for details."
        )

    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
