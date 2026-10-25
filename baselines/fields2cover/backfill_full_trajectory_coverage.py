#!/usr/bin/env python3
"""Add complete-path coverage metrics to an existing Fields2Cover run.

Use this for result folders generated before ``run_experiment.py`` started
reporting ``full_trajectory_coverage``.  It does not re-plan the route: it
evaluates the already serialized ``path.csv`` under the new metric definition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fields2cover as f2c

from trajectory_coverage import evaluate_full_trajectory_coverage, load_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill complete-path fixed-tool coverage for an existing run."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    metrics_path = result_dir / "metrics.json"
    path_csv = result_dir / "path.csv"
    if not metrics_path.is_file() or not path_csv.is_file():
        raise FileNotFoundError("result-dir must contain metrics.json and path.csv")

    with metrics_path.open(encoding="utf-8") as input_file:
        metrics = json.load(input_file)
    robot = metrics["robot"]
    coverage_width_m = float(robot["coverage_width_m"])
    clearance_m = float(robot["obstacle_clearance_m"])

    # On older runs the top-level values were the productive-swath result.  On
    # newer runs keep the separately recorded audit field intact.
    swath_only = metrics.get("swath_only_coverage")
    if swath_only is None:
        evaluation_area_m2 = float(metrics["field_area_m2"])
        swath_only = {
            "model": "Fields2Cover OBJ_FieldCoverage over productive swaths only",
            "evaluation_area_m2": evaluation_area_m2,
            "coverage_ratio": float(metrics["coverage_ratio"]),
            "coverage_percent": float(metrics["coverage_percent"]),
            "covered_area_m2": float(metrics["covered_area_m2"]),
            "uncovered_area_m2": float(metrics["uncovered_area_m2"]),
        }

    raw_field = f2c.Parser().importCellJson(str(args.map.resolve()))
    field = f2c.Cell.buffer(raw_field, -clearance_m)
    full_trajectory = evaluate_full_trajectory_coverage(
        load_path(str(path_csv)),
        raw_field,
        field,
        coverage_width_m,
        clearance_m,
    )
    metrics.update(
        {
            "coverage_ratio": full_trajectory["coverage_ratio"],
            "coverage_percent": full_trajectory["coverage_percent"],
            "covered_area_m2": full_trajectory["covered_area_m2"],
            "uncovered_area_m2": full_trajectory["uncovered_area_m2"],
            "full_trajectory_coverage": full_trajectory,
            "swath_only_coverage": swath_only,
        }
    )
    with metrics_path.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)
        output_file.write("\n")

    print(
        f"{result_dir.name}: {full_trajectory['coverage_percent']:.2f}% "
        "complete-path coverage"
    )


if __name__ == "__main__":
    main()
