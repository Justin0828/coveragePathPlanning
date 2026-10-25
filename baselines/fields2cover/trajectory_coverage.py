"""Full-trajectory fixed-tool coverage measured with Fields2Cover geometry."""

from __future__ import annotations

import fields2cover as f2c


def load_path(path_csv: str) -> f2c.Path:
    """Load the serialized Fields2Cover path used for an existing run."""
    path = f2c.Path()
    path.loadFile(path_csv)
    return path


def evaluate_full_trajectory_coverage(
    path: f2c.Path,
    raw_field: f2c.Cell,
    evaluation_field: f2c.Cell,
    coverage_width_m: float,
    evaluation_clearance_m: float,
) -> dict:
    """Sweep the fixed tool over every serialized path state.

    ``evaluation_field`` must be the same Fields2Cover ``Cell`` supplied to
    ``planCovRoute``/``planCovPath``.  This deliberately avoids OGR's Python
    default buffer settings, which differ from Fields2Cover's flat-cap,
    mitre-join geometry and would otherwise change the comparison denominator.
    """
    if coverage_width_m <= 0.0:
        raise ValueError("coverage_width_m must be positive")
    if evaluation_field.area() <= 0.0:
        raise ValueError("evaluation field must have positive area")

    centerline = f2c.LineString()
    for index in range(path.size()):
        centerline.addPoint(path.getState(index).point)
    if centerline.size() < 2:
        raise ValueError("path must contain at least two states")

    swept_footprint = f2c.Cell.buffer(centerline, coverage_width_m / 2.0)
    covered_region = f2c.Cells.intersection(swept_footprint, evaluation_field)
    escaped_footprint = f2c.Cells(swept_footprint).difference(f2c.Cells(raw_field))

    evaluation_area_m2 = evaluation_field.area()
    covered_area_m2 = covered_region.area()
    coverage_ratio = covered_area_m2 / evaluation_area_m2
    return {
        "model": "complete path centerline buffered by half the coverage width",
        "geometry_engine": "Fields2Cover flat-cap, mitre-join buffer",
        "coverage_width_m": coverage_width_m,
        "evaluation_region": "safety-buffered free field",
        "evaluation_clearance_m": evaluation_clearance_m,
        "evaluation_area_m2": evaluation_area_m2,
        "full_trajectory_swept_area_m2": swept_footprint.area(),
        "covered_area_m2": covered_area_m2,
        "uncovered_area_m2": evaluation_area_m2 - covered_area_m2,
        "coverage_ratio": coverage_ratio,
        "coverage_percent": coverage_ratio * 100.0,
        "swept_area_outside_raw_field_m2": escaped_footprint.area(),
    }
