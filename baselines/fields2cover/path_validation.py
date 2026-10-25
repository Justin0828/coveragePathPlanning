"""Geometry checks for the robot footprint along a planned path."""

from __future__ import annotations

from pathlib import Path

from osgeo import ogr


def _load_field_geometry(map_path: Path):
    source = ogr.Open(str(map_path))
    if source is None:
        raise RuntimeError(f"Unable to open map for validation: {map_path}")
    layer = source.GetLayer(0)
    feature = layer.GetNextFeature()
    if feature is None or feature.GetGeometryRef() is None:
        raise RuntimeError(f"Map does not contain a polygon feature: {map_path}")
    return feature.GetGeometryRef().Clone()


def _load_path_line(path_csv: Path):
    line = ogr.Geometry(ogr.wkbLineString)
    with path_csv.open(encoding="utf-8") as input_file:
        for row in input_file:
            columns = row.split()
            if len(columns) < 2:
                continue
            line.AddPoint_2D(float(columns[0]), float(columns[1]))
    if line.GetPointCount() < 2:
        raise RuntimeError(f"Path must contain at least two states: {path_csv}")
    return line


def validate_path(
    path_csv: Path,
    map_path: Path,
    body_width_m: float,
    area_tolerance_m2: float = 1e-6,
    length_tolerance_m: float = 1e-6,
) -> dict:
    """Validate a width-only swept footprint against the original free space.

    The returned footprint is the centerline buffered by half the robot body
    width. Any portion outside the original Polygon is a boundary or obstacle
    collision.
    """
    field = _load_field_geometry(map_path)
    centerline = _load_path_line(path_csv)
    footprint = centerline.Buffer(body_width_m / 2.0)
    escaped_footprint = footprint.Difference(field)
    escaped_centerline = centerline.Difference(field)

    swept_area_outside_field_m2 = escaped_footprint.Area()
    centerline_outside_field_length_m = escaped_centerline.Length()
    collision_free = (
        swept_area_outside_field_m2 <= area_tolerance_m2
        and centerline_outside_field_length_m <= length_tolerance_m
    )
    return {
        "model": "centerline buffered by half the robot body width",
        "body_width_m": body_width_m,
        "swept_area_m2": footprint.Area(),
        "swept_area_outside_field_m2": swept_area_outside_field_m2,
        "centerline_outside_field_length_m": centerline_outside_field_length_m,
        "area_tolerance_m2": area_tolerance_m2,
        "length_tolerance_m": length_tolerance_m,
        "collision_free": collision_free,
    }


def evaluate_full_trajectory_coverage(
    path_csv: Path,
    map_path: Path,
    coverage_width_m: float,
    evaluation_clearance_m: float = 0.0,
) -> dict:
    """Measure fixed-tool coverage from the *complete* planned path.

    ``OBJ_FieldCoverage`` evaluates only productive swaths.  That is useful for
    swath-planner diagnostics, but it is not comparable with an evaluator that
    counts a tool footprint during turns and transit motions as well.  This
    function buffers every state in ``path.csv`` into one continuous swept
    footprint, then clips it to the requested free-space evaluation region.

    The evaluation region is the original field when ``evaluation_clearance_m``
    is zero.  Passing the same clearance used during route generation instead
    evaluates against the safety-buffered field, which preserves the historical
    Fields2Cover denominator while changing only the coverage numerator.
    """
    if coverage_width_m <= 0.0:
        raise ValueError("coverage_width_m must be positive")
    if evaluation_clearance_m < 0.0:
        raise ValueError("evaluation_clearance_m must be non-negative")

    raw_field = _load_field_geometry(map_path)
    evaluation_field = raw_field.Buffer(-evaluation_clearance_m)
    if evaluation_field is None or evaluation_field.IsEmpty() or evaluation_field.Area() <= 0.0:
        raise ValueError("Evaluation clearance removed all workable area")

    centerline = _load_path_line(path_csv)
    swept_footprint = centerline.Buffer(coverage_width_m / 2.0)
    covered_region = swept_footprint.Intersection(evaluation_field)
    escaped_footprint = swept_footprint.Difference(raw_field)

    evaluation_area_m2 = evaluation_field.Area()
    covered_area_m2 = covered_region.Area()
    coverage_ratio = covered_area_m2 / evaluation_area_m2
    return {
        "model": "complete path centerline buffered by half the coverage width",
        "coverage_width_m": coverage_width_m,
        "evaluation_region": (
            "safety-buffered free field"
            if evaluation_clearance_m > 0.0
            else "original free field"
        ),
        "evaluation_clearance_m": evaluation_clearance_m,
        "evaluation_area_m2": evaluation_area_m2,
        "full_trajectory_swept_area_m2": swept_footprint.Area(),
        "covered_area_m2": covered_area_m2,
        "uncovered_area_m2": evaluation_area_m2 - covered_area_m2,
        "coverage_ratio": coverage_ratio,
        "coverage_percent": coverage_ratio * 100.0,
        "swept_area_outside_raw_field_m2": escaped_footprint.Area(),
    }
