# Fields2Cover baseline

These scripts run the fixed-width Fields2Cover baseline used in the paper. They were developed
next to a checkout of Fields2Cover (https://github.com/Fields2Cover/Fields2Cover, BSD-3-Clause)
and need its Python bindings (`import fields2cover`) and GDAL's Python bindings (`osgeo`).
Fields2Cover itself is not redistributed here.

The paper's runs are `results/robot_baseline_test1_tool1p5_cruise2_turn0p5/` (map `simple`)
and `results/robot_baseline_parkingGraph_tool1p5_cruise2_turn0p5/` (map `complex`): 1.50 m
tool width, 2.0 m/s on swaths and 0.5 m/s on turns, as in `configs/robots.json`.

---

# Fields2Cover experiments

This experiment configuration covers the complete free field without a
headland margin. Fields2Cover generates swaths, orders them, and connects them
with Dubins paths using the configured minimum turning radius. No additional
obstacle inflation or collision rejection is applied by the experiment layer.

## Inputs

- `maps/*.geojson`: Metric GeoJSON polygons. Interior rings are obstacles.
- `configs/robots.json`: Robot dimensions, turning parameters, and speeds.
- `configs/planners.json`: Swath, route, and path planning options.

The baseline currently uses a `0.01 m` minimum turning radius. Maximum
curvature is derived as `1 / min_turning_radius_m`.

Inspect the robot profile:

```bash
python3 experiments/robot_config.py robot_baseline
```

## Run

Use a new run name each time:

```bash
python3 experiments/run_experiment.py \
  --robot robot_baseline \
  --map test1.geojson \
  --run-name robot_baseline_test1
```

Each run produces:

- `metrics.json`: complete-path fixed-tool coverage, productive-swath-only
  coverage (kept separately for audit), path length, execution-time estimates,
  and planning wall time.
- `route.png`: Coverage swaths and their ordering.
- `path.png`: Continuous Dubins path.
- `path.csv`: Serialized path states.

`planning_wall_time_s` is computer computation time. `estimated_task_time_s`
uses cruise speed on coverage swaths and turn speed on connecting paths.

`coverage_ratio` is the comparison metric: the fixed-width tool swept along
the complete serialized path, including turns and transfers.  It is evaluated
inside the same safety-buffered field used to generate the route.  The previous
swath-only result remains available as `swath_only_coverage.coverage_ratio`.
The sweep and the safety domain both use Fields2Cover's flat-cap, mitre-join
geometry, so the reported denominator exactly matches the planning field.

To apply this revised metric to an existing run without generating a new path:

```bash
python3 experiments/backfill_full_trajectory_coverage.py \
  --result-dir experiments/results/robot_baseline_test1_cruise2_turn0p5 \
  --map experiments/maps/test1.geojson
```
