# BiRect: coverage path planning for a swing-arm concrete troweling robot

This repository contains the code, maps, experiment configurations and results for the paper

> **BiRect: Coverage Path Planning via Bidirectional Rectangular Decomposition for a Swing-Arm Concrete Troweling Robot**
> Yuzhi Pan, Jia-Rui Lin. Submitted to *Automation in Construction* (under review).

BiRect plans an executable coverage route for a troweling robot whose circular disc is carried on a swing arm. On a known 2-D binary obstacle map it (1) builds bidirectional (row and column) rectangle candidates, fills holes and completes connectivity, (2) orders the rectangles on a feasible transition graph, and (3) turns every visit into coordinated base and swing-arm motion. Coverage ratio and execution time are evaluated from the area actually swept by the disc along the complete trajectory.

## Repository layout

```text
algorithms/            Rectangle segmentation, connectivity completion, graph building and
                       single-rectangle coverage (frozen; hashes in algorithms/MANIFEST.json)
generate_map/          Map generators (random obstacles, Fields2Cover GeoJSON -> raster)
src/                   Pipeline entry point (main_pipeline.py), config loading, visualization
experiments/
  configs/                         Shared robot configuration (robot_config_default.json)
  fields2cover_comparison/         BiRect side of the comparison with Fields2Cover
  segmentation_order_comparison/   Row-only / column-only / row-then-column candidate scans
  vehicle_speed_comparison/        Base speed vs. arm angular velocity (rho = v / omega_a)
  connectivity_repair_ablation/    Additional ablation of connectivity completion (not reported in the paper)
baselines/fields2cover/  Fields2Cover baseline runs (scripts, configs, GeoJSON maps, results)
tests/                 Unit and integration tests
docs/                  Problem/algorithm notes and the original Chinese README
```

Each experiment directory has `configs/`, `maps/`, a runner script (where applicable) and `results/`.

## Maps

All maps are binary occupancy images at 0.5 m/pixel.

| Name in the paper | File | Used in |
|---|---|---|
| `simple` | `map_test1.png` | all three experiments |
| `complex` | `parkingGraph.png` (rasterised from Fields2Cover's `parkingGraph.geojson`) | all three experiments |
| `scan_comb_corridors` | `experiments/segmentation_order_comparison/maps/scan_comb_corridors.png` | scan-strategy experiment |
| `scan_mixed_weave` | `experiments/segmentation_order_comparison/maps/scan_mixed_weave.png` | scan-strategy experiment |
| `scan_offset_rooms` | `experiments/segmentation_order_comparison/maps/scan_offset_rooms.png` | scan-strategy experiment |

The three directional maps were selected with `experiments/segmentation_order_comparison/search_directional_maps.py`. The Fields2Cover baseline uses the metric GeoJSON versions in `baselines/fields2cover/maps/`.

## Environment

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t .   # optional check (36 tests)
```

The Fields2Cover baseline additionally needs the Fields2Cover Python bindings (https://github.com/Fields2Cover/Fields2Cover, BSD-3-Clause) and GDAL's Python bindings (`osgeo`); see `baselines/fields2cover/README.md`.

## Reproducing the experiments

Run all commands from the repository root. Re-running a configuration overwrites its results.

```bash
# 1. BiRect vs. Fields2Cover (BiRect side)
python -m src.main_pipeline --config experiments/fields2cover_comparison/configs/experiment_map_test1.json
python -m src.main_pipeline --config experiments/fields2cover_comparison/configs/experiment_parkingGraph.json

# 1b. Fields2Cover baseline (requires Fields2Cover bindings)
python3 baselines/fields2cover/run_experiment.py --robot robot_baseline --map test1.geojson \
    --run-name robot_baseline_test1_tool1p5_cruise2_turn0p5
python3 baselines/fields2cover/run_experiment.py --robot robot_baseline --map parkingGraph.geojson \
    --run-name robot_baseline_parkingGraph_tool1p5_cruise2_turn0p5

# 2. Candidate-scan strategies on the five maps
python experiments/segmentation_order_comparison/run_comparison.py

# 3. Base speed / arm angular velocity coupling (also writes the plots)
python experiments/vehicle_speed_comparison/run_comparison.py
```

## Results layout

Results produced for the paper are committed (videos `*.mp4` are excluded; rerun to regenerate them).

```text
experiments/fields2cover_comparison/results/<map>/                  metrics.json, config_snapshot.json,
                                                                    segmentation.png, traversal_order.png, coverage_final.png
experiments/segmentation_order_comparison/results/<map>/<strategy>/ same files per strategy
experiments/segmentation_order_comparison/results/<map>/comparison_summary.json
experiments/vehicle_speed_comparison/results/<map>/speed_<v>/       per-speed runs
experiments/vehicle_speed_comparison/results/<map>/comparison_summary.{json,csv}, speed_ratio_curve.{png,pdf}
baselines/fields2cover/results/<run>/                               metrics.json, validation.json, path.csv, path.png, route.png
```

Reported metrics: full-trajectory coverage is `coverage.full_trajectory_disc.coverage_ratio` and execution time is `trajectory.duration_seconds` (summed over continuous motion segments, not Python run time). For the baseline, the motion-feasibility check is `path_validation.collision_free` and the area leaving the valid workspace is `path_validation.swept_area_outside_field_m2`.

## Citation

The paper is under review. Citation details will be added once it is published.

## License

No license file has been added yet.
