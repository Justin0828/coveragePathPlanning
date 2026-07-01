from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from src.algorithm_api import bg, conn_methods, rc, seg, seg_methods
from src.configuration import REPOSITORY_ROOT, load_experiment_config
from src.main_pipeline import (
    _motion_duration_metrics,
    _traversal_reachability_metrics,
    build_parametric_coverage_masks,
    compute_coverage_metrics,
    segment_map_into_rectangles,
)
from src.visualization import _pad_frame_to_even_dimensions, render_coverage_image, save_traversal_order


class AlgorithmIntegrityTest(unittest.TestCase):
    def test_frozen_algorithm_hashes(self) -> None:
        manifest_path = REPOSITORY_ROOT / "algorithms/MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for filename, expected in manifest["sha256"].items():
            actual = hashlib.sha256((manifest_path.parent / filename).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, filename)


class ConfigurationTest(unittest.TestCase):
    def test_default_config_resolves_research_directories(self) -> None:
        config = load_experiment_config()
        self.assertTrue(Path(config.map_path).is_file())
        self.assertEqual(
            Path(config.map_path).parent,
            REPOSITORY_ROOT / "experiments/fields2cover_comparison/maps",
        )
        self.assertEqual(
            Path(config.metrics_path).parent,
            REPOSITORY_ROOT / "experiments/fields2cover_comparison/results/map_test1",
        )
        self.assertEqual(config.experiment_name, "map_test1")
        self.assertEqual(
            Path(config.robot_config_path),
            REPOSITORY_ROOT / "experiments/configs/robot_config_default.json",
        )
        self.assertEqual(config.candidate_scan_strategy, "row_then_col")
        self.assertEqual(config.connectivity_strategy, "candidate_bridge")

    def test_segmentation_order_configs_are_grouped_by_map_then_strategy(self) -> None:
        root = REPOSITORY_ROOT / "experiments/segmentation_order_comparison/configs"
        for map_name in ("map_test1", "parkingGraph"):
            for strategy in seg_methods.SUPPORTED_CANDIDATE_SCAN_STRATEGIES:
                config = load_experiment_config(
                    root / strategy / f"experiment_{map_name}.json"
                )
                self.assertEqual(config.experiment_name, map_name)
                self.assertEqual(config.candidate_scan_strategy, strategy)
                self.assertEqual(Path(config.map_path).stem, map_name)
                self.assertEqual(
                    Path(config.map_path).parent,
                    REPOSITORY_ROOT
                    / "experiments/segmentation_order_comparison/maps",
                )
                self.assertEqual(
                    Path(config.output_directory),
                    REPOSITORY_ROOT
                    / "experiments/segmentation_order_comparison/results"
                    / map_name
                    / strategy,
                )
                self.assertEqual(
                    Path(config.robot_config_path),
                    REPOSITORY_ROOT / "experiments/configs/robot_config_default.json",
                )

    def test_seed_makes_random_expansion_reproducible(self) -> None:
        config = load_experiment_config()
        with tempfile.TemporaryDirectory() as directory:
            preview = str(Path(directory) / "segmentation.png")
            isolated = replace(config, segmentation_preview_path=preview)
            _, _, first, _ = segment_map_into_rectangles(isolated)
            _, _, second, _ = segment_map_into_rectangles(isolated)
        coordinates = lambda rects: [(r.x1, r.y1, r.x2, r.y2) for r in rects]
        self.assertEqual(coordinates(first), coordinates(second))

    def test_connectivity_ablation_configs_are_paired(self) -> None:
        root = REPOSITORY_ROOT / "experiments/connectivity_repair_ablation"
        expected = {
            "with_repair": "candidate_bridge",
            "without_repair": "none",
        }
        for map_name in ("map_test1", "parkingGraph"):
            for variant, strategy in expected.items():
                config = load_experiment_config(
                    root / "configs" / variant / f"experiment_{map_name}.json"
                )
                self.assertEqual(config.experiment_name, map_name)
                self.assertEqual(config.connectivity_strategy, strategy)
                self.assertEqual(
                    Path(config.map_path),
                    root / "maps" / f"{map_name}.png",
                )
                self.assertEqual(
                    Path(config.output_directory),
                    root / "results" / map_name / variant,
                )

    def test_nonpositive_candidate_top_k_is_rejected(self) -> None:
        config_path = (
            REPOSITORY_ROOT
            / "experiments/fields2cover_comparison/configs/experiment_map_test1.json"
        )
        document = json.loads(config_path.read_text(encoding="utf-8"))
        document["segmentation"]["candidate_top_k"] = 0
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "invalid.json"
            invalid_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(invalid_path)

    def test_unknown_connectivity_strategy_is_rejected(self) -> None:
        config_path = (
            REPOSITORY_ROOT
            / "experiments/fields2cover_comparison/configs/experiment_map_test1.json"
        )
        document = json.loads(config_path.read_text(encoding="utf-8"))
        document["connectivity"] = {"strategy": "unknown"}
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "invalid.json"
            invalid_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(invalid_path)

    def test_vehicle_speed_configs_change_only_linear_speed(self) -> None:
        root = REPOSITORY_ROOT / "experiments/vehicle_speed_comparison"
        robots = []
        for path in sorted((root / "configs/robots").glob("robot_speed_*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            speed = document.pop("speed_limit")
            robots.append((speed, document))
        self.assertEqual(
            [speed for speed, _ in robots],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        )
        self.assertTrue(all(document == robots[0][1] for _, document in robots))
        self.assertEqual(robots[0][1]["arm_angular_velocity_limit"], 90.0)

        for map_name in ("map_test1", "parkingGraph"):
            config = load_experiment_config(
                root / "configs" / f"experiment_{map_name}.json"
            )
            self.assertEqual(config.work_speed_policy, "commanded")
            self.assertEqual(Path(config.map_path), root / "maps" / f"{map_name}.png")

    def test_default_work_speed_policy_remains_coverage_safe(self) -> None:
        self.assertEqual(load_experiment_config().work_speed_policy, "coverage_safe")


class SegmentationStrategyTest(unittest.TestCase):
    def test_candidate_top_k_limits_each_scanline_after_deduplication(self) -> None:
        grid = np.zeros((3, 4), dtype=np.bool_)
        grid[2, 3] = 1

        row_top_1 = seg.generateCandidates(grid, 0.0, 1.0, 1, 1)
        row_top_2 = seg.generateCandidates(grid, 0.0, 1.0, 1, 2)
        row_all = seg.generateCandidates(grid, 0.0, 1.0, 1, None)
        col_top_1 = seg.generateColCandidates(grid, 0.0, 1.0, 1, 1)
        col_top_2 = seg.generateColCandidates(grid, 0.0, 1.0, 1, 2)

        self.assertEqual(len(row_top_1), 3)
        self.assertEqual(len(row_top_2), 5)
        self.assertEqual(len(row_all), 5)
        self.assertEqual(len(col_top_1), 4)
        self.assertEqual(len(col_top_2), 7)
        self.assertEqual(
            [(r.x1, r.y1, r.x2, r.y2) for r in row_top_1],
            [(0, 2, 3, 3), (0, 1, 3, 3), (0, 0, 3, 3)],
        )

    def test_connectivity_pool_is_not_limited_by_candidate_top_k(self) -> None:
        grid = np.zeros((3, 4), dtype=np.bool_)
        raw_map = np.full((3, 4), 255, dtype=np.uint8)
        grid[2, 3] = 1
        raw_map[2, 3] = 0
        result = seg_methods.select_candidates(
            strategy="row_only",
            grid_map=grid,
            raw_map=raw_map,
            min_edge_length=0.0,
            pixel_size=1.0,
            grid_size=1,
            candidate_top_k=1,
            alpha=0.0,
            beta=0.0,
        )

        self.assertEqual(result.stats["row_candidate_count"], 3)
        self.assertEqual(result.stats["connectivity_candidate_count"], 5)
        self.assertEqual(len(result.connectivity_candidates), 5)

    def test_candidates_never_include_obstacle_cells(self) -> None:
        grid = np.zeros((6, 7), dtype=np.bool_)
        grid[1, 2] = 1
        grid[3, 4] = 1
        grid[5, 0] = 1

        candidates = (
            seg.generateCandidates(grid, 0.0, 1.0, 1, None)
            + seg.generateColCandidates(grid, 0.0, 1.0, 1, None)
        )

        self.assertGreater(len(candidates), 0)
        self.assertTrue(all(seg.isRectObstacleFree(rect, grid, 1.0, 1) for rect in candidates))

    def test_mask_and_random_expansion_use_world_y_without_second_flip(self) -> None:
        grid = np.zeros((4, 4), dtype=np.bool_)
        grid[0, 1] = 1
        bottom_rect = seg.Rect(0.0, 0.0, 1.0, 1.0)

        masked = seg.maskRects(grid, [bottom_rect], 1.0, 1)
        self.assertEqual(masked[0, 0], 1)
        self.assertEqual(masked[3, 0], 0)

        expanded = seg.randomExpansionSession(
            grid,
            [],
            pixel_size=1.0,
            grid_size=1,
            min_edge_length=0.0,
            max_consecutive_failures=1,
            seed=3,
        )
        self.assertGreater(len(expanded), 0)
        self.assertTrue(all(seg.isRectObstacleFree(rect, grid, 1.0, 1) for rect in expanded))

    def test_connectivity_repair_rejects_obstacle_crossing_bridge(self) -> None:
        grid = np.zeros((2, 5), dtype=np.bool_)
        grid[:, 2] = 1
        selected = [bg.Rect(0, 0, 2, 2), bg.Rect(3, 0, 5, 2)]
        unsafe_bridge = bg.Rect(0, 0, 5, 2)

        repaired = bg.repairConnectivity(
            selected,
            [unsafe_bridge],
            min_edge_length=0.0,
            grid_map=grid,
            pixel_size=1.0,
            grid_size=1,
        )

        self.assertEqual(repaired, selected)

    def test_single_scan_methods_do_not_generate_other_orientation(self) -> None:
        config = load_experiment_config()
        raw_map = seg.loadMap(config.map_path)
        grid_map = seg.resterizeMap(raw_map, config.grid_size)
        common = {
            "grid_map": grid_map,
            "raw_map": raw_map,
            "min_edge_length": config.min_edge_length,
            "pixel_size": config.pixel_size,
            "grid_size": config.grid_size,
            "candidate_top_k": config.candidate_top_k,
            "alpha": config.alpha,
            "beta": config.beta,
        }
        row = seg_methods.select_candidates("row_only", **common)
        col = seg_methods.select_candidates("col_only", **common)
        combined = seg_methods.select_candidates("row_then_col", **common)

        self.assertGreater(row.stats["row_candidate_count"], 0)
        self.assertEqual(row.stats["column_candidate_count"], 0)
        self.assertEqual(col.stats["row_candidate_count"], 0)
        self.assertGreater(col.stats["column_candidate_count"], 0)
        self.assertEqual(combined.stats["scan_stage_order"], ["row", "col"])


class WorkSpeedPolicyTest(unittest.TestCase):
    def _run_sweep(self, policy: str) -> tuple[list[dict], list[rc.MotionSegment]]:
        robot = rc.Robot(
            disc_radius=0.75,
            arm_length=1.2,
            speed_limit=3.0,
            angular_velocity_limit=30.0,
            arm_angle_limit=(-90.0, 90.0),
            arm_angular_velocity_limit=90.0,
            work_speed_policy=policy,
        )
        segments: list[rc.MotionSegment] = []
        poses = rc._generateSweepingMovePoses(
            robot,
            (0.0, 0.0),
            (6.0, 0.0),
            0.0,
            -45.0,
            45.0,
            robot.speed_limit,
            robot.arm_angular_velocity_limit,
            2.0,
            segments,
        )
        return poses, segments

    def test_commanded_policy_uses_configured_vehicle_speed(self) -> None:
        poses, segments = self._run_sweep("commanded")
        self.assertAlmostEqual(max(pose["car_speed"] for pose in poses), 3.0)
        self.assertAlmostEqual(segments[0].duration, 2.0)

    def test_safe_policy_preserves_coverage_speed_clamp(self) -> None:
        poses, segments = self._run_sweep("coverage_safe")
        self.assertAlmostEqual(max(pose["car_speed"] for pose in poses), 1.5)
        self.assertAlmostEqual(segments[0].duration, 4.0)


class ConnectivityStrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = np.zeros((2, 6), dtype=np.bool_)
        self.selected = [
            bg.Rect(0.0, 0.0, 2.0, 2.0),
            bg.Rect(4.0, 0.0, 6.0, 2.0),
        ]
        self.bridge = bg.Rect(2.0, 0.0, 4.0, 2.0)

    def _apply(self, strategy: str):
        return conn_methods.apply_connectivity_strategy(
            strategy=strategy,
            selected_rectangles=self.selected,
            connectivity_candidates=[self.bridge],
            min_edge_length=0.0,
            grid_map=self.grid,
            pixel_size=1.0,
            grid_size=1,
        )

    def test_none_is_an_identity_and_reports_disconnection(self) -> None:
        result = self._apply("none")

        self.assertEqual(result.rectangles, self.selected)
        self.assertIsNot(result.rectangles, self.selected)
        self.assertEqual(result.bridge_rectangles, [])
        self.assertEqual(result.stats["component_count_before"], 2)
        self.assertEqual(result.stats["component_count_after"], 2)
        self.assertFalse(result.stats["fully_connected_after"])

    def test_candidate_bridge_connects_components_without_reordering(self) -> None:
        result = self._apply("candidate_bridge")

        self.assertEqual(result.rectangles[:2], self.selected)
        self.assertEqual(result.bridge_rectangles, [self.bridge])
        self.assertEqual(result.stats["component_count_before"], 2)
        self.assertEqual(result.stats["component_count_after"], 1)
        self.assertTrue(result.stats["fully_connected_after"])
        self.assertEqual(result.stats["connectivity_bridge_count"], 1)
        self.assertAlmostEqual(result.stats["connectivity_bridge_area_m2"], 4.0)

    def test_pre_repair_hash_is_strategy_independent(self) -> None:
        without = self._apply("none")
        with_repair = self._apply("candidate_bridge")
        self.assertEqual(
            without.stats["pre_repair_rectangles_sha256"],
            with_repair.stats["pre_repair_rectangles_sha256"],
        )

    def test_real_map_pre_repair_rectangles_match_between_variants(self) -> None:
        root = REPOSITORY_ROOT / "experiments/connectivity_repair_ablation/configs"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with_config = replace(
                load_experiment_config(
                    root / "with_repair/experiment_map_test1.json"
                ),
                segmentation_preview_path=str(output / "with.png"),
            )
            without_config = replace(
                load_experiment_config(
                    root / "without_repair/experiment_map_test1.json"
                ),
                segmentation_preview_path=str(output / "without.png"),
            )
            _, _, _, with_stats = segment_map_into_rectangles(with_config)
            _, _, _, without_stats = segment_map_into_rectangles(without_config)

        self.assertEqual(
            with_stats["pre_repair_rectangles_sha256"],
            without_stats["pre_repair_rectangles_sha256"],
        )
        self.assertEqual(
            with_stats["start_rectangle_coordinates_before_connectivity"],
            without_stats["start_rectangle_coordinates_before_connectivity"],
        )
        self.assertEqual(without_stats["connectivity_bridge_count"], 0)


class ContinuousDurationTest(unittest.TestCase):
    def test_duration_comes_from_motion_segments(self) -> None:
        segments = [
            rc.MotionSegment(
                "work", True, 1.25, (0.0, 0.0), (1.0, 0.0),
                0.0, 0.0, 0.0, 0.0,
            ),
            rc.MotionSegment(
                "transition", False, 2.5, (1.0, 0.0), (3.0, 0.0),
                0.0, 0.0, 0.0, 0.0,
            ),
        ]

        metrics = _motion_duration_metrics(segments)

        self.assertEqual(metrics["duration_source"], "continuous_motion_segments")
        self.assertAlmostEqual(metrics["duration_seconds"], 3.75)
        self.assertAlmostEqual(metrics["work_duration_seconds"], 1.25)
        self.assertAlmostEqual(metrics["non_work_duration_seconds"], 2.5)
        self.assertEqual(
            metrics["phase_duration_seconds"],
            {"work": 1.25, "transition": 2.5},
        )


class ReachabilityMetricsTest(unittest.TestCase):
    def test_disconnected_rectangles_are_reported_as_unreachable(self) -> None:
        rectangles = [
            bg.Rect(0.0, 0.0, 2.0, 2.0),
            bg.Rect(4.0, 0.0, 6.0, 2.0),
        ]
        adjacency = bg.getAdjacencyGraph(rectangles, min_edge_length=0.0)
        raw_map = np.full((2, 6), 255, dtype=np.uint8)

        metrics = _traversal_reachability_metrics(
            rectangles,
            adjacency,
            start_rect=0,
            raw_map=raw_map,
            pixel_size=1.0,
        )

        self.assertEqual(metrics["reachable_rectangles"], [0])
        self.assertEqual(metrics["unreachable_rectangles"], [1])
        self.assertEqual(metrics["reachable_rectangle_ratio"], 0.5)
        self.assertFalse(metrics["all_rectangles_reachable"])
        self.assertAlmostEqual(
            metrics["reachable_partition"]["free_coverage_ratio"],
            4.0 / 12.0,
        )


class TraversalCostConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = load_experiment_config().robot

    def test_coverage_cost_uses_robot_speed_turn_rate_and_geometry(self) -> None:
        rect = bg.Rect(0.0, 0.0, 10.0, 4.0)

        default_cost = bg.computeCoverageCost(rect, self.robot)
        slower_cost = bg.computeCoverageCost(
            rect,
            replace(self.robot, speed_limit=self.robot.speed_limit / 2.0),
        )
        faster_turn_cost = bg.computeCoverageCost(
            rect,
            replace(
                self.robot,
                angular_velocity_limit=self.robot.angular_velocity_limit * 2.0,
            ),
        )
        longer_arm_cost = bg.computeCoverageCost(
            rect,
            replace(self.robot, arm_length=2.0),
        )

        self.assertGreater(slower_cost, default_cost)
        self.assertLess(faster_turn_cost, default_cost)
        self.assertLess(longer_arm_cost, default_cost)

    def test_transition_cost_uses_robot_speed_and_turn_rate(self) -> None:
        rect_a = bg.Rect(0.0, 0.0, 2.0, 2.0)
        rect_b = bg.Rect(10.0, 0.0, 12.0, 2.0)

        cost = bg.computeTransitionCost(rect_a, rect_b, self.robot)
        expected = (
            10.0 / self.robot.speed_limit
            + 180.0 / self.robot.angular_velocity_limit
        )
        self.assertAlmostEqual(cost, expected)

        changed = replace(
            self.robot,
            speed_limit=self.robot.speed_limit * 2.0,
            angular_velocity_limit=self.robot.angular_velocity_limit * 2.0,
        )
        self.assertLess(bg.computeTransitionCost(rect_a, rect_b, changed), cost)

    def test_invalid_cost_configuration_is_rejected(self) -> None:
        rect = bg.Rect(0.0, 0.0, 2.0, 2.0)
        with self.assertRaises(ValueError):
            bg.computeCoverageCost(rect, replace(self.robot, speed_limit=0.0))
        with self.assertRaises(ValueError):
            bg.computeCoverageCost(rect, replace(self.robot, arm_length=0.0))
        with self.assertRaises(ValueError):
            bg.computeCoverageCost(rect, replace(self.robot, arm_angle_limit=(10.0, 90.0)))


class VariableArmCoverageTest(unittest.TestCase):
    def test_local_coverage_generates_sweeping_arm_angles(self) -> None:
        robot = rc.Robot(
            disc_radius=0.75,
            arm_length=1.2,
            car_width=0.8,
            car_half_length=1.0,
            pivot_to_car_center=0.3,
            speed_limit=1.0,
            angular_velocity_limit=30.0,
            arm_angle_limit=(-90.0, 90.0),
            arm_angular_velocity_limit=90.0,
        )
        rect = rc.Rect(0.0, 0.0, 6.0, 3.0)
        poses = rc.boustrophedonCoverage(
            rect,
            in_point=(0.0, 3.0),
            out_point=(6.0, 0.0),
            robot=rc.findRobotPoseByInPoint(rect, (0.0, 3.0), robot),
            sample_rate=4.0,
        )
        arm_angles = [pose["arm_angle"] for pose in poses]
        self.assertLess(min(arm_angles), -1.0)
        self.assertGreater(max(arm_angles), 1.0)

    def test_work_pass_oscillates_at_fixed_angular_speed(self) -> None:
        robot = rc.Robot(
            disc_radius=0.75,
            arm_length=1.2,
            speed_limit=2.0,
            angular_velocity_limit=30.0,
            arm_angle_limit=(-90.0, 90.0),
            arm_angular_velocity_limit=90.0,
        )
        segments: list[rc.MotionSegment] = []

        poses = rc._generateSweepingMovePoses(
            robot,
            node_a=(0.0, 0.0),
            node_b=(12.0, 0.0),
            heading=0.0,
            arm_angle_start=-45.0,
            arm_angle_end=45.0,
            speed_limit=2.0,
            arm_angular_velocity_limit=90.0,
            sample_rate=10.0,
            motion_segments=segments,
        )

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.arm_motion, "oscillating")
        self.assertEqual(abs(segment.arm_angular_velocity), 90.0)
        velocities = {np.sign(pose["arm_angular_velocity"]) for pose in poses}
        self.assertEqual(velocities, {-1.0, 1.0})
        # 90° half-sweep takes 1 s, so a 1.5 m disc diameter limits speed to 1.5 m/s.
        self.assertAlmostEqual(max(pose["car_speed"] for pose in poses), 1.5)

    def test_repeated_sweep_produces_a_wide_continuous_strip(self) -> None:
        robot = rc.Robot(
            disc_radius=0.75,
            arm_length=1.2,
            pivot_to_car_center=0.0,
            speed_limit=2.0,
            angular_velocity_limit=30.0,
            arm_angle_limit=(-90.0, 90.0),
            arm_angular_velocity_limit=90.0,
        )
        segments: list[rc.MotionSegment] = []
        rc._generateSweepingMovePoses(
            robot,
            node_a=(2.0, 5.0),
            node_b=(18.0, 5.0),
            heading=0.0,
            arm_angle_start=-45.0,
            arm_angle_end=45.0,
            speed_limit=2.0,
            arm_angular_velocity_limit=90.0,
            sample_rate=2.0,
            motion_segments=segments,
        )

        work, _, _ = build_parametric_coverage_masks(
            segments,
            (20, 40),
            robot,
            0.5,
        )
        interior_column = work[:, 20]
        covered_rows = np.flatnonzero(interior_column)
        self.assertGreaterEqual(len(covered_rows), 5)
        self.assertTrue(np.all(np.diff(covered_rows) == 1))


class VisualizationConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_map = np.full((20, 20), 255, dtype=np.uint8)
        self.robot = rc.Robot(
            disc_radius=1.0,
            arm_length=1.0,
            car_width=0.8,
            car_half_length=1.0,
            pivot_to_car_center=0.0,
            speed_limit=1.0,
            angular_velocity_limit=30.0,
            arm_angle_limit=(-90.0, 90.0),
            arm_angular_velocity_limit=90.0,
        )
        self.poses = [{
            "heading": 0.0,
            "arm_angle": 0.0,
            "car_speed": 0.0,
            "car_angular_velocity": 0.0,
            "car_position": (5.0, 5.0),
        }]

    def test_blue_pixels_match_coverage_metric(self) -> None:
        metrics = compute_coverage_metrics(self.poses, self.raw_map, self.robot, 1.0)
        color = (30, 144, 255)
        image = render_coverage_image(
            self.raw_map,
            self.raw_map,
            metrics["covered_map"],
            color,
        )
        blue_pixels = int(np.sum(np.all(image == color, axis=2)))
        self.assertEqual(blue_pixels, metrics["covered_free_px"])

    def test_parametric_metrics_do_not_depend_on_pose_sampling_density(self) -> None:
        raw_map = np.full((10, 10), 255, dtype=np.uint8)
        robot = rc.Robot(
            disc_radius=0.75,
            arm_length=1.0,
            car_width=0.8,
            car_half_length=1.0,
            pivot_to_car_center=0.0,
            speed_limit=1.0,
            angular_velocity_limit=30.0,
            arm_angle_limit=(-90.0, 90.0),
            arm_angular_velocity_limit=90.0,
        )
        segment = rc.MotionSegment(
            phase="work",
            coverage_active=True,
            duration=2.0,
            car_position_start=(2.0, 5.0),
            car_position_end=(8.0, 5.0),
            heading_start=0.0,
            heading_end=0.0,
            arm_angle_start=-45.0,
            arm_angle_end=45.0,
        )
        sparse = compute_coverage_metrics(
            [],
            raw_map,
            robot,
            0.5,
            motion_segments=[segment],
        )
        dense = compute_coverage_metrics(
            [{"unused": True}] * 1000,
            raw_map,
            robot,
            0.5,
            motion_segments=[segment],
        )

        self.assertEqual(sparse["coverage_source"], "parametric_swept_disc")
        np.testing.assert_array_equal(
            sparse["work_covered_map"],
            dense["work_covered_map"],
        )

    def test_fixed_arm_straight_motion_is_a_capsule(self) -> None:
        robot = replace(self.robot, disc_radius=1.0, arm_length=1.0, pivot_to_car_center=0.0)
        segment = rc.MotionSegment(
            phase="work",
            coverage_active=True,
            duration=4.0,
            car_position_start=(3.0, 5.0),
            car_position_end=(7.0, 5.0),
            heading_start=180.0,
            heading_end=180.0,
            arm_angle_start=0.0,
            arm_angle_end=0.0,
        )

        work, trajectory, _ = build_parametric_coverage_masks(
            [segment],
            (10, 12),
            robot,
            1.0,
        )

        self.assertTrue(work[4, 3])
        self.assertTrue(work[4, 7])
        self.assertFalse(work[2, 5])
        np.testing.assert_array_equal(work, trajectory)

    def test_stationary_arm_rotation_has_an_uncovered_inner_region(self) -> None:
        robot = replace(self.robot, disc_radius=0.5, arm_length=2.0, pivot_to_car_center=0.0)
        segment = rc.MotionSegment(
            phase="arm_setup",
            coverage_active=True,
            duration=1.0,
            car_position_start=(5.0, 5.0),
            car_position_end=(5.0, 5.0),
            heading_start=0.0,
            heading_end=0.0,
            arm_angle_start=0.0,
            arm_angle_end=90.0,
        )

        work, _, _ = build_parametric_coverage_masks(
            [segment],
            (20, 20),
            robot,
            0.5,
        )

        self.assertFalse(work[10, 10])  # pivot 附近属于内凹区域
        self.assertTrue(work[7, 7])     # 摆臂圆弧附近被扫掠

    def test_inactive_motion_only_contributes_to_full_trajectory(self) -> None:
        active = rc.MotionSegment(
            "work", True, 1.0, (2.0, 2.0), (6.0, 2.0), 180.0, 180.0, 0.0, 0.0
        )
        inactive = rc.MotionSegment(
            "transition", False, 1.0, (2.0, 7.0), (6.0, 7.0), 180.0, 180.0, 0.0, 0.0
        )
        work, trajectory, _ = build_parametric_coverage_masks(
            [active, inactive],
            (10, 10),
            self.robot,
            1.0,
        )

        self.assertGreater(np.sum(trajectory), np.sum(work))
        self.assertTrue(np.all(work <= trajectory))

    def test_traversal_figure_is_written(self) -> None:
        rects = [bg.Rect(0, 0, 5, 5), bg.Rect(5, 0, 10, 5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.png"
            save_traversal_order(self.raw_map, rects, [0, 1], 1.0, path)
            self.assertGreater(path.stat().st_size, 0)

    def test_odd_sized_video_frame_is_padded_to_even_dimensions(self) -> None:
        frame = np.full((131, 170, 3), 255, dtype=np.uint8)
        padded = _pad_frame_to_even_dimensions(frame)

        self.assertEqual(padded.shape, (132, 170, 3))
        np.testing.assert_array_equal(padded[:131], frame)
        self.assertTrue(np.all(padded[131] == 0))


if __name__ == "__main__":
    unittest.main()
