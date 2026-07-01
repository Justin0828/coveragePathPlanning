"""Run the row-only, col-only and row-then-col segmentation ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.configuration import PipelineConfig, load_experiment_config  # noqa: E402
from src.main_pipeline import run_experiment  # noqa: E402


STRATEGIES = ("row_only", "col_only", "row_then_col")
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
RESULT_ROOT = Path(__file__).resolve().parent / "results"


def _config_path(strategy: str, map_name: str) -> Path:
    return CONFIG_ROOT / strategy / f"experiment_{map_name}.json"


def available_maps() -> list[str]:
    """Return map names that have a configuration for every strategy."""
    configured_per_strategy = []
    for strategy in STRATEGIES:
        configured_per_strategy.append(
            {
                path.stem.removeprefix("experiment_")
                for path in (CONFIG_ROOT / strategy).glob("experiment_*.json")
            }
        )
    return sorted(set.intersection(*configured_per_strategy))


def _summary_row(report: dict) -> dict:
    segmentation = report["segmentation"]
    return {
        "strategy": segmentation["candidate_scan_strategy"],
        "result_directory": str(
            Path(report["artifacts"]["metrics_json"]["path"]).parent
        ),
        "candidate_count": segmentation["candidate_count_total"],
        "row_selected_count": segmentation["row_selected_count"],
        "column_selected_count": segmentation["column_selected_count"],
        "greedy_rectangle_count": segmentation["greedy_rectangle_count"],
        "greedy_partition_coverage_ratio": segmentation["greedy_partition"]["free_coverage_ratio"],
        "expanded_rectangle_count": segmentation["expanded_rectangle_count"],
        "connectivity_bridge_count": segmentation["connectivity_bridge_count"],
        "final_rectangle_count": segmentation["final_rectangle_count"],
        "final_partition_coverage_ratio": segmentation["final_partition"]["free_coverage_ratio"],
        "final_partition_overlap_ratio": segmentation["final_partition"]["free_overlap_ratio"],
        "coverage_source": report["coverage"]["source"],
        "partition_coverage_ratio": report["coverage"]["partition_coverage_ratio"],
        "work_disc_coverage_ratio": report["coverage"]["work_disc"]["coverage_ratio"],
        "full_trajectory_disc_coverage_ratio": report["coverage"]["full_trajectory_disc"]["coverage_ratio"],
        "coverage_ratio": report["coverage"]["work_disc"]["coverage_ratio"],
        "trajectory_duration_seconds": report["trajectory"]["duration_seconds"],
        "segmentation_runtime_seconds": report["runtime_seconds"]["segmentation"],
        "total_runtime_seconds": report["runtime_seconds"]["total"],
    }


def _controlled_signature(config: PipelineConfig) -> dict:
    """Return fields that must remain identical across scan strategies."""
    payload = config.snapshot()
    for key in (
        "config_path",
        "output_directory",
        "output_video_path",
        "segmentation_preview_path",
        "traversal_preview_path",
        "coverage_image_path",
        "metrics_path",
        "config_snapshot_path",
        "candidate_scan_strategy",
    ):
        payload.pop(key)
    return payload


def _write_json(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)


def _run_map_comparison(map_name: str, strategies: list[str]) -> dict:
    configs = []
    for strategy in strategies:
        config_path = _config_path(strategy, map_name)
        if not config_path.is_file():
            raise FileNotFoundError(f"地图 {map_name!r} 缺少 {strategy!r} 配置：{config_path}")
        config = load_experiment_config(config_path)
        if config.experiment_name != map_name:
            raise ValueError(
                f"{config_path} 的 experiment_name 必须与地图名一致：{map_name!r}"
            )
        if Path(config.map_path).stem != map_name:
            raise ValueError(f"{config_path} 的 map.path 与地图名 {map_name!r} 不一致。")
        expected_output = (RESULT_ROOT / map_name / strategy).resolve()
        if Path(config.output_directory) != expected_output:
            raise ValueError(
                f"{config_path} 的输出目录必须是 {expected_output}。"
            )
        configs.append(config)

    signatures = [_controlled_signature(config) for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            f"地图 {map_name!r} 的策略配置除扫描策略和输出路径外存在其他差异。"
        )
    rows = [_summary_row(run_experiment(config)) for config in configs]

    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "comparison": "segmentation_candidate_scan_order",
        "experiment_name": map_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "controlled_variables": {
            "map": configs[0].snapshot()["map_path"],
            "robot_config": configs[0].snapshot()["robot_config_path"],
            "random_seed": configs[0].random_seed,
            "note": "Only candidate_scan_strategy differs between configurations.",
        },
        "results": rows,
    }
    _write_json(summary, RESULT_ROOT / map_name / "comparison_summary.json")
    return summary


def run_comparison(strategies: list[str], map_names: list[str] | None = None) -> dict:
    selected_maps = map_names or available_maps()
    if not selected_maps:
        raise ValueError("没有找到三种策略配置齐全的地图实验。")
    summaries = [
        _run_map_comparison(map_name, strategies)
        for map_name in selected_maps
    ]
    return {
        "schema_version": "1.0",
        "status": "completed",
        "comparison": "segmentation_candidate_scan_order",
        "maps": summaries,
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare row-only, col-only and row-then-col candidate scans."
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGIES,
        dest="strategies",
        help="只运行指定策略；可重复传入。默认运行全部三种策略。",
    )
    parser.add_argument(
        "--map",
        action="append",
        choices=available_maps(),
        dest="map_names",
        help="只运行指定地图；可重复传入。默认运行所有配置齐全的地图。",
    )
    args = parser.parse_args(argv)
    try:
        summary = run_comparison(
            args.strategies or list(STRATEGIES),
            args.map_names,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
