"""Run the paired connectivity-repair ablation on configured maps."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.configuration import PipelineConfig, load_experiment_config  # noqa: E402
from src.main_pipeline import run_experiment  # noqa: E402


VARIANTS = ("without_repair", "with_repair")
EXPECTED_STRATEGIES = {
    "without_repair": "none",
    "with_repair": "candidate_bridge",
}
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
RESULT_ROOT = Path(__file__).resolve().parent / "results"


def _config_path(variant: str, map_name: str) -> Path:
    return CONFIG_ROOT / variant / f"experiment_{map_name}.json"


def available_maps() -> list[str]:
    """Return map names with a configuration for both variants."""
    configured = []
    for variant in VARIANTS:
        configured.append({
            path.stem.removeprefix("experiment_")
            for path in (CONFIG_ROOT / variant).glob("experiment_*.json")
        })
    return sorted(set.intersection(*configured))


def _controlled_signature(config: PipelineConfig) -> dict:
    """Return fields that must be identical across the paired variants."""
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
        "connectivity_strategy",
    ):
        payload.pop(key)
    return payload


def _summary_row(report: dict, variant: str) -> dict:
    segmentation = report["segmentation"]
    traversal = report["traversal"]
    coverage = report["coverage"]
    trajectory = report["trajectory"]
    return {
        "variant": variant,
        "connectivity_strategy": segmentation["connectivity_strategy"],
        "result_directory": str(
            Path(report["artifacts"]["metrics_json"]["path"]).parent
        ),
        "pre_repair_rectangles_sha256": segmentation["pre_repair_rectangles_sha256"],
        "component_count_before": segmentation["component_count_before"],
        "component_count_after": segmentation["component_count_after"],
        "fully_connected_after": segmentation["fully_connected_after"],
        "bridge_count": segmentation["connectivity_bridge_count"],
        "bridge_area_m2": segmentation["connectivity_bridge_area_m2"],
        "final_rectangle_count": segmentation["final_rectangle_count"],
        "final_partition_coverage_ratio": segmentation["final_partition"]["free_coverage_ratio"],
        "final_partition_overlap_ratio": segmentation["final_partition"]["free_overlap_ratio"],
        "reachable_rectangle_count": traversal["reachable_rectangle_count"],
        "unreachable_rectangle_count": traversal["unreachable_rectangle_count"],
        "reachable_rectangle_ratio": traversal["reachable_rectangle_ratio"],
        "reachable_partition_coverage_ratio": traversal["reachable_partition"]["free_coverage_ratio"],
        "all_rectangles_reachable": traversal["all_rectangles_reachable"],
        "revisit_count": traversal["revisit_count"],
        "transition_count": traversal["transition_count"],
        "work_disc_coverage_ratio": coverage["work_disc"]["coverage_ratio"],
        "covered_free_area_m2": coverage["covered_free_area_m2"],
        "duration_seconds": trajectory["duration_seconds"],
        "work_duration_seconds": trajectory["work_duration_seconds"],
        "non_work_duration_seconds": trajectory["non_work_duration_seconds"],
        "seconds_per_covered_m2": coverage["execution_seconds_per_covered_m2"],
        "segmentation_runtime_seconds": report["runtime_seconds"]["segmentation"],
        "traversal_runtime_seconds": report["runtime_seconds"]["traversal"],
        "total_runtime_seconds": report["runtime_seconds"]["total"],
    }


def _paired_delta(rows: dict[str, dict]) -> dict:
    """Return with-repair minus without-repair for interpretable metrics."""
    with_repair = rows["with_repair"]
    without_repair = rows["without_repair"]
    fields = (
        "component_count_after",
        "bridge_count",
        "bridge_area_m2",
        "final_rectangle_count",
        "final_partition_coverage_ratio",
        "final_partition_overlap_ratio",
        "reachable_rectangle_count",
        "unreachable_rectangle_count",
        "reachable_rectangle_ratio",
        "reachable_partition_coverage_ratio",
        "revisit_count",
        "transition_count",
        "work_disc_coverage_ratio",
        "covered_free_area_m2",
        "duration_seconds",
        "work_duration_seconds",
        "non_work_duration_seconds",
        "seconds_per_covered_m2",
        "segmentation_runtime_seconds",
        "traversal_runtime_seconds",
        "total_runtime_seconds",
    )
    return {
        field: (
            with_repair[field] - without_repair[field]
            if with_repair[field] is not None and without_repair[field] is not None
            else None
        )
        for field in fields
    }


def _write_json(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)


def _write_csv(summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        for result in summary["results"]:
            rows.append({"map": summary["experiment_name"], **result})
    output = RESULT_ROOT / "comparison_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()) if rows else ["map"],
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def _run_map_ablation(map_name: str, variants: list[str]) -> dict:
    configs: list[PipelineConfig] = []
    for variant in variants:
        config_path = _config_path(variant, map_name)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"地图 {map_name!r} 缺少 {variant!r} 配置：{config_path}"
            )
        config = load_experiment_config(config_path)
        if config.connectivity_strategy != EXPECTED_STRATEGIES[variant]:
            raise ValueError(
                f"{config_path} 的 connectivity.strategy 应为 "
                f"{EXPECTED_STRATEGIES[variant]!r}。"
            )
        if config.experiment_name != map_name or Path(config.map_path).stem != map_name:
            raise ValueError(f"{config_path} 的实验名、地图名与 {map_name!r} 不一致。")
        expected_output = (RESULT_ROOT / map_name / variant).resolve()
        if Path(config.output_directory) != expected_output:
            raise ValueError(f"{config_path} 的输出目录必须是 {expected_output}。")
        configs.append(config)

    signatures = [_controlled_signature(config) for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            f"地图 {map_name!r} 的配对配置除连通性策略和输出路径外存在差异。"
        )

    reports = {
        variant: run_experiment(config)
        for variant, config in zip(variants, configs)
    }
    hashes = {
        report["segmentation"]["pre_repair_rectangles_sha256"]
        for report in reports.values()
    }
    if len(hashes) != 1:
        raise ValueError(
            f"地图 {map_name!r} 的两组补全前矩形不一致，消融实验无效。"
        )
    start_rectangles = {
        json.dumps(report["traversal"]["start_rectangle_coordinates"], sort_keys=True)
        for report in reports.values()
    }
    if len(start_rectangles) != 1:
        raise ValueError(f"地图 {map_name!r} 的两组起始矩形不一致。")

    rows_by_variant = {
        variant: _summary_row(report, variant)
        for variant, report in reports.items()
    }
    with_repair = rows_by_variant.get("with_repair")
    ablation_triggered = bool(
        with_repair and with_repair["bridge_count"] > 0
    )
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "comparison": "connectivity_repair_ablation",
        "experiment_name": map_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "controlled_variables": {
            "map": configs[0].snapshot()["map_path"],
            "robot_config": configs[0].snapshot()["robot_config_path"],
            "candidate_scan_strategy": configs[0].candidate_scan_strategy,
            "random_seed": configs[0].random_seed,
            "pre_repair_rectangles_sha256": next(iter(hashes)),
            "note": "Only connectivity_strategy and output paths differ.",
        },
        "ablation_triggered": ablation_triggered,
        "interpretation": (
            "bridge_repair_treatment"
            if ablation_triggered
            else "natural_negative_control_no_bridge_inserted"
        ),
        "results": [rows_by_variant[variant] for variant in variants],
        "delta_with_minus_without": (
            _paired_delta(rows_by_variant)
            if set(VARIANTS).issubset(rows_by_variant)
            else None
        ),
    }
    _write_json(summary, RESULT_ROOT / map_name / "comparison_summary.json")
    return summary


def run_ablation(
    variants: list[str],
    map_names: list[str] | None = None,
) -> dict:
    selected_maps = map_names or available_maps()
    if not selected_maps:
        raise ValueError("没有找到两种连通性策略配置齐全的地图。")
    summaries = [
        _run_map_ablation(map_name, variants)
        for map_name in selected_maps
    ]
    if set(variants) == set(VARIANTS):
        _write_csv(summaries)
    return {
        "schema_version": "1.0",
        "status": "completed",
        "comparison": "connectivity_repair_ablation",
        "maps": summaries,
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare coverage with and without connectivity repair."
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=VARIANTS,
        dest="variants",
        help="只运行指定变体；可重复传入。默认运行配对实验。",
    )
    parser.add_argument(
        "--map",
        action="append",
        choices=available_maps(),
        dest="map_names",
        help="只运行指定地图；可重复传入。默认运行全部两张地图。",
    )
    args = parser.parse_args(argv)
    try:
        payload = run_ablation(
            args.variants or list(VARIANTS),
            args.map_names,
        )
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
