"""Plot coverage-duration trade-offs for the vehicle-speed experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_rows(summary_csv: str | Path) -> list[dict[str, str]]:
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_comparison(summary_csv: str | Path, output_directory: str | Path) -> list[Path]:
    rows = _read_rows(summary_csv)
    if not rows:
        raise ValueError("汇总 CSV 中没有可绘制的数据。")
    rows.sort(key=lambda row: float(row["configured_speed_mps"]))
    speeds = [float(row["configured_speed_mps"]) for row in rows]
    durations = [float(row["duration_seconds"]) for row in rows]
    coverage = [100.0 * float(row["work_disc_coverage_ratio"]) for row in rows]

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    tradeoff_path = output_directory / "coverage_duration_curve.png"
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.plot(durations, coverage, marker="o", linewidth=1.8)
    duration_midpoint = (min(durations) + max(durations)) / 2.0
    for speed, duration, ratio in zip(speeds, durations, coverage):
        align_right = duration > duration_midpoint
        axis.annotate(
            f"{speed:g} m/s",
            (duration, ratio),
            xytext=(-5 if align_right else 5, 5),
            textcoords="offset points",
            fontsize=8,
            ha="right" if align_right else "left",
        )
    axis.set_xlabel("Total coverage duration (s)")
    axis.set_ylabel("Work-disc coverage ratio (%)")
    axis.set_title("Coverage–duration trade-off")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(tradeoff_path, dpi=200)
    plt.close(fig)

    speed_path = output_directory / "speed_effect_curve.png"
    fig, coverage_axis = plt.subplots(figsize=(7.2, 5.0))
    duration_axis = coverage_axis.twinx()
    coverage_axis.plot(speeds, coverage, color="tab:blue", marker="o", label="Coverage")
    duration_axis.plot(speeds, durations, color="tab:orange", marker="s", label="Duration")
    coverage_axis.set_xlabel("Commanded vehicle speed (m/s)")
    coverage_axis.set_ylabel("Work-disc coverage ratio (%)", color="tab:blue")
    duration_axis.set_ylabel("Total coverage duration (s)", color="tab:orange")
    coverage_axis.tick_params(axis="y", labelcolor="tab:blue")
    duration_axis.tick_params(axis="y", labelcolor="tab:orange")
    coverage_axis.set_title("Effect of vehicle speed")
    coverage_axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(speed_path, dpi=200)
    plt.close(fig)
    return [tradeoff_path, speed_path]
