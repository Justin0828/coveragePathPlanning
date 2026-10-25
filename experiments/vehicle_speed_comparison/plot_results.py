"""Plot coverage-duration trade-offs for the vehicle-speed experiment."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


# Palette and typography adapted from the supplied TeX paper figures: muted,
# print-friendly colours, serif type, black axes, and unobtrusive y-gridlines.
COVERAGE_COLOR = "#42678D"
DURATION_COLOR = "#B45A5A"
GRID_COLOR = "#B9B9B9"


def _read_rows(summary_csv: str | Path) -> pd.DataFrame:
    rows = pd.read_csv(summary_csv)
    if rows.empty:
        raise ValueError("汇总 CSV 中没有可绘制的数据。")
    rows = rows.sort_values("configured_speed_mps").copy()
    rows["rho_m_per_deg"] = (
        rows["configured_speed_mps"] / rows["arm_angular_velocity_deg_s"]
    )
    coverage_column = (
        "full_trajectory_disc_coverage_ratio"
        if "full_trajectory_disc_coverage_ratio" in rows.columns
        else "work_disc_coverage_ratio"
    )
    rows["coverage_percent"] = 100.0 * rows[coverage_column]
    return rows


def _set_paper_style() -> None:
    """Apply the compact paper style shared by every speed-experiment plot."""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _style_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    """Use a boxed axis and horizontal reference grid, without point guides."""
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    axis.tick_params(direction="out", length=3.2, width=0.8, color="black")
    if grid:
        axis.grid(
            True,
            axis="y",
            linestyle=":",
            linewidth=0.75,
            color=GRID_COLOR,
            zorder=0,
        )
    else:
        axis.grid(False)


def plot_comparison(summary_csv: str | Path, output_directory: str | Path) -> list[Path]:
    rows = _read_rows(summary_csv)
    speeds = rows["configured_speed_mps"].to_numpy()
    durations = rows["duration_seconds"].to_numpy()
    coverage = rows["coverage_percent"].to_numpy()
    rho = rows["rho_m_per_deg"].to_numpy()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    _set_paper_style()

    tradeoff_path = output_directory / "coverage_duration_curve.png"
    fig, axis = plt.subplots(figsize=(5.3, 3.5))
    axis.plot(
        durations,
        coverage,
        color=COVERAGE_COLOR,
        marker="o",
        markersize=5.5,
        markeredgecolor="black",
        markeredgewidth=0.45,
        linewidth=1.35,
    )
    duration_midpoint = (min(durations) + max(durations)) / 2.0
    for speed, duration, ratio in zip(speeds, durations, coverage):
        align_right = duration > duration_midpoint
        axis.annotate(
            f"{speed:g} m/s",
            (duration, ratio),
            xytext=(-5 if align_right else 5, 5),
            textcoords="offset points",
            fontsize=8.5,
            ha="right" if align_right else "left",
        )
    axis.set_xlabel("Total coverage duration (s)")
    axis.set_ylabel("Full-trajectory coverage ratio (%)")
    axis.set_title("Coverage–duration trade-off")
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(tradeoff_path, dpi=200)
    plt.close(fig)

    speed_path = output_directory / "speed_effect_curve.png"
    fig, coverage_axis = plt.subplots(figsize=(5.3, 3.5))
    duration_axis = coverage_axis.twinx()
    coverage_axis.plot(
        speeds,
        coverage,
        color=COVERAGE_COLOR,
        marker="o",
        markersize=5.5,
        markeredgecolor="black",
        markeredgewidth=0.45,
        linewidth=1.35,
        label="Coverage",
    )
    duration_axis.plot(
        speeds,
        durations,
        color=DURATION_COLOR,
        marker="s",
        markersize=5.0,
        markeredgecolor="black",
        markeredgewidth=0.45,
        linewidth=1.35,
        label="Duration",
    )
    coverage_axis.set_xlabel("Commanded vehicle speed (m/s)")
    coverage_axis.set_ylabel("Full-trajectory coverage ratio (%)", color=COVERAGE_COLOR)
    duration_axis.set_ylabel("Total coverage duration (s)", color=DURATION_COLOR)
    coverage_axis.tick_params(axis="y", labelcolor=COVERAGE_COLOR)
    duration_axis.tick_params(axis="y", labelcolor=DURATION_COLOR)
    coverage_axis.set_title("Effect of vehicle speed")
    _style_axis(coverage_axis)
    _style_axis(duration_axis, grid=False)
    lines = coverage_axis.get_lines() + duration_axis.get_lines()
    coverage_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        handlelength=2.1,
        columnspacing=1.3,
    )
    fig.tight_layout()
    fig.savefig(speed_path, dpi=200)
    plt.close(fig)

    ratio_png_path = output_directory / "speed_ratio_curve.png"
    ratio_pdf_path = output_directory / "speed_ratio_curve.pdf"
    display_names = {"map_test1": "map_simple"}
    map_name = display_names.get(output_directory.name, output_directory.name)
    fig, coverage_axis = plt.subplots(figsize=(7.1, 4.4))
    duration_axis = coverage_axis.twinx()
    coverage_axis.plot(
        rho,
        coverage,
        color=COVERAGE_COLOR,
        marker="o",
        markersize=5.8,
        markeredgecolor="black",
        markeredgewidth=0.45,
        linewidth=1.45,
        label="Full-trajectory coverage",
    )
    duration_axis.plot(
        rho,
        durations,
        color=DURATION_COLOR,
        marker="s",
        markersize=5.2,
        markeredgecolor="black",
        markeredgewidth=0.45,
        linewidth=1.45,
        label="Total duration",
    )

    reference = rows.loc[rows["configured_speed_mps"].sub(1.0).abs().idxmin()]
    coverage_axis.scatter(
        [reference["rho_m_per_deg"]],
        [reference["coverage_percent"]],
        s=45,
        color=COVERAGE_COLOR,
        edgecolor="black",
        linewidth=0.45,
        zorder=5,
    )
    duration_axis.scatter(
        [reference["rho_m_per_deg"]],
        [reference["duration_seconds"]],
        s=41,
        color=DURATION_COLOR,
        marker="s",
        edgecolor="black",
        linewidth=0.45,
        zorder=5,
    )
    coverage_axis.annotate(
        "1.0 m/s",
        (reference["rho_m_per_deg"], reference["coverage_percent"]),
        xytext=(6, 7),
        textcoords="offset points",
        fontsize=8,
        color="black",
    )

    coverage_axis.set_xlabel(
        r"Vehicle speed $v$ (m/s)   |   Speed-to-arm-rate ratio $\rho$ (m/deg)",
        labelpad=28,
    )
    coverage_axis.set_ylabel("Full-trajectory coverage (%)", color=COVERAGE_COLOR)
    duration_axis.set_ylabel("Total duration (s)", color=DURATION_COLOR)
    coverage_axis.tick_params(axis="y", colors=COVERAGE_COLOR)
    duration_axis.tick_params(axis="y", colors=DURATION_COLOR)
    coverage_axis.set_xticks(rho)
    coverage_axis.set_xticklabels([])
    # A single bottom axis carries both experimental controls.  At each tick,
    # speed is placed to the left and its corresponding ratio to the right.
    bottom_axis_transform = coverage_axis.get_xaxis_transform()
    for speed, ratio in zip(speeds, rho):
        coverage_axis.annotate(
            f"{speed:g}",
            xy=(ratio, 0),
            xycoords=bottom_axis_transform,
            xytext=(-4, -8),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=9.5,
            color="black",
            clip_on=False,
        )
        coverage_axis.annotate(
            f"{ratio:.5f}",
            xy=(ratio, 0),
            xycoords=bottom_axis_transform,
            xytext=(4, -8),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9.5,
            color="black",
            clip_on=False,
        )
    coverage_axis.set_ylim(
        max(0.0, coverage.min() - 2.0),
        min(100.0, coverage.max() + 1.5),
    )
    duration_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    _style_axis(coverage_axis)
    _style_axis(duration_axis, grid=False)

    coverage_axis.set_title(map_name, pad=10)

    lines = coverage_axis.get_lines() + duration_axis.get_lines()
    labels = [line.get_label() for line in lines]
    coverage_axis.legend(
        lines,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        handlelength=2.1,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.105, right=0.875, bottom=0.205, top=0.90)
    # Keep both maps on an identical fixed canvas for side-by-side use.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(ratio_png_path, dpi=600)
        fig.savefig(ratio_pdf_path)
    plt.close(fig)

    return [tradeoff_path, speed_path, ratio_png_path, ratio_pdf_path]
