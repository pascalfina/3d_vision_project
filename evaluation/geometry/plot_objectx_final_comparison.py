#!/usr/bin/env python3
"""Create poster-ready plot alternatives for Object-X final geometry.

The source is ``current_aggregate_tables.md``.  This script intentionally:

- uses only median-error columns, derived median-Chamfer, and P/R/F1 med;
- excludes all ``final_vs_raw_*`` rows;
- avoids mean-error columns and rainbow palettes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "evaluation" / "outputs" / "geometry" / "current_aggregate_tables.md"
OUT_DIR = REPO_ROOT / "evaluation" / "outputs" / "geometry" / "plots" / "objectx_final_comparison"
SOURCE_TABLE = OUT_DIR / "objectx_final_plot_source.md"

ORDER = [
    ("3RScan", "SAM2 + MUSt3R"),
    ("3RScan", "SAMObject + Pi3X"),
    ("ScanNet", "SAM2 + MUSt3R"),
    ("ScanNet", "SAMObject + Pi3X"),
]

LABELS = {
    ("3RScan", "SAM2 + MUSt3R"): "3RScan\nSAM2 + MUSt3R",
    ("3RScan", "SAMObject + Pi3X"): "3RScan\nSAMObject + Pi3X",
    ("ScanNet", "SAM2 + MUSt3R"): "ScanNet\nSAM2 + MUSt3R",
    ("ScanNet", "SAMObject + Pi3X"): "ScanNet\nSAMObject + Pi3X",
}

# Primary-color palette plus green. Dataset/method pairs remain distinct while
# avoiding rainbow-style categorical colors.
COLORS = {
    "3RScan / SAM2 + MUSt3R": "#0057B8",
    "3RScan / SAMObject + Pi3X": "#D00000",
    "ScanNet / SAM2 + MUSt3R": "#F2C300",
    "ScanNet / SAMObject + Pi3X": "#00843D",
}

METHOD_MARKERS = {
    "SAM2 + MUSt3R": "o",
    "SAMObject + Pi3X": "s",
}

THEME = {
    "bg": "#F4F7FB",
    "panel": "#FFFFFF",
    "text": "#17202C",
    "muted": "#596574",
    "grid": "#D6DEE8",
    "edge": "#182230",
    "blue": "#2364AD",
}

DARK_THEME = {
    "bg": "#0D1726",
    "panel": "#152234",
    "text": "#EEF4FB",
    "muted": "#B8C3D3",
    "grid": "#33465F",
    "edge": "#E9F2FC",
    "blue": "#79AEEB",
}


@dataclass(frozen=True)
class MetricRow:
    dataset: str
    pipeline: str
    comparison: str
    runs: int
    a_metric: str
    a_median_m: float
    b_metric: str
    b_median_m: float
    chamfer_m: float
    p5: float
    r5: float
    f5: float

    @property
    def label(self) -> str:
        return f"{self.dataset} / {self.pipeline}"

    @property
    def stacked_label(self) -> str:
        return LABELS[(self.dataset, self.pipeline)]

    @property
    def short_label(self) -> str:
        method = self.pipeline.replace("SAMObject", "SAMObj").replace("MUSt3R", "M3R")
        return f"{self.dataset}\n{method}"


def _parse_float(value: str) -> float:
    return float(value) if value and value != "n/a" else float("nan")


def _normalise_section(section: str) -> tuple[str, str] | None:
    section = section.replace("Object-X Final:", "").strip()
    if section.startswith("ScanNet "):
        return "ScanNet", section.removeprefix("ScanNet ").strip()
    if section.startswith("3RScan:"):
        return "3RScan", section.removeprefix("3RScan:").strip()
    return None


def parse_summary(path: Path, group: str = "best") -> list[MetricRow]:
    rows: list[MetricRow] = []
    current: tuple[str, str] | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = _normalise_section(line.removeprefix("## ").strip())
            continue
        if current is None or not line.startswith("|"):
            continue
        if "---" in line or "Group" in line:
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 13:
            continue
        row_group, comparison = parts[0], parts[1]
        if row_group != group:
            continue
        if comparison not in {"final_vs_gt", "final_vs_objectx_input"}:
            continue

        dataset, pipeline = current
        rows.append(
            MetricRow(
                dataset=dataset,
                pipeline=pipeline,
                comparison=comparison,
                runs=int(parts[2]),
                a_metric=parts[3].replace(" error", ""),
                a_median_m=_parse_float(parts[5]),
                b_metric=parts[6].replace(" error", ""),
                b_median_m=_parse_float(parts[8]),
                chamfer_m=_parse_float(parts[9]),
                p5=_parse_float(parts[10]),
                r5=_parse_float(parts[11]),
                f5=_parse_float(parts[12]),
            )
        )

    ordered: list[MetricRow] = []
    for dataset, pipeline in ORDER:
        for comparison in ("final_vs_gt", "final_vs_objectx_input"):
            match = [
                row
                for row in rows
                if row.dataset == dataset and row.pipeline == pipeline and row.comparison == comparison
            ]
            if not match:
                raise RuntimeError(f"Missing row for {dataset} / {pipeline} / {comparison}")
            ordered.append(match[0])
    return ordered


def clean_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".svg", ".pdf", ".html", ".md"}:
            path.unlink()


def setup_style(theme: dict[str, str] = THEME) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme["bg"],
            "axes.facecolor": theme["panel"],
            "axes.edgecolor": theme["grid"],
            "axes.labelcolor": theme["text"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "text.color": theme["text"],
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.facecolor": theme["bg"],
            "savefig.edgecolor": theme["bg"],
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.16)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def split_rows(rows: list[MetricRow], comparison: str) -> list[MetricRow]:
    return [row for row in rows if row.comparison == comparison]


def colors(rows: list[MetricRow]) -> list[str]:
    return [COLORS[row.label] for row in rows]


def cm(value_m: float) -> float:
    return value_m * 100.0


def labels(rows: list[MetricRow]) -> list[str]:
    return [row.stacked_label for row in rows]


def contrast_text(hex_color: str) -> str:
    rgb = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#111111" if luminance > 0.52 else "#FFFFFF"


def style_axes(ax: plt.Axes, theme: dict[str, str] = THEME, grid: str | None = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(theme["grid"])
    ax.tick_params(axis="both", colors=theme["muted"])
    ax.grid(False)


def barh_metric(
    ax: plt.Axes,
    rows: list[MetricRow],
    values_m: list[float],
    title: str,
    *,
    theme: dict[str, str] = THEME,
    xmax: float | None = None,
) -> None:
    values_cm = np.array(values_m) * 100.0
    y = np.arange(len(rows))
    if xmax is None:
        xmax = float(np.nanmax(values_cm) * 1.25)
    ax.barh(y, values_cm, color=colors(rows), edgecolor=theme["edge"], linewidth=0.7)
    for yy, value in zip(y, values_cm):
        ax.text(value + xmax * 0.025, yy, f"{value:.1f} cm", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y, labels(rows), fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("cm, lower is better")
    ax.set_title(title, pad=12)
    style_axes(ax, theme, grid="x")


def grouped_score_bars(
    ax: plt.Axes,
    rows: list[MetricRow],
    title: str,
    *,
    theme: dict[str, str] = THEME,
) -> None:
    metrics = [("p5", "Precision"), ("r5", "Recall"), ("f5", "F1")]
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(rows))
    for row, color, offset in zip(rows, colors(rows), offsets):
        vals = [getattr(row, key) for key, _ in metrics]
        bars = ax.bar(x + offset, vals, width, color=color, edgecolor=theme["edge"], linewidth=0.7, label=row.label)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x, [name for _, name in metrics], fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score, higher is better")
    ax.set_title(title, pad=12)
    style_axes(ax, theme, grid="y")


def overview_dashboard(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt_rows = split_rows(rows, "final_vs_gt")
    fig = plt.figure(figsize=(14.6, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_s = fig.add_subplot(grid[1, :])
    fig.suptitle("Object-X Output: Geometry vs. Ground Truth", fontsize=24, fontweight="bold", y=1.06)
    barh_metric(ax_a, gt_rows, [row.a_median_m for row in gt_rows], "Pred -> GT median error", theme=theme)
    barh_metric(ax_b, gt_rows, [row.b_median_m for row in gt_rows], "GT -> Pred median error", theme=theme)
    grouped_score_bars(ax_s, gt_rows, "5 cm scores against GT", theme=theme)
    ax_s.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, stem)


def preservation_dashboard(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    input_rows = split_rows(rows, "final_vs_objectx_input")
    fig = plt.figure(figsize=(14.6, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_s = fig.add_subplot(grid[1, :])
    fig.suptitle("Object-X Output: Geometry Preservation", fontsize=24, fontweight="bold", y=1.06)
    barh_metric(ax_a, input_rows, [row.a_median_m for row in input_rows], "Final -> Input median error", theme=theme)
    barh_metric(ax_b, input_rows, [row.b_median_m for row in input_rows], "Input -> Final median error", theme=theme)
    grouped_score_bars(ax_s, input_rows, "5 cm scores against Object-X input", theme=theme)
    ax_s.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, stem)


def combined_dashboard(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    fig = plt.figure(figsize=(15.0, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.0])
    fig.suptitle("Object-X Final Geometry: Accuracy and Preservation", fontsize=24, fontweight="bold", y=1.04)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    ax4 = fig.add_subplot(grid[1, :])

    barh_metric(ax1, gt, [row.a_median_m for row in gt], "Pred -> GT", theme=theme)
    barh_metric(ax2, gt, [row.b_median_m for row in gt], "GT -> Pred", theme=theme)

    input_error = np.array([cm(row.chamfer_m) for row in inp])
    gt_error = np.array([cm(row.chamfer_m) for row in gt])
    legend_handles = []
    for idx, (row, xval, yval, color) in enumerate(zip(gt, input_error, gt_error, colors(gt)), start=1):
        ax3.scatter(xval, yval, s=320, color=color, edgecolor=theme["edge"], linewidth=1.0)
        ax3.text(
            xval,
            yval,
            str(idx),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=contrast_text(color),
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=theme["edge"], markersize=7, label=f"{idx}. {row.short_label.replace(chr(10), ' / ')}")
        )
    ax3.set_xlabel("Final vs input Chamfer (cm)")
    ax3.set_ylabel("Final vs GT Chamfer (cm)")
    ax3.set_title("Chamfer tradeoff")
    x_pad = max(float(np.nanmax(input_error) - np.nanmin(input_error)) * 0.22, 0.18)
    y_pad = max(float(np.nanmax(gt_error) - np.nanmin(gt_error)) * 0.22, 0.55)
    ax3.set_xlim(max(0.0, float(np.nanmin(input_error)) - x_pad), float(np.nanmax(input_error)) + x_pad)
    ax3.set_ylim(max(0.0, float(np.nanmin(gt_error)) - y_pad), float(np.nanmax(gt_error)) + y_pad)
    ax3.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        fontsize=6.3,
        frameon=False,
        borderaxespad=0.0,
    )
    style_axes(ax3, theme)

    width = 0.34
    x = np.arange(len(gt))
    gt_f1 = [row.f5 for row in gt]
    in_f1 = [row.f5 for row in inp]
    ax4.bar(x - width / 2, gt_f1, width, color=colors(gt), edgecolor=theme["edge"], label="Final vs GT")
    ax4.bar(x + width / 2, in_f1, width, color="white", edgecolor=colors(gt), linewidth=2.0, label="Final vs input")
    for idx, (a, b) in enumerate(zip(gt_f1, in_f1)):
        ax4.plot([idx - width / 2, idx + width / 2], [a, b], color=theme["muted"], alpha=0.45, linewidth=1.0)
        ax4.text(idx - width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax4.text(idx + width / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_xticks(x, [row.short_label for row in gt], fontweight="bold")
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("F1@5cm, higher is better")
    ax4.set_title("F1: ground-truth quality vs input preservation")
    ax4.legend(loc="upper left")
    style_axes(ax4, theme, grid="y")
    save_figure(fig, stem)


def distance_heatmap(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    row_labels = [row.stacked_label for row in gt]
    cols = ["Pred->GT", "GT->Pred", "Final->Input", "Input->Final"]
    data = np.array(
        [
            [cm(g.a_median_m), cm(g.b_median_m), cm(i.a_median_m), cm(i.b_median_m)]
            for g, i in zip(gt, inp)
        ]
    )
    cmap = LinearSegmentedColormap.from_list("distance_map", ["#FFFFFF", "#C8D7E6", "#557A9B", "#1D334A"])
    fig, ax = plt.subplots(figsize=(13.4, 6.2), constrained_layout=True)
    image = ax.imshow(data, cmap=cmap, aspect="auto")
    fig.suptitle("Median Distance Error Matrix", fontsize=22, fontweight="bold", y=1.04)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            value = data[r, c]
            color = "white" if value > np.nanmax(data) * 0.58 else theme["text"]
            ax.text(c, r, f"{value:.1f} cm", ha="center", va="center", fontweight="bold", color=color)
    ax.set_xticks(np.arange(len(cols)), cols, fontweight="bold", rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontweight="bold")
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_label("cm, lower is better")
    save_figure(fig, stem)


def score_heatmap(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    row_labels = [row.stacked_label for row in gt]
    cols = ["GT P", "GT R", "GT F1", "Input P", "Input R", "Input F1"]
    data = np.array([[g.p5, g.r5, g.f5, i.p5, i.r5, i.f5] for g, i in zip(gt, inp)])
    cmap = LinearSegmentedColormap.from_list("score_map", ["#FFFFFF", "#C9D9EA", "#6D8FB1", "#1F4E79"])
    fig, ax = plt.subplots(figsize=(13.4, 6.2), constrained_layout=True)
    image = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    fig.suptitle("5 cm Score Matrix", fontsize=22, fontweight="bold", y=1.04)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            value = data[r, c]
            color = "white" if value > 0.70 else theme["text"]
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontweight="bold", color=color)
    ax.set_xticks(np.arange(len(cols)), cols, fontweight="bold")
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontweight="bold")
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_label("score, higher is better")
    save_figure(fig, stem)


def radar_quality_net(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    metrics = [
        ("Pred->GT", np.array([cm(row.a_median_m) for row in gt]), False),
        ("GT->Pred", np.array([cm(row.b_median_m) for row in gt]), False),
        ("Final->Input", np.array([cm(row.a_median_m) for row in inp]), False),
        ("Input->Final", np.array([cm(row.b_median_m) for row in inp]), False),
        ("Precision", np.array([row.p5 for row in gt]), True),
        ("Recall", np.array([row.r5 for row in gt]), True),
        ("F1", np.array([row.f5 for row in gt]), True),
    ]
    norm_values: list[list[float]] = []
    for _, vals, higher_is_better in metrics:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if hi - lo < 1e-9:
            norm = np.ones_like(vals)
        elif higher_is_better:
            norm = (vals - lo) / (hi - lo)
        else:
            norm = 1.0 - (vals - lo) / (hi - lo)
        norm_values.append(norm.tolist())
    data = np.array(norm_values).T
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(9.2, 8.2), constrained_layout=True)
    ax = fig.add_subplot(111, polar=True)
    fig.suptitle("Ground-Truth Quality Net", fontsize=22, fontweight="bold", y=1.02)
    ax.set_facecolor(theme["panel"])
    for row, vals, color in zip(gt, data, colors(gt)):
        closed = np.concatenate([vals, vals[:1]])
        ax.plot(angles, closed, color=color, linewidth=2.2, label=row.label)
        ax.fill(angles, closed, color=color, alpha=0.13)
    ax.set_xticks(angles[:-1], [m[0] for m in metrics], fontweight="bold")
    ax.set_yticks([0.25, 0.50, 0.75, 1.0], ["0.25", "0.50", "0.75", "best"], color=theme["muted"])
    ax.set_ylim(0, 1.0)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    save_figure(fig, stem)


def precision_recall_scatter(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    comparisons = [("final_vs_gt", "Against GT"), ("final_vs_objectx_input", "Against Object-X input")]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.2), constrained_layout=True)
    fig.suptitle("Precision/Recall Operating Points", fontsize=22, fontweight="bold", y=1.04)
    for ax, (comparison, title) in zip(axes, comparisons):
        sub = split_rows(rows, comparison)
        legend_handles = []
        for idx, (row, color) in enumerate(zip(sub, colors(sub)), start=1):
            marker = METHOD_MARKERS[row.pipeline]
            ax.scatter(
                row.p5,
                row.r5,
                s=280 + row.f5 * 560,
                color=color,
                edgecolor=theme["edge"],
                linewidth=1.1,
                marker=marker,
                zorder=3,
            )
            ax.text(
                row.p5,
                row.r5,
                str(idx),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=contrast_text(color),
                zorder=4,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor=theme["edge"],
                    markersize=9,
                    label=f"{idx}. {row.label}  F1={row.f5:.2f}",
                )
            )
        ax.plot([0, 1], [0, 1], color=theme["grid"], linestyle="--", linewidth=1)
        if comparison == "final_vs_objectx_input":
            ax.set_xlim(0.86, 1.01)
            ax.set_ylim(0.88, 1.01)
            legend_loc = "lower left"
        else:
            ax.set_xlim(0.33, 0.46)
            ax.set_ylim(0.34, 0.88)
            legend_loc = "upper left"
        ax.set_xlabel("Precision@5cm")
        ax.set_ylabel("Recall@5cm")
        ax.set_title(title)
        ax.legend(handles=legend_handles, loc=legend_loc, fontsize=7.7, frameon=True, framealpha=0.92)
        style_axes(ax, theme, grid="both")
    save_figure(fig, stem)


def directional_error_tradeoff(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    fig, ax = plt.subplots(figsize=(9.6, 7.2), constrained_layout=True)
    fig.suptitle("Directional Error Tradeoff", fontsize=22, fontweight="bold", y=1.04)
    label_offsets = {
        "3RScan / SAM2 + MUSt3R": (0.025, 0.0, "left"),
        "3RScan / SAMObject + Pi3X": (0.025, 0.0, "left"),
        "ScanNet / SAM2 + MUSt3R": (-0.030, 0.0, "right"),
        "ScanNet / SAMObject + Pi3X": (0.025, 0.0, "left"),
    }
    for g, i, color in zip(gt, inp, colors(gt)):
        size = 420 + g.f5 * 1000
        xval = cm(i.a_median_m)
        yval = cm(g.a_median_m)
        ax.scatter(xval, yval, s=size, color=color, edgecolor=theme["edge"], linewidth=1.1, alpha=0.92)
        dx, dy, ha = label_offsets[g.label]
        ax.text(xval + dx, yval + dy, g.short_label, va="center", ha=ha, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Final -> Input median error (cm, lower is better)")
    ax.set_ylabel("Pred -> GT median error (cm, lower is better)")
    ax.text(1.52, 6.02, "lower left is best", fontsize=11, fontweight="bold", color=theme["muted"])
    ax.set_xlim(1.45, 2.2)
    ax.set_ylim(5.9, 7.5)
    style_axes(ax, theme)
    save_figure(fig, stem)


def distance_profile_lines(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    metric_names = ["Pred->GT", "GT->Pred", "Final->Input", "Input->Final"]
    x = np.arange(len(metric_names))
    fig, ax = plt.subplots(figsize=(11.6, 6.4), constrained_layout=True)
    fig.suptitle("Median Error Profiles", fontsize=22, fontweight="bold", y=1.04)
    for g, i, color in zip(gt, inp, colors(gt)):
        values = [cm(g.a_median_m), cm(g.b_median_m), cm(i.a_median_m), cm(i.b_median_m)]
        ax.plot(x, values, marker=METHOD_MARKERS[g.pipeline], color=color, linewidth=2.4, markersize=8, label=g.label)
        for xx, yy in zip(x, values):
            ax.text(xx, yy + 0.35, f"{yy:.1f}", ha="center", fontsize=8.5, fontweight="bold", color=color)
    ax.set_xticks(x, metric_names, fontweight="bold")
    ax.set_ylabel("cm, lower is better")
    ax.set_ylim(0, 8.4)
    ax.legend(ncol=2, loc="upper right")
    style_axes(ax, theme, grid="y")
    save_figure(fig, stem)


def f1_slopegraph(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    fig, ax = plt.subplots(figsize=(9.4, 6.4), constrained_layout=True)
    fig.suptitle("F1 Shift: GT Quality to Input Preservation", fontsize=22, fontweight="bold", y=1.04)
    for g, i, color in zip(gt, inp, colors(gt)):
        ax.plot([0, 1], [g.f5, i.f5], color=color, linewidth=3.0, marker=METHOD_MARKERS[g.pipeline], markersize=8)
        ax.text(-0.02, g.f5, f"{g.short_label}  {g.f5:.2f}", ha="right", va="center", fontsize=9, fontweight="bold")
        ax.text(1.02, i.f5, f"{i.f5:.2f}", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.34, 1.18)
    ax.set_ylim(0.30, 1.02)
    ax.set_xticks([0, 1], ["Final vs GT", "Final vs input"], fontweight="bold")
    ax.set_ylabel("F1@5cm, higher is better")
    style_axes(ax, theme, grid="y")
    save_figure(fig, stem)


def parallel_coordinates(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    metric_defs = [
        ("GT Pred->GT", [cm(row.a_median_m) for row in gt], False),
        ("GT GT->Pred", [cm(row.b_median_m) for row in gt], False),
        ("GT F1", [row.f5 for row in gt], True),
        ("Final->Input", [cm(row.a_median_m) for row in inp], False),
        ("Input->Final", [cm(row.b_median_m) for row in inp], False),
        ("Input F1", [row.f5 for row in inp], True),
    ]
    normalized = []
    for _, vals, higher in metric_defs:
        arr = np.array(vals, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-9:
            score = np.ones_like(arr)
        elif higher:
            score = (arr - lo) / (hi - lo)
        else:
            score = 1 - (arr - lo) / (hi - lo)
        normalized.append(score)
    data = np.array(normalized).T
    x = np.arange(len(metric_defs))
    fig, ax = plt.subplots(figsize=(11.4, 6.4), constrained_layout=True)
    fig.suptitle("Normalized Performance Parallel Coordinates", fontsize=22, fontweight="bold", y=1.04)
    for row, vals, color in zip(gt, data, colors(gt)):
        ax.plot(x, vals, color=color, linewidth=3, marker=METHOD_MARKERS[row.pipeline], markersize=8, label=row.label)
    ax.set_xticks(x, [name for name, _, _ in metric_defs], fontweight="bold", rotation=12)
    ax.set_ylabel("normalized score, higher is better")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style_axes(ax, theme, grid="y")
    save_figure(fig, stem)


def lollipop_directional_errors(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    y = np.arange(len(gt))
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    fig.suptitle("Ground-Truth Directional Errors", fontsize=22, fontweight="bold", y=1.04)
    for yy, row, color in zip(y, gt, colors(gt)):
        pred_to_gt = cm(row.a_median_m)
        gt_to_pred = cm(row.b_median_m)
        ax.plot([gt_to_pred, pred_to_gt], [yy, yy], color="#C9D2DD", linewidth=4, solid_capstyle="round")
        ax.scatter(pred_to_gt, yy, s=220, color=color, edgecolor=theme["edge"], linewidth=0.8, label="Pred -> GT" if yy == 0 else None)
        ax.scatter(gt_to_pred, yy, s=180, color="white", edgecolor=color, linewidth=2.2, label="GT -> Pred" if yy == 0 else None)
        ax.text(pred_to_gt, yy + 0.18, f"{pred_to_gt:.1f}", ha="center", fontsize=8.8, fontweight="bold")
        ax.text(gt_to_pred, yy - 0.18, f"{gt_to_pred:.1f}", ha="center", fontsize=8.8, fontweight="bold")
    ax.set_yticks(y, [row.stacked_label for row in gt], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("median error (cm, lower is better)")
    ax.legend(loc="lower right")
    style_axes(ax, theme)
    save_figure(fig, stem)


def rank_summary(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    # Utility combines GT quality and preservation using only directed medians/scores.
    gt_a = np.array([cm(row.a_median_m) for row in gt])
    gt_b = np.array([cm(row.b_median_m) for row in gt])
    in_a = np.array([cm(row.a_median_m) for row in inp])
    in_b = np.array([cm(row.b_median_m) for row in inp])
    gt_f1 = np.array([row.f5 for row in gt])
    in_f1 = np.array([row.f5 for row in inp])

    def inv_norm(values: np.ndarray) -> np.ndarray:
        return 1 - (values - values.min()) / max(values.max() - values.min(), 1e-9)

    def norm(values: np.ndarray) -> np.ndarray:
        return (values - values.min()) / max(values.max() - values.min(), 1e-9)

    utility = (
        0.20 * inv_norm(gt_a)
        + 0.20 * inv_norm(gt_b)
        + 0.15 * inv_norm(in_a)
        + 0.15 * inv_norm(in_b)
        + 0.20 * norm(gt_f1)
        + 0.10 * norm(in_f1)
    )
    order = np.argsort(utility)[::-1]
    fig, ax = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    fig.suptitle("Balanced Geometry Utility Ranking", fontsize=22, fontweight="bold", y=1.04)
    y = np.arange(len(order))
    ordered_rows = [gt[idx] for idx in order]
    ordered_values = utility[order]
    ax.barh(y, ordered_values, color=[COLORS[row.label] for row in ordered_rows], edgecolor=theme["edge"], linewidth=0.7)
    for yy, value in zip(y, ordered_values):
        ax.text(value + 0.025, yy, f"{value:.2f}", va="center", fontweight="bold")
    ax.set_yticks(y, [row.stacked_label for row in ordered_rows], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("balanced normalized utility, higher is better")
    style_axes(ax, theme)
    save_figure(fig, stem)


def utility_sensitivity_curves(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    gt_a = np.array([cm(row.a_median_m) for row in gt])
    gt_b = np.array([cm(row.b_median_m) for row in gt])
    in_a = np.array([cm(row.a_median_m) for row in inp])
    in_b = np.array([cm(row.b_median_m) for row in inp])
    gt_f1 = np.array([row.f5 for row in gt])
    in_f1 = np.array([row.f5 for row in inp])

    def inv_norm(values: np.ndarray) -> np.ndarray:
        return 1 - (values - values.min()) / max(values.max() - values.min(), 1e-9)

    def norm(values: np.ndarray) -> np.ndarray:
        return (values - values.min()) / max(values.max() - values.min(), 1e-9)

    gt_quality = 0.32 * inv_norm(gt_a) + 0.32 * inv_norm(gt_b) + 0.36 * norm(gt_f1)
    preservation = 0.35 * inv_norm(in_a) + 0.35 * inv_norm(in_b) + 0.30 * norm(in_f1)
    lambdas = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    fig.suptitle("Utility Sensitivity Curve", fontsize=22, fontweight="bold", y=1.04)
    for idx, row in enumerate(gt):
        score = lambdas * gt_quality[idx] + (1 - lambdas) * preservation[idx]
        ax.plot(lambdas, score, color=COLORS[row.label], linewidth=2.8, label=row.label)
    ax.set_xlabel("weight on GT quality  (0 = preservation, 1 = GT quality)")
    ax.set_ylabel("normalized utility, higher is better")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    style_axes(ax, theme)
    save_figure(fig, stem)


def metric_boxplots(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    inp = split_rows(rows, "final_vs_objectx_input")
    data = [
        [cm(row.a_median_m) for row in gt],
        [cm(row.b_median_m) for row in gt],
        [cm(row.a_median_m) for row in inp],
        [cm(row.b_median_m) for row in inp],
    ]
    names = ["Pred->GT", "GT->Pred", "Final->Input", "Input->Final"]
    fig, ax = plt.subplots(figsize=(9.6, 6.2), constrained_layout=True)
    fig.suptitle("Aggregate Distance Distribution Across Configurations", fontsize=22, fontweight="bold", y=1.04)
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=names,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#D00000",
            "markeredgecolor": theme["edge"],
            "markersize": 6,
        },
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#C8D7E6")
        patch.set_edgecolor(theme["edge"])
        patch.set_linewidth(1.2)
    for key in ["whiskers", "caps", "medians", "means"]:
        for artist in bp[key]:
            artist.set_color(theme["edge"])
            artist.set_linewidth(1.2)
    ax.set_ylabel("cm, lower is better")
    style_axes(ax, theme)
    save_figure(fig, stem)


def dataset_method_delta(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    gt = split_rows(rows, "final_vs_gt")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), constrained_layout=True)
    fig.suptitle("Method Delta by Dataset", fontsize=22, fontweight="bold", y=1.04)
    metrics = [
        ("F1@5cm", [row.f5 for row in gt], "higher is better", False),
        ("GT->Pred median error", [cm(row.b_median_m) for row in gt], "cm, lower is better", True),
    ]
    dataset_colors = {"3RScan": "#0057B8", "ScanNet": "#D00000"}
    for ax, (title, vals, ylabel, lower_better) in zip(axes, metrics):
        for dataset in ["3RScan", "ScanNet"]:
            ds_rows = [row for row in gt if row.dataset == dataset]
            x = [0, 1]
            y = [vals[gt.index(row)] for row in ds_rows]
            ax.plot(x, y, marker="o", linewidth=2.8, markersize=8, color=dataset_colors[dataset], label=dataset)
            for xx, yy, row in zip(x, y, ds_rows):
                ax.text(xx, yy, row.pipeline.replace(" + ", "\n+ "), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_xticks([0, 1], ["SAM2 + MUSt3R", "SAMObject + Pi3X"], fontweight="bold")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        style_axes(ax, theme, grid="y")
        if lower_better:
            ax.invert_yaxis()
    axes[0].legend(loc="lower right")
    save_figure(fig, stem)


def small_multiples_score_lines(rows: list[MetricRow], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    comparisons = [("final_vs_gt", "Final vs GT"), ("final_vs_objectx_input", "Final vs input")]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), constrained_layout=True)
    fig.suptitle("Precision / Recall / F1 Profiles", fontsize=22, fontweight="bold", y=1.04)
    score_names = [("p5", "Precision"), ("r5", "Recall"), ("f5", "F1")]
    x = np.arange(3)
    for col, (comparison, comp_title) in enumerate(comparisons):
        sub = split_rows(rows, comparison)
        for row_idx, dataset in enumerate(["3RScan", "ScanNet"]):
            ax = axes[row_idx, col]
            for row in [r for r in sub if r.dataset == dataset]:
                vals = [getattr(row, key) for key, _ in score_names]
                ax.plot(x, vals, marker=METHOD_MARKERS[row.pipeline], linewidth=2.5, markersize=8, color=COLORS[row.label], label=row.pipeline)
                for xx, yy in zip(x, vals):
                    ax.text(xx, yy + 0.018, f"{yy:.2f}", ha="center", fontsize=8, fontweight="bold")
            ax.set_xticks(x, [name for _, name in score_names], fontweight="bold")
            ax.set_ylim(0.30, 1.02)
            ax.set_title(f"{dataset}: {comp_title}")
            style_axes(ax, theme, grid="y")
            if row_idx == 0 and col == 0:
                ax.legend(loc="lower right")
    save_figure(fig, stem)


def write_source_table(rows: list[MetricRow]) -> None:
    lines = [
        "# Object-X Final Plot Source",
        "",
        "Source: `evaluation/outputs/geometry/current_aggregate_tables.md`",
        "",
        "Rows are filtered to `best` group and only `final_vs_gt` plus `final_vs_objectx_input`.",
        "`final_vs_raw_pi3x` and `final_vs_raw_must3r` are intentionally excluded.",
        "Mean-error columns are intentionally excluded. Distances below are medians in meters.",
        "",
        "| Dataset | Pipeline | Comparison | Runs | A metric | A median error | B metric | B median error | Chamfer | P@5cm | R@5cm | F1@5cm |",
        "| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset,
                    row.pipeline,
                    row.comparison,
                    str(row.runs),
                    row.a_metric,
                    f"{row.a_median_m:.4f}",
                    row.b_metric,
                    f"{row.b_median_m:.4f}",
                    f"{row.chamfer_m:.4f}",
                    f"{row.p5:.3f}",
                    f"{row.r5:.3f}",
                    f"{row.f5:.3f}",
                ]
            )
            + " |"
        )
    SOURCE_TABLE.write_text("\n".join(lines) + "\n")


PLOT_DESCRIPTIONS = [
    ("01_overview_gt_dashboard", "GT dashboard", "Pred->GT, GT->Pred, and 5 cm scores against ground truth."),
    ("02_preservation_dashboard", "Input preservation dashboard", "Final output compared to Object-X input geometry."),
    ("03_combined_accuracy_preservation", "Combined summary", "Ground-truth quality and input preservation, with a Chamfer tradeoff scatter."),
    ("04_distance_heatmap", "Directional distance heatmap", "Pred->GT, GT->Pred, Final->Input, and Input->Final medians in centimeters."),
    ("05_score_heatmap", "Score heatmap", "Precision, recall, and F1 for GT and input comparisons."),
    ("06_radar_quality_net", "Radar / net diagram", "Normalized directional-error and score profile."),
    ("07_precision_recall_scatter", "Precision-recall scatter", "Operating points with F1 encoded by bubble size."),
    ("08_directional_error_tradeoff", "Directional error tradeoff", "Final->Input preservation against Pred->GT accuracy."),
    ("09_distance_profile_lines", "Distance line graph", "Median error profiles across four directions."),
    ("10_f1_slopegraph", "F1 slope graph", "F1 shift from GT quality to input preservation."),
    ("11_parallel_coordinates", "Parallel coordinates", "Normalized multi-metric performance profile."),
    ("12_lollipop_directional_errors", "Directional lollipop", "Pred->GT and GT->Pred medians per configuration."),
    ("13_rank_summary", "Balanced rank plot", "Normalized utility rank using medians and F1."),
    ("14_utility_sensitivity_curves", "Utility function graph", "Sensitivity curve over GT-quality vs preservation weighting."),
    ("15_metric_boxplots", "Boxplots", "Directional median-error distribution across configurations."),
    ("16_dataset_method_delta", "Dataset method delta", "Method changes within each dataset."),
    ("17_score_small_multiples", "Score small multiples", "P/R/F1 profiles split by dataset and comparison."),
    ("18_overview_gt_dashboard_dark", "Dark GT dashboard", "High-contrast variant of the GT dashboard."),
]


def write_readme_and_index() -> None:
    readme = [
        "# Object-X Final Comparison Plots",
        "",
        f"Source table: `{SOURCE_TABLE.relative_to(REPO_ROOT)}`",
        "",
        "All plots are generated from `current_aggregate_tables.md`.",
        "Raw comparison rows and mean-error columns are excluded.",
        "Most plots use directed median-error columns; the combined dashboard scatter uses Chamfer.",
        "Distances are displayed in centimeters unless noted otherwise.",
        "",
    ]
    for stem, title, desc in PLOT_DESCRIPTIONS:
        readme.append(f"- `{stem}.png`: {title} - {desc}")
    (OUT_DIR / "README.md").write_text("\n".join(readme) + "\n")

    cards = "\n".join(
        f"""
        <article class="card {'wide' if idx < 6 else ''}">
          <div class="card-head">
            <div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
            <span class="badge">{idx + 1:02d}</span>
          </div>
          <figure><img src="{stem}.png" alt="{title}"></figure>
          <div class="links">
            <a href="{stem}.png">PNG</a>
            <a href="{stem}.svg">SVG</a>
            <a href="{stem}.pdf">PDF</a>
          </div>
        </article>
        """
        for idx, (stem, title, desc) in enumerate(PLOT_DESCRIPTIONS)
    )
    legend_rows = "\n".join(
        f'<div class="legend-row"><span class="dot" style="background:{COLORS[f"{dataset} / {pipeline}"]}"></span>{dataset} / {pipeline}</div>'
        for dataset, pipeline in ORDER
    )
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Object-X Final Geometry Plots</title>
  <style>
    :root {{
      --blue: #2364ad;
      --navy: #0d1726;
      --panel: #ffffff;
      --text: #17202c;
      --muted: #596574;
      --line: #d6dee8;
      --shadow: 0 18px 52px rgba(10, 31, 56, 0.22);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 6%, rgba(255,255,255,0.24), transparent 24rem),
        linear-gradient(135deg, #2364ad 0%, #174f96 54%, #0d2f63 100%);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      min-height: 100vh;
    }}
    main {{ width: min(1560px, calc(100vw - 48px)); margin: 0 auto; padding: 34px 0 58px; }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(330px, 0.75fr);
      gap: 22px;
      margin-bottom: 24px;
    }}
    .hero, .legend, .card {{
      border: 1px solid rgba(255,255,255,0.30);
      background: rgba(255,255,255,0.96);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero, .legend {{ padding: 26px; }}
    h1 {{ margin: 0 0 12px; font-size: clamp(2.1rem, 4vw, 4.6rem); letter-spacing: -0.06em; line-height: 0.95; }}
    .subtitle {{ margin: 0; color: var(--muted); font-size: 1.04rem; line-height: 1.55; max-width: 900px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }}
    .chip {{ border-radius: 999px; padding: 8px 12px; background: #eef4fb; border: 1px solid var(--line); font-weight: 800; }}
    .legend h2 {{ margin: 0 0 14px; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted); }}
    .legend-grid {{ display: grid; gap: 10px; }}
    .legend-row {{ display: flex; align-items: center; gap: 10px; font-weight: 800; }}
    .dot {{ width: 14px; height: 14px; border-radius: 50%; box-shadow: 0 0 0 4px rgba(35,100,173,0.10); }}
    .note {{ margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }}
    .card {{ overflow: hidden; }}
    .card.wide {{ grid-column: 1 / -1; }}
    .card-head {{ display: flex; justify-content: space-between; gap: 18px; padding: 20px 22px 12px; }}
    .card h3 {{ margin: 0 0 6px; font-size: 1.18rem; letter-spacing: -0.02em; }}
    .card p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .badge {{ flex: none; padding: 7px 10px; border-radius: 999px; background: var(--blue); color: white; font-weight: 900; font-size: 0.78rem; }}
    figure {{ margin: 0; padding: 0 16px 16px; background: white; }}
    img {{ display: block; width: 100%; border-radius: 14px; border: 1px solid #e5edf6; }}
    .links {{ display: flex; gap: 10px; padding: 0 22px 20px; }}
    a {{ color: #174f96; font-weight: 900; text-decoration: none; }}
    @media (max-width: 980px) {{ header, .grid {{ grid-template-columns: 1fr; }} .card.wide {{ grid-column: auto; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <section class="hero">
        <h1>Object-X Final Geometry</h1>
        <p class="subtitle">A broader plot suite for the final decoded geometry. It uses robust median aggregates only, excludes raw comparison rows, and focuses on directed final-vs-GT plus final-vs-input behaviour with a Chamfer tradeoff view in the combined dashboard.</p>
        <div class="chips">
          <span class="chip">Median errors only</span>
          <span class="chip">No raw rows</span>
          <span class="chip">Directed errors + Chamfer scatter</span>
          <span class="chip">No gridlines</span>
          <span class="chip">Heatmaps, radar, scatter, line and utility plots</span>
        </div>
      </section>
      <aside class="legend">
        <h2>Color coding</h2>
        <div class="legend-grid">
          {legend_rows}
        </div>
        <p class="note">SVG and PDF versions are generated for poster layout. PNGs are quick-preview versions.</p>
      </aside>
    </header>
    <section class="grid">
      {cards}
    </section>
  </main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(index)


def main() -> None:
    rows = parse_summary(SUMMARY_PATH)
    clean_out_dir()
    write_source_table(rows)
    overview_dashboard(rows, THEME, "01_overview_gt_dashboard")
    preservation_dashboard(rows, THEME, "02_preservation_dashboard")
    combined_dashboard(rows, THEME, "03_combined_accuracy_preservation")
    distance_heatmap(rows, THEME, "04_distance_heatmap")
    score_heatmap(rows, THEME, "05_score_heatmap")
    radar_quality_net(rows, THEME, "06_radar_quality_net")
    precision_recall_scatter(rows, THEME, "07_precision_recall_scatter")
    directional_error_tradeoff(rows, THEME, "08_directional_error_tradeoff")
    distance_profile_lines(rows, THEME, "09_distance_profile_lines")
    f1_slopegraph(rows, THEME, "10_f1_slopegraph")
    parallel_coordinates(rows, THEME, "11_parallel_coordinates")
    lollipop_directional_errors(rows, THEME, "12_lollipop_directional_errors")
    rank_summary(rows, THEME, "13_rank_summary")
    utility_sensitivity_curves(rows, THEME, "14_utility_sensitivity_curves")
    metric_boxplots(rows, THEME, "15_metric_boxplots")
    dataset_method_delta(rows, THEME, "16_dataset_method_delta")
    small_multiples_score_lines(rows, THEME, "17_score_small_multiples")
    overview_dashboard(rows, DARK_THEME, "18_overview_gt_dashboard_dark")
    write_readme_and_index()
    print(f"Wrote {len(PLOT_DESCRIPTIONS)} Object-X final plot sets to {OUT_DIR}")


if __name__ == "__main__":
    main()
