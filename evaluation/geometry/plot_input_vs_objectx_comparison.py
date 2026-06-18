#!/usr/bin/env python3
"""Create comparison plots for input geometry vs Object-X output geometry.

The source is ``current_aggregate_tables.md``.  The plots compare each input
geometry backend against the corresponding final Object-X output, always
relative to ground truth:

- MUSt3R input -> SAM2 + MUSt3R Object-X output
- Pi3X input -> SAMObject + Pi3X Object-X output
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "evaluation" / "outputs" / "geometry" / "current_aggregate_tables.md"
OUT_DIR = REPO_ROOT / "evaluation" / "outputs" / "geometry" / "plots" / "input_vs_objectx_comparison"
SOURCE_TABLE = OUT_DIR / "input_vs_objectx_plot_source.md"

ORDER = [
    ("3RScan", "SAM2 + MUSt3R"),
    ("3RScan", "SAMObject + Pi3X"),
    ("ScanNet", "SAM2 + MUSt3R"),
    ("ScanNet", "SAMObject + Pi3X"),
]

INPUT_METHOD_FOR_PIPELINE = {
    "SAM2 + MUSt3R": "MUSt3R",
    "SAMObject + Pi3X": "Pi3X",
}

SHORT_LABELS = {
    ("3RScan", "SAM2 + MUSt3R"): "3RScan\nSAM2 + M3R",
    ("3RScan", "SAMObject + Pi3X"): "3RScan\nSAMObj + Pi3X",
    ("ScanNet", "SAM2 + MUSt3R"): "ScanNet\nSAM2 + M3R",
    ("ScanNet", "SAMObject + Pi3X"): "ScanNet\nSAMObj + Pi3X",
}

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
    "good": "#00843D",
    "bad": "#D00000",
}


@dataclass(frozen=True)
class PairRow:
    dataset: str
    pipeline: str
    input_runs: int
    output_runs: int
    input_pred_m: float
    input_gt_m: float
    input_chamfer_m: float
    input_p5: float
    input_r5: float
    input_f5: float
    output_pred_m: float
    output_gt_m: float
    output_chamfer_m: float
    output_p5: float
    output_r5: float
    output_f5: float

    @property
    def label(self) -> str:
        return f"{self.dataset} / {self.pipeline}"

    @property
    def short_label(self) -> str:
        return SHORT_LABELS[(self.dataset, self.pipeline)]


def _parse_float(value: str) -> float:
    return float(value) if value and value != "n/a" else float("nan")


def parse_input_best(lines: list[str]) -> dict[tuple[str, str], dict[str, float | int]]:
    rows: dict[tuple[str, str], dict[str, float | int]] = {}
    in_best = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("## "):
            in_best = line == "## Input Geometry Best-40 Error Median Comparison"
            continue
        if not in_best or not line.startswith("|") or "---" in line or "Dataset" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 11:
            continue
        dataset, method = parts[0], parts[1]
        if dataset not in {"3RScan", "ScanNet"}:
            continue
        rows[(dataset, method)] = {
            "runs": int(parts[2]),
            "pred": _parse_float(parts[4]),
            "gt": _parse_float(parts[6]),
            "chamfer": _parse_float(parts[7]),
            "p5": _parse_float(parts[8]),
            "r5": _parse_float(parts[9]),
            "f5": _parse_float(parts[10]),
        }
    return rows


def _normalise_output_section(section: str) -> tuple[str, str] | None:
    section = section.replace("Object-X Final:", "").strip()
    if section.startswith("ScanNet "):
        return "ScanNet", section.removeprefix("ScanNet ").strip()
    if section.startswith("3RScan:"):
        return "3RScan", section.removeprefix("3RScan:").strip()
    return None


def parse_output_gt(lines: list[str]) -> dict[tuple[str, str], dict[str, float | int]]:
    rows: dict[tuple[str, str], dict[str, float | int]] = {}
    current: tuple[str, str] | None = None
    for raw in lines:
        line = raw.strip()
        if line.startswith("## "):
            current = _normalise_output_section(line.removeprefix("## ").strip())
            continue
        if current is None or not line.startswith("|") or "---" in line or "Group" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 13 or parts[0] != "best" or parts[1] != "final_vs_gt":
            continue
        rows[current] = {
            "runs": int(parts[2]),
            "pred": _parse_float(parts[5]),
            "gt": _parse_float(parts[8]),
            "chamfer": _parse_float(parts[9]),
            "p5": _parse_float(parts[10]),
            "r5": _parse_float(parts[11]),
            "f5": _parse_float(parts[12]),
        }
    return rows


def parse_summary(path: Path = SUMMARY_PATH) -> list[PairRow]:
    lines = path.read_text().splitlines()
    input_rows = parse_input_best(lines)
    output_rows = parse_output_gt(lines)
    pairs: list[PairRow] = []
    for dataset, pipeline in ORDER:
        method = INPUT_METHOD_FOR_PIPELINE[pipeline]
        inp = input_rows[(dataset, method)]
        out = output_rows[(dataset, pipeline)]
        pairs.append(
            PairRow(
                dataset=dataset,
                pipeline=pipeline,
                input_runs=int(inp["runs"]),
                output_runs=int(out["runs"]),
                input_pred_m=float(inp["pred"]),
                input_gt_m=float(inp["gt"]),
                input_chamfer_m=float(inp["chamfer"]),
                input_p5=float(inp["p5"]),
                input_r5=float(inp["r5"]),
                input_f5=float(inp["f5"]),
                output_pred_m=float(out["pred"]),
                output_gt_m=float(out["gt"]),
                output_chamfer_m=float(out["chamfer"]),
                output_p5=float(out["p5"]),
                output_r5=float(out["r5"]),
                output_f5=float(out["f5"]),
            )
        )
    return pairs


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": THEME["bg"],
            "axes.facecolor": THEME["panel"],
            "axes.edgecolor": THEME["grid"],
            "axes.labelcolor": THEME["text"],
            "xtick.color": THEME["muted"],
            "ytick.color": THEME["muted"],
            "text.color": THEME["text"],
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.facecolor": THEME["bg"],
            "savefig.edgecolor": THEME["bg"],
        }
    )


def clean_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".svg", ".pdf", ".html", ".md"}:
            path.unlink()


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.16)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def cm(value_m: float) -> float:
    return value_m * 100.0


def row_colors(rows: list[PairRow]) -> list[str]:
    return [COLORS[row.label] for row in rows]


def style_axes(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(THEME["grid"])
    ax.tick_params(axis="both", colors=THEME["muted"])
    ax.grid(False)
    if grid == "x":
        ax.grid(axis="x", color=THEME["grid"], alpha=0.45, linewidth=0.8)
    if grid == "y":
        ax.grid(axis="y", color=THEME["grid"], alpha=0.45, linewidth=0.8)


def contrast_text(hex_color: str) -> str:
    rgb = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#111111" if luminance > 0.52 else "#FFFFFF"


def paired_error_bars(rows: list[PairRow], stem: str) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=True)
    fig.suptitle("Input Geometry vs Object-X Output: Distance to Ground Truth", fontsize=23, fontweight="bold", y=1.04)
    panels = [
        ("Pred -> GT median error", "pred", [row.input_pred_m for row in rows], [row.output_pred_m for row in rows]),
        ("GT -> Pred median error", "gt", [row.input_gt_m for row in rows], [row.output_gt_m for row in rows]),
    ]
    for ax, (title, _, input_vals, output_vals) in zip(axes, panels):
        y = np.arange(len(rows))
        height = 0.34
        input_cm = np.array([cm(value) for value in input_vals])
        output_cm = np.array([cm(value) for value in output_vals])
        xmax = float(max(input_cm.max(), output_cm.max()) * 1.25)
        ax.barh(y - height / 2, input_cm, height, color="white", edgecolor=row_colors(rows), linewidth=2.0, label="Input")
        ax.barh(y + height / 2, output_cm, height, color=row_colors(rows), edgecolor=THEME["edge"], linewidth=0.8, label="Object-X output")
        for yy, value in zip(y - height / 2, input_cm):
            ax.text(value + xmax * 0.018, yy, f"{value:.1f}", va="center", fontsize=9.4, fontweight="bold")
        for yy, value in zip(y + height / 2, output_cm):
            ax.text(value + xmax * 0.018, yy, f"{value:.1f}", va="center", fontsize=9.4, fontweight="bold")
        ax.set_yticks(y, [row.short_label for row in rows], fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel("cm, lower is better")
        ax.set_title(title)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=2,
        borderaxespad=0.0,
        fontsize=10.5,
    )
    save_figure(fig, stem)


def chamfer_f1_scatter(rows: list[PairRow], stem: str) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9.4, 7.0), constrained_layout=True)
    fig.suptitle("GT Quality Tradeoff: Chamfer vs F1", fontsize=22, fontweight="bold", y=1.04)
    label_offsets = {
        "3RScan / SAM2 + MUSt3R": (0.18, -0.010),
        "3RScan / SAMObject + Pi3X": (0.22, -0.010),
        "ScanNet / SAM2 + MUSt3R": (0.26, 0.018),
        "ScanNet / SAMObject + Pi3X": (0.18, 0.018),
    }
    for row, color in zip(rows, row_colors(rows)):
        x0, y0 = cm(row.input_chamfer_m), row.input_f5
        x1, y1 = cm(row.output_chamfer_m), row.output_f5
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.2, "shrinkA": 7, "shrinkB": 7},
            zorder=2,
        )
        ax.scatter(x0, y0, s=190, facecolor="white", edgecolor=color, linewidth=2.0, marker=METHOD_MARKERS[row.pipeline], zorder=3)
        ax.scatter(x1, y1, s=230, color=color, edgecolor=THEME["edge"], linewidth=1.0, marker=METHOD_MARKERS[row.pipeline], zorder=4)
        dx, dy = label_offsets[row.label]
        ax.annotate(
            row.short_label.replace("\n", " / "),
            xy=(x1, y1),
            xytext=(x1 + dx, y1 + dy),
            textcoords="data",
            va="center",
            ha="left",
            fontsize=8.6,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": THEME["muted"], "lw": 0.75, "alpha": 0.65},
            zorder=5,
        )
    ax.set_xlabel("Chamfer from median errors (cm, lower is better)")
    ax.set_ylabel("F1@5cm against GT (higher is better)")
    ax.text(5.75, 0.69, "white = input\nsolid = Object-X output", fontsize=10, color=THEME["muted"], fontweight="bold")
    all_x = [cm(v) for row in rows for v in (row.input_chamfer_m, row.output_chamfer_m)]
    all_y = [v for row in rows for v in (row.input_f5, row.output_f5)]
    ax.set_xlim(max(0.0, min(all_x) - 0.8), max(all_x) + 2.5)
    ax.set_ylim(max(0.0, min(all_y) - 0.08), min(1.0, max(all_y) + 0.10))
    style_axes(ax)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=THEME["edge"], markersize=8, label="Input"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=THEME["blue"], markeredgecolor=THEME["edge"], markersize=8, label="Object-X output"),
    ]
    ax.legend(handles=legend, loc="lower right")
    save_figure(fig, stem)


def error_delta_diverging(rows: list[PairRow], stem: str) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.5), constrained_layout=True)
    fig.suptitle("Object-X Change Relative to Input Geometry", fontsize=22, fontweight="bold", y=1.04)
    panels = [
        ("Pred -> GT median error", [cm(row.output_pred_m - row.input_pred_m) for row in rows], "cm change"),
        ("GT -> Pred median error", [cm(row.output_gt_m - row.input_gt_m) for row in rows], "cm change"),
        ("F1@5cm", [row.output_f5 - row.input_f5 for row in rows], "score change"),
    ]
    for ax, (title, values, xlabel) in zip(axes, panels):
        values_arr = np.array(values)
        y = np.arange(len(rows))
        bar_colors = [THEME["good"] if value <= 0 and "F1" not in title else THEME["bad"] for value in values]
        if "F1" in title:
            bar_colors = [THEME["good"] if value >= 0 else THEME["bad"] for value in values]
        ax.axvline(0, color=THEME["edge"], linewidth=1.0)
        ax.barh(y, values_arr, color=bar_colors, edgecolor=THEME["edge"], linewidth=0.8)
        pad = max(abs(values_arr).max() * 0.16, 0.12 if "F1" not in title else 0.015)
        for yy, value in zip(y, values):
            label = f"{value:+.1f} cm" if "F1" not in title else f"{value:+.2f}"
            ax.text(value + (pad if value >= 0 else -pad), yy, label, va="center", ha="left" if value >= 0 else "right", fontsize=9, fontweight="bold")
        ax.set_yticks(y, [row.short_label for row in rows], fontweight="bold")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        lim = abs(values_arr).max() + pad * 2.0
        ax.set_xlim(-lim, lim)
        style_axes(ax)
    save_figure(fig, stem)


def score_paired_bars(rows: list[PairRow], stem: str) -> None:
    setup_style()
    metrics = [
        ("Precision", "p5", [row.input_p5 for row in rows], [row.output_p5 for row in rows]),
        ("Recall", "r5", [row.input_r5 for row in rows], [row.output_r5 for row in rows]),
        ("F1", "f5", [row.input_f5 for row in rows], [row.output_f5 for row in rows]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.5), constrained_layout=True)
    fig.suptitle("5 cm Scores: Input Geometry vs Object-X Output", fontsize=22, fontweight="bold", y=1.04)
    for ax, (title, _, input_vals, output_vals) in zip(axes, metrics):
        x = np.arange(len(rows))
        width = 0.34
        ax.bar(x - width / 2, input_vals, width, color="white", edgecolor=row_colors(rows), linewidth=2.0, label="Input")
        ax.bar(x + width / 2, output_vals, width, color=row_colors(rows), edgecolor=THEME["edge"], linewidth=0.8, label="Object-X")
        for xx, value in zip(x - width / 2, input_vals):
            ax.text(xx, value + 0.018, f"{value:.2f}", ha="center", fontsize=8.8, fontweight="bold")
        for xx, value in zip(x + width / 2, output_vals):
            ax.text(xx, value + 0.018, f"{value:.2f}", ha="center", fontsize=8.8, fontweight="bold")
        ax.set_xticks(x, [row.short_label for row in rows], rotation=0, fontweight="bold")
        ax.set_ylim(0, 1.04)
        ax.set_title(title)
        style_axes(ax, grid="y")
    axes[0].set_ylabel("score, higher is better")
    axes[0].legend(loc="upper left")
    save_figure(fig, stem)


def directional_arrow_field(rows: list[PairRow], stem: str) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9.4, 7.2), constrained_layout=True)
    fig.suptitle("Directional Error Movement Against Ground Truth", fontsize=22, fontweight="bold", y=1.04)
    for row, color in zip(rows, row_colors(rows)):
        x0, y0 = cm(row.input_pred_m), cm(row.input_gt_m)
        x1, y1 = cm(row.output_pred_m), cm(row.output_gt_m)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.3, "shrinkA": 7, "shrinkB": 7},
            zorder=2,
        )
        ax.scatter(x0, y0, s=190, facecolor="white", edgecolor=color, linewidth=2.0, marker=METHOD_MARKERS[row.pipeline], zorder=3)
        ax.scatter(x1, y1, s=230, color=color, edgecolor=THEME["edge"], linewidth=1.0, marker=METHOD_MARKERS[row.pipeline], zorder=4)
        ax.text(x1 + 0.18, y1, row.short_label.replace("\n", " / "), va="center", fontsize=8.7, fontweight="bold")
    all_x = [cm(v) for row in rows for v in (row.input_pred_m, row.output_pred_m)]
    all_y = [cm(v) for row in rows for v in (row.input_gt_m, row.output_gt_m)]
    ax.set_xlim(max(0.0, min(all_x) - 0.9), max(all_x) + 2.2)
    ax.set_ylim(max(0.0, min(all_y) - 0.8), max(all_y) + 1.8)
    ax.set_xlabel("Pred -> GT median error (cm, lower is better)")
    ax.set_ylabel("GT -> Pred median error (cm, lower is better)")
    ax.text(ax.get_xlim()[0] + 0.1, ax.get_ylim()[0] + 0.12, "lower left is best", fontsize=10, color=THEME["muted"], fontweight="bold")
    style_axes(ax)
    save_figure(fig, stem)


def metric_heatmap(rows: list[PairRow], stem: str) -> None:
    setup_style()
    labels = [row.short_label for row in rows]
    cols = ["Input\nPred->GT", "Output\nPred->GT", "Input\nGT->Pred", "Output\nGT->Pred", "Input\nF1", "Output\nF1"]
    data = np.array(
        [
            [cm(row.input_pred_m), cm(row.output_pred_m), cm(row.input_gt_m), cm(row.output_gt_m), row.input_f5, row.output_f5]
            for row in rows
        ]
    )
    fig, ax = plt.subplots(figsize=(13.5, 6.0), constrained_layout=True)
    fig.suptitle("Input vs Output Metric Matrix", fontsize=22, fontweight="bold", y=1.04)
    # Normalize columns independently so distance and score columns can share one heatmap.
    norm_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        vals = data[:, col]
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            norm_data[:, col] = 0.5
        elif col in {4, 5}:  # F1: higher is better, darker is better.
            norm_data[:, col] = (vals - lo) / (hi - lo)
        else:  # Distances: lower is better, darker is better.
            norm_data[:, col] = 1.0 - (vals - lo) / (hi - lo)
    image = ax.imshow(norm_data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            value = data[r, c]
            text = f"{value:.1f} cm" if c < 4 else f"{value:.2f}"
            ax.text(c, r, text, ha="center", va="center", fontweight="bold", color="#0F1720" if norm_data[r, c] < 0.58 else "white")
    ax.set_xticks(np.arange(len(cols)), cols, fontweight="bold")
    ax.set_yticks(np.arange(len(labels)), labels, fontweight="bold")
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_label("column-normalized quality")
    save_figure(fig, stem)


def score_slopegraph(rows: list[PairRow], stem: str) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.5), constrained_layout=True)
    fig.suptitle("Score Shift From Input to Object-X Output", fontsize=22, fontweight="bold", y=1.04)
    panels = [
        ("Precision", "input_p5", "output_p5"),
        ("Recall", "input_r5", "output_r5"),
        ("F1", "input_f5", "output_f5"),
    ]
    for ax, (title, in_key, out_key) in zip(axes, panels):
        for row, color in zip(rows, row_colors(rows)):
            y0 = getattr(row, in_key)
            y1 = getattr(row, out_key)
            ax.plot([0, 1], [y0, y1], color=color, linewidth=2.6, marker=METHOD_MARKERS[row.pipeline], markersize=8)
            ax.text(-0.04, y0, f"{y0:.2f}", ha="right", va="center", fontsize=8.8, fontweight="bold")
            ax.text(1.04, y1, f"{y1:.2f}", ha="left", va="center", fontsize=8.8, fontweight="bold")
        ax.set_xlim(-0.22, 1.22)
        ax.set_ylim(0.25, 1.02)
        ax.set_xticks([0, 1], ["Input", "Object-X"], fontweight="bold")
        ax.set_title(title)
        style_axes(ax, grid="y")
    axes[0].set_ylabel("score, higher is better")
    handles = [
        Line2D([0], [0], color=COLORS[row.label], marker=METHOD_MARKERS[row.pipeline], linewidth=2.4, label=row.short_label.replace("\n", " / "))
        for row in rows
    ]
    axes[1].legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    save_figure(fig, stem)


def compact_poster_dashboard(rows: list[PairRow], stem: str) -> None:
    setup_style()
    fig = plt.figure(figsize=(15.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.02])
    fig.suptitle("Input Geometry vs Object-X Output", fontsize=24, fontweight="bold", y=1.04)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    ax4 = fig.add_subplot(grid[1, :])

    y = np.arange(len(rows))
    height = 0.34
    pred_input = np.array([cm(row.input_pred_m) for row in rows])
    pred_output = np.array([cm(row.output_pred_m) for row in rows])
    gt_input = np.array([cm(row.input_gt_m) for row in rows])
    gt_output = np.array([cm(row.output_gt_m) for row in rows])
    for ax, title, input_vals, output_vals in [
        (ax1, "Pred -> GT", pred_input, pred_output),
        (ax2, "GT -> Pred", gt_input, gt_output),
    ]:
        xmax = float(max(input_vals.max(), output_vals.max()) * 1.25)
        ax.barh(y - height / 2, input_vals, height, color="white", edgecolor=row_colors(rows), linewidth=2.0)
        ax.barh(y + height / 2, output_vals, height, color=row_colors(rows), edgecolor=THEME["edge"], linewidth=0.8)
        for yy, value in zip(y - height / 2, input_vals):
            ax.text(value + xmax * 0.02, yy, f"{value:.1f}", va="center", fontsize=8.8, fontweight="bold")
        for yy, value in zip(y + height / 2, output_vals):
            ax.text(value + xmax * 0.02, yy, f"{value:.1f}", va="center", fontsize=8.8, fontweight="bold")
        ax.set_yticks(y, [row.short_label for row in rows], fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel("cm, lower is better")
        ax.set_title(title)
        style_axes(ax, grid="x")

    # F1 vs Chamfer scatter.
    for row, color in zip(rows, row_colors(rows)):
        x0, y0 = cm(row.input_chamfer_m), row.input_f5
        x1, y1 = cm(row.output_chamfer_m), row.output_f5
        ax3.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0, "shrinkA": 7, "shrinkB": 7})
        ax3.scatter(x0, y0, s=140, facecolor="white", edgecolor=color, linewidth=2.0, marker=METHOD_MARKERS[row.pipeline])
        ax3.scatter(x1, y1, s=170, color=color, edgecolor=THEME["edge"], linewidth=1.0, marker=METHOD_MARKERS[row.pipeline])
    ax3.set_xlabel("Chamfer (cm)")
    ax3.set_ylabel("F1@5cm")
    ax3.set_title("Quality movement")
    ax3.text(5.7, 0.70, "white=input\nsolid=output", fontsize=9.5, color=THEME["muted"], fontweight="bold")
    all_x = [cm(v) for row in rows for v in (row.input_chamfer_m, row.output_chamfer_m)]
    all_y = [v for row in rows for v in (row.input_f5, row.output_f5)]
    ax3.set_xlim(max(0.0, min(all_x) - 0.8), max(all_x) + 1.9)
    ax3.set_ylim(max(0.0, min(all_y) - 0.08), min(1.0, max(all_y) + 0.10))
    style_axes(ax3)

    metrics = [("Precision", "input_p5", "output_p5"), ("Recall", "input_r5", "output_r5"), ("F1", "input_f5", "output_f5")]
    x = np.arange(len(metrics))
    width = 0.16
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(rows))
    for row, color, offset in zip(rows, row_colors(rows), offsets):
        input_vals = [getattr(row, in_key) for _, in_key, _ in metrics]
        output_vals = [getattr(row, out_key) for _, _, out_key in metrics]
        ax4.bar(x + offset - width / 2, input_vals, width, color="white", edgecolor=color, linewidth=1.8)
        bars = ax4.bar(x + offset + width / 2, output_vals, width, color=color, edgecolor=THEME["edge"], linewidth=0.6, label=row.short_label.replace("\n", " / "))
        for bar, value in zip(bars, output_vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", fontsize=8.2, fontweight="bold")
    ax4.set_xticks(x, [name for name, _, _ in metrics], fontweight="bold")
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("score, higher is better")
    ax4.set_title("5 cm scores against GT")
    ax4.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    style_axes(ax4, grid="y")
    save_figure(fig, stem)


def write_source_table(rows: list[PairRow]) -> None:
    lines = [
        "# Input vs Object-X Plot Source",
        "",
        "Source: `evaluation/outputs/geometry/current_aggregate_tables.md`",
        "",
        "Input rows use the `Input Geometry Best-40 Error Median Comparison` section.",
        "Output rows use each Object-X `best/final_vs_gt` section.",
        "Distances are median errors in meters; Chamfer is the sum of the two displayed median directions.",
        "",
        "| Dataset | Pipeline | Input runs | Output runs | Input Pred->GT | Output Pred->GT | Input GT->Pred | Output GT->Pred | Input Chamfer | Output Chamfer | Input P@5 | Output P@5 | Input R@5 | Output R@5 | Input F1 | Output F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset,
                    row.pipeline,
                    str(row.input_runs),
                    str(row.output_runs),
                    f"{row.input_pred_m:.4f}",
                    f"{row.output_pred_m:.4f}",
                    f"{row.input_gt_m:.4f}",
                    f"{row.output_gt_m:.4f}",
                    f"{row.input_chamfer_m:.4f}",
                    f"{row.output_chamfer_m:.4f}",
                    f"{row.input_p5:.3f}",
                    f"{row.output_p5:.3f}",
                    f"{row.input_r5:.3f}",
                    f"{row.output_r5:.3f}",
                    f"{row.input_f5:.3f}",
                    f"{row.output_f5:.3f}",
                ]
            )
            + " |"
        )
    SOURCE_TABLE.write_text("\n".join(lines) + "\n")


PLOT_DESCRIPTIONS = [
    ("01_paired_directional_errors", "Directional paired bars", "Input and Object-X output Pred->GT / GT->Pred medians."),
    ("02_chamfer_f1_scatter", "Chamfer vs F1 movement", "Arrows show how each configuration moves from input to final output."),
    ("03_error_delta_diverging", "Delta bars", "Output minus input deltas for directed errors and F1."),
    ("04_score_paired_bars", "Score paired bars", "Precision, recall, and F1 against GT before and after Object-X."),
    ("05_directional_arrow_field", "Directional error arrows", "Movement in Pred->GT / GT->Pred error space."),
    ("06_metric_heatmap", "Metric matrix", "Input/output distance and F1 values in one heatmap."),
    ("07_score_slopegraph", "Score slopegraph", "Precision, recall, and F1 shifts from input to Object-X."),
    ("08_compact_poster_dashboard", "Poster dashboard", "Compact combined view of distances, Chamfer/F1, and scores."),
]


def write_readme_and_index() -> None:
    readme = [
        "# Input vs Object-X Geometry Comparison Plots",
        "",
        f"Source table: `{SOURCE_TABLE.relative_to(REPO_ROOT)}`",
        "",
        "These plots compare input geometry against the final Object-X output, always relative to ground truth.",
        "The visual language matches the Object-X final comparison plots.",
        "",
    ]
    for stem, title, desc in PLOT_DESCRIPTIONS:
        readme.append(f"- `{stem}.png`: {title} - {desc}")
    (OUT_DIR / "README.md").write_text("\n".join(readme) + "\n")

    cards = "\n".join(
        f"""
        <article class="card {'wide' if idx in {0, 7} else ''}">
          <div class="card-head">
            <div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
            <span>{idx + 1:02d}</span>
          </div>
          <figure><img src="{stem}.png" alt="{title}"></figure>
          <div class="links"><a href="{stem}.png">PNG</a><a href="{stem}.svg">SVG</a><a href="{stem}.pdf">PDF</a></div>
        </article>
        """
        for idx, (stem, title, desc) in enumerate(PLOT_DESCRIPTIONS)
    )
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Input vs Object-X Geometry Plots</title>
  <style>
    :root {{ --blue:#2364ad; --text:#17202c; --muted:#596574; --line:#d6dee8; --panel:#fff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--text); background:linear-gradient(135deg,#2364ad,#0d2f63); }}
    main {{ width:min(1540px,calc(100vw - 48px)); margin:0 auto; padding:34px 0 58px; }}
    header,.card {{ background:rgba(255,255,255,.96); border:1px solid rgba(255,255,255,.35); border-radius:24px; box-shadow:0 18px 52px rgba(10,31,56,.22); }}
    header {{ padding:28px; margin-bottom:24px; }}
    h1 {{ margin:0 0 10px; font-size:clamp(2rem,4vw,4.4rem); letter-spacing:-.055em; line-height:.95; }}
    .subtitle {{ margin:0; max-width:920px; color:var(--muted); line-height:1.5; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; }}
    .card {{ overflow:hidden; padding-bottom:14px; }}
    .card.wide {{ grid-column:1/-1; }}
    .card-head {{ display:flex; justify-content:space-between; gap:16px; padding:20px 22px 12px; }}
    .card h3 {{ margin:0 0 6px; font-size:1.18rem; }}
    .card p {{ margin:0; color:var(--muted); }}
    .card span {{ align-self:flex-start; color:white; background:var(--blue); border-radius:999px; padding:9px 12px; font-weight:900; }}
    figure {{ margin:0; padding:0 16px; }}
    img {{ width:100%; display:block; border-radius:14px; border:1px solid var(--line); background:#f4f7fb; }}
    .links {{ display:flex; gap:12px; padding:12px 20px 0; font-weight:900; }}
    .links a {{ color:var(--blue); text-decoration:none; }}
    @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} .card.wide {{ grid-column:auto; }} main {{ width:min(100vw - 24px,1540px); }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Input vs Object-X Geometry</h1>
    <p class="subtitle">Poster-ready comparisons of the input geometry and final Object-X output against ground truth. White marks input geometry; filled marks final Object-X output.</p>
  </header>
  <section class="grid">{cards}</section>
</main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(index)


def main() -> None:
    rows = parse_summary()
    clean_out_dir()
    write_source_table(rows)
    paired_error_bars(rows, "01_paired_directional_errors")
    chamfer_f1_scatter(rows, "02_chamfer_f1_scatter")
    error_delta_diverging(rows, "03_error_delta_diverging")
    score_paired_bars(rows, "04_score_paired_bars")
    directional_arrow_field(rows, "05_directional_arrow_field")
    metric_heatmap(rows, "06_metric_heatmap")
    score_slopegraph(rows, "07_score_slopegraph")
    compact_poster_dashboard(rows, "08_compact_poster_dashboard")
    write_readme_and_index()
    print(f"Wrote {len(PLOT_DESCRIPTIONS)} input-vs-Object-X plot sets to {OUT_DIR}")


if __name__ == "__main__":
    main()
