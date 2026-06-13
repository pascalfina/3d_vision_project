#!/usr/bin/env python3
"""Create poster-ready plots for the input geometry median comparison table."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "evaluation" / "outputs" / "geometry" / "input_geometry_median_comparison.md"
OUT_DIR = REPO_ROOT / "evaluation" / "outputs" / "geometry" / "plots" / "input_geometry_comparison"

METHOD_COLORS = {
    "MUSt3R": "#44C2FF",
    "Pi3X": "#FFB454",
}

CONFIG_COLORS = {
    "3RScan / MUSt3R": "#42C6FF",
    "3RScan / Pi3X": "#F6A742",
    "ScanNet / MUSt3R": "#53D6A2",
    "ScanNet / Pi3X": "#F05D7E",
}

DARK = {
    "bg": "#111317",
    "panel": "#171A20",
    "grid": "#343A46",
    "text": "#F0F2F5",
    "muted": "#B8BEC9",
    "edge": "#F0F2F5",
}

LIGHT = {
    "bg": "#F7F3EA",
    "panel": "#FFFDF8",
    "grid": "#D9D1C4",
    "text": "#1B1C1E",
    "muted": "#626872",
    "edge": "#1B1C1E",
}

POSTER_BLUE = {
    "bg": "#2364AD",
    "panel": "#FFFFFF",
    "grid": "#C9D6E6",
    "text": "#15181D",
    "muted": "#5C6676",
    "edge": "#112E4F",
    "outer": "#FFFFFF",
}


def parse_summary(path: Path) -> list[dict[str, object]]:
    preferred_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("|") or "---" in line or "Dataset" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) == 11:
            dataset, method, runs, pred_mean, pred_med, gt_mean, gt_med, chamfer, p5, r5, f5 = parts
            target = preferred_rows
        elif len(parts) == 10:
            dataset, method, runs, pred_mean, pred_med, gt_mean, gt_med, p5, r5, f5 = parts
            chamfer = "nan"
            target = fallback_rows
        else:
            continue
        if dataset not in {"3RScan", "ScanNet"}:
            continue
        target.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": int(runs),
                "pred_mean_m": float(pred_mean),
                "pred_median_m": float(pred_med),
                "gt_mean_m": float(gt_mean),
                "gt_median_m": float(gt_med),
                "chamfer_m": float(chamfer),
                "p5": float(p5),
                "r5": float(r5),
                "f5": float(f5),
                "label": f"{dataset} / {method}",
            }
        )
    rows = preferred_rows if len(preferred_rows) == 4 else fallback_rows
    if len(rows) != 4:
        raise RuntimeError(f"Expected 4 rows in {path}, found {len(rows)}")
    return rows


def _cm_axis_limit(rows: list[dict[str, object]], key: str) -> float:
    max_value = max(float(row[key]) * 100 for row in rows)
    return max(1.0, float(np.ceil(max_value * 1.32 * 2) / 2))


def setup_style(theme: dict[str, str]) -> None:
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
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def style_outer_axis(ax: plt.Axes, theme: dict[str, str]) -> None:
    outer = theme.get("outer")
    if not outer:
        return
    ax.tick_params(axis="both", colors=outer)
    ax.xaxis.label.set_color(outer)
    ax.yaxis.label.set_color(outer)
    ax.title.set_color(outer)


def style_outer_legend(legend: object, theme: dict[str, str]) -> None:
    outer = theme.get("outer")
    if not outer or legend is None:
        return
    for text in legend.get_texts():
        text.set_color(outer)


def grouped_error_bars(rows: list[dict[str, object]], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    datasets = ["3RScan", "ScanNet"]
    methods = ["MUSt3R", "Pi3X"]
    metrics = [
        ("pred_median_m", "Pred -> GT median error", "Predicted geometry to reference", _cm_axis_limit(rows, "pred_median_m")),
        ("gt_median_m", "GT -> Pred median error", "Reference coverage by prediction", _cm_axis_limit(rows, "gt_median_m")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    fig.suptitle("Input Geometry: Median Error Comparison", fontsize=21, fontweight="bold", y=1.04, color=theme.get("outer", theme["text"]))

    for ax, (key, title, subtitle, ylim) in zip(axes, metrics):
        x = np.arange(len(datasets))
        width = 0.33
        for i, method in enumerate(methods):
            values = [
                float(next(row for row in rows if row["dataset"] == dataset and row["method"] == method)[key]) * 100
                for dataset in datasets
            ]
            bars = ax.bar(
                x + (i - 0.5) * width,
                values,
                width,
                color=METHOD_COLORS[method],
                edgecolor=theme["edge"],
                linewidth=0.8,
                label=method,
                alpha=0.92,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ylim * 0.025,
                    f"{value:.1f} cm",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
        ax.set_title(title, pad=18)
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", color=theme["muted"], fontsize=10)
        ax.set_xticks(x, datasets, fontweight="bold")
        ax.set_ylim(0, ylim)
        ax.set_ylabel("error in cm (lower is better)")
        ax.grid(axis="y", color=theme["grid"], alpha=0.55, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        style_outer_axis(ax, theme)

    style_outer_legend(axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98)), theme)
    save_figure(fig, stem)


def score_bars(rows: list[dict[str, object]], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    metrics = [("p5", "Precision @ 5 cm"), ("r5", "Recall @ 5 cm"), ("f5", "F1 @ 5 cm")]
    labels = [str(row["label"]) for row in rows]
    colors = [CONFIG_COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(12.8, 6.0), constrained_layout=True)
    fig.suptitle("Input Geometry: 5 cm Quality Scores", fontsize=21, fontweight="bold", y=1.03, color=theme.get("outer", theme["text"]))

    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(rows))
    for row, color, offset in zip(rows, colors, offsets):
        values = [float(row[key]) for key, _ in metrics]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=str(row["label"]),
            color=color,
            edgecolor=theme["edge"],
            linewidth=0.7,
            alpha=0.94,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_xticks(x, [name for _, name in metrics], fontweight="bold")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("score (higher is better)")
    ax.grid(axis="y", color=theme["grid"], alpha=0.55, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    style_outer_axis(ax, theme)
    style_outer_legend(ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.0)), theme)
    save_figure(fig, stem)


def _draw_dashboard_error_panel(
    ax: plt.Axes,
    ordered: list[dict[str, object]],
    labels: list[str],
    colors: list[str],
    theme: dict[str, str],
    key: str,
    title: str,
    xmax: float,
) -> None:
    y = np.arange(len(ordered))
    values = np.array([float(row[key]) * 100 for row in ordered])
    ax.barh(y, values, color=colors, edgecolor=theme["edge"], linewidth=0.7, alpha=0.94)
    for yy, value in zip(y, values):
        ax.text(value + xmax * 0.025, yy, f"{value:.1f} cm", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y, labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("cm, lower is better")
    ax.set_title(title, pad=12)
    ax.grid(axis="x", color=theme["grid"], alpha=0.55, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    style_outer_axis(ax, theme)


def poster_dashboard(rows: list[dict[str, object]], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    fig = plt.figure(figsize=(14.4, 8.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_error_pred = fig.add_subplot(grid[0, 0])
    ax_error_gt = fig.add_subplot(grid[0, 1])
    ax_scores = fig.add_subplot(grid[1, :])
    fig.suptitle("Input Geometry Quality: 3RScan vs ScanNet", fontsize=23, fontweight="bold", y=1.075, color=theme.get("outer", theme["text"]))

    ordered = rows
    labels = [str(row["label"]).replace(" / ", "\n") for row in ordered]
    colors = [CONFIG_COLORS[str(row["label"])] for row in ordered]

    _draw_dashboard_error_panel(ax_error_pred, ordered, labels, colors, theme, "pred_median_m", "Pred -> GT median error", _cm_axis_limit(ordered, "pred_median_m"))
    _draw_dashboard_error_panel(ax_error_gt, ordered, labels, colors, theme, "gt_median_m", "GT -> Pred median error", _cm_axis_limit(ordered, "gt_median_m"))

    metrics = [("p5", "P@5cm"), ("r5", "R@5cm"), ("f5", "F1@5cm")]
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ordered))
    for row, color, offset in zip(ordered, colors, offsets):
        values = [float(row[key]) for key, _ in metrics]
        ax_scores.plot(
            x + offset,
            values,
            marker="o",
            markersize=8,
            linewidth=2.8,
            color=color,
            label=str(row["label"]),
        )
        for xx, value in zip(x + offset, values):
            ax_scores.text(xx, value + 0.032, f"{value:.2f}", ha="center", fontsize=9.5, fontweight="bold")
    ax_scores.set_xticks(x, [name for _, name in metrics], fontweight="bold")
    ax_scores.set_ylim(0, 1.02)
    ax_scores.set_ylabel("score, higher is better")
    ax_scores.set_title("5 cm precision / recall / F1")
    ax_scores.grid(axis="y", color=theme["grid"], alpha=0.55, linewidth=0.8)
    ax_scores.spines[["top", "right"]].set_visible(False)
    style_outer_axis(ax_scores, theme)
    style_outer_legend(ax_scores.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13)), theme)

    save_figure(fig, stem)


def poster_dashboard_score_bars(rows: list[dict[str, object]], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    fig = plt.figure(figsize=(14.4, 8.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86])
    ax_error_pred = fig.add_subplot(grid[0, 0])
    ax_error_gt = fig.add_subplot(grid[0, 1])
    ax_scores = fig.add_subplot(grid[1, :])
    fig.suptitle("Input Geometry Quality: 3RScan vs ScanNet", fontsize=23, fontweight="bold", y=1.075, color=theme.get("outer", theme["text"]))

    ordered = rows
    labels = [str(row["label"]).replace(" / ", "\n") for row in ordered]
    colors = [CONFIG_COLORS[str(row["label"])] for row in ordered]
    _draw_dashboard_error_panel(ax_error_pred, ordered, labels, colors, theme, "pred_median_m", "Pred -> GT median error", _cm_axis_limit(ordered, "pred_median_m"))
    _draw_dashboard_error_panel(ax_error_gt, ordered, labels, colors, theme, "gt_median_m", "GT -> Pred median error", _cm_axis_limit(ordered, "gt_median_m"))

    metrics = [("p5", "Precision"), ("r5", "Recall"), ("f5", "F1")]
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ordered))
    for row, color, offset in zip(ordered, colors, offsets):
        values = [float(row[key]) for key, _ in metrics]
        bars = ax_scores.bar(
            x + offset,
            values,
            width,
            color=color,
            edgecolor=theme["edge"],
            linewidth=0.7,
            alpha=0.95,
            label=str(row["label"]),
        )
        for bar, value in zip(bars, values):
            ax_scores.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.022,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
    ax_scores.set_title("5 cm scores", pad=12)
    subtitle_color = theme.get("outer", theme["muted"])
    ax_scores.text(0.5, 1.01, "Precision, recall, and F1 are shown as separate bars (higher is better)", transform=ax_scores.transAxes, ha="center", color=subtitle_color, fontsize=10)
    ax_scores.set_xticks(x, [name for _, name in metrics], fontweight="bold")
    ax_scores.set_ylim(0, 1.02)
    ax_scores.set_ylabel("score")
    ax_scores.grid(axis="y", color=theme["grid"], alpha=0.55, linewidth=0.8)
    ax_scores.spines[["top", "right"]].set_visible(False)
    style_outer_axis(ax_scores, theme)
    style_outer_legend(ax_scores.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16)), theme)

    save_figure(fig, stem)


def poster_dashboard_score_tiles(rows: list[dict[str, object]], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    fig = plt.figure(figsize=(14.4, 8.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86])
    ax_error_pred = fig.add_subplot(grid[0, 0])
    ax_error_gt = fig.add_subplot(grid[0, 1])
    ax_scores = fig.add_subplot(grid[1, :])
    fig.suptitle("Input Geometry Quality: 3RScan vs ScanNet", fontsize=23, fontweight="bold", y=1.075, color=theme.get("outer", theme["text"]))

    ordered = rows
    labels = [str(row["label"]).replace(" / ", "\n") for row in ordered]
    colors = [CONFIG_COLORS[str(row["label"])] for row in ordered]
    _draw_dashboard_error_panel(ax_error_pred, ordered, labels, colors, theme, "pred_median_m", "Pred -> GT median error", _cm_axis_limit(ordered, "pred_median_m"))
    _draw_dashboard_error_panel(ax_error_gt, ordered, labels, colors, theme, "gt_median_m", "GT -> Pred median error", _cm_axis_limit(ordered, "gt_median_m"))

    metrics = [("p5", "P@5cm"), ("r5", "R@5cm"), ("f5", "F1@5cm")]
    data = np.array([[float(row[key]) for key, _ in metrics] for row in ordered])
    cmap = LinearSegmentedColormap.from_list("poster_score_map", ["#fff3d0", "#7bdcb6", "#1b9e77"])
    image = ax_scores.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            text_color = "#111317" if value > 0.42 else "#1b1c1e"
            ax_scores.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=15, fontweight="bold", color=text_color)

    ax_scores.set_title("5 cm score matrix", pad=12)
    subtitle_color = theme.get("outer", theme["muted"])
    ax_scores.text(0.5, 1.01, "Darker green means higher score; this avoids line crossings and keeps values readable", transform=ax_scores.transAxes, ha="center", color=subtitle_color, fontsize=10)
    ax_scores.set_xticks(np.arange(len(metrics)), [name for _, name in metrics], fontweight="bold")
    ax_scores.set_yticks(np.arange(len(ordered)), [str(row["label"]) for row in ordered], fontweight="bold")
    ax_scores.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax_scores.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
    ax_scores.grid(which="minor", color=theme["panel"], linewidth=6)
    ax_scores.tick_params(which="minor", bottom=False, left=False)
    ax_scores.spines[:].set_visible(False)
    style_outer_axis(ax_scores, theme)
    cbar = fig.colorbar(image, ax=ax_scores, fraction=0.018, pad=0.018)
    cbar_color = theme.get("outer", theme["muted"])
    cbar.set_label("score, higher is better", color=cbar_color)
    cbar.ax.tick_params(colors=cbar_color)

    save_figure(fig, stem)


def method_dumbbells(rows: list[dict[str, object]], theme: dict[str, str], stem: str) -> None:
    setup_style(theme)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.4), constrained_layout=True)
    fig.suptitle("Method Contrast Within Each Dataset", fontsize=22, fontweight="bold", y=1.03, color=theme.get("outer", theme["text"]))

    error_rows = [
        ("3RScan", "Pred -> GT", "pred_median_m"),
        ("3RScan", "GT -> Pred", "gt_median_m"),
        ("ScanNet", "Pred -> GT", "pred_median_m"),
        ("ScanNet", "GT -> Pred", "gt_median_m"),
    ]
    y = np.arange(len(error_rows))
    ax = axes[0]
    for yy, (dataset, label, key) in zip(y, error_rows):
        vals = {
            method: float(next(row for row in rows if row["dataset"] == dataset and row["method"] == method)[key]) * 100
            for method in ("MUSt3R", "Pi3X")
        }
        ax.plot([vals["MUSt3R"], vals["Pi3X"]], [yy, yy], color=theme["grid"], linewidth=3)
        for method in ("MUSt3R", "Pi3X"):
            ax.scatter(vals[method], yy, s=180, color=METHOD_COLORS[method], edgecolor=theme["edge"], linewidth=1.0, zorder=3)
            ax.text(vals[method] + 0.25, yy + 0.08, f"{vals[method]:.1f}", fontsize=9.5, fontweight="bold")
    ax.set_yticks(y, [f"{dataset} · {label}" for dataset, label, _ in error_rows], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("median error in cm (lower is better)")
    ax.set_title("Distance errors")
    ax.grid(axis="x", color=theme["grid"], alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    style_outer_axis(ax, theme)

    score_rows = [
        ("3RScan", "Precision", "p5"),
        ("3RScan", "Recall", "r5"),
        ("3RScan", "F1", "f5"),
        ("ScanNet", "Precision", "p5"),
        ("ScanNet", "Recall", "r5"),
        ("ScanNet", "F1", "f5"),
    ]
    y = np.arange(len(score_rows))
    ax = axes[1]
    for yy, (dataset, label, key) in zip(y, score_rows):
        vals = {
            method: float(next(row for row in rows if row["dataset"] == dataset and row["method"] == method)[key])
            for method in ("MUSt3R", "Pi3X")
        }
        ax.plot([vals["MUSt3R"], vals["Pi3X"]], [yy, yy], color=theme["grid"], linewidth=3)
        for method in ("MUSt3R", "Pi3X"):
            ax.scatter(vals[method], yy, s=180, color=METHOD_COLORS[method], edgecolor=theme["edge"], linewidth=1.0, zorder=3)
            ax.text(vals[method] + 0.015, yy + 0.08, f"{vals[method]:.2f}", fontsize=9.5, fontweight="bold")
    ax.set_yticks(y, [f"{dataset} · {label}" for dataset, label, _ in score_rows], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("score (higher is better)")
    ax.set_title("5 cm scores")
    ax.grid(axis="x", color=theme["grid"], alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    style_outer_axis(ax, theme)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["MUSt3R"], markeredgecolor=theme["edge"], markersize=10, label="MUSt3R"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["Pi3X"], markeredgecolor=theme["edge"], markersize=10, label="Pi3X"),
    ]
    style_outer_legend(fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02)), theme)
    save_figure(fig, stem)


def write_blue_index() -> None:
    plots = [
        ("08_poster_dashboard_blue", "Blue dashboard", "Main poster-style dashboard with distance errors and line-based 5 cm scores."),
        ("09_poster_dashboard_score_bars_blue", "Blue score-bar dashboard", "Same top layout, with grouped score bars for easier reading on a poster."),
        ("10_poster_dashboard_score_tiles_blue", "Blue score-tile dashboard", "Matrix score alternative that avoids line crossings."),
        ("01_error_grouped_bars_dark", "Dark distance bars", "Older high-contrast option."),
        ("06_poster_dashboard_score_bars_light", "Cream dashboard", "Original cream style used before the blue poster direction."),
    ]
    cards = "\n".join(
        f"""
        <article class="card {'wide' if i < 3 else ''}">
          <div class="card-head">
            <div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
            <span class="badge">{stem.split('_')[0]}</span>
          </div>
          <figure><img src="{stem}.png" alt="{title}"></figure>
          <div class="links">
            <a href="{stem}.png">PNG</a>
            <a href="{stem}.svg">SVG</a>
            <a href="{stem}.pdf">PDF</a>
          </div>
        </article>
        """
        for i, (stem, title, desc) in enumerate(plots)
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Input Geometry Comparison Plots</title>
  <style>
    :root {{
      --blue: #2364ad;
      --panel: #ffffff;
      --text: #111827;
      --muted: #5c6676;
      --line: #c9d6e6;
      --shadow: 0 18px 55px rgba(8, 28, 54, 0.24);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 6%, rgba(255, 255, 255, 0.28), transparent 24rem),
        radial-gradient(circle at 84% 12%, rgba(83, 214, 162, 0.20), transparent 24rem),
        linear-gradient(135deg, #2364ad 0%, #18559d 52%, #0d2f63 100%);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      min-height: 100vh;
    }}
    main {{ width: min(1500px, calc(100vw - 48px)); margin: 0 auto; padding: 34px 0 58px; }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
      gap: 22px;
      margin-bottom: 26px;
    }}
    .hero, .legend, .card {{
      border: 1px solid rgba(255, 255, 255, 0.28);
      background: rgba(255, 255, 255, 0.96);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero, .legend {{ padding: 26px; }}
    h1 {{ margin: 0 0 12px; font-size: clamp(2.0rem, 4vw, 4.4rem); letter-spacing: -0.06em; line-height: 0.95; }}
    .subtitle {{ margin: 0; color: var(--muted); font-size: 1.05rem; line-height: 1.55; max-width: 880px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }}
    .chip {{ border-radius: 999px; padding: 8px 12px; background: #edf4ff; border: 1px solid var(--line); font-weight: 800; }}
    .legend h2 {{ margin: 0 0 14px; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted); }}
    .legend-grid {{ display: grid; gap: 10px; }}
    .legend-row {{ display: flex; align-items: center; gap: 10px; font-weight: 800; }}
    .dot {{ width: 13px; height: 13px; border-radius: 50%; box-shadow: 0 0 0 4px rgba(35, 100, 173, 0.10); }}
    .note {{ margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }}
    .card {{ overflow: hidden; }}
    .card.wide {{ grid-column: 1 / -1; }}
    .card-head {{ display: flex; justify-content: space-between; gap: 18px; padding: 20px 22px 12px; }}
    .card h3 {{ margin: 0 0 6px; font-size: 1.18rem; letter-spacing: -0.02em; }}
    .card p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .badge {{ flex: none; padding: 7px 10px; border-radius: 999px; background: #2364ad; color: white; font-weight: 900; font-size: 0.78rem; }}
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
        <h1>Input Geometry Quality</h1>
        <p class="subtitle">Blue poster-style variants for the 3RScan vs ScanNet input geometry comparison. Same visual language as the Object-X final plots.</p>
        <div class="chips">
          <span class="chip">Median distance errors</span>
          <span class="chip">5 cm precision / recall / F1</span>
          <span class="chip">MUSt3R vs Pi3X</span>
        </div>
      </section>
      <aside class="legend">
        <h2>Color coding</h2>
        <div class="legend-grid">
          <div class="legend-row"><span class="dot" style="background:#42C6FF"></span>3RScan / MUSt3R</div>
          <div class="legend-row"><span class="dot" style="background:#F6A742"></span>3RScan / Pi3X</div>
          <div class="legend-row"><span class="dot" style="background:#53D6A2"></span>ScanNet / MUSt3R</div>
          <div class="legend-row"><span class="dot" style="background:#F05D7E"></span>ScanNet / Pi3X</div>
        </div>
        <p class="note">SVG/PDF are best for final poster export. The older cream and dark options are still kept below.</p>
      </aside>
    </header>
    <section class="grid">
      {cards}
    </section>
  </main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html)


def main() -> None:
    rows = parse_summary(SUMMARY_PATH)
    grouped_error_bars(rows, DARK, "01_error_grouped_bars_dark")
    score_bars(rows, DARK, "02_scores_grouped_bars_dark")
    poster_dashboard(rows, DARK, "03_poster_dashboard_dark")
    method_dumbbells(rows, LIGHT, "04_method_dumbbell_light")
    poster_dashboard(rows, LIGHT, "05_poster_dashboard_light")
    poster_dashboard_score_bars(rows, LIGHT, "06_poster_dashboard_score_bars_light")
    poster_dashboard_score_tiles(rows, LIGHT, "07_poster_dashboard_score_tiles_light")
    poster_dashboard(rows, POSTER_BLUE, "08_poster_dashboard_blue")
    poster_dashboard_score_bars(rows, POSTER_BLUE, "09_poster_dashboard_score_bars_blue")
    poster_dashboard_score_tiles(rows, POSTER_BLUE, "10_poster_dashboard_score_tiles_blue")
    write_blue_index()
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
