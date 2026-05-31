#!/usr/bin/env python3
"""
Schematic of the SAM2Object 3D instance-segmentation evaluation, for slides.

Flow (left -> right):
  shared mesh vertices  ->  IoU matrix (pred x GT)  ->  greedy match @ IoU>=tau
  ->  Precision / Recall / F1 (+ mean matched IoU).

Headless (matplotlib Agg) - no GL / GPU needed. Writes .png and .pdf.

  python src/evaluation/make_methodology_figure.py \
    --out /cluster/scratch/ealegret/sam2object/figures/methodology --example
"""

import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# illustrative (cartoon) IoU matrix: 4 predictions (rows) x 5 GT objects (cols)
IOU = [
    [0.85, 0.05, 0.00, 0.00, 0.00],   # P1 -> G1
    [0.00, 0.10, 0.62, 0.05, 0.00],   # P2 -> G3
    [0.00, 0.00, 0.00, 0.30, 0.20],   # P3 -> best 0.30  (< tau -> FP)
    [0.12, 0.00, 0.00, 0.00, 0.05],   # P4 -> spurious   (FP)
]
TAU = 0.5
MATCHED = {(0, 0), (1, 2)}            # greedy result at tau: TP pairs
ACCENT = "#2c6fbb"
GREEN = "#2e8b57"
RED = "#c0392b"


def box(ax, cx, cy, w, h, text, fc="#eef3f9", ec=ACCENT, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.4,rounding_size=0.8",
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=2))


def draw_matrix(ax, x0, ytop, cell, matched=None):
    """Grid of coloured IoU cells; rows=pred (top->down), cols=GT (left->right)."""
    cmap = plt.cm.Blues
    nr, nc = len(IOU), len(IOU[0])
    for r in range(nr):
        for c in range(nc):
            v = IOU[r][c]
            x = x0 + c * cell
            y = ytop - r * cell
            ax.add_patch(Rectangle((x, y - cell), cell, cell,
                                   fc=cmap(0.12 + 0.72 * v), ec="white", lw=1.2, zorder=2))
            ax.text(x + cell / 2, y - cell / 2, f"{v:.2f}",
                    ha="center", va="center", zorder=3, fontsize=8,
                    color="white" if v > 0.5 else "#333")
        ax.text(x0 - 0.6, ytop - r * cell - cell / 2, f"P{r+1}",
                ha="right", va="center", fontsize=9, color=ACCENT, fontweight="bold")
    for c in range(nc):
        ax.text(x0 + c * cell + cell / 2, ytop + 0.6, f"G{c+1}",
                ha="center", va="bottom", fontsize=9, color="#555", fontweight="bold")

    if matched:
        matched_rows = {r for r, _ in matched}
        matched_cols = {c for _, c in matched}
        for (r, c) in matched:
            x = x0 + c * cell
            y = ytop - r * cell
            ax.add_patch(Rectangle((x, y - cell), cell, cell, fill=False,
                                   ec=GREEN, lw=3, zorder=4))
        for r in range(len(IOU)):                       # unmatched pred row -> FP
            if r not in matched_rows:
                ax.text(x0 + nc * cell + 0.5, ytop - r * cell - cell / 2, "FP",
                        ha="left", va="center", fontsize=9, color=RED, fontweight="bold")
        for c in range(len(IOU[0])):                    # unmatched GT col -> FN
            if c not in matched_cols:
                ax.text(x0 + c * cell + cell / 2, ytop - nr * cell - 0.5, "FN",
                        ha="center", va="top", fontsize=9, color=RED, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/cluster/scratch/ealegret/sam2object/figures/methodology")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--example", action="store_true",
                    help="annotate with the real scene 5341b7e3 headline")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(18, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def stage_title(cx, t):
        ax.text(cx, 95, t, ha="center", va="center", fontsize=13,
                fontweight="bold", color=ACCENT)

    # --- Stage 1: shared vertices --------------------------------------- #
    stage_title(12, "1 · Shared vertices")
    box(ax, 12, 82, 20, 9, "Shared mesh vertices (N)", fc="#fff6e6", ec="#d9a441", bold=True)
    arrow(ax, 12, 77.5, 12, 70.5)
    box(ax, 12, 66, 20, 8, "GT:  objectId per vertex")
    box(ax, 12, 53, 20, 8, "Pred:  instance id per vertex")
    ax.text(12, 43, "1:1 aligned — no registration\n(SAM2Object runs on the GT mesh)",
            ha="center", va="center", fontsize=9, style="italic", color="#555")
    ax.text(12, 33, "instance  =  set of vertices\nsharing a label",
            ha="center", va="center", fontsize=9, color="#333")
    arrow(ax, 22.5, 60, 29, 60)

    # --- Stage 2: IoU matrix -------------------------------------------- #
    stage_title(40, "2 · IoU matrix  (pred × GT)")
    ax.text(40, 88, r"IoU(P,G) = |P ∩ G| / |P ∪ G|", ha="center", va="center",
            fontsize=11, fontweight="bold")
    draw_matrix(ax, x0=31, ytop=78, cell=3.6)
    ax.text(40, 30, "ignored (unannotated / structural)\nvertices removed from the union",
            ha="center", va="center", fontsize=8.5, style="italic", color="#777")
    arrow(ax, 53, 60, 59, 60)

    # --- Stage 3: greedy matching --------------------------------------- #
    stage_title(70, f"3 · Greedy match,  IoU ≥ τ ({TAU})")
    draw_matrix(ax, x0=61, ytop=78, cell=3.6, matched=MATCHED)
    ax.add_patch(Rectangle((62, 33), 2.2, 2.2, fill=False, ec=GREEN, lw=3))
    ax.text(65, 34.1, "= matched (TP)", ha="left", va="center", fontsize=8.5, color=GREEN)
    ax.text(70, 28, "best pairs first, one-to-one;\nleftover pred = FP,  leftover GT = FN",
            ha="center", va="center", fontsize=8.5, style="italic", color="#777")
    arrow(ax, 83, 60, 88, 60)

    # --- Stage 4: metrics ----------------------------------------------- #
    stage_title(92, "4 · Metrics")
    metrics = ("TP = 2   FP = 2   FN = 3\n\n"
               "P = TP/(TP+FP) = 0.50\n"
               "R = TP/(TP+FN) = 0.40\n"
               "F1 = 0.44\n\n"
               "mean matched IoU = 0.74")
    box(ax, 92, 62, 15, 30, metrics, fc="#eef7ee", ec=GREEN, fs=9.5)
    ax.text(92, 40, "+ GT-coverage, pred-purity\n(threshold-free)",
            ha="center", va="center", fontsize=8.5, color="#555")

    # --- footer --------------------------------------------------------- #
    footer = ("Class-agnostic 3D instance segmentation  ·  objects-only & all-instances modes  ·  "
              "reported at IoU 0.25 / 0.50 / 0.75  ·  validated: GT-as-prediction = 1.000 (3RScan + ScanNet)")
    ax.text(50, 12, footer, ha="center", va="center", fontsize=9.5, color="#333")
    if args.example:
        ax.text(50, 5, "Example scene 5341b7e3:  22 GT objects · 15 predicted · 8 matched @ IoU 0.25  "
                       "(one scene — illustrative)", ha="center", va="center",
                fontsize=9, style="italic", color=RED)

    fig.tight_layout()
    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=args.dpi, bbox_inches="tight")
        print(f"[OK] wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
