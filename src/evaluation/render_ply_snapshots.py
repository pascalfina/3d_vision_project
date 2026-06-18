#!/usr/bin/env python3
"""
Headless side-by-side snapshot of two vertex-coloured PLY meshes (GT vs pred),
using matplotlib only (no OpenGL/EGL -- works on a login node without a display).

Renders each mesh from a top-down and an oblique view, GT left, pred right.
Ignored/grey vertices are drawn faint so coloured objects stand out.

Example:
  python src/evaluation/render_ply_snapshots.py \
    --gt  /cluster/scratch/ealegret/sam2object/vis_compare/<scan>_gt.ply \
    --pred /cluster/scratch/ealegret/sam2object/vis_compare/<scan>_pred.ply \
    --out /cluster/scratch/ealegret/sam2object/vis_compare/<scan>_compare.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


def load_xyz_rgb(path):
    v = PlyData.read(path)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    return xyz, rgb


def is_greyish(rgb):  # the "ignored" colour (~0.7) -> draw faint
    return np.all(np.abs(rgb - 0.706) < 0.03, axis=1)


def draw(ax, xyz, rgb, elev, azim, title, point_size):
    grey = is_greyish(rgb)
    alpha = np.where(grey, 0.05, 0.9)
    order = np.argsort(grey)[::-1]  # draw grey first, colours on top
    ax.scatter(xyz[order, 0], xyz[order, 1], xyz[order, 2],
               c=rgb[order], s=point_size, alpha=alpha[order], linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    # equal aspect from data extent
    ext = (xyz.max(0) - xyz.min(0))
    ax.set_box_aspect(ext / ext.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    gt_xyz, gt_rgb = load_xyz_rgb(args.gt)
    pr_xyz, pr_rgb = load_xyz_rgb(args.pred)

    fig = plt.figure(figsize=(12, 11))
    views = [("top", 89, -90), ("oblique", 35, -60)]
    for row, (vname, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, row * 2 + 1, projection="3d")
        draw(ax, gt_xyz, gt_rgb, elev, azim,
             f"GROUND TRUTH ({vname})", args.point_size)
        ax = fig.add_subplot(2, 2, row * 2 + 2, projection="3d")
        draw(ax, pr_xyz, pr_rgb, elev, azim,
             f"PREDICTION ({vname})", args.point_size)

    fig.suptitle("Same colour+place = correct   |   black = false positive   |   "
                 "GT colour missing in pred = missed", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=args.dpi)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
