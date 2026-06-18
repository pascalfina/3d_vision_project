#!/usr/bin/env python3
"""Write a colored debug PLY for SAMObject superpoints."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np
from plyfile import PlyData, PlyElement


def make_palette(max_id: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.integers(35, 255, size=(max_id + 1, 3), dtype=np.uint8)
    if max_id >= 0:
        palette[0] = np.array([220, 220, 220], dtype=np.uint8)
    return palette


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--superpoint-json", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ply = PlyData.read(args.ply)
    vertex = ply["vertex"].data
    with open(args.superpoint_json, "r") as f:
        seg_ids = np.asarray(json.load(f)["segIndices"], dtype=np.int64)

    if len(vertex) != len(seg_ids):
        raise ValueError(
            f"PLY vertex count and segIndices length mismatch: {len(vertex)} vs {len(seg_ids)}"
        )

    natural_ids = np.unique(seg_ids)
    remap = {int(old): idx for idx, old in enumerate(natural_ids)}
    seg_natural = np.array([remap[int(x)] for x in seg_ids], dtype=np.int32)
    palette = make_palette(int(seg_natural.max()), args.seed)
    colors = palette[seg_natural]

    vertex_data = np.empty(
        len(vertex),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("superpointId", "i4"),
        ],
    )
    vertex_data["x"] = vertex["x"]
    vertex_data["y"] = vertex["y"]
    vertex_data["z"] = vertex["z"]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data["superpointId"] = seg_natural

    os.makedirs(osp.dirname(args.out_ply), exist_ok=True)
    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(args.out_ply)

    _, counts = np.unique(seg_natural, return_counts=True)
    print(
        "[SAMObject superpoints debug] "
        f"vertices={len(vertex_data)} superpoints={len(counts)} "
        f"size_min={int(counts.min())} size_median={float(np.median(counts)):.1f} "
        f"size_max={int(counts.max())} out={args.out_ply}"
    )


if __name__ == "__main__":
    main()
