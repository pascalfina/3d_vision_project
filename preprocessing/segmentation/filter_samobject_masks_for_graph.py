#!/usr/bin/env python3
"""Filter scene-level SAMObject masks before 3D graph clustering.

SAMObject should segment objects, but automatic SAM masks can sometimes contain
whole-frame or almost-whole-frame regions.  Those masks are valid 2D tracks, but
they are toxic for SAMObject's 3D graph: if many neighboring superpoints share
one giant track ID, the graph can flood-fill most of the scene into one object.

This script keeps the full SAMObject pipeline intact and only removes mask IDs
that are too large to be useful object evidence for 3D graph clustering.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from glob import glob

import numpy as np
from PIL import Image


def _load_mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.int32, copy=False)


def _save_mask(path: str, mask: np.ndarray) -> None:
    max_id = int(mask.max()) if mask.size else 0
    if max_id <= np.iinfo(np.uint16).max:
        Image.fromarray(mask.astype(np.uint16)).save(path)
    else:
        Image.fromarray(mask.astype(np.int32)).save(path)


def filter_mask(mask: np.ndarray, max_area_ratio: float, max_positive_fraction: float):
    total_pixels = int(mask.size)
    positive_pixels = int((mask > 0).sum())
    if total_pixels == 0 or positive_pixels == 0:
        return mask, []

    out = mask.copy()
    removed = []
    ids, counts = np.unique(mask[mask > 0], return_counts=True)
    for mask_id, count in zip(ids, counts):
        mask_id = int(mask_id)
        count = int(count)
        area_ratio = count / total_pixels
        positive_fraction = count / positive_pixels
        remove = False
        if max_area_ratio > 0 and area_ratio > max_area_ratio:
            remove = True
        if max_positive_fraction > 0 and positive_fraction > max_positive_fraction:
            remove = True
        if remove:
            out[out == mask_id] = 0
            removed.append(
                {
                    "mask_id": mask_id,
                    "pixels": count,
                    "area_ratio": area_ratio,
                    "positive_fraction": positive_fraction,
                }
            )
    return out, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--debug-json-out", default=None)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--max-positive-fraction", type=float, default=0.85)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_paths = sorted(glob(osp.join(args.mask_dir, "maskraw_*.png")))
    if not mask_paths:
        raise FileNotFoundError(f"No maskraw_*.png files found in {args.mask_dir}")

    summary = {
        "mask_dir": args.mask_dir,
        "max_area_ratio": float(args.max_area_ratio),
        "max_positive_fraction": float(args.max_positive_fraction),
        "frames": len(mask_paths),
        "frames_changed": 0,
        "removed_instances": 0,
        "removed_pixels": 0,
        "per_frame": [],
    }

    for path in mask_paths:
        mask = _load_mask(path)
        filtered, removed = filter_mask(
            mask,
            max_area_ratio=float(args.max_area_ratio),
            max_positive_fraction=float(args.max_positive_fraction),
        )
        removed_pixels = sum(item["pixels"] for item in removed)
        if removed:
            summary["frames_changed"] += 1
            summary["removed_instances"] += len(removed)
            summary["removed_pixels"] += int(removed_pixels)
            if not args.dry_run:
                _save_mask(path, filtered)
        summary["per_frame"].append(
            {
                "frame": osp.basename(path),
                "removed": removed,
                "removed_pixels": int(removed_pixels),
            }
        )

    if args.debug_json_out:
        os.makedirs(osp.dirname(args.debug_json_out), exist_ok=True)
        with open(args.debug_json_out, "w") as f:
            json.dump(summary, f, indent=2)

    mode = "dry-run" if args.dry_run else "updated"
    print(
        "[SAMObject mask filter] "
        f"{mode} frames={summary['frames']} changed={summary['frames_changed']} "
        f"removed_instances={summary['removed_instances']} "
        f"removed_pixels={summary['removed_pixels']}"
    )


if __name__ == "__main__":
    main()
