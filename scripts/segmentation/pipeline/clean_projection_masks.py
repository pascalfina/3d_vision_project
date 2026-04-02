#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

from utils import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", default="pred_projection")
    parser.add_argument("--target", default="pred_projection_clean")
    parser.add_argument("--scan-id", action="append", default=[])
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test"],
        default=[],
        help="Optional split(s) to derive scan ids from resplit files.",
    )
    parser.add_argument("--mask-erode-iters", type=int, default=1)
    parser.add_argument("--mask-min-area", type=int, default=128)
    parser.add_argument("--keep-largest", action="store_true", default=True)
    parser.add_argument("--no-keep-largest", dest="keep_largest", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scan_ids(data_root: Path, splits: list[str], explicit: list[str]) -> list[str]:
    scan_ids = list(explicit)
    for split in splits:
        preferred = data_root / "files" / f"{split}_resplit_scans.txt"
        fallback = data_root / "files" / f"{split}_scans.txt"
        split_file = preferred if preferred.exists() else fallback
        if split_file.exists():
            scan_ids.extend(
                [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
            )
    if scan_ids:
        return sorted(set(scan_ids))

    mask_dir = data_root / "files" / args.source / "obj_id_pkl"
    scan_ids = []
    for path in mask_dir.iterdir():
        name = path.name
        if name.endswith(".pkl.gz"):
            scan_ids.append(name[: -len(".pkl.gz")])
        elif name.endswith(".pkl"):
            scan_ids.append(path.stem)
    return sorted(set(scan_ids))


def clean_binary_mask(
    mask: np.ndarray,
    *,
    keep_largest: bool,
    erode_iters: int,
    min_area: int,
) -> np.ndarray:
    cleaned = mask.astype(bool)
    if cleaned.sum() == 0:
        return cleaned.astype(np.uint8)

    if keep_largest:
        labeled, num_labels = ndimage.label(cleaned)
        if num_labels > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            cleaned = labeled == counts.argmax()

    if erode_iters > 0 and cleaned.sum() > 0:
        eroded = ndimage.binary_erosion(cleaned, iterations=erode_iters)
        if eroded.sum() >= max(16, cleaned.sum() // 10):
            cleaned = eroded

    if cleaned.sum() < min_area:
        return np.zeros_like(mask, dtype=np.uint8)
    return cleaned.astype(np.uint8)


def clean_mask_map(
    mask_map: np.ndarray,
    *,
    keep_largest: bool,
    erode_iters: int,
    min_area: int,
) -> tuple[np.ndarray, dict]:
    cleaned_map = np.zeros_like(mask_map)
    stats = {
        "objects_before": 0,
        "objects_after": 0,
        "pixels_before": 0,
        "pixels_after": 0,
    }
    object_ids = [int(x) for x in np.unique(mask_map) if int(x) > 0]
    stats["objects_before"] = len(object_ids)
    for obj_id in object_ids:
        binary = mask_map == obj_id
        stats["pixels_before"] += int(binary.sum())
        cleaned = clean_binary_mask(
            binary,
            keep_largest=keep_largest,
            erode_iters=erode_iters,
            min_area=min_area,
        )
        if cleaned.sum() == 0:
            continue
        cleaned_map[cleaned.astype(bool)] = obj_id
        stats["objects_after"] += 1
        stats["pixels_after"] += int(cleaned.sum())
    return cleaned_map, stats


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    args = parse_args()
    data_root = Path(args.data_root)
    source_dir = data_root / "files" / args.source / "obj_id_pkl"
    target_dir = data_root / "files" / args.target / "obj_id_pkl"
    target_dir.mkdir(parents=True, exist_ok=True)

    scan_ids = load_scan_ids(data_root, args.split, args.scan_id)
    summary = {
        "data_root": str(data_root),
        "source": args.source,
        "target": args.target,
        "scan_count": len(scan_ids),
        "mask_erode_iters": args.mask_erode_iters,
        "mask_min_area": args.mask_min_area,
        "keep_largest": args.keep_largest,
        "scans": {},
    }

    for scan_id in scan_ids:
        src_file = source_dir / f"{scan_id}.pkl"
        gz_src_file = source_dir / f"{scan_id}.pkl.gz"
        dst_file = target_dir / f"{scan_id}.pkl"
        if not src_file.exists() and not gz_src_file.exists():
            raise FileNotFoundError(
                f"Missing source mask file: {src_file} or {gz_src_file}"
            )
        if dst_file.exists() and not args.overwrite:
            print(f"[skip existing] {scan_id}")
            continue

        obj_id_maps = common.load_pkl_data(src_file)
        cleaned_maps = {}
        scan_stats = {
            "frame_count": len(obj_id_maps),
            "frames": {},
            "objects_before": 0,
            "objects_after": 0,
            "pixels_before": 0,
            "pixels_after": 0,
        }
        for frame_idx, mask_map in obj_id_maps.items():
            cleaned_map, frame_stats = clean_mask_map(
                np.asarray(mask_map),
                keep_largest=args.keep_largest,
                erode_iters=args.mask_erode_iters,
                min_area=args.mask_min_area,
            )
            cleaned_maps[frame_idx] = cleaned_map
            scan_stats["frames"][frame_idx] = frame_stats
            scan_stats["objects_before"] += frame_stats["objects_before"]
            scan_stats["objects_after"] += frame_stats["objects_after"]
            scan_stats["pixels_before"] += frame_stats["pixels_before"]
            scan_stats["pixels_after"] += frame_stats["pixels_after"]

        common.ensure_dir(str(dst_file.parent))
        common.write_pkl_data(cleaned_maps, str(dst_file))
        summary["scans"][scan_id] = scan_stats
        print(
            f"[cleaned] {scan_id} objects {scan_stats['objects_before']} -> {scan_stats['objects_after']} "
            f"pixels {scan_stats['pixels_before']} -> {scan_stats['pixels_after']}"
        )

    write_json(
        data_root / "files" / args.target / "cleanup_summary.json",
        summary,
    )
