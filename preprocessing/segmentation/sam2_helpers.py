import re
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def reindex_tracks(merged_binary, id_mapping):
    """Reindex object IDs to 1..N after filtering."""
    new_tracks = {}
    new_mapping = {}
    for new_id, old_id in enumerate(sorted(merged_binary.keys()), start=1):
        new_tracks[new_id] = merged_binary[old_id]
        new_mapping[new_id] = id_mapping[old_id]
    return new_tracks, new_mapping


def prune_short_tracks(
    merged_binary,
    id_mapping,
    *,
    min_frames=2,
    min_peak_area=400,
    min_total_area=1600,
):
    """
    Prune likely-noisy tracks that are very short and tiny.
    Keep a track if it has enough temporal support OR enough area.
    """
    kept_binary = {}
    kept_mapping = {}
    removed = 0
    for obj_id, frame_masks in merged_binary.items():
        areas = [int(mask.sum()) for mask in frame_masks.values()]
        n_frames = len(areas)
        peak_area = max(areas) if areas else 0
        total_area = int(np.sum(areas)) if areas else 0
        keep = (
            n_frames >= int(min_frames)
            or peak_area >= int(min_peak_area)
            or total_area >= int(min_total_area)
        )
        if keep:
            kept_binary[obj_id] = frame_masks
            kept_mapping[obj_id] = id_mapping[obj_id]
        else:
            removed += 1
    kept_binary, kept_mapping = reindex_tracks(kept_binary, kept_mapping)
    return kept_binary, kept_mapping, removed


def parse_scan_frame_idx(frame_path):
    """Extract numeric frame index from filename (e.g. frame-000042 -> 42)."""
    m = re.search(r"frame-(\d+)", Path(frame_path).stem)
    if m is None:
        raise ValueError(f"Cannot extract frame idx from {frame_path}")
    return int(m.group(1))


def colorize_obj_map(obj_map):
    """Convert an integer obj_id map to a deterministic RGB color image."""
    vis = np.zeros((*obj_map.shape, 3), dtype=np.uint8)

    for oid in np.unique(obj_map):
        if oid == 0:
            continue
        rng = np.random.default_rng(int(oid))
        color = rng.integers(60, 255, size=3, dtype=np.uint8)
        vis[obj_map == oid] = color

    return vis


def save_obj_id_and_color_frames(obj_id_imgs: Dict[int, np.ndarray], scan_id: str, output_dir: str):
    obj_dir = Path(output_dir) / "obj_id" / scan_id
    color_dir = Path(output_dir) / "color" / scan_id
    obj_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    for scan_fidx, obj_id_map in obj_id_imgs.items():
        cv2.imwrite(
            str(obj_dir / f"frame-{scan_fidx:06d}.jpg"),
            obj_id_map.astype(np.uint16),
        )
        color_vis = colorize_obj_map(obj_id_map)
        cv2.imwrite(
            str(color_dir / f"frame-{scan_fidx:06d}.jpg"),
            cv2.cvtColor(color_vis, cv2.COLOR_RGB2BGR),
        )
