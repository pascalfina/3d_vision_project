# utils/mask_utils.py
import numpy as np
import cv2
from typing import Dict, List, Tuple


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two binary masks."""
    intersection = np.logical_and(a, b).sum()
    union        = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def merge_tracks_by_iou(
    tracks: Dict[int, Dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
    iou_reduction: str = "max",
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, List[int]]]:
    """
    Merges overlapping tracks from different keyframes using Union-Find.

    Args:
        tracks:        {orig_id → {frame_idx → bool mask}}
        iou_threshold: threshold over the selected IoU reduction over shared
                       frames to consider two tracks the same object
        iou_reduction: "max" (recommended) or "mean"

    Returns:
        merged_tracks: {new_id → {frame_idx → bool mask}}  (re-indexed from 1)
        id_mapping:    {new_id → [fused orig_ids]}
    """
    ids = list(tracks.keys())
    n   = len(ids)
    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare every pair of tracks over their shared frames
    for i in range(n):
        for j in range(i + 1, n):
            shared = set(tracks[ids[i]]) & set(tracks[ids[j]])
            if not shared:
                continue
            ious = [mask_iou(tracks[ids[i]][f], tracks[ids[j]][f]) for f in shared]
            pair_iou = float(np.max(ious)) if iou_reduction == "max" else float(np.mean(ious))
            if pair_iou >= iou_threshold:
                union(i, j)

    # Group track indices by Union-Find component
    groups: Dict[int, List[int]] = {}
    for i, orig_id in enumerate(ids):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(orig_id)

    # Build merged_tracks and id_mapping (re-indexed from 1; 0 is reserved for background)
    merged_tracks: Dict[int, Dict[int, np.ndarray]] = {}
    id_mapping:    Dict[int, List[int]]              = {}

    for new_id, (_, orig_ids) in enumerate(groups.items(), start=1):
        id_mapping[new_id] = orig_ids
        merged: Dict[int, np.ndarray] = {}
        for orig_id in orig_ids:
            for fidx, mask in tracks[orig_id].items():
                if fidx in merged:
                    merged[fidx] = np.logical_or(merged[fidx], mask)
                else:
                    merged[fidx] = mask.copy()   # BUG 2 FIX: còpia, no referència
        merged_tracks[new_id] = merged

    print(f"[Masks] {n} tracks → {len(merged_tracks)} unique objects")
    return merged_tracks, id_mapping


def visualize_masks_on_frame(
    frame: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Overlay coloured masks on an RGB frame."""
    rng = np.random.default_rng(seed)
    vis = frame.copy().astype(np.float32)
    for mask in masks:
        color = rng.integers(80, 230, size=3).astype(np.float32)
        # BUG 3 FIX: clamp explícit per evitar overflow uint8
        vis[mask] = np.clip(vis[mask] * (1 - alpha) + color * alpha, 0, 255)
    return vis.astype(np.uint8)
