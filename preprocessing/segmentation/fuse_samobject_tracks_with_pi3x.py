#!/usr/bin/env python3
"""Fuse SAMObject 2D track masks with Pi3X per-pixel geometry.

This adapter intentionally avoids SAMObject's 3D graph clustering path.  The
SAMObject output is treated as 2D tracklets, while Pi3X supplies the 3D surface
points used to decide conservative tracklet merges and to feed Object-X via
gt_projection masks.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def _frame_id_from_mask_path(path: str) -> str:
    name = osp.basename(path)
    if not name.startswith("maskraw_") or not name.endswith(".png"):
        raise ValueError(f"Unexpected SAMObject mask filename: {name}")
    return name[len("maskraw_") : -len(".png")]


def _camera_points_to_world(points_cam: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    rotation = pose_c2w[:3, :3].astype(np.float32)
    translation = pose_c2w[:3, 3].astype(np.float32)
    return (rotation @ points_cam.T).T + translation[None, :]


def _resize_labels_nearest(labels: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    if labels.shape == (height, width):
        return labels
    resample = getattr(Image, "Resampling", Image).NEAREST
    resized = Image.fromarray(labels).resize((width, height), resample=resample)
    return np.asarray(resized, dtype=labels.dtype)


def _stable_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[idx].astype(np.float32, copy=False)


@dataclass
class Tracklet:
    raw_id: int
    frames: set[str] = field(default_factory=set)
    point_count: int = 0
    voxel_keys: set[tuple[int, int, int]] = field(default_factory=set)
    sampled_chunks: list[np.ndarray] = field(default_factory=list)
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None

    def add_points(
        self,
        frame_id: str,
        points_world: np.ndarray,
        *,
        merge_voxel_size: float,
        max_points_per_frame: int,
    ) -> None:
        if points_world.size == 0:
            return
        self.frames.add(frame_id)
        self.point_count += int(points_world.shape[0])

        pts_min = points_world.min(axis=0)
        pts_max = points_world.max(axis=0)
        self.bbox_min = pts_min if self.bbox_min is None else np.minimum(self.bbox_min, pts_min)
        self.bbox_max = pts_max if self.bbox_max is None else np.maximum(self.bbox_max, pts_max)

        voxels = np.floor(points_world.astype(np.float64) / merge_voxel_size).astype(np.int64)
        unique_voxels = np.unique(voxels, axis=0)
        self.voxel_keys.update(map(tuple, unique_voxels.tolist()))

        self.sampled_chunks.append(_stable_sample(points_world, max_points_per_frame))

    def sampled_points(self, max_points: int) -> np.ndarray:
        if not self.sampled_chunks:
            return np.zeros((0, 3), dtype=np.float32)
        points = np.concatenate(self.sampled_chunks, axis=0)
        return _stable_sample(points, max_points)


class DisjointSet:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a


def _bbox_gap(a: Tracklet, b: Tracklet) -> float:
    if a.bbox_min is None or a.bbox_max is None or b.bbox_min is None or b.bbox_max is None:
        return float("inf")
    gap = np.maximum(np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max), 0.0)
    return float(np.linalg.norm(gap))


def _voxel_overlap_fraction(a: Tracklet, b: Tracklet) -> tuple[int, float]:
    if not a.voxel_keys or not b.voxel_keys:
        return 0, 0.0
    small, large = (
        (a.voxel_keys, b.voxel_keys)
        if len(a.voxel_keys) <= len(b.voxel_keys)
        else (b.voxel_keys, a.voxel_keys)
    )
    overlap = sum(1 for key in small if key in large)
    return overlap, float(overlap / max(1, len(small)))


def _near_point_fraction(a_points: np.ndarray, b_points: np.ndarray, distance: float) -> float:
    if a_points.size == 0 or b_points.size == 0:
        return 0.0
    tree_b = cKDTree(b_points)
    dist_a, _ = tree_b.query(a_points, k=1)
    frac_a = float((dist_a <= distance).mean())

    tree_a = cKDTree(a_points)
    dist_b, _ = tree_a.query(b_points, k=1)
    frac_b = float((dist_b <= distance).mean())
    return max(frac_a, frac_b)


def _should_merge(
    a: Tracklet,
    b: Tracklet,
    *,
    allow_covisible_merge: bool,
    merge_voxel_overlap: float,
    near_merge_min_voxel_overlap: float,
    near_fraction_thr: float,
    near_distance: float,
    max_sample_points: int,
) -> tuple[bool, dict]:
    shared_frames = len(a.frames & b.frames)
    if shared_frames > 0 and not allow_covisible_merge:
        return False, {"shared_frames": shared_frames, "reason": "co_visible"}

    bbox_gap = _bbox_gap(a, b)
    if bbox_gap > near_distance:
        return False, {"shared_frames": shared_frames, "bbox_gap": bbox_gap, "reason": "bbox_far"}

    overlap_count, overlap_fraction = _voxel_overlap_fraction(a, b)
    if overlap_fraction >= merge_voxel_overlap:
        return True, {
            "shared_frames": shared_frames,
            "bbox_gap": bbox_gap,
            "voxel_overlap": overlap_count,
            "voxel_overlap_fraction": overlap_fraction,
            "reason": "voxel_overlap",
        }

    if overlap_fraction < near_merge_min_voxel_overlap:
        return False, {
            "shared_frames": shared_frames,
            "bbox_gap": bbox_gap,
            "voxel_overlap": overlap_count,
            "voxel_overlap_fraction": overlap_fraction,
            "reason": "low_overlap",
        }

    near_fraction = _near_point_fraction(
        a.sampled_points(max_sample_points),
        b.sampled_points(max_sample_points),
        near_distance,
    )
    return (
        near_fraction >= near_fraction_thr,
        {
            "shared_frames": shared_frames,
            "bbox_gap": bbox_gap,
            "voxel_overlap": overlap_count,
            "voxel_overlap_fraction": overlap_fraction,
            "near_fraction": near_fraction,
            "reason": "near_fraction",
        },
    )


def _write_objects_json(root_dir: str, scan_id: str, object_ids: list[int]) -> None:
    files_dir = osp.join(root_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    scan_entry = {
        "scan": scan_id,
        "objects": [
            {"id": int(obj_id), "label": "object", "global_id": int(obj_id)}
            for obj_id in object_ids
        ],
    }
    for filename in ("objects.json", "objects_sam2.json"):
        path = osp.join(files_dir, filename)
        if osp.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = {"scans": []}
        data["scans"] = [entry for entry in data.get("scans", []) if entry.get("scan") != scan_id]
        data["scans"].append(scan_entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _write_debug_ply(
    out_path: str,
    groups: list[list[int]],
    tracklets: dict[int, Tracklet],
    *,
    max_points_per_object: int,
) -> None:
    points_out = []
    colors_out = []
    object_ids_out = []
    track_ids_out = []
    palette = np.array(
        [
            [230, 57, 70],
            [29, 53, 87],
            [42, 157, 143],
            [233, 196, 106],
            [244, 162, 97],
            [69, 123, 157],
            [131, 56, 236],
            [255, 183, 3],
        ],
        dtype=np.uint8,
    )
    for object_id, raw_ids in enumerate(groups, start=1):
        chunks = []
        chunk_track_ids = []
        per_track_limit = max(1, max_points_per_object // max(1, len(raw_ids)))
        for raw_id in raw_ids:
            pts = tracklets[raw_id].sampled_points(per_track_limit)
            if pts.size == 0:
                continue
            chunks.append(pts)
            chunk_track_ids.append(np.full(pts.shape[0], raw_id, dtype=np.int32))
        if not chunks:
            continue
        pts_obj = _stable_sample(np.concatenate(chunks, axis=0), max_points_per_object)
        tids_obj = _stable_sample(
            np.concatenate(chunk_track_ids, axis=0).reshape(-1, 1).astype(np.float32),
            max_points_per_object,
        ).reshape(-1).astype(np.int32)
        points_out.append(pts_obj)
        colors_out.append(np.tile(palette[(object_id - 1) % len(palette)], (pts_obj.shape[0], 1)))
        object_ids_out.append(np.full(pts_obj.shape[0], object_id, dtype=np.int32))
        track_ids_out.append(tids_obj)

    if not points_out:
        return
    points = np.concatenate(points_out, axis=0)
    colors = np.concatenate(colors_out, axis=0)
    object_ids = np.concatenate(object_ids_out, axis=0)
    track_ids = np.concatenate(track_ids_out, axis=0)

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    vertex_data = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("objectId", "i4"),
            ("trackId", "i4"),
        ],
    )
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data["objectId"] = object_ids
    vertex_data["trackId"] = track_ids
    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(out_path)


def _collect_tracklets(args: argparse.Namespace) -> tuple[dict[int, Tracklet], dict]:
    mask_dir = args.mask_dir or osp.join(
        args.sam_root, "2D_masks", args.scan_id, "semantic-sam"
    )
    mask_paths = sorted(glob(osp.join(mask_dir, "maskraw_*.png")))
    if not mask_paths:
        raise FileNotFoundError(f"No SAMObject maskraw_*.png files found in {mask_dir}")

    tracklets: dict[int, Tracklet] = {}
    stats = {
        "mask_dir": mask_dir,
        "pi3x_seq_dir": args.pi3x_seq_dir,
        "input_frames": len(mask_paths),
        "frames_with_pi3x": 0,
        "frames_with_points": 0,
        "raw_points": 0,
    }

    for mask_path in mask_paths:
        frame_id = _frame_id_from_mask_path(mask_path)
        xyz_path = osp.join(args.pi3x_seq_dir, f"frame-{frame_id}.xyz.npy")
        pose_path = osp.join(args.pi3x_seq_dir, f"frame-{frame_id}.pose.txt")
        conf_path = osp.join(args.pi3x_seq_dir, f"frame-{frame_id}.conf.npy")
        if not (osp.exists(xyz_path) and osp.exists(pose_path)):
            continue

        mask = np.asarray(Image.open(mask_path))
        xyz = np.load(xyz_path).astype(np.float32)
        pose_c2w = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
        conf = (
            np.load(conf_path).astype(np.float32)
            if osp.exists(conf_path)
            else np.ones(xyz.shape[:2], dtype=np.float32)
        )
        mask_pi3x = _resize_labels_nearest(mask, xyz.shape[:2])

        valid = mask_pi3x > 0
        valid &= np.isfinite(xyz).all(axis=-1)
        valid &= xyz[..., 2] > 0.0
        valid &= np.abs(xyz).sum(axis=-1) > 0.0
        valid &= conf >= float(args.conf_thr)
        if int(args.pixel_stride) > 1:
            stride_mask = np.zeros_like(valid, dtype=bool)
            stride = int(args.pixel_stride)
            stride_mask[::stride, ::stride] = True
            valid &= stride_mask
        if not valid.any():
            continue

        stats["frames_with_pi3x"] += 1
        frame_points = 0
        for raw_id in np.unique(mask_pi3x[valid]):
            raw_id_int = int(raw_id)
            if raw_id_int <= 0:
                continue
            raw_valid = valid & (mask_pi3x == raw_id_int)
            if not raw_valid.any():
                continue
            pts_world = _camera_points_to_world(xyz[raw_valid], pose_c2w)
            tracklet = tracklets.setdefault(raw_id_int, Tracklet(raw_id=raw_id_int))
            tracklet.add_points(
                frame_id,
                pts_world,
                merge_voxel_size=float(args.merge_voxel_size),
                max_points_per_frame=int(args.max_sample_points_per_frame),
            )
            frame_points += int(pts_world.shape[0])
        if frame_points > 0:
            stats["frames_with_points"] += 1
            stats["raw_points"] += frame_points

    return tracklets, stats


def _merge_tracklets(
    tracklets: dict[int, Tracklet], args: argparse.Namespace
) -> tuple[list[list[int]], list[dict]]:
    kept_ids = [
        raw_id
        for raw_id, tracklet in tracklets.items()
        if len(tracklet.frames) >= int(args.min_track_frames)
        and tracklet.point_count >= int(args.min_track_points)
        and tracklet.voxel_keys
    ]
    kept_ids.sort()
    dsu = DisjointSet(kept_ids)
    merge_events = []

    for i, raw_a in enumerate(kept_ids):
        a = tracklets[raw_a]
        for raw_b in kept_ids[i + 1 :]:
            b = tracklets[raw_b]
            should_merge, details = _should_merge(
                a,
                b,
                allow_covisible_merge=bool(args.allow_covisible_merge),
                merge_voxel_overlap=float(args.merge_voxel_overlap),
                near_merge_min_voxel_overlap=float(args.near_merge_min_voxel_overlap),
                near_fraction_thr=float(args.near_fraction_thr),
                near_distance=float(args.near_distance),
                max_sample_points=int(args.max_merge_sample_points),
            )
            if should_merge:
                dsu.union(raw_a, raw_b)
                details.update({"track_a": raw_a, "track_b": raw_b})
                merge_events.append(details)

    groups_by_root: dict[int, list[int]] = {}
    for raw_id in kept_ids:
        groups_by_root.setdefault(dsu.find(raw_id), []).append(raw_id)
    groups = list(groups_by_root.values())
    groups.sort(
        key=lambda raw_ids: (
            min(min(tracklets[raw_id].frames) for raw_id in raw_ids),
            -sum(tracklets[raw_id].point_count for raw_id in raw_ids),
            min(raw_ids),
        )
    )
    return groups, merge_events


def _group_point_count(tracklets: dict[int, Tracklet], raw_ids: list[int]) -> int:
    return int(sum(tracklets[raw_id].point_count for raw_id in raw_ids))


def _write_projection_masks(
    args: argparse.Namespace,
    raw_to_object_id: dict[int, int],
) -> dict:
    mask_dir = args.mask_dir or osp.join(
        args.sam_root, "2D_masks", args.scan_id, "semantic-sam"
    )
    out = {}
    output_dtype = (
        np.uint16
        if len(set(raw_to_object_id.values())) <= np.iinfo(np.uint16).max
        else np.int32
    )
    for mask_path in sorted(glob(osp.join(mask_dir, "maskraw_*.png"))):
        frame_id = _frame_id_from_mask_path(mask_path)
        mask = np.asarray(Image.open(mask_path))
        obj_map = np.zeros(mask.shape, dtype=output_dtype)
        for raw_id in np.unique(mask):
            raw_id_int = int(raw_id)
            object_id = raw_to_object_id.get(raw_id_int, 0)
            if object_id > 0:
                obj_map[mask == raw_id_int] = int(object_id)
        if np.any(obj_map > 0):
            out[frame_id] = obj_map

    save_dir = osp.join(args.output_root, "files", "gt_projection", "obj_id_pkl")
    os.makedirs(save_dir, exist_ok=True)
    out_path = osp.join(save_dir, f"{args.scan_id}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    return {"path": out_path, "frames": len(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--pi3x-seq-dir", required=True)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--conf-thr", type=float, default=0.10)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--merge-voxel-size", type=float, default=0.05)
    parser.add_argument("--merge-voxel-overlap", type=float, default=0.12)
    parser.add_argument("--near-merge-min-voxel-overlap", type=float, default=0.02)
    parser.add_argument("--near-fraction-thr", type=float, default=0.65)
    parser.add_argument("--near-distance", type=float, default=0.075)
    parser.add_argument("--min-track-frames", type=int, default=2)
    parser.add_argument("--min-track-points", type=int, default=128)
    parser.add_argument("--min-object-points", type=int, default=0)
    parser.add_argument("--max-sample-points-per-frame", type=int, default=2048)
    parser.add_argument("--max-merge-sample-points", type=int, default=20000)
    parser.add_argument("--debug-points-per-object", type=int, default=5000)
    parser.add_argument("--allow-covisible-merge", action="store_true")
    args = parser.parse_args()

    tracklets, stats = _collect_tracklets(args)
    groups, merge_events = _merge_tracklets(tracklets, args)
    groups_before_object_filter = len(groups)
    min_object_points = max(0, int(args.min_object_points))
    if min_object_points > 0:
        groups = [
            raw_ids
            for raw_ids in groups
            if _group_point_count(tracklets, raw_ids) >= min_object_points
        ]
    if not groups:
        raise RuntimeError(
            "No stable SAMObject/Pi3X tracklets survived filtering. "
            "Lower --min-track-frames/--min-track-points/--min-object-points "
            "or check mask/xyz alignment."
        )
    raw_to_object_id = {
        raw_id: object_id
        for object_id, raw_ids in enumerate(groups, start=1)
        for raw_id in raw_ids
    }
    object_ids = list(range(1, len(groups) + 1))

    projection_info = _write_projection_masks(args, raw_to_object_id)
    _write_objects_json(args.output_root, args.scan_id, object_ids)

    debug_dir = osp.join(args.output_root, "files", "samobject_pi3x_fusion", args.scan_id)
    os.makedirs(debug_dir, exist_ok=True)
    debug_payload = {
        "scan_id": args.scan_id,
        "thresholds": {
            "conf_thr": float(args.conf_thr),
            "pixel_stride": int(args.pixel_stride),
            "merge_voxel_size": float(args.merge_voxel_size),
            "merge_voxel_overlap": float(args.merge_voxel_overlap),
            "near_merge_min_voxel_overlap": float(args.near_merge_min_voxel_overlap),
            "near_fraction_thr": float(args.near_fraction_thr),
            "near_distance": float(args.near_distance),
            "min_track_frames": int(args.min_track_frames),
            "min_track_points": int(args.min_track_points),
            "min_object_points": int(args.min_object_points),
            "allow_covisible_merge": bool(args.allow_covisible_merge),
        },
        "collection": stats,
        "raw_tracklets": len(tracklets),
        "kept_tracklets": len(raw_to_object_id),
        "objects_before_object_filter": groups_before_object_filter,
        "objects": len(groups),
        "dropped_objects_by_object_filter": groups_before_object_filter - len(groups),
        "projection": projection_info,
        "merge_events": merge_events,
        "object_tracks": [
            {
                "object_id": object_id,
                "raw_track_ids": raw_ids,
                "frames": sorted({frame for raw_id in raw_ids for frame in tracklets[raw_id].frames}),
                "point_count": int(sum(tracklets[raw_id].point_count for raw_id in raw_ids)),
                "voxel_count": int(sum(len(tracklets[raw_id].voxel_keys) for raw_id in raw_ids)),
            }
            for object_id, raw_ids in enumerate(groups, start=1)
        ],
    }
    debug_json = osp.join(debug_dir, "fusion_stats.json")
    with open(debug_json, "w") as f:
        json.dump(debug_payload, f, indent=2)

    _write_debug_ply(
        osp.join(debug_dir, "objects_sampled.ply"),
        groups,
        tracklets,
        max_points_per_object=int(args.debug_points_per_object),
    )

    print(
        "[samobject-pi3x-fusion] "
        f"raw_tracklets={len(tracklets)} kept_tracklets={len(raw_to_object_id)} "
        f"objects={len(groups)} projection_frames={projection_info['frames']} "
        f"debug={debug_json}"
    )


if __name__ == "__main__":
    main()
