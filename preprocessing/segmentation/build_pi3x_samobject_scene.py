#!/usr/bin/env python3
"""Build a SAMObject-compatible 3D scene from Pi3X geometry.

SAMObject is the object segmenter in this pipeline.  Pi3X only supplies the
3D support surface on which SAMObject can run.  This script therefore builds
the two inputs SAMObject graph clustering needs:

1. labels.instances.annotated.v2.ply
   A stable multi-view Pi3X surfel cloud in world coordinates.
2. mesh.refined.0.010000.segs.v2.json
   Geometry-aware superpoints over that surfel cloud.

The superpoints are not object labels.  They are small, coherent surface
patches used as SAMObject graph primitives.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


@dataclass
class VoxelAccumulator:
    point_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    weight_sum: float = 0.0
    point_count: int = 0
    views: set[int] = field(default_factory=set)


def _camera_points_to_world(points_cam: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    rotation = pose_c2w[:3, :3].astype(np.float32)
    translation = pose_c2w[:3, 3].astype(np.float32)
    return (rotation @ points_cam.T).T + translation[None, :]


def _load_color_for_xyz(color_path: str, shape_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(color_path).convert("RGB")
    height, width = shape_hw
    if image.size != (width, height):
        resample = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((width, height), resample=resample)
    return np.asarray(image, dtype=np.float32) / 255.0


def _iter_frame_paths(sequence_dir: str) -> list[tuple[str, str, str, Optional[str], Optional[str]]]:
    frames = []
    for xyz_path in sorted(glob(osp.join(sequence_dir, "frame-*.xyz.npy"))):
        frame_id = osp.basename(xyz_path).replace("frame-", "").replace(".xyz.npy", "")
        pose_path = osp.join(sequence_dir, f"frame-{frame_id}.pose.txt")
        conf_path = osp.join(sequence_dir, f"frame-{frame_id}.conf.npy")
        color_path = osp.join(sequence_dir, f"frame-{frame_id}.color.jpg")
        if not osp.exists(pose_path):
            continue
        frames.append(
            (
                frame_id,
                xyz_path,
                pose_path,
                conf_path if osp.exists(conf_path) else None,
                color_path if osp.exists(color_path) else None,
            )
        )
    return frames


def _valid_pi3x_pixels(
    xyz: np.ndarray,
    conf: Optional[np.ndarray],
    *,
    conf_thr: float,
    pixel_stride: int,
) -> np.ndarray:
    valid = np.isfinite(xyz).all(axis=-1)
    valid &= xyz[..., 2] > 0.0
    valid &= np.abs(xyz).sum(axis=-1) > 0.0
    if conf is not None:
        valid &= conf >= float(conf_thr)
    if pixel_stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::pixel_stride, ::pixel_stride] = True
        valid &= stride_mask
    return valid


def build_stable_surfels(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frames = _iter_frame_paths(args.sequence_dir)
    if not frames:
        raise FileNotFoundError(f"No Pi3X frames found in {args.sequence_dir}")

    accum: dict[tuple[int, int, int], VoxelAccumulator] = {}
    raw_points = 0
    frames_used = 0

    for view_idx, (_, xyz_path, pose_path, conf_path, color_path) in enumerate(frames):
        xyz = np.load(xyz_path).astype(np.float32)
        conf = np.load(conf_path).astype(np.float32) if conf_path else None
        colors = (
            _load_color_for_xyz(color_path, xyz.shape[:2])
            if color_path
            else np.full((*xyz.shape[:2], 3), 0.5, dtype=np.float32)
        )
        valid = _valid_pi3x_pixels(
            xyz,
            conf,
            conf_thr=float(args.conf_thr),
            pixel_stride=int(args.pixel_stride),
        )
        if not valid.any():
            continue

        weights = conf[valid].astype(np.float64) if conf is not None else np.ones(int(valid.sum()))
        weights = np.clip(weights, 1e-6, None)
        pts_world = _camera_points_to_world(
            xyz[valid].astype(np.float32),
            np.loadtxt(pose_path).reshape(4, 4).astype(np.float32),
        )
        cols = colors[valid].astype(np.float64)
        raw_points += int(pts_world.shape[0])
        frames_used += 1

        voxels = np.floor(pts_world.astype(np.float64) / float(args.voxel_size)).astype(np.int64)
        unique_voxels, inverse, counts = np.unique(
            voxels, axis=0, return_inverse=True, return_counts=True
        )
        point_sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        color_sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        weight_sums = np.zeros(len(unique_voxels), dtype=np.float64)
        np.add.at(point_sums, inverse, pts_world * weights[:, None])
        np.add.at(color_sums, inverse, cols * weights[:, None])
        np.add.at(weight_sums, inverse, weights)

        for key_arr, point_sum, color_sum, weight_sum, count in zip(
            unique_voxels, point_sums, color_sums, weight_sums, counts
        ):
            key = tuple(int(v) for v in key_arr)
            item = accum.setdefault(key, VoxelAccumulator())
            item.point_sum += point_sum
            item.color_sum += color_sum
            item.weight_sum += float(weight_sum)
            item.point_count += int(count)
            item.views.add(view_idx)

    kept_keys = [
        key
        for key, item in accum.items()
        if item.point_count >= int(args.min_points_per_voxel)
        and len(item.views) >= int(args.min_views_per_voxel)
    ]
    kept_keys.sort()
    if not kept_keys:
        raise RuntimeError("No stable Pi3X surfels survived filtering.")

    points = np.stack(
        [accum[key].point_sum / max(accum[key].weight_sum, 1e-6) for key in kept_keys],
        axis=0,
    ).astype(np.float32)
    colors = np.stack(
        [accum[key].color_sum / max(accum[key].weight_sum, 1e-6) for key in kept_keys],
        axis=0,
    ).astype(np.float32)
    colors = np.clip(colors, 0.0, 1.0)
    support_views = np.array([len(accum[key].views) for key in kept_keys], dtype=np.int32)

    stats = {
        "frames_total": len(frames),
        "frames_used": frames_used,
        "raw_points": raw_points,
        "raw_voxels": len(accum),
        "stable_surfels": len(points),
        "voxel_size": float(args.voxel_size),
        "min_views_per_voxel": int(args.min_views_per_voxel),
        "support_views_mean_all_voxels": float(
            np.mean([len(item.views) for item in accum.values()])
        ),
        "support_views_median_all_voxels": float(
            np.median([len(item.views) for item in accum.values()])
        ),
        "support_views_mean_kept": float(support_views.mean()),
        "support_views_median_kept": float(np.median(support_views)),
    }
    return points, colors, support_views, stats


def estimate_normals(points: np.ndarray, *, k: int, chunk_size: int = 8192) -> np.ndarray:
    if len(points) < 4:
        return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(points), 1))
    tree = cKDTree(points)
    k = min(max(4, int(k)), len(points))
    center = points.mean(axis=0)
    normals = np.zeros_like(points, dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        _, nn = tree.query(points[start:end], k=k, workers=-1)
        neigh = points[nn]
        local_mean = neigh.mean(axis=1, keepdims=True)
        centered = neigh - local_mean
        cov = np.matmul(centered.transpose(0, 2, 1), centered) / max(k - 1, 1)
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, :, 0].astype(np.float32)
        orient = np.sum(n * (points[start:end] - center), axis=1) < 0
        n[orient] *= -1.0
        n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-6)
        normals[start:end] = n
    return normals


def build_surface_superpoints(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    support_views: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=float(args.superpoint_radius), workers=-1)

    labels = np.full(len(points), -1, dtype=np.int32)
    order = np.lexsort((np.arange(len(points)), -support_views))
    cos_thr = float(np.cos(np.deg2rad(float(args.normal_angle_deg))))
    color_thr = float(args.color_distance_thr)
    max_points = int(args.max_superpoint_points)
    next_label = 0

    for seed in order:
        seed = int(seed)
        if labels[seed] >= 0:
            continue
        labels[seed] = next_label
        queue = [seed]
        size = 1
        while queue:
            cur = queue.pop()
            for nbr in neighbors[cur]:
                nbr = int(nbr)
                if labels[nbr] >= 0:
                    continue
                if size >= max_points:
                    break
                if float(np.dot(normals[cur], normals[nbr])) < cos_thr:
                    continue
                if float(np.linalg.norm(colors[cur] - colors[nbr])) > color_thr:
                    continue
                labels[nbr] = next_label
                queue.append(nbr)
                size += 1
        next_label += 1

    labels = merge_tiny_superpoints(points, labels, min_size=int(args.min_superpoint_points))
    labels = remap_labels(labels)
    unique, counts = np.unique(labels, return_counts=True)
    stats = {
        "superpoints": int(unique.size),
        "superpoint_radius": float(args.superpoint_radius),
        "normal_angle_deg": float(args.normal_angle_deg),
        "color_distance_thr": float(args.color_distance_thr),
        "min_superpoint_size": int(counts.min()),
        "median_superpoint_size": float(np.median(counts)),
        "mean_superpoint_size": float(counts.mean()),
        "max_superpoint_size": int(counts.max()),
    }
    return labels.astype(np.int32), stats


def merge_tiny_superpoints(points: np.ndarray, labels: np.ndarray, *, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return labels
    unique, counts = np.unique(labels, return_counts=True)
    tiny = unique[counts < min_size]
    if tiny.size == 0:
        return labels

    centroids = np.stack([points[labels == label].mean(axis=0) for label in unique], axis=0)
    size_by_label = {int(label): int(count) for label, count in zip(unique, counts)}
    pos_by_label = {int(label): idx for idx, label in enumerate(unique)}
    large_labels = [int(label) for label in unique if size_by_label[int(label)] >= min_size]
    if not large_labels:
        return labels

    large_pos = np.array([pos_by_label[label] for label in large_labels], dtype=np.int64)
    tree = cKDTree(centroids[large_pos])
    merged = labels.copy()
    for label in tiny:
        label_int = int(label)
        _, nn = tree.query(centroids[pos_by_label[label_int]], k=1)
        target = large_labels[int(nn)]
        merged[labels == label_int] = target
    return merged


def remap_labels(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    mapping = {int(label): idx for idx, label in enumerate(unique)}
    out = np.empty_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == old] = int(new)
    return out


def write_scene_ply(out_ply: str, points: np.ndarray, colors: np.ndarray) -> None:
    os.makedirs(osp.dirname(out_ply), exist_ok=True)
    colors_u8 = (np.clip(colors, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    vertex_data = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors_u8[:, 0]
    vertex_data["green"] = colors_u8[:, 1]
    vertex_data["blue"] = colors_u8[:, 2]
    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(out_ply)


def write_superpoint_json(out_json: str, labels: np.ndarray) -> None:
    os.makedirs(osp.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"segIndices": labels.astype(np.int64).tolist()}, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--superpoint-json-out", required=True)
    parser.add_argument("--debug-json-out", default=None)
    parser.add_argument("--conf-thr", type=float, default=0.10)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--min-views-per-voxel", type=int, default=2)
    parser.add_argument("--min-points-per-voxel", type=int, default=2)
    parser.add_argument("--normal-k", type=int, default=24)
    parser.add_argument("--superpoint-radius", type=float, default=0.09)
    parser.add_argument("--normal-angle-deg", type=float, default=45.0)
    parser.add_argument("--color-distance-thr", type=float, default=0.35)
    parser.add_argument("--min-superpoint-points", type=int, default=8)
    parser.add_argument("--max-superpoint-points", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, colors, support_views, surface_stats = build_stable_surfels(args)
    normals = estimate_normals(points, k=int(args.normal_k))
    superpoints, superpoint_stats = build_surface_superpoints(
        points, colors, normals, support_views, args
    )

    write_scene_ply(args.out_ply, points, colors)
    write_superpoint_json(args.superpoint_json_out, superpoints)

    payload = {
        "surface": surface_stats,
        "superpoints": superpoint_stats,
        "outputs": {
            "ply": args.out_ply,
            "superpoint_json": args.superpoint_json_out,
        },
    }
    if args.debug_json_out:
        os.makedirs(osp.dirname(args.debug_json_out), exist_ok=True)
        with open(args.debug_json_out, "w") as f:
            json.dump(payload, f, indent=2)

    print(
        "[Pi3X SAMObject scene] "
        f"surfels={len(points)} superpoints={superpoint_stats['superpoints']} "
        f"support_views_median={surface_stats['support_views_median_kept']:.2f} "
        f"ply={args.out_ply}"
    )


if __name__ == "__main__":
    main()
