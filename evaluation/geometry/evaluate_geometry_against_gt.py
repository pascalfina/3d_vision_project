#!/usr/bin/env python3
"""Evaluate reconstructed scene geometry against a GT mesh.

This is intentionally label-free: it measures geometry quality before any
object/instance metric.  It supports a reconstructed PLY point cloud or direct
Pi3X frame-*.xyz.npy fusion as prediction input.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Pi3X/reconstruction geometry against a GT mesh."
    )
    pred = parser.add_mutually_exclusive_group(required=True)
    pred.add_argument("--pred-ply", help="Predicted point-cloud PLY.")
    pred.add_argument(
        "--pred-sequence-dir",
        help="Pi3X sequence dir containing frame-*.xyz.npy and frame-*.pose.txt.",
    )
    parser.add_argument("--gt-mesh", required=True, help="Ground-truth mesh path.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Scene identifier stored in report outputs. Defaults to out-dir name.",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="Method/config name stored in report outputs.",
    )
    parser.add_argument("--sample-gt-points", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument(
        "--report-threshold",
        type=float,
        default=0.05,
        help="Primary threshold for paper-style precision/recall/F1 reporting.",
    )
    parser.add_argument(
        "--align",
        choices=[
            "none",
            "centroid",
            "icp",
            "pca",
            "pca_icp",
            "axis_bbox",
            "axis_bbox_icp",
            "coord_yz_flip",
            "coord_yz_flip_icp",
            "rgbd_correspondence",
            "rgbd_correspondence_icp",
            "pose_rigid",
            "pose_rigid_icp",
            "pose_similarity",
            "pose_similarity_icp",
        ],
        default="pca_icp",
        help=(
            "Align prediction to GT before scoring. Use 'none' if global pose "
            "accuracy should be part of the metric."
        ),
    )
    parser.add_argument(
        "--align-scale",
        choices=["none", "uniform"],
        default="uniform",
        help="Optional scale correction for PCA alignment modes.",
    )
    parser.add_argument(
        "--allow-reflection",
        action="store_true",
        help="Allow det<0 PCA alignments. Off by default to avoid hiding handedness bugs.",
    )
    parser.add_argument("--icp-sample-points", type=int, default=50000)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--axis-bbox-refine-top-k", type=int, default=8)
    parser.add_argument("--coord-bbox-low-q", type=float, default=0.0)
    parser.add_argument("--coord-bbox-high-q", type=float, default=100.0)
    parser.add_argument("--rgbd-pred-sequence-dir", default=None)
    parser.add_argument("--rgbd-gt-sequence-dir", default=None)
    parser.add_argument("--rgbd-gt-sequence-zip", default=None)
    parser.add_argument("--rgbd-frame-stride", type=int, default=4)
    parser.add_argument("--rgbd-max-frames", type=int, default=96)
    parser.add_argument("--rgbd-pixel-stride", type=int, default=6)
    parser.add_argument("--rgbd-conf-thr", type=float, default=0.10)
    parser.add_argument("--rgbd-max-correspondences", type=int, default=50000)
    parser.add_argument("--rgbd-trim-quantile", type=float, default=65.0)
    parser.add_argument("--rgbd-fit-iterations", type=int, default=8)
    parser.add_argument(
        "--icp-trim-quantile",
        type=float,
        default=90.0,
        help="Keep this distance percentile of ICP correspondences each iteration.",
    )
    parser.add_argument(
        "--icp-max-correspondence",
        type=float,
        default=0.0,
        help="Optional maximum ICP correspondence distance. 0 disables this gate.",
    )
    parser.add_argument(
        "--pred-voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel downsample size for predicted points in meters.",
    )
    parser.add_argument(
        "--max-pred-points",
        type=int,
        default=600000,
        help="Deterministically subsample prediction after optional voxel downsample.",
    )
    parser.add_argument(
        "--bbox-margin",
        type=float,
        default=0.25,
        help="Margin around pred bbox for the pred_bbox_gt completeness scope.",
    )
    parser.add_argument(
        "--visible-gt-sequence-dir",
        default=None,
        help="Optional sequence dir with _info.txt and frame-*.pose.txt for visible-GT scope.",
    )
    parser.add_argument(
        "--visible-gt-sequence-zip",
        default=None,
        help="Optional sequence.zip with _info.txt and frame-*.pose.txt for visible-GT scope.",
    )
    parser.add_argument("--visible-max-frames", type=int, default=96)
    parser.add_argument("--visible-frame-stride", type=int, default=1)
    parser.add_argument(
        "--visible-depth-rtol",
        type=float,
        default=0.03,
        help="Relative z-buffer tolerance for GT sampled visibility.",
    )
    parser.add_argument(
        "--visible-depth-atol",
        type=float,
        default=0.03,
        help="Absolute z-buffer tolerance in meters for GT sampled visibility.",
    )
    parser.add_argument("--sequence-conf-thr", type=float, default=0.10)
    parser.add_argument("--sequence-pixel-stride", type=int, default=2)
    parser.add_argument("--sequence-max-frames", type=int, default=0)
    parser.add_argument(
        "--sequence-pose-dir",
        default=None,
        help="Optional pose override dir for --pred-sequence-dir camera-frame xyz maps.",
    )
    parser.add_argument(
        "--sequence-pose-zip",
        default=None,
        help="Optional pose override sequence.zip for --pred-sequence-dir camera-frame xyz maps.",
    )
    parser.add_argument(
        "--sequence-camera-axis-signs",
        default="1,1,1",
        help="Comma-separated camera-frame axis signs applied to xyz before pose lift.",
    )
    parser.add_argument(
        "--pose-align-pred-sequence-dir",
        default=None,
        help="Pi3X/pred sequence dir with frame-*.pose.txt for pose-based alignment.",
    )
    parser.add_argument(
        "--pose-align-gt-sequence-dir",
        default=None,
        help="GT sequence dir with frame-*.pose.txt for pose-based alignment.",
    )
    parser.add_argument(
        "--pose-align-gt-sequence-zip",
        default=None,
        help="GT sequence.zip with frame-*.pose.txt for pose-based alignment.",
    )
    parser.add_argument("--pose-align-frame-stride", type=int, default=1)
    parser.add_argument("--pose-align-max-frames", type=int, default=0)
    parser.add_argument("--pose-align-min-pairs", type=int, default=8)
    parser.add_argument("--write-debug-ply", action="store_true")
    parser.add_argument("--write-debug-html", action="store_true")
    parser.add_argument(
        "--debug-html-max-points",
        type=int,
        default=300000,
        help="Maximum points per cloud in the interactive debug HTML overlay.",
    )
    return parser.parse_args()


def load_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = None
    if {"red", "green", "blue"}.issubset(vertex.dtype.names or ()):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]
    if points.size == 0:
        raise ValueError(f"No valid vertices in {path}")
    return points, colors


def _camera_points_to_world(points_cam: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    return (pose_c2w[:3, :3].astype(np.float32) @ points_cam.T).T + pose_c2w[:3, 3]


def parse_axis_signs(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--sequence-camera-axis-signs must contain three comma-separated values")
    signs = np.array([float(part) for part in parts], dtype=np.float32)
    if not np.all(np.isin(signs, [-1.0, 1.0])):
        raise ValueError("--sequence-camera-axis-signs values must be -1 or 1")
    return signs


def load_sequence_points(
    sequence_dir: Path,
    *,
    conf_thr: float,
    pixel_stride: int,
    max_frames: int,
    pose_sequence_dir: Path | None = None,
    pose_sequence_zip: Path | None = None,
    camera_axis_signs: np.ndarray | None = None,
) -> np.ndarray:
    xyz_paths = sorted(sequence_dir.glob("frame-*.xyz.npy"))
    if max_frames and len(xyz_paths) > max_frames:
        positions = np.linspace(0, len(xyz_paths) - 1, max_frames).round().astype(int)
        xyz_paths = [xyz_paths[int(pos)] for pos in positions]
    points = []
    stride = max(1, int(pixel_stride))
    for xyz_path in xyz_paths:
        frame_id = xyz_path.name.replace("frame-", "").replace(".xyz.npy", "")
        pose_path = sequence_dir / f"frame-{frame_id}.pose.txt"
        conf_path = sequence_dir / f"frame-{frame_id}.conf.npy"
        if pose_sequence_dir is None and pose_sequence_zip is None and not pose_path.exists():
            continue
        xyz = np.load(xyz_path).astype(np.float32)
        valid = np.isfinite(xyz).all(axis=-1)
        valid &= xyz[..., 2] > 0.0
        valid &= np.abs(xyz).sum(axis=-1) > 0.0
        if conf_path.exists():
            valid &= np.load(conf_path).astype(np.float32) >= float(conf_thr)
        if stride > 1:
            stride_mask = np.zeros(valid.shape, dtype=bool)
            stride_mask[::stride, ::stride] = True
            valid &= stride_mask
        if not valid.any():
            continue
        if pose_sequence_dir is not None or pose_sequence_zip is not None:
            pose = load_pose_matrix(
                frame_id,
                sequence_dir=pose_sequence_dir,
                sequence_zip=pose_sequence_zip,
            )
        else:
            pose = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
        pts_cam = xyz[valid]
        if camera_axis_signs is not None:
            pts_cam = pts_cam * camera_axis_signs[None, :]
        points.append(_camera_points_to_world(pts_cam, pose))
    if not points:
        raise ValueError(f"No valid Pi3X points found in {sequence_dir}")
    return np.concatenate(points, axis=0).astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points.astype(np.float64) / float(voxel_size)).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    np.add.at(sums, inverse, points)
    return (sums / counts[:, None]).astype(np.float32)


def deterministic_subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    indices.sort()
    return points[indices]


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("Rigid transform estimation needs at least 3 paired points")
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid[None, :]
    dst_centered = dst - dst_centroid[None, :]
    h = src_centered.T @ dst_centered
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_centroid - r @ src_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r
    transform[:3, 3] = t
    return transform


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, dict]:
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("Similarity transform estimation needs at least 3 paired points")
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid[None, :]
    dst_centered = dst - dst_centroid[None, :]
    h = src_centered.T @ dst_centered
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    rotated_src = (r @ src_centered.T).T
    numerator = float(np.sum(dst_centered * rotated_src))
    denominator = float(np.sum(src_centered * src_centered))
    scale = numerator / max(denominator, 1e-12)
    t = dst_centroid - scale * r @ src_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * r
    transform[:3, 3] = t
    aligned = apply_transform(src, transform)
    residual = np.linalg.norm(aligned - dst, axis=1)
    stats = {
        "scale": float(scale),
        "rotation_determinant": float(np.linalg.det(r)),
        "pose_residual_mean": float(np.mean(residual)),
        "pose_residual_median": float(np.median(residual)),
        "pose_residual_p95": float(np.percentile(residual, 95)),
        "pose_residual_max": float(np.max(residual)),
    }
    return transform, stats


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3][None, :]


def centroid_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = dst.mean(axis=0) - src.mean(axis=0)
    return transform


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center[None, :]
    covariance = (centered.T @ centered) / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    values = values[order]
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1.0
    return center, vectors, np.sqrt(np.maximum(values, 1e-12))


def signed_permutation_matrices(*, allow_reflection: bool) -> list[tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]]:
    import itertools

    matrices = []
    for perm in itertools.permutations(range(3)):
        perm_matrix = np.zeros((3, 3), dtype=np.float64)
        for src_axis, dst_axis in enumerate(perm):
            perm_matrix[dst_axis, src_axis] = 1.0
        for signs in itertools.product([-1, 1], repeat=3):
            matrix = perm_matrix @ np.diag(signs)
            if not allow_reflection and np.linalg.det(matrix) < 0:
                continue
            matrices.append((matrix, tuple(int(x) for x in perm), tuple(int(x) for x in signs)))
    return matrices


def mean_bidirectional_distance(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float]:
    src_to_dst = nearest_distances(src, dst)
    dst_to_src = nearest_distances(dst, src)
    src_mean = float(np.mean(src_to_dst)) if len(src_to_dst) else float("inf")
    dst_mean = float(np.mean(dst_to_src)) if len(dst_to_src) else float("inf")
    return src_mean + dst_mean, src_mean, dst_mean


def pca_alignment_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    seed: int,
    sample_points: int,
    scale_mode: str,
    allow_reflection: bool,
) -> tuple[np.ndarray, dict]:
    src = deterministic_subsample(src_points, sample_points, seed).astype(np.float64)
    dst = deterministic_subsample(dst_points, sample_points, seed + 1).astype(np.float64)
    src_center, src_axes, src_scales = pca_basis(src)
    dst_center, dst_axes, dst_scales = pca_basis(dst)
    scale = 1.0
    if scale_mode == "uniform":
        scale = float(np.linalg.norm(dst_scales) / max(np.linalg.norm(src_scales), 1e-12))

    best = None
    candidates_scored = 0
    for signed_perm, perm, signs in signed_permutation_matrices(allow_reflection=allow_reflection):
        rotation = dst_axes @ signed_perm @ src_axes.T
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = scale * rotation
        matrix[:3, 3] = dst_center - scale * rotation @ src_center
        aligned = apply_transform(src, matrix)
        chamfer, src_mean, dst_mean = mean_bidirectional_distance(aligned, dst)
        candidates_scored += 1
        candidate = {
            "transform": matrix,
            "chamfer_l1_mean": chamfer,
            "pred_to_gt_mean": src_mean,
            "gt_to_pred_mean": dst_mean,
            "determinant": float(np.linalg.det(rotation)),
            "scale": float(scale),
            "permutation": perm,
            "signs": signs,
        }
        if best is None or chamfer < best["chamfer_l1_mean"]:
            best = candidate

    assert best is not None
    stats = {
        "pca_sample_source_points": int(len(src)),
        "pca_sample_target_points": int(len(dst)),
        "pca_candidates_scored": int(candidates_scored),
        "scale_mode": scale_mode,
        "allow_reflection": bool(allow_reflection),
        "coarse_chamfer_l1_mean": float(best["chamfer_l1_mean"]),
        "coarse_pred_to_gt_mean": float(best["pred_to_gt_mean"]),
        "coarse_gt_to_pred_mean": float(best["gt_to_pred_mean"]),
        "coarse_determinant": float(best["determinant"]),
        "coarse_scale": float(best["scale"]),
        "coarse_permutation": [int(x) for x in best["permutation"]],
        "coarse_signs": [int(x) for x in best["signs"]],
    }
    return best["transform"], stats


def bbox_oriented_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    rotation: np.ndarray,
    *,
    scale_mode: str,
    low_q: float = 0.0,
    high_q: float = 100.0,
) -> tuple[np.ndarray, float]:
    oriented_src = (rotation @ src_points.astype(np.float64).T).T
    src_min = np.percentile(oriented_src, float(low_q), axis=0)
    src_max = np.percentile(oriented_src, float(high_q), axis=0)
    dst = dst_points.astype(np.float64)
    dst_min = np.percentile(dst, float(low_q), axis=0)
    dst_max = np.percentile(dst, float(high_q), axis=0)
    src_center = 0.5 * (src_min + src_max)
    dst_center = 0.5 * (dst_min + dst_max)
    src_extent = np.maximum(src_max - src_min, 1e-8)
    dst_extent = np.maximum(dst_max - dst_min, 1e-8)
    scale = 1.0
    if scale_mode == "uniform":
        scale = float(np.median(dst_extent / src_extent))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = dst_center - scale * src_center
    return transform, scale


def robust_alignment_score(src: np.ndarray, dst: np.ndarray, transform: np.ndarray) -> dict:
    aligned = apply_transform(src.astype(np.float64), transform).astype(np.float32)
    pred_to_gt = nearest_distances(aligned, dst)
    gt_to_pred = nearest_distances(dst, aligned)
    return {
        "score": float(np.median(pred_to_gt) + np.median(gt_to_pred)),
        "pred_to_gt_median": float(np.median(pred_to_gt)),
        "gt_to_pred_median": float(np.median(gt_to_pred)),
        "pred_to_gt_p95": float(np.percentile(pred_to_gt, 95)),
        "gt_to_pred_p95": float(np.percentile(gt_to_pred, 95)),
    }


def axis_bbox_alignment_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    seed: int,
    sample_points: int,
    refine_with_icp: bool,
    refine_top_k: int,
    iterations: int,
    trim_quantile: float,
    max_correspondence: float,
) -> tuple[np.ndarray, dict]:
    src = deterministic_subsample(src_points, sample_points, seed).astype(np.float32)
    dst = deterministic_subsample(dst_points, sample_points, seed + 1).astype(np.float32)
    candidates = []
    for rotation, perm, signs in signed_permutation_matrices(allow_reflection=False):
        for scale_mode in ("none", "uniform"):
            transform, scale = bbox_oriented_transform(
                src,
                dst,
                rotation,
                scale_mode=scale_mode,
            )
            score = robust_alignment_score(src, dst, transform)
            candidates.append(
                {
                    "transform": transform,
                    "permutation": perm,
                    "signs": signs,
                    "scale_mode": scale_mode,
                    "scale": float(scale),
                    "determinant": float(np.linalg.det(rotation)),
                    **score,
                }
            )
    candidates.sort(key=lambda item: item["score"])
    coarse_top = candidates[: max(1, int(refine_top_k))]
    best = coarse_top[0]
    icp_history = None
    refined_candidates = []

    if refine_with_icp:
        for candidate in coarse_top:
            refined_transform, icp_stats = refine_icp_transform(
                src_points,
                dst_points,
                initial_transform=candidate["transform"],
                seed=seed,
                sample_points=sample_points,
                iterations=iterations,
                trim_quantile=trim_quantile,
                max_correspondence=max_correspondence,
            )
            score = robust_alignment_score(src, dst, refined_transform)
            refined = {
                **candidate,
                "transform": refined_transform,
                "icp": icp_stats,
                "refined_score": score["score"],
                "refined_pred_to_gt_median": score["pred_to_gt_median"],
                "refined_gt_to_pred_median": score["gt_to_pred_median"],
                "refined_pred_to_gt_p95": score["pred_to_gt_p95"],
                "refined_gt_to_pred_p95": score["gt_to_pred_p95"],
            }
            refined_candidates.append(refined)
        refined_candidates.sort(key=lambda item: item["refined_score"])
        best = refined_candidates[0]
        icp_history = best["icp"]

    def slim(candidate: dict) -> dict:
        out = {
            "permutation": [int(x) for x in candidate["permutation"]],
            "signs": [int(x) for x in candidate["signs"]],
            "scale_mode": candidate["scale_mode"],
            "scale": float(candidate["scale"]),
            "score": float(candidate["score"]),
            "pred_to_gt_median": float(candidate["pred_to_gt_median"]),
            "gt_to_pred_median": float(candidate["gt_to_pred_median"]),
            "pred_to_gt_p95": float(candidate["pred_to_gt_p95"]),
            "gt_to_pred_p95": float(candidate["gt_to_pred_p95"]),
        }
        if "refined_score" in candidate:
            out.update(
                {
                    "refined_score": float(candidate["refined_score"]),
                    "refined_pred_to_gt_median": float(candidate["refined_pred_to_gt_median"]),
                    "refined_gt_to_pred_median": float(candidate["refined_gt_to_pred_median"]),
                    "refined_pred_to_gt_p95": float(candidate["refined_pred_to_gt_p95"]),
                    "refined_gt_to_pred_p95": float(candidate["refined_gt_to_pred_p95"]),
                }
            )
        return out

    stats = {
        "axis_bbox_sample_source_points": int(len(src)),
        "axis_bbox_sample_target_points": int(len(dst)),
        "axis_bbox_candidates_scored": int(len(candidates)),
        "axis_bbox_refine_top_k": int(refine_top_k),
        "selected": slim(best),
        "top_coarse_candidates": [slim(item) for item in candidates[: min(8, len(candidates))]],
    }
    if refined_candidates:
        stats["top_refined_candidates"] = [
            slim(item) for item in refined_candidates[: min(8, len(refined_candidates))]
        ]
    if icp_history is not None:
        stats["icp"] = icp_history
    return best["transform"], stats


def refine_icp_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    initial_transform: np.ndarray | None = None,
    seed: int,
    sample_points: int,
    iterations: int,
    trim_quantile: float,
    max_correspondence: float,
) -> tuple[np.ndarray, dict]:
    src = deterministic_subsample(src_points, sample_points, seed).astype(np.float64)
    dst = deterministic_subsample(dst_points, sample_points, seed + 1).astype(np.float64)
    transform = (
        initial_transform.astype(np.float64, copy=True)
        if initial_transform is not None
        else centroid_transform(src, dst)
    )
    tree = cKDTree(dst)
    history = []

    for iteration in range(int(iterations)):
        transformed = apply_transform(src, transform)
        distances, nn = tree.query(transformed, k=1, workers=-1)
        keep = np.isfinite(distances)
        if float(trim_quantile) < 100.0 and np.any(keep):
            cutoff = np.percentile(distances[keep], float(trim_quantile))
            keep &= distances <= cutoff
        if float(max_correspondence) > 0.0:
            keep &= distances <= float(max_correspondence)
        if int(keep.sum()) < 16:
            break
        delta = estimate_rigid_transform(transformed[keep], dst[nn[keep]])
        transform = delta @ transform
        mean_error = float(np.mean(distances[keep]))
        history.append(
            {
                "iteration": int(iteration + 1),
                "mean_error": mean_error,
                "median_error": float(np.median(distances[keep])),
                "kept_correspondences": int(keep.sum()),
            }
        )
        if len(history) >= 2 and abs(history[-2]["mean_error"] - mean_error) < 1e-6:
            break

    stats = {
        "sample_source_points": int(len(src)),
        "sample_target_points": int(len(dst)),
        "iterations_requested": int(iterations),
        "iterations_run": int(len(history)),
        "trim_quantile": float(trim_quantile),
        "max_correspondence": float(max_correspondence),
        "history": history,
    }
    return transform, stats


def load_triangle_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh

        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            geometries = [geom for geom in mesh.geometry.values() if hasattr(geom, "faces")]
            if not geometries:
                raise ValueError("trimesh scene has no triangle geometry")
            mesh = trimesh.util.concatenate(geometries)
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        triangles = np.asarray(mesh.faces).astype(np.int64)
    except Exception:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        triangles = np.asarray(mesh.triangles).astype(np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"Could not load a valid triangle mesh from {path}")
    return vertices, triangles


def sample_mesh_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    n_points: int,
    seed: int,
) -> np.ndarray:
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 1e-12
    tri = tri[valid]
    areas = areas[valid]
    if len(tri) == 0:
        raise ValueError("GT mesh has no non-degenerate triangles")
    probs = areas / areas.sum()
    rng = np.random.default_rng(seed)
    face_idx = rng.choice(len(tri), size=int(n_points), replace=True, p=probs)
    chosen = tri[face_idx]
    u = rng.random(int(n_points), dtype=np.float32)
    v = rng.random(int(n_points), dtype=np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (
        chosen[:, 2] - chosen[:, 0]
    )
    return points.astype(np.float32)


def _parse_info_text(text: str) -> dict:
    parsed = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        parsed[key] = value
    return parsed


def _intrinsic_from_info_values(values: str) -> np.ndarray:
    intr = [float(v) for v in values.split()]
    return np.array(
        [[intr[0], 0.0, intr[2]], [0.0, intr[5], intr[6]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _parse_color_intrinsics(text: str) -> dict:
    data = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        data[key] = value
    width = int(float(data["m_colorWidth"]))
    height = int(float(data["m_colorHeight"]))
    intrinsic = _intrinsic_from_info_values(data["m_calibrationColorIntrinsic"])
    return {"width": width, "height": height, "intrinsic": intrinsic}


def _zip_member_by_basename(zip_path: Path, basename: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [name for name in zf.namelist() if Path(name).name == basename]
    if not candidates:
        raise FileNotFoundError(f"Missing {basename} in {zip_path}")
    candidates.sort(key=lambda value: (len(value), value))
    return candidates[0]


def _read_sequence_text(
    *,
    sequence_dir: Path | None = None,
    sequence_zip: Path | None = None,
    basename: str,
) -> str:
    if sequence_dir is not None:
        path = sequence_dir / basename
        if not path.exists():
            raise FileNotFoundError(f"Missing {basename} in {sequence_dir}")
        return path.read_text()
    if sequence_zip is not None:
        member = _zip_member_by_basename(sequence_zip, basename)
        with zipfile.ZipFile(sequence_zip) as zf:
            return zf.read(member).decode("utf-8")
    raise ValueError("Either sequence_dir or sequence_zip is required")


def load_intrinsics(
    *,
    sequence_dir: Path | None = None,
    sequence_zip: Path | None = None,
) -> dict:
    text = _read_sequence_text(
        sequence_dir=sequence_dir,
        sequence_zip=sequence_zip,
        basename="_info.txt",
    )
    return _parse_color_intrinsics(text)


def load_depth_camera_info(
    *,
    sequence_dir: Path | None = None,
    sequence_zip: Path | None = None,
) -> dict:
    text = _read_sequence_text(
        sequence_dir=sequence_dir,
        sequence_zip=sequence_zip,
        basename="_info.txt",
    )
    data = _parse_info_text(text)
    return {
        "width": int(float(data["m_depthWidth"])),
        "height": int(float(data["m_depthHeight"])),
        "depth_shift": float(data.get("m_depthShift", 1000.0)),
        "intrinsic": _intrinsic_from_info_values(data["m_calibrationDepthIntrinsic"]),
    }


def load_depth_image_meters(
    frame_id: str,
    *,
    sequence_dir: Path | None = None,
    sequence_zip: Path | None = None,
    depth_shift: float,
) -> np.ndarray:
    basename = f"frame-{frame_id}.depth.pgm"
    if sequence_dir is not None:
        image = Image.open(sequence_dir / basename)
    elif sequence_zip is not None:
        member = _zip_member_by_basename(sequence_zip, basename)
        with zipfile.ZipFile(sequence_zip) as zf:
            image = Image.open(io.BytesIO(zf.read(member)))
            image.load()
    else:
        raise ValueError("Either sequence_dir or sequence_zip is required")
    return np.asarray(image, dtype=np.float32) / float(depth_shift)


def _extract_pose_frame_id(path_name: str) -> str | None:
    match = re.search(r"frame-(\d+)\.pose\.txt$", path_name)
    if not match:
        return None
    return match.group(1)


def frame_ids_from_sequence(
    *,
    sequence_dir: Path | None = None,
    sequence_zip: Path | None = None,
    stride: int,
    max_frames: int,
) -> list[str]:
    if sequence_dir is not None:
        names = [p.name for p in sequence_dir.glob("frame-*.pose.txt")]
    elif sequence_zip is not None:
        with zipfile.ZipFile(sequence_zip) as zf:
            names = zf.namelist()
    else:
        raise ValueError("Either sequence_dir or sequence_zip is required")
    frame_ids = sorted({frame_id for name in names if (frame_id := _extract_pose_frame_id(name))})
    frame_ids = frame_ids[:: max(1, int(stride))]
    if max_frames and len(frame_ids) > max_frames:
        positions = np.linspace(0, len(frame_ids) - 1, max_frames).round().astype(int)
        frame_ids = [frame_ids[int(pos)] for pos in positions]
    return frame_ids


def load_pose_matrix(
    frame_id: str,
    *,
    sequence_dir: Path | None = None,
    sequence_zip: Path | None = None,
) -> np.ndarray:
    text = _read_sequence_text(
        sequence_dir=sequence_dir,
        sequence_zip=sequence_zip,
        basename=f"frame-{frame_id}.pose.txt",
    )
    return np.loadtxt(text.splitlines()).reshape(4, 4).astype(np.float32)


def load_pose_centers(
    *,
    sequence_dir: Path | None = None,
    sequence_zip: Path | None = None,
    frame_stride: int,
    max_frames: int,
) -> dict[str, np.ndarray]:
    frame_ids = frame_ids_from_sequence(
        sequence_dir=sequence_dir,
        sequence_zip=sequence_zip,
        stride=frame_stride,
        max_frames=max_frames,
    )
    centers: dict[str, np.ndarray] = {}
    for frame_id in frame_ids:
        pose = load_pose_matrix(frame_id, sequence_dir=sequence_dir, sequence_zip=sequence_zip)
        center = pose[:3, 3].astype(np.float64)
        if np.isfinite(center).all():
            centers[frame_id] = center
    return centers


def pose_alignment_transform(
    *,
    pred_sequence_dir: Path,
    gt_sequence_dir: Path | None,
    gt_sequence_zip: Path | None,
    frame_stride: int,
    max_frames: int,
    min_pairs: int,
    with_scale: bool,
) -> tuple[np.ndarray, dict]:
    pred_centers = load_pose_centers(
        sequence_dir=pred_sequence_dir,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    gt_centers = load_pose_centers(
        sequence_dir=gt_sequence_dir,
        sequence_zip=gt_sequence_zip,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    common_ids = sorted(set(pred_centers) & set(gt_centers))
    if len(common_ids) < int(min_pairs):
        raise ValueError(
            f"Pose alignment needs at least {min_pairs} common frames, got {len(common_ids)}"
        )
    src = np.stack([pred_centers[frame_id] for frame_id in common_ids], axis=0)
    dst = np.stack([gt_centers[frame_id] for frame_id in common_ids], axis=0)
    if with_scale:
        transform, stats = estimate_similarity_transform(src, dst)
    else:
        transform = estimate_rigid_transform(src, dst)
        aligned = apply_transform(src, transform)
        residual = np.linalg.norm(aligned - dst, axis=1)
        stats = {
            "scale": 1.0,
            "rotation_determinant": float(np.linalg.det(transform[:3, :3])),
            "pose_residual_mean": float(np.mean(residual)),
            "pose_residual_median": float(np.median(residual)),
            "pose_residual_p95": float(np.percentile(residual, 95)),
            "pose_residual_max": float(np.max(residual)),
        }
    stats.update(
        {
            "pose_pairs": int(len(common_ids)),
            "first_frame_id": common_ids[0],
            "last_frame_id": common_ids[-1],
            "pred_sequence_dir": str(pred_sequence_dir),
            "gt_sequence_dir": str(gt_sequence_dir) if gt_sequence_dir is not None else None,
            "gt_sequence_zip": str(gt_sequence_zip) if gt_sequence_zip is not None else None,
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames),
        }
    )
    return transform, stats


def collect_rgbd_correspondences(
    *,
    pred_sequence_dir: Path,
    gt_sequence_dir: Path | None,
    gt_sequence_zip: Path | None,
    frame_stride: int,
    max_frames: int,
    pixel_stride: int,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pair Pi3X world points with GT depth world points at matching frames/rays."""

    depth_info = load_depth_camera_info(sequence_dir=gt_sequence_dir, sequence_zip=gt_sequence_zip)
    depth_intrinsic = depth_info["intrinsic"].astype(np.float32)
    depth_shift = float(depth_info["depth_shift"])
    frame_ids = frame_ids_from_sequence(
        sequence_dir=pred_sequence_dir,
        stride=frame_stride,
        max_frames=max_frames,
    )
    if not frame_ids:
        raise FileNotFoundError(f"No Pi3X pose frames found in {pred_sequence_dir}")

    src_chunks = []
    dst_chunks = []
    frames_used = 0
    projected_samples = 0
    depth_valid_samples = 0
    stride = max(1, int(pixel_stride))

    for frame_id in frame_ids:
        xyz_path = pred_sequence_dir / f"frame-{frame_id}.xyz.npy"
        pred_pose_path = pred_sequence_dir / f"frame-{frame_id}.pose.txt"
        conf_path = pred_sequence_dir / f"frame-{frame_id}.conf.npy"
        if not xyz_path.exists() or not pred_pose_path.exists():
            continue

        xyz = np.load(xyz_path).astype(np.float32)
        conf = np.load(conf_path).astype(np.float32) if conf_path.exists() else None
        valid = _valid_sequence_xyz_mask(xyz, conf=conf, conf_thr=conf_thr, pixel_stride=stride)
        if not valid.any():
            continue

        pred_pose = np.loadtxt(pred_pose_path).reshape(4, 4).astype(np.float32)
        gt_pose = load_pose_matrix(frame_id, sequence_dir=gt_sequence_dir, sequence_zip=gt_sequence_zip)
        depth = load_depth_image_meters(
            frame_id,
            sequence_dir=gt_sequence_dir,
            sequence_zip=gt_sequence_zip,
            depth_shift=depth_shift,
        )

        pts_cam = xyz[valid].astype(np.float32)
        z = pts_cam[:, 2]
        projected = (depth_intrinsic @ pts_cam.T).T
        finite = np.isfinite(projected).all(axis=1) & (np.abs(projected[:, 2]) > 1e-8)
        if not np.any(finite):
            continue
        pts_cam = pts_cam[finite]
        z = z[finite]
        uv = projected[finite, :2] / projected[finite, 2:3]
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        if not np.any(inside):
            continue
        pts_cam = pts_cam[inside]
        u = u[inside]
        v = v[inside]
        z = z[inside]
        projected_samples += int(len(pts_cam))

        z_gt = depth[v, u].astype(np.float32)
        depth_valid = z_gt > 0.0
        if not np.any(depth_valid):
            continue
        pts_cam = pts_cam[depth_valid]
        u = u[depth_valid]
        v = v[depth_valid]
        z_gt = z_gt[depth_valid]
        depth_valid_samples += int(len(pts_cam))

        target_cam = np.stack(
            [
                (u.astype(np.float32) - depth_intrinsic[0, 2]) / depth_intrinsic[0, 0] * z_gt,
                (v.astype(np.float32) - depth_intrinsic[1, 2]) / depth_intrinsic[1, 1] * z_gt,
                z_gt,
            ],
            axis=1,
        ).astype(np.float32)
        src_chunks.append(_camera_points_to_world(pts_cam, pred_pose))
        dst_chunks.append(_camera_points_to_world(target_cam, gt_pose))
        frames_used += 1

    if not src_chunks:
        raise RuntimeError("No RGB-D correspondences could be collected")

    src = np.concatenate(src_chunks, axis=0).astype(np.float32)
    dst = np.concatenate(dst_chunks, axis=0).astype(np.float32)
    stats = {
        "frames_considered": int(len(frame_ids)),
        "frames_used": int(frames_used),
        "correspondences_raw": int(len(src)),
        "projected_samples": int(projected_samples),
        "depth_valid_samples": int(depth_valid_samples),
        "frame_stride": int(frame_stride),
        "max_frames": int(max_frames),
        "pixel_stride": int(pixel_stride),
        "conf_thr": float(conf_thr),
        "pred_sequence_dir": str(pred_sequence_dir),
        "gt_sequence_dir": str(gt_sequence_dir) if gt_sequence_dir is not None else None,
        "gt_sequence_zip": str(gt_sequence_zip) if gt_sequence_zip is not None else None,
    }
    return src, dst, stats


def _valid_sequence_xyz_mask(
    xyz: np.ndarray,
    *,
    conf: np.ndarray | None,
    conf_thr: float,
    pixel_stride: int,
) -> np.ndarray:
    valid = np.isfinite(xyz).all(axis=-1)
    valid &= xyz[..., 2] > 0.0
    valid &= np.abs(xyz).sum(axis=-1) > 0.0
    if conf is not None:
        valid &= conf >= float(conf_thr)
    if pixel_stride > 1:
        stride_mask = np.zeros(valid.shape, dtype=bool)
        stride_mask[::pixel_stride, ::pixel_stride] = True
        valid &= stride_mask
    return valid


def fit_trimmed_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    seed: int,
    max_correspondences: int,
    trim_quantile: float,
    iterations: int,
) -> tuple[np.ndarray, dict]:
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("Trimmed similarity needs at least 3 paired correspondences")
    if max_correspondences > 0 and len(src) > max_correspondences:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(src), size=int(max_correspondences), replace=False)
        idx.sort()
        src = src[idx]
        dst = dst[idx]

    keep = np.ones(len(src), dtype=bool)
    history = []
    transform = np.eye(4, dtype=np.float64)
    for iteration in range(max(1, int(iterations))):
        transform, fit_stats = estimate_similarity_transform(src[keep], dst[keep])
        aligned = apply_transform(src.astype(np.float64), transform)
        residual = np.linalg.norm(aligned - dst.astype(np.float64), axis=1)
        cutoff = np.percentile(residual, float(trim_quantile))
        keep = residual <= cutoff
        history.append(
            {
                "iteration": int(iteration + 1),
                "scale": float(fit_stats["scale"]),
                "kept_correspondences": int(keep.sum()),
                "residual_mean": float(np.mean(residual)),
                "residual_median": float(np.median(residual)),
                "residual_p95": float(np.percentile(residual, 95)),
                "trim_cutoff": float(cutoff),
            }
        )
        if keep.sum() < 16:
            break

    aligned = apply_transform(src.astype(np.float64), transform)
    residual = np.linalg.norm(aligned - dst.astype(np.float64), axis=1)
    stats = {
        "fit_correspondences": int(len(src)),
        "kept_correspondences_final": int(keep.sum()),
        "trim_quantile": float(trim_quantile),
        "iterations_requested": int(iterations),
        "iterations_run": int(len(history)),
        "scale": float(abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0)),
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_p95": float(np.percentile(residual, 95)),
        "history": history,
    }
    return transform, stats


def rgbd_correspondence_alignment_transform(
    *,
    pred_sequence_dir: Path,
    gt_sequence_dir: Path | None,
    gt_sequence_zip: Path | None,
    frame_stride: int,
    max_frames: int,
    pixel_stride: int,
    conf_thr: float,
    seed: int,
    max_correspondences: int,
    trim_quantile: float,
    iterations: int,
) -> tuple[np.ndarray, dict]:
    src, dst, corr_stats = collect_rgbd_correspondences(
        pred_sequence_dir=pred_sequence_dir,
        gt_sequence_dir=gt_sequence_dir,
        gt_sequence_zip=gt_sequence_zip,
        frame_stride=frame_stride,
        max_frames=max_frames,
        pixel_stride=pixel_stride,
        conf_thr=conf_thr,
    )
    transform, fit_stats = fit_trimmed_similarity(
        src,
        dst,
        seed=seed,
        max_correspondences=max_correspondences,
        trim_quantile=trim_quantile,
        iterations=iterations,
    )
    return transform, {"rgbd_correspondences": corr_stats, "rgbd_fit": fit_stats}


def visible_gt_mask(
    gt_points: np.ndarray,
    *,
    sequence_dir: Path | None = None,
    sequence_zip: Path | None = None,
    frame_stride: int,
    max_frames: int,
    depth_rtol: float,
    depth_atol: float,
) -> tuple[np.ndarray, dict]:
    info = load_intrinsics(sequence_dir=sequence_dir, sequence_zip=sequence_zip)
    width = int(info["width"])
    height = int(info["height"])
    intrinsic = info["intrinsic"].astype(np.float32)
    frame_ids = frame_ids_from_sequence(
        sequence_dir=sequence_dir,
        sequence_zip=sequence_zip,
        stride=frame_stride,
        max_frames=max_frames,
    )
    if not frame_ids:
        source = sequence_dir if sequence_dir is not None else sequence_zip
        raise FileNotFoundError(f"No pose frames found in {source}")

    points_h = np.concatenate(
        [gt_points.astype(np.float32), np.ones((len(gt_points), 1), dtype=np.float32)],
        axis=1,
    )
    visible = np.zeros(len(gt_points), dtype=bool)
    visible_observations = 0

    for frame_id in frame_ids:
        pose = load_pose_matrix(frame_id, sequence_dir=sequence_dir, sequence_zip=sequence_zip)
        world_to_camera = np.linalg.inv(pose).astype(np.float32)
        cam_h = (world_to_camera @ points_h.T).T
        cam = cam_h[:, :3] / np.clip(cam_h[:, 3:4], 1e-8, np.inf)
        z = cam[:, 2]
        valid = np.isfinite(cam).all(axis=1) & (z > 0.0)
        if not np.any(valid):
            continue

        proj = (intrinsic @ cam.T).T
        finite_proj = valid & np.isfinite(proj).all(axis=1) & (np.abs(proj[:, 2]) > 1e-8)
        u = np.full(len(gt_points), -1, dtype=np.int32)
        v = np.full(len(gt_points), -1, dtype=np.int32)
        uv = proj[finite_proj, :2] / proj[finite_proj, 2:3]
        u[finite_proj] = np.rint(uv[:, 0]).astype(np.int32)
        v[finite_proj] = np.rint(uv[:, 1]).astype(np.int32)
        bounded = finite_proj & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(bounded):
            continue

        idx = np.flatnonzero(bounded)
        pixel = v[idx].astype(np.int64) * width + u[idx].astype(np.int64)
        depth = z[idx]
        order = np.argsort(pixel, kind="stable")
        idx_sorted = idx[order]
        pixel_sorted = pixel[order]
        depth_sorted = depth[order]
        starts = np.r_[0, np.flatnonzero(pixel_sorted[1:] != pixel_sorted[:-1]) + 1]
        ends = np.r_[starts[1:], len(pixel_sorted)]
        min_depth = np.minimum.reduceat(depth_sorted, starts)
        counts = ends - starts
        per_point_min = np.repeat(min_depth, counts)
        tol = np.maximum(float(depth_atol), float(depth_rtol) * per_point_min)
        frame_visible = np.abs(depth_sorted - per_point_min) <= tol
        visible[idx_sorted[frame_visible]] = True
        visible_observations += int(frame_visible.sum())

    stats = {
        "frames_considered": len(frame_ids),
        "visible_points": int(visible.sum()),
        "visible_fraction": float(visible.mean()) if len(visible) else 0.0,
        "visible_observations": int(visible_observations),
        "depth_rtol": float(depth_rtol),
        "depth_atol": float(depth_atol),
    }
    return visible, stats


def distance_summary(distances: np.ndarray) -> dict:
    if len(distances) == 0:
        return {
            "count": 0,
            "mean": None,
            "rmse": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "rmse": float(np.sqrt(np.mean(np.square(distances)))),
        "median": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(np.max(distances)),
    }


def threshold_key(threshold: float) -> str:
    return f"{float(threshold):.3f}m"


def threshold_metrics(pred_to_gt: np.ndarray, gt_to_pred: np.ndarray, thresholds: Iterable[float]) -> dict:
    out = {}
    for threshold in thresholds:
        precision = float(np.mean(pred_to_gt <= threshold)) if len(pred_to_gt) else 0.0
        recall = float(np.mean(gt_to_pred <= threshold)) if len(gt_to_pred) else 0.0
        fscore = 0.0
        if precision + recall > 0.0:
            fscore = 2.0 * precision * recall / (precision + recall)
        key = threshold_key(float(threshold))
        out[key] = {
            "threshold_m": float(threshold),
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }
    return out


def nearest_distances(query: np.ndarray, support: np.ndarray, workers: int = -1) -> np.ndarray:
    tree = cKDTree(support)
    distances, _ = tree.query(query, k=1, workers=workers)
    return distances.astype(np.float32)


def bbox(points: np.ndarray) -> dict:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return {
        "min": mn.round(6).tolist(),
        "max": mx.round(6).tolist(),
        "extent": (mx - mn).round(6).tolist(),
        "diag": float(np.linalg.norm(mx - mn)),
    }


def mask_points_in_bbox(points: np.ndarray, reference: np.ndarray, margin: float) -> np.ndarray:
    mn = reference.min(axis=0) - float(margin)
    mx = reference.max(axis=0) + float(margin)
    return np.all((points >= mn[None, :]) & (points <= mx[None, :]), axis=1)


def error_colors(distances: np.ndarray, clip_distance: float) -> np.ndarray:
    if len(distances) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    x = np.clip(distances / max(float(clip_distance), 1e-6), 0.0, 1.0)
    # Blue -> cyan -> yellow -> red, simple deterministic colormap.
    r = np.clip(2.0 * x - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * x - 1.0), 0.0, 1.0)
    b = np.clip(1.2 - 2.0 * x, 0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255.0).round().astype(np.uint8)


def write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(
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
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def write_overlay_html(
    path: Path,
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    max_points_per_cloud: int,
    title: str,
    details=None,
) -> None:
    import trimesh
    import trimesh.viewer

    path.parent.mkdir(parents=True, exist_ok=True)
    pred_vis = deterministic_subsample(pred_points, max_points_per_cloud, seed=501)
    gt_vis = deterministic_subsample(gt_points, max_points_per_cloud, seed=502)
    pred_colors = np.tile(np.array([[255, 95, 35, 210]], dtype=np.uint8), (len(pred_vis), 1))
    gt_colors = np.tile(np.array([[35, 120, 255, 185]], dtype=np.uint8), (len(gt_vis), 1))

    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.points.PointCloud(gt_vis, colors=gt_colors),
        geom_name="GT mesh sample (blue)",
    )
    scene.add_geometry(
        trimesh.points.PointCloud(pred_vis, colors=pred_colors),
        geom_name="Pi3X output aligned (orange)",
    )
    bounds = scene.bounds
    if bounds is not None and np.isfinite(bounds).all():
        center = bounds.mean(axis=0)
        distance = max(float(np.max(bounds[1] - bounds[0])) * 1.7, 1.0)
        scene.set_camera(angles=(0.75, 0.0, 0.65), distance=distance, center=center)

    html_text = trimesh.viewer.scene_to_html(scene)
    detail_rows = []
    for key, value in (details or {}).items():
        detail_rows.append(
            f'<div class="meta"><b>{html.escape(str(key))}:</b> '
            f'{html.escape(str(value))}</div>'
        )
    detail_html = "\n  ".join(detail_rows)
    legend = f"""
<style>
.objx-overlay-legend {{
  position: fixed;
  top: 16px;
  left: 16px;
  z-index: 9999;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(255,255,255,0.92);
  color: #17202a;
  font: 13px/1.35 sans-serif;
  box-shadow: 0 8px 28px rgba(0,0,0,0.18);
}}
.objx-overlay-legend .title {{ font-weight: 700; margin-bottom: 6px; }}
.objx-overlay-legend .row {{ display: flex; align-items: center; gap: 8px; }}
.objx-overlay-legend .meta {{ max-width: 560px; overflow-wrap: anywhere; margin-top: 3px; }}
.objx-overlay-legend .swatch {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
</style>
<div class="objx-overlay-legend">
  <div class="title">{html.escape(title)}</div>
  <div class="row"><span class="swatch" style="background:#2378ff"></span>GT mesh sample</div>
  <div class="row"><span class="swatch" style="background:#ff5f23"></span>Pi3X output aligned</div>
  <div>points: GT {len(gt_vis):,}, Pi3X {len(pred_vis):,}</div>
  {detail_html}
</div>
"""
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", legend + "\n</body>")
    else:
        html_text += legend
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(html_text, encoding="utf-8")
    tmp_path.replace(path)


def _alignment_value(metrics: dict, key: str):
    alignment = metrics.get("alignment", {})
    if key in alignment:
        return alignment[key]
    for nested_key in ("rgbd_fit", "pose", "icp"):
        nested = alignment.get(nested_key, {})
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return None


def _primary_threshold_payload(thresholds: dict, report_threshold: float) -> tuple[str, dict]:
    preferred = threshold_key(report_threshold)
    if preferred in thresholds:
        return preferred, thresholds[preferred]
    if not thresholds:
        return preferred, {
            "threshold_m": float(report_threshold),
            "precision": None,
            "recall": None,
            "fscore": None,
        }
    return min(
        thresholds.items(),
        key=lambda item: abs(float(item[1].get("threshold_m", report_threshold)) - report_threshold),
    )


def build_report_rows(metrics: dict, report_threshold: float) -> list[dict]:
    rows = []
    pred_summary = metrics["pred_to_gt"]
    alignment = metrics.get("alignment", {})
    rgbd_corr = alignment.get("rgbd_correspondences", {})
    rgbd_fit = alignment.get("rgbd_fit", {})
    for scope, payload in metrics["scopes"].items():
        gt_summary = payload["gt_to_pred"]
        primary_key, primary = _primary_threshold_payload(payload["thresholds"], report_threshold)
        row = {
            "scene_id": metrics.get("scene_id"),
            "method_name": metrics.get("method_name"),
            "scope": scope,
            "scope_description": payload.get("description"),
            "pred_source_type": metrics.get("pred_source", {}).get("type"),
            "pred_source_path": metrics.get("pred_source", {}).get("path"),
            "gt_mesh": metrics.get("gt_mesh"),
            "pred_points_eval": metrics.get("counts", {}).get("pred_points_eval"),
            "gt_points_scope": payload.get("gt_points"),
            "gt_fraction_scope": payload.get("gt_fraction"),
            "accuracy_mean_m": pred_summary.get("mean"),
            "accuracy_rmse_m": pred_summary.get("rmse"),
            "accuracy_median_m": pred_summary.get("median"),
            "accuracy_p90_m": pred_summary.get("p90"),
            "accuracy_p95_m": pred_summary.get("p95"),
            "completeness_mean_m": gt_summary.get("mean"),
            "completeness_rmse_m": gt_summary.get("rmse"),
            "completeness_median_m": gt_summary.get("median"),
            "completeness_p90_m": gt_summary.get("p90"),
            "completeness_p95_m": gt_summary.get("p95"),
            "chamfer_l1_mean_m": payload.get("chamfer_l1_mean"),
            f"precision_at_{primary_key}": primary.get("precision"),
            f"recall_at_{primary_key}": primary.get("recall"),
            f"fscore_at_{primary_key}": primary.get("fscore"),
            "report_threshold_m": primary.get("threshold_m"),
            "alignment_mode": alignment.get("mode"),
            "alignment_scale": alignment.get("effective_uniform_scale")
            if alignment.get("effective_uniform_scale") is not None
            else rgbd_fit.get("scale"),
            "alignment_translation_norm_m": alignment.get("translation_norm"),
            "rgbd_frames_used": rgbd_corr.get("frames_used"),
            "rgbd_correspondences_raw": rgbd_corr.get("correspondences_raw"),
            "rgbd_fit_correspondences": rgbd_fit.get("fit_correspondences"),
            "rgbd_fit_kept_correspondences_final": rgbd_fit.get("kept_correspondences_final"),
            "rgbd_fit_residual_median_m": rgbd_fit.get("residual_median"),
            "rgbd_fit_residual_p95_m": rgbd_fit.get("residual_p95"),
        }
        for threshold_name, threshold_payload in payload["thresholds"].items():
            row[f"precision_at_{threshold_name}"] = threshold_payload.get("precision")
            row[f"recall_at_{threshold_name}"] = threshold_payload.get("recall")
            row[f"fscore_at_{threshold_name}"] = threshold_payload.get("fscore")
        rows.append(row)
    return rows


def _fmt_float(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report_markdown(path: Path, metrics: dict, report_threshold: float) -> None:
    rows = build_report_rows(metrics, report_threshold)
    alignment = metrics.get("alignment", {})
    threshold_name = threshold_key(report_threshold)
    threshold_label = f"@{int(round(report_threshold * 100))}cm"
    lines = [
        "# Geometry Evaluation Summary",
        "",
        f"- Scene: `{metrics.get('scene_id')}`",
        f"- Method: `{metrics.get('method_name')}`",
        f"- Prediction: `{metrics.get('pred_source', {}).get('type')}`",
        f"- Alignment: `{alignment.get('mode')}`",
        f"- GT mesh: `{metrics.get('gt_mesh')}`",
        "",
        f"| Scope | Acc mean | Acc med | Acc p95 | Compl mean | Compl med | Compl p95 | P{threshold_label} | R{threshold_label} | F1{threshold_label} |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        primary_key = f"fscore_at_{threshold_name}"
        precision_key = f"precision_at_{threshold_name}"
        recall_key = f"recall_at_{threshold_name}"
        if primary_key not in row:
            # Fall back to the nearest available threshold if the user supplied
            # a custom threshold list that does not include report_threshold.
            available = sorted(k for k in row if k.startswith("fscore_at_"))
            suffix = available[0].removeprefix("fscore_at_") if available else threshold_name
            precision_key = f"precision_at_{suffix}"
            recall_key = f"recall_at_{suffix}"
            primary_key = f"fscore_at_{suffix}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scope"]),
                    _fmt_float(row["accuracy_mean_m"]),
                    _fmt_float(row["accuracy_median_m"]),
                    _fmt_float(row["accuracy_p95_m"]),
                    _fmt_float(row["completeness_mean_m"]),
                    _fmt_float(row["completeness_median_m"]),
                    _fmt_float(row["completeness_p95_m"]),
                    _fmt_float(row.get(precision_key), digits=3),
                    _fmt_float(row.get(recall_key), digits=3),
                    _fmt_float(row.get(primary_key), digits=3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Alignment QC",
            "",
            f"- Scale: `{_fmt_float(_alignment_value(metrics, 'effective_uniform_scale'))}`",
            f"- Translation norm: `{_fmt_float(alignment.get('translation_norm'))} m`",
            f"- RGB-D frames used: `{alignment.get('rgbd_correspondences', {}).get('frames_used', 'n/a')}`",
            f"- RGB-D raw correspondences: `{alignment.get('rgbd_correspondences', {}).get('correspondences_raw', 'n/a')}`",
            f"- RGB-D fit residual median: `{_fmt_float(alignment.get('rgbd_fit', {}).get('residual_median'))} m`",
            f"- RGB-D fit residual p95: `{_fmt_float(alignment.get('rgbd_fit', {}).get('residual_p95'))} m`",
            "",
            "## Artifacts",
            "",
            "- `metrics.json`: full machine-readable output",
            "- `metrics.csv`: one report row per GT scope",
            "- `report_summary.md`: this report-ready summary",
            "- `overlay_pred_gt.html`: interactive overlay when `--write-debug-html` is enabled",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, metrics: dict, report_threshold: float) -> None:
    rows = build_report_rows(metrics, report_threshold)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_id = args.scene_id or out_dir.name
    method_name = args.method_name or "pi3x_geometry"

    if args.pred_ply:
        pred_points, pred_colors = load_ply_points(Path(args.pred_ply))
        pred_source = {"type": "ply", "path": str(Path(args.pred_ply))}
    else:
        pred_points = load_sequence_points(
            Path(args.pred_sequence_dir),
            conf_thr=args.sequence_conf_thr,
            pixel_stride=args.sequence_pixel_stride,
            max_frames=args.sequence_max_frames,
            pose_sequence_dir=Path(args.sequence_pose_dir) if args.sequence_pose_dir else None,
            pose_sequence_zip=Path(args.sequence_pose_zip) if args.sequence_pose_zip else None,
            camera_axis_signs=parse_axis_signs(args.sequence_camera_axis_signs),
        )
        pred_colors = None
        pred_source = {
            "type": "sequence",
            "path": str(Path(args.pred_sequence_dir)),
            "pose_override_dir": str(Path(args.sequence_pose_dir)) if args.sequence_pose_dir else None,
            "pose_override_zip": str(Path(args.sequence_pose_zip)) if args.sequence_pose_zip else None,
            "camera_axis_signs": [float(x) for x in parse_axis_signs(args.sequence_camera_axis_signs)],
            "conf_thr": float(args.sequence_conf_thr),
            "pixel_stride": int(args.sequence_pixel_stride),
            "max_frames": int(args.sequence_max_frames),
        }

    pred_count_raw = int(len(pred_points))
    pred_points = voxel_downsample(pred_points, args.pred_voxel_size)
    pred_points = deterministic_subsample(pred_points, args.max_pred_points, args.seed)
    if pred_colors is not None and len(pred_colors) != len(pred_points):
        pred_colors = None
    pred_count_eval = int(len(pred_points))

    gt_vertices, gt_triangles = load_triangle_mesh(Path(args.gt_mesh))
    gt_points = sample_mesh_surface(gt_vertices, gt_triangles, args.sample_gt_points, args.seed)

    alignment_stats = {"mode": args.align}
    raw_pred_bbox = bbox(pred_points)
    if args.align == "centroid":
        transform = centroid_transform(pred_points.astype(np.float64), gt_points.astype(np.float64))
        pred_points = apply_transform(pred_points.astype(np.float64), transform).astype(np.float32)
        alignment_stats.update(
            {
                "transform_pred_to_gt": transform.round(9).tolist(),
                "translation_norm": float(np.linalg.norm(transform[:3, 3])),
            }
        )
    elif args.align == "icp":
        transform, icp_stats = refine_icp_transform(
            pred_points,
            gt_points,
            initial_transform=None,
            seed=args.seed,
            sample_points=args.icp_sample_points,
            iterations=args.icp_iterations,
            trim_quantile=args.icp_trim_quantile,
            max_correspondence=args.icp_max_correspondence,
        )
        pred_points = apply_transform(pred_points.astype(np.float64), transform).astype(np.float32)
        alignment_stats.update(icp_stats)
        alignment_stats.update(
            {
                "transform_pred_to_gt": transform.round(9).tolist(),
                "translation_norm": float(np.linalg.norm(transform[:3, 3])),
            }
        )
    elif args.align in {"pose_rigid", "pose_rigid_icp", "pose_similarity", "pose_similarity_icp"}:
        pred_pose_dir = args.pose_align_pred_sequence_dir or args.pred_sequence_dir
        if pred_pose_dir is None:
            raise ValueError(
                f"--align {args.align} requires --pose-align-pred-sequence-dir "
                "when prediction input is not --pred-sequence-dir"
            )
        gt_pose_dir = Path(args.pose_align_gt_sequence_dir) if args.pose_align_gt_sequence_dir else None
        gt_pose_zip = Path(args.pose_align_gt_sequence_zip) if args.pose_align_gt_sequence_zip else None
        if gt_pose_dir is None and gt_pose_zip is None:
            raise ValueError(
                f"--align {args.align} requires --pose-align-gt-sequence-dir "
                "or --pose-align-gt-sequence-zip"
            )
        transform, pose_stats = pose_alignment_transform(
            pred_sequence_dir=Path(pred_pose_dir),
            gt_sequence_dir=gt_pose_dir,
            gt_sequence_zip=gt_pose_zip,
            frame_stride=args.pose_align_frame_stride,
            max_frames=args.pose_align_max_frames,
            min_pairs=args.pose_align_min_pairs,
            with_scale=args.align in {"pose_similarity", "pose_similarity_icp"},
        )
        alignment_stats["pose"] = pose_stats
        if args.align in {"pose_rigid_icp", "pose_similarity_icp"}:
            transform, icp_stats = refine_icp_transform(
                pred_points,
                gt_points,
                initial_transform=transform,
                seed=args.seed,
                sample_points=args.icp_sample_points,
                iterations=args.icp_iterations,
                trim_quantile=args.icp_trim_quantile,
                max_correspondence=args.icp_max_correspondence,
            )
            alignment_stats["icp"] = icp_stats
        pred_points = apply_transform(pred_points.astype(np.float64), transform).astype(np.float32)
        alignment_stats.update(pose_stats)
        alignment_stats.update(
            {
                "transform_pred_to_gt": transform.round(9).tolist(),
                "linear_determinant": float(np.linalg.det(transform[:3, :3])),
                "effective_uniform_scale": float(
                    abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0)
                ),
                "translation_norm": float(np.linalg.norm(transform[:3, 3])),
            }
        )
    elif args.align in {"pca", "pca_icp"}:
        transform, pca_stats = pca_alignment_transform(
            pred_points,
            gt_points,
            seed=args.seed,
            sample_points=args.icp_sample_points,
            scale_mode=args.align_scale,
            allow_reflection=args.allow_reflection,
        )
        alignment_stats.update(pca_stats)
        if args.align == "pca_icp":
            transform, icp_stats = refine_icp_transform(
                pred_points,
                gt_points,
                initial_transform=transform,
                seed=args.seed,
                sample_points=args.icp_sample_points,
                iterations=args.icp_iterations,
                trim_quantile=args.icp_trim_quantile,
                max_correspondence=args.icp_max_correspondence,
            )
            alignment_stats["icp"] = icp_stats
        pred_points = apply_transform(pred_points.astype(np.float64), transform).astype(np.float32)
        alignment_stats.update(
            {
                "transform_pred_to_gt": transform.round(9).tolist(),
                "linear_determinant": float(np.linalg.det(transform[:3, :3])),
                "effective_uniform_scale": float(
                    abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0)
                ),
                "translation_norm": float(np.linalg.norm(transform[:3, 3])),
            }
        )
    elif args.align in {"axis_bbox", "axis_bbox_icp"}:
        transform, axis_stats = axis_bbox_alignment_transform(
            pred_points,
            gt_points,
            seed=args.seed,
            sample_points=args.icp_sample_points,
            refine_with_icp=args.align == "axis_bbox_icp",
            refine_top_k=args.axis_bbox_refine_top_k,
            iterations=args.icp_iterations,
            trim_quantile=args.icp_trim_quantile,
            max_correspondence=args.icp_max_correspondence,
        )
        pred_points = apply_transform(pred_points.astype(np.float64), transform).astype(np.float32)
        alignment_stats.update(axis_stats)
        alignment_stats.update(
            {
                "transform_pred_to_gt": transform.round(9).tolist(),
                "linear_determinant": float(np.linalg.det(transform[:3, :3])),
                "effective_uniform_scale": float(
                    abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0)
                ),
                "translation_norm": float(np.linalg.norm(transform[:3, 3])),
            }
        )
    elif args.align in {"coord_yz_flip", "coord_yz_flip_icp"}:
        rotation = np.diag([1.0, -1.0, -1.0])
        transform, scale = bbox_oriented_transform(
            pred_points,
            gt_points,
            rotation,
            scale_mode="none",
            low_q=args.coord_bbox_low_q,
            high_q=args.coord_bbox_high_q,
        )
        alignment_stats.update(
            {
                "coordinate_conversion": "x,-y,-z",
                "bbox_low_q": float(args.coord_bbox_low_q),
                "bbox_high_q": float(args.coord_bbox_high_q),
                "scale": float(scale),
            }
        )
        if args.align == "coord_yz_flip_icp":
            transform, icp_stats = refine_icp_transform(
                pred_points,
                gt_points,
                initial_transform=transform,
                seed=args.seed,
                sample_points=args.icp_sample_points,
                iterations=args.icp_iterations,
                trim_quantile=args.icp_trim_quantile,
                max_correspondence=args.icp_max_correspondence,
            )
            alignment_stats["icp"] = icp_stats
        pred_points = apply_transform(pred_points.astype(np.float64), transform).astype(np.float32)
        alignment_stats.update(
            {
                "transform_pred_to_gt": transform.round(9).tolist(),
                "linear_determinant": float(np.linalg.det(transform[:3, :3])),
                "effective_uniform_scale": float(
                    abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0)
                ),
                "translation_norm": float(np.linalg.norm(transform[:3, 3])),
            }
        )
    elif args.align in {"rgbd_correspondence", "rgbd_correspondence_icp"}:
        if not args.rgbd_pred_sequence_dir:
            raise ValueError(f"--align {args.align} requires --rgbd-pred-sequence-dir")
        gt_pose_dir = Path(args.rgbd_gt_sequence_dir) if args.rgbd_gt_sequence_dir else None
        gt_pose_zip = Path(args.rgbd_gt_sequence_zip) if args.rgbd_gt_sequence_zip else None
        if gt_pose_dir is None and gt_pose_zip is None:
            raise ValueError(
                f"--align {args.align} requires --rgbd-gt-sequence-dir or --rgbd-gt-sequence-zip"
            )
        transform, rgbd_stats = rgbd_correspondence_alignment_transform(
            pred_sequence_dir=Path(args.rgbd_pred_sequence_dir),
            gt_sequence_dir=gt_pose_dir,
            gt_sequence_zip=gt_pose_zip,
            frame_stride=args.rgbd_frame_stride,
            max_frames=args.rgbd_max_frames,
            pixel_stride=args.rgbd_pixel_stride,
            conf_thr=args.rgbd_conf_thr,
            seed=args.seed,
            max_correspondences=args.rgbd_max_correspondences,
            trim_quantile=args.rgbd_trim_quantile,
            iterations=args.rgbd_fit_iterations,
        )
        alignment_stats.update(rgbd_stats)
        if args.align == "rgbd_correspondence_icp":
            transform, icp_stats = refine_icp_transform(
                pred_points,
                gt_points,
                initial_transform=transform,
                seed=args.seed,
                sample_points=args.icp_sample_points,
                iterations=args.icp_iterations,
                trim_quantile=args.icp_trim_quantile,
                max_correspondence=args.icp_max_correspondence,
            )
            alignment_stats["icp"] = icp_stats
        pred_points = apply_transform(pred_points.astype(np.float64), transform).astype(np.float32)
        alignment_stats.update(
            {
                "transform_pred_to_gt": transform.round(9).tolist(),
                "linear_determinant": float(np.linalg.det(transform[:3, :3])),
                "effective_uniform_scale": float(
                    abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0)
                ),
                "translation_norm": float(np.linalg.norm(transform[:3, 3])),
            }
        )

    pred_to_gt = nearest_distances(pred_points, gt_points)
    gt_to_pred_full = nearest_distances(gt_points, pred_points)

    pred_bbox_mask = mask_points_in_bbox(gt_points, pred_points, args.bbox_margin)
    scopes = {
        "full_gt": {
            "mask": np.ones(len(gt_points), dtype=bool),
            "description": "all sampled GT mesh surface points",
        },
        "pred_bbox_gt": {
            "mask": pred_bbox_mask,
            "description": "GT points inside pred bbox plus margin",
        },
    }

    visibility_stats = None
    if args.visible_gt_sequence_dir or args.visible_gt_sequence_zip:
        visible_mask, visibility_stats = visible_gt_mask(
            gt_points,
            sequence_dir=Path(args.visible_gt_sequence_dir) if args.visible_gt_sequence_dir else None,
            sequence_zip=Path(args.visible_gt_sequence_zip) if args.visible_gt_sequence_zip else None,
            frame_stride=args.visible_frame_stride,
            max_frames=args.visible_max_frames,
            depth_rtol=args.visible_depth_rtol,
            depth_atol=args.visible_depth_atol,
        )
        scopes["visible_gt"] = {
            "mask": visible_mask,
            "description": "GT points visible from the provided camera sequence",
        }

    metrics = {
        "scene_id": scene_id,
        "method_name": method_name,
        "pred_source": pred_source,
        "gt_mesh": str(Path(args.gt_mesh)),
        "units": "meters",
        "thresholds_m": [float(x) for x in args.thresholds],
        "report_threshold_m": float(args.report_threshold),
        "counts": {
            "pred_points_raw": pred_count_raw,
            "pred_points_eval": pred_count_eval,
            "gt_sample_points": int(len(gt_points)),
        },
        "sampling": {
            "seed": int(args.seed),
            "sample_gt_points": int(args.sample_gt_points),
            "pred_voxel_size": float(args.pred_voxel_size),
            "max_pred_points": int(args.max_pred_points),
        },
        "alignment": alignment_stats,
        "bbox": {
            "pred_raw": raw_pred_bbox,
            "pred_eval": bbox(pred_points),
            "gt_sample": bbox(gt_points),
            "pred_bbox_margin": float(args.bbox_margin),
        },
        "pred_to_gt": distance_summary(pred_to_gt),
        "scopes": {},
    }
    if visibility_stats is not None:
        metrics["visible_gt"] = visibility_stats

    for scope_name, scope in scopes.items():
        mask = scope["mask"]
        gt_distances = gt_to_pred_full[mask]
        metrics["scopes"][scope_name] = {
            "description": scope["description"],
            "gt_points": int(mask.sum()),
            "gt_fraction": float(mask.mean()) if len(mask) else 0.0,
            "gt_to_pred": distance_summary(gt_distances),
            "thresholds": threshold_metrics(pred_to_gt, gt_distances, args.thresholds),
            "chamfer_l1_mean": (
                float(np.mean(pred_to_gt) + np.mean(gt_distances))
                if len(pred_to_gt) and len(gt_distances)
                else None
            ),
        }

    metrics["report_rows"] = build_report_rows(metrics, args.report_threshold)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    write_metrics_csv(out_dir / "metrics.csv", metrics, args.report_threshold)
    write_report_markdown(out_dir / "report_summary.md", metrics, args.report_threshold)

    if args.write_debug_ply or args.write_debug_html:
        clip = max(args.thresholds) if args.thresholds else 0.10
        overlay_gt_mask = scopes.get("visible_gt", scopes["full_gt"])["mask"]
        overlay_gt = deterministic_subsample(gt_points[overlay_gt_mask], 300000, args.seed + 101)
        overlay_pred = deterministic_subsample(pred_points, 300000, args.seed + 102)
        if args.write_debug_html:
            write_overlay_html(
                out_dir / "overlay_pred_gt.html",
                overlay_pred,
                overlay_gt,
                max_points_per_cloud=args.debug_html_max_points,
                title=f"{scene_id} | {method_name} | {args.align}",
                details={
                    "scene_id": scene_id,
                    "method": method_name,
                    "pred_source": pred_source.get("type"),
                    "pred_path": pred_source.get("path"),
                    "gt_mesh": str(Path(args.gt_mesh)),
                    "pred_raw_points": pred_count_raw,
                    "pred_eval_points": pred_count_eval,
                },
            )
        if args.write_debug_ply:
            write_point_ply(out_dir / "pred_error_to_gt.ply", pred_points, error_colors(pred_to_gt, clip))
            overlay_points = np.concatenate([overlay_pred, overlay_gt], axis=0)
            overlay_colors = np.concatenate(
                [
                    np.tile(np.array([[255, 90, 30]], dtype=np.uint8), (len(overlay_pred), 1)),
                    np.tile(np.array([[60, 150, 255]], dtype=np.uint8), (len(overlay_gt), 1)),
                ],
                axis=0,
            )
            write_point_ply(out_dir / "overlay_pred_gt.ply", overlay_points, overlay_colors)
            for scope_name, scope in scopes.items():
                mask = scope["mask"]
                write_point_ply(
                    out_dir / f"gt_error_to_pred_{scope_name}.ply",
                    gt_points[mask],
                    error_colors(gt_to_pred_full[mask], clip),
                )

    print(f"[geometry-eval] wrote {metrics_path}")
    print(f"[geometry-eval] alignment={args.align}")
    if args.align != "none":
        print(
            "[geometry-eval] pred->gt transform translation norm: "
            f"{alignment_stats['translation_norm']:.4f} m"
        )
    def fmt(value) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    print(
        "[geometry-eval] pred->gt mean/median/p95: "
        f"{fmt(metrics['pred_to_gt']['mean'])} / "
        f"{fmt(metrics['pred_to_gt']['median'])} / "
        f"{fmt(metrics['pred_to_gt']['p95'])} m"
    )
    for scope_name, payload in metrics["scopes"].items():
        summary = payload["gt_to_pred"]
        print(
            f"[geometry-eval] {scope_name} gt->pred "
            f"n={payload['gt_points']} mean/median/p95="
            f"{fmt(summary['mean'])} / {fmt(summary['median'])} / {fmt(summary['p95'])} m"
        )
        for threshold_key, t_payload in payload["thresholds"].items():
            print(
                f"  @{threshold_key}: precision={t_payload['precision']:.3f} "
                f"recall={t_payload['recall']:.3f} fscore={t_payload['fscore']:.3f}"
            )


if __name__ == "__main__":
    main()
