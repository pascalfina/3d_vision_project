#!/usr/bin/env python3
"""Refine Pi3X support superpoints with SAMObject 2D mask evidence.

The first Pi3X support scene step builds geometry/color/normal superpoints before
SAMObject has produced 2D tracks.  On noisy Pi3X clouds those primitives can span
several 2D object hypotheses, which makes SAMObject's later graph clustering
over-merge.  This script runs after ``mask_convert.py`` and splits only those
geometry superpoints whose points have conflicting SAMObject mask signatures.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from collections import Counter
from glob import glob

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree


def load_points_from_ply(path: str) -> np.ndarray:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)


def load_frame_ids(mask_dir: str, posed_images_dir: str, view_stride: int) -> list[str]:
    frame_ids = []
    for mask_path in sorted(glob(osp.join(mask_dir, "maskraw_*.png"))):
        frame_id = osp.basename(mask_path).replace("maskraw_", "").replace(".png", "")
        if (
            osp.exists(osp.join(posed_images_dir, f"{frame_id}.jpg"))
            and osp.exists(osp.join(posed_images_dir, f"{frame_id}.png"))
            and osp.exists(osp.join(posed_images_dir, f"{frame_id}.txt"))
        ):
            frame_ids.append(frame_id)
    stride = max(1, int(view_stride))
    return frame_ids[::stride]


def load_intrinsic(path: str) -> np.ndarray:
    return np.loadtxt(path).astype(np.float32)[:3, :3]


def load_depth(path: str) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 1000.0


def load_mask(path: str) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32, copy=False)


def collect_point_mask_evidence(
    points_world: np.ndarray,
    *,
    posed_images_dir: str,
    mask_dir: str,
    view_stride: int,
    vis_rtol: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    frame_ids = load_frame_ids(mask_dir, posed_images_dir, view_stride)
    if not frame_ids:
        raise FileNotFoundError(f"No usable mask/posed-image frames found in {mask_dir}")

    color_intrinsic = load_intrinsic(osp.join(posed_images_dir, "intrinsics_color.txt"))
    depth_intrinsic = load_intrinsic(osp.join(posed_images_dir, "intrinsics_depth.txt"))
    point_labels = np.zeros((len(points_world), len(frame_ids)), dtype=np.int32)
    seen_counts = np.zeros(len(points_world), dtype=np.int32)
    points_h = np.concatenate(
        [points_world.astype(np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    visible_total = 0
    labeled_total = 0

    for frame_pos, frame_id in enumerate(frame_ids):
        pose = np.loadtxt(osp.join(posed_images_dir, f"{frame_id}.txt")).astype(np.float32).reshape(4, 4)
        world_to_camera = np.linalg.inv(pose).astype(np.float32)
        points_cam_h = (world_to_camera @ points_h.T).T
        points_cam = points_cam_h[:, :3] / np.clip(points_cam_h[:, 3:4], 1e-8, np.inf)
        z = points_cam[:, 2]

        valid_z = np.isfinite(points_cam).all(axis=1) & (z > 0.0)
        color_pix_h = (color_intrinsic @ points_cam.T).T
        depth_pix_h = (depth_intrinsic @ points_cam.T).T
        color_uv = np.zeros((len(points_world), 2), dtype=np.int32)
        depth_uv = np.zeros((len(points_world), 2), dtype=np.int32)
        color_uv[valid_z] = np.rint(
            color_pix_h[valid_z, :2] / np.clip(color_pix_h[valid_z, 2:3], 1e-8, np.inf)
        ).astype(np.int32)
        depth_uv[valid_z] = np.rint(
            depth_pix_h[valid_z, :2] / np.clip(depth_pix_h[valid_z, 2:3], 1e-8, np.inf)
        ).astype(np.int32)

        mask = load_mask(osp.join(mask_dir, f"maskraw_{frame_id}.png"))
        depth = load_depth(osp.join(posed_images_dir, f"{frame_id}.png"))
        mh, mw = mask.shape[:2]
        dh, dw = depth.shape[:2]
        cu, cv = color_uv[:, 0], color_uv[:, 1]
        du, dv = depth_uv[:, 0], depth_uv[:, 1]
        bounded = (
            valid_z
            & (cu >= 0)
            & (cu < mw)
            & (cv >= 0)
            & (cv < mh)
            & (du >= 0)
            & (du < dw)
            & (dv >= 0)
            & (dv < dh)
        )
        if not np.any(bounded):
            continue

        bounded_idx = np.flatnonzero(bounded)
        captured_depth = depth[dv[bounded_idx], du[bounded_idx]]
        visible = np.isclose(z[bounded_idx], captured_depth, rtol=float(vis_rtol))
        visible &= captured_depth > 0.0
        if not np.any(visible):
            continue

        visible_idx = bounded_idx[visible]
        labels = mask[cv[visible_idx], cu[visible_idx]]
        seen_counts[visible_idx] += 1
        visible_total += int(visible_idx.size)
        valid_label = labels > 0
        if np.any(valid_label):
            point_labels[visible_idx[valid_label], frame_pos] = labels[valid_label]
            labeled_total += int(valid_label.sum())

    stats = {
        "frames": len(frame_ids),
        "view_stride": int(view_stride),
        "visible_point_observations": visible_total,
        "labeled_point_observations": labeled_total,
        "seen_count_quantiles": np.percentile(
            seen_counts, [0, 10, 25, 50, 75, 90, 99, 100]
        ).round(3).tolist(),
        "nonzero_label_count_quantiles": np.percentile(
            (point_labels > 0).sum(axis=1), [0, 10, 25, 50, 75, 90, 99, 100]
        ).round(3).tolist(),
    }
    return point_labels, seen_counts, stats


def dominant_point_signatures(
    point_labels: np.ndarray,
    *,
    min_observations: int,
    min_dominant_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    signatures = np.zeros(point_labels.shape[0], dtype=np.int32)
    dominant_counts = np.zeros(point_labels.shape[0], dtype=np.int32)
    nonzero_counts = (point_labels > 0).sum(axis=1).astype(np.int32)

    for idx, row in enumerate(point_labels):
        nz = row[row > 0]
        if nz.size < min_observations:
            continue
        labels, counts = np.unique(nz, return_counts=True)
        best = int(np.argmax(counts))
        count = int(counts[best])
        ratio = float(count) / float(nz.size)
        if ratio >= min_dominant_ratio:
            signatures[idx] = int(labels[best])
            dominant_counts[idx] = count

    supported = signatures > 0
    ratios = np.zeros(point_labels.shape[0], dtype=np.float32)
    valid = nonzero_counts > 0
    ratios[valid] = dominant_counts[valid] / nonzero_counts[valid]
    stats = {
        "points_with_mask_signature": int(supported.sum()),
        "points_with_mask_signature_fraction": float(supported.mean()),
        "min_observations": int(min_observations),
        "min_dominant_ratio": float(min_dominant_ratio),
        "dominant_ratio_quantiles_supported": (
            np.percentile(ratios[supported], [0, 10, 25, 50, 75, 90, 100]).round(3).tolist()
            if np.any(supported)
            else []
        ),
    }
    return signatures, dominant_counts, nonzero_counts, stats


def remap_labels(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    mapping = {int(old): new for new, old in enumerate(unique)}
    out = np.empty_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == old] = int(new)
    return out


def split_superpoints_by_signature(
    points: np.ndarray,
    base_superpoints: np.ndarray,
    signatures: np.ndarray,
    *,
    min_split_points: int,
    min_signature_fraction: float,
    ambiguous_split_fraction: float,
) -> tuple[np.ndarray, dict]:
    refined = np.full(len(base_superpoints), -1, dtype=np.int32)
    next_id = 0
    split_superpoints = 0
    ambiguous_groups = 0
    signature_groups = 0

    for sp_id in np.unique(base_superpoints):
        members = np.flatnonzero(base_superpoints == sp_id)
        sp_signatures = signatures[members]
        positive = sp_signatures[sp_signatures > 0]
        if positive.size == 0:
            refined[members] = next_id
            next_id += 1
            continue

        counts = Counter(int(x) for x in positive)
        significant = [
            label
            for label, count in counts.items()
            if count >= min_split_points and count / float(len(members)) >= min_signature_fraction
        ]
        significant.sort(key=lambda label: counts[label], reverse=True)

        ambiguous_members = members[sp_signatures == 0]
        should_split_ambiguous = (
            len(significant) > 0
            and ambiguous_members.size >= min_split_points
            and ambiguous_members.size / float(len(members)) >= ambiguous_split_fraction
        )
        if len(significant) <= 1 and not should_split_ambiguous:
            refined[members] = next_id
            next_id += 1
            continue

        split_superpoints += 1
        assigned = np.zeros(len(members), dtype=bool)
        centroids = []
        centroid_labels = []
        for label in significant:
            local = np.flatnonzero(sp_signatures == label)
            global_members = members[local]
            if global_members.size == 0:
                continue
            refined[global_members] = next_id
            assigned[local] = True
            centroids.append(points[global_members].mean(axis=0))
            centroid_labels.append(next_id)
            next_id += 1
            signature_groups += 1

        if should_split_ambiguous:
            refined[ambiguous_members] = next_id
            assigned[sp_signatures == 0] = True
            next_id += 1
            ambiguous_groups += 1

        rest = members[~assigned]
        if rest.size > 0:
            if centroids:
                centroids_arr = np.stack(centroids, axis=0)
                tree = cKDTree(centroids_arr)
                _, nearest = tree.query(points[rest], k=1)
                nearest = np.asarray(nearest, dtype=np.int64)
                for pos, target_pos in enumerate(nearest):
                    refined[rest[pos]] = centroid_labels[int(target_pos)]
            else:
                refined[rest] = next_id
                next_id += 1

    if np.any(refined < 0):
        refined[refined < 0] = next_id
    refined = remap_labels(refined)
    unique, counts = np.unique(refined, return_counts=True)
    stats = {
        "base_superpoints": int(np.unique(base_superpoints).size),
        "refined_superpoints": int(unique.size),
        "split_base_superpoints": int(split_superpoints),
        "signature_groups_created": int(signature_groups),
        "ambiguous_groups_created": int(ambiguous_groups),
        "min_refined_size": int(counts.min()),
        "median_refined_size": float(np.median(counts)),
        "mean_refined_size": float(counts.mean()),
        "max_refined_size": int(counts.max()),
    }
    return refined.astype(np.int32), stats


def build_adjacency(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    radius: float,
    k: int,
) -> tuple[list[tuple[int, int]], dict]:
    tree = cKDTree(points)
    k = min(max(2, int(k)), len(points))
    dists, neighbors = tree.query(points, k=k, distance_upper_bound=float(radius), workers=-1)
    pairs: set[tuple[int, int]] = set()
    for idx in range(len(points)):
        label_i = int(labels[idx])
        for dist, nbr in zip(np.atleast_1d(dists[idx]), np.atleast_1d(neighbors[idx])):
            nbr = int(nbr)
            if nbr >= len(points) or nbr == idx or not np.isfinite(dist):
                continue
            label_j = int(labels[nbr])
            if label_i == label_j:
                continue
            pairs.add((min(label_i, label_j), max(label_i, label_j)))

    degrees = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for left, right in pairs:
        degrees[left] += 1
        degrees[right] += 1
    stats = {
        "superpoint_neighbors": int(len(pairs)),
        "adjacency_radius": float(radius),
        "adjacency_k": int(k),
        "degree_min": int(degrees.min()) if degrees.size else 0,
        "degree_median": float(np.median(degrees)) if degrees.size else 0.0,
        "degree_mean": float(degrees.mean()) if degrees.size else 0.0,
        "degree_max": int(degrees.max()) if degrees.size else 0,
        "isolated_superpoints": int((degrees == 0).sum()) if degrees.size else 0,
    }
    return sorted(pairs), stats


def write_superpoint_json(path: str, labels: np.ndarray) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"segIndices": labels.astype(np.int64).tolist()}, f)


def write_neighbors_json(path: str, pairs: list[tuple[int, int]]) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"superpointNeighbors": [[int(a), int(b)] for a, b in pairs]}, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--posed-images-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--superpoint-json-in", required=True)
    parser.add_argument("--superpoint-json-out", required=True)
    parser.add_argument("--superpoint-neighbors-json-out", required=True)
    parser.add_argument("--debug-json-out", default=None)
    parser.add_argument("--view-stride", type=int, default=1)
    parser.add_argument("--vis-rtol", type=float, default=0.15)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--min-dominant-ratio", type=float, default=0.45)
    parser.add_argument("--min-split-points", type=int, default=6)
    parser.add_argument("--min-signature-fraction", type=float, default=0.12)
    parser.add_argument("--ambiguous-split-fraction", type=float, default=0.25)
    parser.add_argument("--adjacency-radius", type=float, default=0.12)
    parser.add_argument("--adjacency-k", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points_from_ply(args.ply)
    with open(args.superpoint_json_in, "r") as f:
        base_superpoints = np.asarray(json.load(f)["segIndices"], dtype=np.int32)
    if base_superpoints.shape[0] != points.shape[0]:
        raise ValueError(
            f"Superpoint length {base_superpoints.shape[0]} does not match point count {points.shape[0]}"
        )

    point_labels, seen_counts, evidence_stats = collect_point_mask_evidence(
        points,
        posed_images_dir=args.posed_images_dir,
        mask_dir=args.mask_dir,
        view_stride=args.view_stride,
        vis_rtol=args.vis_rtol,
    )
    signatures, dominant_counts, nonzero_counts, signature_stats = dominant_point_signatures(
        point_labels,
        min_observations=int(args.min_observations),
        min_dominant_ratio=float(args.min_dominant_ratio),
    )
    refined_superpoints, split_stats = split_superpoints_by_signature(
        points,
        base_superpoints,
        signatures,
        min_split_points=int(args.min_split_points),
        min_signature_fraction=float(args.min_signature_fraction),
        ambiguous_split_fraction=float(args.ambiguous_split_fraction),
    )
    neighbor_pairs, adjacency_stats = build_adjacency(
        points,
        refined_superpoints,
        radius=float(args.adjacency_radius),
        k=int(args.adjacency_k),
    )

    write_superpoint_json(args.superpoint_json_out, refined_superpoints)
    write_neighbors_json(args.superpoint_neighbors_json_out, neighbor_pairs)

    payload = {
        "inputs": {
            "ply": args.ply,
            "posed_images_dir": args.posed_images_dir,
            "mask_dir": args.mask_dir,
            "superpoint_json_in": args.superpoint_json_in,
        },
        "evidence": evidence_stats,
        "point_signatures": signature_stats,
        "split": split_stats,
        "adjacency": adjacency_stats,
    }
    if args.debug_json_out:
        os.makedirs(osp.dirname(args.debug_json_out), exist_ok=True)
        with open(args.debug_json_out, "w") as f:
            json.dump(payload, f, indent=2)

    print(
        "[Pi3X mask-aware superpoints] "
        f"base={split_stats['base_superpoints']} refined={split_stats['refined_superpoints']} "
        f"split_base={split_stats['split_base_superpoints']} "
        f"signature_points={signature_stats['points_with_mask_signature']} "
        f"neighbors={adjacency_stats['superpoint_neighbors']}"
    )


if __name__ == "__main__":
    main()
