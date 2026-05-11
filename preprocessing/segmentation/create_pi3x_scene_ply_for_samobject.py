#!/usr/bin/env python3
"""Create a Pi3X world-space scene point cloud PLY for SAM2Object graph clustering."""

import argparse
import json
import os
import os.path as osp
from glob import glob

import numpy as np
from plyfile import PlyData, PlyElement


def _camera_points_to_world(points_cam: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    rotation = pose_c2w[:3, :3].astype(np.float32)
    translation = pose_c2w[:3, 3].astype(np.float32)
    return (rotation @ points_cam.T).T + translation[None, :]


def write_point_ply(out_ply: str, points: np.ndarray) -> None:
    os.makedirs(osp.dirname(out_ply), exist_ok=True)
    vertex_data = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(out_ply)


def write_seg_json(out_json: str, seg_ids: np.ndarray) -> None:
    os.makedirs(osp.dirname(out_json), exist_ok=True)
    payload = {"segIndices": seg_ids.astype(np.int64).tolist()}
    with open(out_json, "w") as f:
        json.dump(payload, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--conf-thr", type=float, default=0.10)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--voxel-dedup-size", type=float, default=0.025)
    parser.add_argument("--superpoint-json-out", default=None)
    parser.add_argument("--superpoint-voxel-size", type=float, default=0.25)
    args = parser.parse_args()

    sequence_dir = args.sequence_dir
    xyz_paths = sorted(glob(osp.join(sequence_dir, "frame-*.xyz.npy")))
    if not xyz_paths:
        raise FileNotFoundError(f"No Pi3X xyz maps found in {sequence_dir}")

    all_points = []
    for xyz_path in xyz_paths:
        fid = osp.basename(xyz_path).replace("frame-", "").replace(".xyz.npy", "")
        pose_path = osp.join(sequence_dir, f"frame-{fid}.pose.txt")
        conf_path = osp.join(sequence_dir, f"frame-{fid}.conf.npy")
        if not osp.exists(pose_path):
            continue

        xyz = np.load(xyz_path).astype(np.float32)
        pose_c2w = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
        conf = (
            np.load(conf_path).astype(np.float32)
            if osp.exists(conf_path)
            else np.ones(xyz.shape[:2], dtype=np.float32)
        )

        valid = np.isfinite(xyz).all(axis=-1)
        valid &= xyz[..., 2] > 0.0
        valid &= np.abs(xyz).sum(axis=-1) > 0.0
        valid &= conf > float(args.conf_thr)
        if not valid.any():
            continue

        if int(args.pixel_stride) > 1:
            stride_mask = np.zeros_like(valid, dtype=bool)
            stride_mask[:: int(args.pixel_stride), :: int(args.pixel_stride)] = True
            valid &= stride_mask
        if not valid.any():
            continue

        pts_cam = xyz[valid].astype(np.float32)
        pts_world = _camera_points_to_world(pts_cam, pose_c2w)
        all_points.append(pts_world)

    if not all_points:
        raise RuntimeError("No valid Pi3X world points found for scene PLY export")

    pts_all = np.concatenate(all_points, axis=0)
    bbox_min = pts_all.min(axis=0)
    bbox_max = pts_all.max(axis=0)
    print(
        f"[Pi3X scene ply] points before dedup={len(pts_all)} "
        f"bbox_min={bbox_min.tolist()} bbox_max={bbox_max.tolist()}"
    )

    voxel = np.floor(pts_all / float(args.voxel_dedup_size)).astype(np.int64)
    unique_voxel, unique_idx = np.unique(voxel, axis=0, return_index=True)
    pts_final = pts_all[unique_idx]
    print(
        f"[Pi3X scene ply] points after dedup={len(pts_final)} "
        f"voxel_dedup_size={float(args.voxel_dedup_size):.4f} unique_voxels={len(unique_voxel)}"
    )

    write_point_ply(args.out_ply, pts_final.astype(np.float32))
    if args.superpoint_json_out:
        super_voxel = np.floor(pts_final / float(args.superpoint_voxel_size)).astype(np.int64)
        _, inverse = np.unique(super_voxel, axis=0, return_inverse=True)
        write_seg_json(args.superpoint_json_out, inverse.astype(np.int32))
        print(
            f"[Pi3X scene ply] wrote pseudo-superpoints={int(inverse.max()) + 1 if inverse.size else 0} "
            f"superpoint_voxel_size={float(args.superpoint_voxel_size):.4f} "
            f"path={args.superpoint_json_out}"
        )
    print(args.out_ply)


if __name__ == "__main__":
    main()
