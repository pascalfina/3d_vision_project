#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.voxel_anno import voxelise_features as vf
from utils import scan3r


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Debug a single object in world space by comparing GT voxels, "
            "lifted points, and fused voxels."
        )
    )
    parser.add_argument("--data-root", required=True, help="Predseg/object-level root.")
    parser.add_argument("--baseline-root", required=True, help="Baseline GT root.")
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--obj-id", required=True, type=int)
    parser.add_argument("--label", default="object")
    parser.add_argument("--mask-source", default="pred_projection_clean")
    parser.add_argument(
        "--pose-mode",
        default="raw",
        choices=["invert", "raw"],
        help=(
            "How to interpret frame poses before lifting: "
            "'invert' uses inv(extrinsic), 'raw' uses extrinsic directly."
        ),
    )
    parser.add_argument(
        "--lift-coord-system",
        default="pinhole",
        choices=["scan3r", "pinhole"],
        help=(
            "Coordinate system for lifting depth pixels. "
            "'scan3r' uses the custom [depth,-x,-y] convention, "
            "'pinhole' uses the generic intrinsic-inverse formulation."
        ),
    )
    parser.add_argument(
        "--frame-selection",
        default="diverse_area",
        choices=["all", "top_area", "diverse_area"],
    )
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument(
        "--object-source",
        default="tsdf_masks",
        choices=["lifted_masks", "tsdf_masks", "hybrid_masks"],
    )
    parser.add_argument(
        "--reference-mean-scale-root",
        default=None,
        help="Usually the baseline root, to keep the world placement comparable.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to vis/worldspace_debug/<scan>_<obj>/",
    )
    return parser.parse_args()


def stage_scan(src_scan_dir: Path, dst_scan_dir: Path) -> None:
    if dst_scan_dir.exists():
        shutil.rmtree(dst_scan_dir)
    dst_scan_dir.mkdir(parents=True, exist_ok=True)

    for item in src_scan_dir.iterdir():
        if item.name in {"sequence.zip", "sequence"}:
            continue
        os.symlink(item, dst_scan_dir / item.name)

    src_zip = src_scan_dir / "sequence.zip"
    src_seq = src_scan_dir / "sequence"
    dst_seq = dst_scan_dir / "sequence"
    if src_zip.exists():
        dst_seq.mkdir(exist_ok=True)
        subprocess.run(["unzip", "-qo", str(src_zip), "-d", str(dst_seq)], check=True)
    elif src_seq.exists():
        os.symlink(src_seq, dst_seq)


def resolve_scene_root(data_root: Path, baseline_root: Path, scan_id: str, stage_root: Path) -> Path:
    for root in [data_root, baseline_root]:
        info = root / "scenes" / scan_id / "sequence" / "_info.txt"
        if info.exists():
            return root
    for root in [data_root, baseline_root]:
        scan_dir = root / "scenes" / scan_id
        if (scan_dir / "sequence.zip").exists() or (scan_dir / "sequence").exists():
            staged = stage_root / "scenes" / scan_id
            if not (staged / "sequence" / "_info.txt").exists():
                stage_scan(scan_dir, staged)
            return stage_root
    return baseline_root


@contextmanager
def temp_env(overrides):
    sentinel = object()
    old = {}
    for key, value in overrides.items():
        old[key] = os.environ.get(key, sentinel)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_visible_frames(masks: dict, frame_ids: list[str], obj_id: int):
    selected_ids = []
    selected_masks = []
    for frame_id in frame_ids:
        obj_mask = np.where(masks[frame_id] == int(obj_id), 1, 0).astype(np.uint8)
        if obj_mask.sum() > 0:
            selected_ids.append(frame_id)
            selected_masks.append(obj_mask)
    return selected_ids, selected_masks


def voxel_to_world(voxel_indices: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    voxel = voxel_indices.astype(np.float32) * (1.0 / 64.0)
    voxel = voxel * 2.0 - 1.0
    return voxel * float(scale) + mean[None, :]


def normalize_with_reference(points: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    normalized = (points - mean[None, :]) * (1.0 / (2.0 * float(scale)))
    return np.clip(normalized, -0.5 + 1e-6, 0.5 - 1e-6).astype(np.float32)


def save_colored_points(points: np.ndarray, colors: np.ndarray, output_file: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(output_file), pcd)


def subsample(points: np.ndarray, max_points: int = 30000) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def plot_triptych(gt_world, lifted_world, fused_world, out_file: Path) -> None:
    fig = plt.figure(figsize=(12, 4))
    items = [
        ("GT voxels", gt_world, "#2ca02c"),
        ("Lifted points", lifted_world, "#1f77b4"),
        ("Fused voxels", fused_world, "#d62728"),
    ]
    for idx, (title, pts, color) in enumerate(items, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        pts_s = subsample(pts)
        ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2], s=1.0, c=color, depthshade=False)
        ax.set_title(title)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(gt_world, lifted_world, fused_world, out_file: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for pts, color, label in [
        (gt_world, "#2ca02c", "GT"),
        (lifted_world, "#1f77b4", "lifted"),
        (fused_world, "#d62728", "fused"),
    ]:
        pts_s = subsample(pts)
        ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2], s=1.0, c=color, depthshade=False, label=label)
    ax.legend(loc="upper right")
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    baseline_root = Path(args.baseline_root)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "vis" / "worldspace_debug" / f"{args.scan_id}_{args.obj_id}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="objectx-worlddebug-") as tmpdir:
        stage_root = Path(tmpdir)
        scene_root = resolve_scene_root(data_root, baseline_root, args.scan_id, stage_root)
        scenes_dir = scene_root / "scenes"

        frame_ids = scan3r.load_frame_idxs(data_dir=str(scenes_dir), scan_id=args.scan_id)
        extrinsics = scan3r.load_frame_poses(
            data_dir=str(scene_root), scan_id=args.scan_id, frame_idxs=frame_ids
        )
        depth_intrinsics = scan3r.load_intrinsics(
            data_dir=str(scenes_dir), scan_id=args.scan_id, type="depth"
        )
        depth_shift = vf._load_depth_shift(str(scenes_dir), args.scan_id)
        masks = scan3r.load_masks(str(data_root), args.scan_id, mask_source=args.mask_source)

        vis_frame_ids, vis_masks = build_visible_frames(masks, frame_ids, args.obj_id)
        vis_frame_ids, vis_masks = vf._filter_selected_masks(
            vis_frame_ids, vis_masks, object_source=args.object_source
        )

        with temp_env(
            {
                "OBJECTX_VOXEL_FRAME_SELECTION": args.frame_selection,
                "OBJECTX_VOXEL_LIFT_COORD_SYSTEM": args.lift_coord_system,
                "OBJECTX_VOXEL_REFERENCE_MEAN_SCALE_ROOT": str(
                    Path(args.reference_mean_scale_root or baseline_root)
                ),
            }
        ):
            selected_frame_ids, selected_masks = vf._select_object_frames(
                vis_frame_ids,
                vis_masks,
                extrinsics=extrinsics,
                max_views=args.max_views,
            )

            selected_depths = []
            for frame_id in selected_frame_ids:
                selected_depths.append(
                    scan3r.load_depth_map(
                        str(
                            scenes_dir
                            / args.scan_id
                            / "sequence"
                            / f"frame-{frame_id}.depth.pgm"
                        ),
                        depth_shift,
                    )
                )
            if args.pose_mode == "invert":
                pose_camera_to_world = [
                    np.linalg.inv(extrinsics[fid]) for fid in selected_frame_ids
                ]
            else:
                pose_camera_to_world = [
                    np.array(extrinsics[fid], copy=True) for fid in selected_frame_ids
                ]

            vf.args = argparse.Namespace(visualize=False)

            mean_scale = np.load(
                baseline_root
                / "files"
                / "gs_annotations"
                / args.scan_id
                / str(args.obj_id)
                / "mean_scale_dense.npz"
            )
            gt_voxels = np.load(
                baseline_root
                / "files"
                / "gs_annotations"
                / args.scan_id
                / str(args.obj_id)
                / "voxel_output_dense.npz"
            )["arr_0"][:, :3].astype(np.int32)
            gt_mean = mean_scale["mean"].astype(np.float32)
            gt_scale = float(mean_scale["scale"])

            lifted_world = vf._lift_masked_points(
                selected_masks=selected_masks,
                selected_depths=selected_depths,
                pose_camera_to_world=pose_camera_to_world,
                depth_intrinsics=depth_intrinsics,
            )
            lifted_norm = normalize_with_reference(lifted_world, gt_mean, gt_scale)
            lifted_voxels = vf._voxelize_normalized_points(lifted_norm, dilate_iters=1)
            lifted_voxels = vf._keep_largest_voxel_component(lifted_voxels)

            if args.object_source == "tsdf_masks":
                fused_voxels, fused_mean, fused_scale = vf._build_tsdf_object_voxel_grid(
                    scan_id=args.scan_id,
                    obj_id=args.obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )
            elif args.object_source == "hybrid_masks":
                fused_voxels, fused_mean, fused_scale = vf._build_hybrid_object_voxel_grid(
                    scan_id=args.scan_id,
                    obj_id=args.obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )
            else:
                fused_voxels, fused_mean, fused_scale = vf._build_lifted_object_voxel_grid(
                    scan_id=args.scan_id,
                    obj_id=args.obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )

        gt_world = voxel_to_world(gt_voxels, gt_mean, gt_scale)
        fused_world = voxel_to_world(fused_voxels, fused_mean, fused_scale)
        lifted_voxel_world = voxel_to_world(lifted_voxels, gt_mean, gt_scale)

        save_colored_points(gt_world, np.tile(np.array([[0.1, 0.8, 0.1]]), (len(gt_world), 1)), out_dir / "gt_voxels_world.ply")
        save_colored_points(lifted_world, np.tile(np.array([[0.1, 0.4, 1.0]]), (len(lifted_world), 1)), out_dir / "lifted_points_world.ply")
        save_colored_points(fused_world, np.tile(np.array([[1.0, 0.2, 0.2]]), (len(fused_world), 1)), out_dir / "fused_voxels_world.ply")

        overlay_points = np.concatenate([gt_world, lifted_world, fused_world], axis=0)
        overlay_colors = np.concatenate(
            [
                np.tile(np.array([[0.1, 0.8, 0.1]]), (len(gt_world), 1)),
                np.tile(np.array([[0.1, 0.4, 1.0]]), (len(lifted_world), 1)),
                np.tile(np.array([[1.0, 0.2, 0.2]]), (len(fused_world), 1)),
            ],
            axis=0,
        )
        save_colored_points(overlay_points, overlay_colors, out_dir / "overlay_world.ply")

        plot_triptych(gt_world, lifted_world, fused_world, out_dir / "triptych.png")
        plot_overlay(gt_world, lifted_world, fused_world, out_dir / "overlay.png")

        gt_set = {tuple(map(int, row)) for row in gt_voxels}
        lifted_set = {tuple(map(int, row)) for row in lifted_voxels}
        fused_set = {tuple(map(int, row)) for row in fused_voxels}

        summary = {
            "scan_id": args.scan_id,
            "obj_id": int(args.obj_id),
            "label": args.label,
            "mask_source": args.mask_source,
            "pose_mode": args.pose_mode,
            "lift_coord_system": args.lift_coord_system,
            "frame_selection": args.frame_selection,
            "max_views": int(args.max_views),
            "selected_frames": selected_frame_ids,
            "counts": {
                "gt_voxels": int(len(gt_set)),
                "lifted_points": int(lifted_world.shape[0]),
                "lifted_voxels": int(len(lifted_set)),
                "fused_voxels": int(len(fused_set)),
            },
            "overlap": {
                "lifted_voxel_vs_gt": int(len(lifted_set & gt_set)),
                "fused_voxel_vs_gt": int(len(fused_set & gt_set)),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Saved world-space debug outputs to {out_dir}")


if __name__ == "__main__":
    main()
