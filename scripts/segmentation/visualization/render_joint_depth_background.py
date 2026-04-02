#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import numpy as np
import trimesh
import trimesh.viewer
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.segmentation.visualization.render_depth_background import (
    build_background_from_depth,
    compute_bounds,
    render_frame,
    resolve_obj_ids,
    save_contact_sheet,
    subsample,
)

C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a combined scene using RGB-D background points and a U3DGS "
            "joint Gaussian PLY converted to a point cloud."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--replacement-root", required=True)
    parser.add_argument("--joint-ply", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--obj-id", dest="obj_ids", action="append", type=int, default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mask-source", default="gt_projection")
    parser.add_argument(
        "--background-remove-mode",
        default="loaded",
        choices=["all", "loaded"],
    )
    parser.add_argument("--mask-erode-px", type=int, default=3)
    parser.add_argument("--pose-mode", default="raw", choices=["raw", "invert"])
    parser.add_argument("--lift-coord-system", default="pinhole", choices=["scan3r", "pinhole"])
    parser.add_argument(
        "--frame-selection", default="all", choices=["all", "top_area", "diverse_area"]
    )
    parser.add_argument("--max-views", type=int, default=9999)
    parser.add_argument("--bg-max-points", type=int, default=250000)
    parser.add_argument("--joint-max-points", type=int, default=250000)
    parser.add_argument("--joint-opacity-min", type=float, default=0.0)
    parser.add_argument("--joint-opacity-quantile", type=float, default=0.8)
    parser.add_argument("--joint-scale-quantile", type=float, default=0.95)
    parser.add_argument("--fit-mode", default="all", choices=["objects", "all"])
    parser.add_argument("--fit-margin", type=float, default=1.08)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--fig-height", type=float, default=7.0)
    parser.add_argument("--bg-point-size", type=float, default=0.35)
    parser.add_argument("--joint-point-size", type=float, default=1.0)
    parser.add_argument("--label", default="joint_depth_bg")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_loaded_ids(replacement_root: Path, scan_id: str, obj_ids: list[int]) -> list[int]:
    loaded = []
    for obj_id in obj_ids:
        obj_dir = replacement_root / "files" / "gs_annotations" / scan_id / str(obj_id)
        if (obj_dir / "voxel_output_dense.npz").exists() and (
            obj_dir / "mean_scale_dense.npz"
        ).exists():
            loaded.append(int(obj_id))
    return loaded


def load_joint_points(
    joint_ply: Path,
    *,
    opacity_min: float,
    opacity_quantile: float,
    scale_quantile: float,
):
    ply = PlyData.read(str(joint_ply))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    sh0 = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(
        np.float32
    )
    colors = np.clip(sh0 * C0 + 0.5, 0.0, 1.0)
    opacity_logit = np.asarray(vertex["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    scales = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
    ).astype(np.float32)
    scale_norm = np.linalg.norm(np.exp(scales), axis=1)

    mask = np.ones(len(xyz), dtype=bool)
    if opacity_min > 0.0:
        mask &= opacity >= float(opacity_min)
    if 0.0 < opacity_quantile < 1.0 and len(opacity) > 1:
        mask &= opacity >= np.quantile(opacity, opacity_quantile)
    if 0.0 < scale_quantile < 1.0 and len(scale_norm) > 1:
        mask &= scale_norm <= np.quantile(scale_norm, scale_quantile)

    return (
        xyz[mask].astype(np.float32),
        colors[mask].astype(np.float32),
        {
            "total_points": int(len(xyz)),
            "kept_points": int(mask.sum()),
            "opacity_quantile_threshold": float(np.quantile(opacity, opacity_quantile))
            if 0.0 < opacity_quantile < 1.0 and len(opacity) > 1
            else None,
            "scale_quantile_threshold": float(np.quantile(scale_norm, scale_quantile))
            if 0.0 < scale_quantile < 1.0 and len(scale_norm) > 1
            else None,
        },
    )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    replacement_root = Path(args.replacement_root)
    joint_ply = Path(args.joint_ply)
    obj_ids = resolve_obj_ids(args)
    loaded_ids = load_loaded_ids(replacement_root, args.scan_id, obj_ids)
    remove_obj_ids = obj_ids if args.background_remove_mode == "all" else loaded_ids

    obj_slug = "-".join(str(x) for x in obj_ids[:6])
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("vis")
        / "rendered_joint_depth_bg"
        / f"{args.scan_id}_{obj_slug}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_points, bg_colors, selected_frame_ids = build_background_from_depth(
        data_root=data_root,
        scan_id=args.scan_id,
        remove_obj_ids=remove_obj_ids,
        mask_source=args.mask_source,
        pose_mode=args.pose_mode,
        lift_coord_system=args.lift_coord_system,
        frame_selection=args.frame_selection,
        max_views=args.max_views,
        mask_erode_px=args.mask_erode_px,
    )
    joint_points, joint_colors, joint_stats = load_joint_points(
        joint_ply,
        opacity_min=args.joint_opacity_min,
        opacity_quantile=args.joint_opacity_quantile,
        scale_quantile=args.joint_scale_quantile,
    )

    bg_points, bg_colors = subsample(bg_points, bg_colors, args.bg_max_points)
    joint_points, joint_colors = subsample(
        joint_points, joint_colors, args.joint_max_points
    )
    center, half_extent = compute_bounds(
        bg_points, joint_points, args.fit_mode, args.fit_margin
    )

    rendered = []
    preview = []
    for frame_idx in range(args.num_frames):
        azim = 360.0 * frame_idx / args.num_frames
        frame = render_frame(
            bg_points,
            bg_colors,
            joint_points,
            joint_colors,
            center,
            half_extent,
            azim=azim,
            elev=18.0,
            bg_point_size=args.bg_point_size,
            obj_point_size=args.joint_point_size,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
        )
        rendered.append(frame)
        if frame_idx in {
            0,
            args.num_frames // 4,
            args.num_frames // 2,
            (3 * args.num_frames) // 4,
        }:
            preview.append((frame_idx, frame))

    video_path = out_dir / f"{args.scan_id}_joint_depth_bg_{args.label.replace(' ', '_')}.mp4"
    imageio.mimsave(video_path, rendered, fps=args.fps)
    save_contact_sheet(preview, out_dir / "contact_sheet.png")

    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.points.PointCloud(bg_points, colors=(255.0 * bg_colors).astype(np.uint8)),
        geom_name="depth_background",
    )
    scene.add_geometry(
        trimesh.points.PointCloud(
            joint_points, colors=(255.0 * joint_colors).astype(np.uint8)
        ),
        geom_name="u3dgs_joint",
    )
    bounds = scene.bounds
    scene.set_camera(
        angles=(0.6, 0.0, 0.6),
        distance=max(float(np.max(bounds[1] - bounds[0])) * 1.6, 1.0),
        center=bounds.mean(axis=0),
    )
    html_path = out_dir / f"{args.scan_id}_interactive_{args.label.replace(' ', '_')}.html"
    html_path.write_text(trimesh.viewer.scene_to_html(scene), encoding="utf-8")

    summary = {
        "data_root": str(data_root),
        "replacement_root": str(replacement_root),
        "joint_ply": str(joint_ply),
        "scan_id": args.scan_id,
        "obj_ids": obj_ids,
        "loaded_obj_ids": loaded_ids,
        "background_removed_obj_ids": remove_obj_ids,
        "selected_frame_ids": selected_frame_ids,
        "bg_points": int(len(bg_points)),
        "joint_points": int(len(joint_points)),
        "mask_source": args.mask_source,
        "background_remove_mode": args.background_remove_mode,
        "mask_erode_px": int(args.mask_erode_px),
        "pose_mode": args.pose_mode,
        "lift_coord_system": args.lift_coord_system,
        "joint_filter": {
            "opacity_min": float(args.joint_opacity_min),
            "opacity_quantile": float(args.joint_opacity_quantile),
            "scale_quantile": float(args.joint_scale_quantile),
            **joint_stats,
        },
        "video": str(video_path),
        "html": str(html_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(video_path)
    print(html_path)


if __name__ == "__main__":
    main()
