#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import trimesh.viewer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.segmentation.visualization.render_depth_background import (
    build_background_from_depth,
    resolve_obj_ids,
    subsample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export an interactive HTML scene view with background reconstructed "
            "from depth+pose and SLAT sparse coordinates projected into world space."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--slat-path", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--obj-id", dest="obj_ids", action="append", type=int, default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mask-source", default="gt_projection")
    parser.add_argument(
        "--background-remove-mode",
        default="loaded",
        choices=["all", "loaded"],
        help="Which object masks to remove from the depth background.",
    )
    parser.add_argument("--mask-erode-px", type=int, default=3)
    parser.add_argument("--pose-mode", default="raw", choices=["raw", "invert"])
    parser.add_argument("--lift-coord-system", default="pinhole", choices=["scan3r", "pinhole"])
    parser.add_argument(
        "--frame-selection", default="all", choices=["all", "top_area", "diverse_area"]
    )
    parser.add_argument("--max-views", type=int, default=9999)
    parser.add_argument("--max-bg-points", type=int, default=250000)
    parser.add_argument("--max-slat-points", type=int, default=250000)
    parser.add_argument("--label", default="slat_depth_bg_interactive")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def voxel_to_world(voxel_indices: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    voxel = voxel_indices.astype(np.float32) * (1.0 / 64.0)
    voxel = voxel * 2.0 - 1.0
    return voxel * float(scale) + mean[None, :]


def load_slat_points_with_colors(
    slat_path: Path, selected_obj_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    file = np.load(slat_path)
    coords = file["coords"].astype(np.int32)
    means = file["mean"].astype(np.float32)
    scales = file["scale"].astype(np.float32)
    obj_ids = file["obj_id"].astype(np.int64)

    palette = plt.get_cmap("hsv", len(obj_ids) + 1)
    points_all = []
    colors_all = []
    loaded = []

    for batch_idx, obj_id in enumerate(obj_ids.tolist()):
        if int(obj_id) not in selected_obj_ids:
            continue
        obj_mask = coords[:, 0] == batch_idx
        if not np.any(obj_mask):
            continue
        obj_vox = coords[obj_mask, 1:4]
        obj_world = voxel_to_world(obj_vox, means[batch_idx], float(scales[batch_idx]))
        color = np.array(palette(batch_idx)[:3], dtype=np.float32)
        obj_colors = np.tile(color[None, :], (len(obj_world), 1))
        points_all.append(obj_world)
        colors_all.append(obj_colors)
        loaded.append(int(obj_id))

    if not points_all:
        raise FileNotFoundError(f"No SLAT object coordinates found for {selected_obj_ids}")

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    return points, colors, loaded, int(coords.shape[0])


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    slat_path = Path(args.slat_path)
    obj_ids = resolve_obj_ids(args)
    obj_slug = "-".join(str(x) for x in obj_ids[:6])
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("vis")
        / "interactive_slat_views"
        / f"{args.scan_id}_{obj_slug}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    slat_points, slat_colors, loaded_ids, total_slat_coords = load_slat_points_with_colors(
        slat_path, obj_ids
    )
    remove_obj_ids = obj_ids if args.background_remove_mode == "all" else loaded_ids

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

    bg_points, bg_colors = subsample(bg_points, bg_colors, args.max_bg_points)
    slat_points, slat_colors = subsample(slat_points, slat_colors, args.max_slat_points)

    bg_cloud = trimesh.points.PointCloud(
        bg_points, colors=(255.0 * bg_colors).astype(np.uint8)
    )
    slat_cloud = trimesh.points.PointCloud(
        slat_points, colors=(255.0 * slat_colors).astype(np.uint8)
    )

    scene = trimesh.Scene()
    scene.add_geometry(bg_cloud, geom_name="depth_background")
    scene.add_geometry(slat_cloud, geom_name="slat_sparse_world")

    bounds = scene.bounds
    center = bounds.mean(axis=0)
    extent = np.max(bounds[1] - bounds[0])
    distance = max(float(extent) * 1.6, 1.0)
    scene.set_camera(angles=(0.6, 0.0, 0.6), distance=distance, center=center)

    html = trimesh.viewer.scene_to_html(scene)
    html_path = out_dir / f"{args.scan_id}_interactive_{args.label.replace(' ', '_')}.html"
    html_path.write_text(html, encoding="utf-8")

    summary = {
        "data_root": str(data_root),
        "slat_path": str(slat_path),
        "scan_id": args.scan_id,
        "obj_ids": obj_ids,
        "loaded_obj_ids": loaded_ids,
        "background_removed_obj_ids": remove_obj_ids,
        "selected_frame_ids": selected_frame_ids,
        "bg_points": int(len(bg_points)),
        "slat_points": int(len(slat_points)),
        "total_slat_coords": total_slat_coords,
        "mask_source": args.mask_source,
        "background_remove_mode": args.background_remove_mode,
        "mask_erode_px": int(args.mask_erode_px),
        "pose_mode": args.pose_mode,
        "lift_coord_system": args.lift_coord_system,
        "html": str(html_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(html_path)


if __name__ == "__main__":
    main()
