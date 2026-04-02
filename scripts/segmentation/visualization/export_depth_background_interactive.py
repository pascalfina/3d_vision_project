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
    load_objects_with_colors,
    resolve_obj_ids,
    subsample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export an interactive HTML scene view with background reconstructed "
            "from depth+pose and replacement objects loaded from gs_annotations."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--replacement-root", required=True)
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
    parser.add_argument("--mask-erode-px", type=int, default=2)
    parser.add_argument("--pose-mode", default="raw", choices=["raw", "invert"])
    parser.add_argument("--lift-coord-system", default="pinhole", choices=["scan3r", "pinhole"])
    parser.add_argument(
        "--frame-selection", default="diverse_area", choices=["all", "top_area", "diverse_area"]
    )
    parser.add_argument("--max-views", type=int, default=48)
    parser.add_argument("--max-bg-points", type=int, default=80000)
    parser.add_argument("--max-obj-points", type=int, default=40000)
    parser.add_argument("--label", default="depth_bg_interactive")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    replacement_root = Path(args.replacement_root)
    obj_ids = resolve_obj_ids(args)
    obj_slug = "-".join(str(x) for x in obj_ids[:6])
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("vis")
        / "interactive_depth_views"
        / f"{args.scan_id}_{obj_slug}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_points, obj_colors, loaded_ids = load_objects_with_colors(
        replacement_root, args.scan_id, obj_ids
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
    obj_points, obj_colors = subsample(obj_points, obj_colors, args.max_obj_points)

    bg_cloud = trimesh.points.PointCloud(
        bg_points, colors=(255.0 * bg_colors).astype(np.uint8)
    )
    obj_cloud = trimesh.points.PointCloud(
        obj_points, colors=(255.0 * obj_colors).astype(np.uint8)
    )
    scene = trimesh.Scene()
    scene.add_geometry(bg_cloud, geom_name="depth_background")
    scene.add_geometry(obj_cloud, geom_name="replacement_objects")

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
        "replacement_root": str(replacement_root),
        "scan_id": args.scan_id,
        "obj_ids": obj_ids,
        "loaded_obj_ids": loaded_ids,
        "background_removed_obj_ids": remove_obj_ids,
        "selected_frame_ids": selected_frame_ids,
        "bg_points": int(len(bg_points)),
        "obj_points": int(len(obj_points)),
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
