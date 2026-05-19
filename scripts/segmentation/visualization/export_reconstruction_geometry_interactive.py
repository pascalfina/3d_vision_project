#!/usr/bin/env python3
import argparse
import html
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import trimesh
import trimesh.viewer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.segmentation.visualization.render_depth_background import (
    build_background_from_depth,
    resolve_scene_root,
    subsample,
)
from utils import scan3r


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export an interactive HTML and PLY for raw reconstruction geometry "
            "built directly from pose + xyz/depth outputs, without requiring "
            "pred-ready objects."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument(
        "--mask-root",
        default=None,
        help=(
            "Optional mask root if you want to remove specific objects from the "
            "debug geometry. Omit for pure backend geometry."
        ),
    )
    parser.add_argument(
        "--mask-source",
        default="none",
        help="Mask source to use when --remove-obj-id is provided. Use 'none' for full-frame geometry.",
    )
    parser.add_argument(
        "--remove-obj-id",
        dest="remove_obj_ids",
        action="append",
        type=int,
        default=[],
        help="Optional object id to remove from the geometry using projection masks.",
    )
    parser.add_argument("--pose-mode", default="raw", choices=["raw", "invert"])
    parser.add_argument("--lift-coord-system", default="pinhole", choices=["scan3r", "pinhole"])
    parser.add_argument(
        "--frame-selection",
        default="diverse_area",
        choices=["all", "top_area", "diverse_area"],
    )
    parser.add_argument("--max-views", type=int, default=96)
    parser.add_argument("--mask-erode-px", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--label", default="reconstruction_geometry")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument(
        "--gt-mesh",
        default=None,
        help="Optional GT mesh path. When set, also writes a GT HTML/PLY into --out-dir.",
    )
    parser.add_argument("--gt-label", default="ground_truth_geometry")
    parser.add_argument(
        "--gt-max-points",
        type=int,
        default=300000,
        help="Maximum sampled GT surface points.",
    )
    return parser.parse_args()


def inject_overlay(html_text: str, overlay_html: str) -> str:
    style_block = """
<style>
.objectx-overlay {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 9999;
  pointer-events: none;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.objectx-card {
  width: 340px;
  max-height: calc(100vh - 32px);
  overflow: auto;
  background: rgba(255, 255, 255, 0.92);
  color: #1b1b1b;
  border: 1px solid rgba(0, 0, 0, 0.10);
  border-radius: 12px;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.18);
  padding: 12px 14px;
  pointer-events: auto;
}
.title {
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 6px;
}
.meta {
  font-size: 12px;
  color: #505050;
  margin-bottom: 4px;
}
.swatch-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  margin-top: 8px;
}
.swatch {
  width: 14px;
  height: 14px;
  border-radius: 4px;
  border: 1px solid rgba(0, 0, 0, 0.12);
}
</style>
""".strip()
    injection = style_block + "\n" + overlay_html
    if "</body>" in html_text:
        return html_text.replace("</body>", injection + "\n</body>")
    return html_text + "\n" + injection


def build_overlay(summary: dict) -> str:
    removal = summary["remove_obj_ids"]
    removal_text = ", ".join(str(x) for x in removal) if removal else "none"
    return (
        "<div class='objectx-overlay'><div class='objectx-card'>"
        f"<div class='title'>{html.escape(summary['scan_id'])}</div>"
        f"<div class='meta'>label: {html.escape(summary['label'])}</div>"
        f"<div class='meta'>points: {int(summary['point_count'])}</div>"
        f"<div class='meta'>selected frames: {int(summary['selected_frame_count'])}</div>"
        f"<div class='meta'>camera centers: {int(summary['camera_count'])}</div>"
        f"<div class='meta'>remove obj ids: {html.escape(removal_text)}</div>"
        "<div class='swatch-row'><span class='swatch' style='background: rgb(128,128,128)'></span>"
        "<span>geometry</span></div>"
        "<div class='swatch-row'><span class='swatch' style='background: rgb(220,60,40)'></span>"
        "<span>camera centers</span></div>"
        "</div></div>"
    )


def load_camera_centers(
    data_root: Path,
    scan_id: str,
    frame_ids: list[str],
    pose_mode: str,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="objectx-geomcams-") as tmpdir:
        stage_root = Path(tmpdir)
        scene_root = resolve_scene_root(data_root, scan_id, stage_root)
        poses = scan3r.load_frame_poses(
            data_dir=str(scene_root),
            scan_id=scan_id,
            frame_idxs=frame_ids,
        )
        return np.stack(
            [
                scan3r.camera_center_from_pose(poses[frame_id], pose_mode=pose_mode)
                for frame_id in frame_ids
            ],
            axis=0,
        ).astype(np.float32)


def camera_colors(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    cmap = cm.get_cmap("plasma")
    vals = np.linspace(0.0, 1.0, num=count, dtype=np.float32)
    return (255.0 * cmap(vals)[:, :3]).astype(np.uint8)


def label_slug(label: str) -> str:
    return label.replace(" ", "_")


def load_gt_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geom for geom in mesh.geometry.values() if hasattr(geom, "faces")]
        if not geometries:
            raise ValueError(f"GT scene has no triangle geometry: {path}")
        mesh = trimesh.util.concatenate(geometries)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"GT mesh has no vertices/faces: {path}")
    return mesh


def sample_gt_surface(mesh: trimesh.Trimesh, max_points: int) -> np.ndarray:
    n_points = min(int(max_points), max(len(mesh.faces), 1) * 8)
    n_points = max(n_points, min(len(mesh.vertices), int(max_points)))
    if n_points <= 0:
        raise ValueError("GT max points must be positive")
    points, _face_ids = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float32)


def write_gt_debug_outputs(
    *,
    gt_mesh_path: Path,
    scan_id: str,
    label: str,
    out_dir: Path,
    max_points: int,
) -> dict:
    mesh = load_gt_mesh(gt_mesh_path)
    points = sample_gt_surface(mesh, max_points)
    colors = np.tile(np.array([[55, 130, 255]], dtype=np.uint8), (len(points), 1))
    cloud = trimesh.points.PointCloud(points, colors=colors)

    scene = trimesh.Scene()
    scene.add_geometry(cloud, geom_name="ground_truth_geometry")
    bounds = scene.bounds
    if bounds is not None and np.isfinite(bounds).all():
        center = bounds.mean(axis=0)
        extent = np.max(bounds[1] - bounds[0])
        scene.set_camera(angles=(0.6, 0.0, 0.6), distance=max(float(extent) * 1.6, 1.0), center=center)

    slug = label_slug(label)
    html_path = out_dir / f"{scan_id}_interactive_{slug}.html"
    ply_path = out_dir / f"{scan_id}_{slug}_geometry.ply"
    cloud.export(ply_path)

    summary = {
        "data_root": str(gt_mesh_path.parent.parent.parent) if gt_mesh_path.name else None,
        "gt_mesh": str(gt_mesh_path),
        "scan_id": scan_id,
        "label": label,
        "point_count": int(len(points)),
        "html": str(html_path),
        "geometry_ply": str(ply_path),
    }
    html_text = trimesh.viewer.scene_to_html(scene)
    html_text = inject_overlay(html_text, build_gt_overlay(summary))
    html_path.write_text(html_text, encoding="utf-8")
    return summary


def build_gt_overlay(summary: dict) -> str:
    return (
        "<div class='objectx-overlay'><div class='objectx-card'>"
        f"<div class='title'>{html.escape(summary['scan_id'])}</div>"
        f"<div class='meta'>label: {html.escape(summary['label'])}</div>"
        f"<div class='meta'>points: {int(summary['point_count'])}</div>"
        f"<div class='meta'>GT mesh: {html.escape(Path(summary['gt_mesh']).name)}</div>"
        "<div class='swatch-row'><span class='swatch' style='background: rgb(55,130,255)'></span>"
        "<span>ground truth surface sample</span></div>"
        "</div></div>"
    )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root) if args.mask_root else None
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("vis")
        / "interactive_reconstruction_geometry"
        / f"{args.scan_id}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    points, colors, selected_frame_ids = build_background_from_depth(
        data_root=data_root,
        mask_root=mask_root,
        scan_id=args.scan_id,
        remove_obj_ids=args.remove_obj_ids,
        mask_source=args.mask_source,
        pose_mode=args.pose_mode,
        lift_coord_system=args.lift_coord_system,
        frame_selection=args.frame_selection,
        max_views=args.max_views,
        mask_erode_px=args.mask_erode_px,
    )
    points, colors = subsample(points, colors, args.max_points)

    scene = trimesh.Scene()
    geom_cloud = trimesh.points.PointCloud(
        points, colors=(255.0 * colors).astype(np.uint8)
    )
    scene.add_geometry(geom_cloud, geom_name="reconstruction_geometry")

    cam_centers = np.zeros((0, 3), dtype=np.float32)
    if not args.skip_cameras and selected_frame_ids:
        cam_centers = load_camera_centers(
            data_root=data_root,
            scan_id=args.scan_id,
            frame_ids=selected_frame_ids,
            pose_mode=args.pose_mode,
        )
        cam_cloud = trimesh.points.PointCloud(
            cam_centers, colors=camera_colors(len(cam_centers))
        )
        scene.add_geometry(cam_cloud, geom_name="camera_centers")

    if len(points) == 0:
        raise ValueError("No reconstruction points were produced.")

    bounds = scene.bounds
    center = bounds.mean(axis=0)
    extent = np.max(bounds[1] - bounds[0])
    distance = max(float(extent) * 1.6, 1.0)
    scene.set_camera(angles=(0.6, 0.0, 0.6), distance=distance, center=center)

    label = label_slug(args.label)
    html_path = out_dir / f"{args.scan_id}_interactive_{label}.html"
    ply_path = out_dir / f"{args.scan_id}_{label}_geometry.ply"
    cams_ply_path = out_dir / f"{args.scan_id}_{label}_camera_centers.ply"
    summary_path = out_dir / "summary.json"

    geom_cloud.export(ply_path)
    if len(cam_centers) > 0:
        trimesh.points.PointCloud(
            cam_centers, colors=camera_colors(len(cam_centers))
        ).export(cams_ply_path)

    summary = {
        "data_root": str(data_root),
        "mask_root": str(mask_root) if mask_root is not None else None,
        "scan_id": args.scan_id,
        "label": args.label,
        "remove_obj_ids": [int(x) for x in args.remove_obj_ids],
        "mask_source": args.mask_source,
        "pose_mode": args.pose_mode,
        "lift_coord_system": args.lift_coord_system,
        "frame_selection": args.frame_selection,
        "max_views": int(args.max_views),
        "point_count": int(len(points)),
        "selected_frame_ids": selected_frame_ids,
        "selected_frame_count": int(len(selected_frame_ids)),
        "camera_count": int(len(cam_centers)),
        "html": str(html_path),
        "geometry_ply": str(ply_path),
        "camera_ply": str(cams_ply_path) if len(cam_centers) > 0 else None,
    }

    html_text = trimesh.viewer.scene_to_html(scene)
    html_text = inject_overlay(html_text, build_overlay(summary))
    html_path.write_text(html_text, encoding="utf-8")
    if args.gt_mesh:
        gt_summary = write_gt_debug_outputs(
            gt_mesh_path=Path(args.gt_mesh),
            scan_id=args.scan_id,
            label=args.gt_label,
            out_dir=out_dir,
            max_points=args.gt_max_points,
        )
        summary["ground_truth"] = gt_summary
    summary_path.write_text(json.dumps(summary, indent=2))
    print(html_path)
    if args.gt_mesh:
        print(summary["ground_truth"]["html"])


if __name__ == "__main__":
    main()
