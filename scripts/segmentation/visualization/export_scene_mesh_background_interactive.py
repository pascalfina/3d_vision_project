#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import trimesh.viewer
from plyfile import PlyData


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export an interactive HTML scene view with GT scene mesh background "
            "and replacement objects loaded from voxel annotations."
        )
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--replacement-root", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--obj-id", dest="obj_ids", action="append", type=int, default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--label", default="interactive")
    parser.add_argument("--max-bg-points", type=int, default=80000)
    parser.add_argument("--max-obj-points", type=int, default=40000)
    parser.add_argument("--keep-target-in-background", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def resolve_obj_ids(args) -> list[int]:
    if args.obj_ids:
        return [int(x) for x in args.obj_ids]
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        return [int(x) for x in manifest["selected_obj_ids"]]
    raise ValueError("Provide at least one --obj-id or a --manifest file.")


def voxel_to_world(voxel_indices: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    voxel = voxel_indices.astype(np.float32) * (1.0 / 64.0)
    voxel = voxel * 2.0 - 1.0
    return voxel * float(scale) + mean[None, :]


def subsample(points: np.ndarray, colors: np.ndarray, max_points: int):
    if len(points) <= max_points:
        return points, colors
    step = max(1, len(points) // max_points)
    return points[::step], colors[::step]


def load_scene_background(
    baseline_root: Path, scan_id: str, obj_ids: list[int], keep_target: bool
):
    ply_path = baseline_root / "scenes" / scan_id / "labels.instances.align.annotated.v2.ply"
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    object_ids = np.asarray(vertex["objectId"]).astype(np.int32)
    if not keep_target:
        mask = ~np.isin(object_ids, np.array(obj_ids, dtype=np.int32))
        xyz = xyz[mask]
        rgb = rgb[mask]
    return xyz, rgb


def load_object_points(root: Path, scan_id: str, obj_id: int):
    obj_dir = root / "files" / "gs_annotations" / scan_id / str(obj_id)
    vox = np.load(obj_dir / "voxel_output_dense.npz")["arr_0"][:, :3].astype(np.int32)
    ms = np.load(obj_dir / "mean_scale_dense.npz")
    return voxel_to_world(vox, ms["mean"].astype(np.float32), float(ms["scale"]))


def load_objects_with_colors(root: Path, scan_id: str, obj_ids: list[int]):
    palette = plt.get_cmap("tab20")
    all_points = []
    all_colors = []
    loaded_ids = []
    for idx, obj_id in enumerate(obj_ids):
        obj_dir = root / "files" / "gs_annotations" / scan_id / str(obj_id)
        if not (obj_dir / "voxel_output_dense.npz").exists():
            continue
        if not (obj_dir / "mean_scale_dense.npz").exists():
            continue
        pts = load_object_points(root, scan_id, obj_id)
        color = (255.0 * np.array(palette(idx % 20)[:3], dtype=np.float32)).astype(np.uint8)
        colors = np.tile(color[None, :], (len(pts), 1))
        all_points.append(pts)
        all_colors.append(colors)
        loaded_ids.append(int(obj_id))
    if not all_points:
        raise FileNotFoundError(f"No replacement objects found for {scan_id}: {obj_ids}")
    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0), loaded_ids


def main():
    args = parse_args()
    baseline_root = Path(args.baseline_root)
    replacement_root = Path(args.replacement_root)
    obj_ids = resolve_obj_ids(args)
    obj_slug = "-".join(str(x) for x in obj_ids[:6])
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("vis")
        / "interactive_scene_views"
        / f"{args.scan_id}_{obj_slug}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_points, bg_colors = load_scene_background(
        baseline_root, args.scan_id, obj_ids, args.keep_target_in_background
    )
    obj_points, obj_colors, loaded_ids = load_objects_with_colors(
        replacement_root, args.scan_id, obj_ids
    )

    bg_points, bg_colors = subsample(bg_points, bg_colors, args.max_bg_points)
    obj_points, obj_colors = subsample(obj_points, obj_colors, args.max_obj_points)

    bg_cloud = trimesh.points.PointCloud(bg_points, colors=bg_colors)
    obj_cloud = trimesh.points.PointCloud(obj_points, colors=obj_colors)
    scene = trimesh.Scene()
    scene.add_geometry(bg_cloud, geom_name="background")
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
        "baseline_root": str(baseline_root),
        "replacement_root": str(replacement_root),
        "scan_id": args.scan_id,
        "obj_ids": obj_ids,
        "loaded_obj_ids": loaded_ids,
        "bg_points": int(len(bg_points)),
        "obj_points": int(len(obj_points)),
        "keep_target_in_background": bool(args.keep_target_in_background),
        "html": str(html_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(html_path)


if __name__ == "__main__":
    main()
