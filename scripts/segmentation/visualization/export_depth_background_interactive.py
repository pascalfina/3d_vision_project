#!/usr/bin/env python3
import argparse
import html
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
    load_object_cloud_specs,
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
    parser.add_argument(
        "--mask-root",
        default=None,
        help="Optional root to load object masks from. Defaults to --data-root.",
    )
    parser.add_argument("--replacement-root", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--obj-id", dest="obj_ids", action="append", type=int, default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mask-source", default="gt_projection")
    parser.add_argument(
        "--background-remove-mode",
        default="loaded",
        choices=["all", "loaded", "none"],
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
    parser.add_argument("--min-obj-points-per-object", type=int, default=256)
    parser.add_argument(
        "--background-color-mode",
        default="rgb",
        choices=["rgb", "gray"],
        help="Use original RGB background colors or a neutral gray background for readability.",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help=(
            "Render only the lifted depth/XYZ geometry. This disables object "
            "replacement clouds, object-mask background removal, and the object legend."
        ),
    )
    parser.add_argument(
        "--hide-object-overlay",
        action="store_true",
        help="Do not inject the object legend overlay into the generated HTML.",
    )
    parser.add_argument("--label", default="depth_bg_interactive")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def subsample_to_count(points: np.ndarray, colors: np.ndarray, target_count: int):
    if target_count <= 0 or len(points) <= target_count:
        return points, colors
    indices = np.linspace(0, len(points) - 1, num=target_count, dtype=np.int64)
    return points[indices], colors[indices]


def allocate_object_point_budgets(
    point_counts: list[int], max_total_points: int, min_points_per_object: int
) -> list[int]:
    if not point_counts:
        return []
    total_points = sum(point_counts)
    if max_total_points <= 0 or total_points <= max_total_points:
        return [int(x) for x in point_counts]

    base = [min(int(count), max(1, int(min_points_per_object))) for count in point_counts]
    base_total = sum(base)
    if base_total >= max_total_points:
        order = sorted(range(len(point_counts)), key=lambda idx: point_counts[idx], reverse=True)
        alloc = [0] * len(point_counts)
        for idx in order[:max_total_points]:
            alloc[idx] = 1
        return alloc

    residual = [max(0, int(count) - base[idx]) for idx, count in enumerate(point_counts)]
    remaining_budget = max_total_points - base_total
    residual_total = sum(residual)
    if residual_total <= 0 or remaining_budget <= 0:
        return base

    extras = [0] * len(point_counts)
    fractions = []
    for idx, room in enumerate(residual):
        raw_extra = remaining_budget * float(room) / float(residual_total)
        extra = min(room, int(np.floor(raw_extra)))
        extras[idx] = extra
        fractions.append((raw_extra - extra, idx))

    used = sum(extras)
    leftover = remaining_budget - used
    for _, idx in sorted(fractions, reverse=True):
        if leftover <= 0:
            break
        if extras[idx] < residual[idx]:
            extras[idx] += 1
            leftover -= 1

    return [base[idx] + extras[idx] for idx in range(len(point_counts))]


def subsample_object_cloud_specs(
    object_specs: list[dict], max_total_points: int, min_points_per_object: int
) -> list[dict]:
    budgets = allocate_object_point_budgets(
        [int(spec["point_count"]) for spec in object_specs],
        max_total_points=max_total_points,
        min_points_per_object=min_points_per_object,
    )
    sampled = []
    for spec, budget in zip(object_specs, budgets):
        points, colors = subsample_to_count(
            spec["points"],
            np.tile(spec["color"][None, :], (len(spec["points"]), 1)),
            budget,
        )
        sampled.append(
            {
                **spec,
                "points": points,
                "colors": colors,
                "export_point_count": int(len(points)),
                "geometry_name": f"object_{int(spec['obj_id']):03d}",
            }
        )
    return sampled


def maybe_gray_background(bg_colors: np.ndarray, mode: str) -> np.ndarray:
    if mode != "gray" or len(bg_colors) == 0:
        return bg_colors
    luminance = np.dot(bg_colors[:, :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    luminance = np.clip(0.15 + 0.70 * luminance, 0.0, 1.0).astype(np.float32)
    return np.repeat(luminance[:, None], 3, axis=1)


def build_legend_html(summary: dict) -> str:
    rows = []
    ordered_objects = sorted(
        summary["objects"],
        key=lambda item: (int(item["export_point_count"]), int(item["point_count"])),
        reverse=True,
    )
    for item in ordered_objects:
        rgb = item["color_rgb"]
        color_css = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        rows.append(
            "<div class='legend-row'>"
            f"<span class='swatch' style='background:{color_css}'></span>"
            f"<span class='obj-label'>Objekt {int(item['obj_id'])}</span>"
            f"<span class='obj-meta'>{int(item['export_point_count'])}/{int(item['point_count'])} Punkte</span>"
            "</div>"
        )
    return (
        "<div class='objectx-overlay'>"
        "<div class='objectx-card'>"
        f"<div class='legend-title'>{html.escape(summary['scan_id'])}</div>"
        f"<div class='legend-subtitle'>{len(summary['objects'])} Objekte, "
        f"{int(summary['bg_points'])} Hintergrundpunkte</div>"
        "<details open><summary>Objekt-Legende</summary>"
        "<div class='legend-list'>"
        + "".join(rows)
        + "</div></details></div></div>"
    )


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
  width: 320px;
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
.legend-title {
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 4px;
}
.legend-subtitle {
  font-size: 12px;
  color: #505050;
  margin-bottom: 8px;
}
.legend-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 10px;
}
.legend-row {
  display: grid;
  grid-template-columns: 14px 1fr auto;
  gap: 8px;
  align-items: center;
  font-size: 12px;
}
.swatch {
  width: 14px;
  height: 14px;
  border-radius: 4px;
  border: 1px solid rgba(0, 0, 0, 0.12);
}
.obj-label {
  font-weight: 600;
}
.obj-meta {
  color: #666;
  white-space: nowrap;
}
details summary {
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
}
</style>
""".strip()
    injection = style_block + "\n" + overlay_html
    if "</body>" in html_text:
        return html_text.replace("</body>", injection + "\n</body>")
    return html_text + "\n" + injection


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root) if args.mask_root else data_root
    replacement_root = Path(args.replacement_root)
    obj_ids = [] if args.geometry_only else resolve_obj_ids(args)
    obj_slug = "-".join(str(x) for x in obj_ids[:6])
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("vis")
        / "interactive_depth_views"
        / f"{args.scan_id}_{obj_slug}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.geometry_only:
        object_specs = []
        loaded_ids = []
        remove_obj_ids = []
        mask_source = "none"
    else:
        object_specs = load_object_cloud_specs(
            replacement_root, args.scan_id, obj_ids
        )
        loaded_ids = [int(spec["obj_id"]) for spec in object_specs]
        if args.background_remove_mode == "all":
            remove_obj_ids = obj_ids
        elif args.background_remove_mode == "loaded":
            remove_obj_ids = loaded_ids
        else:
            remove_obj_ids = []
        mask_source = args.mask_source
    bg_points, bg_colors, selected_frame_ids = build_background_from_depth(
        data_root=data_root,
        mask_root=mask_root,
        scan_id=args.scan_id,
        remove_obj_ids=remove_obj_ids,
        mask_source=mask_source,
        pose_mode=args.pose_mode,
        lift_coord_system=args.lift_coord_system,
        frame_selection=args.frame_selection,
        max_views=args.max_views,
        mask_erode_px=args.mask_erode_px,
    )

    bg_points, bg_colors = subsample(bg_points, bg_colors, args.max_bg_points)
    bg_colors = maybe_gray_background(bg_colors, args.background_color_mode)
    if object_specs:
        object_specs = subsample_object_cloud_specs(
            object_specs,
            max_total_points=args.max_obj_points,
            min_points_per_object=args.min_obj_points_per_object,
        )
        obj_points = np.concatenate([spec["points"] for spec in object_specs], axis=0)
    else:
        obj_points = np.zeros((0, 3), dtype=np.float32)

    bg_cloud = trimesh.points.PointCloud(
        bg_points, colors=(255.0 * bg_colors).astype(np.uint8)
    )
    scene = trimesh.Scene()
    scene.add_geometry(bg_cloud, geom_name="depth_background")
    for spec in object_specs:
        obj_cloud = trimesh.points.PointCloud(
            spec["points"], colors=(255.0 * spec["colors"]).astype(np.uint8)
        )
        scene.add_geometry(obj_cloud, geom_name=spec["geometry_name"])

    bounds = scene.bounds
    center = bounds.mean(axis=0)
    extent = np.max(bounds[1] - bounds[0])
    distance = max(float(extent) * 1.6, 1.0)
    scene.set_camera(angles=(0.6, 0.0, 0.6), distance=distance, center=center)

    html = trimesh.viewer.scene_to_html(scene)
    html_path = out_dir / f"{args.scan_id}_interactive_{args.label.replace(' ', '_')}.html"

    summary = {
        "data_root": str(data_root),
        "mask_root": str(mask_root),
        "replacement_root": str(replacement_root),
        "scan_id": args.scan_id,
        "obj_ids": obj_ids,
        "loaded_obj_ids": loaded_ids,
        "background_removed_obj_ids": remove_obj_ids,
        "selected_frame_ids": selected_frame_ids,
        "bg_points": int(len(bg_points)),
        "obj_points": int(len(obj_points)),
        "mask_source": mask_source,
        "background_remove_mode": args.background_remove_mode,
        "mask_erode_px": int(args.mask_erode_px),
        "background_color_mode": args.background_color_mode,
        "geometry_only": bool(args.geometry_only),
        "pose_mode": args.pose_mode,
        "lift_coord_system": args.lift_coord_system,
        "objects": [
            {
                "obj_id": int(spec["obj_id"]),
                "point_count": int(spec["point_count"]),
                "export_point_count": int(spec["export_point_count"]),
                "geometry_name": spec["geometry_name"],
                "color_rgb": [int(x) for x in (255.0 * spec["color"]).astype(np.uint8).tolist()],
            }
            for spec in object_specs
        ],
        "html": str(html_path),
    }
    if not args.geometry_only and not args.hide_object_overlay:
        html = inject_overlay(html, build_legend_html(summary))
    html_path.write_text(html, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(html_path)


if __name__ == "__main__":
    main()
