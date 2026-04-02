#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a scene-level orbit with the aligned scene mesh as background and "
            "a replacement object inserted from voxel annotations."
        )
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--replacement-root", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--obj-id", dest="obj_ids", action="append", type=int, default=[])
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional mixed_scene_manifest.json; if set, selected_obj_ids are used unless --obj-id is provided.",
    )
    parser.add_argument("--label", default="object")
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-bg-points", type=int, default=50000)
    parser.add_argument("--max-obj-points", type=int, default=16000)
    parser.add_argument("--bg-point-size", type=float, default=0.6)
    parser.add_argument("--obj-point-size", type=float, default=2.0)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--bg-color", default="white")
    parser.add_argument(
        "--fit-mode",
        default="objects",
        choices=["objects", "all"],
        help="Zoom either around the replaced objects or around the whole visible scene.",
    )
    parser.add_argument(
        "--fit-margin",
        type=float,
        default=1.45,
        help="Larger values zoom out; smaller values zoom in.",
    )
    parser.add_argument("--fig-width", type=float, default=10.0)
    parser.add_argument("--fig-height", type=float, default=5.8)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--keep-target-in-background", action="store_true")
    return parser.parse_args()


def voxel_to_world(voxel_indices: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    voxel = voxel_indices.astype(np.float32) * (1.0 / 64.0)
    voxel = voxel * 2.0 - 1.0
    return voxel * float(scale) + mean[None, :]


def subsample(points: np.ndarray, colors: np.ndarray, max_points: int):
    if len(points) <= max_points:
        return points, colors
    step = max(1, len(points) // max_points)
    return points[::step], colors[::step]


def resolve_obj_ids(args) -> list[int]:
    if args.obj_ids:
        return [int(x) for x in args.obj_ids]
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        return [int(x) for x in manifest["selected_obj_ids"]]
    raise ValueError("Provide at least one --obj-id or a --manifest file.")


def load_scene_background(
    baseline_root: Path, scan_id: str, obj_ids: list[int], keep_target: bool
):
    ply_path = baseline_root / "scenes" / scan_id / "labels.instances.align.annotated.v2.ply"
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    rgb = (
        np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
        / 255.0
    )
    object_ids = np.asarray(vertex["objectId"]).astype(np.int32)
    if not keep_target:
        remove_ids = set(int(x) for x in obj_ids)
        mask = ~np.isin(object_ids, list(remove_ids))
        xyz = xyz[mask]
        rgb = rgb[mask]
    return xyz, rgb


def load_object_points(root: Path, scan_id: str, obj_id: int):
    obj_dir = root / "files" / "gs_annotations" / scan_id / str(obj_id)
    vox = np.load(obj_dir / "voxel_output_dense.npz")["arr_0"][:, :3].astype(np.int32)
    ms = np.load(obj_dir / "mean_scale_dense.npz")
    world = voxel_to_world(vox, ms["mean"].astype(np.float32), float(ms["scale"]))
    return world


def load_objects_with_colors(root: Path, scan_id: str, obj_ids: list[int]):
    palette = plt.get_cmap("tab20")
    all_points = []
    all_colors = []
    loaded_ids = []
    for idx, obj_id in enumerate(obj_ids):
        obj_dir = root / "files" / "gs_annotations" / scan_id / str(obj_id)
        vox_path = obj_dir / "voxel_output_dense.npz"
        ms_path = obj_dir / "mean_scale_dense.npz"
        if not vox_path.exists() or not ms_path.exists():
            continue
        pts = load_object_points(root, scan_id, obj_id)
        color = np.array(palette(idx % 20)[:3], dtype=np.float32)
        colors = np.tile(color[None, :], (len(pts), 1))
        all_points.append(pts)
        all_colors.append(colors)
        loaded_ids.append(int(obj_id))
    if not all_points:
        raise FileNotFoundError(f"No replacement objects found for {scan_id}: {obj_ids}")
    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0), loaded_ids


def compute_bounds(bg_points: np.ndarray, obj_points: np.ndarray, fit_mode: str, fit_margin: float):
    pts = obj_points if fit_mode == "objects" else np.concatenate([bg_points, obj_points], axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = max(float(np.max(maxs - mins)), 1e-3)
    half = 0.5 * fit_margin * extent
    return center, half


def render_frame(
    bg_points,
    bg_colors,
    obj_points,
    obj_colors,
    center,
    half_extent,
    *,
    azim,
    elev,
    bg_point_size,
    obj_point_size,
    facecolor,
    fig_width,
    fig_height,
):
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=facecolor)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(facecolor)
    ax.scatter(
        bg_points[:, 0],
        bg_points[:, 1],
        bg_points[:, 2],
        c=bg_colors,
        s=bg_point_size,
        depthshade=False,
        linewidths=0,
    )
    ax.scatter(
        obj_points[:, 0],
        obj_points[:, 1],
        obj_points[:, 2],
        c=obj_colors,
        s=obj_point_size,
        depthshade=False,
        linewidths=0,
    )
    ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
    ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return frame


def save_contact_sheet(frames, out_file: Path):
    thumbs = []
    for idx, frame in frames:
        img = Image.fromarray(frame)
        img.thumbnail((360, 220))
        canvas = Image.new("RGB", (380, 250), "white")
        canvas.paste(img, ((380 - img.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 226), f"frame {idx}", fill="black")
        thumbs.append(canvas)
    sheet = Image.new("RGB", (380 * len(thumbs), 250), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (i * 380, 0))
    sheet.save(out_file)


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
        / "rendered_scene_mesh_bg"
        / f"{args.scan_id}_{obj_slug}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_points, bg_colors = load_scene_background(
        baseline_root, args.scan_id, obj_ids, args.keep_target_in_background
    )
    obj_points, obj_colors, loaded_obj_ids = load_objects_with_colors(
        replacement_root, args.scan_id, obj_ids
    )

    bg_points, bg_colors = subsample(bg_points, bg_colors, args.max_bg_points)
    obj_points, obj_colors = subsample(obj_points, obj_colors, args.max_obj_points)

    center, half_extent = compute_bounds(
        bg_points, obj_points, args.fit_mode, args.fit_margin
    )
    facecolor = args.bg_color

    rendered = []
    preview_frames = []
    for frame_idx in range(args.num_frames):
        azim = 360.0 * frame_idx / args.num_frames
        frame = render_frame(
            bg_points,
            bg_colors,
            obj_points,
            obj_colors,
            center,
            half_extent,
            azim=azim,
            elev=args.elev,
            bg_point_size=args.bg_point_size,
            obj_point_size=args.obj_point_size,
            facecolor=facecolor,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
        )
        rendered.append(frame)
        if frame_idx in {0, args.num_frames // 4, args.num_frames // 2, (3 * args.num_frames) // 4}:
            preview_frames.append((frame_idx, frame))

    video_path = out_dir / f"{args.scan_id}_scene_mesh_bg_{args.label.replace(' ', '_')}.mp4"
    imageio.mimsave(video_path, rendered, fps=args.fps)
    save_contact_sheet(preview_frames, out_dir / "contact_sheet.png")

    summary = {
        "baseline_root": str(baseline_root),
        "replacement_root": str(replacement_root),
        "scan_id": args.scan_id,
        "obj_ids": obj_ids,
        "loaded_obj_ids": loaded_obj_ids,
        "label": args.label,
        "num_frames": args.num_frames,
        "bg_points": int(len(bg_points)),
        "obj_points": int(len(obj_points)),
        "keep_target_in_background": bool(args.keep_target_in_background),
        "fit_mode": args.fit_mode,
        "fit_margin": args.fit_margin,
        "video": str(video_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(video_path)
    print(out_dir / "contact_sheet.png")


if __name__ == "__main__":
    main()
