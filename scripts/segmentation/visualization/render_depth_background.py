#!/usr/bin/env python3
import argparse
import colorsys
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.voxel_anno import voxelise_features as vf
from utils import scan3r


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a scene-level orbit with background reconstructed from depth+pose "
            "and replacement objects loaded from gs_annotations."
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
        choices=["all", "loaded"],
        help="Which object masks to remove from the depth background.",
    )
    parser.add_argument(
        "--mask-erode-px",
        type=int,
        default=2,
        help="Shrink removal masks to preserve nearby floor/background pixels.",
    )
    parser.add_argument("--pose-mode", default="raw", choices=["raw", "invert"])
    parser.add_argument("--lift-coord-system", default="pinhole", choices=["scan3r", "pinhole"])
    parser.add_argument("--frame-selection", default="diverse_area", choices=["all", "top_area", "diverse_area"])
    parser.add_argument("--max-views", type=int, default=48)
    parser.add_argument("--bg-max-points", type=int, default=60000)
    parser.add_argument("--obj-max-points", type=int, default=30000)
    parser.add_argument("--fit-mode", default="objects", choices=["objects", "all"])
    parser.add_argument("--fit-margin", type=float, default=1.35)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--fig-height", type=float, default=7.0)
    parser.add_argument("--bg-point-size", type=float, default=0.5)
    parser.add_argument("--obj-point-size", type=float, default=2.0)
    parser.add_argument("--label", default="depth_bg")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


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


def resolve_obj_ids(args) -> list[int]:
    if args.obj_ids:
        return [int(x) for x in args.obj_ids]
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        return [int(x) for x in manifest["selected_obj_ids"]]
    raise ValueError("Provide at least one --obj-id or a --manifest file.")


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


def resolve_scene_root(data_root: Path, scan_id: str, stage_root: Path) -> Path:
    scene_dirnames = []
    for env_key in ("OBJECTX_VIS_SCENES_DIRNAME", "OBJECTX_SCENES_DIRNAME"):
        value = os.getenv(env_key, "").strip()
        if value and value not in scene_dirnames:
            scene_dirnames.append(value)
    if "scenes" not in scene_dirnames:
        scene_dirnames.append("scenes")

    for scene_dirname in scene_dirnames:
        scan_dir = data_root / scene_dirname / scan_id
        info = scan_dir / "sequence" / "_info.txt"
        if scene_dirname == "scenes" and info.exists():
            return data_root
        if (scan_dir / "sequence.zip").exists() or (scan_dir / "sequence").exists():
            staged = stage_root / "scenes" / scan_id
            stage_scan(scan_dir, staged)
            return stage_root
    raise FileNotFoundError(f"Could not resolve staged scene for {scan_id}")


def voxel_to_world(voxel_indices: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    voxel = voxel_indices.astype(np.float32) * (1.0 / 64.0)
    voxel = voxel * 2.0 - 1.0
    return voxel * float(scale) + mean[None, :]


def subsample(points: np.ndarray, colors: np.ndarray, max_points: int):
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices], colors[indices]


def load_object_points(root: Path, scan_id: str, obj_id: int):
    obj_dir = root / "files" / "gs_annotations" / scan_id / str(obj_id)
    vox = np.load(obj_dir / "voxel_output_dense.npz")["arr_0"][:, :3].astype(np.int32)
    ms = np.load(obj_dir / "mean_scale_dense.npz")
    return voxel_to_world(vox, ms["mean"].astype(np.float32), float(ms["scale"]))


def generate_object_color(index: int) -> np.ndarray:
    # Use a golden-ratio hue sequence so neighboring objects remain visually distinct.
    hue = (0.618033988749895 * index) % 1.0
    saturation = 0.72 + 0.18 * ((index % 3) / 2.0)
    value = 0.88 + 0.10 * ((index % 2))
    return np.array(
        colorsys.hsv_to_rgb(hue, min(saturation, 0.95), min(value, 0.98)),
        dtype=np.float32,
    )


def load_object_cloud_specs(root: Path, scan_id: str, obj_ids: list[int]):
    specs = []
    for idx, obj_id in enumerate(obj_ids):
        obj_dir = root / "files" / "gs_annotations" / scan_id / str(obj_id)
        if not (obj_dir / "voxel_output_dense.npz").exists():
            continue
        if not (obj_dir / "mean_scale_dense.npz").exists():
            continue
        pts = load_object_points(root, scan_id, obj_id)
        color = generate_object_color(idx)
        specs.append(
            {
                "obj_id": int(obj_id),
                "points": pts,
                "color": color,
                "point_count": int(len(pts)),
            }
        )
    if not specs:
        raise FileNotFoundError(f"No replacement objects found for {scan_id}: {obj_ids}")
    return specs


def load_objects_with_colors(root: Path, scan_id: str, obj_ids: list[int]):
    pts_all, cols_all, loaded = [], [], []
    for spec in load_object_cloud_specs(root, scan_id, obj_ids):
        pts = spec["points"]
        color = spec["color"]
        cols = np.tile(color[None, :], (len(pts), 1))
        pts_all.append(pts)
        cols_all.append(cols)
        loaded.append(int(spec["obj_id"]))
    return np.concatenate(pts_all, axis=0), np.concatenate(cols_all, axis=0), loaded


def load_color_frame_array(
    scenes_dir: Path,
    scan_id: str,
    frame_id: str,
    width: int,
    height: int,
) -> np.ndarray:
    image = (
        Image.open(scenes_dir / scan_id / "sequence" / f"frame-{frame_id}.color.jpg")
        .convert("RGB")
        .resize((width, height), resample=Image.BILINEAR)
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(np.uint8, copy=False)
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    size = 2 * pixels + 1
    eroded = pil.filter(ImageFilter.MinFilter(size=size))
    return (np.asarray(eroded, dtype=np.uint8) > 0).astype(np.uint8)


def _filter_by_background_voxel_support(
    points: np.ndarray,
    colors: np.ndarray,
    frame_indices: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    voxel_size = float(os.getenv("OBJECTX_VIS_BG_VOXEL_SIZE", "0.0"))
    min_points = int(os.getenv("OBJECTX_VIS_BG_MIN_POINTS_PER_VOXEL", "0"))
    min_views = int(os.getenv("OBJECTX_VIS_BG_MIN_VIEWS_PER_VOXEL", "0"))
    if voxel_size <= 0.0 or (min_points <= 1 and min_views <= 1):
        return points, colors, frame_indices
    if points.shape[0] < 64:
        return points, colors, frame_indices

    voxel = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(
        voxel, axis=0, return_inverse=True, return_counts=True
    )
    keep_voxel = np.ones_like(counts, dtype=bool)
    if min_points > 1:
        keep_voxel &= counts >= min_points
    if min_views > 1 and frame_indices is not None:
        pairs = np.stack([inverse, frame_indices.astype(np.int64, copy=False)], axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        view_counts = np.bincount(unique_pairs[:, 0], minlength=counts.shape[0])
        keep_voxel &= view_counts >= min_views

    keep = keep_voxel[inverse]
    min_keep_fraction = float(os.getenv("OBJECTX_VIS_BG_MIN_KEEP_FRACTION", "0.03"))
    min_keep = max(32, int(min_keep_fraction * points.shape[0]))
    if int(keep.sum()) < min_keep:
        print(
            "[depthbg] voxel-support filter skipped: "
            f"would keep {int(keep.sum())}/{int(points.shape[0])} points"
        )
        return points, colors, frame_indices

    print(
        "[depthbg] voxel-support filter kept "
        f"{int(keep.sum())}/{int(points.shape[0])} points "
        f"(voxel_size={voxel_size}, min_points={min_points}, min_views={min_views})"
    )
    next_frame_indices = frame_indices[keep] if frame_indices is not None else None
    return points[keep], colors[keep], next_frame_indices


def filter_points_with_colors(
    points: np.ndarray,
    colors: np.ndarray,
    frame_indices: Optional[np.ndarray] = None,
):
    if len(points) == 0:
        return points.astype(np.float32), colors.astype(np.float32)

    keep = np.all(np.isfinite(points), axis=1)
    filtered_points = points[keep]
    filtered_colors = colors[keep]
    filtered_frame_indices = frame_indices[keep] if frame_indices is not None else None
    if filtered_points.shape[0] < 64:
        return filtered_points.astype(np.float32), filtered_colors.astype(np.float32)

    lower_q = float(os.getenv("OBJECTX_VOXEL_LIFT_TRIM_LOW_Q", "1.0"))
    upper_q = float(os.getenv("OBJECTX_VOXEL_LIFT_TRIM_HIGH_Q", "99.0"))
    if 0.0 <= lower_q < upper_q <= 100.0:
        lower = np.percentile(filtered_points, lower_q, axis=0)
        upper = np.percentile(filtered_points, upper_q, axis=0)
        keep = np.all((filtered_points >= lower) & (filtered_points <= upper), axis=1)
        if keep.sum() >= max(32, int(0.2 * filtered_points.shape[0])):
            filtered_points = filtered_points[keep]
            filtered_colors = filtered_colors[keep]
            if filtered_frame_indices is not None:
                filtered_frame_indices = filtered_frame_indices[keep]

    if filtered_points.shape[0] >= 64:
        center = np.median(filtered_points, axis=0)
        distances = np.linalg.norm(filtered_points - center[None, :], axis=1)
        keep_q = float(os.getenv("OBJECTX_VOXEL_LIFT_KEEP_RADIUS_Q", "99.0"))
        keep_radius = np.percentile(distances, keep_q)
        keep = distances <= keep_radius
        if keep.sum() >= max(32, int(0.2 * filtered_points.shape[0])):
            filtered_points = filtered_points[keep]
            filtered_colors = filtered_colors[keep]
            if filtered_frame_indices is not None:
                filtered_frame_indices = filtered_frame_indices[keep]

    filtered_points, filtered_colors, _ = _filter_by_background_voxel_support(
        filtered_points,
        filtered_colors,
        filtered_frame_indices,
    )

    return filtered_points.astype(np.float32), filtered_colors.astype(np.float32)


def build_background_from_depth(
    data_root: Path,
    mask_root: Optional[Path],
    scan_id: str,
    remove_obj_ids: list[int],
    mask_source: str,
    pose_mode: str,
    lift_coord_system: str,
    frame_selection: str,
    max_views: int,
    mask_erode_px: int,
):
    with tempfile.TemporaryDirectory(prefix="objectx-depthbg-") as tmpdir:
        stage_root = Path(tmpdir)
        scene_root = resolve_scene_root(data_root, scan_id, stage_root)
        scenes_dir = scene_root / "scenes"
        mask_data_root = mask_root if mask_root is not None else data_root

        frame_ids = scan3r.load_frame_idxs(data_dir=str(scenes_dir), scan_id=scan_id)
        extrinsics = scan3r.load_frame_poses(
            data_dir=str(scene_root), scan_id=scan_id, frame_idxs=frame_ids
        )
        depth_intrinsics = scan3r.load_intrinsics(data_dir=str(scenes_dir), scan_id=scan_id, type="depth")
        depth_shift = vf._load_depth_shift(str(scenes_dir), scan_id)
        masks = scan3r.load_masks(str(mask_data_root), scan_id, mask_source=mask_source)

        vis_frame_ids = []
        vis_masks = []
        for frame_id in frame_ids:
            union_mask = np.isin(
                masks[frame_id], np.array(remove_obj_ids, dtype=np.int32)
            ).astype(np.uint8)
            union_mask = erode_mask(union_mask, mask_erode_px)
            # We want background, so keep frames that contain at least some non-object pixels.
            if union_mask.size > 0:
                vis_frame_ids.append(frame_id)
                vis_masks.append(union_mask)

        with temp_env(
            {
                "OBJECTX_VOXEL_FRAME_SELECTION": frame_selection,
                "OBJECTX_VOXEL_LIFT_COORD_SYSTEM": lift_coord_system,
                "OBJECTX_VOXEL_POSE_MODE": pose_mode,
            }
        ):
            selected_frame_ids, selected_masks = vf._select_object_frames(
                vis_frame_ids,
                vis_masks,
                extrinsics=extrinsics,
                max_views=max_views,
            )
            dropped_ids = vf._compute_scene_pose_jump_dropped(frame_ids, extrinsics)
            selected_frame_ids, selected_masks = vf._filter_frames_by_pose_jumps(
                selected_frame_ids,
                selected_masks,
                dropped_ids=dropped_ids,
            )

        depth_width = int(depth_intrinsics["width"])
        depth_height = int(depth_intrinsics["height"])
        intrinsic = depth_intrinsics["intrinsic_mat"].astype(np.float32)
        intrinsic_inv = np.linalg.inv(intrinsic).astype(np.float32)
        pixel_stride = max(1, int(os.getenv("OBJECTX_VOXEL_LIFT_PIXEL_STRIDE", "1")))
        coord_system = lift_coord_system.strip().lower()

        vis_depth_source = os.getenv("OBJECTX_VIS_DEPTH_SOURCE", "").strip().lower()
        if vis_depth_source:
            depth_source = vis_depth_source
        else:
            depth_source = (
                os.getenv("OBJECTX_VOXEL_DEPTH_SOURCE", "default").strip().lower()
            )
        prefer_raw = depth_source in {"raw_if_available", "must3r_raw_if_available"}
        prefer_xyz = depth_source in {
            "must3r_xyz_if_available",
            "xyz_if_available",
            "xyz",
        }
        raw_conf_thr = float(os.getenv("OBJECTX_VIS_RAW_DEPTH_CONF_THR", "0.15"))
        raw_depth_max = float(os.getenv("OBJECTX_VIS_RAW_DEPTH_MAX", "0.0"))

        lifted_points = []
        lifted_colors = []
        lifted_frame_indices = []
        xyz_frames_used = 0
        intrinsic_frames_used = 0
        for frame_index, (frame_id, union_mask) in enumerate(
            zip(selected_frame_ids, selected_masks)
        ):
            sequence_dir = scenes_dir / scan_id / "sequence"
            raw_path = sequence_dir / f"frame-{frame_id}.depth_raw.npy"
            conf_path = sequence_dir / f"frame-{frame_id}.conf.npy"
            xyz_path = sequence_dir / f"frame-{frame_id}.xyz.npy"

            xyz_map = None
            if prefer_xyz and xyz_path.exists():
                xyz_map = np.load(str(xyz_path)).astype(np.float32)
                xyz_map[~np.isfinite(xyz_map)] = 0.0
                if raw_conf_thr > 0.0 and conf_path.exists():
                    conf_map = np.load(str(conf_path)).astype(np.float32)
                    if conf_map.shape[:2] == xyz_map.shape[:2]:
                        xyz_map[conf_map < raw_conf_thr] = 0.0
                if raw_depth_max > 0.0:
                    xyz_map[xyz_map[..., 2] > raw_depth_max] = 0.0

            if xyz_map is None:
                if prefer_raw and raw_path.exists():
                    depth_map = np.load(str(raw_path)).astype(np.float32)
                    if raw_conf_thr > 0.0 and conf_path.exists():
                        conf_map = np.load(str(conf_path)).astype(np.float32)
                        if conf_map.shape != depth_map.shape:
                            conf_map = np.array(
                                Image.fromarray(conf_map).resize(
                                    (depth_map.shape[1], depth_map.shape[0]),
                                    resample=Image.BILINEAR,
                                ),
                                dtype=np.float32,
                            )
                        depth_map = np.where(conf_map >= raw_conf_thr, depth_map, 0.0)
                else:
                    depth_map = scan3r.load_depth_map(
                        str(sequence_dir / f"frame-{frame_id}.depth.pgm"),
                        depth_shift,
                    )
                if raw_depth_max > 0.0:
                    depth_map = np.where(depth_map <= raw_depth_max, depth_map, 0.0)

            bg_mask = 1 - union_mask.astype(np.uint8)
            if xyz_map is not None:
                map_height, map_width = xyz_map.shape[:2]
            else:
                map_height, map_width = depth_height, depth_width
            mask_depth = np.array(
                Image.fromarray((bg_mask > 0).astype(np.uint8)).resize(
                    (map_width, map_height), resample=Image.NEAREST
                ),
                dtype=bool,
            )
            y, x = np.nonzero(mask_depth)
            if x.size == 0:
                continue

            if xyz_map is not None:
                cam_xyz = xyz_map[y, x].astype(np.float32)
                valid = (
                    np.isfinite(cam_xyz).all(axis=1)
                    & (np.abs(cam_xyz).sum(axis=1) > 0.0)
                    & (cam_xyz[:, 2] > 0.0)
                )
                if not np.any(valid):
                    continue
                x = x[valid]
                y = y[valid]
                cam_xyz = cam_xyz[valid]
                if pixel_stride > 1:
                    x = x[::pixel_stride]
                    y = y[::pixel_stride]
                    cam_xyz = cam_xyz[::pixel_stride]
                cam_points = cam_xyz.T
                xyz_frames_used += 1
            else:
                depth = depth_map[y, x]
                valid = depth > 0.0
                if not np.any(valid):
                    continue
                x = x[valid]
                y = y[valid]
                depth = depth[valid].astype(np.float32)

                if pixel_stride > 1:
                    x = x[::pixel_stride]
                    y = y[::pixel_stride]
                    depth = depth[::pixel_stride]

                pixels = np.stack(
                    [
                        x.astype(np.float32),
                        y.astype(np.float32),
                        np.ones_like(x, dtype=np.float32),
                    ],
                    axis=0,
                )
                if coord_system in {"scan3r", "kitti"}:
                    x3 = (x.astype(np.float32) - intrinsic[0, 2]) * depth / intrinsic[0, 0]
                    y3 = (y.astype(np.float32) - intrinsic[1, 2]) * depth / intrinsic[1, 1]
                    cam_points = np.stack([depth, -x3, -y3], axis=0)
                else:
                    cam_points = (intrinsic_inv @ pixels) * depth[None, :]
                intrinsic_frames_used += 1

            if pose_mode == "invert":
                camera_to_world = np.linalg.inv(extrinsics[frame_id])
            else:
                camera_to_world = np.array(extrinsics[frame_id], copy=True)

            world_points = (
                camera_to_world[:3, :3].astype(np.float32) @ cam_points
                + camera_to_world[:3, 3:4].astype(np.float32)
            )
            # Load color at the same resolution as the xyz/depth map we
            # indexed with (y, x). Pi3X writes xyz at its native net size
            # (e.g. 476x266), which differs from the Tango depth grid
            # (224x172).
            color_image = load_color_frame_array(
                scenes_dir, scan_id, frame_id, map_width, map_height
            )
            point_colors = color_image[y, x]
            frame_points = world_points.T
            lifted_points.append(frame_points)
            lifted_colors.append(point_colors)
            lifted_frame_indices.append(
                np.full(frame_points.shape[0], frame_index, dtype=np.int32)
            )

        print(
            f"[depthbg] lifted frames: xyz={xyz_frames_used} "
            f"pinhole={intrinsic_frames_used}"
        )

        if not lifted_points:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                selected_frame_ids,
            )

        bg_points = np.concatenate(lifted_points, axis=0).astype(np.float32)
        bg_colors = np.concatenate(lifted_colors, axis=0).astype(np.float32)
        bg_frame_indices = np.concatenate(lifted_frame_indices, axis=0)
        bg_points, bg_colors = filter_points_with_colors(
            bg_points,
            bg_colors,
            bg_frame_indices,
        )
        return bg_points, bg_colors, selected_frame_ids


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
    fig_width,
    fig_height,
):
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.scatter(bg_points[:, 0], bg_points[:, 1], bg_points[:, 2], c=bg_colors, s=bg_point_size, depthshade=False, linewidths=0)
    ax.scatter(obj_points[:, 0], obj_points[:, 1], obj_points[:, 2], c=obj_colors, s=obj_point_size, depthshade=False, linewidths=0)
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
    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root) if args.mask_root else data_root
    replacement_root = Path(args.replacement_root)
    obj_ids = resolve_obj_ids(args)
    obj_slug = "-".join(str(x) for x in obj_ids[:6])
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("vis")
        / "rendered_depth_bg"
        / f"{args.scan_id}_{obj_slug}_{args.label.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_points, obj_colors, loaded_ids = load_objects_with_colors(
        replacement_root, args.scan_id, obj_ids
    )
    remove_obj_ids = obj_ids if args.background_remove_mode == "all" else loaded_ids
    bg_points, bg_colors, selected_frame_ids = build_background_from_depth(
        data_root=data_root,
        mask_root=mask_root,
        scan_id=args.scan_id,
        remove_obj_ids=remove_obj_ids,
        mask_source=args.mask_source,
        pose_mode=args.pose_mode,
        lift_coord_system=args.lift_coord_system,
        frame_selection=args.frame_selection,
        max_views=args.max_views,
        mask_erode_px=args.mask_erode_px,
    )
    bg_points, bg_colors = subsample(bg_points, bg_colors, args.bg_max_points)
    obj_points, obj_colors = subsample(obj_points, obj_colors, args.obj_max_points)
    center, half_extent = compute_bounds(bg_points, obj_points, args.fit_mode, args.fit_margin)

    rendered = []
    preview = []
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
            elev=18.0,
            bg_point_size=args.bg_point_size,
            obj_point_size=args.obj_point_size,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
        )
        rendered.append(frame)
        if frame_idx in {0, args.num_frames // 4, args.num_frames // 2, (3 * args.num_frames) // 4}:
            preview.append((frame_idx, frame))

    video_path = out_dir / f"{args.scan_id}_depth_bg_{args.label.replace(' ', '_')}.mp4"
    imageio.mimsave(video_path, rendered, fps=args.fps)
    save_contact_sheet(preview, out_dir / "contact_sheet.png")

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
        "mask_source": args.mask_source,
        "background_remove_mode": args.background_remove_mode,
        "mask_erode_px": int(args.mask_erode_px),
        "pose_mode": args.pose_mode,
        "lift_coord_system": args.lift_coord_system,
        "video": str(video_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(video_path)
    print(out_dir / "contact_sheet.png")


if __name__ == "__main__":
    main()
