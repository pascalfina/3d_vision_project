#!/usr/bin/env python3
import argparse
import json
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import scan3r


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render debug previews for predicted-segmentation experiments: "
            "2D mask overlays for the strongest frames and PNG previews for "
            "existing lifted/voxel PLY files."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root containing files/, scenes/, and mask projections.",
    )
    parser.add_argument("--scan-id", required=True, help="3RScan scene id.")
    parser.add_argument(
        "--object-id",
        action="append",
        required=True,
        help="Object id to visualize. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mask-source",
        default="pred_projection",
        help="Mask source under files/, e.g. pred_projection or gt_projection.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Directory containing PLY debug outputs. Defaults to <repo>/vis.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for PNG previews. Defaults to <repo>/vis/previews/<scan_id>/.",
    )
    parser.add_argument(
        "--top-k-frames",
        type=int,
        default=3,
        help="How many top-area frames to render per object.",
    )
    parser.add_argument(
        "--frame-pad",
        type=int,
        default=40,
        help="Padding around the mask crop in pixels.",
    )
    parser.add_argument(
        "--ply-kind",
        action="append",
        default=None,
        help=(
            "PLY suffix to preview, e.g. lifted_points_world, "
            "lifted_points_normalized, voxel. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--crop-only",
        action="store_true",
        help="Only save cropped overlay views, not full-frame overlays.",
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return cleaned.strip("_") or "unknown"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_object_labels(data_root: Path, scan_id: str) -> dict[str, str]:
    objects_path = data_root / "files" / "objects.json"
    with open(objects_path) as f:
        scans = json.load(f)["scans"]
    for scan in scans:
        if scan["scan"] == scan_id:
            return {str(obj["id"]): obj.get("label", "unknown") for obj in scan["objects"]}
    raise KeyError(f"Scan {scan_id} not found in {objects_path}")


def top_frames_for_object(masks: dict, object_id: int, top_k: int) -> list[tuple[int, str]]:
    areas = []
    for frame_id, mask in masks.items():
        area = int((mask == object_id).sum())
        if area > 0:
            areas.append((area, frame_id))
    areas.sort(reverse=True)
    return areas[:top_k]


def load_color_frame(data_root: Path, scan_id: str, frame_id: str) -> Image.Image:
    scan_dir = data_root / "scenes" / scan_id
    sequence_dir = scan_dir / "sequence"
    filename = f"frame-{frame_id}.color.jpg"

    file_path = sequence_dir / filename
    if file_path.exists():
        return Image.open(file_path).convert("RGB")

    sequence_zip = scan_dir / "sequence.zip"
    if sequence_zip.exists():
        with zipfile.ZipFile(sequence_zip) as zf:
            with zf.open(filename) as f:
                return Image.open(BytesIO(f.read())).convert("RGB")

    raise FileNotFoundError(
        f"Could not find {filename} in {sequence_dir} or {sequence_zip}"
    )


def render_overlay(
    image: Image.Image,
    mask: np.ndarray,
    object_id: int,
    crop_only: bool,
    frame_pad: int,
    output_prefix: Path,
) -> list[Path]:
    arr = np.array(image)
    object_mask = mask == object_id

    if not np.any(object_mask):
        return []

    overlay = arr.copy()
    overlay[object_mask] = (
        0.65 * overlay[object_mask] + 0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)
    full = Image.fromarray(overlay)

    ys, xs = np.where(object_mask)
    draw = ImageDraw.Draw(full)
    draw.rectangle(
        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        outline=(255, 255, 0),
        width=3,
    )

    outputs = []
    if not crop_only:
        full_out = output_prefix.with_name(output_prefix.name + "_overlay_full.png")
        full.save(full_out)
        outputs.append(full_out)

    x0 = max(0, int(xs.min()) - frame_pad)
    y0 = max(0, int(ys.min()) - frame_pad)
    x1 = min(arr.shape[1], int(xs.max()) + frame_pad)
    y1 = min(arr.shape[0], int(ys.max()) + frame_pad)
    crop = full.crop((x0, y0, x1, y1))
    crop_out = output_prefix.with_name(output_prefix.name + "_overlay_crop.png")
    crop.save(crop_out)
    outputs.append(crop_out)
    return outputs


def render_ply_preview(src: Path, output_file: Path) -> Optional[Path]:
    if not src.exists():
        return None

    pcd = o3d.io.read_point_cloud(str(src))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(pts) // 25000)
    pts_s = pts[::step]
    cols_s = cols[::step] if cols is not None and len(cols) == len(pts) else "#4C78A8"
    ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2], c=cols_s, s=1.5, depthshade=False)

    mins = pts_s.min(axis=0)
    maxs = pts_s.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_title(src.name)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_file


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique_object_ids(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    scan_id = args.scan_id
    object_ids = unique_object_ids(args.object_id)
    label_map = load_object_labels(data_root, scan_id)
    masks = scan3r.load_masks(str(data_root), scan_id, mask_source=args.mask_source)

    repo_root = repo_root_from_script()
    vis_dir = Path(args.vis_dir) if args.vis_dir else repo_root / "vis"
    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / "vis" / "previews" / scan_id
    )
    ensure_dir(out_root)

    ply_kinds = args.ply_kind or ["lifted_points_world", "lifted_points_normalized", "voxel"]
    manifest = {"scan_id": scan_id, "objects": []}

    for object_id in object_ids:
        label = label_map.get(object_id, "unknown")
        object_dir = ensure_dir(out_root / f"obj_{object_id}_{sanitize_label(label)}")
        overlays_dir = ensure_dir(object_dir / "overlays")
        plys_dir = ensure_dir(object_dir / "ply_previews")

        frame_entries = top_frames_for_object(masks, int(object_id), args.top_k_frames)
        object_record = {
            "object_id": object_id,
            "label": label,
            "top_frames": frame_entries,
            "overlay_outputs": [],
            "ply_outputs": [],
        }

        for area, frame_id in frame_entries:
            image = load_color_frame(data_root, scan_id, frame_id)
            output_prefix = overlays_dir / f"{scan_id}_{object_id}_{frame_id}"
            outputs = render_overlay(
                image=image,
                mask=masks[frame_id],
                object_id=int(object_id),
                crop_only=args.crop_only,
                frame_pad=args.frame_pad,
                output_prefix=output_prefix,
            )
            object_record["overlay_outputs"].extend(str(p) for p in outputs)
            print(
                f"[overlay] scan={scan_id} obj={object_id} label={label} "
                f"frame={frame_id} area={area}"
            )
            for path in outputs:
                print(path)

        for kind in ply_kinds:
            src = vis_dir / f"{scan_id}_{object_id}_{kind}.ply"
            out = plys_dir / f"{scan_id}_{object_id}_{kind}.png"
            rendered = render_ply_preview(src, out)
            if rendered is not None:
                object_record["ply_outputs"].append(str(rendered))
                print(f"[ply-preview] {rendered}")
            else:
                print(f"[skip missing ply] {src}")

        manifest["objects"].append(object_record)

    manifest_path = out_root / f"{scan_id}_debug_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
