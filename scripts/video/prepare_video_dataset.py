#!/usr/bin/env python3
"""Convert one or more RGB videos into a small 3RScan-like dataset root.

The produced layout is intentionally minimal but matches what the RGB-only
Object-X preprocessing pipeline needs:

  <output-root>/
    scenes/<scene-id>/sequence/_info.txt
    scenes/<scene-id>/sequence/frame-000000.color.jpg
    files/{train,val,test}_scans.txt
    files/{train,val,test}_resplit_scans.txt
    files/3RScan.json
    files/objects.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="Zip archive containing videos.")
    source.add_argument("--video", type=Path, nargs="+", help="One or more video files.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scene-prefix", default="custom_video")
    parser.add_argument("--max-frames", type=int, default=220)
    parser.add_argument("--max-long-side", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--focal-width-ratio", type=float, default=0.92)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep extracted source videos under <output-root>/source_videos. By default they are removed after frame export.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "video"


def list_videos_from_zip(zip_path: Path, extract_dir: Path, force: bool) -> list[Path]:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    video_paths: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            info
            for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in VIDEO_EXTS
        ]
        if not members:
            raise RuntimeError(f"No video files found in {zip_path}")
        for info in members:
            out_name = slugify(Path(info.filename).stem) + Path(info.filename).suffix.lower()
            out_path = extract_dir / out_name
            if not out_path.exists() or force:
                print(f"[video-dataset] extracting {info.filename} -> {out_path}")
                with zf.open(info, "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            video_paths.append(out_path)
    return video_paths


def target_size(width: int, height: int, max_long_side: int) -> tuple[int, int]:
    if max(width, height) <= max_long_side:
        out_w, out_h = width, height
    else:
        scale = float(max_long_side) / float(max(width, height))
        out_w = max(2, int(round(width * scale)))
        out_h = max(2, int(round(height * scale)))
    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(2, out_w), max(2, out_h)


def sampled_indices(total_frames: int, max_frames: int) -> list[int]:
    n_out = min(max(1, int(max_frames)), int(total_frames))
    if n_out == total_frames:
        return list(range(total_frames))
    if n_out == 1:
        return [0]
    return [
        int(round(i * (total_frames - 1) / float(n_out - 1)))
        for i in range(n_out)
    ]


def write_info(path: Path, width: int, height: int, n_frames: int, focal_ratio: float) -> None:
    fx = fy = float(width) * float(focal_ratio)
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    depth_w, depth_h = 224, 172
    dfx = fx * (float(depth_w) / float(width))
    dfy = fy * (float(depth_h) / float(height))
    dcx = (float(depth_w) - 1.0) * 0.5
    dcy = (float(depth_h) - 1.0) * 0.5
    ident_4x4 = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    path.write_text(
        "\n".join(
            [
                "m_versionNumber = 4",
                "m_sensorName = custom RGB video (synthetic calibration)",
                f"m_colorWidth = {width}",
                f"m_colorHeight = {height}",
                f"m_depthWidth = {depth_w}",
                f"m_depthHeight = {depth_h}",
                "m_depthShift = 1000",
                f"m_calibrationColorIntrinsic = {fx:.6f} 0 {cx:.6f} 0 0 {fy:.6f} {cy:.6f} 0 0 0 1 0 0 0 0 1 ",
                f"m_calibrationColorExtrinsic = {ident_4x4} ",
                f"m_calibrationDepthIntrinsic = {dfx:.6f} 0 {dcx:.6f} 0 0 {dfy:.6f} {dcy:.6f} 0 0 0 1 0 0 0 0 1 ",
                f"m_calibrationDepthExtrinsic = {ident_4x4} ",
                f"m_frames.size = {n_frames}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_video(
    video_path: Path,
    output_root: Path,
    scene_id: str,
    max_frames: int,
    max_long_side: int,
    jpeg_quality: int,
    focal_ratio: float,
    force: bool,
) -> dict:
    scene_dir = output_root / "scenes" / scene_id
    sequence_dir = scene_dir / "sequence"
    if sequence_dir.exists():
        if not force:
            existing = sorted(sequence_dir.glob("frame-*.color.jpg"))
            if existing and (sequence_dir / "_info.txt").exists():
                print(f"[video-dataset] keeping existing scene {scene_id}: {sequence_dir}")
                return {
                    "scene_id": scene_id,
                    "video": str(video_path),
                    "frames": len(existing),
                    "sequence_dir": str(sequence_dir),
                    "kept_existing": True,
                }
            raise FileExistsError(f"{sequence_dir} exists but is incomplete; pass --force")
        shutil.rmtree(sequence_dir)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}")
    out_w, out_h = target_size(src_w, src_h, max_long_side)
    indices = sampled_indices(total_frames, max_frames)
    selected = set(indices)

    print(
        "[video-dataset] "
        f"scene={scene_id} source={video_path.name} "
        f"frames={total_frames} fps={fps:.3f} size={src_w}x{src_h} "
        f"export={out_w}x{out_h} selected={len(indices)}"
    )

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    written = 0
    frame_idx = 0
    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in selected:
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            out_path = sequence_dir / f"frame-{written:06d}.color.jpg"
            if not cv2.imwrite(str(out_path), frame, encode_params):
                raise OSError(f"Failed to write {out_path}")
            written += 1
        frame_idx += 1
    cap.release()

    if written == 0:
        raise RuntimeError(f"No frames exported from {video_path}")
    write_info(sequence_dir / "_info.txt", out_w, out_h, written, focal_ratio)
    manifest = {
        "scene_id": scene_id,
        "video": str(video_path),
        "source_frames": total_frames,
        "source_fps": fps,
        "source_size": f"{src_w}x{src_h}",
        "exported_frames": written,
        "exported_size": f"{out_w}x{out_h}",
        "max_frames": max_frames,
        "max_long_side": max_long_side,
        "focal_width_ratio": focal_ratio,
        "sequence_dir": str(sequence_dir),
    }
    (scene_dir / "custom_video_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[video-dataset] wrote {written} frames to {sequence_dir}")
    return manifest


def write_dataset_files(output_root: Path, scene_ids: list[str], split: str) -> None:
    files_dir = output_root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        payload = "\n".join(scene_ids) + "\n" if name == split else ""
        (files_dir / f"{name}_scans.txt").write_text(payload, encoding="utf-8")
        (files_dir / f"{name}_resplit_scans.txt").write_text(payload, encoding="utf-8")
    scan_payload = [
        {
            "reference": scene_id,
            "scans": [],
            "type": "validation" if split == "val" else split,
            "ambiguity": [],
            "generated": True,
        }
        for scene_id in scene_ids
    ]
    (files_dir / "3RScan.json").write_text(json.dumps(scan_payload, indent=2), encoding="utf-8")
    objects_payload = {
        "scans": [
            {
                "scan": scene_id,
                "objects": [],
                "generated": True,
            }
            for scene_id in scene_ids
        ]
    }
    (files_dir / "objects.json").write_text(json.dumps(objects_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    extract_dir = args.output_root / "source_videos"
    if args.zip:
        video_paths = list_videos_from_zip(args.zip, extract_dir, force=args.force)
    else:
        video_paths = [path.resolve() for path in args.video]

    manifests = []
    for video_path in sorted(video_paths, key=lambda p: p.name):
        scene_id = f"{slugify(args.scene_prefix)}_{slugify(video_path.stem)}"
        manifests.append(
            prepare_video(
                video_path=video_path,
                output_root=args.output_root,
                scene_id=scene_id,
                max_frames=args.max_frames,
                max_long_side=args.max_long_side,
                jpeg_quality=args.jpeg_quality,
                focal_ratio=args.focal_width_ratio,
                force=args.force,
            )
        )
    scene_ids = [item["scene_id"] for item in manifests]
    write_dataset_files(args.output_root, scene_ids, split=args.split)
    (args.output_root / "custom_video_dataset_manifest.json").write_text(
        json.dumps({"scenes": manifests, "split": args.split}, indent=2),
        encoding="utf-8",
    )
    if args.zip:
        if args.keep_extracted:
            print(f"[video-dataset] kept extracted videos at {extract_dir}")
        elif extract_dir.exists():
            shutil.rmtree(extract_dir)
            print(f"[video-dataset] removed extracted source videos at {extract_dir}")
    print("[video-dataset] scenes:")
    for scene_id in scene_ids:
        print(f"  {scene_id}")
    print(f"[video-dataset] output_root={args.output_root}")


if __name__ == "__main__":
    main()
