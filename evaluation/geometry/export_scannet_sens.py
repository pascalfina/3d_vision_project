#!/usr/bin/env python3
"""Export ScanNet .sens files into Object-X/geometry-eval friendly folders.

The ScanNet SDK bundled in the course zip ships a Python 2 reader.  This file is
the small Python 3 subset we need: color images, depth images, poses and
intrinsics in the usual ScanNet layout:

    data/color/<frame>.jpg
    data/depth/<frame>.png
    data/pose/<frame>.txt
    data/intrinsic/intrinsic_*.txt
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import zlib
from math import ceil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


@dataclass
class FrameRecord:
    camera_to_world: np.ndarray
    color_data: bytes
    depth_data: bytes


@dataclass
class SensorHeader:
    intrinsic_color: np.ndarray
    extrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_depth: np.ndarray
    color_compression_type: str
    depth_compression_type: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    num_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ScanNet .sens RGB-D data.")
    parser.add_argument("--filename", required=True, help="Input .sens file.")
    parser.add_argument("--output-path", required=True, help="Output data directory.")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--auto-frame-skip-for-max",
        action="store_true",
        help="Increase frame skip so --max-frames samples the full video span.",
    )
    parser.add_argument("--export-depth-images", action="store_true", default=True)
    parser.add_argument("--export-color-images", action="store_true", default=True)
    parser.add_argument("--export-poses", action="store_true", default=True)
    parser.add_argument("--export-intrinsics", action="store_true", default=True)
    parser.add_argument(
        "--compat-sequence-path",
        help=(
            "Optional 3RScan/Object-X compatible sequence directory with "
            "frame-XXXXXX.color.jpg symlinks and _info.txt."
        ),
    )
    return parser.parse_args()


def _read_exact(handle, n_bytes: int) -> bytes:
    data = handle.read(n_bytes)
    if len(data) != n_bytes:
        raise EOFError(f"Unexpected end of .sens file while reading {n_bytes} bytes")
    return data


def read_header(handle) -> SensorHeader:
    version = struct.unpack("I", _read_exact(handle, 4))[0]
    if version != 4:
        raise ValueError(f"Unsupported ScanNet .sens version {version}; expected 4")
    strlen = struct.unpack("Q", _read_exact(handle, 8))[0]
    _sensor_name = _read_exact(handle, strlen)
    intrinsic_color = np.asarray(struct.unpack("f" * 16, _read_exact(handle, 16 * 4)), dtype=np.float32).reshape(4, 4)
    extrinsic_color = np.asarray(struct.unpack("f" * 16, _read_exact(handle, 16 * 4)), dtype=np.float32).reshape(4, 4)
    intrinsic_depth = np.asarray(struct.unpack("f" * 16, _read_exact(handle, 16 * 4)), dtype=np.float32).reshape(4, 4)
    extrinsic_depth = np.asarray(struct.unpack("f" * 16, _read_exact(handle, 16 * 4)), dtype=np.float32).reshape(4, 4)
    color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", _read_exact(handle, 4))[0]]
    depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", _read_exact(handle, 4))[0]]
    color_width = struct.unpack("I", _read_exact(handle, 4))[0]
    color_height = struct.unpack("I", _read_exact(handle, 4))[0]
    depth_width = struct.unpack("I", _read_exact(handle, 4))[0]
    depth_height = struct.unpack("I", _read_exact(handle, 4))[0]
    depth_shift = struct.unpack("f", _read_exact(handle, 4))[0]
    num_frames = struct.unpack("Q", _read_exact(handle, 8))[0]
    return SensorHeader(
        intrinsic_color=intrinsic_color,
        extrinsic_color=extrinsic_color,
        intrinsic_depth=intrinsic_depth,
        extrinsic_depth=extrinsic_depth,
        color_compression_type=color_compression_type,
        depth_compression_type=depth_compression_type,
        color_width=color_width,
        color_height=color_height,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_shift=depth_shift,
        num_frames=int(num_frames),
    )


def read_frame(handle) -> FrameRecord:
    pose = np.asarray(struct.unpack("f" * 16, _read_exact(handle, 16 * 4)), dtype=np.float32).reshape(4, 4)
    _timestamp_color = struct.unpack("Q", _read_exact(handle, 8))[0]
    _timestamp_depth = struct.unpack("Q", _read_exact(handle, 8))[0]
    color_size = struct.unpack("Q", _read_exact(handle, 8))[0]
    depth_size = struct.unpack("Q", _read_exact(handle, 8))[0]
    color_data = _read_exact(handle, color_size)
    depth_data = _read_exact(handle, depth_size)
    return FrameRecord(camera_to_world=pose, color_data=color_data, depth_data=depth_data)


def decode_color(frame: FrameRecord, compression_type: str) -> Image.Image:
    if compression_type not in {"jpeg", "png"}:
        raise ValueError(f"Unsupported ScanNet color compression: {compression_type}")
    image = Image.open(io.BytesIO(frame.color_data))
    image.load()
    return image.convert("RGB")


def decode_depth(frame: FrameRecord, header: SensorHeader) -> np.ndarray:
    if header.depth_compression_type != "zlib_ushort":
        raise ValueError(f"Unsupported ScanNet depth compression: {header.depth_compression_type}")
    depth_data = zlib.decompress(frame.depth_data)
    return np.frombuffer(depth_data, dtype=np.uint16).reshape(header.depth_height, header.depth_width)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.9f")


def write_info_file(path: Path, header: SensorHeader, frame_count: int) -> None:
    def flat(matrix: np.ndarray) -> str:
        return " ".join(f"{float(v):.9f}" for v in np.asarray(matrix).reshape(-1))

    lines = [
        "m_versionNumber = 4",
        "m_sensorName = ScanNet",
        f"m_colorWidth = {int(header.color_width)}",
        f"m_colorHeight = {int(header.color_height)}",
        f"m_depthWidth = {int(header.depth_width)}",
        f"m_depthHeight = {int(header.depth_height)}",
        f"m_depthShift = {float(header.depth_shift):.9f}",
        f"m_frames.size = {int(frame_count)}",
        f"m_calibrationColorIntrinsic = {flat(header.intrinsic_color)}",
        f"m_calibrationColorExtrinsic = {flat(header.extrinsic_color)}",
        f"m_calibrationDepthIntrinsic = {flat(header.intrinsic_depth)}",
        f"m_calibrationDepthExtrinsic = {flat(header.extrinsic_depth)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel_src = os.path.relpath(src, dst.parent)
    try:
        dst.symlink_to(rel_src)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def write_compat_frame_links(compat_dir: Path, frame_id: int, color_path: Path, depth_path: Path, pose_path: Path) -> None:
    frame_name = f"frame-{frame_id:06d}"
    safe_symlink_or_copy(color_path, compat_dir / f"{frame_name}.color.jpg")
    safe_symlink_or_copy(depth_path, compat_dir / f"{frame_name}.depth.pgm")
    safe_symlink_or_copy(pose_path, compat_dir / f"{frame_name}.pose.txt")


def export_sens(args: argparse.Namespace) -> None:
    sens_path = Path(args.filename)
    out_dir = Path(args.output_path)
    compat_dir = Path(args.compat_sequence_path) if args.compat_sequence_path else None
    color_dir = out_dir / "color"
    depth_dir = out_dir / "depth"
    pose_dir = out_dir / "pose"
    intrinsic_dir = out_dir / "intrinsic"
    for path in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        path.mkdir(parents=True, exist_ok=True)

    frame_skip = max(1, int(args.frame_skip))
    written = 0
    with sens_path.open("rb") as handle:
        header = read_header(handle)
        if args.auto_frame_skip_for_max and args.max_frames:
            frame_skip = max(frame_skip, int(ceil(float(header.num_frames) / float(args.max_frames))))
        if args.export_intrinsics:
            write_matrix(intrinsic_dir / "intrinsic_color.txt", header.intrinsic_color)
            write_matrix(intrinsic_dir / "extrinsic_color.txt", header.extrinsic_color)
            write_matrix(intrinsic_dir / "intrinsic_depth.txt", header.intrinsic_depth)
            write_matrix(intrinsic_dir / "extrinsic_depth.txt", header.extrinsic_depth)

        for frame_idx in range(header.num_frames):
            frame = read_frame(handle)
            if frame_idx % frame_skip != 0:
                continue
            if args.max_frames and written >= int(args.max_frames):
                break
            output_frame_id = written
            pose_path = pose_dir / f"{output_frame_id}.txt"
            color_path = color_dir / f"{output_frame_id}.jpg"
            depth_path = depth_dir / f"{output_frame_id}.png"
            if args.export_poses:
                write_matrix(pose_path, frame.camera_to_world)
            if args.export_color_images:
                decode_color(frame, header.color_compression_type).save(color_path)
            if args.export_depth_images:
                depth = decode_depth(frame, header)
                Image.fromarray(depth).save(depth_path)
            if compat_dir is not None:
                write_compat_frame_links(compat_dir, output_frame_id, color_path, depth_path, pose_path)
            written += 1

        if compat_dir is not None:
            write_info_file(compat_dir / "_info.txt", header, written)

    print(
        f"[scannet-export] {sens_path.name}: exported {written}/{header.num_frames} frames "
        f"(frame_skip={frame_skip}, max_frames={int(args.max_frames)}) to {out_dir}"
    )
    if compat_dir is not None:
        print(f"[scannet-export] compatibility sequence: {compat_dir}")


def main() -> None:
    export_sens(parse_args())


if __name__ == "__main__":
    main()
