import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple

def save_masks(masks_list, output_dir, frame_idx, save_format="jpg", scene_id=""):
    """Save masks per object in the new format."""
    # This will be called differently now
    pass

def save_depth(depth, output_dir, frame_idx):
    """Save depth as .pgm."""
    depth_mm = (depth * 1000).astype(np.uint16)  # Convert to mm
    cv2.imwrite(os.path.join(output_dir, f"frame-{int(frame_idx):06d}.depth.pgm"), depth_mm)

def save_poses(poses, output_dir, scene_id=None, frame_idxs=None):
    """Save poses per frame as .txt."""
    if frame_idxs is None:
        frame_idxs = range(len(poses))
    for frame_idx, pose in zip(frame_idxs, poses):
        np.savetxt(
            os.path.join(output_dir, f"frame-{int(frame_idx):06d}.pose.txt"),
            pose,
        )


def _info_intrinsic_line(intrinsic_mat: np.ndarray) -> str:
    fx = float(intrinsic_mat[0, 0])
    fy = float(intrinsic_mat[1, 1])
    cx = float(intrinsic_mat[0, 2])
    cy = float(intrinsic_mat[1, 2])
    values = [
        fx, 0.0, cx, 0.0,
        0.0, fy, cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    return " ".join(f"{value:.8f}" for value in values)


def save_sequence_info(
    output_dir,
    color_size,
    depth_size,
    color_intrinsic,
    depth_intrinsic,
    n_frames=None,
    depth_shift=1000.0,
    metadata=None,
):
    """Write an Object-X/3RScan-style `_info.txt` plus optional camera metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    color_width, color_height = color_size
    depth_width, depth_height = depth_size
    lines = [
        f"m_colorWidth = {int(color_width)}",
        f"m_colorHeight = {int(color_height)}",
        f"m_depthWidth = {int(depth_width)}",
        f"m_depthHeight = {int(depth_height)}",
        f"m_calibrationColorIntrinsic = {_info_intrinsic_line(np.asarray(color_intrinsic, dtype=np.float32))}",
        f"m_calibrationDepthIntrinsic = {_info_intrinsic_line(np.asarray(depth_intrinsic, dtype=np.float32))}",
        f"m_depthShift = {float(depth_shift):.8f}",
    ]
    if n_frames is not None:
        lines.append(f"m_frames.size = {int(n_frames)}")

    (output_path / "_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if metadata is not None:
        with open(output_path / "predicted_camera_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

def load_frames_from_scene(scene_dir, ext=".color.jpg", resize=None):
    """Load frames from scene directory."""
    frame_paths = sorted(Path(scene_dir).glob(f"*{ext}"))
    frames = []
    for path in frame_paths:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, resize)
        frames.append(img)
    return frames, [str(p) for p in frame_paths]

def select_keyframes(frame_paths, strategy="stride", stride=10, n_keyframes=20):
    """Select keyframes."""
    if strategy == "stride":
        return list(range(0, len(frame_paths), stride))
    elif strategy == "uniform":
        indices = np.linspace(0, len(frame_paths)-1, n_keyframes, dtype=int)
        return indices.tolist()
    else:
        return list(range(len(frame_paths)))
