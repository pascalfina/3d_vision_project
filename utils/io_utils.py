import os
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

def load_frames_from_scene(scene_dir, ext=".color.jpg", resize=None):
    """Load frames from scene directory."""
    frame_paths = sorted(Path(scene_dir).glob(f"*{ext}"))
    frames = []
    for path in frame_paths:
        img = cv2.imread(str(path))
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
