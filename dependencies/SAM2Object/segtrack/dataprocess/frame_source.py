"""
Single source of truth for "where a dataset's frames live and in what order".

Both extract_only_jpg.py and get_posed_images.py import this so they cannot drift:
they enumerate the *same* natsorted frame list, then write each frame under its
*position* `i` as a zero-padded 6-digit name ({i:06d}). This is what makes the
SAM2-tracking positional index match the posed_images basename and the
maskraw_{i:06d}.png masks (see mask_convert.py / seg_tracking.py / sam2object.py).

Per-dataset, only the source layout differs:
  - 3RScan: <scenes_dir>/<scene>/sequence/frame-NNNNNN.{color.jpg,depth.pgm,pose.txt}
            intrinsics from sequence/_info.txt
            (3RScan sequences are gap-free, so {i:06d} reproduces the original ids)
  - ScanNet: <scannet_src>/<scene>/NNNNN.{jpg,png,txt} (pre-extracted posed_images)
            intrinsics from intrinsic.txt / depth_intrinsic.txt
            (5-digit, step-10 numbering -> re-indexing by position is the fix)
"""

import os
from collections import namedtuple
from os.path import isdir, join

import numpy as np

opj = os.path.join

FrameRef = namedtuple("FrameRef", ["color_path", "depth_path", "pose_path"])

DEFAULT_SCANNET_POSED_SRC = "/cluster/project/cvg/data/scannet/posed_images"


def _list_frames_scannet(scene, scannet_src):
    scene_dir = opj(scannet_src, scene)
    if not isdir(scene_dir):
        print(f"[SKIP] No ScanNet posed_images folder found for {scene} ({scene_dir})")
        return []
    frames = []
    for name in os.listdir(scene_dir):
        if not name.endswith(".jpg"):
            continue
        stem = name[:-len(".jpg")]          # "00010"
        if not stem.isdigit():              # skip any stray non-frame jpgs
            continue
        frames.append((
            int(stem),
            FrameRef(
                color_path=opj(scene_dir, name),
                depth_path=opj(scene_dir, stem + ".png"),
                pose_path=opj(scene_dir, stem + ".txt"),
            ),
        ))
    frames.sort(key=lambda x: x[0])
    return [f for _, f in frames]


def _list_frames_3rscan(scene, scenes_dir):
    seq = opj(scenes_dir, scene, "sequence")
    if not isdir(seq):
        print(f"[SKIP] No sequence folder found for {scene} ({seq})")
        return []
    frames = []
    for name in os.listdir(seq):
        if not name.endswith(".color.jpg"):
            continue
        # frame-000123.color.jpg -> 000123
        frame_id = name[len("frame-"):-len(".color.jpg")]
        frames.append((
            int(frame_id),
            FrameRef(
                color_path=opj(seq, name),
                depth_path=opj(seq, f"frame-{frame_id}.depth.pgm"),
                pose_path=opj(seq, f"frame-{frame_id}.pose.txt"),
            ),
        ))
    frames.sort(key=lambda x: x[0])
    return [f for _, f in frames]


def list_frames(dataset, scene, scannet_src=None, scenes_dir=None):
    """Return the natsorted, paired list of FrameRef for one scene.

    The position of each FrameRef in the returned list IS its output index `i`.
    """
    if dataset == "ScanNet":
        return _list_frames_scannet(scene, scannet_src or DEFAULT_SCANNET_POSED_SRC)
    elif dataset == "3RScan":
        return _list_frames_3rscan(scene, scenes_dir)
    raise ValueError(f"Unknown DATASET: {dataset}")


def _parse_3rscan_intrinsics(info_path, key):
    """Read a 4x4 intrinsic from 3RScan sequence/_info.txt (e.g.
    m_calibrationColorIntrinsic / m_calibrationDepthIntrinsic)."""
    with open(info_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(key):
            values = [float(v) for v in line.split("=")[1].strip().split()]
            if len(values) == 16:
                return np.array(values, dtype=np.float32).reshape(4, 4)
            elif len(values) == 9:
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = np.array(values, dtype=np.float32).reshape(3, 3)
                return mat
            raise ValueError(f"Unexpected intrinsic size for {key}: {len(values)} values")
    raise FileNotFoundError(f"Could not find {key} in {info_path}")


def read_intrinsics(dataset, scene, scannet_src=None, scenes_dir=None):
    """Return (K_color, K_depth) as 4x4 float32 arrays for one scene."""
    if dataset == "ScanNet":
        scene_dir = opj(scannet_src or DEFAULT_SCANNET_POSED_SRC, scene)
        k_color = np.loadtxt(opj(scene_dir, "intrinsic.txt")).astype(np.float32)
        k_depth = np.loadtxt(opj(scene_dir, "depth_intrinsic.txt")).astype(np.float32)
        return k_color, k_depth
    elif dataset == "3RScan":
        info_path = opj(scenes_dir, scene, "sequence", "_info.txt")
        k_color = _parse_3rscan_intrinsics(info_path, "m_calibrationColorIntrinsic")
        k_depth = _parse_3rscan_intrinsics(info_path, "m_calibrationDepthIntrinsic")
        return k_color, k_depth
    raise ValueError(f"Unknown DATASET: {dataset}")


def link_or_copy(src, dst):
    """Idempotently point dst at src via a symlink (src is read-only project data,
    dst lives on scratch). Replaces any existing dst link/file."""
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)
