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
    cv2.imwrite(os.path.join(output_dir, f"frame-{frame_idx:04d}.depth.pgm"), depth_mm)

def save_poses(poses, output_dir, scene_id):
    """Save poses per frame as .txt."""
    for i, pose in enumerate(poses):
        np.savetxt(os.path.join(output_dir, f"frame-{i:04d}.pose.txt"), pose)

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


def _frame_score_metrics(frame):
    if frame is None or frame.size == 0:
        return {
            "sharpness": 0.0,
            "edge_density": 0.0,
            "upper_edge_density": 0.0,
            "lower_heavy_edges": 0.0,
            "contrast": 0.0,
            "clip_frac": 1.0,
        }

    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side > 256:
        scale = 256.0 / float(max_side)
        frame = cv2.resize(
            frame,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 80, 160)
    split = max(1, edges.shape[0] // 2)

    edge_density = float(edges.mean() / 255.0)
    upper_edge_density = float(edges[:split].mean() / 255.0)
    lower_edge_density = float(edges[split:].mean() / 255.0)

    return {
        "sharpness": float(np.log1p(float(lap.var()))),
        "edge_density": edge_density,
        "upper_edge_density": upper_edge_density,
        "lower_heavy_edges": float(max(0.0, lower_edge_density - upper_edge_density)),
        "contrast": float(gray.std() / 255.0),
        "clip_frac": float(((gray < 12) | (gray > 245)).mean()),
    }


def _zscore(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _quality_scores(frames):
    metrics = [_frame_score_metrics(frame) for frame in frames]
    sharpness = _zscore([m["sharpness"] for m in metrics])
    edge_density = _zscore([m["edge_density"] for m in metrics])
    upper_edges = _zscore([m["upper_edge_density"] for m in metrics])
    lower_heavy = _zscore([m["lower_heavy_edges"] for m in metrics])
    contrast = _zscore([m["contrast"] for m in metrics])
    clip_frac = _zscore([m["clip_frac"] for m in metrics])

    scores = (
        1.35 * sharpness
        + 1.00 * edge_density
        + 0.90 * upper_edges
        + 0.60 * contrast
        - 0.85 * lower_heavy
        - 0.60 * clip_frac
    )
    return scores.astype(np.float32)


def _pick_best_in_ranges(scores, ranges):
    chosen = []
    used = set()
    for start, end in ranges:
        if end <= start:
            continue
        local = scores[start:end]
        order = np.argsort(local)[::-1]
        picked = None
        for rel_idx in order:
            idx = start + int(rel_idx)
            if idx not in used:
                picked = idx
                break
        if picked is None:
            picked = start + int(np.argmax(local))
        chosen.append(picked)
        used.add(picked)
    return sorted(chosen)


def _candidate_groups_in_ranges(scores, ranges, top_k):
    groups = []
    for start, end in ranges:
        if end <= start:
            continue
        local = scores[start:end]
        order = np.argsort(local)[::-1]
        group = []
        for rel_idx in order[:top_k]:
            group.append(start + int(rel_idx))
        if not group:
            group = [start + int(np.argmax(local))]
        groups.append(group)
    return groups


def select_keyframe_candidate_groups(
    frame_paths,
    strategy="stride",
    stride=10,
    n_keyframes=20,
    frames=None,
    top_k=4,
):
    num_frames = len(frame_paths)
    if num_frames == 0:
        return []

    if strategy == "quality_uniform":
        if frames is None:
            raise ValueError("quality_uniform requires loaded frames")
        scores = _quality_scores(frames)
        edges = np.linspace(0, num_frames, n_keyframes + 1, dtype=int)
        ranges = [(int(edges[i]), int(edges[i + 1])) for i in range(n_keyframes)]
        return _candidate_groups_in_ranges(scores, ranges, top_k=top_k)

    if strategy == "quality_stride":
        if frames is None:
            raise ValueError("quality_stride requires loaded frames")
        scores = _quality_scores(frames)
        ranges = [
            (start, min(start + stride, num_frames))
            for start in range(0, num_frames, stride)
        ]
        return _candidate_groups_in_ranges(scores, ranges, top_k=top_k)

    return [[idx] for idx in select_keyframes(frame_paths, strategy, stride, n_keyframes, frames=frames)]


def select_keyframes(frame_paths, strategy="stride", stride=10, n_keyframes=20, frames=None):
    """Select keyframes."""
    num_frames = len(frame_paths)
    if num_frames == 0:
        return []

    if strategy == "stride":
        return list(range(0, num_frames, stride))

    if strategy == "uniform":
        indices = np.linspace(0, num_frames - 1, n_keyframes, dtype=int)
        return indices.tolist()

    if strategy == "quality_uniform":
        if frames is None:
            raise ValueError("quality_uniform requires loaded frames")
        scores = _quality_scores(frames)
        edges = np.linspace(0, num_frames, n_keyframes + 1, dtype=int)
        ranges = [(int(edges[i]), int(edges[i + 1])) for i in range(n_keyframes)]
        return _pick_best_in_ranges(scores, ranges)

    if strategy == "quality_stride":
        if frames is None:
            raise ValueError("quality_stride requires loaded frames")
        scores = _quality_scores(frames)
        ranges = [
            (start, min(start + stride, num_frames))
            for start in range(0, num_frames, stride)
        ]
        return _pick_best_in_ranges(scores, ranges)

    return list(range(num_frames))
