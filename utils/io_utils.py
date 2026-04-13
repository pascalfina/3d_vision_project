import os
import numpy as np
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
    import cv2

    frame_paths = sorted(Path(scene_dir).glob(f"*{ext}"))
    frames = []
    for path in frame_paths:
        img = cv2.imread(str(path))
        if resize:
            img = cv2.resize(img, resize)
        frames.append(img)
    return frames, [str(p) for p in frame_paths]


def _frame_score_metrics(frame):
    import cv2

    if frame is None or frame.size == 0:
        return {
            "sharpness": 0.0,
            "edge_density": 0.0,
            "edge_spread": 0.0,
            "edge_entropy": 0.0,
            "texture_regular": 0.0,
            "edge_concentration": 0.0,
            "center_bias": 0.0,
            "border_heavy_edges": 0.0,
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
    edge_density = float(edges.mean() / 255.0)

    grid = 3
    gh = np.linspace(0, edges.shape[0], grid + 1, dtype=int)
    gw = np.linspace(0, edges.shape[1], grid + 1, dtype=int)
    cell_edges = []
    for gy in range(grid):
        for gx in range(grid):
            cell = edges[gh[gy] : gh[gy + 1], gw[gx] : gw[gx + 1]]
            if cell.size > 0:
                cell_edges.append(float(cell.mean() / 255.0))
    cell_edges = np.asarray(cell_edges, dtype=np.float32)
    active_cells = float((cell_edges > 0.025).mean()) if cell_edges.size else 0.0
    edge_balance = float(
        1.0 - (cell_edges.std() / max(cell_edges.mean(), 1e-4))
    ) if cell_edges.size else 0.0
    edge_spread = max(0.0, 0.55 * active_cells + 0.45 * edge_balance)
    if cell_edges.size and float(cell_edges.sum()) > 1e-6:
        cell_prob = cell_edges / float(cell_edges.sum())
        edge_entropy = float(
            -(cell_prob * np.log(cell_prob + 1e-8)).sum() / np.log(max(len(cell_prob), 2))
        )
        edge_cv = float(cell_edges.std() / max(cell_edges.mean(), 1e-4))
        edge_concentration = float(
            np.clip(cell_edges.max() / max(cell_edges.mean(), 1e-4) - 1.8, 0.0, 2.0) / 2.0
        )
        texture_regular = float(
            np.clip((active_cells - 0.45) / 0.45, 0.0, 1.0)
            * np.clip((0.70 - edge_cv) / 0.70, 0.0, 1.0)
        )
    else:
        edge_entropy = 0.0
        edge_concentration = 0.0
        texture_regular = 0.0

    margin_y = max(1, int(round(edges.shape[0] * 0.12)))
    margin_x = max(1, int(round(edges.shape[1] * 0.12)))
    border_mask = np.zeros_like(edges, dtype=np.bool_)
    border_mask[:margin_y, :] = True
    border_mask[-margin_y:, :] = True
    border_mask[:, :margin_x] = True
    border_mask[:, -margin_x:] = True
    border_edge_density = float(edges[border_mask].mean() / 255.0)
    inner_edge_density = float(edges[~border_mask].mean() / 255.0) if (~border_mask).any() else 0.0
    center_y0 = max(0, int(round(edges.shape[0] * 0.20)))
    center_y1 = min(edges.shape[0], int(round(edges.shape[0] * 0.80)))
    center_x0 = max(0, int(round(edges.shape[1] * 0.20)))
    center_x1 = min(edges.shape[1], int(round(edges.shape[1] * 0.80)))
    center_mask = np.zeros_like(edges, dtype=np.bool_)
    center_mask[center_y0:center_y1, center_x0:center_x1] = True
    center_edge_density = float(edges[center_mask].mean() / 255.0) if center_mask.any() else 0.0
    peripheral_edge_density = float(edges[~center_mask].mean() / 255.0) if (~center_mask).any() else 0.0

    return {
        "sharpness": float(np.log1p(float(lap.var()))),
        "edge_density": edge_density,
        "edge_spread": float(edge_spread),
        "edge_entropy": float(edge_entropy),
        "texture_regular": float(texture_regular),
        "edge_concentration": float(edge_concentration),
        "center_bias": float(max(0.0, center_edge_density - peripheral_edge_density)),
        "border_heavy_edges": float(max(0.0, border_edge_density - inner_edge_density)),
        "contrast": float(gray.std() / 255.0),
        "clip_frac": float(((gray < 12) | (gray > 245)).mean()),
    }


def _scene_metric_raw_score(metrics):
    return float(
        1.35 * metrics["sharpness"]
        + 1.00 * metrics["edge_density"]
        + 0.95 * metrics["edge_spread"]
        + 0.85 * metrics["edge_entropy"]
        + 0.70 * metrics["contrast"]
        - 0.80 * metrics["texture_regular"]
        - 0.65 * metrics["edge_concentration"]
        - 0.40 * metrics["center_bias"]
        - 0.90 * metrics["border_heavy_edges"]
        - 0.60 * metrics["clip_frac"]
    )


def _best_rotated_frame_metrics(frame):
    candidates = []
    for rot_k in range(4):
        rotated = np.ascontiguousarray(np.rot90(frame, rot_k))
        metrics = _frame_score_metrics(rotated)
        candidates.append((_scene_metric_raw_score(metrics), rot_k, rotated, metrics))
    best_score, best_rot_k, best_frame, best_metrics = max(candidates, key=lambda x: x[0])
    best_metrics = dict(best_metrics)
    best_metrics["rotation"] = int(best_rot_k)
    best_metrics["scene_score_raw"] = float(best_score)
    return best_frame, best_metrics


def _zscore(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _normalize_scores(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-6:
        return np.ones_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _pool_grid_mean(grid, out_h, out_w):
    grid = np.asarray(grid, dtype=np.float32)
    if grid.size == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    row_chunks = np.array_split(np.arange(grid.shape[0]), out_h)
    col_chunks = np.array_split(np.arange(grid.shape[1]), out_w)
    pooled = np.zeros((out_h, out_w), dtype=np.float32)
    for oy, rows in enumerate(row_chunks):
        for ox, cols in enumerate(col_chunks):
            cell = grid[np.ix_(rows, cols)]
            if cell.size > 0:
                pooled[oy, ox] = float(cell.mean())
    return pooled


def _quality_scores(frames):
    metrics = [_best_rotated_frame_metrics(frame)[1] for frame in frames]
    sharpness = _zscore([m["sharpness"] for m in metrics])
    edge_density = _zscore([m["edge_density"] for m in metrics])
    edge_spread = _zscore([m["edge_spread"] for m in metrics])
    edge_entropy = _zscore([m["edge_entropy"] for m in metrics])
    texture_regular = _zscore([m["texture_regular"] for m in metrics])
    edge_concentration = _zscore([m["edge_concentration"] for m in metrics])
    center_bias = _zscore([m["center_bias"] for m in metrics])
    border_heavy = _zscore([m["border_heavy_edges"] for m in metrics])
    contrast = _zscore([m["contrast"] for m in metrics])
    clip_frac = _zscore([m["clip_frac"] for m in metrics])

    scores = (
        1.35 * sharpness
        + 1.00 * edge_density
        + 0.95 * edge_spread
        + 0.85 * edge_entropy
        + 0.70 * contrast
        - 0.80 * texture_regular
        - 0.65 * edge_concentration
        - 0.40 * center_bias
        - 0.90 * border_heavy
        - 0.60 * clip_frac
    )
    return scores.astype(np.float32)


def _frame_descriptor(frame):
    import cv2

    if frame is None or frame.size == 0:
        return np.zeros(256, dtype=np.float32)

    frame, best_metrics = _best_rotated_frame_metrics(frame)

    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side > 192:
        scale = 192.0 / float(max_side)
        frame = cv2.resize(
            frame,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    edges = cv2.Canny(gray, 80, 160)
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    edge_grid = np.zeros((6, 6), dtype=np.float32)
    grad_grid = np.zeros((6, 6), dtype=np.float32)
    gh = np.linspace(0, edges.shape[0], 7, dtype=int)
    gw = np.linspace(0, edges.shape[1], 7, dtype=int)
    for gy in range(6):
        for gx in range(6):
            cell = edges[gh[gy] : gh[gy + 1], gw[gx] : gw[gx + 1]]
            if cell.size > 0:
                edge_grid[gy, gx] = float(cell.mean() / 255.0)
            grad_cell = grad_mag[gh[gy] : gh[gy + 1], gw[gx] : gw[gx + 1]]
            if grad_cell.size > 0:
                grad_grid[gy, gx] = float(grad_cell.mean())

    gray_thumb = cv2.resize(gray_f, (10, 10), interpolation=cv2.INTER_AREA).astype(np.float32)
    center_thumb = cv2.resize(gray_f, (6, 6), interpolation=cv2.INTER_AREA).astype(np.float32)
    pooled_gray = _pool_grid_mean(gray_thumb, 5, 5)
    pooled_edges = _pool_grid_mean(edge_grid, 3, 3)
    row_edge = edge_grid.mean(axis=1).astype(np.float32)
    col_edge = edge_grid.mean(axis=0).astype(np.float32)
    row_grad = grad_grid.mean(axis=1).astype(np.float32)
    col_grad = grad_grid.mean(axis=0).astype(np.float32)
    row_gray = gray_thumb.mean(axis=1).astype(np.float32)
    col_gray = gray_thumb.mean(axis=0).astype(np.float32)
    row_edge_grad = np.diff(row_edge).astype(np.float32) if row_edge.size > 1 else np.zeros(0, dtype=np.float32)
    col_edge_grad = np.diff(col_edge).astype(np.float32) if col_edge.size > 1 else np.zeros(0, dtype=np.float32)
    top_gray = float(gray_thumb[:5].mean())
    bottom_gray = float(gray_thumb[5:].mean())
    left_gray = float(gray_thumb[:, :5].mean())
    right_gray = float(gray_thumb[:, 5:].mean())
    top_edge = float(edge_grid[:3].mean())
    bottom_edge = float(edge_grid[3:].mean())
    left_edge = float(edge_grid[:, :3].mean())
    right_edge = float(edge_grid[:, 3:].mean())

    metrics = dict(best_metrics)
    descriptor = np.concatenate(
        [
            (0.85 * gray_thumb).reshape(-1),
            (0.70 * pooled_gray).reshape(-1),
            (1.55 * edge_grid).reshape(-1),
            (1.10 * grad_grid).reshape(-1),
            (0.85 * pooled_edges).reshape(-1),
            (0.65 * center_thumb).reshape(-1),
            (1.05 * row_edge).reshape(-1),
            (1.05 * col_edge).reshape(-1),
            (0.80 * row_grad).reshape(-1),
            (0.80 * col_grad).reshape(-1),
            (0.55 * row_gray).reshape(-1),
            (0.55 * col_gray).reshape(-1),
            (0.70 * row_edge_grad).reshape(-1),
            (0.70 * col_edge_grad).reshape(-1),
            np.array(
                [
                    metrics["sharpness"],
                    metrics["edge_density"],
                    metrics["edge_spread"],
                    metrics["border_heavy_edges"],
                    metrics["contrast"],
                    metrics["clip_frac"],
                    metrics["rotation"] / 3.0,
                    top_gray,
                    bottom_gray,
                    left_gray,
                    right_gray,
                    top_edge,
                    bottom_edge,
                    left_edge,
                    right_edge,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    if norm > 1e-6:
        descriptor /= norm
    return descriptor


def _frame_descriptors(frames):
    if not frames:
        return np.zeros((0, 64), dtype=np.float32)
    return np.stack([_frame_descriptor(frame) for frame in frames], axis=0)


def _temporal_risk_scores(scores, descriptors):
    scores = np.asarray(scores, dtype=np.float32)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    num_frames = int(scores.size)
    if num_frames == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    def _normalize_temporal(values):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return values
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo < 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - lo) / (hi - lo)).astype(np.float32)

    quality = _normalize_scores(scores)
    appearance_jump = np.zeros(num_frames, dtype=np.float32)
    transition_jump = np.zeros(num_frames, dtype=np.float32)
    novelty_to_recent = np.zeros(num_frames, dtype=np.float32)
    score_rise = np.zeros(num_frames, dtype=np.float32)
    score_drop = np.zeros(num_frames, dtype=np.float32)

    for idx in range(num_frames):
        prev_idx = max(0, idx - 1)
        next_idx = min(num_frames - 1, idx + 1)
        if idx > 0:
            prev_sim = float(np.clip(descriptors[idx] @ descriptors[prev_idx], -1.0, 1.0))
            appearance_jump[idx] = 1.0 - prev_sim
            score_rise[idx] = float(max(0.0, quality[idx] - quality[prev_idx]))
            score_drop[idx] = float(max(0.0, quality[prev_idx] - quality[idx]))
        if 0 < idx < num_frames - 1:
            prev_sim = float(np.clip(descriptors[idx] @ descriptors[idx - 1], -1.0, 1.0))
            next_sim = float(np.clip(descriptors[idx] @ descriptors[idx + 1], -1.0, 1.0))
            transition_jump[idx] = 0.5 * ((1.0 - prev_sim) + (1.0 - next_sim))
        recent_start = max(0, idx - 6)
        if idx > recent_start:
            recent = descriptors[recent_start:idx]
            novelty_to_recent[idx] = 1.0 - float(np.max(recent @ descriptors[idx]))

    discovery = (
        1.10 * novelty_to_recent
        + 0.55 * score_rise
        + 0.35 * appearance_jump
        + 0.20 * transition_jump
    )
    repair = (
        0.95 * appearance_jump
        + 0.85 * transition_jump
        + 0.55 * score_drop
        + 0.30 * novelty_to_recent
    )
    return _normalize_temporal(discovery), _normalize_temporal(repair)


def _mmr_select_candidates(candidate_ids, scores, descriptors, target_count, min_gap, diversity_weight=0.75):
    candidate_ids = [int(idx) for idx in candidate_ids]
    if not candidate_ids or target_count <= 0:
        return []

    score_values = np.asarray([scores[idx] for idx in candidate_ids], dtype=np.float32)
    score_min = float(score_values.min()) if score_values.size else 0.0
    score_max = float(score_values.max()) if score_values.size else 0.0
    base_scores = {}
    for idx in candidate_ids:
        raw = float(scores[idx])
        if score_max - score_min < 1e-6:
            base_scores[idx] = 1.0
        else:
            base_scores[idx] = (raw - score_min) / (score_max - score_min)

    selected = []
    remaining = list(candidate_ids)
    while remaining and len(selected) < target_count:
        valid = [
            idx for idx in remaining
            if all(abs(idx - prev) >= min_gap for prev in selected)
        ]
        pool = valid if valid else remaining

        def candidate_score(idx):
            base = base_scores[idx]
            if not selected:
                return base
            desc = descriptors[idx]
            max_sim = max(float(np.clip(desc @ descriptors[prev], -1.0, 1.0)) for prev in selected)
            nearest_gap = min(abs(idx - prev) for prev in selected)
            gap_bonus = min(1.0, float(nearest_gap) / max(float(min_gap * 2), 1.0))
            return base + diversity_weight * (1.0 - max_sim) + 0.15 * gap_bonus

        best_idx = max(pool, key=candidate_score)
        selected.append(int(best_idx))
        remaining = [idx for idx in remaining if idx != best_idx]

    return selected


def _greedy_diverse_order(
    candidate_ids,
    scores,
    descriptors,
    target_count,
    min_gap,
    diversity_weight=1.10,
    similarity_caps=(0.84, 0.88, 0.92, 0.96),
):
    candidate_ids = [int(idx) for idx in candidate_ids]
    if not candidate_ids or target_count <= 0:
        return []

    score_values = np.asarray([scores[idx] for idx in candidate_ids], dtype=np.float32)
    score_min = float(score_values.min()) if score_values.size else 0.0
    score_max = float(score_values.max()) if score_values.size else 0.0
    base_scores = {}
    for idx in candidate_ids:
        raw = float(scores[idx])
        if score_max - score_min < 1e-6:
            base_scores[idx] = 1.0
        else:
            base_scores[idx] = (raw - score_min) / (score_max - score_min)

    selected = []
    remaining = list(candidate_ids)
    active_cap_idx = 0
    caps = [float(v) for v in similarity_caps]

    while remaining and len(selected) < target_count:
        valid = [
            idx for idx in remaining
            if all(abs(idx - prev) >= min_gap for prev in selected)
        ]
        if not valid:
            valid = list(remaining)

        capped = []
        if selected:
            while active_cap_idx < len(caps):
                cap = caps[active_cap_idx]
                capped = []
                for idx in valid:
                    desc = descriptors[idx]
                    max_sim = max(float(np.clip(desc @ descriptors[prev], -1.0, 1.0)) for prev in selected)
                    if max_sim <= cap:
                        capped.append(idx)
                if capped:
                    break
                active_cap_idx += 1
        else:
            capped = list(valid)

        pool = capped if capped else valid

        def candidate_score(idx):
            base = base_scores[idx]
            if not selected:
                return base
            desc = descriptors[idx]
            sims = [float(np.clip(desc @ descriptors[prev], -1.0, 1.0)) for prev in selected]
            max_sim = max(sims)
            mean_sim = float(np.mean(sims))
            nearest_gap = min(abs(idx - prev) for prev in selected)
            gap_bonus = min(1.0, float(nearest_gap) / max(float(min_gap * 2), 1.0))
            return (
                0.80 * base
                + diversity_weight * (1.0 - max_sim)
                + 0.45 * (1.0 - mean_sim)
                + 0.12 * gap_bonus
            )

        best_idx = max(pool, key=candidate_score)
        selected.append(int(best_idx))
        remaining = [idx for idx in remaining if idx != best_idx]

    return selected


def _temporal_front_guard(candidate_ids, front_target, min_gap):
    candidate_ids = [int(idx) for idx in candidate_ids]
    if not candidate_ids or front_target <= 0 or min_gap <= 0:
        return candidate_ids

    front = []
    deferred = []
    for idx in candidate_ids:
        if len(front) < front_target and all(abs(idx - prev) >= min_gap for prev in front):
            front.append(idx)
        else:
            deferred.append(idx)
    return front + deferred


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


def _greedy_spaced_order(scores, min_gap):
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)[::-1]
    chosen = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - prev) >= min_gap for prev in chosen):
            chosen.append(idx)
    for idx in order:
        idx = int(idx)
        if idx not in chosen:
            chosen.append(idx)
    return chosen


def _quality_global_candidates(scores, descriptors, n_keyframes, top_k):
    scores = np.asarray(scores, dtype=np.float32)
    num_frames = int(scores.size)
    if num_frames == 0:
        return []

    quality = _normalize_scores(scores)
    discovery_risk, repair_risk = _temporal_risk_scores(scores, descriptors)
    temporal_signal = _normalize_scores(0.55 * discovery_risk + 0.45 * repair_risk)
    base_gap = max(20, int(num_frames / max(n_keyframes * 3, 1)))
    cluster_count = min(num_frames, max(n_keyframes * 3, n_keyframes + 6))
    candidate_count = min(
        num_frames,
        max(cluster_count * max(2, top_k), n_keyframes * max(10, top_k + 6)),
    )
    quality_floor = max(0.12, float(np.quantile(quality, 0.18))) if quality.size else 0.12

    seed_candidates = _greedy_spaced_order(0.62 * quality + 0.38 * temporal_signal, min_gap=max(12, base_gap // 2))
    centers = []
    for idx in seed_candidates:
        idx = int(idx)
        if quality[idx] >= quality_floor or temporal_signal[idx] >= 0.35:
            centers.append(idx)
        if len(centers) >= min(cluster_count, max(2, n_keyframes // 2)):
            break
    if not centers:
        centers = [int(np.argmax(quality))]
    while len(centers) < cluster_count:
        sims = descriptors @ descriptors[centers].T
        desc_novelty = 1.0 - sims.max(axis=1)
        time_gap = np.full(num_frames, float(num_frames), dtype=np.float32)
        for prev in centers:
            time_gap = np.minimum(time_gap, np.abs(np.arange(num_frames, dtype=np.float32) - float(prev)))
        time_bonus = np.clip(time_gap / max(float(base_gap * 2), 1.0), 0.0, 1.0)
        objective = 0.18 * quality + 0.52 * desc_novelty + 0.12 * time_bonus + 0.18 * temporal_signal
        objective[centers] = -1.0
        remaining_needed = cluster_count - len(centers)
        eligible_mask = quality >= quality_floor
        eligible_remaining = int(np.count_nonzero(eligible_mask & (objective >= 0.0)))
        if eligible_remaining >= remaining_needed:
            objective[~eligible_mask] = -1.0
        else:
            objective[~eligible_mask] -= 0.25
        next_idx = int(np.argmax(objective))
        if next_idx in centers:
            break
        centers.append(next_idx)

    center_desc = descriptors[centers]
    sims = descriptors @ center_desc.T
    assignments = np.argmax(sims, axis=1)
    local_gap = max(12, base_gap // 2)

    cluster_lists = []
    cluster_priority = []
    for cluster_id, center_idx in enumerate(centers):
        members = np.where(assignments == cluster_id)[0].tolist()
        members.sort(
            key=lambda idx: (
                0.58 * quality[idx]
                + 0.32 * float(sims[idx, cluster_id])
                + 0.10 * min(1.0, abs(int(idx) - int(center_idx)) / max(float(base_gap), 1.0))
            ),
            reverse=True,
        )
        picked = []
        if int(center_idx) in members and quality[int(center_idx)] >= quality_floor:
            picked.append(int(center_idx))
        preferred_members = [idx for idx in members if quality[int(idx)] >= quality_floor]
        fallback_members = [idx for idx in members if quality[int(idx)] < quality_floor]
        for idx in preferred_members + fallback_members:
            idx = int(idx)
            if idx in picked:
                continue
            if all(abs(idx - prev) >= local_gap for prev in picked):
                picked.append(idx)
            if len(picked) >= max(1, top_k):
                break
        cluster_lists.append(picked)
        cluster_priority.append(
            0.65 * quality[int(center_idx)]
            + 0.35 * (
                float(np.mean([quality[idx] for idx in picked])) if picked else 0.0
            )
        )

    ordered_clusters = list(np.argsort(np.asarray(cluster_priority, dtype=np.float32))[::-1])
    pool = []
    used = set()

    def add(idx):
        idx = int(idx)
        if idx not in used:
            pool.append(idx)
            used.add(idx)

    max_rounds = max((len(lst) for lst in cluster_lists), default=0)
    for round_idx in range(max_rounds):
        for cluster_id in ordered_clusters:
            lst = cluster_lists[cluster_id]
            if round_idx < len(lst):
                add(lst[round_idx])
            if len(pool) >= candidate_count:
                break
        if len(pool) >= candidate_count:
            break

    risk_order = _greedy_spaced_order(temporal_signal, min_gap=max(10, base_gap // 2))
    for idx in risk_order[: max(n_keyframes * 3, cluster_count)]:
        if quality[int(idx)] >= quality_floor or temporal_signal[int(idx)] >= 0.45:
            add(idx)
        if len(pool) >= candidate_count:
            break

    global_order = _greedy_spaced_order(scores, min_gap=base_gap)
    for idx in global_order:
        add(idx)
        if len(pool) >= candidate_count:
            break

    ordered_pool = _greedy_diverse_order(
        pool,
        scores,
        descriptors,
        target_count=min(len(pool), candidate_count),
        min_gap=max(16, base_gap // 2),
        diversity_weight=1.10,
        similarity_caps=(0.82, 0.86, 0.90, 0.94),
    )
    front_target = min(
        len(ordered_pool),
        max(
            n_keyframes + 2 * max(1, top_k),
            int(round(1.5 * n_keyframes)),
        ),
    )
    front_gap = max(base_gap, 18)
    ordered_pool = _temporal_front_guard(
        ordered_pool,
        front_target=front_target,
        min_gap=front_gap,
    )
    return ordered_pool


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

    if strategy == "quality_global":
        if frames is None:
            raise ValueError("quality_global requires loaded frames")
        scores = _quality_scores(frames)
        descriptors = _frame_descriptors(frames)
        return [_quality_global_candidates(scores, descriptors, n_keyframes, top_k)]

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

    if strategy == "quality_global":
        if frames is None:
            raise ValueError("quality_global requires loaded frames")
        scores = _quality_scores(frames)
        descriptors = _frame_descriptors(frames)
        ordered = _quality_global_candidates(scores, descriptors, n_keyframes, top_k=2)
        min_gap = max(18, int(num_frames / max(n_keyframes * 2, 1)))
        selected = _mmr_select_candidates(
            ordered,
            scores,
            descriptors,
            target_count=n_keyframes,
            min_gap=min_gap,
            diversity_weight=0.90,
        )
        return sorted(selected)

    return list(range(num_frames))
