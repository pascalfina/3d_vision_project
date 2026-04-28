"""
Image-only keyframe refinement for SAM2 preview selection.

The module keeps three concerns separate:
1. candidate description from preview masks
2. set optimization over described candidates
3. diagnostics and regression-friendly summaries
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM2_REPO = REPO_ROOT / "dependencies" / "sam2"
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))


def _parse_int(value, default):
    if value is None:
        return int(default)
    return int(value)


def _parse_float(value, default):
    if value is None:
        return float(default)
    return float(value)


def _normalize_scores(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-6:
        return np.ones_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def frame_preview_quality(frame):
    import cv2

    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side > 256:
        scale = 256.0 / float(max_side)
        frame = cv2.resize(
            frame,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    lap_var = float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()))
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    clip_frac = float(((gray < 12) | (gray > 245)).mean())
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
    edge_balance = (
        float(1.0 - (cell_edges.std() / max(cell_edges.mean(), 1e-4)))
        if cell_edges.size
        else 0.0
    )
    edge_spread = max(0.0, 0.55 * active_cells + 0.45 * edge_balance)

    margin_y = max(1, int(round(edges.shape[0] * 0.12)))
    margin_x = max(1, int(round(edges.shape[1] * 0.12)))
    border_mask = np.zeros_like(edges, dtype=np.bool_)
    border_mask[:margin_y, :] = True
    border_mask[-margin_y:, :] = True
    border_mask[:, :margin_x] = True
    border_mask[:, -margin_x:] = True
    border_edge_density = float(edges[border_mask].mean() / 255.0)
    inner_edge_density = float(edges[~border_mask].mean() / 255.0) if (~border_mask).any() else 0.0
    return (
        1.25 * lap_var
        + 0.80 * edge_density
        + 0.95 * edge_spread
        + 0.55 * contrast
        - 0.90 * max(0.0, border_edge_density - inner_edge_density)
        - 0.45 * clip_frac
    )


def rotate_norm_xy(center_x, center_y, rot_k):
    if rot_k == 0:
        return center_x, center_y
    if rot_k == 1:
        return center_y, 1.0 - center_x
    if rot_k == 2:
        return 1.0 - center_x, 1.0 - center_y
    return 1.0 - center_y, center_x


def rotate_margin_touches(left, top, right, bottom, rot_k):
    if rot_k == 0:
        return left, top, right, bottom
    if rot_k == 1:
        return top, right, bottom, left
    if rot_k == 2:
        return right, bottom, left, top
    return bottom, left, top, right


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _array_entropy(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    prob = values / values.sum()
    return float(
        -(prob * np.log(prob + 1e-8)).sum() / np.log(max(len(prob), 2))
    )


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


def describe_preview_masks(frame, masks, preview_top_k, descriptor_grid):
    import cv2

    usable = masks[:preview_top_k]
    mask_count = len(usable)
    total_area = float(sum(m["area"] for m in usable))
    largest_area = 0.0
    tiny_masks = 0
    union_mask = None
    pred_ious = []
    stability_scores = []
    frame_area = float(frame.shape[0] * frame.shape[1])
    rotated_thumbs = []
    frame_quality_by_rot = []
    for rot_k in range(4):
        rotated = np.ascontiguousarray(np.rot90(frame, rot_k))
        rotated_thumbs.append(rotated)
        frame_quality_by_rot.append(frame_preview_quality(rotated))

    rot_stats = [
        {
            "useful_masks": 0,
            "weighted_utility": 0.0,
            "border_dominated": 0,
            "bottom_heavy_masks": 0,
            "grid": np.zeros((3, 3), dtype=np.bool_),
            "useful_hist": np.zeros((3, 3), dtype=np.float32),
            "texture_grid": np.zeros((3, 3), dtype=np.bool_),
            "texture_areas": [],
            "useful_area_fracs": [],
            "useful_area_mass": 0.0,
            "bottom_useful_mass": 0.0,
            "object_scores": [],
            "useful_union": None,
        }
        for _ in range(4)
    ]
    area_fracs = []

    for mask in usable:
        seg = mask.get("segmentation")
        area = float(mask.get("area", 0.0))
        if seg is None or area <= 0.0:
            continue
        seg = seg.astype(bool, copy=False)
        pred_iou = float(mask.get("predicted_iou", 0.0))
        stability = float(mask.get("stability_score", pred_iou))
        pred_ious.append(pred_iou)
        stability_scores.append(stability)
        if union_mask is None:
            union_mask = np.zeros_like(seg, dtype=np.bool_)
        union_mask |= seg
        h = int(seg.shape[0])
        w = int(seg.shape[1])
        bbox = mask.get("bbox") or [0, 0, w, h]
        x, y, bw, bh = [float(v) for v in bbox]
        largest_area = max(largest_area, area)
        area_frac = area / max(frame_area, 1.0)
        area_fracs.append(area_frac)
        if area_frac < 0.0025:
            tiny_masks += 1

        bbox_fill = area / max(float(bw * bh), 1.0)
        shape_ratio = min(float(bw), float(bh)) / max(max(float(bw), float(bh)), 1.0)
        center_y_raw = (y + 0.5 * bh) / max(float(h), 1.0)
        center_x_raw = (x + 0.5 * bw) / max(float(w), 1.0)

        touch_margin = 8.0
        left = x <= touch_margin
        top = y <= touch_margin
        right = (x + bw) >= (w - touch_margin)
        bottom = (y + bh) >= (h - touch_margin)

        for rot_k, stats in enumerate(rot_stats):
            center_x, center_y = rotate_norm_xy(center_x_raw, center_y_raw, rot_k)
            grid_y = min(2, max(0, int(center_y * 3.0)))
            grid_x = min(2, max(0, int(center_x * 3.0)))
            center_dist = float(np.hypot(center_x - 0.5, center_y - 0.5) / 0.70710678)
            vertical_pref = max(0.0, 1.0 - center_y)
            central_pref = max(0.0, 1.0 - center_dist)
            position_weight = max(0.25, 0.55 * vertical_pref + 0.45 * central_pref)

            l, t, r, b = rotate_margin_touches(left, top, right, bottom, rot_k)
            touches = int(l) + int(t) + int(r) + int(b)
            if touches >= 2 and area >= 0.12 * float(h * w):
                stats["border_dominated"] += 1

            if area_frac < 0.0015:
                size_score = 0.0
            elif area_frac < 0.004:
                size_score = 0.30
            elif area_frac < 0.02:
                size_score = 1.00
            elif area_frac < 0.12:
                size_score = 0.82
            elif area_frac < 0.28:
                size_score = 0.55
            else:
                size_score = 0.22

            fill_score = _clip01((bbox_fill - 0.18) / 0.62)
            shape_score = max(0.30, min(1.0, shape_ratio))
            quality_score = 0.55 * pred_iou + 0.45 * stability
            touch_score = 1.0 if touches == 0 else (0.86 if touches == 1 else 0.65)
            bottom_score = 0.82 if center_y >= 0.82 else 1.0

            object_score = (
                size_score
                * (0.35 + 0.65 * fill_score)
                * (0.35 + 0.65 * quality_score)
                * (0.45 + 0.55 * shape_score)
                * (0.35 + 0.65 * position_weight)
                * touch_score
                * bottom_score
            )

            useful_union = stats["useful_union"]
            if useful_union is None:
                novel_frac = 1.0
                novel_area_frac = area_frac
            else:
                novel_pixels = int(np.count_nonzero(seg & ~useful_union))
                novel_frac = float(novel_pixels / max(area, 1.0))
                novel_area_frac = float(novel_pixels / max(frame_area, 1.0))
            novelty_weight = _clip01((novel_frac - 0.10) / 0.75)
            effective_object_score = object_score * (0.25 + 0.75 * novelty_weight)

            is_useful = (
                object_score >= 0.22
                and novelty_weight >= 0.18
                and novel_area_frac >= 0.0012
            )
            is_texture = (
                effective_object_score < 0.16
                and touches <= 1
                and 0.0015 <= area_frac <= 0.025
                and bbox_fill >= 0.22
            )

            if is_texture:
                stats["texture_grid"][grid_y, grid_x] = True
                stats["texture_areas"].append(area_frac)

            if is_useful:
                useful_mass = area_frac * max(float(effective_object_score), 0.2)
                stats["useful_masks"] += 1
                stats["grid"][grid_y, grid_x] = True
                stats["useful_hist"][grid_y, grid_x] += float(effective_object_score)
                stats["useful_area_fracs"].append(area_frac)
                stats["useful_area_mass"] += useful_mass
                if center_y >= 0.78 and area_frac >= 0.003:
                    stats["bottom_heavy_masks"] += 1
                    stats["bottom_useful_mass"] += useful_mass
                stats["object_scores"].append(float(effective_object_score))
                stats["weighted_utility"] += float(effective_object_score)
                if useful_union is None:
                    stats["useful_union"] = seg.copy()
                else:
                    useful_union |= seg
            elif touches <= 1 and 0.02 < area_frac <= 0.35 and object_score >= 0.12:
                stats["weighted_utility"] += 0.12 * float(effective_object_score)
            elif center_y >= 0.78 and area_frac >= 0.003 and object_score < 0.18:
                stats["bottom_heavy_masks"] += 1

    union_area = float(union_mask.sum()) if union_mask is not None else 0.0
    dominance_ratio = largest_area / max(total_area, 1.0)
    tiny_mask_ratio = float(tiny_masks / max(mask_count, 1))
    mean_pred_iou = float(np.mean(pred_ious)) if pred_ious else 0.0
    mean_stability = float(np.mean(stability_scores)) if stability_scores else mean_pred_iou
    union_ratio = float(union_area / max(frame_area, 1.0))
    area_fracs = np.asarray(area_fracs, dtype=np.float32)
    scale_hist = np.histogram(
        area_fracs,
        bins=np.asarray([0.0, 0.002, 0.01, 0.04, 0.18, 1.0], dtype=np.float32),
    )[0].astype(np.float32)
    scale_diversity = int((scale_hist > 0).sum())
    scale_entropy = _array_entropy(scale_hist)

    rotation_summaries = []
    for rot_k, stats in enumerate(rot_stats):
        useful_masks = int(stats["useful_masks"])
        rot_grid_cells = int(stats["grid"].sum())
        object_scores = sorted(stats["object_scores"], reverse=True)
        top_object_sum = float(sum(object_scores[:10]))
        texture_count = len(stats["texture_areas"])
        texture_grid_cells = int(stats["texture_grid"].sum())
        if texture_count > 1:
            texture_areas = np.asarray(stats["texture_areas"], dtype=np.float32)
            texture_area_cv = float(texture_areas.std() / max(texture_areas.mean(), 1e-6))
        else:
            texture_area_cv = 1.0
        texture_penalty = float(
            _clip01((texture_count - 6) / 14.0)
            * (texture_grid_cells / 9.0)
            * _clip01((0.85 - texture_area_cv) / 0.85)
        )
        noise_penalty = _clip01(((mask_count - useful_masks) / max(mask_count, 1) - 0.45) / 0.40)
        dominance_penalty = _clip01((dominance_ratio - 0.55) / 0.30)
        border_penalty = _clip01(stats["border_dominated"] / max(useful_masks, 1))
        bottom_penalty = (
            _clip01(stats["bottom_useful_mass"] / max(stats["useful_area_mass"], 1e-6))
            if stats["useful_area_mass"] > 0.0
            else 0.0
        )
        useful_count_score = min(1.0, useful_masks / 12.0)
        coverage_score = rot_grid_cells / 9.0
        useful_hist = stats["useful_hist"].reshape(-1)
        useful_hist_sum = float(useful_hist.sum())
        useful_hist_prob = useful_hist / useful_hist_sum if useful_hist_sum > 1e-6 else useful_hist
        coverage_entropy = _array_entropy(useful_hist)
        useful_area_fracs = np.asarray(stats["useful_area_fracs"], dtype=np.float32)
        useful_union = stats["useful_union"]
        useful_union_ratio = (
            float(useful_union.sum() / max(frame_area, 1.0))
            if useful_union is not None
            else 0.0
        )
        overlap_penalty = (
            _clip01(1.0 - useful_union_ratio / max(float(useful_area_fracs.sum()), 1e-6))
            if useful_area_fracs.size > 0
            else 0.0
        )
        useful_scale_hist = np.histogram(
            useful_area_fracs,
            bins=np.asarray([0.0, 0.002, 0.01, 0.04, 0.18, 1.0], dtype=np.float32),
        )[0].astype(np.float32) if useful_area_fracs.size > 0 else np.zeros(5, dtype=np.float32)
        useful_scale_entropy = _array_entropy(useful_scale_hist)
        concentration_penalty = (
            float(_clip01(useful_hist.max() / max(useful_hist_sum, 1e-6) - 0.42) / 0.58)
            if useful_hist_sum > 0.0
            else 0.0
        )
        object_strength = min(1.0, top_object_sum / 4.0)
        union_mid_score = max(0.0, 1.0 - abs(union_ratio - 0.42) / 0.42)
        useful_union_mid_score = max(0.0, 1.0 - abs(useful_union_ratio - 0.36) / 0.36)
        quality_norm = _clip01((frame_quality_by_rot[rot_k] - 2.5) / 5.0)

        context_score = (
            2.10 * useful_count_score
            + 2.25 * coverage_score
            + 2.00 * coverage_entropy
            + 1.30 * object_strength
            + 0.80 * scale_entropy
            + 0.90 * useful_scale_entropy
            + 0.75 * quality_norm
            + 0.35 * union_mid_score
            + 1.05 * useful_union_mid_score
            + 0.15 * float(stats["weighted_utility"])
            - 1.45 * texture_penalty
            - 1.45 * overlap_penalty
            - 0.72 * dominance_penalty
            - 1.00 * noise_penalty
            - 0.48 * bottom_penalty
            - 0.85 * concentration_penalty
            - 0.22 * border_penalty
            - 0.22 * tiny_mask_ratio
        )

        rotated_union = np.ascontiguousarray(np.rot90(union_mask, rot_k)) if union_mask is not None else None
        rotated_useful_union = np.ascontiguousarray(np.rot90(useful_union, rot_k)) if useful_union is not None else None
        occ_grid = np.zeros((descriptor_grid, descriptor_grid), dtype=np.float32)
        useful_occ_grid = np.zeros((descriptor_grid, descriptor_grid), dtype=np.float32)
        if rotated_union is not None and rotated_union.size > 0:
            gh = np.linspace(0, rotated_union.shape[0], descriptor_grid + 1, dtype=int)
            gw = np.linspace(0, rotated_union.shape[1], descriptor_grid + 1, dtype=int)
            for gy in range(descriptor_grid):
                for gx in range(descriptor_grid):
                    cell = rotated_union[gh[gy] : gh[gy + 1], gw[gx] : gw[gx + 1]]
                    if cell.size > 0:
                        occ_grid[gy, gx] = float(cell.mean())
                    if rotated_useful_union is not None:
                        useful_cell = rotated_useful_union[gh[gy] : gh[gy + 1], gw[gx] : gw[gx + 1]]
                        if useful_cell.size > 0:
                            useful_occ_grid[gy, gx] = float(useful_cell.mean())

        thumb = cv2.resize(rotated_thumbs[rot_k], (12, 12), interpolation=cv2.INTER_AREA)
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        thumb_gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        thumb_edges = cv2.Canny((thumb_gray * 255.0).astype(np.uint8), 40, 120).astype(np.float32) / 255.0
        thumb_small = cv2.resize(thumb_gray, (6, 6), interpolation=cv2.INTER_AREA).astype(np.float32)
        occ_row = occ_grid.mean(axis=1).astype(np.float32)
        occ_col = occ_grid.mean(axis=0).astype(np.float32)
        useful_row = useful_occ_grid.mean(axis=1).astype(np.float32)
        useful_col = useful_occ_grid.mean(axis=0).astype(np.float32)
        occ_row_grad = np.diff(occ_row).astype(np.float32) if occ_row.size > 1 else np.zeros(0, dtype=np.float32)
        occ_col_grad = np.diff(occ_col).astype(np.float32) if occ_col.size > 1 else np.zeros(0, dtype=np.float32)
        useful_row_grad = np.diff(useful_row).astype(np.float32) if useful_row.size > 1 else np.zeros(0, dtype=np.float32)
        useful_col_grad = np.diff(useful_col).astype(np.float32) if useful_col.size > 1 else np.zeros(0, dtype=np.float32)
        pooled_useful_occ = _pool_grid_mean(useful_occ_grid, 3, 3)
        pooled_occ = _pool_grid_mean(occ_grid, 3, 3)
        useful_top = float(useful_occ_grid[: descriptor_grid // 2].mean()) if useful_occ_grid.size else 0.0
        useful_bottom = float(useful_occ_grid[descriptor_grid // 2 :].mean()) if useful_occ_grid.size else 0.0
        useful_left = float(useful_occ_grid[:, : descriptor_grid // 2].mean()) if useful_occ_grid.size else 0.0
        useful_right = float(useful_occ_grid[:, descriptor_grid // 2 :].mean()) if useful_occ_grid.size else 0.0
        occ_top = float(occ_grid[: descriptor_grid // 2].mean()) if occ_grid.size else 0.0
        occ_bottom = float(occ_grid[descriptor_grid // 2 :].mean()) if occ_grid.size else 0.0
        occ_left = float(occ_grid[:, : descriptor_grid // 2].mean()) if occ_grid.size else 0.0
        occ_right = float(occ_grid[:, descriptor_grid // 2 :].mean()) if occ_grid.size else 0.0
        if rotated_useful_union is not None and rotated_useful_union.any():
            ys, xs = np.nonzero(rotated_useful_union)
            useful_center = np.array(
                [
                    float(xs.mean() / max(rotated_useful_union.shape[1] - 1, 1)),
                    float(ys.mean() / max(rotated_useful_union.shape[0] - 1, 1)),
                ],
                dtype=np.float32,
            )
        else:
            useful_center = np.array([0.5, 0.5], dtype=np.float32)
        if rotated_union is not None and rotated_union.any():
            ys_u, xs_u = np.nonzero(rotated_union)
            union_center = np.array(
                [
                    float(xs_u.mean() / max(rotated_union.shape[1] - 1, 1)),
                    float(ys_u.mean() / max(rotated_union.shape[0] - 1, 1)),
                ],
                dtype=np.float32,
            )
        else:
            union_center = np.array([0.5, 0.5], dtype=np.float32)
        descriptor = np.concatenate(
            [
                (2.4 * useful_occ_grid).reshape(-1),
                (1.2 * occ_grid).reshape(-1),
                (1.1 * useful_hist_prob).reshape(-1),
                (0.35 * thumb_rgb).reshape(-1),
                np.array(
                    [
                        useful_masks / max(preview_top_k, 1),
                        rot_grid_cells / 9.0,
                        union_ratio,
                        useful_union_ratio,
                        min(1.0, dominance_ratio),
                        min(1.0, frame_quality_by_rot[rot_k] / 10.0),
                        min(1.0, mean_pred_iou),
                        min(1.0, mean_stability),
                        tiny_mask_ratio,
                        min(1.0, stats["bottom_heavy_masks"] / max(mask_count, 1)),
                        min(1.0, texture_grid_cells / 9.0),
                        texture_penalty,
                        overlap_penalty,
                        concentration_penalty,
                        scale_entropy,
                        coverage_entropy,
                        useful_scale_entropy,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        norm = float(np.linalg.norm(descriptor))
        if norm > 1e-6:
            descriptor /= norm

        family_descriptor = np.concatenate(
            [
                (3.20 * useful_occ_grid).reshape(-1),
                (1.90 * occ_grid).reshape(-1),
                (1.80 * pooled_useful_occ).reshape(-1),
                (1.10 * pooled_occ).reshape(-1),
                (1.35 * useful_row).reshape(-1),
                (1.35 * useful_col).reshape(-1),
                (1.15 * useful_row_grad).reshape(-1),
                (1.15 * useful_col_grad).reshape(-1),
                (0.90 * occ_row).reshape(-1),
                (0.90 * occ_col).reshape(-1),
                (0.70 * occ_row_grad).reshape(-1),
                (0.70 * occ_col_grad).reshape(-1),
                (0.95 * useful_hist_prob).reshape(-1),
                (0.60 * thumb_edges).reshape(-1),
                (0.35 * thumb_small).reshape(-1),
                np.array(
                    [
                        useful_center[0],
                        useful_center[1],
                        union_center[0],
                        union_center[1],
                        useful_top,
                        useful_bottom,
                        useful_left,
                        useful_right,
                        occ_top,
                        occ_bottom,
                        occ_left,
                        occ_right,
                        union_ratio,
                        useful_union_ratio,
                        min(1.0, dominance_ratio),
                        float(rot_grid_cells) / 9.0,
                        min(1.0, frame_quality_by_rot[rot_k] / 10.0),
                        float(texture_penalty),
                        float(overlap_penalty),
                        float(concentration_penalty),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        family_norm = float(np.linalg.norm(family_descriptor))
        if family_norm > 1e-6:
            family_descriptor /= family_norm

        rotation_summaries.append(
            {
                "mask_count": mask_count,
                "useful_masks": useful_masks,
                "utility_score": float(1000.0 * context_score),
                "context_score": float(context_score),
                "frame_quality": float(frame_quality_by_rot[rot_k]),
                "grid_cells": rot_grid_cells,
                "union_ratio": union_ratio,
                "useful_union_ratio": useful_union_ratio,
                "dominance_ratio": float(dominance_ratio),
                "tiny_mask_ratio": tiny_mask_ratio,
                "mean_pred_iou": mean_pred_iou,
                "mean_stability": mean_stability,
                "bottom_heavy_masks": int(stats["bottom_heavy_masks"]),
                "texture_grid_cells": texture_grid_cells,
                "texture_penalty": float(texture_penalty),
                "overlap_penalty": float(overlap_penalty),
                "concentration_penalty": float(concentration_penalty),
                "scale_entropy": float(scale_entropy),
                "useful_scale_entropy": float(useful_scale_entropy),
                "coverage_entropy": float(coverage_entropy),
                "top_object_sum": top_object_sum,
                "rotation": int(rot_k),
                "descriptor": descriptor,
                "family_descriptor": family_descriptor,
                "grid_vec": stats["grid"].astype(np.float32).reshape(-1),
                "coverage_vec": useful_occ_grid.reshape(-1).astype(np.float32),
                "focus_vec": useful_hist_prob.astype(np.float32),
            }
        )

    return {
        "best_rotation": int(np.argmax([entry["utility_score"] for entry in rotation_summaries])),
        "rotation_stats": rotation_summaries,
    }


def _pair_similarity(entry_a, entry_b):
    return float(np.clip(entry_a["stats"]["descriptor"] @ entry_b["stats"]["descriptor"], -1.0, 1.0))


def _family_descriptor(entry):
    stats = entry["stats"]
    return stats.get("family_descriptor", stats["descriptor"])


def _family_similarity(entry_a, entry_b):
    return float(np.clip(_family_descriptor(entry_a) @ _family_descriptor(entry_b), -1.0, 1.0))


def _selection_pair_similarity(entry_a, entry_b):
    desc_sim = _pair_similarity(entry_a, entry_b)
    family_sim = _family_similarity(entry_a, entry_b)
    family_a = int(entry_a.get("view_family", -1))
    family_b = int(entry_b.get("view_family", -1))
    if family_a >= 0 and family_a == family_b:
        return max(desc_sim, family_sim)
    return float(np.clip(0.30 * desc_sim + 0.70 * family_sim, -1.0, 1.0))


def _entries_respect_gap(entries, min_frame_gap):
    idxs = sorted(int(entry["idx"]) for entry in entries)
    return all((b - a) >= min_frame_gap for a, b in zip(idxs, idxs[1:]))


def _entries_respect_similarity(entries, similarity_cap):
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if _selection_pair_similarity(entries[i], entries[j]) > similarity_cap:
                return False
    return True


def _assign_view_families(all_entries, family_similarity_threshold=0.88):
    family_similarity_threshold = float(family_similarity_threshold)
    ordered = sorted(
        all_entries,
        key=lambda x: (
            x.get("overview_score", 0.0),
            x.get("support_score", 0.0),
            x["stats"]["context_score"],
            x["stats"]["useful_union_ratio"],
        ),
        reverse=True,
    )

    families = []
    for entry in ordered:
        descriptor = _family_descriptor(entry)
        best_family = None
        best_score = -1.0
        for family_idx, family in enumerate(families):
            member_desc = np.stack(family["member_descriptors"], axis=0)
            member_sims = np.clip(member_desc @ descriptor, -1.0, 1.0)
            max_member_sim = float(member_sims.max()) if member_sims.size else -1.0
            topk = min(3, int(member_sims.size))
            topk_mean = float(np.mean(np.partition(member_sims, -topk)[-topk:])) if topk > 0 else -1.0
            anchor_sim = float(np.clip(descriptor @ family["anchor"], -1.0, 1.0))
            centroid_sim = float(np.clip(descriptor @ family["centroid"], -1.0, 1.0))
            if (
                max_member_sim < family_similarity_threshold
                or topk_mean < family_similarity_threshold - 0.02
                or anchor_sim < family_similarity_threshold - 0.04
                or centroid_sim < family_similarity_threshold - 0.05
            ):
                continue
            score = (
                0.42 * max_member_sim
                + 0.28 * topk_mean
                + 0.20 * anchor_sim
                + 0.10 * centroid_sim
            )
            if score > best_score:
                best_score = score
                best_family = family_idx
        if best_family is None:
            families.append(
                {
                    "anchor": descriptor.copy(),
                    "centroid": descriptor.copy(),
                    "member_descriptors": [descriptor.copy()],
                    "members": [entry],
                }
            )
        else:
            families[best_family]["members"].append(entry)
            families[best_family]["member_descriptors"].append(descriptor.copy())
            centroid = np.mean(
                np.stack(families[best_family]["member_descriptors"], axis=0),
                axis=0,
            ).astype(np.float32)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-6:
                centroid /= norm
            families[best_family]["centroid"] = centroid

    for family_id, family in enumerate(families):
        for member in family["members"]:
            member["view_family"] = int(family_id)
            member["view_family_size"] = int(len(family["members"]))
            member["view_family_similarity"] = float(
                np.clip(_family_descriptor(member) @ family["centroid"], -1.0, 1.0)
            )
    return families


def _compute_temporal_role_signals(all_entries):
    if not all_entries:
        return {}

    order = sorted(range(len(all_entries)), key=lambda idx: int(all_entries[idx]["idx"]))
    num_entries = len(order)
    appearance_jump = np.zeros(num_entries, dtype=np.float32)
    transition_jump = np.zeros(num_entries, dtype=np.float32)
    discovery_novelty = np.zeros(num_entries, dtype=np.float32)
    focus_novelty = np.zeros(num_entries, dtype=np.float32)
    grid_novelty = np.zeros(num_entries, dtype=np.float32)
    mask_rise = np.zeros(num_entries, dtype=np.float32)
    mask_drop = np.zeros(num_entries, dtype=np.float32)
    coverage_rise = np.zeros(num_entries, dtype=np.float32)
    coverage_drop = np.zeros(num_entries, dtype=np.float32)
    context_rise = np.zeros(num_entries, dtype=np.float32)
    context_drop = np.zeros(num_entries, dtype=np.float32)
    quality_gain = np.zeros(num_entries, dtype=np.float32)
    quality_drop = np.zeros(num_entries, dtype=np.float32)

    def _neighbor_positions(pos):
        neigh = []
        for delta in (-2, -1, 1, 2):
            npos = pos + delta
            if 0 <= npos < num_entries:
                neigh.append(npos)
        return neigh

    for pos, entry_idx in enumerate(order):
        entry = all_entries[entry_idx]
        neighbors_pos = _neighbor_positions(pos)
        if not neighbors_pos:
            continue
        neighbors = [all_entries[order[npos]] for npos in neighbors_pos]

        sims = [_pair_similarity(entry, other) for other in neighbors]
        appearance_jump[pos] = max(0.0, 1.0 - float(min(sims)))

        if 0 < pos < num_entries - 1:
            prev_entry = all_entries[order[pos - 1]]
            next_entry = all_entries[order[pos + 1]]
            transition_jump[pos] = max(0.0, 1.0 - _pair_similarity(prev_entry, next_entry))

        cur_cov = np.asarray(entry["stats"]["coverage_vec"], dtype=np.float32)
        cur_focus = np.asarray(entry["stats"]["focus_vec"], dtype=np.float32)
        cur_masks = float(entry["stats"]["useful_masks"])
        cur_cov_ratio = float(entry["stats"]["useful_union_ratio"])
        cur_ctx = float(entry["stats"]["context_score"])
        cur_quality = float(entry["stats"].get("frame_quality", 0.0))

        neigh_cov = np.max(
            np.stack([np.asarray(other["stats"]["coverage_vec"], dtype=np.float32) for other in neighbors], axis=0),
            axis=0,
        )
        neigh_focus = np.max(
            np.stack([np.asarray(other["stats"]["focus_vec"], dtype=np.float32) for other in neighbors], axis=0),
            axis=0,
        )
        neigh_masks = float(np.mean([float(other["stats"]["useful_masks"]) for other in neighbors]))
        neigh_cov_ratio = float(np.mean([float(other["stats"]["useful_union_ratio"]) for other in neighbors]))
        neigh_ctx = float(np.mean([float(other["stats"]["context_score"]) for other in neighbors]))
        neigh_quality = float(np.mean([float(other["stats"].get("frame_quality", 0.0)) for other in neighbors]))

        discovery_novelty[pos] = float(np.clip(cur_cov - neigh_cov, 0.0, 1.0).mean())
        focus_novelty[pos] = float(np.clip(cur_focus - neigh_focus, 0.0, 1.0).mean())
        grid_novelty[pos] = float(
            ((cur_cov > 0.08) & (neigh_cov < 0.04)).mean()
        )

        mask_rise[pos] = max(0.0, cur_masks - neigh_masks) / max(max(cur_masks, neigh_masks), 1.0)
        mask_drop[pos] = max(0.0, neigh_masks - cur_masks) / max(neigh_masks, 1.0)
        coverage_rise[pos] = max(0.0, cur_cov_ratio - neigh_cov_ratio)
        coverage_drop[pos] = max(0.0, neigh_cov_ratio - cur_cov_ratio)
        context_rise[pos] = max(0.0, cur_ctx - neigh_ctx) / max(abs(neigh_ctx), 1.0)
        context_drop[pos] = max(0.0, neigh_ctx - cur_ctx) / max(abs(neigh_ctx), 1.0)
        quality_gain[pos] = max(0.0, cur_quality - neigh_quality) / 10.0
        quality_drop[pos] = max(0.0, neigh_quality - cur_quality) / 10.0

    discovery_raw = (
        1.35 * discovery_novelty
        + 0.90 * focus_novelty
        + 0.75 * grid_novelty
        + 0.75 * mask_rise
        + 0.55 * coverage_rise
        + 0.35 * context_rise
        + 0.40 * appearance_jump
        + 0.20 * quality_gain
    )
    repair_raw = (
        1.25 * appearance_jump
        + 0.90 * transition_jump
        + 0.85 * coverage_drop
        + 0.80 * mask_drop
        + 0.55 * context_drop
        + 0.35 * quality_drop
        + 0.25 * np.maximum(coverage_rise, discovery_novelty)
    )

    discovery_score = _normalize_scores(discovery_raw)
    repair_score = _normalize_scores(repair_raw)

    signals = {}
    for pos, entry_idx in enumerate(order):
        signals[entry_idx] = {
            "appearance_jump": float(appearance_jump[pos]),
            "transition_jump": float(transition_jump[pos]),
            "discovery_novelty": float(discovery_novelty[pos]),
            "focus_novelty": float(focus_novelty[pos]),
            "grid_novelty": float(grid_novelty[pos]),
            "mask_rise": float(mask_rise[pos]),
            "mask_drop": float(mask_drop[pos]),
            "coverage_rise": float(coverage_rise[pos]),
            "coverage_drop": float(coverage_drop[pos]),
            "context_rise": float(context_rise[pos]),
            "context_drop": float(context_drop[pos]),
            "quality_gain": float(quality_gain[pos]),
            "quality_drop": float(quality_drop[pos]),
            "discovery_score": float(discovery_score[pos]),
            "repair_score": float(repair_score[pos]),
        }
    return signals


def _prepare_entry_roles(all_entries):
    context_scores = np.asarray([entry["stats"]["context_score"] for entry in all_entries], dtype=np.float32)
    cov_scores = np.asarray([entry["stats"]["useful_union_ratio"] for entry in all_entries], dtype=np.float32)
    grid_scores = np.asarray([entry["stats"]["grid_cells"] for entry in all_entries], dtype=np.float32)
    quality_scores = np.asarray([entry["stats"]["frame_quality"] for entry in all_entries], dtype=np.float32)
    penalty_scores = np.asarray(
        [
            entry["stats"]["texture_penalty"]
            + entry["stats"]["overlap_penalty"]
            + 0.8 * entry["stats"]["concentration_penalty"]
            + 0.4 * max(0.0, 0.18 - entry["stats"]["useful_union_ratio"])
            for entry in all_entries
        ],
        dtype=np.float32,
    )

    ctx_norm = _normalize_scores(context_scores)
    cov_norm = _normalize_scores(cov_scores)
    grid_norm = _normalize_scores(grid_scores)
    qual_norm = _normalize_scores(quality_scores)
    pen_norm = _normalize_scores(penalty_scores)
    temporal_signals = _compute_temporal_role_signals(all_entries)

    ctx_thr = float(np.quantile(context_scores, 0.30)) if context_scores.size else 0.0
    if ctx_thr > 5.0:
        ctx_thr = max(8.6, ctx_thr)
    cov_thr = max(0.15, float(np.quantile(cov_scores, 0.30)) - 0.01) if cov_scores.size else 0.15
    grid_thr = max(5.0, float(np.quantile(grid_scores, 0.30))) if grid_scores.size else 5.0
    overview_raw = (
        1.60 * ctx_norm
        + 1.45 * cov_norm
        + 1.00 * grid_norm
        + 0.55 * qual_norm
        - 0.85 * pen_norm
    )
    discovery_signal = np.asarray(
        [temporal_signals[idx]["discovery_score"] for idx in range(len(all_entries))],
        dtype=np.float32,
    )
    repair_signal = np.asarray(
        [temporal_signals[idx]["repair_score"] for idx in range(len(all_entries))],
        dtype=np.float32,
    )
    appearance_jump = np.asarray(
        [temporal_signals[idx]["appearance_jump"] for idx in range(len(all_entries))],
        dtype=np.float32,
    )
    discovery_novelty = np.asarray(
        [temporal_signals[idx]["discovery_novelty"] for idx in range(len(all_entries))],
        dtype=np.float32,
    )
    discovery_raw = (
        1.10 * discovery_signal
        + 0.70 * ctx_norm
        + 0.55 * cov_norm
        + 0.45 * grid_norm
        + 0.35 * qual_norm
        + 0.35 * appearance_jump
        + 0.45 * discovery_novelty
        - 0.35 * pen_norm
    )
    repair_raw = (
        1.20 * repair_signal
        + 0.45 * appearance_jump
        + 0.35 * cov_norm
        + 0.25 * grid_norm
        + 0.20 * qual_norm
        - 0.20 * pen_norm
    )
    discovery_norm = _normalize_scores(discovery_raw)
    repair_norm = _normalize_scores(repair_raw)
    support_raw = (
        0.45 * overview_raw
        + 0.75 * discovery_norm
        + 0.85 * repair_norm
        + 0.30 * qual_norm
        - 0.20 * pen_norm
    )
    overview_norm = _normalize_scores(overview_raw)
    support_norm = _normalize_scores(support_raw)
    quality_thr = float(np.quantile(quality_scores, 0.18)) if quality_scores.size else 0.0
    if quality_thr > 5.0:
        quality_thr = max(7.2, quality_thr)
    discovery_thr = max(0.55, float(np.quantile(discovery_norm, 0.65))) if discovery_norm.size else 0.55
    repair_thr = max(0.55, float(np.quantile(repair_norm, 0.65))) if repair_norm.size else 0.55

    for idx, (entry, overview_score, support_score, discovery_score, repair_score) in enumerate(
        zip(all_entries, overview_norm, support_norm, discovery_norm, repair_norm)
    ):
        stats = entry["stats"]
        entry["discovery_score"] = float(discovery_score)
        entry["repair_score"] = float(repair_score)
        entry["overview_score"] = float(overview_score)
        entry["support_score"] = float(support_score)
        entry.update(temporal_signals.get(idx, {}))
        entry["is_overview_candidate"] = bool(
            stats["context_score"] >= ctx_thr
            and stats["useful_union_ratio"] >= cov_thr
            and stats["grid_cells"] >= int(round(grid_thr))
            and stats["texture_penalty"] <= 0.24
            and stats["concentration_penalty"] <= 0.30
        )
        entry["is_discovery_candidate"] = bool(
            entry["discovery_score"] >= discovery_thr
            and stats["frame_quality"] >= max(0.0, quality_thr - 0.45)
            and (
                entry["discovery_novelty"] >= 0.035
                or entry["mask_rise"] >= 0.08
                or entry["coverage_rise"] >= 0.03
                or entry["grid_novelty"] >= 0.08
            )
        )
        entry["is_repair_candidate"] = bool(
            entry["repair_score"] >= repair_thr
            and stats["frame_quality"] >= max(0.0, quality_thr - 0.65)
            and (
                entry["appearance_jump"] >= 0.08
                or entry["transition_jump"] >= 0.08
                or entry["mask_drop"] >= 0.08
                or entry["coverage_drop"] >= 0.03
            )
        )
        entry["is_support_candidate"] = bool(
            stats["context_score"] >= max(8.4, ctx_thr - 1.0)
            and stats["useful_union_ratio"] >= max(0.14, cov_thr - 0.04)
            and stats["grid_cells"] >= max(5, int(round(grid_thr)) - 1)
            and stats["texture_penalty"] <= 0.26
        )
    return {
        "context_threshold": float(ctx_thr),
        "coverage_threshold": float(cov_thr),
        "grid_threshold": int(round(grid_thr)),
        "quality_threshold": float(quality_thr),
        "discovery_threshold": float(discovery_thr),
        "repair_threshold": float(repair_thr),
    }


def selection_objective(entries, total_frames, target_count, diversity_weight, min_frame_gap, similarity_cap):
    if not entries:
        return -1e9
    base_mean = float(np.mean([entry["global_base_score"] for entry in entries]))
    context_mean = float(np.mean([entry["stats"]["context_score"] for entry in entries]))
    coverage_union = np.max(np.stack([entry["stats"]["coverage_vec"] for entry in entries], axis=0), axis=0)
    focus_union = np.max(np.stack([entry["stats"]["focus_vec"] for entry in entries], axis=0), axis=0)
    grid_union = np.max(np.stack([entry["stats"]["grid_vec"] for entry in entries], axis=0), axis=0)

    coverage_score = float(coverage_union.mean())
    focus_score = float(focus_union.mean())
    grid_score = float(grid_union.mean())
    bins_hit = float((coverage_union > 0.10).mean())

    if len(entries) > 1:
        sims = []
        overlaps = []
        idxs = sorted(int(entry["idx"]) for entry in entries)
        gaps = np.diff(idxs)
        for i in range(len(entries)):
            cov_i = entries[i]["stats"]["coverage_vec"]
            for j in range(i + 1, len(entries)):
                cov_j = entries[j]["stats"]["coverage_vec"]
                sims.append(_selection_pair_similarity(entries[i], entries[j]))
                overlaps.append(
                    float(np.minimum(cov_i, cov_j).sum() / max(np.maximum(cov_i, cov_j).sum(), 1e-6))
                )
        sims = np.asarray(sims, dtype=np.float32)
        overlaps = np.asarray(overlaps, dtype=np.float32)
        diversity = float(np.mean(1.0 - sims)) if sims.size else 0.0
        redundancy_penalty = float(np.mean(np.clip(sims - similarity_cap, 0.0, 1.0))) if sims.size else 0.0
        overlap_penalty = float(overlaps.mean()) if overlaps.size else 0.0
        gap_score = float(np.clip(gaps.mean() / max(float(total_frames / max(target_count, 1)), 1.0), 0.0, 1.0))
        min_gap_score = float(np.clip(gaps.min() / max(float(min_frame_gap), 1.0), 0.0, 1.2))
    else:
        diversity = 0.0
        redundancy_penalty = 0.0
        overlap_penalty = 0.0
        gap_score = 0.0
        min_gap_score = 0.0

    return (
        0.30 * base_mean
        + 0.40 * context_mean
        + 2.40 * coverage_score
        + 1.50 * focus_score
        + 0.90 * grid_score
        + 0.90 * bins_hit
        + 2.00 * diversity_weight * diversity
        + 0.30 * gap_score
        + 0.55 * min_gap_score
        - 1.30 * redundancy_penalty
        - 1.10 * overlap_penalty
    )


def _family_counts(entries):
    counts = {}
    for entry in entries:
        family_id = int(entry.get("view_family", -1))
        counts[family_id] = counts.get(family_id, 0) + 1
    return counts


def _family_diversity_stats(entries):
    counts = _family_counts(entries)
    if not counts:
        return {
            "unique_families": 0,
            "max_family_count": 0,
            "repeat_penalty": 0.0,
            "overcrowded_penalty": 0.0,
        }
    repeat_penalty = float(sum(max(0, count - 1) for count in counts.values()))
    overcrowded_penalty = float(sum(max(0, count - 2) for count in counts.values()))
    return {
        "unique_families": int(len(counts)),
        "max_family_count": int(max(counts.values())),
        "repeat_penalty": repeat_penalty,
        "overcrowded_penalty": overcrowded_penalty,
    }


def _quality_selection_objective(
    entries,
    total_frames,
    target_count,
    diversity_weight,
    min_frame_gap,
    similarity_cap,
):
    if not entries:
        return -1e9
    support_scores = np.asarray([float(entry.get("support_score", 0.0)) for entry in entries], dtype=np.float32)
    overview_scores = np.asarray([float(entry.get("overview_score", 0.0)) for entry in entries], dtype=np.float32)
    family_stats = _family_diversity_stats(entries)
    return (
        selection_objective(
            entries,
            total_frames=total_frames,
            target_count=target_count,
            diversity_weight=diversity_weight,
            min_frame_gap=min_frame_gap,
            similarity_cap=similarity_cap,
        )
        + 1.10 * float(support_scores.mean())
        + 2.10 * float(support_scores.min())
        + 0.55 * float(overview_scores.mean())
        + 0.80 * (family_stats["unique_families"] / max(len(entries), 1))
        - 0.35 * family_stats["repeat_penalty"]
        - 0.90 * family_stats["overcrowded_penalty"]
    )


def _quality_swap_preserves_diversity(current_entries, trial_entries, quality_family_cap):
    current_family_stats = _family_diversity_stats(current_entries)
    trial_family_stats = _family_diversity_stats(trial_entries)
    if trial_family_stats["max_family_count"] > quality_family_cap:
        return False, trial_family_stats
    if trial_family_stats["unique_families"] < current_family_stats["unique_families"]:
        return False, trial_family_stats
    if trial_family_stats["overcrowded_penalty"] > current_family_stats["overcrowded_penalty"]:
        return False, trial_family_stats
    if (
        trial_family_stats["repeat_penalty"] > current_family_stats["repeat_penalty"]
        and trial_family_stats["unique_families"] == current_family_stats["unique_families"]
    ):
        return False, trial_family_stats
    return True, trial_family_stats


def _phase_objective(entries, total_frames, target_count, diversity_weight, min_frame_gap, similarity_cap, score_key):
    if not entries:
        return -1e9
    role_scores = np.asarray(
        [
            float(
                entry.get(
                    score_key,
                    entry.get("support_score", entry.get("overview_score", 0.0)),
                )
            )
            for entry in entries
        ],
        dtype=np.float32,
    )
    min_role = float(role_scores.min())
    mean_role = float(role_scores.mean())
    full_objective = selection_objective(
        entries,
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=min_frame_gap,
        similarity_cap=similarity_cap,
    )
    return 3.40 * min_role + 1.80 * mean_role + 0.60 * full_objective


def _greedy_phase_select(
    candidates,
    count,
    current,
    total_frames,
    target_count,
    diversity_weight,
    min_frame_gap,
    similarity_cap,
    score_key,
    max_per_family=None,
):
    selected = list(current)
    remaining = [entry for entry in candidates if int(entry["idx"]) not in {int(sel["idx"]) for sel in current}]

    while remaining and len(selected) < len(current) + count:
        family_counts = {}
        for sel in selected:
            family_id = int(sel.get("view_family", -1))
            family_counts[family_id] = family_counts.get(family_id, 0) + 1
        valid = [
            entry for entry in remaining
            if all(abs(int(entry["idx"]) - int(sel["idx"])) >= min_frame_gap for sel in selected)
            and all(_selection_pair_similarity(entry, sel) <= similarity_cap for sel in selected)
            and (
                max_per_family is None
                or family_counts.get(int(entry.get("view_family", -1)), 0) < int(max_per_family)
            )
        ]
        if not valid:
            break
        best_by_family = {}
        for entry in valid:
            family_id = int(entry.get("view_family", -1))
            current_best = best_by_family.get(family_id)
            entry_key = (
                float(entry.get(score_key, entry.get("support_score", entry.get("overview_score", 0.0)))),
                float(entry["stats"]["context_score"]),
                float(entry["stats"]["useful_union_ratio"]),
            )
            if current_best is None:
                best_by_family[family_id] = entry
                continue
            current_key = (
                float(current_best.get(score_key, current_best.get("support_score", current_best.get("overview_score", 0.0)))),
                float(current_best["stats"]["context_score"]),
                float(current_best["stats"]["useful_union_ratio"]),
            )
            if entry_key > current_key:
                best_by_family[family_id] = entry
        valid = list(best_by_family.values())
        if not selected:
            best_entry = max(
                valid,
                key=lambda entry: (
                    float(entry.get(score_key, entry.get("support_score", entry.get("overview_score", 0.0)))),
                    float(entry.get("overview_score", 0.0)),
                    float(entry["stats"]["context_score"]),
                    float(entry["stats"]["useful_union_ratio"]),
                    float(entry["stats"].get("frame_quality", 0.0)),
                ),
            )
        else:
            best_entry = max(
                valid,
                key=lambda entry: (
                    _phase_objective(
                        selected + [entry],
                        total_frames=total_frames,
                        target_count=target_count,
                        diversity_weight=diversity_weight,
                        min_frame_gap=min_frame_gap,
                        similarity_cap=similarity_cap,
                        score_key=score_key,
                    ),
                    float(entry.get(score_key, entry.get("support_score", entry.get("overview_score", 0.0)))),
                    entry["stats"]["context_score"],
                    entry["stats"]["useful_union_ratio"],
                ),
            )
        selected.append(best_entry)
        remaining = [entry for entry in remaining if int(entry["idx"]) != int(best_entry["idx"])]
    return selected


def _temporal_guard_entries(entries, front_target, min_gap, score_key):
    if not entries or front_target <= 0 or min_gap <= 0:
        return list(entries)

    ordered = sorted(
        entries,
        key=lambda entry: (
            float(entry.get(score_key, 0.0)),
            float(entry.get("overview_score", 0.0)),
            float(entry["stats"]["context_score"]),
            float(entry["stats"]["useful_union_ratio"]),
        ),
        reverse=True,
    )
    front = []
    deferred = []
    for entry in ordered:
        idx = int(entry["idx"])
        if len(front) < front_target and all(abs(idx - int(prev["idx"])) >= min_gap for prev in front):
            front.append(entry)
        else:
            deferred.append(entry)
    return front + deferred


def _rank_entries(entries, score_key):
    return sorted(
        entries,
        key=lambda entry: (
            float(entry.get(score_key, 0.0)),
            float(entry.get("overview_score", 0.0)),
            float(entry["stats"]["context_score"]),
            float(entry["stats"]["useful_union_ratio"]),
        ),
        reverse=True,
    )


def _build_rejected_rescue_set(
    rejected_entries,
    target_count,
    total_frames,
    diversity_weight,
    min_frame_gap,
    similarity_cap,
    anchor_target,
    support_target,
):
    if not rejected_entries or target_count <= 0:
        return []

    rescue_gap = max(32, min_frame_gap - 8)
    rescue_similarity = min(0.96, similarity_cap + 0.02)
    rescue_anchor_cap = 1
    rescue_support_cap = 3

    overview_pool = _rank_entries(
        [entry for entry in rejected_entries if entry.get("is_overview_candidate", False)],
        "overview_score",
    )
    if len(overview_pool) < anchor_target:
        overview_pool = _rank_entries(rejected_entries, "overview_score")

    selected = _greedy_phase_select(
        overview_pool,
        count=anchor_target,
        current=[],
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=rescue_gap,
        similarity_cap=rescue_similarity,
        score_key="overview_score",
        max_per_family=rescue_anchor_cap,
    )

    support_pool = sorted(
        [entry for entry in rejected_entries if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected}],
        key=lambda entry: (
            float(entry.get("discovery_score", 0.0)) + float(entry.get("repair_score", 0.0)),
            float(entry.get("support_score", 0.0)),
            float(entry["stats"].get("frame_quality", 0.0)),
            float(entry["stats"].get("useful_union_ratio", 0.0)),
        ),
        reverse=True,
    )
    selected = _greedy_phase_select(
        support_pool,
        count=max(0, target_count - len(selected)),
        current=selected,
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=rescue_gap,
        similarity_cap=rescue_similarity,
        score_key="support_score",
        max_per_family=rescue_support_cap,
    )
    return sorted(selected, key=lambda entry: int(entry["idx"]))


def _entry_quality_value(entry):
    return float(
        1.35 * float(entry.get("support_score", 0.0))
        + 0.55 * float(entry.get("overview_score", 0.0))
        + 0.40 * float(entry.get("discovery_score", 0.0))
        + 0.42 * float(entry.get("repair_score", 0.0))
        + 0.035 * float(entry["stats"]["context_score"])
        + 0.85 * float(entry["stats"]["useful_union_ratio"])
        + 0.085 * float(entry["stats"].get("frame_quality", 0.0))
    )


def _selection_role_counts(entries):
    counts = {"overview": 0, "discovery": 0, "repair": 0}
    for entry in entries:
        role = entry.get("selection_role")
        if role in counts:
            counts[role] += 1
    return counts


def _assign_final_selection_roles(entries, overview_target, discovery_target, repair_target):
    ordered = sorted(entries, key=lambda entry: int(entry["idx"]))
    if not ordered:
        return ordered

    for entry in ordered:
        entry["selection_role"] = ""

    used = set()

    def _pick(role, count, primary_key, secondary_keys=()):
        if count <= 0:
            return
        ranked = sorted(
            ordered,
            key=lambda entry: (
                primary_key(entry),
                *[key(entry) for key in secondary_keys],
                float(entry["stats"].get("frame_quality", 0.0)),
                float(entry["stats"].get("context_score", 0.0)),
                float(entry["stats"].get("useful_union_ratio", 0.0)),
            ),
            reverse=True,
        )
        picked = 0
        for entry in ranked:
            idx = int(entry["idx"])
            if idx in used:
                continue
            entry["selection_role"] = role
            used.add(idx)
            picked += 1
            if picked >= count:
                break

    _pick(
        "discovery",
        int(discovery_target),
        lambda entry: float(entry.get("discovery_score", 0.0)),
        secondary_keys=(
            lambda entry: float(entry.get("discovery_novelty", 0.0)),
            lambda entry: float(entry.get("mask_rise", 0.0)),
            lambda entry: float(entry.get("coverage_rise", 0.0)),
        ),
    )
    _pick(
        "repair",
        int(repair_target),
        lambda entry: float(entry.get("repair_score", 0.0)),
        secondary_keys=(
            lambda entry: float(entry.get("appearance_jump", 0.0)),
            lambda entry: float(entry.get("transition_jump", 0.0)),
            lambda entry: float(entry.get("mask_drop", 0.0)),
            lambda entry: float(entry.get("coverage_drop", 0.0)),
        ),
    )
    remaining_overview = max(0, int(overview_target))
    _pick(
        "overview",
        remaining_overview,
        lambda entry: float(entry.get("overview_score", 0.0)),
        secondary_keys=(
            lambda entry: float(entry.get("support_score", 0.0)),
        ),
    )

    for entry in ordered:
        if not entry.get("selection_role"):
            entry["selection_role"] = "overview"

    return ordered


def _role_fill_score(entry, overview_need=0, discovery_need=0, repair_need=0):
    base = (
        1.00 * float(entry.get("support_score", 0.0))
        + 0.35 * float(entry["stats"].get("frame_quality", 0.0))
        + 0.28 * float(entry["stats"].get("context_score", 0.0))
        + 0.90 * float(entry["stats"].get("useful_union_ratio", 0.0))
    )
    if overview_need > 0:
        base += 1.10 * float(entry.get("overview_score", 0.0))
    if discovery_need > 0:
        base += (
            1.55 * float(entry.get("discovery_score", 0.0))
            + 0.70 * float(entry.get("discovery_novelty", 0.0))
            + 0.45 * float(entry.get("mask_rise", 0.0))
            + 0.25 * float(entry.get("coverage_rise", 0.0))
        )
    if repair_need > 0:
        base += (
            1.55 * float(entry.get("repair_score", 0.0))
            + 0.60 * float(entry.get("appearance_jump", 0.0))
            + 0.55 * float(entry.get("transition_jump", 0.0))
            + 0.45 * float(entry.get("mask_drop", 0.0))
            + 0.25 * float(entry.get("coverage_drop", 0.0))
        )
    return float(base)


def _compute_final_fill_floors(selected_entries):
    if not selected_entries:
        return 0.64, 8.9, 0.16, 0.0, 0.56, 8.4, 0.13, 0.0

    support_vals = np.asarray([float(entry.get("support_score", 0.0)) for entry in selected_entries], dtype=np.float32)
    ctx_vals = np.asarray([float(entry["stats"]["context_score"]) for entry in selected_entries], dtype=np.float32)
    cov_vals = np.asarray([float(entry["stats"]["useful_union_ratio"]) for entry in selected_entries], dtype=np.float32)
    fq_vals = np.asarray([float(entry["stats"].get("frame_quality", 0.0)) for entry in selected_entries], dtype=np.float32)

    strict_support = max(0.64, float(np.quantile(support_vals, 0.18)) - 0.18)
    strict_context = max(8.9, float(np.quantile(ctx_vals, 0.18)) - 1.55)
    strict_cov = max(0.16, float(np.quantile(cov_vals, 0.18)) - 0.10)
    strict_fq = max(0.0, float(np.quantile(fq_vals, 0.18)) - 0.20)

    relaxed_support = max(0.56, strict_support - 0.08)
    relaxed_context = max(8.4, strict_context - 0.55)
    relaxed_cov = max(0.13, strict_cov - 0.04)
    relaxed_fq = max(0.0, strict_fq - 0.20)
    return (
        float(strict_support),
        float(strict_context),
        float(strict_cov),
        float(strict_fq),
        float(relaxed_support),
        float(relaxed_context),
        float(relaxed_cov),
        float(relaxed_fq),
    )


def _meets_explicit_quality_floor(entry, support_floor, context_floor, coverage_floor, frame_quality_floor=0.0):
    return (
        float(entry.get("support_score", 0.0)) >= float(support_floor)
        and float(entry["stats"]["context_score"]) >= float(context_floor)
        and float(entry["stats"]["useful_union_ratio"]) >= float(coverage_floor)
        and float(entry["stats"].get("frame_quality", 0.0)) >= float(frame_quality_floor)
    )


def _family_duplicate_gain_requirement(current_entries, victim, candidate):
    current_counts = _family_counts(current_entries)
    victim_family = int(victim.get("view_family", -1))
    candidate_family = int(candidate.get("view_family", -1))

    if candidate_family == victim_family:
        return 0.08

    remaining_counts = dict(current_counts)
    if victim_family in remaining_counts:
        remaining_counts[victim_family] -= 1
        if remaining_counts[victim_family] <= 0:
            remaining_counts.pop(victim_family, None)

    duplicate_after = remaining_counts.get(candidate_family, 0) >= 1
    victim_was_duplicate = current_counts.get(victim_family, 0) > 1
    if duplicate_after and not victim_was_duplicate:
        return 0.16
    if duplicate_after:
        return 0.10
    return 0.02


def _final_polish_selected_set(
    selected_entries,
    all_entries,
    target_count,
    total_frames,
    diversity_weight,
    min_frame_gap,
    similarity_cap,
):
    if len(selected_entries) < target_count:
        return sorted(selected_entries, key=lambda entry: int(entry["idx"])), []

    polish_gap = max(16, min_frame_gap // 3)
    polish_similarity = min(0.97, similarity_cap + 0.03)
    polish_family_cap = max(4, int(np.ceil(target_count / 3)))
    notes = []
    selected = sorted(selected_entries, key=lambda entry: int(entry["idx"]))
    current_quality_objective = _quality_selection_objective(
        selected,
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=polish_gap,
        similarity_cap=polish_similarity,
    )
    current_family_stats = _family_diversity_stats(selected)

    rejected_pool = _rank_entries(
        [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected}
        ],
        "support_score",
    )
    rejected_pool = _temporal_guard_entries(
        rejected_pool,
        front_target=max(target_count * 2, 16),
        min_gap=max(12, polish_gap // 2),
        score_key="support_score",
    )

    improved = True
    while improved:
        improved = False
        weak_selected = sorted(
            selected,
            key=lambda entry: (
                _entry_quality_value(entry),
                float(entry["stats"]["useful_union_ratio"]),
                float(entry["stats"]["context_score"]),
            ),
        )
        for candidate in rejected_pool[: max(target_count * 3, 24)]:
            if improved:
                break
            for victim in weak_selected[: max(6, target_count // 2)]:
                if int(candidate["idx"]) == int(victim["idx"]):
                    continue
                trial = [
                    entry
                    for entry in selected
                    if int(entry["idx"]) != int(victim["idx"])
                ] + [candidate]
                if not _entries_respect_gap(trial, polish_gap):
                    continue
                if not _entries_respect_similarity(trial, polish_similarity):
                    continue
                allowed, trial_family_stats = _quality_swap_preserves_diversity(
                    selected,
                    trial,
                    quality_family_cap=polish_family_cap,
                )
                if not allowed:
                    continue

                trial_quality_objective = _quality_selection_objective(
                    trial,
                    total_frames=total_frames,
                    target_count=target_count,
                    diversity_weight=diversity_weight,
                    min_frame_gap=polish_gap,
                    similarity_cap=polish_similarity,
                )
                quality_gain = _entry_quality_value(candidate) - _entry_quality_value(victim)
                required_gain = _family_duplicate_gain_requirement(selected, victim, candidate)
                objective_gain = trial_quality_objective - current_quality_objective
                same_family = int(candidate.get("view_family", -1)) == int(victim.get("view_family", -1))
                frame_quality_gain = float(candidate["stats"].get("frame_quality", 0.0)) - float(
                    victim["stats"].get("frame_quality", 0.0)
                )
                family_improved = (
                    trial_family_stats["unique_families"] > current_family_stats["unique_families"]
                    or trial_family_stats["repeat_penalty"] < current_family_stats["repeat_penalty"]
                    or trial_family_stats["overcrowded_penalty"] < current_family_stats["overcrowded_penalty"]
                )
                if same_family:
                    stronger_same_family = (
                        float(candidate.get("overview_score", 0.0)) >= float(victim.get("overview_score", 0.0)) + 0.03
                        or float(candidate["stats"]["context_score"]) >= float(victim["stats"]["context_score"]) + 0.25
                        or float(candidate["stats"]["useful_union_ratio"]) >= float(victim["stats"]["useful_union_ratio"]) + 0.04
                    )
                    if not stronger_same_family:
                        continue
                if quality_gain <= 0.0:
                    if not family_improved or objective_gain < 0.08 or frame_quality_gain < -0.05:
                        continue
                if not family_improved and frame_quality_gain < -0.05:
                    continue
                if not family_improved and quality_gain < required_gain:
                    continue
                if quality_gain < required_gain and objective_gain < max(0.03, required_gain * 0.75):
                    continue

                selected = sorted(trial, key=lambda entry: int(entry["idx"]))
                current_quality_objective = trial_quality_objective
                current_family_stats = trial_family_stats
                notes.append(
                    f"final polish swap {int(victim['idx'])}->{int(candidate['idx'])} "
                    f"(gain={quality_gain:.3f}, fq_gain={frame_quality_gain:.3f}, family requirement={required_gain:.3f})"
                )
                improved = True
                break

    return selected, notes


def _force_fill_entries(
    selected_entries,
    all_entries,
    target_count,
    total_frames,
    diversity_weight,
    support_gap,
    support_similarity_cap,
):
    if len(selected_entries) >= target_count:
        return selected_entries, []

    notes = []
    selected = sorted(selected_entries, key=lambda entry: int(entry["idx"]))
    (
        strict_support_floor,
        strict_context_floor,
        strict_coverage_floor,
        strict_frame_quality_floor,
        relaxed_support_floor,
        relaxed_context_floor,
        relaxed_coverage_floor,
        relaxed_frame_quality_floor,
    ) = _compute_final_fill_floors(selected)

    def _remaining():
        return [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected}
        ]

    staged_specs = [
        (
            "force fill stage1",
            max(24, support_gap // 2),
            min(0.96, support_similarity_cap + 0.02),
            4,
            (strict_support_floor, strict_context_floor, strict_coverage_floor, strict_frame_quality_floor),
        ),
        (
            "force fill stage2",
            max(12, support_gap // 3),
            min(0.99, support_similarity_cap + 0.05),
            max(target_count, 6),
            (relaxed_support_floor, relaxed_context_floor, relaxed_coverage_floor, relaxed_frame_quality_floor),
        ),
    ]

    for label, stage_gap, stage_similarity, stage_family_cap, stage_floors in staged_specs:
        if len(selected) >= target_count:
            break
        support_floor, context_floor, coverage_floor, frame_quality_floor = stage_floors
        pool = _rank_entries(
            [
                entry
                for entry in _remaining()
                if _meets_explicit_quality_floor(
                    entry,
                    support_floor=support_floor,
                    context_floor=context_floor,
                    coverage_floor=coverage_floor,
                    frame_quality_floor=frame_quality_floor,
                )
            ],
            "support_score",
        )
        pool = _temporal_guard_entries(
            pool,
            front_target=max(target_count * 2, 12),
            min_gap=max(8, stage_gap),
            score_key="support_score",
        )
        before = len(selected)
        selected = _greedy_phase_select(
            pool,
            count=target_count - len(selected),
            current=selected,
            total_frames=total_frames,
            target_count=target_count,
            diversity_weight=diversity_weight,
            min_frame_gap=stage_gap,
            similarity_cap=stage_similarity,
            score_key="support_score",
            max_per_family=stage_family_cap,
        )
        if len(selected) > before:
            notes.append(
                f"{label} {before}->{len(selected)} gap {stage_gap} similarity {stage_similarity:.2f} "
                f"family cap {stage_family_cap} floor s>={support_floor:.2f} ctx>={context_floor:.2f} cov>={coverage_floor:.2f} fq>={frame_quality_floor:.2f}"
            )

    if len(selected) < target_count:
        pool = _rank_entries(
            [
                entry
                for entry in _remaining()
                if _meets_explicit_quality_floor(
                    entry,
                    support_floor=relaxed_support_floor,
                    context_floor=relaxed_context_floor,
                    coverage_floor=relaxed_coverage_floor,
                    frame_quality_floor=relaxed_frame_quality_floor,
                )
            ],
            "support_score",
        )
        pool = _temporal_guard_entries(
            pool,
            front_target=max(target_count * 2, 12),
            min_gap=max(8, support_gap // 4),
            score_key="support_score",
        )
        before = len(selected)
        for entry in pool:
            if len(selected) >= target_count:
                break
            if int(entry["idx"]) in {int(sel["idx"]) for sel in selected}:
                continue
            selected.append(entry)
        if len(selected) > before:
            notes.append(
                f"force fill final append {before}->{len(selected)} from remaining ranked pool "
                f"(floor s>={relaxed_support_floor:.2f} ctx>={relaxed_context_floor:.2f} cov>={relaxed_coverage_floor:.2f} fq>={relaxed_frame_quality_floor:.2f})"
            )

    if len(selected) < target_count:
        pool = _rank_entries(_remaining(), "support_score")
        pool = _temporal_guard_entries(
            pool,
            front_target=max(target_count * 2, 12),
            min_gap=max(6, support_gap // 5),
            score_key="support_score",
        )
        before = len(selected)
        guarantee_gap = max(6, support_gap // 5)
        guarantee_similarity = min(0.99, support_similarity_cap + 0.06)
        guarantee_family_cap = max(2, int(np.ceil(target_count / 4)))
        selected = _greedy_phase_select(
            pool,
            count=target_count - len(selected),
            current=selected,
            total_frames=total_frames,
            target_count=target_count,
            diversity_weight=diversity_weight,
            min_frame_gap=guarantee_gap,
            similarity_cap=guarantee_similarity,
            score_key="support_score",
            max_per_family=guarantee_family_cap,
        )
        if len(selected) < target_count:
            for entry in pool:
                if len(selected) >= target_count:
                    break
                if int(entry["idx"]) in {int(sel["idx"]) for sel in selected}:
                    continue
                selected.append(entry)
        if len(selected) > before:
            notes.append(
                f"force fill guarantee {before}->{len(selected)} from remaining ranked pool "
                f"(gap {guarantee_gap} similarity {guarantee_similarity:.2f} family cap {guarantee_family_cap})"
            )

    return sorted(selected, key=lambda entry: int(entry["idx"])), notes


def optimize_keyframe_set(
    all_entries,
    target_count,
    total_frames,
    min_frame_gap,
    diversity_weight=0.95,
    similarity_cap=0.90,
    candidate_subset_size=None,
):
    if not all_entries or target_count <= 0:
        return []

    role_thresholds = _prepare_entry_roles(all_entries)
    family_similarity_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_VIEW_FAMILY_SIMILARITY"),
        0.88,
    )
    families = _assign_view_families(
        all_entries,
        family_similarity_threshold=family_similarity_threshold,
    )

    if target_count <= 4:
        overview_target = min(int(target_count), max(3, int(target_count - 1)))
        discovery_target = 0
        repair_target = 0
    else:
        overview_target = max(3, int(round(target_count * 0.42)))
        discovery_target = max(2, int(round(target_count * 0.25)))
        repair_target = max(1, target_count - overview_target - discovery_target)
        total_target = overview_target + discovery_target + repair_target
        while total_target > target_count and overview_target > 3:
            overview_target -= 1
            total_target -= 1
        while total_target > target_count and discovery_target > 2:
            discovery_target -= 1
            total_target -= 1
        while total_target < target_count:
            repair_target += 1
            total_target += 1

    anchor_target = int(overview_target)
    support_target = int(discovery_target + repair_target)
    overview_gap = int(min_frame_gap)
    discovery_gap = max(36, overview_gap - 10)
    repair_gap = max(28, overview_gap - 18)
    support_gap = max(32, overview_gap - 12)
    overview_similarity_cap = min(0.97, float(similarity_cap) + 0.04)
    discovery_similarity_cap = min(0.96, max(float(similarity_cap), 0.91) + 0.03)
    repair_similarity_cap = min(0.97, max(float(similarity_cap), 0.91) + 0.04)
    support_similarity_cap = min(0.96, max(float(similarity_cap), 0.92))
    anchor_family_cap = 1
    discovery_family_cap = 2
    repair_family_cap = 2
    support_family_cap = 2
    support_min_support_score = _parse_float(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_SUPPORT_MIN_SCORE"),
        0.72,
    )
    support_min_context = max(0.0, role_thresholds["context_threshold"] - 0.2)
    support_min_coverage = max(0.22, role_thresholds["coverage_threshold"] - 0.01)
    support_relaxed_min_support_score = max(0.62, support_min_support_score - 0.10)
    support_relaxed_min_context = max(0.0, support_min_context - 0.7)
    support_relaxed_min_coverage = max(0.18, support_min_coverage - 0.05)
    discovery_min_score = max(0.58, role_thresholds["discovery_threshold"] - 0.06)
    repair_min_score = max(0.56, role_thresholds["repair_threshold"] - 0.06)
    temporal_min_quality = max(0.0, role_thresholds["quality_threshold"] - 0.55)
    fallback_notes = []

    def _meets_support_quality(entry):
        return (
            entry["support_score"] >= support_min_support_score
            and entry["stats"]["context_score"] >= support_min_context
            and entry["stats"]["useful_union_ratio"] >= support_min_coverage
        )

    def _meets_discovery_quality(entry):
        return (
            entry["discovery_score"] >= discovery_min_score
            and entry["stats"]["frame_quality"] >= temporal_min_quality
            and (
                entry.get("is_discovery_candidate", False)
                or entry.get("discovery_novelty", 0.0) >= 0.04
                or entry.get("mask_rise", 0.0) >= 0.10
                or entry.get("coverage_rise", 0.0) >= 0.03
            )
        )

    def _meets_repair_quality(entry):
        return (
            entry["repair_score"] >= repair_min_score
            and entry["stats"]["frame_quality"] >= max(0.0, temporal_min_quality - 0.2)
            and (
                entry.get("is_repair_candidate", False)
                or entry.get("appearance_jump", 0.0) >= 0.08
                or entry.get("transition_jump", 0.0) >= 0.08
                or entry.get("mask_drop", 0.0) >= 0.08
                or entry.get("coverage_drop", 0.0) >= 0.03
            )
        )

    def _support_rank(entry):
        return (
            entry["support_score"],
            entry.get("repair_score", 0.0),
            entry.get("discovery_score", 0.0),
            entry["overview_score"],
            entry["stats"]["context_score"],
            entry["stats"]["useful_union_ratio"],
        )

    def _discovery_rank(entry):
        return (
            entry["discovery_score"],
            entry.get("discovery_novelty", 0.0),
            entry.get("mask_rise", 0.0),
            entry.get("grid_novelty", 0.0),
            entry["support_score"],
            entry["stats"]["frame_quality"],
        )

    def _repair_rank(entry):
        return (
            entry["repair_score"],
            entry.get("appearance_jump", 0.0),
            entry.get("transition_jump", 0.0),
            entry.get("mask_drop", 0.0),
            entry.get("coverage_drop", 0.0),
            entry["support_score"],
            entry["stats"]["frame_quality"],
        )

    def _meets_support_quality_relaxed(entry):
        return (
            entry["support_score"] >= support_relaxed_min_support_score
            and entry["stats"]["context_score"] >= support_relaxed_min_context
            and entry["stats"]["useful_union_ratio"] >= support_relaxed_min_coverage
        )

    def _selection_role_deficits(entries):
        counts = _selection_role_counts(entries)
        return {
            "overview": max(0, int(overview_target) - counts["overview"]),
            "discovery": max(0, int(discovery_target) - counts["discovery"]),
            "repair": max(0, int(repair_target) - counts["repair"]),
        }

    def _choose_fill_role(entry, deficits):
        candidates = []
        if deficits["discovery"] > 0 and _meets_discovery_quality(entry):
            candidates.append(("discovery", _role_fill_score(entry, discovery_need=deficits["discovery"])))
        if deficits["repair"] > 0 and _meets_repair_quality(entry):
            candidates.append(("repair", _role_fill_score(entry, repair_need=deficits["repair"])))
        if deficits["overview"] > 0 and entry.get("is_overview_candidate", False):
            candidates.append(("overview", _role_fill_score(entry, overview_need=deficits["overview"])))
        if not candidates:
            candidates = [
                ("discovery", _role_fill_score(entry, discovery_need=1) if _meets_discovery_quality(entry) else -1e9),
                ("repair", _role_fill_score(entry, repair_need=1) if _meets_repair_quality(entry) else -1e9),
                ("overview", _role_fill_score(entry, overview_need=1) if entry.get("is_overview_candidate", False) else -1e9),
            ]
        best_role, best_score = max(candidates, key=lambda item: item[1])
        return best_role if best_score > -1e8 else "overview"

    def _tag_new_entries(previous_entries, current_entries, default_role):
        prev_ids = {int(entry["idx"]) for entry in previous_entries}
        tagged = list(current_entries)
        running_selected = list(previous_entries)
        for entry in tagged:
            if int(entry["idx"]) in prev_ids:
                continue
            deficits = _selection_role_deficits(running_selected)
            role = default_role
            if default_role == "fill":
                role = _choose_fill_role(entry, deficits)
            entry["selection_role"] = role
            running_selected.append(entry)
        return tagged

    def _ensure_selection_roles(entries):
        ordered = sorted(entries, key=lambda entry: int(entry["idx"]))
        running = []
        for entry in ordered:
            if not entry.get("selection_role"):
                entry["selection_role"] = _choose_fill_role(entry, _selection_role_deficits(running))
            running.append(entry)
        return ordered

    def _rank_fill_pool(entries, current_entries):
        deficits = _selection_role_deficits(current_entries)
        ranked = []
        for entry in entries:
            fill_score = _role_fill_score(
                entry,
                overview_need=deficits["overview"],
                discovery_need=deficits["discovery"],
                repair_need=deficits["repair"],
            )
            entry["fill_score"] = float(fill_score)
            ranked.append(entry)
        return sorted(
            ranked,
            key=lambda entry: (
                float(entry.get("fill_score", 0.0)),
                float(entry["stats"].get("frame_quality", 0.0)),
                float(entry["stats"].get("context_score", 0.0)),
                float(entry["stats"].get("useful_union_ratio", 0.0)),
            ),
            reverse=True,
        )

    overview_pool = sorted(
        [entry for entry in all_entries if entry["is_overview_candidate"]],
        key=lambda x: (
            x["overview_score"],
            x["stats"]["context_score"],
            x["stats"]["useful_union_ratio"],
            x["stats"]["grid_cells"],
        ),
        reverse=True,
    )
    if len(overview_pool) < anchor_target:
        fallback_notes.append(
            f"overview pool too small ({len(overview_pool)}/{anchor_target}); extending with top overview scores"
        )
        overview_pool = sorted(
            all_entries,
            key=lambda x: (
                x["overview_score"],
                x["stats"]["context_score"],
                x["stats"]["useful_union_ratio"],
                x["stats"]["grid_cells"],
            ),
            reverse=True,
        )[: max(anchor_target * 3, target_count * 3)]

    selected_entries = _greedy_phase_select(
        overview_pool,
        count=overview_target,
        current=[],
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=overview_gap,
        similarity_cap=overview_similarity_cap,
        score_key="overview_score",
        max_per_family=anchor_family_cap,
    )
    selected_entries = _tag_new_entries([], selected_entries, "overview")

    if len(selected_entries) < overview_target:
        fallback_notes.append(
            f"overview phase filled only {len(selected_entries)}/{overview_target}; relaxing overview gap/similarity"
        )
        previous_entries = list(selected_entries)
        selected_entries = _greedy_phase_select(
            overview_pool,
            count=overview_target,
            current=[],
            total_frames=total_frames,
            target_count=target_count,
            diversity_weight=diversity_weight,
            min_frame_gap=max(40, overview_gap - 12),
            similarity_cap=min(0.99, overview_similarity_cap + 0.02),
            score_key="overview_score",
            max_per_family=max(1, anchor_family_cap + 1),
        )
        selected_entries = _tag_new_entries([], selected_entries, "overview")

    discovery_pool = sorted(
        [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            and _meets_discovery_quality(entry)
        ],
        key=_discovery_rank,
        reverse=True,
    )
    selected_family_ids = {int(entry.get("view_family", -1)) for entry in selected_entries}
    discovery_pool_new_families = [
        entry for entry in discovery_pool
        if int(entry.get("view_family", -1)) not in selected_family_ids
    ]
    previous_entries = list(selected_entries)
    selected_entries = _greedy_phase_select(
        discovery_pool_new_families,
        count=discovery_target,
        current=selected_entries,
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=discovery_gap,
        similarity_cap=discovery_similarity_cap,
        score_key="discovery_score",
        max_per_family=1,
    )
    selected_entries = _tag_new_entries(previous_entries, selected_entries, "discovery")

    remaining_discovery = max(0, overview_target + discovery_target - len(selected_entries))
    if remaining_discovery > 0:
        discovery_pool_repeat = sorted(
            [
                entry
                for entry in discovery_pool
                if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            ],
            key=_discovery_rank,
            reverse=True,
        )
        previous_entries = list(selected_entries)
        selected_entries = _greedy_phase_select(
            discovery_pool_repeat,
            count=remaining_discovery,
            current=selected_entries,
            total_frames=total_frames,
            target_count=target_count,
            diversity_weight=diversity_weight,
            min_frame_gap=max(32, discovery_gap - 4),
            similarity_cap=min(0.97, discovery_similarity_cap + 0.01),
            score_key="discovery_score",
            max_per_family=discovery_family_cap,
        )
        selected_entries = _tag_new_entries(previous_entries, selected_entries, "discovery")

    repair_pool = sorted(
        [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            and _meets_repair_quality(entry)
        ],
        key=_repair_rank,
        reverse=True,
    )
    repair_pool_new_families = [
        entry
        for entry in repair_pool
        if int(entry.get("view_family", -1))
        not in {int(sel.get("view_family", -1)) for sel in selected_entries}
    ]
    previous_entries = list(selected_entries)
    selected_entries = _greedy_phase_select(
        repair_pool_new_families,
        count=repair_target,
        current=selected_entries,
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=repair_gap,
        similarity_cap=repair_similarity_cap,
        score_key="repair_score",
        max_per_family=1,
    )
    selected_entries = _tag_new_entries(previous_entries, selected_entries, "repair")

    remaining_role_slots = max(0, overview_target + discovery_target + repair_target - len(selected_entries))
    if remaining_role_slots > 0:
        repair_pool_repeat = sorted(
            [
                entry
                for entry in all_entries
                if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
                and _meets_repair_quality(entry)
            ],
            key=_repair_rank,
            reverse=True,
        )
        before_repair = len(selected_entries)
        selected_before_repair = list(selected_entries)
        selected_entries = _greedy_phase_select(
            repair_pool_repeat,
            count=remaining_role_slots,
            current=selected_entries,
            total_frames=total_frames,
            target_count=target_count,
            diversity_weight=diversity_weight,
            min_frame_gap=max(24, repair_gap - 4),
            similarity_cap=min(0.98, repair_similarity_cap + 0.01),
            score_key="repair_score",
            max_per_family=repair_family_cap,
        )
        selected_entries = _tag_new_entries(selected_before_repair, selected_entries, "repair")
        if len(selected_entries) > before_repair:
            fallback_notes.append(
                f"repair phase filled {before_repair}->{len(selected_entries)} "
                f"(score>={repair_min_score:.2f}, quality>={max(0.0, temporal_min_quality - 0.2):.2f})"
            )

    support_pool = _rank_fill_pool(
        [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            and entry["is_support_candidate"]
            and _meets_support_quality(entry)
        ],
        selected_entries,
    )

    if len(selected_entries) < target_count:
        relaxed_new_family_pool = _rank_fill_pool(
            [
                entry
                for entry in all_entries
                if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
                and _meets_support_quality_relaxed(entry)
                and int(entry.get("view_family", -1))
                not in {int(sel.get("view_family", -1)) for sel in selected_entries}
            ],
            selected_entries,
        )
        if relaxed_new_family_pool:
            before_relaxed_new = len(selected_entries)
            previous_entries = list(selected_entries)
            selected_entries = _greedy_phase_select(
                relaxed_new_family_pool,
                count=target_count - len(selected_entries),
                current=selected_entries,
                total_frames=total_frames,
                target_count=target_count,
                diversity_weight=diversity_weight,
                min_frame_gap=max(36, support_gap - 4),
                similarity_cap=min(0.95, support_similarity_cap + 0.01),
                score_key="fill_score",
                max_per_family=1,
            )
            selected_entries = _tag_new_entries(previous_entries, selected_entries, "fill")
            if len(selected_entries) > before_relaxed_new:
                fallback_notes.append(
                    f"relaxed support accepted new families {before_relaxed_new}->{len(selected_entries)} "
                    f"(score>={support_relaxed_min_support_score:.2f}, ctx>={support_relaxed_min_context:.2f}, cov>={support_relaxed_min_coverage:.2f})"
                )

    if len(selected_entries) < target_count:
        top_rejected_pool = [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            and _meets_support_quality_relaxed(entry)
        ]
        top_rejected_pool = _rank_fill_pool(top_rejected_pool, selected_entries)
        top_rejected_pool = _temporal_guard_entries(
            top_rejected_pool,
            front_target=max(target_count * 2, 12),
            min_gap=max(24, support_gap // 2),
            score_key="fill_score",
        )
        top_rejected_family_cap = max(support_family_cap + 1, 3)
        before_top_rejected = len(selected_entries)
        previous_entries = list(selected_entries)
        selected_entries = _greedy_phase_select(
            top_rejected_pool,
            count=target_count - len(selected_entries),
            current=selected_entries,
            total_frames=total_frames,
            target_count=target_count,
            diversity_weight=diversity_weight,
            min_frame_gap=support_gap,
            similarity_cap=support_similarity_cap,
            score_key="fill_score",
            max_per_family=top_rejected_family_cap,
        )
        selected_entries = _tag_new_entries(previous_entries, selected_entries, "fill")
        if len(selected_entries) > before_top_rejected:
            fallback_notes.append(
                f"top_rejected fallback filled {before_top_rejected}->{len(selected_entries)} with family cap {top_rejected_family_cap}"
            )

    if len(selected_entries) < target_count:
        remaining_all = [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            and _meets_support_quality_relaxed(entry)
        ]
        remaining_all = _rank_fill_pool(remaining_all, selected_entries)
        remaining_all = _temporal_guard_entries(
            remaining_all,
            front_target=max(target_count * 2, 12),
            min_gap=max(20, support_gap // 2),
            score_key="fill_score",
        )
        relaxed_family_cap = max(support_family_cap + 1, 3)
        if remaining_all:
            fallback_notes.append(
                f"support phase filled only {len(selected_entries)}/{target_count}; fallback from top rejected gap {support_gap}->{max(32, support_gap - 8)}, similarity {support_similarity_cap:.2f}->{min(0.97, support_similarity_cap + 0.02):.2f}, family cap {relaxed_family_cap} (quality floor kept)"
            )
            previous_entries = list(selected_entries)
            selected_entries = _greedy_phase_select(
                remaining_all,
                count=target_count - len(selected_entries),
                current=selected_entries,
                total_frames=total_frames,
                target_count=target_count,
                diversity_weight=diversity_weight,
                min_frame_gap=max(32, support_gap - 8),
                similarity_cap=min(0.97, support_similarity_cap + 0.02),
                score_key="fill_score",
                max_per_family=relaxed_family_cap,
            )
            selected_entries = _tag_new_entries(previous_entries, selected_entries, "fill")
        else:
            fallback_notes.append(
                f"support phase filled only {len(selected_entries)}/{target_count}; no remaining support candidates passed the quality floor"
            )

    quality_gap = max(32, support_gap - 8)
    quality_similarity_cap = min(0.97, support_similarity_cap + 0.02)
    quality_family_cap = max(support_family_cap + 2, 4)

    current_quality_objective = _quality_selection_objective(
        selected_entries,
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=quality_gap,
        similarity_cap=quality_similarity_cap,
    )
    current_family_stats = _family_diversity_stats(selected_entries)

    improved = True
    while improved:
        improved = False
        rejected_quality = [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            and _meets_support_quality(entry)
        ]
        rejected_quality = _rank_fill_pool(rejected_quality, selected_entries)
        weak_selected = sorted(
            selected_entries,
            key=lambda x: (
                x["support_score"],
                x["overview_score"],
                x["stats"]["context_score"],
                x["stats"]["useful_union_ratio"],
            ),
        )
        for candidate in rejected_quality[: max(target_count * 3, 24)]:
            if improved:
                break
            for victim in weak_selected:
                if int(candidate["idx"]) == int(victim["idx"]):
                    continue
                if (
                    candidate["support_score"] <= victim["support_score"] + 0.03
                    and candidate["overview_score"] <= victim["overview_score"] + 0.03
                ):
                    continue
                trial = [
                    entry
                    for entry in selected_entries
                    if int(entry["idx"]) != int(victim["idx"])
                ] + [candidate]
                if not _entries_respect_gap(trial, quality_gap):
                    continue
                if not _entries_respect_similarity(trial, quality_similarity_cap):
                    continue
                allowed, trial_family_stats = _quality_swap_preserves_diversity(
                    selected_entries,
                    trial,
                    quality_family_cap=quality_family_cap,
                )
                if not allowed:
                    continue
                trial_quality_objective = _quality_selection_objective(
                    trial,
                    total_frames=total_frames,
                    target_count=target_count,
                    diversity_weight=diversity_weight,
                    min_frame_gap=quality_gap,
                    similarity_cap=quality_similarity_cap,
                )
                quality_gain = _entry_quality_value(candidate) - _entry_quality_value(victim)
                required_gain = _family_duplicate_gain_requirement(selected_entries, victim, candidate)
                same_family = int(candidate.get("view_family", -1)) == int(victim.get("view_family", -1))
                if same_family:
                    if float(candidate.get("overview_score", 0.0)) < float(victim.get("overview_score", 0.0)) - 1e-6:
                        continue
                    if float(candidate.get("global_base_score", 0.0)) < float(victim.get("global_base_score", 0.0)) - 1e-6:
                        continue
                    if quality_gain < required_gain:
                        continue
                if trial_quality_objective > current_quality_objective + 1e-6:
                    selected_entries = sorted(trial, key=lambda x: int(x["idx"]))
                    fallback_notes.append(
                        f"quality swap {int(victim['idx'])}->{int(candidate['idx'])} with family cap {quality_family_cap}"
                    )
                    current_quality_objective = trial_quality_objective
                    current_family_stats = trial_family_stats
                    improved = True
                    break

    if len(selected_entries) < target_count:
        quality_fill_pool = [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            and _meets_support_quality_relaxed(entry)
        ]
        quality_fill_pool = _rank_fill_pool(quality_fill_pool, selected_entries)
        before_quality_fill = len(selected_entries)
        previous_entries = list(selected_entries)
        selected_entries = _greedy_phase_select(
            quality_fill_pool,
            count=target_count - len(selected_entries),
            current=selected_entries,
            total_frames=total_frames,
            target_count=target_count,
            diversity_weight=diversity_weight,
            min_frame_gap=quality_gap,
            similarity_cap=quality_similarity_cap,
            score_key="fill_score",
            max_per_family=quality_family_cap,
        )
        selected_entries = _tag_new_entries(previous_entries, selected_entries, "fill")
        if len(selected_entries) > before_quality_fill:
            fallback_notes.append(
                f"quality fill added {before_quality_fill}->{len(selected_entries)} with family cap {quality_family_cap}"
            )

    final_rejected_pool = _rank_fill_pool(
        [
            entry
            for entry in all_entries
            if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}
            and _meets_support_quality_relaxed(entry)
        ],
        selected_entries,
    )
    final_rejected_pool = _temporal_guard_entries(
        final_rejected_pool,
        front_target=max(target_count * 2, 12),
        min_gap=max(20, support_gap // 2),
        score_key="fill_score",
    )
    if len(selected_entries) < target_count and final_rejected_pool:
        rescue_entries = _build_rejected_rescue_set(
            rejected_entries=final_rejected_pool,
            target_count=target_count,
            total_frames=total_frames,
            diversity_weight=diversity_weight,
            min_frame_gap=support_gap,
            similarity_cap=support_similarity_cap,
            anchor_target=anchor_target,
            support_target=support_target,
        )
        if rescue_entries:
            rescue_quality_objective = _quality_selection_objective(
                rescue_entries,
                total_frames=total_frames,
                target_count=target_count,
                diversity_weight=diversity_weight,
                min_frame_gap=quality_gap,
                similarity_cap=quality_similarity_cap,
            )
            rescue_better = (
                len(rescue_entries) > len(selected_entries)
                or (
                    len(rescue_entries) == len(selected_entries)
                    and rescue_quality_objective > current_quality_objective + 1e-6
                )
            )
            if rescue_better:
                fallback_notes.append(
                    f"rejected-set rescue replaced selected {len(selected_entries)}->{len(rescue_entries)}"
                )
                selected_entries = rescue_entries
                current_quality_objective = rescue_quality_objective

    if len(selected_entries) < target_count:
        selected_entries, force_fill_notes = _force_fill_entries(
            selected_entries=selected_entries,
            all_entries=all_entries,
            target_count=target_count,
            total_frames=total_frames,
            diversity_weight=diversity_weight,
            support_gap=support_gap,
            support_similarity_cap=support_similarity_cap,
        )
        fallback_notes.extend(force_fill_notes)

    if len(selected_entries) >= target_count:
        selected_entries, final_polish_notes = _final_polish_selected_set(
            selected_entries=selected_entries,
            all_entries=all_entries,
            target_count=target_count,
            total_frames=total_frames,
            diversity_weight=diversity_weight,
            min_frame_gap=support_gap,
            similarity_cap=support_similarity_cap,
        )
        fallback_notes.extend(final_polish_notes)

    selected_entries = _assign_final_selection_roles(
        selected_entries,
        overview_target=overview_target,
        discovery_target=discovery_target,
        repair_target=repair_target,
    )
    current_objective = selection_objective(
        selected_entries,
        total_frames=total_frames,
        target_count=target_count,
        diversity_weight=diversity_weight,
        min_frame_gap=support_gap,
        similarity_cap=support_similarity_cap,
    )

    selected_entries = sorted(selected_entries, key=lambda x: int(x["idx"]))
    role_counts = _selection_role_counts(selected_entries)
    diagnostics = {
        "objective": float(current_objective),
        "active_gap": int(support_gap),
        "active_similarity_cap": float(support_similarity_cap),
        "anchor_target": int(anchor_target),
        "support_target": int(support_target),
        "overview_target": int(overview_target),
        "discovery_target": int(discovery_target),
        "repair_target": int(repair_target),
        "anchor_family_cap": int(anchor_family_cap),
        "discovery_family_cap": int(discovery_family_cap),
        "repair_family_cap": int(repair_family_cap),
        "support_family_cap": int(support_family_cap),
        "support_min_support_score": float(support_min_support_score),
        "support_min_context": float(support_min_context),
        "support_min_coverage": float(support_min_coverage),
        "support_relaxed_min_support_score": float(support_relaxed_min_support_score),
        "support_relaxed_min_context": float(support_relaxed_min_context),
        "support_relaxed_min_coverage": float(support_relaxed_min_coverage),
        "discovery_min_score": float(discovery_min_score),
        "repair_min_score": float(repair_min_score),
        "temporal_min_quality": float(temporal_min_quality),
        "view_family_count": int(len(families)),
        "view_family_similarity_threshold": float(family_similarity_threshold),
        "overview_pool_size": int(len(overview_pool)),
        "discovery_pool_size": int(len(discovery_pool)),
        "repair_pool_size": int(len(repair_pool)),
        "support_pool_size": int(len(support_pool)),
        "selected_role_counts": {key: int(value) for key, value in role_counts.items()},
        "fallback_notes": fallback_notes,
    }
    return selected_entries, diagnostics


def _build_preview_generator(cfg, device, preview_points_per_side, min_mask_region_area):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2(f"configs/sam2.1/{cfg['model_cfg']}", cfg["checkpoint"], device=device)
    gen = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=preview_points_per_side,
        pred_iou_thresh=cfg.get("pred_iou_thresh", 0.86),
        stability_score_thresh=cfg.get("stability_score_thresh", 0.92),
        stability_score_offset=cfg.get("stability_score_offset", 1.0),
        box_nms_thresh=cfg.get("box_nms_thresh", 0.7),
        crop_n_layers=cfg.get("crop_n_layers", 1),
        crop_n_points_downscale_factor=cfg.get("crop_n_points_downscale_factor", 2),
        min_mask_region_area=min_mask_region_area,
        output_mode=cfg.get("output_mode", "binary_mask"),
        points_per_batch=cfg.get("points_per_batch", 64),
    )
    return sam2, gen


def _save_contact_sheet(frames, entries, out_path, scene_rotation, cols=4, thumb_hw=(180, 320)):
    import cv2

    if not entries:
        return
    th, tw = thumb_hw
    tiles = []
    for entry in entries:
        idx = int(entry["idx"])
        frame = np.ascontiguousarray(np.rot90(frames[idx], scene_rotation))
        thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        label = (
            f"{idx} | ctx {entry['stats']['context_score']:.2f} | "
            f"cov {entry['stats']['useful_union_ratio']:.2f}"
        )
        cv2.putText(thumb, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(thumb)
    rows = []
    for i in range(0, len(tiles), cols):
        row_tiles = tiles[i : i + cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(cv2.hconcat(row_tiles))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.vconcat(rows))


def _save_similarity_matrix(entries, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["idx"] + [int(entry["idx"]) for entry in entries]
        writer.writerow(header)
        for row_entry in entries:
            row = [int(row_entry["idx"])]
            for col_entry in entries:
                row.append(f"{_pair_similarity(row_entry, col_entry):.4f}")
            writer.writerow(row)


def _entry_summary(entry):
    stats = entry["stats"]
    return {
        "idx": int(entry["idx"]),
        "selection_role": str(entry.get("selection_role", "")),
        "view_family": int(entry.get("view_family", -1)),
        "view_family_size": int(entry.get("view_family_size", 0)),
        "view_family_similarity": float(entry.get("view_family_similarity", 0.0)),
        "base_score": float(entry.get("global_base_score", entry.get("base_score", 0.0))),
        "overview_score": float(entry.get("overview_score", 0.0)),
        "discovery_score": float(entry.get("discovery_score", 0.0)),
        "repair_score": float(entry.get("repair_score", 0.0)),
        "support_score": float(entry.get("support_score", 0.0)),
        "is_overview_candidate": bool(entry.get("is_overview_candidate", False)),
        "is_discovery_candidate": bool(entry.get("is_discovery_candidate", False)),
        "is_repair_candidate": bool(entry.get("is_repair_candidate", False)),
        "is_support_candidate": bool(entry.get("is_support_candidate", False)),
        "appearance_jump": float(entry.get("appearance_jump", 0.0)),
        "transition_jump": float(entry.get("transition_jump", 0.0)),
        "discovery_novelty": float(entry.get("discovery_novelty", 0.0)),
        "mask_rise": float(entry.get("mask_rise", 0.0)),
        "mask_drop": float(entry.get("mask_drop", 0.0)),
        "coverage_rise": float(entry.get("coverage_rise", 0.0)),
        "coverage_drop": float(entry.get("coverage_drop", 0.0)),
        "utility_score": float(stats["utility_score"]),
        "context_score": float(stats["context_score"]),
        "useful_masks": int(stats["useful_masks"]),
        "grid_cells": int(stats["grid_cells"]),
        "frame_quality": float(stats["frame_quality"]),
        "useful_union_ratio": float(stats["useful_union_ratio"]),
        "texture_penalty": float(stats["texture_penalty"]),
        "overlap_penalty": float(stats["overlap_penalty"]),
        "concentration_penalty": float(stats["concentration_penalty"]),
    }


def _save_selection_diagnostics(
    scene_id,
    frames,
    all_entries,
    selected_entries,
    scene_rotation,
    diag_root,
    selection_diag,
    preview_config,
):
    diag_root = Path(diag_root)
    scene_dir = diag_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    selected_path = scene_dir / "selected_keyframes.jpg"
    rejected_path = scene_dir / "top_rejected_keyframes.jpg"
    similarity_path = scene_dir / "selected_similarity.csv"
    summary_path = scene_dir / "selection_summary.json"

    selected_ids = {int(entry["idx"]) for entry in selected_entries}
    rejected_entries = [entry for entry in all_entries if int(entry["idx"]) not in selected_ids]
    rejected_entries = sorted(
        rejected_entries,
        key=lambda x: (x.get("overview_score", 0.0), x.get("support_score", 0.0), x["stats"]["context_score"]),
        reverse=True,
    )
    rejected_entries = _temporal_guard_entries(
        rejected_entries,
        front_target=min(12, len(rejected_entries)),
        min_gap=max(24, int(len(frames) / max(len(selected_entries) * 4, 1))),
        score_key="support_score",
    )[: min(12, len(rejected_entries))]

    _save_contact_sheet(frames, selected_entries, selected_path, scene_rotation=scene_rotation)
    if rejected_entries:
        _save_contact_sheet(frames, rejected_entries, rejected_path, scene_rotation=scene_rotation)
    _save_similarity_matrix(selected_entries, similarity_path)

    summary = {
        "scene_id": scene_id,
        "scene_rotation": int(scene_rotation),
        "preview_config": preview_config,
        "selection": selection_diag,
        "selected": [_entry_summary(entry) for entry in selected_entries],
        "top_rejected": [_entry_summary(entry) for entry in rejected_entries],
        "candidate_count": int(len(all_entries)),
        "selected_indices": [int(entry["idx"]) for entry in selected_entries],
        "artifacts": {
            "selected_contact_sheet": str(selected_path),
            "rejected_contact_sheet": str(rejected_path),
            "similarity_csv": str(similarity_path),
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def _zscore(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


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

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
        return np.zeros(311, dtype=np.float32)

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

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
        return np.zeros((0, 311), dtype=np.float32)
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

    seed_candidates = _greedy_spaced_order(
        0.62 * quality + 0.38 * temporal_signal,
        min_gap=max(12, base_gap // 2),
    )
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
            + 0.35 * (float(np.mean([quality[idx] for idx in picked])) if picked else 0.0)
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

    return pool[:candidate_count]


def _mmr_select_candidates(
    candidate_ids, scores, descriptors, target_count, min_gap, diversity_weight=0.90
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
    while remaining and len(selected) < target_count:
        valid = [idx for idx in remaining if all(abs(idx - prev) >= min_gap for prev in selected)]
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


def select_keyframe_candidate_groups(
    frame_paths,
    strategy="quality_global",
    stride=10,
    n_keyframes=20,
    frames=None,
    top_k=8,
):
    import cv2

    num_frames = len(frame_paths)
    if num_frames == 0:
        return []
    if frames is None:
        frames = []
        for path in frame_paths:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to read frame for heuristic selection: {path}")
            frames.append(img)

    if strategy == "quality_global":
        scores = _quality_scores(frames)
        descriptors = _frame_descriptors(frames)
        return [_quality_global_candidates(scores, descriptors, n_keyframes, top_k)]

    # Fallback for unsupported strategies.
    if strategy == "stride":
        return [[idx] for idx in range(0, num_frames, stride)]
    if strategy == "uniform":
        indices = np.linspace(0, num_frames - 1, n_keyframes, dtype=int).tolist()
        return [[int(idx)] for idx in indices]
    return [[idx] for idx in range(num_frames)]


def select_keyframes_heuristic(
    frame_paths,
    frames,
    sam2_cfg,
    *,
    n_keyframes=20,
    stride=10,
    device="cuda",
    preview_candidates=8,
    refine_preview=True,
    scene_id="scene",
):
    if not frame_paths:
        return []

    candidate_groups = select_keyframe_candidate_groups(
        frame_paths,
        strategy="quality_global",
        stride=stride,
        n_keyframes=n_keyframes,
        frames=frames,
        top_k=preview_candidates,
    )
    if not candidate_groups:
        return []

    fallback = sorted(int(idx) for idx in candidate_groups[0][:n_keyframes])
    if not refine_preview:
        return fallback

    if not isinstance(sam2_cfg, dict):
        raise ValueError("sam2_cfg must be provided for heuristic preview refinement")

    selected = refine_keyframes_with_mask_preview(
        frames=frames,
        candidate_groups=candidate_groups,
        cfg=sam2_cfg,
        device=device,
        target_count=n_keyframes,
        scene_id=scene_id,
    )
    return sorted(int(idx) for idx in selected)


def refine_keyframes_with_mask_preview(
    frames,
    candidate_groups,
    cfg,
    device="cuda",
    target_count=None,
    scene_id="scene",
    diagnostics_root=None,
):
    import torch

    if not candidate_groups:
        return []

    preview_points_per_side = _parse_int(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_PREVIEW_POINTS_PER_SIDE"),
        max(8, int(cfg.get("points_per_side", 32)) // 2),
    )
    preview_top_k = _parse_int(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_PREVIEW_TOP_K_MASKS"),
        cfg.get("max_obj_ids", 50),
    )
    min_mask_region_area = _parse_int(
        os.environ.get("OBJECTX_SAM2_MIN_MASK_REGION_AREA"),
        cfg.get("min_mask_region_area", 200),
    )
    diversity_weight = _parse_float(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_PREVIEW_DIVERSITY_WEIGHT"),
        0.95,
    )
    descriptor_grid = _parse_int(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_PREVIEW_DESCRIPTOR_GRID"),
        6,
    )
    similarity_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_PREVIEW_SIMILARITY_CAP"),
        0.90,
    )
    target_count = int(target_count) if target_count is not None else len(candidate_groups)
    min_frame_gap = _parse_int(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_PREVIEW_MIN_FRAME_GAP"),
        max(18, int(len(frames) / max(target_count * 3, 1))),
    )

    print(
        "[SAM2] Refining keyframes with preview "
        f"(points_per_side={preview_points_per_side}, top_k_masks={preview_top_k}, "
        f"diversity_weight={diversity_weight}, similarity_cap={similarity_cap})"
    )

    sam2, gen = _build_preview_generator(cfg, device, preview_points_per_side, min_mask_region_area)
    candidate_entries = []
    try:
        for group_idx, group in enumerate(candidate_groups, start=1):
            scored = []
            group_size = len(group)
            for item_idx, idx in enumerate(group, start=1):
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    masks = gen.generate(frames[idx])
                masks = sorted(masks, key=lambda x: x["area"], reverse=True)
                preview = describe_preview_masks(
                    frame=frames[idx],
                    masks=masks,
                    preview_top_k=preview_top_k,
                    descriptor_grid=descriptor_grid,
                )
                scored.append({"idx": int(idx), "preview": preview})
                if group_size > 8 and (item_idx % 8 == 0 or item_idx == group_size):
                    print(
                        f"[SAM2] preview group {group_idx} progress {item_idx}/{group_size}",
                        flush=True,
                    )
            candidate_entries.append(scored)
    finally:
        del gen, sam2
        torch.cuda.empty_cache()

    preview_rotation_scores = np.zeros(4, dtype=np.float32)
    preview_rotation_counts = np.zeros(4, dtype=np.float32)
    for group in candidate_entries:
        for entry in group:
            for rot_entry in entry["preview"]["rotation_stats"]:
                rot_k = int(rot_entry["rotation"])
                preview_rotation_scores[rot_k] += float(rot_entry["utility_score"])
                preview_rotation_counts[rot_k] += 1.0
    preview_rotation_means = preview_rotation_scores / np.maximum(preview_rotation_counts, 1.0)
    scene_rotation = int(np.argmax(preview_rotation_means))
    print(
        "[SAM2] preview scene rotation "
        f"means={[round(float(x), 1) for x in preview_rotation_means.tolist()]} -> {scene_rotation}",
        flush=True,
    )

    for group_idx, group in enumerate(candidate_entries, start=1):
        for entry in group:
            entry["stats"] = entry["preview"]["rotation_stats"][scene_rotation]
        scored_txt = ", ".join(
            f"{entry['idx']}:{entry['stats']['mask_count']}m/{entry['stats']['useful_masks']}u/"
            f"{entry['stats']['grid_cells']}g/{int(entry['stats']['utility_score'])}s"
            for entry in group
        )
        print(f"[SAM2] preview group {group_idx}: {scored_txt}", flush=True)

    flat_entries = {}
    for group in candidate_entries:
        for entry in group:
            idx = int(entry["idx"])
            current = flat_entries.get(idx)
            if current is None or entry["stats"]["utility_score"] > current["stats"]["utility_score"]:
                flat_entries[idx] = entry

    all_entries = list(flat_entries.values())
    raw_scores = np.asarray([entry["stats"]["utility_score"] for entry in all_entries], dtype=np.float32)
    norm_scores = _normalize_scores(raw_scores)
    for entry, score in zip(all_entries, norm_scores):
        entry["global_base_score"] = float(score)

    selected_entries, selection_diag = optimize_keyframe_set(
        all_entries=all_entries,
        target_count=target_count,
        total_frames=len(frames),
        min_frame_gap=min_frame_gap,
        diversity_weight=diversity_weight,
        similarity_cap=similarity_cap,
    )
    selected = [int(entry["idx"]) for entry in selected_entries]
    print(
        "[SAM2] preview set selection "
        f"mode=overview_discovery_repair candidates={len(all_entries)} min_frame_gap={selection_diag['active_gap']} "
        f"similarity_cap={selection_diag['active_similarity_cap']:.2f} "
        f"overview={selection_diag['overview_target']} "
        f"discovery={selection_diag['discovery_target']} "
        f"repair={selection_diag['repair_target']} "
        f"families={selection_diag['view_family_count']} "
        f"family_caps={selection_diag['anchor_family_cap']}/{selection_diag['discovery_family_cap']}/{selection_diag['repair_family_cap']} "
        f"-> {selected} (objective={selection_diag['objective']:.3f})",
        flush=True,
    )
    if selection_diag["fallback_notes"]:
        for note in selection_diag["fallback_notes"]:
            print(f"[SAM2] preview fallback: {note}", flush=True)

    diag_root = diagnostics_root or os.environ.get(
        "OBJECTX_SEG_KEYFRAME_DIAG_ROOT",
        str(REPO_ROOT / "debug" / "keyframe_selection"),
    )
    summary_path = _save_selection_diagnostics(
        scene_id=scene_id,
        frames=frames,
        all_entries=all_entries,
        selected_entries=selected_entries,
        scene_rotation=scene_rotation,
        diag_root=diag_root,
        selection_diag=selection_diag,
        preview_config={
            "points_per_side": preview_points_per_side,
            "top_k_masks": preview_top_k,
            "descriptor_grid": descriptor_grid,
            "diversity_weight": diversity_weight,
            "min_frame_gap": min_frame_gap,
            "similarity_cap": similarity_cap,
        },
    )
    print(f"[SAM2] keyframe diagnostics: {summary_path}", flush=True)
    return selected
