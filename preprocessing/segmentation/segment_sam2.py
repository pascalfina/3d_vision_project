"""
SAM2: AutomaticMaskGenerator (grid) on keyframes + VideoPredictor propagation.
"""
import gc, os, re, shutil, tempfile, pickle, sys
import torch, numpy as np, cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM2_REPO = REPO_ROOT / "dependencies" / "sam2"
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.mask_utils import visualize_masks_on_frame

POPCOUNT_LUT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1).astype(np.uint8)


def ensure_sam2_postprocess_ready(device="cuda"):
    """Fail fast if the optional SAM2 post-processing extension is unavailable."""
    from sam2.utils.misc import get_connected_components

    requested_device = str(device).split(":", 1)[0]
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM2 post-processing requires CUDA, but torch.cuda.is_available() is False."
            )
        test_mask = torch.zeros((1, 1, 8, 8), dtype=torch.bool, device="cuda")
        test_mask[:, :, 2:4, 2:4] = True
        try:
            labels, areas = get_connected_components(test_mask)
        except Exception as e:
            raise RuntimeError(
                "SAM2 CUDA post-processing extension is not operational. "
                "Please rebuild sam2._C before running the pipeline."
            ) from e
        if labels.shape != test_mask.shape or areas.shape != test_mask.shape:
            raise RuntimeError(
                "SAM2 CUDA post-processing extension returned unexpected output shapes."
            )
        print("[SAM2] post-processing extension check: CUDA OK", flush=True)
    else:
        try:
            import sam2._C  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "SAM2 post-processing extension import failed even for non-CUDA mode."
            ) from e
        print("[SAM2] post-processing extension check: import OK", flush=True)


def _parse_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_int(value, default):
    if value is None:
        return int(default)
    return int(value)


def _parse_float(value, default):
    if value is None:
        return float(default)
    return float(value)


def segment_keyframes(frames, keyframe_idxs, cfg, device="cuda"):
    """Run grid 32x32 on each keyframe → list of SAM2 masks."""
    print("\n[SAM2] Step 1: Segmenting keyframes with grid")
    sam2 = build_sam2(cfg["model_cfg"], cfg["checkpoint"], device=device)
    points_per_side = _parse_int(
        os.environ.get("OBJECTX_SAM2_POINTS_PER_SIDE"),
        cfg.get("points_per_side", 32),
    )
    min_mask_region_area = _parse_int(
        os.environ.get("OBJECTX_SAM2_MIN_MASK_REGION_AREA"),
        cfg.get("min_mask_region_area", 200),
    )
    gen = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
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
    keyframe_masks = {}
    for idx in keyframe_idxs:
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            masks = gen.generate(frames[idx])
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        keyframe_masks[idx] = masks
        print(f"  Frame {idx:5d}: {len(masks):3d} màscares")
    del gen, sam2
    torch.cuda.empty_cache()
    return keyframe_masks


def refine_keyframes_with_mask_preview(
    frames,
    candidate_groups,
    cfg,
    device="cuda",
    target_count=None,
):
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
        0.85,
    )
    descriptor_grid = _parse_int(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_PREVIEW_DESCRIPTOR_GRID"),
        4,
    )

    print(
        "[SAM2] Refining keyframes with preview "
        f"(points_per_side={preview_points_per_side}, top_k_masks={preview_top_k}, "
        f"diversity_weight={diversity_weight})"
    )
    sam2 = build_sam2(cfg["model_cfg"], cfg["checkpoint"], device=device)
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

    def frame_preview_quality(frame):
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
        edge_balance = float(
            1.0 - (cell_edges.std() / max(cell_edges.mean(), 1e-4))
        ) if cell_edges.size else 0.0
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

    def score_preview_masks(frame, masks):
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
                "useful_novel_mass": 0.0,
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
            pred_iou = float(mask.get("predicted_iou", 0.0))
            stability = float(mask.get("stability_score", pred_iou))
            pred_ious.append(pred_iou)
            stability_scores.append(stability)
            seg = seg.astype(bool, copy=False)
            if union_mask is None:
                union_mask = np.zeros_like(seg, dtype=np.bool_)
            union_mask |= seg
            row_sums = seg.sum(axis=1, dtype=np.int64)
            if row_sums.sum() <= 0:
                continue
            h = int(seg.shape[0])
            w = int(seg.shape[1])
            bbox = mask.get("bbox") or [0, 0, w, h]
            x, y, bw, bh = [float(v) for v in bbox]
            bbox_fill = area / max(float(bw * bh), 1.0)
            shape_ratio = min(float(bw), float(bh)) / max(max(float(bw), float(bh)), 1.0)
            center_y_raw = (y + 0.5 * bh) / max(float(h), 1.0)
            center_x_raw = (x + 0.5 * bw) / max(float(w), 1.0)
            largest_area = max(largest_area, area)
            area_frac = area / max(frame_area, 1.0)
            area_fracs.append(area_frac)
            touch_margin = 8.0
            if area_frac < 0.0025:
                tiny_masks += 1
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
                    size_score = 0.40
                else:
                    size_score = 0.10

                fill_score = float(np.clip((bbox_fill - 0.18) / 0.62, 0.0, 1.0))
                shape_score = max(0.30, min(1.0, shape_ratio))
                quality_score = 0.55 * pred_iou + 0.45 * stability
                touch_score = 1.0 if touches == 0 else (0.82 if touches == 1 else 0.50)
                bottom_score = 0.72 if center_y >= 0.82 else 1.0
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
                novelty_weight = float(np.clip((novel_frac - 0.10) / 0.75, 0.0, 1.0))
                effective_object_score = object_score * (0.30 + 0.70 * novelty_weight)
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
                    stats["useful_masks"] += 1
                    stats["grid"][grid_y, grid_x] = True
                    stats["useful_hist"][grid_y, grid_x] += float(effective_object_score)
                    stats["useful_area_fracs"].append(area_frac)
                    useful_mass = area_frac * max(float(effective_object_score), 0.2)
                    stats["useful_area_mass"] += useful_mass
                    stats["useful_novel_mass"] += novel_area_frac
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
                    stats["weighted_utility"] += 0.18 * float(effective_object_score)
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
        scale_prob = scale_hist[scale_hist > 0]
        if scale_prob.size > 0:
            scale_prob = scale_prob / scale_prob.sum()
            scale_entropy = float(
                -(scale_prob * np.log(scale_prob + 1e-8)).sum()
                / np.log(max(len(scale_hist), 2))
            )
        else:
            scale_entropy = 0.0

        rotation_summaries = []
        for rot_k, stats in enumerate(rot_stats):
            rot_grid_cells = int(stats["grid"].sum())
            useful_masks = int(stats["useful_masks"])
            object_scores = sorted(stats["object_scores"], reverse=True)
            top_object_sum = float(sum(object_scores[:12]))
            top_object_mean = float(np.mean(object_scores[:8])) if object_scores else 0.0
            texture_count = len(stats["texture_areas"])
            texture_grid_cells = int(stats["texture_grid"].sum())
            if texture_count > 1:
                texture_areas = np.asarray(stats["texture_areas"], dtype=np.float32)
                texture_area_cv = float(texture_areas.std() / max(texture_areas.mean(), 1e-6))
            else:
                texture_area_cv = 1.0
            texture_penalty = float(
                np.clip((texture_count - 6) / 14.0, 0.0, 1.0)
                * (texture_grid_cells / 9.0)
                * np.clip((0.85 - texture_area_cv) / 0.85, 0.0, 1.0)
            )
            noise_penalty = float(
                np.clip(((mask_count - useful_masks) / max(mask_count, 1) - 0.45) / 0.40, 0.0, 1.0)
            )
            dominance_penalty = float(np.clip((dominance_ratio - 0.55) / 0.30, 0.0, 1.0))
            border_penalty = float(np.clip(stats["border_dominated"] / max(useful_masks, 1), 0.0, 1.0))
            bottom_penalty = float(
                np.clip(stats["bottom_useful_mass"] / max(stats["useful_area_mass"], 1e-6), 0.0, 1.0)
            ) if stats["useful_area_mass"] > 0.0 else 0.0
            useful_count_score = min(1.0, useful_masks / 12.0)
            coverage_score = rot_grid_cells / 9.0
            useful_hist = stats["useful_hist"].reshape(-1)
            useful_hist_sum = float(useful_hist.sum())
            if useful_hist_sum > 1e-6:
                useful_hist_prob = useful_hist[useful_hist > 0] / useful_hist_sum
                coverage_entropy = float(
                    -(useful_hist_prob * np.log(useful_hist_prob + 1e-8)).sum()
                    / np.log(max(useful_hist.size, 2))
                )
            else:
                coverage_entropy = 0.0
            useful_area_fracs = np.asarray(stats["useful_area_fracs"], dtype=np.float32)
            useful_union = stats["useful_union"]
            useful_union_ratio = (
                float(useful_union.sum() / max(frame_area, 1.0))
                if useful_union is not None
                else 0.0
            )
            overlap_penalty = float(
                np.clip(
                    1.0 - useful_union_ratio / max(float(useful_area_fracs.sum()), 1e-6),
                    0.0,
                    1.0,
                )
            ) if useful_area_fracs.size > 0 else 0.0
            if useful_area_fracs.size > 0:
                useful_scale_hist = np.histogram(
                    useful_area_fracs,
                    bins=np.asarray([0.0, 0.002, 0.01, 0.04, 0.18, 1.0], dtype=np.float32),
                )[0].astype(np.float32)
                useful_scale_prob = useful_scale_hist[useful_scale_hist > 0]
                if useful_scale_prob.size > 0:
                    useful_scale_prob = useful_scale_prob / useful_scale_prob.sum()
                    useful_scale_entropy = float(
                        -(useful_scale_prob * np.log(useful_scale_prob + 1e-8)).sum()
                        / np.log(max(len(useful_scale_hist), 2))
                    )
                else:
                    useful_scale_entropy = 0.0
            else:
                useful_scale_entropy = 0.0
            object_strength = min(1.0, top_object_sum / 4.0)
            union_mid_score = max(0.0, 1.0 - abs(union_ratio - 0.42) / 0.42)
            useful_union_mid_score = max(0.0, 1.0 - abs(useful_union_ratio - 0.36) / 0.36)
            quality_norm = float(np.clip((frame_quality_by_rot[rot_k] - 2.5) / 5.0, 0.0, 1.0))
            concentration_penalty = float(
                np.clip(
                    useful_hist.max() / max(useful_hist_sum, 1e-6) - 0.42,
                    0.0,
                    0.58,
                )
            ) if useful_hist_sum > 0.0 else 0.0
            context_score = (
                2.20 * useful_count_score
                + 2.00 * coverage_score
                + 1.70 * coverage_entropy
                + 1.25 * object_strength
                + 0.95 * scale_entropy
                + 0.85 * useful_scale_entropy
                + 0.70 * quality_norm
                + 0.35 * union_mid_score
                + 0.90 * useful_union_mid_score
                + 0.20 * float(stats["weighted_utility"])
                - 1.80 * texture_penalty
                - 1.65 * overlap_penalty
                - 1.45 * dominance_penalty
                - 1.10 * noise_penalty
                - 1.00 * bottom_penalty
                - 0.90 * concentration_penalty
                - 0.65 * border_penalty
                - 0.45 * tiny_mask_ratio
            )
            rot_score = 1000.0 * context_score
            occ_grid = np.zeros((descriptor_grid, descriptor_grid), dtype=np.float32)
            if union_mask is not None and union_mask.size > 0:
                rotated_union = np.ascontiguousarray(np.rot90(union_mask, rot_k))
                gh = np.linspace(0, rotated_union.shape[0], descriptor_grid + 1, dtype=int)
                gw = np.linspace(0, rotated_union.shape[1], descriptor_grid + 1, dtype=int)
                for gy in range(descriptor_grid):
                    for gx in range(descriptor_grid):
                        cell = rotated_union[gh[gy] : gh[gy + 1], gw[gx] : gw[gx + 1]]
                        if cell.size > 0:
                            occ_grid[gy, gx] = float(cell.mean())

            thumb = cv2.resize(rotated_thumbs[rot_k], (12, 12), interpolation=cv2.INTER_AREA)
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            bottom_heavy_masks = int(stats["bottom_heavy_masks"])
            useful_hist_grid = stats["useful_hist"].astype(np.float32)
            useful_hist_norm = useful_hist_grid.reshape(-1)
            useful_hist_norm_sum = float(useful_hist_norm.sum())
            if useful_hist_norm_sum > 1e-6:
                useful_hist_norm /= useful_hist_norm_sum
            descriptor = np.concatenate(
                [
                    (2.4 * occ_grid).reshape(-1),
                    (1.25 * useful_hist_norm).reshape(-1),
                    (0.45 * thumb_rgb).reshape(-1),
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
                            min(1.0, bottom_heavy_masks / max(mask_count, 1)),
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
            desc_norm = float(np.linalg.norm(descriptor))
            if desc_norm > 1e-6:
                descriptor /= desc_norm

            rotation_summaries.append(
                {
                    "mask_count": mask_count,
                    "useful_masks": useful_masks,
                    "weighted_utility": float(stats["weighted_utility"]),
                    "utility_score": float(rot_score),
                    "total_area": total_area,
                    "grid_cells": rot_grid_cells,
                    "union_ratio": union_ratio,
                    "useful_union_ratio": useful_union_ratio,
                    "frame_quality": float(frame_quality_by_rot[rot_k]),
                    "dominance_ratio": float(dominance_ratio),
                    "tiny_mask_ratio": tiny_mask_ratio,
                    "mean_pred_iou": mean_pred_iou,
                    "mean_stability": mean_stability,
                    "bottom_heavy_masks": bottom_heavy_masks,
                    "border_dominated": int(stats["border_dominated"]),
                    "texture_grid_cells": texture_grid_cells,
                    "texture_penalty": float(texture_penalty),
                    "overlap_penalty": float(overlap_penalty),
                    "concentration_penalty": float(concentration_penalty),
                    "scale_entropy": float(scale_entropy),
                    "useful_scale_entropy": float(useful_scale_entropy),
                    "coverage_entropy": float(coverage_entropy),
                    "scale_diversity": int(scale_diversity),
                    "top_object_sum": top_object_sum,
                    "context_score": float(context_score),
                    "rotation": int(rot_k),
                    "descriptor": descriptor,
                    "grid_vec": stats["grid"].astype(np.float32).reshape(-1),
                }
            )

        return {
            "best_rotation": int(np.argmax([entry["utility_score"] for entry in rotation_summaries])),
            "rotation_stats": rotation_summaries,
        }

    candidate_entries = []
    try:
        for group_idx, group in enumerate(candidate_groups, start=1):
            scored = []
            group_size = len(group)
            for item_idx, idx in enumerate(group, start=1):
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    masks = gen.generate(frames[idx])
                masks = sorted(masks, key=lambda x: x["area"], reverse=True)
                preview_stats = score_preview_masks(frames[idx], masks)
                scored.append({"idx": int(idx), "preview": preview_stats})
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
        print(
            f"[SAM2] preview group {group_idx}: {scored_txt}",
            flush=True,
        )
        raw_scores = np.asarray(
            [entry["stats"]["utility_score"] for entry in group],
            dtype=np.float32,
        )
        raw_min = float(raw_scores.min()) if raw_scores.size else 0.0
        raw_max = float(raw_scores.max()) if raw_scores.size else 0.0
        for entry, raw_score in zip(group, raw_scores):
            if raw_max - raw_min < 1e-6:
                entry["base_score"] = 1.0
            else:
                entry["base_score"] = float((raw_score - raw_min) / (raw_max - raw_min))

    target_count = int(target_count) if target_count is not None else len(candidate_entries)
    min_frame_gap = _parse_int(
        os.environ.get("OBJECTX_SAM2_KEYFRAME_PREVIEW_MIN_FRAME_GAP"),
        max(18, int(len(frames) / max(target_count * 3, 1))),
    )

    flat_entries = {}
    for group_idx, group in enumerate(candidate_entries):
        for entry in group:
            idx = int(entry["idx"])
            entry["group_idx"] = group_idx
            current = flat_entries.get(idx)
            if current is None or entry["stats"]["utility_score"] > current["stats"]["utility_score"]:
                flat_entries[idx] = entry

    all_entries = list(flat_entries.values())
    raw_scores = np.asarray(
        [entry["stats"]["utility_score"] for entry in all_entries],
        dtype=np.float32,
    )
    raw_min = float(raw_scores.min()) if raw_scores.size else 0.0
    raw_max = float(raw_scores.max()) if raw_scores.size else 0.0
    for entry, raw_score in zip(all_entries, raw_scores):
        if raw_max - raw_min < 1e-6:
            entry["global_base_score"] = 1.0
        else:
            entry["global_base_score"] = float((raw_score - raw_min) / (raw_max - raw_min))

    def mmr_score(entry, selected_entries):
        base = float(entry["global_base_score"])
        if not selected_entries:
            return base
        desc = entry["stats"]["descriptor"]
        sims = [
            float(np.clip(desc @ sel["stats"]["descriptor"], -1.0, 1.0))
            for sel in selected_entries
        ]
        max_sim = max(sims) if sims else 0.0
        min_gap = min(abs(int(entry["idx"]) - int(sel["idx"])) for sel in selected_entries)
        gap_bonus = min(1.0, float(min_gap) / max(float(min_frame_gap * 2), 1.0))
        overlap_penalty = max(0.0, max_sim - 0.86)
        return 0.55 * base + 1.55 * diversity_weight * (1.0 - max_sim) + 0.35 * gap_bonus - 0.90 * overlap_penalty

    active_gap = int(min_frame_gap)

    def selection_objective(entries):
        if not entries:
            return -1e9
        base_mean = float(np.mean([entry["global_base_score"] for entry in entries]))
        context_mean = float(np.mean([entry["stats"]["context_score"] for entry in entries]))
        grid_cover = float(
            np.max(np.stack([entry["stats"]["grid_vec"] for entry in entries], axis=0), axis=0).mean()
        )
        if len(entries) > 1:
            descs = np.stack([entry["stats"]["descriptor"] for entry in entries], axis=0)
            sims = np.clip(descs @ descs.T, -1.0, 1.0)
            iu = np.triu_indices(len(entries), k=1)
            diversity = float(np.mean(1.0 - sims[iu])) if iu[0].size > 0 else 0.0
            redundancy_penalty = float(np.mean(np.clip(sims[iu] - 0.88, 0.0, 1.0))) if iu[0].size > 0 else 0.0
            sorted_indices = np.sort([int(entry["idx"]) for entry in entries])
            gaps = np.diff(sorted_indices)
            gap_score = float(
                np.clip(gaps.mean() / max(float(len(frames) / max(target_count, 1)), 1.0), 0.0, 1.0)
            ) if gaps.size > 0 else 0.0
            min_gap_score = float(
                np.clip(gaps.min() / max(float(active_gap), 1.0), 0.0, 1.15)
            ) if gaps.size > 0 else 0.0
        else:
            diversity = 0.0
            redundancy_penalty = 0.0
            gap_score = 0.0
            min_gap_score = 0.0
        return (
            0.45 * base_mean
            + 1.85 * diversity_weight * diversity
            + 0.90 * grid_cover
            + 0.45 * context_mean
            + 0.35 * gap_score
            + 0.45 * min_gap_score
            - 1.10 * redundancy_penalty
        )

    def respects_gap(entries):
        idxs = sorted(int(entry["idx"]) for entry in entries)
        return all((b - a) >= active_gap for a, b in zip(idxs, idxs[1:]))

    selected_entries = []
    remaining = sorted(
        all_entries,
        key=lambda x: (
            x["global_base_score"],
            x["stats"]["useful_masks"],
            x["stats"]["grid_cells"],
            x["stats"]["frame_quality"],
        ),
        reverse=True,
    )

    while remaining and len(selected_entries) < target_count:
        valid = [
            entry
            for entry in remaining
            if all(abs(int(entry["idx"]) - int(sel["idx"])) >= min_frame_gap for sel in selected_entries)
        ]
        if not valid:
            break
        pool = valid
        best_entry = max(
            pool,
            key=lambda entry: (
                mmr_score(entry, selected_entries),
                entry["stats"]["useful_masks"],
                entry["stats"]["grid_cells"],
                entry["stats"]["frame_quality"],
            ),
        )
        selected_entries.append(best_entry)
        remaining = [entry for entry in remaining if int(entry["idx"]) != int(best_entry["idx"])]

    if len(selected_entries) < target_count:
        print(
            "[SAM2] preview set selection "
            f"could only fill {len(selected_entries)}/{target_count} with hard gap={min_frame_gap}; "
            "retrying with expanded spacing fallback",
            flush=True,
        )
        relaxed_gap = max(32, min_frame_gap - 12)
        active_gap = int(relaxed_gap)
        remaining = sorted(
            [entry for entry in all_entries if int(entry["idx"]) not in {int(sel["idx"]) for sel in selected_entries}],
            key=lambda x: (
                x["global_base_score"],
                x["stats"]["context_score"],
                x["stats"]["grid_cells"],
            ),
            reverse=True,
        )
        while remaining and len(selected_entries) < target_count:
            valid = [
                entry
                for entry in remaining
                if all(abs(int(entry["idx"]) - int(sel["idx"])) >= relaxed_gap for sel in selected_entries)
            ]
            if not valid:
                break
            best_entry = max(
                valid,
                key=lambda entry: (
                    mmr_score(entry, selected_entries),
                    entry["stats"]["context_score"],
                    entry["stats"]["grid_cells"],
                ),
            )
            selected_entries.append(best_entry)
            remaining = [entry for entry in remaining if int(entry["idx"]) != int(best_entry["idx"])]

    selected_ids = {int(entry["idx"]) for entry in selected_entries}
    candidate_subset = sorted(
        all_entries,
        key=lambda x: (
            x["global_base_score"],
            x["stats"]["context_score"],
            x["stats"]["useful_masks"],
            x["stats"]["grid_cells"],
        ),
        reverse=True,
    )[: max(24, target_count * 4)]
    current_objective = selection_objective(selected_entries)
    for _ in range(3):
        best_swap = None
        for slot_idx, existing in enumerate(selected_entries):
            kept = [entry for i, entry in enumerate(selected_entries) if i != slot_idx]
            for candidate in candidate_subset:
                cand_idx = int(candidate["idx"])
                if cand_idx == int(existing["idx"]) or cand_idx in selected_ids:
                    continue
                trial_entries = kept + [candidate]
                if not respects_gap(trial_entries):
                    continue
                trial_objective = selection_objective(trial_entries)
                if trial_objective > current_objective + 1e-4:
                    best_swap = (slot_idx, candidate, cand_idx, trial_objective)
                    current_objective = trial_objective
        if best_swap is None:
            break
        slot_idx, candidate, cand_idx, trial_objective = best_swap
        removed_idx = int(selected_entries[slot_idx]["idx"])
        selected_ids.remove(removed_idx)
        selected_entries[slot_idx] = candidate
        selected_ids.add(cand_idx)
        current_objective = trial_objective

    selected_entries = sorted(selected_entries, key=lambda x: int(x["idx"]))
    selected = [int(entry["idx"]) for entry in selected_entries]
    best_objective = float(selection_objective(selected_entries))
    print(
        "[SAM2] preview set selection "
        f"mode=global_greedy candidates={len(all_entries)} min_frame_gap={active_gap} "
        f"-> {selected} (objective={best_objective:.3f})",
        flush=True,
    )
    return selected


def _parse_scan_frame_idx(frame_path):
    """Extract numeric frame index from filename (e.g. frame-000042 → 42)."""
    m = re.search(r"frame-(\d+)", Path(frame_path).stem)
    if m is None:
        raise ValueError(f"Cannot extract frame idx from {frame_path}")
    return int(m.group(1))


def _frame_key(frame_idx):
    return f"{int(frame_idx):06d}"


def _parse_dtype(value, default):
    dtype_name = str(value or default).strip().lower()
    mapping = {
        "float16": np.float16,
        "fp16": np.float16,
        "float32": np.float32,
        "fp32": np.float32,
        "uint16": np.uint16,
        "int32": np.int32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{value}'")
    return mapping[dtype_name]


def _prepare_work_dir(output_dir, scan_id, cfg):
    base_dir = (
        os.environ.get("OBJECTX_SAM2_TMP_ROOT")
        or cfg.get("tmp_root")
        or output_dir
    )
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"sam2_{scan_id}_", dir=str(base_path)))


def _pack_mask(mask_bin):
    return np.packbits(mask_bin.reshape(-1), bitorder="little")


def _packed_intersection_count(a, b):
    return int(POPCOUNT_LUT[np.bitwise_and(a, b)].sum(dtype=np.int64))


def _count_tracks(keyframe_masks, max_obj):
    return sum(min(len(masks), max_obj) for masks in keyframe_masks.values())


def _mask_bbox_stats(mask_entry):
    bbox = mask_entry.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x, y, w, h = [float(v) for v in bbox]
        return x, y, w, h
    seg = mask_entry["segmentation"].astype(bool, copy=False)
    ys, xs = np.nonzero(seg)
    if ys.size == 0 or xs.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)


def _score_propagation_mask(mask_entry, frame_shape):
    h, w = frame_shape
    frame_area = max(float(h * w), 1.0)
    area = float(mask_entry.get("area", 0.0))
    pred_iou = float(mask_entry.get("predicted_iou", 0.0))
    stability = float(mask_entry.get("stability_score", pred_iou))
    x, y, bw, bh = _mask_bbox_stats(mask_entry)
    bbox_area = max(float(bw * bh), 1.0)
    bbox_fill = area / bbox_area
    area_frac = area / frame_area
    shape_ratio = min(float(bw), float(bh)) / max(max(float(bw), float(bh)), 1.0) if bw > 0 and bh > 0 else 0.0
    center_x = (x + 0.5 * bw) / max(float(w), 1.0)
    center_y = (y + 0.5 * bh) / max(float(h), 1.0)
    margin = 8.0
    touches = int(x <= margin) + int(y <= margin) + int((x + bw) >= (w - margin)) + int((y + bh) >= (h - margin))

    if area_frac < 0.0010:
        size_score = 0.08
    elif area_frac < 0.0025:
        size_score = 0.35
    elif area_frac < 0.015:
        size_score = 1.00
    elif area_frac < 0.08:
        size_score = 0.88
    elif area_frac < 0.25:
        size_score = 0.55
    else:
        size_score = 0.22

    quality = 0.52 * pred_iou + 0.48 * stability
    touch_penalty = 0.18 * max(0, touches - 1)
    bottom_penalty = 0.18 if center_y >= 0.82 and area_frac < 0.02 else 0.0
    dominant_penalty = 0.16 if area_frac > 0.22 and bbox_fill < 0.30 and touches >= 2 else 0.0
    score = (
        1.10 * quality
        + 0.42 * bbox_fill
        + 0.34 * size_score
        + 0.18 * max(0.0, shape_ratio)
        - touch_penalty
        - bottom_penalty
        - dominant_penalty
    )

    return {
        "area": area,
        "area_frac": area_frac,
        "pred_iou": pred_iou,
        "stability": stability,
        "bbox_fill": bbox_fill,
        "shape_ratio": shape_ratio,
        "center_x": center_x,
        "center_y": center_y,
        "touches": touches,
        "score": float(score),
    }


def _prepare_keyframe_masks_for_propagation(kf_idx, masks, max_obj, frame_shape):
    candidate_multiplier = _parse_int(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_CANDIDATE_MULTIPLIER"),
        2,
    )
    hard_filter_enabled = _parse_bool(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_HARD_FILTER_ENABLED"),
        False,
    )
    min_keep = _parse_int(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_MIN_KEEP"),
        max(12, min(24, max_obj // 2 if max_obj > 0 else 12)),
    )
    min_score = _parse_float(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_MIN_SCORE"),
        0.90,
    )
    min_pred_iou = _parse_float(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_MIN_PRED_IOU"),
        0.72,
    )
    min_stability = _parse_float(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_MIN_STABILITY"),
        0.82,
    )
    tiny_area_frac = _parse_float(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_TINY_AREA_FRAC"),
        0.0012,
    )
    dedupe_enabled = _parse_bool(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_DEDUPE_ENABLED"),
        False,
    )
    dedupe_iou = _parse_float(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_DEDUPE_IOU"),
        0.82,
    )
    dedupe_containment = _parse_float(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_DEDUPE_CONTAINMENT"),
        0.94,
    )
    replace_margin = _parse_float(
        os.environ.get("OBJECTX_SAM2_PROP_MASK_REPLACE_MARGIN"),
        0.08,
    )

    candidate_limit = min(len(masks), max(max_obj, max_obj * candidate_multiplier))
    candidate_masks = masks[:candidate_limit]
    scored = []
    hard_filtered = 0
    hard_filter_reasons = {
        "tiny_weak": 0,
        "border_dominant": 0,
        "bottom_sliver": 0,
        "low_score": 0,
    }

    for mask_entry in candidate_masks:
        stats = _score_propagation_mask(mask_entry, frame_shape)
        reject_reason = None
        if hard_filter_enabled and (
            stats["area_frac"] < tiny_area_frac
            and (
                stats["touches"] >= 1
                or stats["pred_iou"] < min_pred_iou
                or stats["stability"] < min_stability
            )
        ):
            reject_reason = "tiny_weak"
        elif hard_filter_enabled and (
            stats["area_frac"] > 0.18
            and stats["touches"] >= 2
            and stats["bbox_fill"] < 0.28
            and stats["pred_iou"] < 0.84
        ):
            reject_reason = "border_dominant"
        elif hard_filter_enabled and (
            stats["center_y"] >= 0.86
            and stats["area_frac"] < 0.015
            and stats["touches"] >= 1
            and stats["pred_iou"] < 0.84
        ):
            reject_reason = "bottom_sliver"
        elif hard_filter_enabled and (
            stats["score"] < min_score
            and stats["area_frac"] < 0.02
            and stats["pred_iou"] < 0.88
        ):
            reject_reason = "low_score"

        if reject_reason is not None:
            hard_filtered += 1
            hard_filter_reasons[reject_reason] += 1
            continue
        scored.append((mask_entry, stats))

    scored.sort(
        key=lambda item: (
            item[1]["score"],
            item[1]["pred_iou"],
            item[1]["stability"],
            item[1]["bbox_fill"],
            item[1]["area"],
        ),
        reverse=True,
    )

    kept = []
    deduped = 0
    if dedupe_enabled:
        for mask_entry, stats in scored:
            seg = mask_entry["segmentation"].astype(bool, copy=False)
            duplicate_idx = None
            replace_existing = False
            for idx, kept_entry in enumerate(kept):
                kept_seg = kept_entry["segmentation"]
                intersection = int(np.count_nonzero(seg & kept_seg))
                if intersection == 0:
                    continue
                union = stats["area"] + kept_entry["stats"]["area"] - intersection
                if union <= 0:
                    continue
                iou = float(intersection / union)
                containment = float(intersection / max(1.0, min(stats["area"], kept_entry["stats"]["area"])))
                if iou >= dedupe_iou or containment >= dedupe_containment:
                    duplicate_idx = idx
                    replace_existing = (
                        stats["score"] > kept_entry["stats"]["score"] + replace_margin
                        and stats["pred_iou"] >= kept_entry["stats"]["pred_iou"] - 0.02
                    )
                    break
            if duplicate_idx is None:
                kept.append(
                    {
                        "mask": mask_entry,
                        "stats": stats,
                        "segmentation": seg,
                    }
                )
            elif replace_existing:
                kept[duplicate_idx] = {
                    "mask": mask_entry,
                    "stats": stats,
                    "segmentation": seg,
                }
                deduped += 1
            else:
                deduped += 1
    else:
        kept = [
            {
                "mask": mask_entry,
                "stats": stats,
                "segmentation": mask_entry["segmentation"].astype(bool, copy=False),
            }
            for mask_entry, stats in scored
        ]

    kept.sort(
        key=lambda item: (
            item["stats"]["score"],
            item["stats"]["pred_iou"],
            item["stats"]["stability"],
            item["stats"]["area"],
        ),
        reverse=True,
    )

    selected = kept[:max_obj]

    prepared_masks = [item["mask"] for item in selected]
    print(
        "[SAM2] keyframe prefilter "
        f"{int(kf_idx)}: raw={len(masks)} "
        f"candidate_pool={candidate_limit} "
        f"hard_filter_enabled={int(hard_filter_enabled)} "
        f"hard_filtered={hard_filtered} "
        f"dedupe_enabled={int(dedupe_enabled)} "
        f"deduped={deduped} "
        f"selected={len(prepared_masks)} "
        f"reasons={hard_filter_reasons}",
        flush=True,
    )
    return prepared_masks


def _run_chunk_propagation(
    predictor,
    tmp_dir,
    chunk_masks,
    kf_idx,
    num_frames,
    device,
    offload_video_to_cpu,
    offload_state_to_cpu,
    async_loading_frames,
    on_prob,
):
    seen_flags = np.zeros((len(chunk_masks), num_frames), dtype=np.bool_)

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(tmp_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        for local_oid, mask_entry in enumerate(chunk_masks):
            predictor.add_new_mask(
                state,
                frame_idx=int(kf_idx),
                obj_id=local_oid,
                mask=mask_entry["segmentation"].astype(np.uint8),
            )

        for reverse in (False, True):
            for fidx, oids, logits in predictor.propagate_in_video(state, reverse=reverse):
                for oid, logit in zip(oids, logits):
                    oid = int(oid)
                    fidx = int(fidx)
                    if seen_flags[oid, fidx]:
                        continue
                    prob_map = torch.sigmoid(logit[0]).float().cpu().numpy()
                    on_prob(oid, fidx, prob_map)
                    seen_flags[oid, fidx] = True

    predictor.reset_state(state)
    del state
    torch.cuda.empty_cache()
    return seen_flags


def _merge_packed_tracks(
    track_masks,
    track_seen,
    track_areas,
    track_key_positions,
    iou_threshold,
    relaxed_iou_threshold,
    containment_threshold,
    peak_iou_threshold,
    same_keyframe_containment_threshold,
    same_keyframe_peak_iou_threshold,
    fragment_track_max_frames,
    fragment_containment_threshold,
    fragment_peak_iou_threshold,
    fragment_overlap_fraction,
    short_track_area_ratio_cap,
    max_keyframe_hops,
    min_shared_frames,
    area_ratio_cap,
):
    num_tracks = int(track_seen.shape[0])
    parent = list(range(num_tracks))
    peak_areas = track_areas.max(axis=1).astype(np.float32, copy=False)
    track_lengths = track_seen.sum(axis=1).astype(np.int32, copy=False)
    compared_pairs = 0
    skipped_same_keyframe = 0
    compared_same_keyframe = 0
    skipped_far_keyframe = 0
    skipped_area_ratio = 0
    skipped_shared_frames = 0
    direct_iou_merges = 0
    containment_merges = 0
    same_keyframe_merges = 0
    fragment_merges = 0

    valid_key_positions = track_key_positions[track_key_positions >= 0]
    num_keyframes = int(np.unique(valid_key_positions).size) if valid_key_positions.size else 0
    effective_max_keyframe_hops = int(max_keyframe_hops)
    if effective_max_keyframe_hops >= 0 and num_keyframes >= 8:
        effective_max_keyframe_hops = max(effective_max_keyframe_hops, min(4, max(0, num_keyframes - 1)))
    effective_area_ratio_cap = float(area_ratio_cap)
    if effective_area_ratio_cap > 0.0 and num_keyframes >= 8:
        effective_area_ratio_cap = max(effective_area_ratio_cap, 16.0)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    print(
        "[SAM2] merging "
        f"{num_tracks} tracks with IoU threshold {iou_threshold} "
        f"(relaxed_iou_threshold={relaxed_iou_threshold}, "
        f"containment_threshold={containment_threshold}, "
        f"peak_iou_threshold={peak_iou_threshold}, "
        f"same_keyframe_containment_threshold={same_keyframe_containment_threshold}, "
        f"same_keyframe_peak_iou_threshold={same_keyframe_peak_iou_threshold}, "
        f"fragment_track_max_frames={fragment_track_max_frames}, "
        f"fragment_containment_threshold={fragment_containment_threshold}, "
        f"fragment_peak_iou_threshold={fragment_peak_iou_threshold}, "
        f"fragment_overlap_fraction={fragment_overlap_fraction}, "
        f"short_track_area_ratio_cap={short_track_area_ratio_cap}, "
        f"max_keyframe_hops={effective_max_keyframe_hops}, "
        f"min_shared_frames={min_shared_frames}, "
        f"area_ratio_cap={effective_area_ratio_cap})",
        flush=True,
    )
    for i in range(num_tracks):
        if i == 0 or (i + 1) % 8 == 0 or (i + 1) == num_tracks:
            print(f"[SAM2] merge progress {i + 1}/{num_tracks}", flush=True)
        key_pos_i = int(track_key_positions[i])
        area_i = max(float(peak_areas[i]), 1.0)
        len_i = max(int(track_lengths[i]), 1)
        for j in range(i + 1, num_tracks):
            key_pos_j = int(track_key_positions[j])
            same_keyframe = key_pos_i == key_pos_j
            if effective_max_keyframe_hops >= 0 and abs(key_pos_i - key_pos_j) > effective_max_keyframe_hops:
                skipped_far_keyframe += 1
                continue
            len_j = max(int(track_lengths[j]), 1)
            short_pair = min(len_i, len_j) <= int(fragment_track_max_frames)
            shared = np.flatnonzero(track_seen[i] & track_seen[j])
            overlap_fraction = float(shared.size / max(1, min(len_i, len_j)))
            required_shared_frames = int(min_shared_frames)
            if same_keyframe:
                required_shared_frames = max(1, int(min_shared_frames) - 1)
            elif short_pair and overlap_fraction >= float(fragment_overlap_fraction):
                required_shared_frames = max(1, int(min_shared_frames) - 1)
            if shared.size < required_shared_frames:
                if same_keyframe:
                    skipped_same_keyframe += 1
                else:
                    skipped_shared_frames += 1
                continue
            area_j = max(float(peak_areas[j]), 1.0)
            pair_area_ratio_cap = float(effective_area_ratio_cap)
            if short_pair:
                pair_area_ratio_cap = max(pair_area_ratio_cap, float(short_track_area_ratio_cap))
            if pair_area_ratio_cap > 0.0 and (max(area_i, area_j) / min(area_i, area_j)) > pair_area_ratio_cap:
                skipped_area_ratio += 1
                continue
            compared_pairs += 1
            if same_keyframe:
                compared_same_keyframe += 1

            packed_i = track_masks[i, shared]
            packed_j = track_masks[j, shared]
            intersections = POPCOUNT_LUT[np.bitwise_and(packed_i, packed_j)].sum(
                axis=1, dtype=np.int64
            )
            union_areas = (
                track_areas[i, shared].astype(np.int64)
                + track_areas[j, shared].astype(np.int64)
                - intersections
            )
            min_areas = np.minimum(
                track_areas[i, shared].astype(np.int64),
                track_areas[j, shared].astype(np.int64),
            )
            valid = union_areas > 0
            if not np.any(valid):
                continue

            iou_vals = intersections[valid] / union_areas[valid]
            containment_vals = intersections[valid] / np.maximum(min_areas[valid], 1)
            mean_iou = float(np.mean(iou_vals))
            weighted_iou = float(intersections[valid].sum() / union_areas[valid].sum())
            weighted_containment = float(intersections[valid].sum() / np.maximum(min_areas[valid].sum(), 1))
            topk = min(3, int(iou_vals.size))
            top_iou = float(np.mean(np.sort(iou_vals)[-topk:])) if topk > 0 else 0.0
            top_containment = float(np.mean(np.sort(containment_vals)[-topk:])) if topk > 0 else 0.0
            peak_iou = float(np.max(iou_vals)) if iou_vals.size else 0.0
            peak_containment = float(np.max(containment_vals)) if containment_vals.size else 0.0

            merge_reason = None
            if mean_iou >= iou_threshold or weighted_iou >= iou_threshold:
                merge_reason = "direct_iou"
            elif same_keyframe:
                if (
                    (weighted_containment >= same_keyframe_containment_threshold or peak_containment >= same_keyframe_containment_threshold)
                    and (top_iou >= relaxed_iou_threshold or peak_iou >= same_keyframe_peak_iou_threshold)
                ):
                    merge_reason = "same_keyframe"
            elif short_pair:
                if (
                    overlap_fraction >= fragment_overlap_fraction
                    and (weighted_containment >= fragment_containment_threshold or top_containment >= fragment_containment_threshold or peak_containment >= fragment_containment_threshold + 0.06)
                    and (top_iou >= relaxed_iou_threshold or peak_iou >= fragment_peak_iou_threshold)
                ):
                    merge_reason = "fragment"
            elif (
                (weighted_containment >= containment_threshold or top_containment >= containment_threshold)
                and (top_iou >= relaxed_iou_threshold or peak_iou >= peak_iou_threshold)
            ):
                merge_reason = "containment"

            if merge_reason is not None:
                union(i, j)
                if merge_reason == "direct_iou":
                    direct_iou_merges += 1
                elif merge_reason == "same_keyframe":
                    same_keyframe_merges += 1
                elif merge_reason == "fragment":
                    fragment_merges += 1
                else:
                    containment_merges += 1

    groups = {}
    for orig_idx in range(num_tracks):
        root = find(orig_idx)
        groups.setdefault(root, []).append(orig_idx)

    id_mapping = {}
    orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
    for new_id, orig_indices in enumerate(groups.values(), start=1):
        orig_ids = [idx + 1 for idx in orig_indices]
        id_mapping[new_id] = orig_ids
        for orig_id in orig_ids:
            orig_to_new[orig_id] = new_id

    print(
        "[SAM2] merge candidate summary: "
        f"compared={compared_pairs}, "
        f"same_keyframe_skipped={skipped_same_keyframe}, "
        f"same_keyframe_compared={compared_same_keyframe}, "
        f"far_keyframe_skipped={skipped_far_keyframe}, "
        f"shared_frames_skipped={skipped_shared_frames}, "
        f"area_ratio_skipped={skipped_area_ratio}, "
        f"direct_iou_merges={direct_iou_merges}, "
        f"containment_merges={containment_merges}, "
        f"same_keyframe_merges={same_keyframe_merges}, "
        f"fragment_merges={fragment_merges}",
        flush=True,
    )
    print(f"[SAM2] {num_tracks} tracks -> {len(id_mapping)} merged objects")
    return id_mapping, orig_to_new


def _compute_group_masks_for_frames(track_masks, track_areas, orig_indices, frames):
    frames = np.asarray(frames, dtype=np.int64)
    if frames.size == 0:
        return np.zeros((0, track_masks.shape[2]), dtype=np.uint8), np.zeros(0, dtype=np.int64)
    if len(orig_indices) == 1:
        idx = int(orig_indices[0])
        packed = np.asarray(track_masks[idx, frames], dtype=np.uint8)
        areas = track_areas[idx, frames].astype(np.int64, copy=False)
        return packed, areas
    packed = np.bitwise_or.reduce(track_masks[orig_indices][:, frames], axis=0)
    areas = POPCOUNT_LUT[packed].sum(axis=1, dtype=np.int64)
    return packed, areas


def _compute_group_mask_for_frame(track_masks, track_areas, orig_indices, frame_idx, cache=None):
    cache_key = None
    if cache is not None:
        cache_key = (tuple(orig_indices), int(frame_idx))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    packed, areas = _compute_group_masks_for_frames(
        track_masks=track_masks,
        track_areas=track_areas,
        orig_indices=orig_indices,
        frames=np.asarray([int(frame_idx)], dtype=np.int64),
    )
    result = (packed[0], int(areas[0]))
    if cache is not None:
        cache[cache_key] = result
    return result


def _compute_temporal_boundary_similarity(
    track_masks,
    track_areas,
    source_meta,
    target_meta,
    max_gap,
    boundary_window,
    frame_mask_cache=None,
):
    source_frames = source_meta["frames"]
    target_frames = target_meta["frames"]
    if source_frames.size == 0 or target_frames.size == 0:
        return None

    if source_frames[-1] < target_frames[0]:
        gap = int(target_frames[0] - source_frames[-1] - 1)
        if gap > max_gap:
            return None
        source_candidates = source_frames[max(0, source_frames.size - boundary_window) :]
        target_candidates = target_frames[:boundary_window]
    elif target_frames[-1] < source_frames[0]:
        gap = int(source_frames[0] - target_frames[-1] - 1)
        if gap > max_gap:
            return None
        source_candidates = source_frames[:boundary_window]
        target_candidates = target_frames[max(0, target_frames.size - boundary_window) :]
    else:
        return None

    best = None
    for source_frame in source_candidates:
        source_mask, source_area = _compute_group_mask_for_frame(
            track_masks=track_masks,
            track_areas=track_areas,
            orig_indices=source_meta["orig_indices"],
            frame_idx=int(source_frame),
            cache=frame_mask_cache,
        )
        for target_frame in target_candidates:
            target_mask, target_area = _compute_group_mask_for_frame(
                track_masks=track_masks,
                track_areas=track_areas,
                orig_indices=target_meta["orig_indices"],
                frame_idx=int(target_frame),
                cache=frame_mask_cache,
            )
            intersection = _packed_intersection_count(source_mask, target_mask)
            union_area = int(source_area) + int(target_area) - int(intersection)
            if union_area <= 0:
                continue
            min_area = max(1, min(int(source_area), int(target_area)))
            iou = float(intersection / union_area)
            containment = float(intersection / min_area)
            score = containment + 0.65 * iou - 0.03 * gap
            if best is None or score > best["score"]:
                best = {
                    "gap": int(gap),
                    "iou": iou,
                    "containment": containment,
                    "score": float(score),
                    "source_frame": int(source_frame),
                    "target_frame": int(target_frame),
                }
    return best


def _collect_group_metadata(id_mapping, track_masks, track_seen, track_areas, track_key_positions=None):
    metadata = {}
    for group_id, orig_ids in id_mapping.items():
        orig_indices = [orig_id - 1 for orig_id in orig_ids]
        group_seen = np.any(track_seen[orig_indices], axis=0)
        frames = np.flatnonzero(group_seen)
        if frames.size == 0:
            continue
        _, areas = _compute_group_masks_for_frames(
            track_masks=track_masks,
            track_areas=track_areas,
            orig_indices=orig_indices,
            frames=frames,
        )
        if track_key_positions is not None:
            key_positions = [
                int(track_key_positions[idx])
                for idx in orig_indices
                if int(track_key_positions[idx]) >= 0
            ]
        else:
            key_positions = []
        metadata[int(group_id)] = {
            "group_id": int(group_id),
            "orig_ids": list(orig_ids),
            "orig_indices": orig_indices,
            "frames": frames,
            "n_frames": int(frames.size),
            "frame_span": int(frames[-1] - frames[0] + 1),
            "area_mean": float(np.mean(areas)),
            "area_max": float(np.max(areas)),
            "orig_track_count": int(len(orig_indices)),
            "key_positions": sorted(set(key_positions)),
            "keyframe_support_count": int(len(set(key_positions))),
        }
    return metadata


def _rebuild_group_mapping_from_parent(id_mapping, parent, num_tracks):
    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    merged_groups = {}
    for group_id, orig_ids in id_mapping.items():
        root = find(int(group_id))
        merged_groups.setdefault(root, []).extend(orig_ids)

    new_id_mapping = {}
    orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
    ordered_groups = sorted(
        (sorted(orig_ids) for orig_ids in merged_groups.values()),
        key=lambda ids: ids[0],
    )
    for new_id, orig_ids in enumerate(ordered_groups, start=1):
        new_id_mapping[new_id] = orig_ids
        for orig_id in orig_ids:
            orig_to_new[int(orig_id)] = new_id
    return new_id_mapping, orig_to_new


def _cleanup_fragment_groups(
    id_mapping,
    track_masks,
    track_seen,
    track_areas,
    cleanup_enabled,
    cleanup_max_frames,
    cleanup_area_mean_cap,
    cleanup_min_shared_frames,
    cleanup_overlap_fraction,
    cleanup_containment_threshold,
    cleanup_peak_iou_threshold,
    cleanup_area_ratio_cap,
    cleanup_target_length_ratio,
    cleanup_target_area_ratio,
    cleanup_temporal_gap_max,
    cleanup_temporal_boundary_window,
    cleanup_temporal_containment_threshold,
    cleanup_temporal_peak_iou_threshold,
):
    num_tracks = int(track_seen.shape[0])
    if not cleanup_enabled or len(id_mapping) <= 1:
        orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
        for new_id, orig_ids in id_mapping.items():
            for orig_id in orig_ids:
                orig_to_new[int(orig_id)] = int(new_id)
        return id_mapping, orig_to_new

    metadata = _collect_group_metadata(id_mapping, track_masks, track_seen, track_areas)
    if not metadata:
        orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
        return {}, orig_to_new

    parent = {int(group_id): int(group_id) for group_id in metadata}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    fragment_candidates = [
        group_id
        for group_id, meta in metadata.items()
        if meta["n_frames"] <= cleanup_max_frames and meta["area_mean"] <= cleanup_area_mean_cap
    ]
    fragment_candidates.sort(
        key=lambda group_id: (
            metadata[group_id]["n_frames"],
            metadata[group_id]["area_mean"],
            metadata[group_id]["area_max"],
        )
    )

    cleanup_compared = 0
    cleanup_merges = 0
    cleanup_shared_skipped = 0
    cleanup_overlap_skipped = 0
    cleanup_area_ratio_skipped = 0
    cleanup_strength_skipped = 0
    cleanup_temporal_gap_skipped = 0
    cleanup_temporal_compared = 0
    temporal_cleanup_merges = 0
    fragment_mask_cache = {}
    frame_mask_cache = {}

    print(
        "[SAM2] fragment cleanup "
        f"enabled={int(cleanup_enabled)} "
        f"(max_frames={cleanup_max_frames}, area_mean_cap={cleanup_area_mean_cap}, "
        f"min_shared_frames={cleanup_min_shared_frames}, overlap_fraction={cleanup_overlap_fraction}, "
        f"containment_threshold={cleanup_containment_threshold}, peak_iou_threshold={cleanup_peak_iou_threshold}, "
        f"area_ratio_cap={cleanup_area_ratio_cap}, target_length_ratio={cleanup_target_length_ratio}, "
        f"target_area_ratio={cleanup_target_area_ratio}, temporal_gap_max={cleanup_temporal_gap_max}, "
        f"temporal_boundary_window={cleanup_temporal_boundary_window}, "
        f"temporal_containment_threshold={cleanup_temporal_containment_threshold}, "
        f"temporal_peak_iou_threshold={cleanup_temporal_peak_iou_threshold})",
        flush=True,
    )

    for frag_group_id in fragment_candidates:
        frag_root = find(int(frag_group_id))
        if frag_root != int(frag_group_id):
            continue
        frag_meta = metadata[frag_group_id]
        frag_frames = frag_meta["frames"]
        frag_n_frames = frag_meta["n_frames"]
        frag_area_mean = frag_meta["area_mean"]
        frag_area_max = max(frag_meta["area_max"], 1.0)

        frag_cache = fragment_mask_cache.get(frag_group_id)
        if frag_cache is None:
            frag_cache = _compute_group_masks_for_frames(
                track_masks=track_masks,
                track_areas=track_areas,
                orig_indices=frag_meta["orig_indices"],
                frames=frag_frames,
            )
            fragment_mask_cache[frag_group_id] = frag_cache
        frag_packed_all, frag_areas_all = frag_cache

        best_target = None
        best_score = -1e9

        for cand_group_id, cand_meta in metadata.items():
            cand_root = find(int(cand_group_id))
            if cand_root != int(cand_group_id) or cand_root == frag_root:
                continue

            strong_length = cand_meta["n_frames"] >= max(
                frag_n_frames + 2,
                int(np.ceil(frag_n_frames * cleanup_target_length_ratio)),
            )
            strong_area = cand_meta["area_mean"] >= frag_area_mean * cleanup_target_area_ratio
            if not (strong_length or strong_area):
                cleanup_strength_skipped += 1
                continue

            shared_frames, frag_idx, cand_idx = np.intersect1d(
                frag_frames,
                cand_meta["frames"],
                assume_unique=True,
                return_indices=True,
            )
            cand_area_max = max(cand_meta["area_max"], 1.0)
            if (max(frag_area_max, cand_area_max) / min(frag_area_max, cand_area_max)) > cleanup_area_ratio_cap:
                cleanup_area_ratio_skipped += 1
                continue

            score = None
            merge_mode = None
            overlap_fraction = 0.0
            if shared_frames.size >= cleanup_min_shared_frames:
                overlap_fraction = float(shared_frames.size / max(frag_n_frames, 1))
                if overlap_fraction < cleanup_overlap_fraction:
                    cleanup_overlap_skipped += 1
                    continue

                cleanup_compared += 1
                frag_packed = frag_packed_all[frag_idx]
                frag_areas = frag_areas_all[frag_idx]
                cand_packed, cand_areas = _compute_group_masks_for_frames(
                    track_masks=track_masks,
                    track_areas=track_areas,
                    orig_indices=cand_meta["orig_indices"],
                    frames=shared_frames,
                )
                intersections = POPCOUNT_LUT[np.bitwise_and(frag_packed, cand_packed)].sum(
                    axis=1, dtype=np.int64
                )
                union_areas = frag_areas.astype(np.int64) + cand_areas.astype(np.int64) - intersections
                min_areas = np.minimum(frag_areas.astype(np.int64), cand_areas.astype(np.int64))
                valid = union_areas > 0
                if not np.any(valid):
                    continue

                iou_vals = intersections[valid] / union_areas[valid]
                containment_vals = intersections[valid] / np.maximum(min_areas[valid], 1)
                weighted_containment = float(intersections[valid].sum() / np.maximum(min_areas[valid].sum(), 1))
                topk = min(3, int(iou_vals.size))
                top_containment = float(np.mean(np.sort(containment_vals)[-topk:])) if topk > 0 else 0.0
                peak_iou = float(np.max(iou_vals)) if iou_vals.size else 0.0

                if (
                    weighted_containment < cleanup_containment_threshold
                    and top_containment < cleanup_containment_threshold
                ) or peak_iou < cleanup_peak_iou_threshold:
                    continue

                score = (
                    1.20 * weighted_containment
                    + 0.80 * top_containment
                    + 0.55 * peak_iou
                    + 0.35 * overlap_fraction
                    + 0.10 * np.log1p(cand_meta["n_frames"])
                    + 0.08 * np.log1p(cand_meta["area_mean"] / max(frag_area_mean, 1.0))
                )
                merge_mode = "shared"
            else:
                temporal_stats = _compute_temporal_boundary_similarity(
                    track_masks=track_masks,
                    track_areas=track_areas,
                    source_meta=frag_meta,
                    target_meta=cand_meta,
                    max_gap=cleanup_temporal_gap_max,
                    boundary_window=cleanup_temporal_boundary_window,
                    frame_mask_cache=frame_mask_cache,
                )
                if temporal_stats is None:
                    cleanup_temporal_gap_skipped += 1
                    continue
                cleanup_temporal_compared += 1
                if (
                    temporal_stats["containment"] < cleanup_temporal_containment_threshold
                    or temporal_stats["iou"] < cleanup_temporal_peak_iou_threshold
                ):
                    continue
                score = (
                    1.25 * temporal_stats["containment"]
                    + 0.70 * temporal_stats["iou"]
                    + 0.12 * np.log1p(cand_meta["n_frames"])
                    + 0.10 * np.log1p(cand_meta["area_mean"] / max(frag_area_mean, 1.0))
                    - 0.04 * temporal_stats["gap"]
                )
                merge_mode = "temporal"

            if score > best_score:
                best_score = score
                best_target = (cand_group_id, merge_mode)

        if best_target is not None:
            cand_group_id, merge_mode = best_target
            union(frag_root, int(cand_group_id))
            cleanup_merges += 1
            if merge_mode == "temporal":
                temporal_cleanup_merges += 1

    new_id_mapping, orig_to_new = _rebuild_group_mapping_from_parent(
        id_mapping=id_mapping,
        parent=parent,
        num_tracks=num_tracks,
    )
    print(
        "[SAM2] fragment cleanup summary: "
        f"candidates={len(fragment_candidates)}, "
        f"compared={cleanup_compared}, "
        f"shared_frames_skipped={cleanup_shared_skipped}, "
        f"overlap_skipped={cleanup_overlap_skipped}, "
        f"area_ratio_skipped={cleanup_area_ratio_skipped}, "
        f"strength_skipped={cleanup_strength_skipped}, "
        f"temporal_gap_skipped={cleanup_temporal_gap_skipped}, "
        f"temporal_compared={cleanup_temporal_compared}, "
        f"cleanup_merges={cleanup_merges}, "
        f"temporal_cleanup_merges={temporal_cleanup_merges}",
        flush=True,
    )
    if cleanup_merges > 0:
        print(
            f"[SAM2] fragment cleanup reduced objects {len(id_mapping)} -> {len(new_id_mapping)}",
            flush=True,
        )
    return new_id_mapping, orig_to_new


def _prune_redundant_fragment_groups(
    id_mapping,
    track_masks,
    track_seen,
    track_areas,
    prune_enabled,
    prune_max_frames,
    prune_area_mean_cap,
    prune_area_max_cap,
    prune_min_shared_frames,
    prune_overlap_fraction,
    prune_containment_threshold,
    prune_peak_containment_threshold,
    prune_target_length_ratio,
    prune_target_area_ratio,
):
    num_tracks = int(track_seen.shape[0])
    if not prune_enabled or len(id_mapping) <= 1:
        orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
        for new_id, orig_ids in id_mapping.items():
            for orig_id in orig_ids:
                orig_to_new[int(orig_id)] = int(new_id)
        return id_mapping, orig_to_new

    metadata = _collect_group_metadata(id_mapping, track_masks, track_seen, track_areas)
    if not metadata:
        orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
        return {}, orig_to_new

    prune_candidates = [
        group_id
        for group_id, meta in metadata.items()
        if (
            meta["n_frames"] <= prune_max_frames
            and meta["area_mean"] <= prune_area_mean_cap
            and meta["area_max"] <= prune_area_max_cap
        )
    ]
    prune_candidates.sort(
        key=lambda group_id: (
            metadata[group_id]["n_frames"],
            metadata[group_id]["area_mean"],
            metadata[group_id]["area_max"],
        )
    )

    mask_cache = {}
    pruned_groups = set()
    compared = 0
    shared_frames_skipped = 0
    overlap_skipped = 0
    strength_skipped = 0
    containment_skipped = 0
    pruned_count = 0

    print(
        "[SAM2] fragment prune "
        f"enabled={int(prune_enabled)} "
        f"(max_frames={prune_max_frames}, area_mean_cap={prune_area_mean_cap}, "
        f"area_max_cap={prune_area_max_cap}, min_shared_frames={prune_min_shared_frames}, "
        f"overlap_fraction={prune_overlap_fraction}, containment_threshold={prune_containment_threshold}, "
        f"peak_containment_threshold={prune_peak_containment_threshold}, "
        f"target_length_ratio={prune_target_length_ratio}, target_area_ratio={prune_target_area_ratio})",
        flush=True,
    )

    for cand_group_id in prune_candidates:
        if cand_group_id in pruned_groups:
            continue
        cand_meta = metadata[cand_group_id]
        cand_frames = cand_meta["frames"]
        cand_n_frames = cand_meta["n_frames"]
        cand_area_mean = cand_meta["area_mean"]
        cand_area_max = cand_meta["area_max"]

        cand_cache = mask_cache.get(cand_group_id)
        if cand_cache is None:
            cand_cache = _compute_group_masks_for_frames(
                track_masks=track_masks,
                track_areas=track_areas,
                orig_indices=cand_meta["orig_indices"],
                frames=cand_frames,
            )
            mask_cache[cand_group_id] = cand_cache
        cand_packed_all, cand_areas_all = cand_cache

        best_target = None
        best_score = -1e9
        for target_group_id, target_meta in metadata.items():
            if target_group_id == cand_group_id or target_group_id in pruned_groups:
                continue
            strong_length = target_meta["n_frames"] >= max(
                cand_n_frames + 1,
                int(np.ceil(cand_n_frames * prune_target_length_ratio)),
            )
            strong_area = (
                target_meta["area_mean"] >= cand_area_mean * prune_target_area_ratio
                or target_meta["area_max"] >= cand_area_max * max(1.35, prune_target_area_ratio)
            )
            if not (strong_length or strong_area):
                strength_skipped += 1
                continue

            shared_frames, cand_idx, target_idx = np.intersect1d(
                cand_frames,
                target_meta["frames"],
                assume_unique=True,
                return_indices=True,
            )
            if shared_frames.size < prune_min_shared_frames:
                shared_frames_skipped += 1
                continue
            overlap_fraction = float(shared_frames.size / max(cand_n_frames, 1))
            if overlap_fraction < prune_overlap_fraction:
                overlap_skipped += 1
                continue

            compared += 1
            cand_packed = cand_packed_all[cand_idx]
            cand_areas = cand_areas_all[cand_idx]
            target_packed, target_areas = _compute_group_masks_for_frames(
                track_masks=track_masks,
                track_areas=track_areas,
                orig_indices=target_meta["orig_indices"],
                frames=shared_frames,
            )
            intersections = POPCOUNT_LUT[np.bitwise_and(cand_packed, target_packed)].sum(
                axis=1, dtype=np.int64
            )
            valid = cand_areas > 0
            if not np.any(valid):
                continue
            containment_vals = intersections[valid] / np.maximum(cand_areas[valid], 1)
            weighted_containment = float(intersections[valid].sum() / np.maximum(cand_areas[valid].sum(), 1))
            peak_containment = float(np.max(containment_vals)) if containment_vals.size else 0.0
            if (
                weighted_containment < prune_containment_threshold
                and peak_containment < prune_peak_containment_threshold
            ):
                containment_skipped += 1
                continue

            score = (
                1.30 * weighted_containment
                + 0.70 * peak_containment
                + 0.30 * overlap_fraction
                + 0.08 * np.log1p(target_meta["n_frames"])
                + 0.08 * np.log1p(target_meta["area_mean"] / max(cand_area_mean, 1.0))
            )
            if score > best_score:
                best_score = score
                best_target = target_group_id

        if best_target is not None:
            pruned_groups.add(int(cand_group_id))
            pruned_count += 1

    kept_groups = [
        (group_id, id_mapping[group_id])
        for group_id in sorted(id_mapping)
        if int(group_id) not in pruned_groups
    ]
    new_id_mapping = {}
    orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
    for new_id, (_, orig_ids) in enumerate(kept_groups, start=1):
        new_id_mapping[new_id] = list(orig_ids)
        for orig_id in orig_ids:
            orig_to_new[int(orig_id)] = int(new_id)

    print(
        "[SAM2] fragment prune summary: "
        f"candidates={len(prune_candidates)}, "
        f"compared={compared}, "
        f"shared_frames_skipped={shared_frames_skipped}, "
        f"overlap_skipped={overlap_skipped}, "
        f"strength_skipped={strength_skipped}, "
        f"containment_skipped={containment_skipped}, "
        f"pruned={pruned_count}",
        flush=True,
    )
    if pruned_count > 0:
        print(
            f"[SAM2] fragment prune reduced objects {len(id_mapping)} -> {len(new_id_mapping)}",
            flush=True,
        )
    return new_id_mapping, orig_to_new


def _prune_low_support_groups(
    id_mapping,
    track_masks,
    track_seen,
    track_areas,
    track_key_positions,
    prune_enabled,
    prune_max_frames,
    prune_max_frame_span,
    prune_area_mean_cap,
    prune_area_max_cap,
    prune_max_orig_tracks,
    prune_max_keyframe_support,
    prune_min_shared_frames,
    prune_overlap_fraction,
    prune_containment_threshold,
    prune_peak_containment_threshold,
    prune_target_length_ratio,
    prune_target_area_ratio,
):
    num_tracks = int(track_seen.shape[0])
    if not prune_enabled or len(id_mapping) <= 1:
        orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
        for new_id, orig_ids in id_mapping.items():
            for orig_id in orig_ids:
                orig_to_new[int(orig_id)] = int(new_id)
        return id_mapping, orig_to_new

    metadata = _collect_group_metadata(
        id_mapping=id_mapping,
        track_masks=track_masks,
        track_seen=track_seen,
        track_areas=track_areas,
        track_key_positions=track_key_positions,
    )
    if not metadata:
        orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
        return {}, orig_to_new

    prune_candidates = []
    too_many_frames = 0
    too_large = 0
    too_many_tracks = 0
    too_many_keyframes = 0
    for group_id, meta in metadata.items():
        if meta["n_frames"] > prune_max_frames or meta["frame_span"] > prune_max_frame_span:
            too_many_frames += 1
            continue
        if meta["area_mean"] > prune_area_mean_cap or meta["area_max"] > prune_area_max_cap:
            too_large += 1
            continue
        if meta["orig_track_count"] > prune_max_orig_tracks:
            too_many_tracks += 1
            continue
        if meta["keyframe_support_count"] > prune_max_keyframe_support:
            too_many_keyframes += 1
            continue
        prune_candidates.append(int(group_id))

    prune_candidates.sort(
        key=lambda group_id: (
            metadata[group_id]["n_frames"],
            metadata[group_id]["frame_span"],
            metadata[group_id]["area_mean"],
            metadata[group_id]["area_max"],
        )
    )

    mask_cache = {}
    pruned_groups = set()
    compared = 0
    shared_frames_skipped = 0
    overlap_skipped = 0
    strength_skipped = 0
    containment_skipped = 0
    pruned_count = 0

    for cand_group_id in prune_candidates:
        if cand_group_id in pruned_groups:
            continue
        cand_meta = metadata[cand_group_id]
        cand_frames = cand_meta["frames"]
        cand_n_frames = cand_meta["n_frames"]
        cand_area_mean = cand_meta["area_mean"]
        cand_area_max = cand_meta["area_max"]

        cand_cache = mask_cache.get(cand_group_id)
        if cand_cache is None:
            cand_cache = _compute_group_masks_for_frames(
                track_masks=track_masks,
                track_areas=track_areas,
                orig_indices=cand_meta["orig_indices"],
                frames=cand_frames,
            )
            mask_cache[cand_group_id] = cand_cache
        cand_packed_all, cand_areas_all = cand_cache

        best_target = None
        best_score = -1e9
        for target_group_id, target_meta in metadata.items():
            if target_group_id == cand_group_id or target_group_id in pruned_groups:
                continue
            strong_length = target_meta["n_frames"] >= max(
                cand_n_frames + 1,
                int(np.ceil(cand_n_frames * prune_target_length_ratio)),
            )
            strong_area = (
                target_meta["area_mean"] >= cand_area_mean * prune_target_area_ratio
                or target_meta["area_max"] >= cand_area_max * max(1.35, prune_target_area_ratio)
            )
            strong_support = (
                target_meta["orig_track_count"] > cand_meta["orig_track_count"]
                or target_meta["keyframe_support_count"] > cand_meta["keyframe_support_count"]
                or target_meta["n_frames"] > cand_n_frames
            )
            if not ((strong_length or strong_area) and strong_support):
                strength_skipped += 1
                continue

            shared_frames, cand_idx, target_idx = np.intersect1d(
                cand_frames,
                target_meta["frames"],
                assume_unique=True,
                return_indices=True,
            )
            if shared_frames.size < prune_min_shared_frames:
                shared_frames_skipped += 1
                continue
            overlap_fraction = float(shared_frames.size / max(cand_n_frames, 1))
            if overlap_fraction < prune_overlap_fraction:
                overlap_skipped += 1
                continue

            compared += 1
            cand_packed = cand_packed_all[cand_idx]
            cand_areas = cand_areas_all[cand_idx]
            target_cache = mask_cache.get(target_group_id)
            if target_cache is None:
                target_cache = _compute_group_masks_for_frames(
                    track_masks=track_masks,
                    track_areas=track_areas,
                    orig_indices=target_meta["orig_indices"],
                    frames=target_meta["frames"],
                )
                mask_cache[target_group_id] = target_cache
            target_packed_all, _ = target_cache
            target_packed = target_packed_all[target_idx]

            intersections = POPCOUNT_LUT[np.bitwise_and(cand_packed, target_packed)].sum(
                axis=1, dtype=np.int64
            )
            valid = cand_areas > 0
            if not np.any(valid):
                continue
            containment_vals = intersections[valid] / np.maximum(cand_areas[valid], 1)
            weighted_containment = float(
                intersections[valid].sum() / np.maximum(cand_areas[valid].sum(), 1)
            )
            peak_containment = float(np.max(containment_vals)) if containment_vals.size else 0.0
            if (
                weighted_containment < prune_containment_threshold
                and peak_containment < prune_peak_containment_threshold
            ):
                containment_skipped += 1
                continue

            score = (
                1.40 * weighted_containment
                + 0.75 * peak_containment
                + 0.20 * overlap_fraction
                + 0.06 * np.log1p(target_meta["n_frames"])
                + 0.06 * np.log1p(target_meta["area_mean"] / max(cand_area_mean, 1.0))
                + 0.04 * float(target_meta["keyframe_support_count"] > cand_meta["keyframe_support_count"])
            )
            if score > best_score:
                best_score = score
                best_target = target_group_id

        if best_target is not None:
            pruned_groups.add(int(cand_group_id))
            pruned_count += 1

    kept_groups = [
        (group_id, id_mapping[group_id])
        for group_id in sorted(id_mapping)
        if int(group_id) not in pruned_groups
    ]
    new_id_mapping = {}
    orig_to_new = np.zeros(num_tracks + 1, dtype=np.int32)
    for new_id, (_, orig_ids) in enumerate(kept_groups, start=1):
        new_id_mapping[new_id] = list(orig_ids)
        for orig_id in orig_ids:
            orig_to_new[int(orig_id)] = int(new_id)

    print(
        "[SAM2] low-support prune summary: "
        f"enabled={int(prune_enabled)}, "
        f"max_frames={prune_max_frames}, "
        f"max_frame_span={prune_max_frame_span}, "
        f"area_mean_cap={prune_area_mean_cap}, "
        f"area_max_cap={prune_area_max_cap}, "
        f"max_orig_tracks={prune_max_orig_tracks}, "
        f"max_keyframe_support={prune_max_keyframe_support}, "
        f"min_shared_frames={prune_min_shared_frames}, "
        f"overlap_fraction={prune_overlap_fraction}, "
        f"containment_threshold={prune_containment_threshold}, "
        f"peak_containment_threshold={prune_peak_containment_threshold}, "
        f"target_length_ratio={prune_target_length_ratio}, "
        f"target_area_ratio={prune_target_area_ratio}, "
        f"candidates={len(prune_candidates)}, "
        f"compared={compared}, "
        f"shared_frames_skipped={shared_frames_skipped}, "
        f"overlap_skipped={overlap_skipped}, "
        f"strength_skipped={strength_skipped}, "
        f"containment_skipped={containment_skipped}, "
        f"pruned={pruned_count}, "
        f"too_many_frames={too_many_frames}, "
        f"too_large={too_large}, "
        f"too_many_tracks={too_many_tracks}, "
        f"too_many_keyframes={too_many_keyframes}",
        flush=True,
    )
    if pruned_count > 0:
        print(
            f"[SAM2] low-support prune reduced objects {len(id_mapping)} -> {len(new_id_mapping)}",
            flush=True,
        )
    return new_id_mapping, orig_to_new


def _summarize_merged_groups(id_mapping, track_masks, track_seen, track_areas, tmp_to_scan_fidx):
    object_stats = {}
    num_frames = int(track_seen.shape[1])
    print(f"[SAM2] summarizing {len(id_mapping)} merged objects", flush=True)

    for obj_idx, (new_id, orig_ids) in enumerate(id_mapping.items(), start=1):
        if obj_idx == 1 or obj_idx % 8 == 0 or obj_idx == len(id_mapping):
            print(f"[SAM2] summary progress {obj_idx}/{len(id_mapping)}", flush=True)
        orig_indices = [orig_id - 1 for orig_id in orig_ids]
        group_seen = np.any(track_seen[orig_indices], axis=0)
        shared_frames = np.flatnonzero(group_seen)
        if shared_frames.size == 0:
            continue

        if len(orig_indices) == 1:
            areas = track_areas[orig_indices[0], shared_frames].astype(np.int64)
        else:
            merged_masks = np.bitwise_or.reduce(track_masks[orig_indices][:, shared_frames], axis=0)
            areas = POPCOUNT_LUT[merged_masks].sum(axis=1, dtype=np.int64)

        frame_idxs = [int(tmp_to_scan_fidx[int(tmp_fidx)]) for tmp_fidx in shared_frames]

        object_stats[new_id] = {
            "id": str(new_id),
            "n_frames": len(frame_idxs),
            "first_frame": int(frame_idxs[0]),
            "last_frame": int(frame_idxs[-1]),
            "area_mean": float(np.mean(areas)),
            "area_max": float(np.max(areas)),
        }

    return object_stats



def visualize_results(frames, obj_id_imgs, frame_paths, out_dir, stride=5):
    """Visualize obj_id_map as a color overlay and save to disk."""
    os.makedirs(out_dir, exist_ok=True)
    for fidx in range(0, len(frames), stride):
        scan_fidx = _frame_key(_parse_scan_frame_idx(frame_paths[fidx]))
        obj_map = obj_id_imgs.get(scan_fidx)
        if obj_map is None or obj_map.max() == 0:
            continue
        unique_ids = np.unique(obj_map[obj_map > 0])
        masks = [(obj_map == uid) for uid in unique_ids]
        vis = visualize_masks_on_frame(frames[fidx], masks)
        cv2.imwrite(
            os.path.join(out_dir, f"vis_{fidx:06d}.jpg"),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def colorize_obj_map(obj_map):
    """Convert an integer obj_id map to a deterministic RGB color image."""
    vis = np.zeros((*obj_map.shape, 3), dtype=np.uint8)

    for oid in np.unique(obj_map):
        if oid == 0:
            continue # background stays black
        rng = np.random.default_rng(int(oid))
        color = rng.integers(60, 255, size=3, dtype=np.uint8)
        vis[obj_map == oid] = color

    return vis


def propagate_masks(frame_paths, keyframe_masks, cfg, output_dir, scan_id, device="cuda"):
    """
    Propagate SAM2 masks across all frames and save in VLSG/Object-X format:
      obj_id/<scan_id>/frame-xxxxxx.png   — uint16 lossless ID map
      obj_id_pkl/<scan_id>.pkl            — {frame_idx: np.ndarray int32}
      color/<scan_id>/frame-xxxxxx.jpg    — RGB visualization
    Returns:
        obj_id_imgs:    saved to disk under output_dir
        object_stats:   merged-track summary compatible with objects.json export
    """
    print("\n[SAM2] Step 2: Propagating with VideoPredictor")
    offload_video_to_cpu = _parse_bool(
        os.environ.get("OBJECTX_SAM2_OFFLOAD_VIDEO_TO_CPU"),
        cfg.get("offload_video_to_cpu", True),
    )
    offload_state_to_cpu = _parse_bool(
        os.environ.get("OBJECTX_SAM2_OFFLOAD_STATE_TO_CPU"),
        cfg.get("offload_state_to_cpu", True),
    )
    async_loading_frames = _parse_bool(
        os.environ.get("OBJECTX_SAM2_ASYNC_LOADING_FRAMES"),
        cfg.get("async_loading_frames", False),
    )
    print(
        "[SAM2] VideoPredictor memory mode: "
        f"offload_video_to_cpu={int(offload_video_to_cpu)} "
        f"offload_state_to_cpu={int(offload_state_to_cpu)} "
        f"async_loading_frames={int(async_loading_frames)}"
    )
    max_obj = _parse_int(
        os.environ.get("OBJECTX_SAM2_MAX_OBJ_IDS"),
        cfg.get("max_obj_ids", 50),
    )
    prob_thr = _parse_float(
        os.environ.get("OBJECTX_SAM2_MASK_PROB_THRESHOLD"),
        cfg.get("mask_prob_threshold", 0.5),
    )
    iou_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_IOU_THRESHOLD"),
        cfg.get("merge_iou_threshold", 0.5),
    )
    relaxed_iou_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_RELAXED_IOU_THRESHOLD"),
        cfg.get("merge_relaxed_iou_threshold", 0.24),
    )
    containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_containment_threshold", 0.78),
    )
    peak_iou_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_PEAK_IOU_THRESHOLD"),
        cfg.get("merge_peak_iou_threshold", 0.40),
    )
    same_keyframe_containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_SAME_KEYFRAME_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_same_keyframe_containment_threshold", 0.86),
    )
    same_keyframe_peak_iou_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_SAME_KEYFRAME_PEAK_IOU_THRESHOLD"),
        cfg.get("merge_same_keyframe_peak_iou_threshold", 0.46),
    )
    fragment_track_max_frames = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_TRACK_MAX_FRAMES"),
        cfg.get("merge_fragment_track_max_frames", 12),
    )
    fragment_containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_fragment_containment_threshold", 0.84),
    )
    fragment_peak_iou_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PEAK_IOU_THRESHOLD"),
        cfg.get("merge_fragment_peak_iou_threshold", 0.32),
    )
    fragment_overlap_fraction = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_OVERLAP_FRACTION"),
        cfg.get("merge_fragment_overlap_fraction", 0.35),
    )
    short_track_area_ratio_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_SHORT_TRACK_AREA_RATIO_CAP"),
        cfg.get("merge_short_track_area_ratio_cap", 24.0),
    )
    fragment_cleanup_enabled = _parse_bool(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_ENABLED"),
        cfg.get("merge_fragment_cleanup_enabled", True),
    )
    fragment_cleanup_max_frames = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_MAX_FRAMES"),
        cfg.get("merge_fragment_cleanup_max_frames", 18),
    )
    fragment_cleanup_area_mean_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_AREA_MEAN_CAP"),
        cfg.get("merge_fragment_cleanup_area_mean_cap", 12000.0),
    )
    fragment_cleanup_min_shared_frames = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_MIN_SHARED_FRAMES"),
        cfg.get("merge_fragment_cleanup_min_shared_frames", 1),
    )
    fragment_cleanup_overlap_fraction = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_OVERLAP_FRACTION"),
        cfg.get("merge_fragment_cleanup_overlap_fraction", 0.45),
    )
    fragment_cleanup_containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_fragment_cleanup_containment_threshold", 0.84),
    )
    fragment_cleanup_peak_iou_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_PEAK_IOU_THRESHOLD"),
        cfg.get("merge_fragment_cleanup_peak_iou_threshold", 0.26),
    )
    fragment_cleanup_area_ratio_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_AREA_RATIO_CAP"),
        cfg.get("merge_fragment_cleanup_area_ratio_cap", 32.0),
    )
    fragment_cleanup_target_length_ratio = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_TARGET_LENGTH_RATIO"),
        cfg.get("merge_fragment_cleanup_target_length_ratio", 1.5),
    )
    fragment_cleanup_target_area_ratio = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_TARGET_AREA_RATIO"),
        cfg.get("merge_fragment_cleanup_target_area_ratio", 1.75),
    )
    fragment_cleanup_temporal_gap_max = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_TEMPORAL_GAP_MAX"),
        cfg.get("merge_fragment_cleanup_temporal_gap_max", 6),
    )
    fragment_cleanup_temporal_boundary_window = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_TEMPORAL_BOUNDARY_WINDOW"),
        cfg.get("merge_fragment_cleanup_temporal_boundary_window", 2),
    )
    fragment_cleanup_temporal_containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_TEMPORAL_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_fragment_cleanup_temporal_containment_threshold", 0.72),
    )
    fragment_cleanup_temporal_peak_iou_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_CLEANUP_TEMPORAL_PEAK_IOU_THRESHOLD"),
        cfg.get("merge_fragment_cleanup_temporal_peak_iou_threshold", 0.18),
    )
    fragment_prune_enabled = _parse_bool(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_ENABLED"),
        cfg.get("merge_fragment_prune_enabled", True),
    )
    fragment_prune_max_frames = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_MAX_FRAMES"),
        cfg.get("merge_fragment_prune_max_frames", 5),
    )
    fragment_prune_area_mean_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_AREA_MEAN_CAP"),
        cfg.get("merge_fragment_prune_area_mean_cap", 5000.0),
    )
    fragment_prune_area_max_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_AREA_MAX_CAP"),
        cfg.get("merge_fragment_prune_area_max_cap", 12000.0),
    )
    fragment_prune_min_shared_frames = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_MIN_SHARED_FRAMES"),
        cfg.get("merge_fragment_prune_min_shared_frames", 1),
    )
    fragment_prune_overlap_fraction = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_OVERLAP_FRACTION"),
        cfg.get("merge_fragment_prune_overlap_fraction", 0.50),
    )
    fragment_prune_containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_fragment_prune_containment_threshold", 0.94),
    )
    fragment_prune_peak_containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_PEAK_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_fragment_prune_peak_containment_threshold", 0.98),
    )
    fragment_prune_target_length_ratio = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_TARGET_LENGTH_RATIO"),
        cfg.get("merge_fragment_prune_target_length_ratio", 1.5),
    )
    fragment_prune_target_area_ratio = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_FRAGMENT_PRUNE_TARGET_AREA_RATIO"),
        cfg.get("merge_fragment_prune_target_area_ratio", 1.4),
    )
    low_support_prune_enabled = _parse_bool(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_ENABLED"),
        cfg.get("merge_low_support_prune_enabled", False),
    )
    low_support_prune_max_frames = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_MAX_FRAMES"),
        cfg.get("merge_low_support_prune_max_frames", 3),
    )
    low_support_prune_max_frame_span = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_MAX_FRAME_SPAN"),
        cfg.get("merge_low_support_prune_max_frame_span", 24),
    )
    low_support_prune_area_mean_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_AREA_MEAN_CAP"),
        cfg.get("merge_low_support_prune_area_mean_cap", 2500.0),
    )
    low_support_prune_area_max_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_AREA_MAX_CAP"),
        cfg.get("merge_low_support_prune_area_max_cap", 7000.0),
    )
    low_support_prune_max_orig_tracks = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_MAX_ORIG_TRACKS"),
        cfg.get("merge_low_support_prune_max_orig_tracks", 1),
    )
    low_support_prune_max_keyframe_support = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_MAX_KEYFRAME_SUPPORT"),
        cfg.get("merge_low_support_prune_max_keyframe_support", 1),
    )
    low_support_prune_min_shared_frames = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_MIN_SHARED_FRAMES"),
        cfg.get("merge_low_support_prune_min_shared_frames", 2),
    )
    low_support_prune_overlap_fraction = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_OVERLAP_FRACTION"),
        cfg.get("merge_low_support_prune_overlap_fraction", 0.80),
    )
    low_support_prune_containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_low_support_prune_containment_threshold", 0.98),
    )
    low_support_prune_peak_containment_threshold = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_PEAK_CONTAINMENT_THRESHOLD"),
        cfg.get("merge_low_support_prune_peak_containment_threshold", 0.995),
    )
    low_support_prune_target_length_ratio = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_TARGET_LENGTH_RATIO"),
        cfg.get("merge_low_support_prune_target_length_ratio", 1.5),
    )
    low_support_prune_target_area_ratio = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_LOW_SUPPORT_PRUNE_TARGET_AREA_RATIO"),
        cfg.get("merge_low_support_prune_target_area_ratio", 1.5),
    )
    max_keyframe_hops = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_KEYFRAME_HOPS"),
        cfg.get("merge_keyframe_hops", 1),
    )
    min_shared_frames = _parse_int(
        os.environ.get("OBJECTX_SAM2_MERGE_MIN_SHARED_FRAMES"),
        cfg.get("merge_min_shared_frames", 3),
    )
    area_ratio_cap = _parse_float(
        os.environ.get("OBJECTX_SAM2_MERGE_AREA_RATIO_CAP"),
        cfg.get("merge_area_ratio_cap", 8.0),
    )
    prob_cache_dtype = _parse_dtype(
        os.environ.get("OBJECTX_SAM2_PROB_CACHE_DTYPE"),
        cfg.get("prob_cache_dtype", "float16"),
    )
    obj_chunk_size = int(
        os.environ.get(
            "OBJECTX_SAM2_PROPAGATE_OBJECTS_PER_CHUNK",
            cfg.get("propagate_objects_per_chunk", 8),
        )
    )
    obj_chunk_size = max(1, obj_chunk_size)

    work_dir = _prepare_work_dir(output_dir, scan_id, cfg)
    tmp_dir = work_dir / "frames"
    cache_dir = work_dir / "cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_to_scan_fidx = {}  # maps tmp sequential index → original scan frame index
    predictor = None

    try:
        # ── Copy frames to tmp dir with sequential names for SAM2 ──
        for tmp_idx, fp in enumerate(frame_paths):
            fp = Path(fp)
            dst = tmp_dir / f"{tmp_idx:06d}{fp.suffix.lower()}"
            shutil.copy2(fp, dst)
            tmp_to_scan_fidx[tmp_idx] = _parse_scan_frame_idx(fp)

        if frame_paths:
            sample_frame = cv2.imread(str(frame_paths[0]))
            H, W = sample_frame.shape[:2]
        else:
            H, W = 540, 960

        num_frames = len(frame_paths)
        scan_fidxs_in_order = [_frame_key(tmp_to_scan_fidx[i]) for i in range(num_frames)]
        ordered_keyframe_items = sorted(keyframe_masks.items())
        prepared_keyframe_items = []
        for kf_idx, masks in ordered_keyframe_items:
            prepared_masks = _prepare_keyframe_masks_for_propagation(
                kf_idx=kf_idx,
                masks=masks,
                max_obj=max_obj,
                frame_shape=(H, W),
            )
            prepared_keyframe_items.append((kf_idx, prepared_masks))
        num_tracks = sum(len(masks) for _, masks in prepared_keyframe_items)
        packed_len = (H * W + 7) // 8
        track_id_dtype = np.uint16 if num_tracks < np.iinfo(np.uint16).max else np.int32

        track_mask_path = cache_dir / "track_masks.dat"
        track_seen = np.zeros((num_tracks, num_frames), dtype=np.bool_)
        track_areas = np.zeros((num_tracks, num_frames), dtype=np.int32)
        track_key_positions = np.full(num_tracks, -1, dtype=np.int16)
        obj_id_mm = np.memmap(
            cache_dir / "obj_id_maps.dat",
            mode="w+",
            dtype=track_id_dtype,
            shape=(num_frames, H, W),
        )
        score_mm = np.memmap(
            cache_dir / "score_maps.dat",
            mode="w+",
            dtype=prob_cache_dtype,
            shape=(num_frames, H, W),
        )
        obj_id_mm[:] = 0
        score_mm[:] = 0
        if num_tracks == 0:
            object_stats = {}
            orig_to_new = np.zeros(1, dtype=np.int32)
        else:
            track_masks = np.memmap(
                track_mask_path,
                mode="w+",
                dtype=np.uint8,
                shape=(num_tracks, num_frames, packed_len),
            )
            global_track_id = 1

            # Pass 1: store original binary tracks exactly as before, but on disk.
            for key_pos, (kf_idx, limited_masks) in enumerate(prepared_keyframe_items):
                n_objs = len(limited_masks)
                print(f"  Keyframe {kf_idx}: {n_objs} objects")
                for chunk_start in range(0, n_objs, obj_chunk_size):
                    chunk_masks = limited_masks[chunk_start : chunk_start + obj_chunk_size]
                    chunk_end = chunk_start + len(chunk_masks) - 1
                    chunk_label = "full-batch" if len(chunk_masks) == n_objs else "chunk"
                    print(
                        f"    pass1 {chunk_label} {chunk_start}-{chunk_end} "
                        f"of keyframe {kf_idx} (size={len(chunk_masks)})"
                    )

                    predictor = build_sam2_video_predictor(
                        cfg["model_cfg"],
                        cfg["checkpoint"],
                        device=device,
                    )
                    chunk_gid_start = global_track_id
                    chunk_gid_stop = chunk_gid_start + len(chunk_masks)
                    track_key_positions[chunk_gid_start - 1 : chunk_gid_stop - 1] = key_pos

                    def on_prob_pass1(local_oid, fidx, prob_map):
                        orig_id = chunk_gid_start + local_oid
                        mask_bin = prob_map >= prob_thr
                        mask_area = int(mask_bin.sum())
                        if mask_area <= 0:
                            return
                        track_masks[orig_id - 1, fidx] = _pack_mask(mask_bin)
                        track_seen[orig_id - 1, fidx] = True
                        track_areas[orig_id - 1, fidx] = mask_area
                        update = mask_bin & (prob_map > score_mm[fidx])
                        if not np.any(update):
                            return
                        score_mm[fidx][update] = prob_map[update].astype(prob_cache_dtype, copy=False)
                        obj_id_mm[fidx][update] = int(orig_id)

                    _run_chunk_propagation(
                        predictor=predictor,
                        tmp_dir=tmp_dir,
                        chunk_masks=chunk_masks,
                        kf_idx=kf_idx,
                        num_frames=num_frames,
                        device=device,
                        offload_video_to_cpu=offload_video_to_cpu,
                        offload_state_to_cpu=offload_state_to_cpu,
                        async_loading_frames=async_loading_frames,
                        on_prob=on_prob_pass1,
                    )

                    global_track_id += len(chunk_masks)
                    del predictor
                    predictor = None
                    gc.collect()
                    torch.cuda.empty_cache()

            track_masks.flush()
            id_mapping, orig_to_new = _merge_packed_tracks(
                track_masks=track_masks,
                track_seen=track_seen,
                track_areas=track_areas,
                track_key_positions=track_key_positions,
                iou_threshold=iou_threshold,
                relaxed_iou_threshold=relaxed_iou_threshold,
                containment_threshold=containment_threshold,
                peak_iou_threshold=peak_iou_threshold,
                same_keyframe_containment_threshold=same_keyframe_containment_threshold,
                same_keyframe_peak_iou_threshold=same_keyframe_peak_iou_threshold,
                fragment_track_max_frames=fragment_track_max_frames,
                fragment_containment_threshold=fragment_containment_threshold,
                fragment_peak_iou_threshold=fragment_peak_iou_threshold,
                fragment_overlap_fraction=fragment_overlap_fraction,
                short_track_area_ratio_cap=short_track_area_ratio_cap,
                max_keyframe_hops=max_keyframe_hops,
                min_shared_frames=min_shared_frames,
                area_ratio_cap=area_ratio_cap,
            )
            id_mapping, orig_to_new = _cleanup_fragment_groups(
                id_mapping=id_mapping,
                track_masks=track_masks,
                track_seen=track_seen,
                track_areas=track_areas,
                cleanup_enabled=fragment_cleanup_enabled,
                cleanup_max_frames=fragment_cleanup_max_frames,
                cleanup_area_mean_cap=fragment_cleanup_area_mean_cap,
                cleanup_min_shared_frames=fragment_cleanup_min_shared_frames,
                cleanup_overlap_fraction=fragment_cleanup_overlap_fraction,
                cleanup_containment_threshold=fragment_cleanup_containment_threshold,
                cleanup_peak_iou_threshold=fragment_cleanup_peak_iou_threshold,
                cleanup_area_ratio_cap=fragment_cleanup_area_ratio_cap,
                cleanup_target_length_ratio=fragment_cleanup_target_length_ratio,
                cleanup_target_area_ratio=fragment_cleanup_target_area_ratio,
                cleanup_temporal_gap_max=fragment_cleanup_temporal_gap_max,
                cleanup_temporal_boundary_window=fragment_cleanup_temporal_boundary_window,
                cleanup_temporal_containment_threshold=fragment_cleanup_temporal_containment_threshold,
                cleanup_temporal_peak_iou_threshold=fragment_cleanup_temporal_peak_iou_threshold,
            )
            id_mapping, orig_to_new = _prune_redundant_fragment_groups(
                id_mapping=id_mapping,
                track_masks=track_masks,
                track_seen=track_seen,
                track_areas=track_areas,
                prune_enabled=fragment_prune_enabled,
                prune_max_frames=fragment_prune_max_frames,
                prune_area_mean_cap=fragment_prune_area_mean_cap,
                prune_area_max_cap=fragment_prune_area_max_cap,
                prune_min_shared_frames=fragment_prune_min_shared_frames,
                prune_overlap_fraction=fragment_prune_overlap_fraction,
                prune_containment_threshold=fragment_prune_containment_threshold,
                prune_peak_containment_threshold=fragment_prune_peak_containment_threshold,
                prune_target_length_ratio=fragment_prune_target_length_ratio,
                prune_target_area_ratio=fragment_prune_target_area_ratio,
            )
            id_mapping, orig_to_new = _prune_low_support_groups(
                id_mapping=id_mapping,
                track_masks=track_masks,
                track_seen=track_seen,
                track_areas=track_areas,
                track_key_positions=track_key_positions,
                prune_enabled=low_support_prune_enabled,
                prune_max_frames=low_support_prune_max_frames,
                prune_max_frame_span=low_support_prune_max_frame_span,
                prune_area_mean_cap=low_support_prune_area_mean_cap,
                prune_area_max_cap=low_support_prune_area_max_cap,
                prune_max_orig_tracks=low_support_prune_max_orig_tracks,
                prune_max_keyframe_support=low_support_prune_max_keyframe_support,
                prune_min_shared_frames=low_support_prune_min_shared_frames,
                prune_overlap_fraction=low_support_prune_overlap_fraction,
                prune_containment_threshold=low_support_prune_containment_threshold,
                prune_peak_containment_threshold=low_support_prune_peak_containment_threshold,
                prune_target_length_ratio=low_support_prune_target_length_ratio,
                prune_target_area_ratio=low_support_prune_target_area_ratio,
            )
            object_stats = _summarize_merged_groups(
                id_mapping=id_mapping,
                track_masks=track_masks,
                track_seen=track_seen,
                track_areas=track_areas,
                tmp_to_scan_fidx=tmp_to_scan_fidx,
            )
            del track_masks
            gc.collect()
            track_mask_path.unlink(missing_ok=True)
            print("[SAM2] remapping per-pixel winners to merged ids", flush=True)
            for tmp_fidx in range(num_frames):
                if tmp_fidx == 0 or (tmp_fidx + 1) % 64 == 0 or (tmp_fidx + 1) == num_frames:
                    print(f"[SAM2] remap progress {tmp_fidx + 1}/{num_frames}", flush=True)
                obj_id_mm[tmp_fidx] = orig_to_new[
                    np.asarray(obj_id_mm[tmp_fidx], dtype=np.int32)
                ].astype(track_id_dtype, copy=False)

        if not object_stats:
            print("[SAM2] Avís: cap objecte detectat en aquesta escena")

        # ── Save in VLSG/Object-X format ──
        obj_dir = os.path.join(output_dir, "obj_id", scan_id)
        pkl_dir = os.path.join(output_dir, "obj_id_pkl")
        color_dir = os.path.join(output_dir, "color", scan_id)
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(obj_dir, exist_ok=True)
        os.makedirs(pkl_dir, exist_ok=True)

        obj_id_imgs = {}
        for tmp_fidx, scan_fidx in enumerate(scan_fidxs_in_order):
            obj_id_map = np.asarray(obj_id_mm[tmp_fidx])
            obj_id_imgs[scan_fidx] = obj_id_map
            # obj_id real
            cv2.imwrite(
                os.path.join(obj_dir, f"frame-{scan_fidx}.jpg"),
                obj_id_map.astype(np.uint16)
            )

            # visualization color
            color_vis = colorize_obj_map(obj_id_map)
            cv2.imwrite(
                os.path.join(color_dir, f"frame-{scan_fidx}.jpg"),
                cv2.cvtColor(color_vis, cv2.COLOR_RGB2BGR)
            )

        # Save full int32 maps as pickle
        with open(os.path.join(pkl_dir, f"{scan_id}.pkl"), "wb") as f:
            pickle.dump(obj_id_imgs, f)

        n_empty = sum(1 for m in obj_id_imgs.values() if m.max() == 0)
        print(f"[SAM2] {len(object_stats)} objects after fusion")
        print(f"[SAM2] {len(obj_id_imgs)} frames saved ({n_empty} empty / background-only)")
        obj_id_imgs.clear()

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if predictor is not None:
            del predictor
        torch.cuda.empty_cache()

    return {}, object_stats
