from dataclasses import dataclass
from typing import Any, Mapping


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


@dataclass(frozen=True)
class Sam2ModelConfig:
    model_cfg: str
    checkpoint: str


@dataclass(frozen=True)
class Sam2KeyframeConfig:
    points_per_side: int
    pred_iou_thresh: float
    stability_score_thresh: float
    stability_score_offset: float
    box_nms_thresh: float
    crop_n_layers: int
    crop_n_points_downscale_factor: int
    min_mask_region_area: int
    output_mode: str
    points_per_batch: int


@dataclass(frozen=True)
class Sam2PropagationConfig:
    max_obj_ids: int
    mask_prob_threshold: float
    propagation_mode: str
    global_prompt_keyframes: Any
    global_max_total_obj_ids: int
    temporal_vote_weight: float
    min_mask_pixels_per_frame: int
    objects_per_chunk: int
    hard_filter_enabled: bool
    candidate_multiplier: int
    min_keep: int
    min_mask_score: float
    min_pred_iou: float
    min_stability: float
    tiny_area_frac: float
    dedupe_enabled: bool
    dedupe_iou: float
    dedupe_containment: float


@dataclass(frozen=True)
class Sam2MergeConfig:
    iou_threshold: float
    iou_reduction: str
    relaxed_iou_threshold: float
    containment_threshold: float
    peak_iou_threshold: float
    min_shared_frames: int


@dataclass(frozen=True)
class Sam2PruneConfig:
    enabled: bool
    min_frames: int
    min_peak_area: int
    min_total_area: int


@dataclass(frozen=True)
class Sam2RuntimeConfig:
    model: Sam2ModelConfig
    keyframes: Sam2KeyframeConfig
    propagation: Sam2PropagationConfig
    merge: Sam2MergeConfig
    prune: Sam2PruneConfig


# Phase 1 note: environment overrides are intentionally limited to a small,
# explicit set to keep behavior stable and readable.
def build_runtime_config(raw_cfg: Mapping[str, Any]) -> Sam2RuntimeConfig:
    model_cfg = str(raw_cfg["model_cfg"])
    checkpoint = str(raw_cfg["checkpoint"])

    keyframes = Sam2KeyframeConfig(
        points_per_side=_to_int(raw_cfg.get("points_per_side"), 32),
        pred_iou_thresh=_to_float(raw_cfg.get("pred_iou_thresh"), 0.95),
        stability_score_thresh=_to_float(raw_cfg.get("stability_score_thresh"), 0.92),
        stability_score_offset=_to_float(raw_cfg.get("stability_score_offset"), 1.0),
        box_nms_thresh=_to_float(raw_cfg.get("box_nms_thresh"), 0.5),
        crop_n_layers=_to_int(raw_cfg.get("crop_n_layers"), 0),
        crop_n_points_downscale_factor=_to_int(raw_cfg.get("crop_n_points_downscale_factor"), 2),
        min_mask_region_area=_to_int(raw_cfg.get("min_mask_region_area"), 1200),
        output_mode=str(raw_cfg.get("output_mode", "binary_mask")),
        points_per_batch=_to_int(raw_cfg.get("points_per_batch"), 64),
    )

    propagation = Sam2PropagationConfig(
        max_obj_ids=_to_int(raw_cfg.get("max_obj_ids"), 50),
        mask_prob_threshold=_to_float(raw_cfg.get("mask_prob_threshold"), 0.5),
        propagation_mode=str(raw_cfg.get("propagation_mode", "per_keyframe")).strip().lower(),
        global_prompt_keyframes=raw_cfg.get("global_prompt_keyframes", "all"),
        global_max_total_obj_ids=_to_int(raw_cfg.get("global_max_total_obj_ids"), 0),
        temporal_vote_weight=_to_float(raw_cfg.get("temporal_vote_weight"), 0.15),
        min_mask_pixels_per_frame=_to_int(raw_cfg.get("min_mask_pixels_per_frame"), 0),
        objects_per_chunk=_to_int(raw_cfg.get("propagate_objects_per_chunk"), 8),
        hard_filter_enabled=_to_bool(raw_cfg.get("prop_mask_hard_filter_enabled"), True),
        candidate_multiplier=_to_int(raw_cfg.get("prop_mask_candidate_multiplier"), 2),
        min_keep=_to_int(raw_cfg.get("prop_mask_min_keep"), 4),
        min_mask_score=_to_float(raw_cfg.get("prop_mask_min_score"), 0.15),
        min_pred_iou=_to_float(raw_cfg.get("prop_mask_min_pred_iou"), 0.70),
        min_stability=_to_float(raw_cfg.get("prop_mask_min_stability"), 0.70),
        tiny_area_frac=_to_float(raw_cfg.get("prop_mask_tiny_area_frac"), 0.0008),
        dedupe_enabled=_to_bool(raw_cfg.get("prop_mask_dedupe_enabled"), True),
        dedupe_iou=_to_float(raw_cfg.get("prop_mask_dedupe_iou"), 0.85),
        dedupe_containment=_to_float(raw_cfg.get("prop_mask_dedupe_containment"), 0.92),
    )

    merge = Sam2MergeConfig(
        iou_threshold=_to_float(raw_cfg.get("merge_iou_threshold"), 0.6),
        iou_reduction=str(raw_cfg.get("merge_iou_reduction", "max")),
        relaxed_iou_threshold=_to_float(raw_cfg.get("merge_relaxed_iou_threshold"), 0.24),
        containment_threshold=_to_float(raw_cfg.get("merge_containment_threshold"), 0.78),
        peak_iou_threshold=_to_float(raw_cfg.get("merge_peak_iou_threshold"), 0.40),
        min_shared_frames=_to_int(raw_cfg.get("merge_min_shared_frames"), 2),
    )

    prune = Sam2PruneConfig(
        enabled=_to_bool(raw_cfg.get("prune_short_tracks"), True),
        min_frames=_to_int(raw_cfg.get("track_min_frames"), 2),
        min_peak_area=_to_int(raw_cfg.get("track_min_peak_area"), 400),
        min_total_area=_to_int(raw_cfg.get("track_min_total_area"), 1600),
    )

    return Sam2RuntimeConfig(
        model=Sam2ModelConfig(model_cfg=model_cfg, checkpoint=checkpoint),
        keyframes=keyframes,
        propagation=propagation,
        merge=merge,
        prune=prune,
    )
