"""
SAM2: AutomaticMaskGenerator (grid) on keyframes + VideoPredictor propagation.
"""
import os, re, shutil, tempfile, pickle
import torch, numpy as np, cv2
from pathlib import Path
from typing import Dict, List
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.mask_utils import merge_tracks_by_iou, visualize_masks_on_frame


def _reindex_tracks(merged_binary, id_mapping):
    """Reindex object IDs to 1..N after filtering."""
    new_tracks = {}
    new_mapping = {}
    for new_id, old_id in enumerate(sorted(merged_binary.keys()), start=1):
        new_tracks[new_id] = merged_binary[old_id]
        new_mapping[new_id] = id_mapping[old_id]
    return new_tracks, new_mapping


def _prune_short_tracks(
    merged_binary,
    id_mapping,
    *,
    min_frames=2,
    min_peak_area=400,
    min_total_area=1600,
):
    """
    Prune likely-noisy tracks that are very short and tiny.
    Keep a track if it has enough temporal support OR enough area.
    """
    kept_binary = {}
    kept_mapping = {}
    removed = 0
    for obj_id, frame_masks in merged_binary.items():
        areas = [int(mask.sum()) for mask in frame_masks.values()]
        n_frames = len(areas)
        peak_area = max(areas) if areas else 0
        total_area = int(np.sum(areas)) if areas else 0
        keep = (
            n_frames >= int(min_frames)
            or peak_area >= int(min_peak_area)
            or total_area >= int(min_total_area)
        )
        if keep:
            kept_binary[obj_id] = frame_masks
            kept_mapping[obj_id] = id_mapping[obj_id]
        else:
            removed += 1
    kept_binary, kept_mapping = _reindex_tracks(kept_binary, kept_mapping)
    return kept_binary, kept_mapping, removed


def segment_keyframes(frames, keyframe_idxs, cfg, device="cuda"):
    """Run grid 32x32 on each keyframe → list of SAM2 masks."""
    print("\n[SAM2] Step 1: Segmenting keyframes with grid")
    sam2 = build_sam2(f"configs/sam2.1/{cfg['model_cfg']}", cfg["checkpoint"], device=device)
    gen = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=cfg.get("points_per_side", 32),
        pred_iou_thresh=cfg.get("pred_iou_thresh", 0.95),
        stability_score_thresh=cfg.get("stability_score_thresh", 0.92),
        stability_score_offset=cfg.get("stability_score_offset", 1.0),
        box_nms_thresh=cfg.get("box_nms_thresh", 0.5),
        crop_n_layers=cfg.get("crop_n_layers", 0),
        crop_n_points_downscale_factor=cfg.get("crop_n_points_downscale_factor", 2),
        min_mask_region_area=cfg.get("min_mask_region_area", 1200),
        output_mode=cfg.get("output_mode", "binary_mask"),
        points_per_batch=cfg.get("points_per_batch", 64),
    )
    keyframe_masks = {}
    for idx in keyframe_idxs:
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            masks = gen.generate(frames[idx])
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        keyframe_masks[idx] = masks
        print(f"  Frame {idx:5d}: {len(masks):3d} masks")
    del gen, sam2
    torch.cuda.empty_cache()
    return keyframe_masks


def _parse_scan_frame_idx(frame_path):
    """Extract numeric frame index from filename (e.g. frame-000042 → 42)."""
    m = re.search(r"frame-(\d+)", Path(frame_path).stem)
    if m is None:
        raise ValueError(f"Cannot extract frame idx from {frame_path}")
    return int(m.group(1))



def visualize_results(frames, obj_id_imgs, frame_paths, out_dir, stride=5):
    """Visualize obj_id_map as a color overlay and save to disk."""
    os.makedirs(out_dir, exist_ok=True)
    for fidx in range(0, len(frames), stride):
        scan_fidx = _parse_scan_frame_idx(frame_paths[fidx])
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
      obj_id/<scan_id>/frame-xxxxxx.jpg   — jpg ID map
      obj_id_pkl/<scan_id>.pkl            — {frame_idx: np.ndarray int32}
      color/<scan_id>/frame-xxxxxx.jpg    — RGB visualization
    Returns:
        obj_id_imgs:   {scan_fidx → (H,W) int32}
        merged_binary: {new_id   → {scan_fidx → bool mask}}
    """
    print("\n[SAM2] Step 2: Propagating with VideoPredictor")
    predictor = build_sam2_video_predictor(
        f"configs/sam2.1/{cfg['model_cfg']}",
        cfg["checkpoint"],
        device=device,
    )

    max_obj  = cfg.get("max_obj_ids", 50)
    prob_thr = cfg.get("mask_prob_threshold", 0.5)
    propagation_mode = str(cfg.get("propagation_mode", "per_keyframe")).strip().lower()
    global_prompt_keyframes = cfg.get("global_prompt_keyframes", "all")
    global_max_total_obj_ids = int(cfg.get("global_max_total_obj_ids", 0))

    all_binary = {}  # {global_id → {tmp_fidx → bool mask}}
    all_probs  = {}  # {global_id → {tmp_fidx → float32 prob_map}}
    global_counter = 1  # 0 = background

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    tmp_to_scan_fidx = {}  # maps tmp sequential index → original scan frame index

    try:
        # ── Copy frames to tmp dir with sequential names for SAM2 ──
        for tmp_idx, fp in enumerate(frame_paths):
            fp = Path(fp)
            dst = tmp_dir / f"{tmp_idx:06d}{fp.suffix.lower()}"
            shutil.copy2(fp, dst)
            tmp_to_scan_fidx[tmp_idx] = _parse_scan_frame_idx(fp)

        # ── Propagation strategy ──
        # "per_keyframe": one predictor state per keyframe (legacy behaviour).
        # "global": single predictor state seeded with masks from many/all keyframes.
        if propagation_mode == "global":
            keyframe_items = sorted(keyframe_masks.items(), key=lambda x: int(x[0]))
            if isinstance(global_prompt_keyframes, int):
                n_seed_kf = max(1, int(global_prompt_keyframes))
                keyframe_items = keyframe_items[:n_seed_kf]
            elif str(global_prompt_keyframes).strip().lower() != "all":
                try:
                    n_seed_kf = max(1, int(global_prompt_keyframes))
                    keyframe_items = keyframe_items[:n_seed_kf]
                except ValueError:
                    pass

            print(
                f"[SAM2] global propagation with {len(keyframe_items)} seed keyframes "
                f"(max_obj_ids/keyframe={max_obj}, global_max_total_obj_ids={global_max_total_obj_ids})"
            )

            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                state = predictor.init_state(video_path=str(tmp_dir))

                # Add all prompts first
                for kf_idx, masks in keyframe_items:
                    n_objs = min(len(masks), max_obj)
                    print(f"  Seed keyframe {kf_idx}: {n_objs} objectes")
                    if n_objs == 0:
                        continue
                    for m in masks[:n_objs]:
                        if global_max_total_obj_ids > 0 and global_counter > global_max_total_obj_ids:
                            break
                        gid = global_counter
                        predictor.add_new_mask(
                            state,
                            frame_idx=int(kf_idx),
                            obj_id=int(gid),
                            mask=m["segmentation"].astype(np.uint8),
                        )
                        all_probs[gid] = {}
                        global_counter += 1
                    if global_max_total_obj_ids > 0 and global_counter > global_max_total_obj_ids:
                        print("[SAM2] reached global_max_total_obj_ids cap")
                        break

                # If no prompt was added, continue with empty outputs.
                if all_probs:
                    # Forward propagation
                    for fidx, oids, logits in predictor.propagate_in_video(state):
                        for oid, logit in zip(oids, logits):
                            oid_int = int(oid)
                            if oid_int not in all_probs:
                                all_probs[oid_int] = {}
                            all_probs[oid_int][fidx] = \
                                torch.sigmoid(logit[0]).cpu().numpy().astype(np.float32)

                    # Backward propagation
                    for fidx, oids, logits in predictor.propagate_in_video(state, reverse=True):
                        for oid, logit in zip(oids, logits):
                            oid_int = int(oid)
                            if oid_int not in all_probs:
                                all_probs[oid_int] = {}
                            if fidx not in all_probs[oid_int]:
                                all_probs[oid_int][fidx] = \
                                    torch.sigmoid(logit[0]).cpu().numpy().astype(np.float32)

                    all_binary = {
                        gid: {fidx: (prob >= prob_thr) for fidx, prob in frame_probs.items()}
                        for gid, frame_probs in all_probs.items()
                    }

                predictor.reset_state(state)
        else:
            # masks[kf_idx] keys: segmentation, area, bbox, predicted_iou,
            #                     point_coords, stability_score, crop_box
            for kf_idx, masks in keyframe_masks.items():
                n_objs = min(len(masks), max_obj)
                print(f"  Keyframe {kf_idx}: {n_objs} objectes")
                if n_objs == 0:
                    # SAM2 video predictor requires at least one prompt (point/mask)
                    # before propagate_in_video; skip empty keyframes safely.
                    continue

                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    state = predictor.init_state(video_path=str(tmp_dir))
                    for local_oid, m in enumerate(masks[:n_objs]):
                        predictor.add_new_mask(
                            state, frame_idx=int(kf_idx),
                            obj_id=local_oid,
                            mask=m["segmentation"].astype(np.uint8),
                        )

                    local_probs = {i: {} for i in range(n_objs)}

                    # Forward propagation
                    for fidx, oids, logits in predictor.propagate_in_video(state):
                        for oid, logit in zip(oids, logits):
                            local_probs[int(oid)][fidx] = \
                                torch.sigmoid(logit[0]).cpu().numpy().astype(np.float32)

                    # Backward propagation (fill frames before keyframe)
                    for fidx, oids, logits in predictor.propagate_in_video(state, reverse=True):
                        for oid, logit in zip(oids, logits):
                            if fidx not in local_probs[int(oid)]:
                                local_probs[int(oid)][fidx] = \
                                    torch.sigmoid(logit[0]).cpu().numpy().astype(np.float32)

                # Store probs and binary masks under the same global_id
                for local_oid in range(n_objs):
                    gid = global_counter
                    all_probs[gid]  = local_probs[local_oid]
                    all_binary[gid] = {
                        fidx: (prob >= prob_thr)
                        for fidx, prob in local_probs[local_oid].items()
                    }
                    global_counter += 1

                predictor.reset_state(state)

        # ── Merge overlapping tracks by IoU on binary masks ──
        merge_iou_thr = float(cfg.get("merge_iou_threshold", 0.6))
        merge_iou_reduction = str(cfg.get("merge_iou_reduction", "max"))
        merged_binary, id_mapping = merge_tracks_by_iou(
            all_binary,
            iou_threshold=merge_iou_thr,
            iou_reduction=merge_iou_reduction,
        )

        # ── Prune short/tiny tracks (high impact, low cost denoising) ──
        prune_short_tracks = bool(cfg.get("prune_short_tracks", True))
        if prune_short_tracks:
            merged_binary, id_mapping, n_removed = _prune_short_tracks(
                merged_binary,
                id_mapping,
                min_frames=int(cfg.get("track_min_frames", 2)),
                min_peak_area=int(cfg.get("track_min_peak_area", 400)),
                min_total_area=int(cfg.get("track_min_total_area", 1600)),
            )
            print(f"[SAM2] short-track prune removed {n_removed} tracks")

        # ── Reconstruct probability maps aligned to new merged IDs ──
        # For each new ID, take pixel-wise max across all fused original IDs
        merged_probs = {}
        for new_id, orig_ids in id_mapping.items():
            fused = {}
            for orig_id in orig_ids:
                for fidx, prob in all_probs[orig_id].items():
                    if fidx not in fused:
                        fused[fidx] = prob.copy()
                    else:
                        np.maximum(fused[fidx], prob, out=fused[fidx]) # pixel-wise max
            merged_probs[new_id] = fused

        # Guard: handle scenes with zero detections
        if not merged_probs:
            print("[SAM2] Avís: cap objecte detectat en aquesta escena")
            H, W = next(iter(
                cv2.imread(str(frame_paths[0])).shape[:2]
                for _ in [None]
            )) if frame_paths else (540, 960)
        else:
            sample_prob = next(iter(next(iter(merged_probs.values())).values()))
            H, W = sample_prob.shape

        # ── Initialise ALL frames as zero-maps (background) ──
        # Ensures frames with no detections still appear on disk
        all_scan_fidxs = sorted({
            tmp_to_scan_fidx[i] for i in range(len(frame_paths))
        })
        obj_id_imgs = {f: np.zeros((H, W), dtype=np.int32) for f in all_scan_fidxs}
        score_imgs  = {f: np.zeros((H, W), dtype=np.float32) for f in all_scan_fidxs}
        # Temporal-support prior: tracks seen in more frames get a small boost.
        # This is a coherent alternative to "per-pixel majority across frames"
        # (which is not stable under camera motion).
        temporal_vote_weight = float(cfg.get("temporal_vote_weight", 0.15))
        track_support = {
            obj_id: len(frame_dict) for obj_id, frame_dict in merged_binary.items()
        }
        max_support = max(track_support.values()) if track_support else 1

        # ── Assign each pixel to the highest-confidence object ──
        min_mask_pixels_per_frame = int(cfg.get("min_mask_pixels_per_frame", 0))
        for new_id, frame_dict in merged_binary.items():
            support_norm = float(track_support.get(new_id, 0)) / float(max_support)
            for tmp_fidx, mask_bin in frame_dict.items():
                if int(mask_bin.sum()) < min_mask_pixels_per_frame:
                    continue
                scan_fidx = tmp_to_scan_fidx[tmp_fidx]
                prob_map  = merged_probs[new_id].get(tmp_fidx)

                if prob_map is not None:
                    # Multiplicative temporal prior: amplifies confident tracks
                    # with longer support, without replacing SAM2 confidence.
                    effective_score = prob_map * (1.0 + temporal_vote_weight * support_norm)
                    update = mask_bin & (effective_score > score_imgs[scan_fidx])
                    score_imgs[scan_fidx][update] = effective_score[update]
                else:
                    update = mask_bin & (obj_id_imgs[scan_fidx] == 0)

                obj_id_imgs[scan_fidx][update] = int(new_id)

        # ── Save in VLSG/Object-X format ──
        obj_dir = os.path.join(output_dir, "obj_id", scan_id)
        pkl_dir = os.path.join(output_dir, "obj_id_pkl")
        color_dir = os.path.join(output_dir, "color", scan_id)
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(obj_dir, exist_ok=True)
        os.makedirs(pkl_dir, exist_ok=True)

        for scan_fidx, obj_id_map in obj_id_imgs.items():
            # obj_id real
            cv2.imwrite(
                os.path.join(obj_dir, f"frame-{scan_fidx:06d}.jpg"),
                obj_id_map.astype(np.uint16)
            )

            # visualization color
            color_vis = colorize_obj_map(obj_id_map)
            cv2.imwrite(
                os.path.join(color_dir, f"frame-{scan_fidx:06d}.jpg"),
                cv2.cvtColor(color_vis, cv2.COLOR_RGB2BGR)
            )

        # Save full int32 maps as pickle using zero-padded string frame ids
        # to match the canonical 3RScan/Object-X style ("000042").
        obj_id_imgs_for_pkl = {
            f"{int(scan_fidx):06d}": obj_id_map
            for scan_fidx, obj_id_map in obj_id_imgs.items()
        }
        with open(os.path.join(pkl_dir, f"{scan_id}.pkl"), "wb") as f:
            pickle.dump(obj_id_imgs_for_pkl, f)

        merged_binary_scan = {
            new_id: {
                tmp_to_scan_fidx[tmp_fidx]: mask_bin
                for tmp_fidx, mask_bin in frame_dict.items()
            }
            for new_id, frame_dict in merged_binary.items()
        }

        n_empty = sum(1 for m in obj_id_imgs.values() if m.max() == 0)
        print(f"[SAM2] {len(merged_binary)} objects after fusion")
        print(f"[SAM2] {len(obj_id_imgs)} frames saved ({n_empty} empty / background-only)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        del predictor
        torch.cuda.empty_cache()

    return obj_id_imgs, merged_binary_scan
