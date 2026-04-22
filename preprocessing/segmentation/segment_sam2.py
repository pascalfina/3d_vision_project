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


def segment_keyframes(frames, keyframe_idxs, cfg, device="cuda"):
    """Run grid 32x32 on each keyframe → list of SAM2 masks."""
    print("\n[SAM2] Step 1: Segmenting keyframes with grid")
    sam2 = build_sam2(f"configs/sam2.1/{cfg['model_cfg']}", cfg["checkpoint"], device=device)
    gen = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=cfg.get("points_per_side", 32),
        pred_iou_thresh=cfg.get("pred_iou_thresh", 0.86),
        stability_score_thresh=cfg.get("stability_score_thresh", 0.92),
        stability_score_offset=cfg.get("stability_score_offset", 1.0),
        box_nms_thresh=cfg.get("box_nms_thresh", 0.7),
        crop_n_layers=cfg.get("crop_n_layers", 1),
        crop_n_points_downscale_factor=cfg.get("crop_n_points_downscale_factor", 2),
        min_mask_region_area=cfg.get("min_mask_region_area", 200),
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
      obj_id/<scan_id>/frame-xxxxxx.png   — uint16 lossless ID map
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

        # ── Propagate from each keyframe ──
        # masks[kf_idx] keys: segmentation, area, bbox, predicted_iou,
        #                     point_coords, stability_score, crop_box
        for kf_idx, masks in keyframe_masks.items():
            n_objs = min(len(masks), max_obj)
            print(f"  Keyframe {kf_idx}: {n_objs} objectes")

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
        merged_binary, id_mapping = merge_tracks_by_iou(all_binary, iou_threshold=0.5)

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

        # ── Assign each pixel to the highest-confidence object ──
        for new_id, frame_dict in merged_binary.items():
            for tmp_fidx, mask_bin in frame_dict.items():
                scan_fidx = tmp_to_scan_fidx[tmp_fidx]
                prob_map  = merged_probs[new_id].get(tmp_fidx)

                if prob_map is not None:
                    update = mask_bin & (prob_map > score_imgs[scan_fidx])
                    score_imgs[scan_fidx][update] = prob_map[update]
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

        # Save full int32 maps as pickle
        with open(os.path.join(pkl_dir, f"{scan_id}.pkl"), "wb") as f:
            pickle.dump(obj_id_imgs, f)

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
