"""
SAM2: AutomaticMaskGenerator (grid) on keyframes + VideoPredictor propagation.
"""
import os, shutil, tempfile, pickle
import sys
import torch, numpy as np, cv2
from pathlib import Path
from dataclasses import dataclass

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAM2_VENDOR_ROOT = _REPO_ROOT / "models" / "sam2"
if str(_SAM2_VENDOR_ROOT) not in sys.path:
    # Ensure vendored SAM2 package is importable when running scripts directly.
    sys.path.insert(0, str(_SAM2_VENDOR_ROOT))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.mask_utils import merge_tracks_by_iou, visualize_masks_on_frame
from sam2_config import build_runtime_config
from sam2_helpers import (
    parse_scan_frame_idx,
    prune_short_tracks,
    save_obj_id_and_color_frames,
)


def ensure_sam2_postprocess_ready(device="cuda"):
    """Fail fast if the SAM2 post-processing extension is unavailable."""
    from sam2.utils.misc import get_connected_components

    requested_device = str(device).split(":", 1)[0]
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("SAM2 post-processing requires CUDA, but CUDA is unavailable.")
        test_mask = torch.zeros((1, 1, 8, 8), dtype=torch.bool, device="cuda")
        test_mask[:, :, 2:4, 2:4] = True
        labels, areas = get_connected_components(test_mask)
        if labels.shape != test_mask.shape or areas.shape != test_mask.shape:
            raise RuntimeError("SAM2 post-processing extension returned invalid tensor shapes.")
        print("[SAM2] post-processing extension check: CUDA OK", flush=True)
    else:
        import sam2._C  # noqa: F401
        print("[SAM2] post-processing extension check: import OK", flush=True)


def _sam2_apply_postprocessing_enabled():
    """Control SAM2 C++ post-processing usage via env var."""
    return str(os.environ.get("OBJECTX_SAM2_APPLY_POSTPROCESS", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }


def segment_keyframes(frames, keyframe_idxs, cfg, device="cuda"):
    """Run grid 32x32 on each keyframe → list of SAM2 masks."""
    print("\n[SAM2] Step 1: Segmenting keyframes with grid")
    rcfg = build_runtime_config(cfg)
    sam2 = build_sam2(
        f"configs/sam2.1/{rcfg.model.model_cfg}",
        rcfg.model.checkpoint,
        device=device,
        apply_postprocessing=_sam2_apply_postprocessing_enabled(),
    )
    gen = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=rcfg.keyframes.points_per_side,
        pred_iou_thresh=rcfg.keyframes.pred_iou_thresh,
        stability_score_thresh=rcfg.keyframes.stability_score_thresh,
        stability_score_offset=rcfg.keyframes.stability_score_offset,
        box_nms_thresh=rcfg.keyframes.box_nms_thresh,
        crop_n_layers=rcfg.keyframes.crop_n_layers,
        crop_n_points_downscale_factor=rcfg.keyframes.crop_n_points_downscale_factor,
        min_mask_region_area=rcfg.keyframes.min_mask_region_area,
        output_mode=rcfg.keyframes.output_mode,
        points_per_batch=rcfg.keyframes.points_per_batch,
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


def visualize_results(frames, obj_id_imgs, frame_paths, out_dir, stride=5):
    """Visualize obj_id_map as a color overlay and save to disk."""
    os.makedirs(out_dir, exist_ok=True)
    for fidx in range(0, len(frames), stride):
        scan_fidx = parse_scan_frame_idx(frame_paths[fidx])
        obj_map = obj_id_imgs.get(scan_fidx)
        if obj_map is None or obj_map.max() == 0:
            continue
        unique_ids = np.unique(obj_map[obj_map > 0])
        masks = [(obj_map == uid) for uid in unique_ids]
        vis = visualize_masks_on_frame(frames[fidx], masks)
        cv2.imwrite(
            os.path.join(out_dir, f"vis_{fidx:06d}.jpg"),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


@dataclass
class PropagationResult:
    all_binary: dict
    all_probs: dict


@dataclass
class MergeResult:
    merged_binary: dict
    id_mapping: dict
    merged_probs: dict


def _copy_frames_for_predictor(frame_paths, tmp_dir):
    """Copy scene frames into SAM2 sequential naming and return tmp->scan index map."""
    tmp_to_scan_fidx = {}
    for tmp_idx, fp in enumerate(frame_paths):
        fp_path = Path(fp)
        dst = tmp_dir / f"{tmp_idx:06d}{fp_path.suffix.lower()}"
        shutil.copy2(fp_path, dst)
        tmp_to_scan_fidx[tmp_idx] = parse_scan_frame_idx(fp_path)
    return tmp_to_scan_fidx


def _score_mask_for_propagation(mask_entry, frame_shape):
    h, w = frame_shape
    frame_area = max(float(h * w), 1.0)
    area = float(mask_entry.get("area", 0.0))
    pred_iou = float(mask_entry.get("predicted_iou", 0.0))
    stability = float(mask_entry.get("stability_score", pred_iou))
    bbox = mask_entry.get("bbox") or [0.0, 0.0, float(w), float(h)]
    _, _, bw, bh = [float(v) for v in bbox]
    bbox_area = max(float(bw * bh), 1.0)
    bbox_fill = area / bbox_area
    area_frac = area / frame_area
    quality = 0.52 * pred_iou + 0.48 * stability
    score = 0.58 * quality + 0.42 * bbox_fill
    return {
        "score": float(score),
        "pred_iou": pred_iou,
        "stability": stability,
        "area_frac": area_frac,
    }


def _prepare_keyframe_masks_for_propagation(masks, max_obj, frame_shape, rcfg):
    prop_cfg = rcfg.propagation
    candidate_limit = min(len(masks), max(max_obj, max_obj * prop_cfg.candidate_multiplier))
    candidates = masks[:candidate_limit]

    scored = []
    for mask in candidates:
        stats = _score_mask_for_propagation(mask, frame_shape)
        if prop_cfg.hard_filter_enabled:
            if (
                stats["score"] < prop_cfg.min_mask_score
                or stats["pred_iou"] < prop_cfg.min_pred_iou
                or stats["stability"] < prop_cfg.min_stability
                or stats["area_frac"] < prop_cfg.tiny_area_frac
            ):
                continue
        scored.append((mask, stats))

    scored.sort(key=lambda x: (x[1]["score"], x[1]["pred_iou"], x[1]["stability"]), reverse=True)

    if prop_cfg.dedupe_enabled:
        selected = []
        selected_segs = []
        selected_ids = set()
        for mask, _ in scored:
            seg = mask["segmentation"].astype(bool, copy=False)
            keep = True
            for kept_seg in selected_segs:
                intersection = np.logical_and(seg, kept_seg).sum()
                if intersection <= 0:
                    continue
                union = np.logical_or(seg, kept_seg).sum()
                iou = float(intersection / max(union, 1))
                containment = float(intersection / max(seg.sum(), 1))
                if iou >= prop_cfg.dedupe_iou or containment >= prop_cfg.dedupe_containment:
                    keep = False
                    break
            if keep:
                selected.append(mask)
                selected_segs.append(seg)
                selected_ids.add(id(mask))
            if len(selected) >= max_obj:
                break
    else:
        selected = [m for m, _ in scored[:max_obj]]
        selected_ids = {id(m) for m in selected}

    if len(selected) < min(prop_cfg.min_keep, max_obj):
        fallback = [m for m in masks[:max_obj] if id(m) not in selected_ids]
        for mask in fallback:
            selected.append(mask)
            selected_ids.add(id(mask))
            if len(selected) >= min(prop_cfg.min_keep, max_obj):
                break
            if len(selected) >= max_obj:
                break

    return selected[:max_obj]


def _track_overlap_stats(track_a, track_b):
    shared = set(track_a.keys()) & set(track_b.keys())
    if not shared:
        return 0, 0.0, 0.0
    peak_iou = 0.0
    peak_containment = 0.0
    valid = 0
    for fidx in shared:
        a = track_a[fidx]
        b = track_b[fidx]
        inter = float(np.logical_and(a, b).sum())
        if inter <= 0:
            continue
        valid += 1
        union = float(np.logical_or(a, b).sum())
        iou = inter / max(union, 1.0)
        containment = max(
            inter / max(float(a.sum()), 1.0),
            inter / max(float(b.sum()), 1.0),
        )
        peak_iou = max(peak_iou, iou)
        peak_containment = max(peak_containment, containment)
    return valid, peak_iou, peak_containment


def _merge_tracks_with_overlap_heuristics(merged_binary, id_mapping, rcfg):
    ids = sorted(merged_binary.keys())
    parent = {obj_id: obj_id for obj_id in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, ida in enumerate(ids):
        for idb in ids[i + 1 :]:
            shared, peak_iou, peak_containment = _track_overlap_stats(
                merged_binary[ida],
                merged_binary[idb],
            )
            if shared < rcfg.merge.min_shared_frames:
                continue
            if peak_iou >= rcfg.merge.iou_threshold:
                union(ida, idb)
            elif peak_iou >= rcfg.merge.relaxed_iou_threshold and peak_containment >= rcfg.merge.containment_threshold:
                union(ida, idb)
            elif peak_iou >= rcfg.merge.peak_iou_threshold and peak_containment >= rcfg.merge.containment_threshold:
                union(ida, idb)

    groups = {}
    for old_id in ids:
        root = find(old_id)
        groups.setdefault(root, []).append(old_id)

    new_binary = {}
    new_mapping = {}
    for new_id, (_, members) in enumerate(sorted(groups.items(), key=lambda x: min(x[1])), start=1):
        frame_union = {}
        for mid in members:
            for fidx, mask in merged_binary[mid].items():
                if fidx in frame_union:
                    frame_union[fidx] = np.logical_or(frame_union[fidx], mask)
                else:
                    frame_union[fidx] = mask.copy()
        new_binary[new_id] = frame_union
        src_ids = []
        for mid in members:
            src_ids.extend(id_mapping.get(mid, [mid]))
        new_mapping[new_id] = sorted(set(int(x) for x in src_ids))
    return new_binary, new_mapping


def _prune_low_support_tracks(merged_binary, id_mapping, max_frames=2, max_peak_area=3500):
    kept_binary = {}
    kept_mapping = {}
    removed = 0
    for obj_id, frame_masks in merged_binary.items():
        n_frames = len(frame_masks)
        peak_area = max((int(m.sum()) for m in frame_masks.values()), default=0)
        if n_frames <= int(max_frames) and peak_area <= int(max_peak_area):
            removed += 1
            continue
        kept_binary[obj_id] = frame_masks
        kept_mapping[obj_id] = id_mapping[obj_id]
    return kept_binary, kept_mapping, removed


def _propagate_global(
    predictor,
    tmp_dir,
    keyframe_masks,
    max_obj,
    prob_thr,
    global_prompt_keyframes,
    global_max_total_obj_ids,
    device,
    rcfg,
):
    all_binary = {}
    all_probs = {}
    global_counter = 1

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

    print(f"[SAM2] global propagation with {len(keyframe_items)} seed keyframes "
          f"(max_obj_ids/keyframe={max_obj}, global_max_total_obj_ids={global_max_total_obj_ids})")

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(tmp_dir))
        for kf_idx, masks in keyframe_items:
            n_objs = min(len(masks), max_obj)
            print(f"  Seed keyframe {kf_idx}: {n_objs} objects")
            if n_objs == 0:
                continue
            prepared_masks = _prepare_keyframe_masks_for_propagation(
                masks,
                max_obj=n_objs,
                frame_shape=masks[0]["segmentation"].shape if masks else (540, 960),
                rcfg=rcfg,
            ) if masks else []
            for m in prepared_masks:
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

        if all_probs:
            for fidx, oids, logits in predictor.propagate_in_video(state):
                for oid, logit in zip(oids, logits):
                    oid_int = int(oid)
                    if oid_int not in all_probs:
                        all_probs[oid_int] = {}
                    all_probs[oid_int][fidx] = torch.sigmoid(logit[0]).cpu().numpy().astype(np.float32)

            for fidx, oids, logits in predictor.propagate_in_video(state, reverse=True):
                for oid, logit in zip(oids, logits):
                    oid_int = int(oid)
                    if oid_int not in all_probs:
                        all_probs[oid_int] = {}
                    if fidx not in all_probs[oid_int]:
                        all_probs[oid_int][fidx] = torch.sigmoid(logit[0]).cpu().numpy().astype(np.float32)

            all_binary = {
                gid: {fidx: (prob >= prob_thr) for fidx, prob in frame_probs.items()}
                for gid, frame_probs in all_probs.items()
            }

        predictor.reset_state(state)
    return PropagationResult(all_binary=all_binary, all_probs=all_probs)


def _propagate_per_keyframe(predictor, tmp_dir, keyframe_masks, max_obj, prob_thr, device, rcfg):
    all_binary = {}
    all_probs = {}
    global_counter = 1

    for kf_idx, masks in keyframe_masks.items():
        n_objs = min(len(masks), max_obj)
        print(f"  Keyframe {kf_idx}: {n_objs} objects")
        if n_objs == 0:
            continue

        frame_shape = masks[0]["segmentation"].shape if masks else (540, 960)
        prepared_masks = _prepare_keyframe_masks_for_propagation(
            masks,
            max_obj=n_objs,
            frame_shape=frame_shape,
            rcfg=rcfg,
        )
        if not prepared_masks:
            continue

        chunk_size = max(1, int(rcfg.propagation.objects_per_chunk))
        for chunk_start in range(0, len(prepared_masks), chunk_size):
            chunk_masks = prepared_masks[chunk_start : chunk_start + chunk_size]
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                state = predictor.init_state(video_path=str(tmp_dir))
                for local_oid, m in enumerate(chunk_masks):
                    predictor.add_new_mask(
                        state,
                        frame_idx=int(kf_idx),
                        obj_id=local_oid,
                        mask=m["segmentation"].astype(np.uint8),
                    )

                local_probs = {i: {} for i in range(len(chunk_masks))}
                for fidx, oids, logits in predictor.propagate_in_video(state):
                    for oid, logit in zip(oids, logits):
                        local_probs[int(oid)][fidx] = torch.sigmoid(logit[0]).cpu().numpy().astype(np.float32)

                for fidx, oids, logits in predictor.propagate_in_video(state, reverse=True):
                    for oid, logit in zip(oids, logits):
                        if fidx not in local_probs[int(oid)]:
                            local_probs[int(oid)][fidx] = torch.sigmoid(logit[0]).cpu().numpy().astype(np.float32)

            for local_oid in range(len(chunk_masks)):
                gid = global_counter
                all_probs[gid] = local_probs[local_oid]
                all_binary[gid] = {
                    fidx: (prob >= prob_thr)
                    for fidx, prob in local_probs[local_oid].items()
                }
                global_counter += 1

            predictor.reset_state(state)

    return PropagationResult(all_binary=all_binary, all_probs=all_probs)


def _merge_and_fuse_tracks(propagation_result, rcfg):
    """Merge propagated tracks and fuse per-track probability maps."""
    merged_binary, id_mapping = merge_tracks_by_iou(
        propagation_result.all_binary,
        iou_threshold=rcfg.merge.iou_threshold,
        iou_reduction=rcfg.merge.iou_reduction,
    )

    if rcfg.prune.enabled:
        merged_binary, id_mapping, n_removed = prune_short_tracks(
            merged_binary,
            id_mapping,
            min_frames=rcfg.prune.min_frames,
            min_peak_area=rcfg.prune.min_peak_area,
            min_total_area=rcfg.prune.min_total_area,
        )
        print(f"[SAM2] short-track prune removed {n_removed} tracks")

    merged_binary, id_mapping = _merge_tracks_with_overlap_heuristics(
        merged_binary, id_mapping, rcfg
    )
    merged_binary, id_mapping, n_removed = _prune_low_support_tracks(
        merged_binary, id_mapping
    )
    if n_removed > 0:
        print(f"[SAM2] low-support prune removed {n_removed} tracks")

    merged_probs = {}
    for new_id, orig_ids in id_mapping.items():
        fused = {}
        for orig_id in orig_ids:
            for fidx, prob in propagation_result.all_probs[orig_id].items():
                if fidx not in fused:
                    fused[fidx] = prob.copy()
                else:
                    np.maximum(fused[fidx], prob, out=fused[fidx])
        merged_probs[new_id] = fused
    return MergeResult(
        merged_binary=merged_binary,
        id_mapping=id_mapping,
        merged_probs=merged_probs,
    )


def _compose_obj_id_images(merged_binary, merged_probs, tmp_to_scan_fidx, frame_paths, rcfg):
    if not merged_probs:
        print("[SAM2] Warning: no objects detected in this scene")
        H, W = next(iter(cv2.imread(str(frame_paths[0])).shape[:2] for _ in [None])) if frame_paths else (540, 960)
    else:
        sample_prob = next(iter(next(iter(merged_probs.values())).values()))
        H, W = sample_prob.shape

    all_scan_fidxs = sorted({tmp_to_scan_fidx[i] for i in range(len(frame_paths))})
    obj_id_imgs = {f: np.zeros((H, W), dtype=np.int32) for f in all_scan_fidxs}
    score_imgs = {f: np.zeros((H, W), dtype=np.float32) for f in all_scan_fidxs}
    temporal_vote_weight = rcfg.propagation.temporal_vote_weight
    track_support = {obj_id: len(frame_dict) for obj_id, frame_dict in merged_binary.items()}
    max_support = max(track_support.values()) if track_support else 1
    min_mask_pixels_per_frame = rcfg.propagation.min_mask_pixels_per_frame

    for new_id, frame_dict in merged_binary.items():
        support_norm = float(track_support.get(new_id, 0)) / float(max_support)
        for tmp_fidx, mask_bin in frame_dict.items():
            if int(mask_bin.sum()) < min_mask_pixels_per_frame:
                continue
            scan_fidx = tmp_to_scan_fidx[tmp_fidx]
            prob_map = merged_probs[new_id].get(tmp_fidx)
            if prob_map is not None:
                effective_score = prob_map * (1.0 + temporal_vote_weight * support_norm)
                update = mask_bin & (effective_score > score_imgs[scan_fidx])
                score_imgs[scan_fidx][update] = effective_score[update]
            else:
                update = mask_bin & (obj_id_imgs[scan_fidx] == 0)
            obj_id_imgs[scan_fidx][update] = int(new_id)
    return obj_id_imgs


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
    rcfg = build_runtime_config(cfg)
    predictor = build_sam2_video_predictor(
        f"configs/sam2.1/{rcfg.model.model_cfg}",
        rcfg.model.checkpoint,
        device=device,
        apply_postprocessing=_sam2_apply_postprocessing_enabled(),
    )

    max_obj = rcfg.propagation.max_obj_ids
    prob_thr = rcfg.propagation.mask_prob_threshold
    propagation_mode = rcfg.propagation.propagation_mode
    global_prompt_keyframes = rcfg.propagation.global_prompt_keyframes
    global_max_total_obj_ids = rcfg.propagation.global_max_total_obj_ids

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    try:
        tmp_to_scan_fidx = _copy_frames_for_predictor(frame_paths, tmp_dir)

        if propagation_mode == "global":
            propagation_result = _propagate_global(
                predictor,
                tmp_dir,
                keyframe_masks,
                max_obj,
                prob_thr,
                global_prompt_keyframes,
                global_max_total_obj_ids,
                device,
                rcfg,
            )
        else:
            propagation_result = _propagate_per_keyframe(
                predictor,
                tmp_dir,
                keyframe_masks,
                max_obj,
                prob_thr,
                device,
                rcfg,
            )

        merge_result = _merge_and_fuse_tracks(propagation_result, rcfg)
        obj_id_imgs = _compose_obj_id_images(
            merge_result.merged_binary,
            merge_result.merged_probs,
            tmp_to_scan_fidx,
            frame_paths,
            rcfg,
        )

        # ── Save in VLSG/Object-X format ──
        pkl_dir = os.path.join(output_dir, "obj_id_pkl")
        os.makedirs(pkl_dir, exist_ok=True)
        save_obj_id_and_color_frames(obj_id_imgs, scan_id=scan_id, output_dir=output_dir)

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
            for new_id, frame_dict in merge_result.merged_binary.items()
        }

        n_empty = sum(1 for m in obj_id_imgs.values() if m.max() == 0)
        print(f"[SAM2] {len(merge_result.merged_binary)} objects after fusion")
        print(f"[SAM2] {len(obj_id_imgs)} frames saved ({n_empty} empty / background-only)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        del predictor
        torch.cuda.empty_cache()

    return obj_id_imgs, merged_binary_scan
