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


def refine_keyframes_with_mask_preview(frames, candidate_groups, cfg, device="cuda"):
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

    print(
        "[SAM2] Refining keyframes with preview "
        f"(points_per_side={preview_points_per_side}, top_k_masks={preview_top_k})"
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

    def score_preview_masks(masks):
        usable = masks[:preview_top_k]
        mask_count = len(usable)
        total_area = float(sum(m["area"] for m in usable))
        useful_masks = 0
        weighted_utility = 0.0

        for mask in usable:
            seg = mask.get("segmentation")
            area = float(mask.get("area", 0.0))
            if seg is None or area <= 0.0:
                continue
            seg = seg.astype(bool, copy=False)
            row_sums = seg.sum(axis=1, dtype=np.int64)
            if row_sums.sum() <= 0:
                continue
            h = int(seg.shape[0])
            split = max(1, h // 2)
            upper_area = float(row_sums[:split].sum())
            upper_ratio = upper_area / max(area, 1.0)
            row_idx = np.arange(h, dtype=np.float32)
            center_y = float((row_sums.astype(np.float32) * row_idx).sum() / area)
            center_y /= max(float(h - 1), 1.0)
            position_weight = max(0.35, 1.20 - 0.80 * center_y)
            weighted_utility += np.sqrt(area) * position_weight * (0.70 + upper_ratio)
            if upper_ratio >= 0.18 or center_y <= 0.62:
                useful_masks += 1

        return {
            "mask_count": mask_count,
            "useful_masks": useful_masks,
            "weighted_utility": float(weighted_utility),
            "total_area": total_area,
        }

    selected = []
    try:
        for group_idx, group in enumerate(candidate_groups, start=1):
            best_idx = None
            best_score = None
            scored = []
            for idx in group:
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    masks = gen.generate(frames[idx])
                masks = sorted(masks, key=lambda x: x["area"], reverse=True)
                preview_stats = score_preview_masks(masks)
                score = (
                    preview_stats["useful_masks"],
                    preview_stats["weighted_utility"],
                    preview_stats["mask_count"],
                    preview_stats["total_area"],
                )
                scored.append((idx, preview_stats))
                if best_score is None or score > best_score:
                    best_idx = idx
                    best_score = score
            scored_txt = ", ".join(
                f"{idx}:{stats['mask_count']}m/{stats['useful_masks']}u/{int(stats['weighted_utility'])}w"
                for idx, stats in scored
            )
            print(
                f"[SAM2] preview group {group_idx}: {scored_txt} -> select {best_idx}",
                flush=True,
            )
            selected.append(int(best_idx))
    finally:
        del gen, sam2
        torch.cuda.empty_cache()

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
    max_keyframe_hops,
    min_shared_frames,
    area_ratio_cap,
):
    num_tracks = int(track_seen.shape[0])
    parent = list(range(num_tracks))
    peak_areas = track_areas.max(axis=1).astype(np.float32, copy=False)
    compared_pairs = 0
    skipped_same_keyframe = 0
    skipped_far_keyframe = 0
    skipped_area_ratio = 0

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
        f"(max_keyframe_hops={max_keyframe_hops}, "
        f"min_shared_frames={min_shared_frames}, "
        f"area_ratio_cap={area_ratio_cap})",
        flush=True,
    )
    for i in range(num_tracks):
        if i == 0 or (i + 1) % 8 == 0 or (i + 1) == num_tracks:
            print(f"[SAM2] merge progress {i + 1}/{num_tracks}", flush=True)
        key_pos_i = int(track_key_positions[i])
        area_i = max(float(peak_areas[i]), 1.0)
        for j in range(i + 1, num_tracks):
            key_pos_j = int(track_key_positions[j])
            if key_pos_i == key_pos_j:
                skipped_same_keyframe += 1
                continue
            if max_keyframe_hops >= 0 and abs(key_pos_i - key_pos_j) > max_keyframe_hops:
                skipped_far_keyframe += 1
                continue
            area_j = max(float(peak_areas[j]), 1.0)
            if area_ratio_cap > 0.0 and (max(area_i, area_j) / min(area_i, area_j)) > area_ratio_cap:
                skipped_area_ratio += 1
                continue
            shared = np.flatnonzero(track_seen[i] & track_seen[j])
            if shared.size < min_shared_frames:
                continue
            compared_pairs += 1

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
            valid = union_areas > 0
            if not np.any(valid):
                continue

            mean_iou = float(np.mean(intersections[valid] / union_areas[valid]))
            if mean_iou >= iou_threshold:
                union(i, j)

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
        f"far_keyframe_skipped={skipped_far_keyframe}, "
        f"area_ratio_skipped={skipped_area_ratio}",
        flush=True,
    )
    print(f"[SAM2] {num_tracks} tracks -> {len(id_mapping)} merged objects")
    return id_mapping, orig_to_new


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
        num_tracks = _count_tracks(dict(ordered_keyframe_items), max_obj)
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
            for key_pos, (kf_idx, masks) in enumerate(ordered_keyframe_items):
                n_objs = min(len(masks), max_obj)
                limited_masks = masks[:n_objs]
                print(f"  Keyframe {kf_idx}: {n_objs} objectes")
                for chunk_start in range(0, n_objs, obj_chunk_size):
                    chunk_masks = limited_masks[chunk_start : chunk_start + obj_chunk_size]
                    chunk_end = chunk_start + len(chunk_masks) - 1
                    print(
                        f"    pass1 chunk {chunk_start}-{chunk_end} "
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
                max_keyframe_hops=max_keyframe_hops,
                min_shared_frames=min_shared_frames,
                area_ratio_cap=area_ratio_cap,
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
