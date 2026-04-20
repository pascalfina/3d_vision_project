"""
MUSt3R: predicts poses (N,4,4) and depth maps (H,W) from RGB only.
Depth is extracted from the Z channel of the pointmaps (pts3d[...,2]).

Chunked mode: when OBJECTX_MUST3R_CHUNK_SIZE > 0, long sequences are split
into overlapping windows. Each chunk runs independently (peak VRAM scales
with chunk length, not total length). Subsequent chunks are aligned to the
first via Umeyama similarity (SE(3)+scale) computed on overlap-frame camera
centers; depths are scaled accordingly.
"""
import os, re, sys, numpy as np, torch, cv2
from pathlib import Path
from typing import List, Optional, Tuple
from utils.io_utils import (
    save_confidence,
    save_depth,
    save_depth_raw,
    save_poses,
    save_xyz_map,
)
from utils import scan3r


def _frame_id_from_path(frame_path: str, fallback_idx: int) -> str:
    match = re.search(r"frame-(\d+)", Path(frame_path).stem)
    if match:
        return match.group(1)
    return f"{int(fallback_idx):06d}"


def _to_numpy(t):
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _ransac_scale(
    ratios: np.ndarray, n_iters: int = 400, sample_size: int = 5,
    thresh: float = 0.15, seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """RANSAC over pair distance ratios. Returns (scale, inlier_mask)."""
    n = len(ratios)
    rng = np.random.default_rng(seed)
    if n < sample_size:
        return float(np.median(ratios)), np.ones(n, dtype=bool)
    best_inliers = None
    best_count = 0
    for _ in range(n_iters):
        idx = rng.choice(n, sample_size, replace=False)
        s = float(np.median(ratios[idx]))
        if not (s > 0):
            continue
        inl = np.abs(ratios / s - 1.0) <= thresh
        c = int(inl.sum())
        if c > best_count:
            best_count = c
            best_inliers = inl
    if best_inliers is None or best_count < max(3, n // 10):
        med = float(np.median(ratios))
        return med, np.ones(n, dtype=bool)
    return float(np.median(ratios[best_inliers])), best_inliers


def _kabsch(src: np.ndarray, dst: np.ndarray, weights: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted Kabsch: returns (R, t) so that dst ≈ R @ src + t."""
    if weights is None:
        weights = np.ones(len(src))
    w = weights / max(weights.sum(), 1e-12)
    mu_s = (src * w[:, None]).sum(axis=0)
    mu_d = (dst * w[:, None]).sum(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = (src_c * w[:, None]).T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[-1, -1] = -1
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def _compute_sim3(
    src: np.ndarray, dst: np.ndarray, min_motion: float = 0.02,
    mad_k: float = 2.5, refine_iters: int = 2,
) -> Optional[Tuple[float, np.ndarray, np.ndarray, dict]]:
    """Robust similarity (s, R, t) so that dst ≈ s*R@src + t.

    - Scale from MAD-filtered median of pairwise translation-distance ratios.
    - Rotation from weighted Kabsch, down-weighting high-residual frames.
    - Refines weights over `refine_iters` iterations.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < 3:
        return None

    pair_idx = []
    ds_pairs = []
    dd_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ds = np.linalg.norm(src[i] - src[j])
            dd = np.linalg.norm(dst[i] - dst[j])
            if ds > min_motion and dd > min_motion:
                pair_idx.append((i, j))
                ds_pairs.append(ds)
                dd_pairs.append(dd)
    if len(ds_pairs) < 3:
        return None
    ratios = np.asarray(dd_pairs) / np.asarray(ds_pairs)

    # Primary: RANSAC over pair-distance ratios
    scale_rs, inlier = _ransac_scale(ratios)
    # Fallback / refinement via MAD on RANSAC inliers
    sub = ratios[inlier] if inlier.sum() >= 3 else ratios
    med = np.median(sub)
    mad = np.median(np.abs(sub - med))
    if mad > 1e-12:
        mad_mask = np.abs(ratios - med) <= mad_k * mad
        combined = inlier & mad_mask
        if combined.sum() >= max(3, len(ratios) // 4):
            inlier = combined
    scale = float(np.median(ratios[inlier]))
    inlier_frac = float(inlier.mean())

    # Per-frame weights: count inlier pairs each frame participates in
    frame_inlier_count = np.zeros(n)
    frame_pair_count = np.zeros(n)
    for (i, j), is_in in zip(pair_idx, inlier):
        frame_pair_count[i] += 1
        frame_pair_count[j] += 1
        if is_in:
            frame_inlier_count[i] += 1
            frame_inlier_count[j] += 1
    weights = frame_inlier_count / np.maximum(frame_pair_count, 1.0)

    # Initial Kabsch
    src_scaled = src * scale
    R, t = _kabsch(src_scaled, dst, weights=weights)

    # Iterate: down-weight high-residual frames
    for _ in range(refine_iters):
        aligned = src_scaled @ R.T + t
        resid = np.linalg.norm(aligned - dst, axis=1)
        rmed = np.median(resid)
        rmad = np.median(np.abs(resid - rmed))
        if rmad > 1e-12:
            sigma = 1.4826 * rmad
            new_w = weights * np.exp(-0.5 * (resid / max(sigma, 1e-3)) ** 2)
        else:
            new_w = weights
        if new_w.sum() < 1e-9:
            break
        R, t = _kabsch(src_scaled, dst, weights=new_w)

    aligned = src_scaled @ R.T + t
    resid = np.linalg.norm(aligned - dst, axis=1)
    info = {
        "scale": scale,
        "n_pairs": len(ds_pairs),
        "inlier_frac": inlier_frac,
        "ratio_p05": float(np.percentile(ratios, 5)),
        "ratio_p95": float(np.percentile(ratios, 95)),
        "resid_mean": float(resid.mean()),
        "resid_median": float(np.median(resid)),
        "resid_max": float(resid.max()),
    }
    return scale, R.astype(np.float64), t.astype(np.float64), info


def _apply_sim3(poses: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply similarity (scale,R,t) to a batch of 4x4 c2w poses."""
    out = poses.astype(np.float64).copy()
    out[:, :3, :3] = R @ out[:, :3, :3]
    out[:, :3, 3] = scale * (out[:, :3, 3] @ R.T) + t
    return out.astype(np.float32)


def _detect_intra_chunk_outliers(
    poses_local: np.ndarray, k: float = 3.0, abs_cap: float = 0.8,
) -> np.ndarray:
    """Flag frames with broken local poses based on step distance to neighbors.

    A frame is 'bad' if its step to prev AND step to next both exceed
    (median + k * 1.4826 * MAD) or an absolute cap. Returns reliability mask.
    """
    centers = poses_local[:, :3, 3].astype(np.float64)
    n = len(centers)
    if n < 4:
        return np.ones(n, dtype=bool)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    med = float(np.median(steps))
    mad = float(np.median(np.abs(steps - med)))
    rel_limit = med + k * 1.4826 * mad if mad > 1e-9 else med * 3.0
    limit = max(rel_limit, abs_cap)
    reliable = np.ones(n, dtype=bool)
    for i in range(n):
        s_prev = steps[i - 1] if i > 0 else 0.0
        s_next = steps[i] if i < n - 1 else 0.0
        # Endpoint: one big step alone flags it
        if i == 0 and s_next > limit:
            reliable[i] = False
        elif i == n - 1 and s_prev > limit:
            reliable[i] = False
        elif i > 0 and i < n - 1 and s_prev > limit and s_next > limit:
            reliable[i] = False
    return reliable


def _refine_bundle(
    chunks: List[List[int]],
    chunk_centers_local: List[np.ndarray],
    sim3_list: List[Tuple[float, np.ndarray, np.ndarray]],
    reliable_masks: List[np.ndarray],
    iters: int = 5,
    tol: float = 0.005,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Pose-graph bundle refinement across chunks.

    For each non-anchor chunk ci, re-fit sim3 so its local centers align to the
    weighted consensus of globalized centers from all OTHER chunks covering the
    same global frames. Iterates until max relative scale delta < tol.
    """
    n_chunks = len(chunks)
    if n_chunks <= 2:
        return sim3_list
    sim3 = [(s, R.copy(), t.copy()) for (s, R, t) in sim3_list]

    for it in range(iters):
        max_ds = 0.0
        # Globalize all centers with current sim3
        globalized = []
        for ci, (s, R, t) in enumerate(sim3):
            g = (chunk_centers_local[ci] * s) @ R.T + t
            globalized.append(g)

        # Consensus per global frame: mean across chunks containing it
        # (only contributions from reliable frames count)
        gframe_sum: dict = {}
        gframe_cnt: dict = {}
        for ci, cidx in enumerate(chunks):
            for li, gi in enumerate(cidx):
                if not reliable_masks[ci][li]:
                    continue
                gframe_sum[gi] = gframe_sum.get(gi, np.zeros(3)) + globalized[ci][li]
                gframe_cnt[gi] = gframe_cnt.get(gi, 0) + 1

        for ci in range(1, n_chunks):
            cidx = chunks[ci]
            # Target = consensus excluding this chunk's own contribution
            src_list, dst_list = [], []
            for li, gi in enumerate(cidx):
                if not reliable_masks[ci][li]:
                    continue
                cnt = gframe_cnt.get(gi, 0)
                if cnt <= 1:
                    continue
                # Remove own contribution
                other_mean = (
                    (gframe_sum[gi] - globalized[ci][li]) / (cnt - 1)
                )
                src_list.append(chunk_centers_local[ci][li])
                dst_list.append(other_mean)
            if len(src_list) < 3:
                continue
            align = _compute_sim3(np.stack(src_list), np.stack(dst_list))
            if align is None:
                continue
            s_new, R_new, t_new, _ = align
            s_old = sim3[ci][0]
            max_ds = max(max_ds, abs(s_new - s_old) / max(s_old, 1e-9))
            sim3[ci] = (s_new, R_new, t_new)
        print(f"[MUSt3R] bundle iter {it+1}: max_rel_scale_delta={max_ds:.5f}")
        if max_ds < tol:
            break
    return sim3


def _build_chunks(n: int, chunk_size: int, overlap: int) -> List[List[int]]:
    if chunk_size <= 0 or n <= chunk_size:
        return [list(range(n))]
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(list(range(start, end)))
        if end == n:
            break
        start += step
    return chunks


def _run_must3r_chunk(
    model, chunk_paths, output_pointmaps_dir, resolution, min_conf_thr, device,
    execution_mode, num_mem_images, num_refinements_iterations,
    vidseq_local_context_size, keyframe_interval, slam_local_context_size,
    subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile,
    persistent_state=None,
) -> dict:
    """Run MUSt3R on a single chunk of frame paths.

    Returns dict with CPU numpy arrays: poses_c2w (Nc,4,4), depth_z (list of HxW),
    conf (list of HxW), and persistent_state (MUSt3R memory bank to chain into
    the next chunk). No global state is retained on GPU after return.
    """
    from must3r.demo.gradio import get_reconstructed_scene

    with torch.no_grad():
        scene, _, state_out = get_reconstructed_scene(
            outdir=output_pointmaps_dir or "/tmp/must3r_tmp",
            viser_server=None,
            should_save_glb=False,
            model=model,
            retrieval=None,
            device=device,
            verbose=True,
            image_size=resolution,
            amp="bf16",
            filelist=chunk_paths,
            min_conf_thr=min_conf_thr,
            as_pointcloud=True,
            transparent_cams=False,
            local_pointmaps=False,
            cam_size=0.05,
            num_mem_images=min(max(1, int(num_mem_images)), len(chunk_paths)),
            max_bs=1,
            render_once=False,
            camera_conf_thr=0.0,
            num_refinements_iterations=int(num_refinements_iterations),
            execution_mode=execution_mode,
            vidseq_local_context_size=int(vidseq_local_context_size),
            keyframe_interval=int(keyframe_interval),
            slam_local_context_size=int(slam_local_context_size),
            subsample=int(subsample),
            min_conf_keyframe=float(min_conf_keyframe),
            keyframe_overlap_thr=float(keyframe_overlap_thr),
            overlap_percentile=float(overlap_percentile),
            persistent_state=persistent_state,
            return_persistent_state=True,
        )

    poses_c2w = torch.stack([o["c2w"] for o in scene.x_out], dim=0)
    poses_c2w = poses_c2w.detach().cpu().numpy().astype(np.float32)

    depth_z = []
    xyz_maps = []
    confs = []
    for o in scene.x_out:
        pts_local = _to_numpy(o["pts3d_local"]).astype(np.float32)
        conf_map = _to_numpy(o["conf"]).astype(np.float32)
        depth_z.append(pts_local[..., 2])
        xyz_maps.append(pts_local)
        confs.append(conf_map)

    del scene
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "poses_c2w": poses_c2w,
        "depth_z": depth_z,
        "xyz": xyz_maps,
        "conf": confs,
        "persistent_state": state_out,
    }


def run_must3r_on_scene(
    frame_paths, checkpoint, output_depth_dir, output_poses_dir,
    output_pointmaps_dir, scene_id,
    resolution=512, min_conf_thr=1.5, device="cuda",
    execution_mode="linseq", num_mem_images=50, num_refinements_iterations=0,
    vidseq_local_context_size=0, keyframe_interval=3, slam_local_context_size=0,
    subsample=2, min_conf_keyframe=1.5, keyframe_overlap_thr=0.05,
    overlap_percentile=85, depth_mask_mode="hard", save_raw_depth=False,
    save_confidence_maps=False, pose_jump_max_translation=0.0,
    pose_jump_max_z_translation=0.0, pose_jump_max_rotation_deg=0.0,
    pose_jump_relative_factor=0.0, zero_invalid_pose_depths=False,
):
    """
    Runs MUSt3R in multi-view mode over the entire scene.
    Returns: poses (N,4,4), depths list of (H,W).
    """
    n_frames = len(frame_paths)
    chunk_size = int(os.environ.get("OBJECTX_MUST3R_CHUNK_SIZE", "0"))
    chunk_overlap = int(os.environ.get("OBJECTX_MUST3R_CHUNK_OVERLAP", "15"))
    chunks = _build_chunks(n_frames, chunk_size, chunk_overlap)

    print(
        f"\n[MUSt3R] Predicting poses+depth for {n_frames} frames "
        f"({len(chunks)} chunk{'s' if len(chunks)>1 else ''}, "
        f"chunk_size={chunk_size or n_frames}, overlap={chunk_overlap})"
    )
    must3r_path = os.environ.get("MUST3R_PATH", "must3r")
    if must3r_path not in sys.path:
        sys.path.insert(0, must3r_path)

    from must3r.model import load_model

    model = load_model(
        checkpoint, device=device, img_size=resolution,
        encoder=None, decoder=None, memory_mode=None,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    chunk_kwargs = dict(
        output_pointmaps_dir=output_pointmaps_dir, resolution=resolution,
        min_conf_thr=min_conf_thr, device=device, execution_mode=execution_mode,
        num_mem_images=num_mem_images,
        num_refinements_iterations=num_refinements_iterations,
        vidseq_local_context_size=vidseq_local_context_size,
        keyframe_interval=keyframe_interval,
        slam_local_context_size=slam_local_context_size, subsample=subsample,
        min_conf_keyframe=min_conf_keyframe,
        keyframe_overlap_thr=keyframe_overlap_thr,
        overlap_percentile=overlap_percentile,
    )

    # Persistent-state mode: MUSt3R memory bank is chained across chunks, so
    # poses are already in a single global frame. No sim3 alignment needed.
    global_poses: dict = {}
    global_depth_z: dict = {}
    global_xyz: dict = {}
    global_conf: dict = {}
    state = None
    for ci, cidx in enumerate(chunks):
        chunk_paths = [frame_paths[i] for i in cidx]
        print(
            f"[MUSt3R] chunk {ci+1}/{len(chunks)}: frames "
            f"{cidx[0]}..{cidx[-1]} ({len(cidx)} frames) "
            f"{'(persistent state)' if state is not None else '(fresh)'}"
        )
        res = _run_must3r_chunk(
            model, chunk_paths, persistent_state=state, **chunk_kwargs
        )
        state = res["persistent_state"]
        try:
            nmem = 0 if state is None else len(state.get("keyframes", []))
        except Exception:
            nmem = -1
        print(f"[MUSt3R] chunk {ci+1} done, persistent keyframes={nmem}")
        poses_local = res["poses_c2w"]
        depths_local = res["depth_z"]
        xyz_local = res["xyz"]
        confs_local = res["conf"]
        for li, gi in enumerate(cidx):
            if gi in global_poses:
                continue  # first writer wins
            global_poses[gi] = poses_local[li]
            global_depth_z[gi] = depths_local[li].astype(np.float32)
            global_xyz[gi] = xyz_local[li].astype(np.float32)
            global_conf[gi] = confs_local[li]

    # Assemble in global frame order
    poses_c2w_np = np.stack(
        [global_poses[i] for i in range(n_frames)], axis=0
    ).astype(np.float32)
    depth_z_list = [global_depth_z[i] for i in range(n_frames)]
    xyz_list = [global_xyz[i] for i in range(n_frames)]
    conf_list = [global_conf[i] for i in range(n_frames)]

    frame_ids = [_frame_id_from_path(p, i) for i, p in enumerate(frame_paths)]

    pose_filter = scan3r.detect_pose_jump_outliers(
        frame_ids,
        {frame_ids[i]: poses_c2w_np[i] for i in range(len(frame_ids))},
        pose_mode="raw",
        max_translation=float(pose_jump_max_translation),
        max_z_translation=float(pose_jump_max_z_translation),
        max_rotation_deg=float(pose_jump_max_rotation_deg),
        relative_step_factor=float(pose_jump_relative_factor),
    )
    invalid_frame_ids = set(pose_filter["dropped_frame_ids"])
    if invalid_frame_ids:
        print(
            "[MUSt3R] pose_jump_filter "
            f"kept={len(frame_ids) - len(invalid_frame_ids)}/{len(frame_ids)} "
            f"dropped={len(invalid_frame_ids)} "
            f"median_step={pose_filter['median_step']:.4f} "
            f"p95_step={pose_filter['raw_step_p95']:.4f} "
            f"limit={pose_filter['translation_limit']:.4f} "
            "mode=non_destructive"
        )
        preview = ", ".join(
            f"{item['frame_id']}<-{item['prev_frame_id']} step={item['step']:.2f} z={item['z_step']:.2f}"
            for item in pose_filter["dropped"][:10]
        )
        if preview:
            print(f"[MUSt3R] dropped pose jumps: {preview}")

    poses_c2w_t = torch.from_numpy(poses_c2w_np)

    depths = []
    masked_valid_fractions = []
    raw_valid_fractions = []
    for i, (depth_raw_full, xyz_full, conf_np) in enumerate(
        zip(depth_z_list, xyz_list, conf_list)
    ):
        raw_depth = depth_raw_full.astype(np.float32).copy()
        raw_depth[~np.isfinite(raw_depth)] = 0.0
        raw_depth[raw_depth <= 0.0] = 0.0
        mask_np = conf_np > min_conf_thr
        masked_depth = raw_depth.copy()
        if depth_mask_mode == "hard":
            masked_depth[~mask_np] = 0.0
        elif depth_mask_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported depth_mask_mode={depth_mask_mode!r}; expected 'hard' or 'none'"
            )

        legacy_frame_id = f"{i:04d}"
        if legacy_frame_id != frame_ids[i]:
            for stale_path in [
                Path(output_depth_dir) / f"frame-{legacy_frame_id}.depth.pgm",
                Path(output_depth_dir) / f"frame-{legacy_frame_id}.depth_raw.npy",
                Path(output_depth_dir) / f"frame-{legacy_frame_id}.conf.npy",
                Path(output_depth_dir) / f"frame-{legacy_frame_id}.xyz.npy",
                Path(output_poses_dir) / f"frame-{legacy_frame_id}.pose.txt",
            ]:
                if stale_path.exists():
                    stale_path.unlink()

        masked_depth = cv2.resize(masked_depth, (224, 172), interpolation=cv2.INTER_NEAREST)
        raw_depth_resized = cv2.resize(raw_depth, (224, 172), interpolation=cv2.INTER_NEAREST)
        conf_resized = cv2.resize(conf_np.astype(np.float32), (224, 172), interpolation=cv2.INTER_LINEAR)
        xyz_resized = cv2.resize(
            xyz_full.astype(np.float32), (224, 172), interpolation=cv2.INTER_NEAREST
        )
        xyz_resized[~np.isfinite(xyz_resized)] = 0.0
        if frame_ids[i] in invalid_frame_ids and zero_invalid_pose_depths:
            masked_depth.fill(0.0)
            raw_depth_resized.fill(0.0)
            conf_resized.fill(0.0)
            xyz_resized.fill(0.0)
        masked_valid_fractions.append(float((masked_depth > 0).mean()))
        raw_valid_fractions.append(float((raw_depth_resized > 0).mean()))
        depths.append(masked_depth)
        save_depth(masked_depth, output_depth_dir, frame_id=frame_ids[i])
        if save_raw_depth:
            save_depth_raw(raw_depth_resized, output_depth_dir, frame_id=frame_ids[i])
        if save_confidence_maps:
            save_confidence(conf_resized, output_depth_dir, frame_id=frame_ids[i])
        save_xyz_map(xyz_resized, output_depth_dir, frame_id=frame_ids[i])

    save_poses(poses_c2w_t, output_poses_dir, scene_id, frame_ids=frame_ids)
    print(
        f"[MUSt3R] Poses: {poses_c2w_t.shape} (camera_to_world, 3RScan convention) | "
        f"Depth shape: {depths[0].shape}"
    )
    if masked_valid_fractions:
        print(
            "[MUSt3R] depth coverage "
            f"masked_mean={np.mean(masked_valid_fractions):.4f} "
            f"masked_median={np.median(masked_valid_fractions):.4f} "
            f"raw_mean={np.mean(raw_valid_fractions):.4f} "
            f"raw_median={np.median(raw_valid_fractions):.4f}"
        )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return poses_c2w_t, depths


def depth_per_instance(depth_map, masks, p_low=5.0, p_high=95.0):
    """
    Robust per-instance depth, filtering border outliers
    via percentile clipping before computing statistics.
    """
    results = []
    for i, mask in enumerate(masks):
        vals = depth_map[mask]
        vals = vals[vals > 0]
        if len(vals) < 10:
            results.append({"obj_id": i, "median_depth": 0.0}); continue
        lo, hi = np.percentile(vals, [p_low, p_high])
        filt = vals[(vals >= lo) & (vals <= hi)]
        results.append({
            "obj_id": i, "median_depth": float(np.median(filt)),
            "mean_depth": float(np.mean(filt)), "std_depth": float(np.std(filt)),
            "n_pixels": int(mask.sum())
        })
    return results
