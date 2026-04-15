"""
MUSt3R: predicts poses (N,4,4) and depth maps (H,W) from RGB only.
Depth is extracted from the Z channel of the pointmaps (pts3d[...,2]).
"""
import os, re, sys, numpy as np, torch, cv2
from pathlib import Path
from typing import List, Optional, Tuple
from utils.io_utils import save_confidence, save_depth, save_depth_raw, save_poses
from utils import scan3r


def _frame_id_from_path(frame_path: str, fallback_idx: int) -> str:
    match = re.search(r"frame-(\d+)", Path(frame_path).stem)
    if match:
        return match.group(1)
    return f"{int(fallback_idx):06d}"


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
    print(f"\n[MUSt3R] Predicting poses+depth for {len(frame_paths)} frames")
    must3r_path = os.environ.get("MUST3R_PATH", "must3r")
    if must3r_path not in sys.path:
        sys.path.insert(0, must3r_path)

    from must3r.model import load_model
    from must3r.demo.gradio import get_reconstructed_scene
    from dust3r.utils.image import load_images
    
    # Load checkpoint and create model with args
    model = load_model(
        checkpoint,
        device=device,
        img_size=resolution,
        encoder=None,
        decoder=None,
        memory_mode=None,
    )
    torch.cuda.empty_cache()

    with torch.no_grad():
        scene, _ = get_reconstructed_scene(
            outdir=output_pointmaps_dir or "/tmp/must3r_tmp",
            viser_server=None,
            should_save_glb=False,
            model=model,
            retrieval=None,
            device=device,
            verbose=True,
            image_size=resolution,
            amp="bf16",
            filelist=frame_paths,
            min_conf_thr=min_conf_thr,
            as_pointcloud=True,
            transparent_cams=False,
            local_pointmaps=False,
            cam_size=0.05,
            num_mem_images=min(max(1, int(num_mem_images)), len(frame_paths)),
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
        ) # dict_keys(['x_out', 'imgs', 'true_shape', 'focals', 'cams2world', 'image_list'])
    
    frame_ids = [_frame_id_from_path(path, i) for i, path in enumerate(frame_paths)]

    # Extract poses, pts3d, conf_masks from the scene object
    pts3d = [o["pts3d"] for o in scene.x_out] # 3D pointmap of each frame in global coords
    conf_masks = [o["conf"] > min_conf_thr for o in scene.x_out]
    poses_c2w = torch.stack([o["c2w"] for o in scene.x_out], dim=0) # Shape: (N, 4, 4)
    poses_c2w_np = poses_c2w.detach().cpu().numpy().astype(np.float32)
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
        last_valid_idx = 0
        for idx, frame_id in enumerate(frame_ids):
            if frame_id in invalid_frame_ids:
                poses_c2w_np[idx] = poses_c2w_np[last_valid_idx]
            else:
                last_valid_idx = idx
        print(
            "[MUSt3R] pose_jump_filter "
            f"kept={len(frame_ids) - len(invalid_frame_ids)}/{len(frame_ids)} "
            f"dropped={len(invalid_frame_ids)} "
            f"median_step={pose_filter['median_step']:.4f} "
            f"p95_step={pose_filter['raw_step_p95']:.4f} "
            f"limit={pose_filter['translation_limit']:.4f}"
        )
        preview = ", ".join(
            f"{item['frame_id']}<-{item['prev_frame_id']} step={item['step']:.2f} z={item['z_step']:.2f}"
            for item in pose_filter["dropped"][:10]
        )
        if preview:
            print(f"[MUSt3R] dropped pose jumps: {preview}")
    poses_c2w = torch.from_numpy(poses_c2w_np).to(device=poses_c2w.device, dtype=poses_c2w.dtype)
    poses_w2c = torch.linalg.inv(poses_c2w)
    
    # o["pts3d"]       : 3D pointmap of the frame in global coordinates
    # o["pts3d_local"] : pointmap in local camera coordinates
    # o["conf"]        : per-pixel/point confidence
    # o["focal"]       : estimated focal length
    # o["c2w"]         : camera-to-world pose

    # Depth → Z channel of each local pointmap
    depths = []
    pts3d_local = [o["pts3d_local"] for o in scene.x_out]

    masked_valid_fractions = []
    raw_valid_fractions = []
    for i, (pts_local, mask, conf_map) in enumerate(zip(pts3d_local, conf_masks, [o["conf"] for o in scene.x_out])):
        pts_local_np = pts_local.cpu().numpy() if hasattr(pts_local, "cpu") else pts_local
        mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else mask
        conf_np = conf_map.cpu().numpy() if hasattr(conf_map, "cpu") else conf_map
        legacy_frame_id = f"{i:04d}"
        if legacy_frame_id != frame_ids[i]:
            for stale_path in [
                Path(output_depth_dir) / f"frame-{legacy_frame_id}.depth.pgm",
                Path(output_depth_dir) / f"frame-{legacy_frame_id}.depth_raw.npy",
                Path(output_depth_dir) / f"frame-{legacy_frame_id}.conf.npy",
                Path(output_poses_dir) / f"frame-{legacy_frame_id}.pose.txt",
            ]:
                if stale_path.exists():
                    stale_path.unlink()

        raw_depth = pts_local_np[..., 2].astype(np.float32)
        raw_depth[~np.isfinite(raw_depth)] = 0.0
        raw_depth[raw_depth <= 0.0] = 0.0
        masked_depth = raw_depth.copy()
        if depth_mask_mode == "hard":
            masked_depth[~mask_np] = 0.0
        elif depth_mask_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported depth_mask_mode={depth_mask_mode!r}; expected 'hard' or 'none'"
            )

        masked_depth = cv2.resize(masked_depth, (224, 172), interpolation=cv2.INTER_NEAREST)
        raw_depth_resized = cv2.resize(raw_depth, (224, 172), interpolation=cv2.INTER_NEAREST)
        conf_resized = cv2.resize(conf_np.astype(np.float32), (224, 172), interpolation=cv2.INTER_LINEAR)
        if frame_ids[i] in invalid_frame_ids and zero_invalid_pose_depths:
            masked_depth.fill(0.0)
            raw_depth_resized.fill(0.0)
            conf_resized.fill(0.0)
        masked_valid_fractions.append(float((masked_depth > 0).mean()))
        raw_valid_fractions.append(float((raw_depth_resized > 0).mean()))
        depths.append(masked_depth)
        save_depth(masked_depth, output_depth_dir, frame_id=frame_ids[i])
        if save_raw_depth:
            save_depth_raw(raw_depth_resized, output_depth_dir, frame_id=frame_ids[i])
        if save_confidence_maps:
            save_confidence(conf_resized, output_depth_dir, frame_id=frame_ids[i])

    save_poses(poses_w2c, output_poses_dir, scene_id, frame_ids=frame_ids)
    print(f"[MUSt3R] Poses: {poses_w2c.shape} | Depth shape: {depths[0].shape}")
    if masked_valid_fractions:
        print(
            "[MUSt3R] depth coverage "
            f"masked_mean={np.mean(masked_valid_fractions):.4f} "
            f"masked_median={np.median(masked_valid_fractions):.4f} "
            f"raw_mean={np.mean(raw_valid_fractions):.4f} "
            f"raw_median={np.median(raw_valid_fractions):.4f}"
        )
    del model; torch.cuda.empty_cache()
    return poses_w2c, depths


def depth_per_instance(depth_map, masks, p_low=5.0, p_high=95.0):
    """
    Robust per-instance depth, filtering border outliers
    via percentile clipping before computing statistics.
    """
    results = []
    for i, mask in enumerate(masks):
        vals = depth_map[mask]
        vals = vals[vals > 0] # discard zero/invalid depth
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
