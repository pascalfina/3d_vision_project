"""
MASt3R-SfM backend: predicts poses (N,4,4) and depth maps (H,W) from RGB only
via global sparse bundle adjustment.

Unlike MUSt3R this is single-pass: MASt3R-SfM handles long sequences internally
through a sparsified scene graph (sliding-window or logarithmic) plus global
bundle adjustment, so no chunking/persistent-state logic is needed. All frames
are jointly optimised, which addresses the intra-sequence pose drift that the
MUSt3R pipeline leaves behind.
"""
import os, re, sys, tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import cv2

from utils.io_utils import save_confidence, save_depth, save_depth_raw, save_poses
from utils import scan3r


def _frame_id_from_path(frame_path: str, fallback_idx: int) -> str:
    match = re.search(r"frame-(\d+)", Path(frame_path).stem)
    if match:
        return match.group(1)
    return f"{fallback_idx:06d}"


def _ensure_mast3r_on_path() -> None:
    mast3r_root = os.environ.get(
        "MAST3R_PATH",
        str(Path(__file__).resolve().parents[2] / "dependencies" / "mast3r"),
    )
    dust3r_root = os.path.join(mast3r_root, "dust3r")
    for p in (dust3r_root, mast3r_root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _reshape_depth(flat_depth: np.ndarray, H: int, W: int) -> np.ndarray:
    """MASt3R returns flat per-pixel depth — reshape to (H, W)."""
    n = int(np.prod(flat_depth.shape))
    if n == H * W:
        return flat_depth.reshape(H, W)
    # Occasionally the canonical depth grid is smaller than the input image
    # (subsampled); fall back to a square-ish reshape and hope the caller
    # resizes downstream.
    side = int(round(np.sqrt(n)))
    if side * side == n:
        return flat_depth.reshape(side, side)
    raise ValueError(
        f"cannot reshape MASt3R depth of length {n} to ({H},{W}) or square"
    )


def run_mast3r_sfm_on_scene(
    frame_paths: List[str],
    checkpoint: str,
    output_depth_dir: str,
    output_poses_dir: str,
    output_pointmaps_dir: Optional[str],
    scene_id: str,
    resolution: int = 512,
    min_conf_thr: float = 1.0,
    device: str = "cuda",
    scene_graph: str = "swin-15",
    shared_intrinsics: bool = True,
    lr1: float = 0.07,
    niter1: int = 300,
    lr2: float = 0.01,
    niter2: int = 300,
    opt_depth: bool = True,
    matching_conf_thr: float = 0.0,
    loss_dust3r_w: float = 0.01,
    subsample: int = 8,
    depth_mask_mode: str = "hard",
    save_raw_depth: bool = False,
    save_confidence_maps: bool = False,
    pose_jump_max_translation: float = 0.0,
    pose_jump_max_z_translation: float = 0.0,
    pose_jump_max_rotation_deg: float = 0.0,
    pose_jump_relative_factor: float = 0.0,
    zero_invalid_pose_depths: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Runs MASt3R-SfM (forward MASt3R + global sparse BA) over the entire scene.
    Returns: (poses_w2c (N,4,4) torch tensor, depths list of (172,224) np.ndarray).
    """
    _ensure_mast3r_on_path()
    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    import mast3r.utils.path_to_dust3r  # noqa
    from dust3r.utils.image import load_images

    n_frames = len(frame_paths)
    print(
        f"\n[MASt3R-SfM] Predicting poses+depth for {n_frames} frames "
        f"(scene_graph={scene_graph}, shared_intrinsics={shared_intrinsics}, "
        f"niter1={niter1}, niter2={niter2}, subsample={subsample})"
    )

    print(f"[MASt3R-SfM] loading model from {checkpoint}")
    model = AsymmetricMASt3R.from_pretrained(checkpoint).to(device)
    if device == "cuda":
        torch.cuda.empty_cache()

    imgs = load_images(frame_paths, size=resolution, verbose=False)
    _, _, H_net, W_net = imgs[0]["img"].shape
    print(f"[MASt3R-SfM] loaded {len(imgs)} images, network size={H_net}x{W_net}")

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"[MASt3R-SfM] pairs: {len(pairs)}")

    # Keep only lightweight metadata in RAM and reload image tensors lazily per
    # pair inside sparse_ga. This reduces host memory pressure substantially and
    # is important when the MASt3R cache lives on tmpfs-backed /tmp.
    for img, frame_path in zip(imgs, frame_paths):
        img["source_path"] = frame_path
        img["load_size"] = int(resolution)
        img["img"] = None
    del imgs
    import gc
    gc.collect()

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix=f"mast3r_sfm_{scene_id}_")
    os.makedirs(cache_dir, exist_ok=True)
    stale_tmp_files = 0
    for stale_path in Path(cache_dir).rglob("*.tmp.*"):
        try:
            stale_path.unlink()
            stale_tmp_files += 1
        except OSError:
            pass
    print(f"[MASt3R-SfM] cache_dir={cache_dir}")
    if stale_tmp_files:
        print(f"[MASt3R-SfM] cleaned {stale_tmp_files} stale temp files from cache")

    scene = sparse_global_alignment(
        frame_paths, pairs, cache_dir, model,
        lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2,
        device=device, opt_depth=opt_depth,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
        loss_dust3r_w=loss_dust3r_w,
        subsample=subsample,
    )

    poses_c2w = scene.get_im_poses().detach().cpu().numpy().astype(np.float32)
    focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
    print(
        f"[MASt3R-SfM] BA done | poses={poses_c2w.shape} "
        f"focal_median={float(np.median(focals)):.2f}px"
    )

    _, dense_depths, dense_confs = scene.get_dense_pts3d(
        clean_depth=True, subsample=subsample
    )

    # dense_depths and dense_confs are lists of flat torch tensors.
    # Per-pixel grid: (H_net // subsample) * (W_net // subsample).
    H_sub = H_net // subsample
    W_sub = W_net // subsample
    depth_z_list: List[np.ndarray] = []
    conf_list: List[np.ndarray] = []
    for i in range(n_frames):
        d = dense_depths[i]
        c = dense_confs[i]
        d_np = d.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(d) else np.asarray(d, dtype=np.float32)
        c_np = c.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(c) else np.asarray(c, dtype=np.float32)
        d_np = _reshape_depth(d_np, H_sub, W_sub)
        c_np = _reshape_depth(c_np, H_sub, W_sub)
        depth_z_list.append(d_np)
        conf_list.append(c_np)

    frame_ids = [_frame_id_from_path(p, i) for i, p in enumerate(frame_paths)]

    pose_filter = scan3r.detect_pose_jump_outliers(
        frame_ids,
        {frame_ids[i]: poses_c2w[i] for i in range(len(frame_ids))},
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
                poses_c2w[idx] = poses_c2w[last_valid_idx]
            else:
                last_valid_idx = idx
        print(
            "[MASt3R-SfM] pose_jump_filter "
            f"kept={len(frame_ids) - len(invalid_frame_ids)}/{len(frame_ids)} "
            f"dropped={len(invalid_frame_ids)} "
            f"median_step={pose_filter['median_step']:.4f} "
            f"p95_step={pose_filter['raw_step_p95']:.4f} "
            f"limit={pose_filter['translation_limit']:.4f}"
        )
        preview = ", ".join(
            f"{item['frame_id']}<-{item['prev_frame_id']} "
            f"step={item['step']:.2f} z={item['z_step']:.2f}"
            for item in pose_filter["dropped"][:10]
        )
        if preview:
            print(f"[MASt3R-SfM] dropped pose jumps: {preview}")

    poses_c2w_t = torch.from_numpy(poses_c2w)
    poses_w2c = torch.linalg.inv(poses_c2w_t)

    depths: List[np.ndarray] = []
    masked_valid_fractions = []
    raw_valid_fractions = []
    for i, (depth_raw_full, conf_np) in enumerate(zip(depth_z_list, conf_list)):
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
                Path(output_poses_dir) / f"frame-{legacy_frame_id}.pose.txt",
            ]:
                if stale_path.exists():
                    stale_path.unlink()

        masked_depth = cv2.resize(masked_depth, (224, 172), interpolation=cv2.INTER_NEAREST)
        raw_depth_resized = cv2.resize(raw_depth, (224, 172), interpolation=cv2.INTER_NEAREST)
        conf_resized = cv2.resize(conf_np, (224, 172), interpolation=cv2.INTER_LINEAR)
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
    print(f"[MASt3R-SfM] Poses: {poses_w2c.shape} | Depth shape: {depths[0].shape}")
    if masked_valid_fractions:
        print(
            "[MASt3R-SfM] depth coverage "
            f"masked_mean={np.mean(masked_valid_fractions):.4f} "
            f"masked_median={np.median(masked_valid_fractions):.4f} "
            f"raw_mean={np.mean(raw_valid_fractions):.4f} "
            f"raw_median={np.median(raw_valid_fractions):.4f}"
        )
    del model, scene
    if device == "cuda":
        torch.cuda.empty_cache()
    return poses_w2c, depths
