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
    """MASt3R returns per-pixel depth; fail fast if the layout is unexpected."""
    arr = np.asarray(flat_depth, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape == (H, W):
            return arr
        raise ValueError(
            f"unexpected MASt3R depth shape {arr.shape}; expected ({H}, {W})"
        )
    if arr.ndim != 1:
        raise ValueError(
            f"unexpected MASt3R depth rank {arr.ndim}; expected flat vector or 2D grid"
        )
    if int(arr.size) != H * W:
        raise ValueError(
            f"cannot reshape MASt3R depth of length {int(arr.size)} to ({H}, {W})"
        )
    return arr.reshape(H, W)


def _reshape_pointmap(flat_pts3d: np.ndarray, H: int, W: int) -> np.ndarray:
    arr = np.asarray(flat_pts3d, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape == (H, W, 3):
            return arr
        raise ValueError(
            f"unexpected MASt3R pointmap shape {arr.shape}; expected ({H}, {W}, 3)"
        )
    if arr.ndim != 2 or arr.shape[-1] != 3:
        raise ValueError(
            f"unexpected MASt3R pointmap shape {arr.shape}; expected (N, 3) or ({H}, {W}, 3)"
        )
    if int(arr.shape[0]) != H * W:
        raise ValueError(
            f"cannot reshape MASt3R pointmap of shape {arr.shape} to ({H}, {W}, 3)"
        )
    return arr.reshape(H, W, 3)


def _infer_dense_grid_shape(
    pointmap: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    H_net: int,
    W_net: int,
    subsample: int,
) -> tuple[int, int]:
    full_n = H_net * W_net
    sub_h = H_net // subsample
    sub_w = W_net // subsample
    sub_n = sub_h * sub_w

    point_n = int(np.asarray(pointmap).shape[0]) if np.asarray(pointmap).ndim >= 1 else int(np.asarray(pointmap).size)
    depth_n = int(np.asarray(depth).size)
    conf_n = int(np.asarray(conf).size)

    sizes = {point_n, depth_n, conf_n}
    if sizes == {full_n}:
        return H_net, W_net
    if sizes == {sub_n}:
        return sub_h, sub_w

    raise ValueError(
        "inconsistent MASt3R dense output sizes: "
        f"pointmap={np.asarray(pointmap).shape} depth={np.asarray(depth).shape} conf={np.asarray(conf).shape}; "
        f"expected either full-res {H_net}x{W_net} ({full_n}) or subsampled {sub_h}x{sub_w} ({sub_n})"
    )


def _build_depth_conf_mask(
    conf_map: np.ndarray,
    raw_depth: np.ndarray,
    min_conf_thr: float,
    smooth_sigma: float = 0.0,
    close_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    conf_eval = np.asarray(conf_map, dtype=np.float32)
    if smooth_sigma > 0.0:
        conf_eval = cv2.GaussianBlur(
            conf_eval,
            (0, 0),
            sigmaX=float(smooth_sigma),
            sigmaY=float(smooth_sigma),
        )

    conf_mask = conf_eval > float(min_conf_thr)
    if close_px > 0:
        kernel_size = 2 * int(close_px) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        conf_mask = cv2.morphologyEx(
            conf_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

    valid_depth = np.isfinite(raw_depth) & (raw_depth > 0.0)
    conf_mask &= valid_depth
    return conf_mask, conf_eval


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
    conf_smooth_sigma: float = 0.0,
    conf_close_px: int = 0,
    save_raw_depth: bool = False,
    save_confidence_maps: bool = False,
    pose_jump_max_translation: float = 0.0,
    pose_jump_max_z_translation: float = 0.0,
    pose_jump_max_rotation_deg: float = 0.0,
    pose_jump_relative_factor: float = 0.0,
    zero_invalid_pose_depths: bool = False,
    cache_dir: Optional[str] = None,
    loss3d_backward_chunk_pairs: int = 64,
    loss2d_backward_chunk_pairs: int = 64,
    loss_dust3r_backward_chunk_pairs: int = 32,
    streaming_condense: bool = True,
    fast_loss_path: bool = False,
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
    if (H_net % subsample) != 0 or (W_net % subsample) != 0:
        valid = [d for d in range(1, min(H_net, W_net) + 1) if H_net % d == 0 and W_net % d == 0]
        raise ValueError(
            f"invalid MASt3R subsample={subsample} for network size {H_net}x{W_net}; "
            f"valid divisors are {valid}"
        )

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
        loss3d_backward_chunk_pairs=loss3d_backward_chunk_pairs,
        loss2d_backward_chunk_pairs=loss2d_backward_chunk_pairs,
        loss_dust3r_backward_chunk_pairs=loss_dust3r_backward_chunk_pairs,
        streaming_condense=streaming_condense,
        fast_loss_path=fast_loss_path,
    )

    poses_c2w = scene.get_im_poses().detach().cpu().numpy().astype(np.float32)
    focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
    print(
        f"[MASt3R-SfM] BA done | poses={poses_c2w.shape} "
        f"focal_median={float(np.median(focals)):.2f}px"
    )

    print(
        f"[MASt3R-SfM] dense export start clean_depth=True subsample={subsample} "
        f"n_frames={n_frames}"
    )
    dense_pts3d, dense_depths, dense_confs = scene.get_dense_pts3d(
        clean_depth=True, subsample=subsample
    )
    print(
        f"[MASt3R-SfM] dense export loaded pointmaps={len(dense_pts3d)} "
        f"depths={len(dense_depths)} confs={len(dense_confs)}"
    )

    pointmap_list: List[np.ndarray] = []
    depth_z_list: List[np.ndarray] = []
    conf_list: List[np.ndarray] = []
    dense_grid_shape: Optional[tuple[int, int]] = None
    for i in range(n_frames):
        if i == 0 or (i + 1) % 100 == 0 or (i + 1) == n_frames:
            print(f"[MASt3R-SfM] dense reshape idx={i + 1}/{n_frames}")
        p = dense_pts3d[i]
        d = dense_depths[i]
        c = dense_confs[i]
        p_np = p.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(p) else np.asarray(p, dtype=np.float32)
        d_np = d.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(d) else np.asarray(d, dtype=np.float32)
        c_np = c.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(c) else np.asarray(c, dtype=np.float32)
        H_dense, W_dense = _infer_dense_grid_shape(
            p_np, d_np, c_np, H_net=H_net, W_net=W_net, subsample=subsample
        )
        if dense_grid_shape is None:
            dense_grid_shape = (H_dense, W_dense)
            print(
                f"[MASt3R-SfM] dense grid shape={H_dense}x{W_dense} "
                f"(network={H_net}x{W_net}, subsample={subsample})"
            )
        p_np = _reshape_pointmap(p_np, H_dense, W_dense)
        d_np = _reshape_depth(d_np, H_dense, W_dense)
        c_np = _reshape_depth(c_np, H_dense, W_dense)
        pointmap_list.append(p_np)
        depth_z_list.append(d_np)
        conf_list.append(c_np)

    print("[MASt3R-SfM] dense reshape done")

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

    if output_pointmaps_dir:
        os.makedirs(output_pointmaps_dir, exist_ok=True)

    print(
        f"[MASt3R-SfM] export start frames={n_frames} "
        f"depth_dir={output_depth_dir} poses_dir={output_poses_dir}"
    )
    depths: List[np.ndarray] = []
    base_mask_valid_fractions = []
    processed_mask_valid_fractions = []
    masked_pre_resize_valid_fractions = []
    masked_valid_fractions = []
    raw_source_valid_fractions = []
    raw_valid_fractions = []
    for i, (pointmap_full, depth_raw_full, conf_np) in enumerate(zip(pointmap_list, depth_z_list, conf_list)):
        if i == 0 or (i + 1) % 100 == 0 or (i + 1) == n_frames:
            print(f"[MASt3R-SfM] export loop idx={i + 1}/{n_frames}")
        raw_depth = depth_raw_full.astype(np.float32).copy()
        raw_depth[~np.isfinite(raw_depth)] = 0.0
        raw_depth[raw_depth <= 0.0] = 0.0
        raw_source_valid_fractions.append(float((raw_depth > 0).mean()))
        base_mask_valid_fractions.append(float((conf_np > min_conf_thr).mean()))
        mask_np, conf_eval = _build_depth_conf_mask(
            conf_np,
            raw_depth,
            min_conf_thr=min_conf_thr,
            smooth_sigma=conf_smooth_sigma,
            close_px=conf_close_px,
        )
        processed_mask_valid_fractions.append(float(mask_np.mean()))
        masked_depth = raw_depth.copy()
        if depth_mask_mode == "hard":
            masked_depth[~mask_np] = 0.0
        elif depth_mask_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported depth_mask_mode={depth_mask_mode!r}; expected 'hard' or 'none'"
            )
        masked_pre_resize_valid_fractions.append(float((masked_depth > 0).mean()))

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
        conf_resized = cv2.resize(conf_eval, (224, 172), interpolation=cv2.INTER_LINEAR)
        if frame_ids[i] in invalid_frame_ids and zero_invalid_pose_depths:
            masked_depth.fill(0.0)
        masked_valid_fractions.append(float((masked_depth > 0).mean()))
        raw_valid_fractions.append(float((raw_depth_resized > 0).mean()))
        depths.append(masked_depth)
        save_depth(masked_depth, output_depth_dir, frame_id=frame_ids[i])
        if save_raw_depth:
            save_depth_raw(raw_depth_resized, output_depth_dir, frame_id=frame_ids[i])
        if save_confidence_maps:
            save_confidence(conf_resized, output_depth_dir, frame_id=frame_ids[i])
        if output_pointmaps_dir:
            np.save(
                os.path.join(output_pointmaps_dir, f"frame-{frame_ids[i]}.pointmap.npy"),
                pointmap_full,
            )

    save_poses(poses_w2c, output_poses_dir, scene_id, frame_ids=frame_ids)
    print("[MASt3R-SfM] export done")
    print(f"[MASt3R-SfM] Poses: {poses_w2c.shape} | Depth shape: {depths[0].shape}")
    if masked_valid_fractions:
        print(
            "[MASt3R-SfM] confidence mask coverage "
            f"base_mean={np.mean(base_mask_valid_fractions):.4f} "
            f"processed_mean={np.mean(processed_mask_valid_fractions):.4f} "
            f"sigma={float(conf_smooth_sigma):.2f} "
            f"close_px={int(conf_close_px)}"
        )
        print(
            "[MASt3R-SfM] depth coverage "
            f"raw_source_mean={np.mean(raw_source_valid_fractions):.4f} "
            f"raw_source_median={np.median(raw_source_valid_fractions):.4f} "
            f"masked_preresize_mean={np.mean(masked_pre_resize_valid_fractions):.4f} "
            f"masked_export_mean={np.mean(masked_valid_fractions):.4f} "
            f"masked_export_median={np.median(masked_valid_fractions):.4f} "
            f"raw_export_mean={np.mean(raw_valid_fractions):.4f} "
            f"raw_export_median={np.median(raw_valid_fractions):.4f} "
            f"invalid_frames={len(invalid_frame_ids)} "
            f"zero_invalid_pose_depths={int(bool(zero_invalid_pose_depths))}"
        )
    del model, scene
    if device == "cuda":
        torch.cuda.empty_cache()
    return poses_w2c, depths
