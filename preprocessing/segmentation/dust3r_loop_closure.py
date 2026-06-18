"""DUSt3R-based global pose refinement (a.k.a. loop closure) for MUSt3R.

This module is *off by default*. Enable via:
    OBJECTX_MUST3R_USE_DUST3R_LOOP_CLOSURE=1

What it does:
    After MUSt3R has produced per-frame poses + xyz + conf, this hook runs
    DUSt3R inference on a global pair graph (default: ``swin-12``) and feeds
    the pairwise outputs into DUSt3R's ``cloud_opt`` global aligner. The
    resulting globally-consistent c2w poses *replace* MUSt3R poses, while
    MUSt3R's per-frame xyz/depth are kept (they are sharper than the
    optimizer's depth) — but rescaled so they match the new world scale via
    the existing robust ``_compute_sim3`` over camera centers.

Why bother:
    MUSt3R's vidslam memory works locally but accumulates drift across long
    sequences. DUSt3R's PointCloudOptimizer does Bundle Adjustment over a
    global pair graph and fixes that drift cleanly.

Outputs:
    ``(poses_c2w_np_refined, depth_z_list_rescaled, xyz_list_rescaled,
       conf_list)``. Conf is unchanged.

Soft-fail: any error returns the inputs unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_DEFAULT_DUST3R_WEIGHTS = "models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"


def _resolve_weights(repo_root: Path) -> Optional[Path]:
    env = os.environ.get("OBJECTX_DUST3R_WEIGHTS", "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None
    p = repo_root / _DEFAULT_DUST3R_WEIGHTS
    return p if p.exists() else None


def _ensure_dust3r_importable(repo_root: Path) -> None:
    candidates = [
        repo_root / "dependencies" / "must3r" / "dust3r",
        repo_root / "dependencies" / "mast3r" / "dust3r",
    ]
    for c in candidates:
        if (c / "dust3r" / "__init__.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return


def _to_numpy(x) -> np.ndarray:
    import torch as _t
    if isinstance(x, _t.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _camera_centers(poses_c2w: np.ndarray) -> np.ndarray:
    return poses_c2w[:, :3, 3].astype(np.float64)


def _apply_sim3_to_poses(
    poses_c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Apply sim3 (s, R, t) to camera-to-world transforms.

    World-to-world' map is x' = s R x + t, so a c2w pose [R_c|t_c] maps to
    [R R_c | s R t_c + t].
    """
    out = poses_c2w.copy().astype(np.float64)
    R = R.astype(np.float64)
    t = t.astype(np.float64)
    for i in range(out.shape[0]):
        R_c = out[i, :3, :3]
        t_c = out[i, :3, 3]
        out[i, :3, :3] = R @ R_c
        out[i, :3, 3] = float(scale) * (R @ t_c) + t
    return out.astype(np.float32)


def apply_dust3r_loop_closure(
    frame_paths: List[str],
    poses_c2w_np: np.ndarray,
    depth_z_list: List[np.ndarray],
    xyz_list: List[np.ndarray],
    conf_list: List[np.ndarray],
    *,
    device: str = "cuda",
    scene_graph: str = "swin-12",
    niter: int = 300,
    schedule: str = "cosine",
    lr: float = 0.01,
    init: str = "mst",
    image_size: int = 512,
    batch_size: int = 1,
    align_to_must3r_world: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Run DUSt3R cloud_opt to globally refine MUSt3R poses.

    Returns ``(poses_c2w_np_refined, depth_z_list_rescaled, xyz_list_rescaled,
    conf_list)``. On any failure all inputs are returned unchanged.

    If ``align_to_must3r_world`` is True (default), refined poses are
    sim3-aligned back into MUSt3R's world frame so downstream code keeps a
    consistent coordinate system. Otherwise we keep DUSt3R's optimizer world
    and only rescale MUSt3R cam-frame quantities.
    """
    repo_root = Path(__file__).resolve().parents[2]
    _ensure_dust3r_importable(repo_root)

    weights_path = _resolve_weights(repo_root)
    if weights_path is None:
        print(
            "[DUSt3R-loop] weights not found; expected at "
            f"{repo_root / _DEFAULT_DUST3R_WEIGHTS} or via OBJECTX_DUST3R_WEIGHTS. "
            "Skipping loop closure."
        )
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    try:
        import argparse as _argparse
        import torch
        try:
            torch.serialization.add_safe_globals([_argparse.Namespace])
        except Exception:
            pass
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    except Exception as exc:
        print(f"[DUSt3R-loop] failed to import DUSt3R: {exc}. Skipping.")
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    # Reuse the robust sim3 helper from depth_pose_must3r so we are consistent
    # with the existing MUSt3R/MAST3R-SfM fusion code.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from preprocessing.segmentation.depth_pose_must3r import _compute_sim3
    except Exception as exc:
        print(f"[DUSt3R-loop] failed to import _compute_sim3: {exc}. Skipping.")
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    n = len(frame_paths)
    if n < 3:
        print(f"[DUSt3R-loop] need >=3 frames, got {n}. Skipping.")
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    print(
        f"[DUSt3R-loop] enabled: weights={weights_path.name} scene_graph={scene_graph} "
        f"niter={niter} lr={lr} init={init} image_size={image_size} "
        f"align_to_must3r_world={align_to_must3r_world}"
    )

    _orig_torch_load = torch.load
    def _trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _trusted_load
    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(str(weights_path)).to(device)
        model.eval()
    except Exception as exc:
        print(f"[DUSt3R-loop] failed to load DUSt3R model: {exc}. Skipping.")
        return poses_c2w_np, depth_z_list, xyz_list, conf_list
    finally:
        torch.load = _orig_torch_load

    try:
        imgs = load_images(
            frame_paths,
            size=image_size,
            verbose=False,
            patch_size=getattr(model, "patch_size", 16),
            square_ok=False,
        )
        pairs = make_pairs(
            imgs,
            scene_graph=scene_graph,
            prefilter=None,
            symmetrize=True,
        )
    except Exception as exc:
        print(f"[DUSt3R-loop] failed to build pair graph: {exc}. Skipping.")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    out = None
    try:
        out = inference(pairs, model, device, batch_size=batch_size, verbose=verbose)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and batch_size > 1:
            torch.cuda.empty_cache()
            print(f"[DUSt3R-loop] OOM at bs={batch_size}, retrying bs=1")
            try:
                out = inference(pairs, model, device, batch_size=1, verbose=verbose)
            except Exception as exc2:
                print(f"[DUSt3R-loop] inference failed even at bs=1: {exc2}. Skipping.")
        else:
            print(f"[DUSt3R-loop] inference failed: {exc}. Skipping.")
    except Exception as exc:
        print(f"[DUSt3R-loop] inference failed: {exc}. Skipping.")
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if out is None:
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    try:
        scene = global_aligner(
            out,
            device=device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=verbose,
        )
        scene.compute_global_alignment(
            init=init, niter=int(niter), schedule=schedule, lr=float(lr)
        )
    except Exception as exc:
        print(f"[DUSt3R-loop] cloud_opt failed: {exc}. Skipping.")
        if device == "cuda":
            import torch as _torch
            _torch.cuda.empty_cache()
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    try:
        dust_poses_c2w = _to_numpy(scene.get_im_poses()).astype(np.float64)  # (N,4,4)
    except Exception as exc:
        print(f"[DUSt3R-loop] failed to read poses from cloud_opt: {exc}. Skipping.")
        return poses_c2w_np, depth_z_list, xyz_list, conf_list
    finally:
        try:
            del scene
        except NameError:
            pass
        if device == "cuda":
            import torch as _torch
            _torch.cuda.empty_cache()

    if dust_poses_c2w.shape[0] != n:
        print(
            f"[DUSt3R-loop] pose count mismatch (got {dust_poses_c2w.shape[0]}, "
            f"expected {n}). Skipping."
        )
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    must_centers = _camera_centers(poses_c2w_np)
    dust_centers = _camera_centers(dust_poses_c2w)
    finite_mask = np.all(np.isfinite(must_centers), axis=1) & np.all(
        np.isfinite(dust_centers), axis=1
    )
    if int(finite_mask.sum()) < 5:
        print("[DUSt3R-loop] not enough finite camera centers for sim3. Skipping.")
        return poses_c2w_np, depth_z_list, xyz_list, conf_list

    if align_to_must3r_world:
        sim3 = _compute_sim3(
            src=dust_centers[finite_mask],
            dst=must_centers[finite_mask],
        )
        if sim3 is None:
            print("[DUSt3R-loop] sim3 alignment failed. Skipping.")
            return poses_c2w_np, depth_z_list, xyz_list, conf_list
        scale, R, t, info = sim3
        refined_poses = _apply_sim3_to_poses(dust_poses_c2w, scale, R, t)
        depth_scaled = [d.astype(np.float32) * float(scale) for d in depth_z_list]
        xyz_scaled = [x.astype(np.float32) * float(scale) for x in xyz_list]
        print(
            f"[DUSt3R-loop] sim3 DUSt3R->MUSt3R scale={scale:.4f} "
            f"n_pairs={info['n_pairs']} inlier_frac={info['inlier_frac']:.3f} "
            f"resid median={info['resid_median']:.3f} "
            "(MUSt3R poses replaced by DUSt3R cloud_opt; cam-frame depth/xyz rescaled)"
        )
        return (
            refined_poses.astype(np.float32),
            depth_scaled,
            xyz_scaled,
            conf_list,
        )
    else:
        sim3 = _compute_sim3(
            src=must_centers[finite_mask],
            dst=dust_centers[finite_mask],
        )
        if sim3 is None:
            print("[DUSt3R-loop] sim3 alignment failed. Skipping.")
            return poses_c2w_np, depth_z_list, xyz_list, conf_list
        scale, _R, _t, info = sim3
        depth_scaled = [d.astype(np.float32) * float(scale) for d in depth_z_list]
        xyz_scaled = [x.astype(np.float32) * float(scale) for x in xyz_list]
        print(
            f"[DUSt3R-loop] world=DUSt3R; MUSt3R cam-frame depth/xyz rescaled "
            f"by sim3 scale={scale:.4f} n_pairs={info['n_pairs']} "
            f"inlier_frac={info['inlier_frac']:.3f}"
        )
        return (
            dust_poses_c2w.astype(np.float32),
            depth_scaled,
            xyz_scaled,
            conf_list,
        )
