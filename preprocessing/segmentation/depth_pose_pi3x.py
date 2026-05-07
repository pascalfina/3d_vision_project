"""Pi3X backend for pose+depth estimation.

Pi3X is a permutation-equivariant multi-view geometry model — newer
generation than DUSt3R/MUSt3R/MASt3R-SfM. Unlike the DUSt3R family it does
NOT need a fixed reference view, predicts cam-frame point maps + camera
poses + confidence in one forward pass over N images, and natively outputs
*metric-scaled* geometry.

We wrap the model's chunked inference helper ``Pi3XVO`` (which handles
chunking + sim3 alignment of overlapping chunks internally) so this backend
behaves like a single drop-in alternative to MUSt3R / MASt3R-SfM in our
pipeline. Outputs are saved in the same on-disk format as the other
backends so voxelise / render / pred-ready stages do not need to change.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from utils.io_utils import (
    save_confidence,
    save_depth,
    save_depth_raw,
    save_poses,
    save_xyz_map,
)
from utils import scan3r


_DEFAULT_PI3X_WEIGHTS = (
    "/work/courses/3dv/team35/pafina/models/pi3/Pi3X.safetensors"
)


def _frame_id_from_path(frame_path: str, fallback_idx: int) -> str:
    match = re.search(r"frame-(\d+)", Path(frame_path).stem)
    if match:
        return match.group(1)
    return f"{fallback_idx:06d}"


def _ensure_pi3_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pi3_root = os.environ.get(
        "PI3_PATH",
        str(repo_root / "dependencies" / "pi3"),
    )
    if pi3_root not in sys.path:
        sys.path.insert(0, pi3_root)
    # Pi3 imports ``from models.curope import cuRoPE2D`` for the CUDA-fast
    # rotary embedding kernel. We ship a copy of the compiled curope inside
    # ``dependencies/pi3/pi3/models/curope/`` so Pi3 has its own kernel
    # rather than borrowing MUSt3R/MAST3R's. To make the bare ``models.``
    # import work we add the inner ``pi3/`` directory to sys.path.
    #
    # The compiled .so is GPU-architecture-specific. If a user runs on a
    # different node than where it was built, set
    # ``OBJECTX_PI3X_DISABLE_CUROPE=1`` to fall back to the slow PyTorch
    # rotary embedding (or rebuild the kernel via
    # ``cd .../pi3/pi3/models/curope && python setup.py build_ext --inplace``).
    disable_curope = os.environ.get(
        "OBJECTX_PI3X_DISABLE_CUROPE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not disable_curope:
        pi3_inner = str(Path(pi3_root) / "pi3")
        if pi3_inner not in sys.path:
            sys.path.insert(0, pi3_inner)
        _patch_curope_for_contiguity()


def _patch_curope_for_contiguity() -> None:
    """Make the CUDA cuRoPE2D wrapper accept non-contiguous inputs.

    The shipped wrapper does ``tokens.transpose(1, 2)`` (non-contiguous view)
    and feeds that straight into the CUDA kernel, which assumes contiguous
    storage. MUSt3R's call sites happen to pass tensors where this still
    works, but Pi3's attention layer passes a query tensor that triggers
    ``RuntimeError: tokens are not contiguous``. We replace ``forward`` with
    a contiguous-safe version that runs the kernel on a contiguous copy and
    writes the result back into the original tensor (preserving the
    in-place semantics the caller expects).
    """
    try:
        from models.curope.curope2d import cuRoPE2D, cuRoPE2D_func
    except Exception:
        return
    if getattr(cuRoPE2D, "_objectx_contig_patched", False):
        return

    def _patched_forward(self, tokens, positions):
        transposed = tokens.transpose(1, 2).contiguous()
        cuRoPE2D_func.apply(transposed, positions, self.base, self.F0)
        tokens.copy_(transposed.transpose(1, 2))
        return tokens

    cuRoPE2D.forward = _patched_forward
    cuRoPE2D._objectx_contig_patched = True


def _resolve_weights() -> Optional[Path]:
    env = os.environ.get("OBJECTX_PI3X_WEIGHTS", "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None
    p = Path(_DEFAULT_PI3X_WEIGHTS)
    return p if p.exists() else None


def _load_images_from_paths(
    frame_paths: List[str], pixel_limit: int = 255000
) -> torch.Tensor:
    """Replicate Pi3's ``load_images_as_tensor`` but accept an explicit list
    of file paths instead of a directory or video. Returns (N, 3, H, W) in
    [0, 1] with H, W rounded to multiples of 14.
    """
    from PIL import Image
    from torchvision import transforms
    import math as _math

    if not frame_paths:
        return torch.empty(0)

    sources = []
    for p in frame_paths:
        try:
            sources.append(Image.open(p).convert("RGB"))
        except Exception as exc:
            raise FileNotFoundError(f"Pi3X failed to open {p}: {exc}")

    W_orig, H_orig = sources[0].size
    aspect = W_orig / max(H_orig, 1)

    # Pi3 requires dimensions divisible by 14. The upstream helper rounds W/H
    # independently, which can noticeably distort very small full-scene runs
    # (e.g. 960x540 -> 210x112). Prefer aspect-preserving candidates and only
    # trade a small amount of pixel area for a much cleaner geometry input.
    max_patches = max(1, int(pixel_limit) // (14 * 14))
    max_k = max(1, min(max_patches, int(_math.ceil(W_orig / 14))))
    max_m = max(1, min(max_patches, int(_math.ceil(H_orig / 14))))
    best = None
    for k in range(1, max_k + 1):
        for m in range(1, max_m + 1):
            patches = k * m
            if patches > max_patches:
                continue
            cand_w, cand_h = k * 14, m * 14
            cand_aspect = cand_w / max(cand_h, 1)
            aspect_err = abs(cand_aspect / aspect - 1.0)
            area = cand_w * cand_h
            # Strongly prefer aspect, but avoid tiny exact-aspect candidates.
            score = (aspect_err + 0.15 * (1.0 - area / max(pixel_limit, 1)), -area)
            if best is None or score < best[0]:
                best = (score, cand_w, cand_h, aspect_err)
    if best is None:
        TARGET_W, TARGET_H, aspect_err = 14, 14, 0.0
    else:
        _, TARGET_W, TARGET_H, aspect_err = best
    print(
        f"[Pi3X] uniform image size set to {TARGET_W}x{TARGET_H} "
        f"(input was {W_orig}x{H_orig}, pixel_limit={pixel_limit}, "
        f"aspect_err={aspect_err:.4f})"
    )
    to_tensor = transforms.ToTensor()
    tensor_list = []
    for img_pil in sources:
        resized = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        tensor_list.append(to_tensor(resized))
    return torch.stack(tensor_list, dim=0)


def _select_even_anchor_indices(n_frames: int, anchor_count: int) -> Optional[List[int]]:
    if n_frames <= 0 or anchor_count <= 1:
        return None
    anchor_count = min(int(anchor_count), n_frames)
    anchors = np.linspace(0, n_frames - 1, anchor_count)
    indices = sorted({int(round(v)) for v in anchors})
    # Rounding can collapse neighboring anchors for very short sequences.
    if len(indices) < anchor_count:
        for idx in range(n_frames):
            if len(indices) >= anchor_count:
                break
            indices.append(idx)
        indices = sorted(set(indices))
    return indices


def _load_scaled_color_intrinsics(
    frame_paths: List[str], target_h: int, target_w: int
) -> Optional[np.ndarray]:
    if not frame_paths:
        return None
    info_path = Path(frame_paths[0]).parent / "_info.txt"
    if not info_path.exists():
        return None

    color_w = color_h = None
    color_k = None
    for line in info_path.read_text().splitlines():
        if "m_colorWidth" in line:
            color_w = float(line.split("=", 1)[1].strip())
        elif "m_colorHeight" in line:
            color_h = float(line.split("=", 1)[1].strip())
        elif "m_calibrationColorIntrinsic" in line:
            values = [float(v) for v in line.split("=", 1)[1].split()]
            if len(values) >= 11:
                color_k = np.array(
                    [
                        [values[0], 0.0, values[2]],
                        [0.0, values[5], values[6]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
    if color_w is None or color_h is None or color_k is None:
        return None

    scaled = color_k.copy()
    scaled[0, 0] *= float(target_w) / color_w
    scaled[0, 2] *= float(target_w) / color_w
    scaled[1, 1] *= float(target_h) / color_h
    scaled[1, 2] *= float(target_h) / color_h
    return scaled.astype(np.float32)


def _intrinsics_batch(
    frame_paths: List[str],
    target_h: int,
    target_w: int,
    device: str,
) -> Optional[torch.Tensor]:
    k = _load_scaled_color_intrinsics(frame_paths, target_h, target_w)
    if k is None:
        return None
    intrinsics = np.broadcast_to(k[None], (len(frame_paths), 3, 3)).copy()
    return torch.from_numpy(intrinsics).unsqueeze(0).to(device)


def _edge_filter_conf(local_depth: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    from pi3.utils.geometry import depth_edge

    edge = depth_edge(local_depth, rtol=0.03)
    conf = conf.clone()
    conf[edge] = 0
    return conf


def _run_local_geometry_chunks(
    model,
    frame_paths: List[str],
    *,
    pixel_limit: int,
    chunk_size: int,
    overlap: int,
    conf_thr: float,
    device: str,
    dtype: torch.dtype,
    pose_priors_c2w: np.ndarray,
    use_intrinsics: bool,
    global_depth_maps: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    # global_depth_maps: (N, H_global, W_global) camera-frame Z from the fullscene pass.
    # Passing these anchors Pi3X's internal scale normalization (dep_median) to the same
    # scene-level scale in every chunk. Without them, each chunk independently normalises
    # its pose translations by its own RMS inter-frame distance, producing a different
    # dep_median per chunk and therefore a different absolute depth scale — the root cause
    # of the scale_ratio > 1 rejection.
    imgs = _load_images_from_paths(frame_paths, pixel_limit=int(pixel_limit))
    if imgs.ndim != 4 or imgs.shape[0] != len(frame_paths):
        raise RuntimeError(
            f"Pi3X local pass produced unexpected image tensor {tuple(imgs.shape)}"
        )
    imgs = imgs.to(device)
    n_frames, _, h_net, w_net = imgs.shape
    intrinsics = (
        _intrinsics_batch(frame_paths, h_net, w_net, device)
        if use_intrinsics
        else None
    )
    poses = torch.from_numpy(pose_priors_c2w.astype(np.float32)).unsqueeze(0).to(device)

    # Pre-resize global depth maps to local resolution once, reuse per chunk.
    global_depths_local: Optional[np.ndarray] = None
    if global_depth_maps is not None and global_depth_maps.shape[0] == n_frames:
        resized = []
        for i in range(n_frames):
            d = cv2.resize(
                global_depth_maps[i].astype(np.float32),
                (w_net, h_net),
                interpolation=cv2.INTER_LINEAR,
            )
            resized.append(d)
        global_depths_local = np.stack(resized, axis=0)  # (N, h_net, w_net)
        print(f"[Pi3X] local pass: using global depth priors for scale anchoring "
              f"(resized to {w_net}x{h_net})")

    step = max(1, int(chunk_size) - max(0, int(overlap)))
    local_points_out: list[Optional[torch.Tensor]] = [None] * n_frames
    conf_out: list[Optional[torch.Tensor]] = [None] * n_frames
    print(
        f"[Pi3X] high-res local geometry pass "
        f"(pixel_limit={pixel_limit}, chunk_size={chunk_size}, overlap={overlap}, "
        f"use_intrinsics={use_intrinsics})"
    )
    with torch.no_grad():
        for start_idx in range(0, n_frames, step):
            end_idx = min(start_idx + int(chunk_size), n_frames)
            if end_idx <= start_idx:
                break
            chunk_imgs = imgs[None, start_idx:end_idx]
            model_kwargs = {"with_prior": True}
            if intrinsics is not None:
                model_kwargs["intrinsics"] = intrinsics[:, start_idx:end_idx]
                model_kwargs["mask_add_ray"] = torch.ones(
                    (1, end_idx - start_idx), dtype=torch.bool, device=device
                )
                model_kwargs["poses"] = poses[:, start_idx:end_idx]
                model_kwargs["mask_add_pose"] = torch.ones(
                    (1, end_idx - start_idx), dtype=torch.bool, device=device
                )
            if global_depths_local is not None:
                chunk_depths = torch.from_numpy(
                    global_depths_local[start_idx:end_idx]
                ).unsqueeze(0).to(device)
                model_kwargs["depths"] = chunk_depths
                model_kwargs["mask_add_depth"] = torch.ones(
                    (1, end_idx - start_idx), dtype=torch.bool, device=device
                )

            print(
                f"  > Local geometry chunk: [{start_idx} : {end_idx}] "
                f"(Length: {end_idx - start_idx})"
            )
            with torch.amp.autocast("cuda", dtype=dtype):
                pred = model(chunk_imgs, **model_kwargs)

            local_points = pred["local_points"][0].detach()
            conf = torch.sigmoid(pred["conf"][0, ..., 0]).detach()
            conf = _edge_filter_conf(local_points[..., 2], conf)

            keep_from = start_idx if start_idx == 0 else min(end_idx, start_idx + int(overlap))
            for frame_idx in range(keep_from, end_idx):
                local_idx = frame_idx - start_idx
                if local_points_out[frame_idx] is None:
                    local_points_out[frame_idx] = local_points[local_idx].cpu()
                    conf_out[frame_idx] = conf[local_idx].cpu()

            del pred, local_points, conf, chunk_imgs
            if device == "cuda":
                torch.cuda.empty_cache()
            if end_idx == n_frames:
                break

    missing = [i for i, value in enumerate(local_points_out) if value is None]
    if missing:
        raise RuntimeError(f"Pi3X local geometry pass missed frames: {missing[:20]}")

    local_np = torch.stack(local_points_out, dim=0).to(torch.float32).numpy()
    conf_np = torch.stack(conf_out, dim=0).to(torch.float32).numpy()
    print(
        f"[Pi3X] high-res local geometry output: "
        f"local_points={local_np.shape} conf={conf_np.shape}"
    )
    return local_np, conf_np


def _global_pts_to_cam_frame(
    global_pts: np.ndarray, pose_c2w: np.ndarray
) -> np.ndarray:
    """Transform (H, W, 3) world-frame points into the camera frame.

    Uses the full matrix inverse rather than R.T because Pi3XVO's sim3
    alignment between chunks may bake a scale factor into the rotation
    block (so R is not necessarily orthonormal). ``np.linalg.inv`` handles
    both cases correctly and matches the model's own
    ``points = R @ local + t`` relationship for any R.
    """
    R = pose_c2w[:3, :3].astype(np.float64)
    t = pose_c2w[:3, 3].astype(np.float64)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if R is singular (degenerate pose).
        R_inv = np.linalg.pinv(R)
    pts = global_pts.astype(np.float64)
    flat = pts.reshape(-1, 3)
    cam_flat = (flat - t[None, :]) @ R_inv.T
    cam = cam_flat.reshape(*pts.shape)
    return cam.astype(np.float32)


def _smooth_depth_scale(
    ref_z: np.ndarray,
    pred_z: np.ndarray,
    sigma: float = 20.0,
) -> np.ndarray:
    """Per-pixel scale map (ref_z / pred_z) blurred over ~sigma pixels.

    Maps all surface points along their viewing rays to the reference depth
    (MUSt3R) while preserving Pi3X's local angular geometry structure within
    each smoothing neighbourhood.  Invalid pixels (zero in either input) are
    initialised to 1.0 so the Gaussian fill-in from valid neighbours takes over.
    Result is clipped to [0.25, 4.0].
    """
    scale_map = np.ones(pred_z.shape, dtype=np.float32)
    valid = (pred_z > 0) & (ref_z > 0)
    if valid.any():
        scale_map[valid] = np.clip(
            ref_z[valid] / (pred_z[valid] + 1e-8), 0.25, 4.0
        )
    if sigma > 0:
        scale_map = cv2.GaussianBlur(scale_map, (0, 0), sigmaX=float(sigma))
    return scale_map


def _load_external_poses(frame_paths: List[str], pose_dir: str) -> np.ndarray:
    """Load c2w pose matrices from .pose.txt files for the given frame paths."""
    poses = []
    for i, p in enumerate(frame_paths):
        fid = _frame_id_from_path(p, i)
        pose_file = Path(pose_dir) / f"frame-{fid}.pose.txt"
        if not pose_file.exists():
            raise FileNotFoundError(f"External pose file not found: {pose_file}")
        mat = np.loadtxt(str(pose_file), dtype=np.float64).reshape(4, 4)
        poses.append(mat.astype(np.float32))
    return np.stack(poses, axis=0)


def _load_external_xyz_maps(
    frame_paths: List[str], xyz_dir: str
) -> Optional[np.ndarray]:
    """Load camera-frame xyz maps from .xyz.npy files for the given frame paths.

    Returns None if any file is missing.
    """
    maps = []
    for i, p in enumerate(frame_paths):
        fid = _frame_id_from_path(p, i)
        xyz_file = Path(xyz_dir) / f"frame-{fid}.xyz.npy"
        if not xyz_file.exists():
            print(
                f"[Pi3X] external xyz file not found: {xyz_file}, "
                "skipping external depth priors"
            )
            return None
        maps.append(np.load(str(xyz_file)).astype(np.float32))
    return np.stack(maps, axis=0)


def run_pi3x_on_scene(
    frame_paths: List[str],
    checkpoint: Optional[str],
    output_depth_dir: str,
    output_poses_dir: str,
    output_pointmaps_dir: Optional[str],
    scene_id: str,
    *,
    pixel_limit: int = 255000,
    chunk_size: int = 16,
    overlap: int = 6,
    conf_thr: float = 0.05,
    device: str = "cuda",
    save_raw_depth: bool = False,
    save_confidence_maps: bool = False,
    pose_jump_max_translation: float = 0.0,
    pose_jump_max_z_translation: float = 0.0,
    pose_jump_max_rotation_deg: float = 0.0,
    pose_jump_relative_factor: float = 0.0,
    zero_invalid_pose_depths: bool = False,
    align_mode: str = "se3",
    anchor_count: int = 0,
    use_intrinsics: bool = False,
    local_pixel_limit: int = 0,
    local_chunk_size: int = 25,
    local_overlap: int = 8,
    local_scale_max_ratio: float = 1.35,
    output_native_resolution: bool = False,
    frame_stride: int = 1,
    external_pose_dir: Optional[str] = None,
):
    """Runs Pi3X over the entire scene with chunked sim3-aligned inference.

    ``frame_stride``: when > 1, Pi3X only processes every ``frame_stride``-th
    frame (always including the first and last). Skipped frames receive no
    pose/xyz/depth/conf output, so downstream voxelise / render only see the
    selected subset. This trades raw point density for higher per-frame
    spatial resolution at fixed GPU memory budget.

    Returns:
        (poses_c2w_t (N',4,4) torch tensor for selected frames,
         depths list of (172, 224) np arrays for selected frames).
    """
    _ensure_pi3_on_path()

    # Optional frame subsampling to trade temporal density for spatial
    # resolution under a fixed GPU memory budget. Always include the last
    # frame so the trajectory endpoint is covered. Stale outputs from prior
    # full-stride runs are removed so downstream voxelise sees only the new
    # selected subset.
    stride = max(1, int(frame_stride))
    if stride > 1 and len(frame_paths) > 1:
        original_total = len(frame_paths)
        keep_indices = list(range(0, original_total, stride))
        if keep_indices[-1] != original_total - 1:
            keep_indices.append(original_total - 1)
        kept_set = set(keep_indices)
        kept_frame_ids = {
            _frame_id_from_path(p, i) for i, p in enumerate(frame_paths)
            if i in kept_set
        }
        sampled_paths = [frame_paths[i] for i in keep_indices]
        print(
            f"[Pi3X] frame_stride={stride}: subsampling {len(sampled_paths)} of "
            f"{original_total} frames for inference (dropping "
            f"{original_total - len(sampled_paths)} frames). Skipped frames "
            "will have no Pi3X pose/xyz/depth/conf output."
        )
        # Remove stale per-frame outputs for frames that won't be processed
        # this run. Color/_info.txt are preserved (they live in the same dir
        # but are extracted from the baseline zip, not written by Pi3X).
        skipped_frame_ids = [
            _frame_id_from_path(p, i) for i, p in enumerate(frame_paths)
            if i not in kept_set
        ]
        stale_suffixes = (
            ".pose.txt",
            ".depth.pgm",
            ".depth_raw.npy",
            ".conf.npy",
            ".xyz.npy",
        )
        n_removed = 0
        for stale_dir in {output_depth_dir, output_poses_dir}:
            if not stale_dir:
                continue
            for fid in skipped_frame_ids:
                for suffix in stale_suffixes:
                    p = Path(stale_dir) / f"frame-{fid}{suffix}"
                    if p.exists():
                        try:
                            p.unlink()
                            n_removed += 1
                        except Exception:
                            pass
        if n_removed:
            print(
                f"[Pi3X] removed {n_removed} stale per-frame output files for "
                "subsampled-out frames"
            )
        frame_paths = sampled_paths

    weights_path = checkpoint
    if not weights_path:
        resolved = _resolve_weights()
        if resolved is None:
            raise FileNotFoundError(
                "Pi3X weights not found. Set OBJECTX_PI3X_WEIGHTS or place "
                f"the file at {_DEFAULT_PI3X_WEIGHTS}."
            )
        weights_path = str(resolved)

    n_frames = len(frame_paths)
    print(
        f"\n[Pi3X] Predicting poses+depth for {n_frames} frames "
        f"(chunk_size={chunk_size}, overlap={overlap}, "
        f"conf_thr={conf_thr}, pixel_limit={pixel_limit}, "
        f"align_mode={align_mode}, anchor_count={anchor_count}, "
        f"use_intrinsics={use_intrinsics}, local_pixel_limit={local_pixel_limit})"
    )
    print(f"[Pi3X] loading model from {weights_path}")

    try:
        import argparse as _argparse
        torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception:
        pass

    from pi3.models.pi3x import Pi3X
    from pi3.pipe.pi3x_vo import Pi3XVO

    model = Pi3X().to(device).eval()
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(weights_path)
    else:
        _orig_load = torch.load
        def _trusted_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _trusted_load
        try:
            weight = torch.load(weights_path, map_location=device)
        finally:
            torch.load = _orig_load
    model.load_state_dict(weight, strict=False)
    if device == "cuda":
        torch.cuda.empty_cache()

    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )

    if external_pose_dir:
        print(
            f"[Pi3X] external_pose_dir set → skipping Pi3XVO global pass; "
            f"loading poses from {external_pose_dir}"
        )
        if int(local_pixel_limit) <= 0:
            raise ValueError(
                "external_pose_dir requires local_pixel_limit > 0 "
                "(no global-pass depth fallback available)"
            )
        poses_c2w_np = _load_external_poses(frame_paths, external_pose_dir)
        if poses_c2w_np.shape[0] != n_frames:
            raise RuntimeError(
                f"Loaded {poses_c2w_np.shape[0]} external poses but expected {n_frames}"
            )
        print(f"[Pi3X] loaded {n_frames} external c2w poses")
        global_local_pts_all: Optional[np.ndarray] = _load_external_xyz_maps(
            frame_paths, external_pose_dir
        )
        if global_local_pts_all is not None:
            print(
                f"[Pi3X] loaded external xyz maps {global_local_pts_all.shape} "
                "for local-pass depth anchoring and scale validation"
            )
        global_conf_all: Optional[np.ndarray] = None
        local_geometry_source = "ext_poses"
        out_h, out_w = (172, 224)
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        pipe = Pi3XVO(model)

        imgs = _load_images_from_paths(frame_paths, pixel_limit=int(pixel_limit))
        if imgs.ndim != 4 or imgs.shape[0] != n_frames:
            raise RuntimeError(
                f"Pi3X image preprocessing produced unexpected tensor shape "
                f"{tuple(imgs.shape)}"
            )
        imgs = imgs.to(device)
        _, _, H_net, W_net = imgs.shape
        print(f"[Pi3X] loaded {n_frames} images at {H_net}x{W_net}")
        global_intrinsics = (
            _intrinsics_batch(frame_paths, H_net, W_net, device)
            if use_intrinsics
            else None
        )
        if global_intrinsics is not None:
            k0 = global_intrinsics[0, 0].detach().cpu().numpy()
            print(
                "[Pi3X] using scaled color intrinsics "
                f"fx={k0[0,0]:.3f} fy={k0[1,1]:.3f} "
                f"cx={k0[0,2]:.3f} cy={k0[1,2]:.3f}"
            )

        imgs_batched = imgs.unsqueeze(0)  # (1, T, 3, H, W)

        print(
            f"[Pi3X] running chunked Pi3XVO (dtype={dtype}, "
            f"chunk_size={chunk_size}, overlap={overlap})"
        )
        anchor_indices = _select_even_anchor_indices(n_frames, int(anchor_count))
        if anchor_indices is not None:
            print(f"[Pi3X] anchor-aligned Pi3XVO anchors={anchor_indices}")
        with torch.no_grad():
            out = pipe(
                imgs=imgs_batched,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                conf_thre=float(conf_thr),
                inject_condition=None,
                dtype=dtype,
                align_mode=str(align_mode).strip().lower(),
                anchor_indices=anchor_indices,
                intrinsics=global_intrinsics,
            )
        # out['points']        (1, N, H, W, 3)  global world frame, metric scale
        # out['local_points']  (1, N, H, W, 3)  camera-frame XYZ, metric scale
        # out['camera_poses']  (1, N, 4, 4)     rigid c2w, metric translation
        # out['conf']          (1, N, H, W)     sigmoid'd confidence in [0, 1]
        global_pts_all = out["points"][0].detach().to("cpu", dtype=torch.float32).numpy()
        local_pts_all = (
            out.get("local_points", out["points"])[0]
            .detach()
            .to("cpu", dtype=torch.float32)
            .numpy()
        )
        poses_c2w_np = out["camera_poses"][0].detach().to(
            "cpu", dtype=torch.float32
        ).numpy()
        conf_all = out["conf"][0].detach().to("cpu", dtype=torch.float32).numpy()
        del pipe, out, imgs, imgs_batched
        if device == "cuda":
            torch.cuda.empty_cache()

        if poses_c2w_np.shape[0] != n_frames:
            raise RuntimeError(
                f"Pi3X returned {poses_c2w_np.shape[0]} poses but expected {n_frames}"
            )
        if global_pts_all.shape != (n_frames, H_net, W_net, 3):
            raise RuntimeError(
                f"Pi3X global pts shape {global_pts_all.shape} does not match "
                f"({n_frames}, {H_net}, {W_net}, 3)"
            )
        if local_pts_all.shape != (n_frames, H_net, W_net, 3):
            raise RuntimeError(
                f"Pi3X local pts shape {local_pts_all.shape} does not match "
                f"({n_frames}, {H_net}, {W_net}, 3)"
            )

        global_local_pts_all = local_pts_all
        global_conf_all = conf_all
        local_geometry_source = "fullscene"
        out_h, out_w = (H_net, W_net) if output_native_resolution else (172, 224)
        if output_native_resolution:
            print(
                "[Pi3X] writing native-resolution xyz/depth/conf maps "
                f"({out_w}x{out_h})"
            )

    precomputed_local_scale_factors: Optional[List[float]] = None
    if int(local_pixel_limit) > 0:
        local_pts_all, conf_all = _run_local_geometry_chunks(
            model,
            frame_paths,
            pixel_limit=int(local_pixel_limit),
            chunk_size=int(local_chunk_size),
            overlap=int(local_overlap),
            conf_thr=float(conf_thr),
            device=device,
            dtype=dtype,
            pose_priors_c2w=poses_c2w_np,
            use_intrinsics=bool(use_intrinsics),
            global_depth_maps=(
                global_local_pts_all[..., 2] if global_local_pts_all is not None else None
            ),
        )
        local_geometry_source = (
            f"highres_chunks(pixel_limit={int(local_pixel_limit)},"
            f"chunk_size={int(local_chunk_size)},overlap={int(local_overlap)})"
        )
        candidate_scales: List[float] = []
        if global_local_pts_all is not None:
            for i in range(n_frames):
                candidate_xyz = cv2.resize(
                    local_pts_all[i].astype(np.float32),
                    (out_w, out_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                candidate_conf = cv2.resize(
                    conf_all[i].astype(np.float32),
                    (out_w, out_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                global_xyz = cv2.resize(
                    global_local_pts_all[i].astype(np.float32),
                    (out_w, out_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                if global_conf_all is not None:
                    global_conf = cv2.resize(
                        global_conf_all[i].astype(np.float32),
                        (out_w, out_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    valid_scale = (
                        (candidate_xyz[..., 2] > 0.0)
                        & (global_xyz[..., 2] > 0.0)
                        & (candidate_conf >= float(conf_thr))
                        & (global_conf >= float(conf_thr))
                    )
                else:
                    valid_scale = (
                        (candidate_xyz[..., 2] > 0.0)
                        & (global_xyz[..., 2] > 0.0)
                        & (candidate_conf >= float(conf_thr))
                    )
                if int(valid_scale.sum()) >= 256:
                    ratio = global_xyz[..., 2][valid_scale] / (
                        candidate_xyz[..., 2][valid_scale] + 1e-8
                    )
                    finite_ratio = ratio[np.isfinite(ratio)]
                    if finite_ratio.size:
                        scale = float(np.median(finite_ratio))
                        candidate_scales.append(float(np.clip(scale, 0.25, 4.0)))
                        continue
                candidate_scales.append(1.0)

            scale_arr = np.asarray(candidate_scales, dtype=np.float32)
            valid_scales = scale_arr[np.isfinite(scale_arr) & (scale_arr > 0.0)]
            if valid_scales.size:
                scale_ratio = float(valid_scales.max() / max(valid_scales.min(), 1e-8))
            else:
                scale_ratio = float("inf")
            if scale_ratio > float(local_scale_max_ratio):
                print(
                    "[Pi3X] high-res local geometry rejected: "
                    f"scale_ratio={scale_ratio:.3f} > "
                    f"max_ratio={float(local_scale_max_ratio):.3f}; "
                    + (
                        "falling back to external xyz maps"
                        if external_pose_dir
                        else "falling back to fullscene geometry"
                    )
                )
                local_pts_all = global_local_pts_all
                conf_all = (
                    np.ones(
                        (n_frames,) + global_local_pts_all.shape[1:3],
                        dtype=np.float32,
                    )
                    if external_pose_dir
                    else global_conf_all
                )
                local_geometry_source = (
                    "ext_xyz_fallback" if external_pose_dir else "fullscene"
                )
            else:
                precomputed_local_scale_factors = candidate_scales
                if output_native_resolution and local_pts_all[0].shape[:2] != (out_h, out_w):
                    out_h, out_w = local_pts_all[0].shape[:2]
                    print(
                        f"[Pi3X] output resolution updated to local pass resolution "
                        f"({out_w}x{out_h})"
                    )
        else:
            # No global depth reference → accept local pass unconditionally with scale=1.0
            print(
                "[Pi3X] no external xyz maps found; accepting local pass without scale validation"
            )
            precomputed_local_scale_factors = [1.0] * n_frames
            if output_native_resolution and local_pts_all[0].shape[:2] != (out_h, out_w):
                out_h, out_w = local_pts_all[0].shape[:2]
                print(
                    f"[Pi3X] output resolution updated to local pass resolution "
                    f"({out_w}x{out_h})"
                )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    pose_scales = np.linalg.norm(poses_c2w_np[:, :3, :3], axis=1)
    pose_scale_median = np.median(pose_scales, axis=1)
    print(
        "[Pi3X] pose-scale summary "
        f"median={float(np.median(pose_scale_median)):.4f} "
        f"min={float(np.min(pose_scale_median)):.4f} "
        f"max={float(np.max(pose_scale_median)):.4f}"
    )

    print(f"[Pi3X] local geometry source: {local_geometry_source}")

    frame_ids = [_frame_id_from_path(p, i) for i, p in enumerate(frame_paths)]
    pose_filter = scan3r.detect_pose_jump_outliers(
        frame_ids,
        {frame_ids[i]: poses_c2w_np[i] for i in range(n_frames)},
        pose_mode="raw",
        max_translation=float(pose_jump_max_translation),
        max_z_translation=float(pose_jump_max_z_translation),
        max_rotation_deg=float(pose_jump_max_rotation_deg),
        relative_step_factor=float(pose_jump_relative_factor),
    )
    invalid_frame_ids = set(pose_filter["dropped_frame_ids"])
    if invalid_frame_ids:
        print(
            "[Pi3X] pose_jump_filter "
            f"kept={n_frames - len(invalid_frame_ids)}/{n_frames} "
            f"dropped={len(invalid_frame_ids)} "
            f"median_step={pose_filter['median_step']:.4f} "
            f"p95_step={pose_filter['raw_step_p95']:.4f} "
            f"limit={pose_filter['translation_limit']:.4f} "
            "mode=non_destructive"
        )
        preview = ", ".join(
            f"{item['frame_id']}<-{item['prev_frame_id']} "
            f"step={item['step']:.2f} z={item['z_step']:.2f}"
            for item in pose_filter["dropped"][:10]
        )
        if preview:
            print(f"[Pi3X] dropped pose jumps: {preview}")

    poses_c2w_t = torch.from_numpy(poses_c2w_np)

    depths_resized: List[np.ndarray] = []
    masked_valid_fractions: List[float] = []
    raw_valid_fractions: List[float] = []
    local_scale_factors: List[float] = []
    for i in range(n_frames):
        xyz_full = local_pts_all[i].astype(np.float32, copy=True)
        xyz_full[~np.isfinite(xyz_full)] = 0.0
        xyz_full[..., 2] = np.where(xyz_full[..., 2] > 0.0, xyz_full[..., 2], 0.0)

        conf_np = conf_all[i].astype(np.float32)
        conf_resized = cv2.resize(
            conf_np, (out_w, out_h), interpolation=cv2.INTER_LINEAR
        )
        xyz_resized = cv2.resize(
            xyz_full, (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )
        xyz_resized[~np.isfinite(xyz_resized)] = 0.0
        xyz_resized[..., 2] = np.where(
            xyz_resized[..., 2] > 0.0, xyz_resized[..., 2], 0.0
        )

        if local_geometry_source not in ("fullscene", "ext_xyz_fallback"):
            if global_local_pts_all is not None:
                # Pixel-wise scale: align each Pi3X surface point to the MUSt3R
                # depth along the same viewing ray, smoothed over ~20 px to
                # preserve Pi3X's local angular geometry while removing
                # inter-chunk scale discontinuities.
                ref_z = cv2.resize(
                    global_local_pts_all[i][..., 2].astype(np.float32),
                    (out_w, out_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                scale_map = _smooth_depth_scale(ref_z, xyz_resized[..., 2])
                xyz_resized *= scale_map[..., None]
                local_scale_factors.append(float(np.median(scale_map)))
            elif precomputed_local_scale_factors is not None:
                scale = float(precomputed_local_scale_factors[i])
                xyz_resized *= scale
                local_scale_factors.append(scale)

        raw_resized = xyz_resized[..., 2].astype(np.float32)
        masked_resized = np.where(
            conf_resized > float(conf_thr), raw_resized, 0.0
        ).astype(np.float32)

        if frame_ids[i] in invalid_frame_ids and zero_invalid_pose_depths:
            masked_resized.fill(0.0)
            raw_resized.fill(0.0)
            conf_resized.fill(0.0)
            xyz_resized.fill(0.0)

        masked_valid_fractions.append(float((masked_resized > 0).mean()))
        raw_valid_fractions.append(float((raw_resized > 0).mean()))
        depths_resized.append(masked_resized)

        save_depth(masked_resized, output_depth_dir, frame_id=frame_ids[i])
        if save_raw_depth:
            save_depth_raw(raw_resized, output_depth_dir, frame_id=frame_ids[i])
        if save_confidence_maps:
            save_confidence(conf_resized, output_depth_dir, frame_id=frame_ids[i])
        save_xyz_map(xyz_resized, output_depth_dir, frame_id=frame_ids[i])

    save_poses(poses_c2w_t, output_poses_dir, scene_id, frame_ids=frame_ids)
    print(
        f"[Pi3X] Poses: {poses_c2w_t.shape} (camera_to_world, 3RScan convention) | "
        f"Depth shape: {depths_resized[0].shape}"
    )
    if masked_valid_fractions:
        masked_arr = np.asarray(masked_valid_fractions)
        raw_arr = np.asarray(raw_valid_fractions)
        print(
            f"[Pi3X] depth coverage masked_mean={masked_arr.mean():.4f} "
            f"masked_median={np.median(masked_arr):.4f} "
            f"raw_mean={raw_arr.mean():.4f} raw_median={np.median(raw_arr):.4f}"
        )
    if local_scale_factors:
        scale_arr = np.asarray(local_scale_factors, dtype=np.float32)
        print(
            "[Pi3X] high-res local scale-to-fullscene "
            f"median={np.median(scale_arr):.4f} "
            f"min={scale_arr.min():.4f} max={scale_arr.max():.4f}"
        )
    return poses_c2w_t, depths_resized
