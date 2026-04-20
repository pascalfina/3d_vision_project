"""
Hybrid fuser: MASt3R-SfM poses + MUSt3R depth/xyz/confidence.

Rationale: MASt3R-SfM's feature-matching + global BA gives globally consistent,
drift-controlled camera poses. MUSt3R's vidslam gives temporally smooth, dense
per-pixel depth + cam-frame XYZ. We keep MASt3R-SfM poses as the world frame
and bring MUSt3R depth/xyz into that scale by a robust sim3 alignment of the
two camera-center sequences.

Only the scale factor from sim3 is applied to MUSt3R cam-frame quantities
(depth, xyz). Rotation / translation from sim3 describe how the two world
frames relate; they do NOT transform cam-frame points (which are local to the
camera). World-frame unprojection uses the MASt3R-SfM pose directly.

Output sequence directory matches the layout voxelise_features / renderers
expect:
  frame-<id>.color.jpg      copied from MASt3R-SfM (identical source frames)
  frame-<id>.pose.txt       MASt3R-SfM w2c pose (unmodified)
  frame-<id>.depth_raw.npy  MUSt3R raw depth * scale
  frame-<id>.depth.pgm      MUSt3R masked depth * scale (conf > thr)
  frame-<id>.conf.npy       MUSt3R confidence (unchanged; used by voxelise)
  frame-<id>.xyz.npy        MUSt3R cam-frame XYZ * scale
  _info.txt                 copied from MASt3R-SfM sequence

Only frame IDs present in BOTH backends are written. Frames missing in one
side or with degenerate data are skipped so the scene dir never mixes
pinhole-pose frames with Tango-fallback placeholders.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.segmentation.depth_pose_must3r import _compute_sim3  # noqa: E402
from utils.io_utils import (  # noqa: E402
    save_confidence,
    save_depth,
    save_depth_raw,
    save_xyz_map,
)


_POSE_ID_RE = re.compile(r"frame-(\d+)\.pose\.txt$")


def _enumerate_frame_ids(sequence_dir: Path) -> List[str]:
    ids = []
    for p in sorted(sequence_dir.glob("frame-*.pose.txt")):
        m = _POSE_ID_RE.search(p.name)
        if m:
            ids.append(m.group(1))
    return ids


def _load_pose(path: Path) -> np.ndarray:
    return np.loadtxt(path).astype(np.float64)


def _invert_se3(T: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T)


def _camera_center(c2w: np.ndarray) -> np.ndarray:
    return c2w[:3, 3]


def _is_identity_like(pose: np.ndarray, tol: float = 1e-5) -> bool:
    return bool(np.allclose(pose, np.eye(4), atol=tol))


def _raw_depth_has_content(path: Path) -> bool:
    if not path.exists():
        return False
    d = np.load(path)
    return bool(np.any(d > 0))


def _xyz_has_content(path: Path) -> bool:
    if not path.exists():
        return False
    d = np.load(path)
    return bool(np.any(d[..., 2] > 0))


def _copy_color(src_dir: Path, dst_dir: Path, fid: str) -> None:
    src = src_dir / f"frame-{fid}.color.jpg"
    dst = dst_dir / f"frame-{fid}.color.jpg"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def _zeros_like_depth(depth_shape: Tuple[int, int]) -> np.ndarray:
    return np.zeros(depth_shape, dtype=np.float32)


def fuse(
    must3r_sequence_dir: Path,
    mast3r_sfm_sequence_dir: Path,
    output_sequence_dir: Path,
    min_conf_thr: float,
    min_alignment_frames: int,
) -> Dict[str, object]:
    """Write a hybrid sequence dir (MASt3R-SfM poses + MUSt3R depth/xyz).

    Returns the summary written to ``fuse_summary.json``.
    """
    output_sequence_dir.mkdir(parents=True, exist_ok=True)

    # _info.txt: copy from MASt3R-SfM sequence (matches the pose source). The
    # hybrid output's unprojection is xyz-based (voxelise OBJECTX_VOXEL_DEPTH_SOURCE
    # = must3r_xyz_if_available), so the Tango-intrinsic values carried inside
    # _info.txt are not read for back-projection. We still copy it because
    # depth_shift etc. metadata are required by downstream loaders.
    src_info = mast3r_sfm_sequence_dir / "_info.txt"
    if not src_info.exists():
        raise FileNotFoundError(f"missing {src_info}")
    shutil.copy2(src_info, output_sequence_dir / "_info.txt")

    must3r_ids = set(_enumerate_frame_ids(must3r_sequence_dir))
    mast3r_ids = set(_enumerate_frame_ids(mast3r_sfm_sequence_dir))
    common_ids = sorted(must3r_ids & mast3r_ids)
    if not common_ids:
        raise RuntimeError("no common frame IDs between the two backends")

    print(
        f"[fuse] frame IDs: must3r={len(must3r_ids)} "
        f"mast3r_sfm={len(mast3r_ids)} common={len(common_ids)}"
    )

    # Load poses. MUSt3R pose.txt is c2w (camera-to-world); MASt3R-SfM pose.txt
    # is w2c. Keep both as c2w internally for alignment on camera centers.
    must3r_c2w: Dict[str, np.ndarray] = {}
    mast3r_c2w: Dict[str, np.ndarray] = {}
    mast3r_w2c: Dict[str, np.ndarray] = {}
    for fid in common_ids:
        must3r_c2w[fid] = _load_pose(
            must3r_sequence_dir / f"frame-{fid}.pose.txt"
        )
        w2c = _load_pose(mast3r_sfm_sequence_dir / f"frame-{fid}.pose.txt")
        mast3r_w2c[fid] = w2c
        mast3r_c2w[fid] = _invert_se3(w2c)

    # Alignment candidates: both sides produce non-degenerate poses and non-empty
    # depth / xyz on the MUSt3R side.
    alignment_ids: List[str] = []
    for fid in common_ids:
        must3r_center = _camera_center(must3r_c2w[fid])
        mast3r_center = _camera_center(mast3r_c2w[fid])
        if not (
            np.all(np.isfinite(must3r_center))
            and np.all(np.isfinite(mast3r_center))
        ):
            continue
        if _is_identity_like(must3r_c2w[fid]) or _is_identity_like(
            mast3r_c2w[fid]
        ):
            continue
        depth_ok = _raw_depth_has_content(
            must3r_sequence_dir / f"frame-{fid}.depth_raw.npy"
        )
        xyz_ok = _xyz_has_content(
            must3r_sequence_dir / f"frame-{fid}.xyz.npy"
        )
        if depth_ok and xyz_ok:
            alignment_ids.append(fid)

    print(f"[fuse] alignment candidates: {len(alignment_ids)}/{len(common_ids)}")
    if len(alignment_ids) < min_alignment_frames:
        raise RuntimeError(
            f"only {len(alignment_ids)} alignment frames; need at least "
            f"{min_alignment_frames}"
        )

    # sim3: src=must3r centers, dst=mast3r centers -> scale brings MUSt3R scale
    # into MASt3R-SfM world scale. Only the scale factor is needed for
    # cam-frame quantities (depth, xyz); rotation/translation describe the
    # world-frame relationship which is discarded (we keep MASt3R-SfM world).
    src_centers = np.stack([_camera_center(must3r_c2w[fid]) for fid in alignment_ids])
    dst_centers = np.stack([_camera_center(mast3r_c2w[fid]) for fid in alignment_ids])
    sim3 = _compute_sim3(src_centers, dst_centers)
    if sim3 is None:
        raise RuntimeError(
            "sim3 alignment failed (insufficient motion or degenerate points)"
        )
    scale, _R, _t, info = sim3
    print(
        f"[fuse] sim3: scale={scale:.4f} n_pairs={info['n_pairs']} "
        f"inlier_frac={info['inlier_frac']:.3f} "
        f"resid median={info['resid_median']:.3f} max={info['resid_max']:.3f}"
    )

    # Probe one MUSt3R depth shape so we can zero frames that lack depth.
    probe_path = must3r_sequence_dir / f"frame-{alignment_ids[0]}.depth_raw.npy"
    depth_shape = np.load(probe_path).shape
    xyz_probe_path = must3r_sequence_dir / f"frame-{alignment_ids[0]}.xyz.npy"
    xyz_shape = np.load(xyz_probe_path).shape

    n_written = 0
    n_zeroed = 0
    for fid in common_ids:
        # Pose: MASt3R-SfM w2c, unchanged.
        np.savetxt(
            output_sequence_dir / f"frame-{fid}.pose.txt",
            mast3r_w2c[fid],
            fmt="%.18e",
        )
        _copy_color(mast3r_sfm_sequence_dir, output_sequence_dir, fid)

        must3r_raw_path = must3r_sequence_dir / f"frame-{fid}.depth_raw.npy"
        must3r_conf_path = must3r_sequence_dir / f"frame-{fid}.conf.npy"
        must3r_xyz_path = must3r_sequence_dir / f"frame-{fid}.xyz.npy"

        usable = (
            must3r_raw_path.exists()
            and must3r_conf_path.exists()
            and must3r_xyz_path.exists()
            and _raw_depth_has_content(must3r_raw_path)
            and _xyz_has_content(must3r_xyz_path)
        )

        if usable:
            depth_raw = np.load(must3r_raw_path).astype(np.float32) * float(scale)
            conf_map = np.load(must3r_conf_path).astype(np.float32)
            xyz_map = np.load(must3r_xyz_path).astype(np.float32) * float(scale)
            depth_raw[~np.isfinite(depth_raw)] = 0.0
            xyz_map[~np.isfinite(xyz_map)] = 0.0
            mask = conf_map > float(min_conf_thr)
            depth_masked = np.where(mask, depth_raw, 0.0).astype(np.float32)
        else:
            depth_raw = _zeros_like_depth(depth_shape)
            depth_masked = _zeros_like_depth(depth_shape)
            conf_map = _zeros_like_depth(depth_shape)
            xyz_map = np.zeros(xyz_shape, dtype=np.float32)
            n_zeroed += 1

        save_depth(depth_masked, str(output_sequence_dir), frame_id=fid)
        save_depth_raw(depth_raw, str(output_sequence_dir), frame_id=fid)
        save_confidence(conf_map, str(output_sequence_dir), frame_id=fid)
        save_xyz_map(xyz_map, str(output_sequence_dir), frame_id=fid)
        n_written += 1

    summary = {
        "must3r_sequence_dir": str(must3r_sequence_dir),
        "mast3r_sfm_sequence_dir": str(mast3r_sfm_sequence_dir),
        "output_sequence_dir": str(output_sequence_dir),
        "n_common": len(common_ids),
        "n_alignment": len(alignment_ids),
        "n_written": n_written,
        "n_zeroed": n_zeroed,
        "sim3_scale": float(scale),
        "sim3_info": {k: float(v) for k, v in info.items()},
        "min_conf_thr": float(min_conf_thr),
        "pose_source": "mast3r_sfm (w2c)",
        "depth_source": "must3r (scaled by sim3 scale)",
    }
    with open(output_sequence_dir / "fuse_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fuse] written={n_written} zeroed={n_zeroed} out={output_sequence_dir}")
    return summary


def _resolve_scene_dirs(
    reconstruction_root: Path,
    scene_id: str,
    must3r_scenes_dirname: str,
    mast3r_sfm_scenes_dirname: str,
    output_scenes_dirname: str,
) -> Tuple[Path, Path, Path]:
    must3r_seq = reconstruction_root / must3r_scenes_dirname / scene_id / "sequence"
    mast3r_seq = reconstruction_root / mast3r_sfm_scenes_dirname / scene_id / "sequence"
    out_seq = reconstruction_root / output_scenes_dirname / scene_id / "sequence"
    for path in (must3r_seq, mast3r_seq):
        if not path.exists():
            raise FileNotFoundError(f"required sequence dir missing: {path}")
    return must3r_seq, mast3r_seq, out_seq


def _link_scene_siblings(
    baseline_root: Optional[Path],
    reconstruction_root: Path,
    scene_id: str,
    output_scenes_dirname: str,
    mast3r_sfm_scenes_dirname: str,
) -> None:
    """Mirror the non-sequence scene files into the hybrid scene dir.

    Downstream voxelise / render steps read siblings (labels.*.ply, data.npy,
    mesh.refined.*, semseg.v2.json, etc.) next to ``sequence/``. Neither
    MUSt3R nor MASt3R-SfM rewrites these — they live only in the baseline
    scenes/. Symlink them in so the hybrid scene dir is self-contained.
    """
    output_scene_dir = reconstruction_root / output_scenes_dirname / scene_id
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    def _link_from(source_scene_dir: Path) -> None:
        if not source_scene_dir.exists():
            return
        for item in source_scene_dir.iterdir():
            if item.name == "sequence":
                continue
            target = output_scene_dir / item.name
            if target.exists() or target.is_symlink():
                continue
            os.symlink(item, target)

    # Baseline carries the GT meshes / labels / segments which downstream
    # readers still consume (e.g. gt_mesh fallback, rendering siblings).
    if baseline_root is not None:
        _link_from(baseline_root / "scenes" / scene_id)
    # Fall back to anything extra the mast3r_sfm scene dir accumulated.
    _link_from(reconstruction_root / mast3r_sfm_scenes_dirname / scene_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse MASt3R-SfM poses with MUSt3R depth/xyz into a scene "
            "directory compatible with voxelise_features."
        )
    )
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, required=True)
    parser.add_argument(
        "--must3r-scenes-dirname",
        type=str,
        default="scenes_sam2_must3r",
        help="Directory name under reconstruction-root holding MUSt3R poses+depth.",
    )
    parser.add_argument(
        "--mast3r-sfm-scenes-dirname",
        type=str,
        default="scenes_sam2_mast3r_sfm",
        help="Directory name under reconstruction-root holding MASt3R-SfM poses+depth.",
    )
    parser.add_argument(
        "--output-scenes-dirname",
        type=str,
        default="scenes_sam2_mast3r_sfm_must3r",
        help="Directory name under reconstruction-root for fused output.",
    )
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--min-conf-thr", type=float, default=0.4)
    parser.add_argument("--min-alignment-frames", type=int, default=20)
    args = parser.parse_args()

    must3r_seq, mast3r_seq, out_seq = _resolve_scene_dirs(
        reconstruction_root=args.reconstruction_root,
        scene_id=args.scene_id,
        must3r_scenes_dirname=args.must3r_scenes_dirname,
        mast3r_sfm_scenes_dirname=args.mast3r_sfm_scenes_dirname,
        output_scenes_dirname=args.output_scenes_dirname,
    )
    _link_scene_siblings(
        baseline_root=args.baseline_root,
        reconstruction_root=args.reconstruction_root,
        scene_id=args.scene_id,
        output_scenes_dirname=args.output_scenes_dirname,
        mast3r_sfm_scenes_dirname=args.mast3r_sfm_scenes_dirname,
    )
    fuse(
        must3r_sequence_dir=must3r_seq,
        mast3r_sfm_sequence_dir=mast3r_seq,
        output_sequence_dir=out_seq,
        min_conf_thr=args.min_conf_thr,
        min_alignment_frames=args.min_alignment_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
