#!/usr/bin/env python3
"""Generate lightweight ScanNet Pi3X geometry profiles from a scene TSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MUST3R_ENV = {
    "OBJECTX_MUST3R_EXECUTION_MODE": "vidslam",
    "OBJECTX_MUST3R_SLAM_LOCAL_CONTEXT_SIZE": "8",
    "OBJECTX_MUST3R_NUM_REFINEMENTS_ITERATIONS": "2",
    "OBJECTX_MUST3R_NUM_MEM_IMAGES": "64",
    "OBJECTX_MUST3R_RESOLUTION": "512",
    "OBJECTX_MUST3R_SUBSAMPLE": "1",
    "OBJECTX_MUST3R_CHUNK_SIZE": "0",
    "OBJECTX_MUST3R_CHUNK_OVERLAP": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OBJECTX_MUST3R_MIN_CONF_KEYFRAME": "1.5",
    "OBJECTX_MUST3R_KEYFRAME_OVERLAP_THR": "0.05",
    "OBJECTX_MUST3R_OVERLAP_PERCENTILE": "85",
    "OBJECTX_MUST3R_MIN_CONF_THR": "1.2",
    "OBJECTX_MUST3R_DEPTH_MASK_MODE": "hard",
    "OBJECTX_MUST3R_SAVE_RAW_DEPTH": "1",
    "OBJECTX_MUST3R_SAVE_CONFIDENCE": "1",
    "OBJECTX_MUST3R_POSE_JUMP_MAX_TRANSLATION": "1.5",
    "OBJECTX_MUST3R_POSE_JUMP_MAX_Z_TRANSLATION": "1.0",
    "OBJECTX_MUST3R_POSE_JUMP_MAX_ROTATION_DEG": "0.0",
    "OBJECTX_MUST3R_POSE_JUMP_RELATIVE_FACTOR": "10.0",
    "OBJECTX_MUST3R_ZERO_INVALID_POSE_DEPTHS": "1",
    "OBJECTX_MUST3R_USE_DUST3R_REFINEMENT": "0",
    "OBJECTX_DUST3R_PAIR_STRIDES": "1",
    "OBJECTX_DUST3R_WEIGHT": "0.5",
    "OBJECTX_DUST3R_MIN_CONF": "1.0",
    "OBJECTX_DUST3R_IMAGE_SIZE": "512",
    "OBJECTX_DUST3R_BATCH_SIZE": "1",
    "OBJECTX_MUST3R_USE_DUST3R_LOOP_CLOSURE": "0",
    "OBJECTX_MUST3R_USE_DUST3R_INIT": "0",
    "OBJECTX_DUST3R_INIT_WINDOW": "16",
    "OBJECTX_DUST3R_INIT_SCENE_GRAPH": "complete",
    "OBJECTX_DUST3R_INIT_NITER": "300",
    "OBJECTX_DUST3R_INIT_LR": "0.01",
    "OBJECTX_DUST3R_INIT_INIT": "mst",
}


PI3X_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OBJECTX_PI3X_PIXEL_LIMIT": "80000",
    "OBJECTX_PI3X_CHUNK_SIZE": "512",
    "OBJECTX_PI3X_OVERLAP": "0",
    "OBJECTX_PI3X_ANCHOR_COUNT": "0",
    "OBJECTX_PI3X_FRAME_STRIDE": "1",
    "OBJECTX_PI3X_CONF_THR": "0.10",
    "OBJECTX_PI3X_ALIGN_MODE": "sim3",
    "OBJECTX_PI3X_USE_INTRINSICS": "1",
    "OBJECTX_PI3X_LOCAL_PIXEL_LIMIT": "130000",
    "OBJECTX_PI3X_LOCAL_CHUNK_SIZE": "45",
    "OBJECTX_PI3X_LOCAL_OVERLAP": "8",
    "OBJECTX_PI3X_LOCAL_SCALE_MAX_RATIO": "2.50",
    "OBJECTX_PI3X_OUTPUT_NATIVE_RES": "1",
    "OBJECTX_PI3X_SAVE_RAW_DEPTH": "1",
    "OBJECTX_PI3X_SAVE_CONFIDENCE": "1",
    "OBJECTX_PI3X_POSE_JUMP_MAX_TRANSLATION": "1.5",
    "OBJECTX_PI3X_POSE_JUMP_MAX_Z_TRANSLATION": "1.0",
    "OBJECTX_PI3X_POSE_JUMP_MAX_ROTATION_DEG": "0.0",
    "OBJECTX_PI3X_POSE_JUMP_RELATIVE_FACTOR": "10.0",
    "OBJECTX_PI3X_ZERO_INVALID_POSE_DEPTHS": "1",
    "OBJECTX_PI3X_WEIGHTS": "/work/courses/3dv/team35/pafina/models/pi3/Pi3X.safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ScanNet Pi3X scene profiles.")
    parser.add_argument(
        "--scene-table",
        type=Path,
        default=Path("configs/workflows/scene_profiles/scannet_under300_scenes.tsv"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("configs/workflows/scene_profiles"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/workflows/scene_profiles/scannet_under300_profiles.txt"),
    )
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--prune-stale",
        action="store_true",
        help="Remove generated scannet_scene*_pi3x.json profiles not in the current manifest.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_scene_table(path: Path, max_scenes: int) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("rank\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        scene_id = parts[1]
        frames = int(parts[2])
        rows.append((scene_id, frames))
        if max_scenes and len(rows) >= max_scenes:
            break
    return rows


def profile_name(scene_id: str) -> str:
    return f"scannet_{scene_id}_pi3x"


def make_profile(scene_id: str, frame_count: int) -> dict:
    name = profile_name(scene_id)
    short = scene_id
    work_root = "${SCANNET_WORK_ROOT}"
    recon_root = f"{work_root}/objectx-data-scannet-{short}-pi3x"
    must3r_root = f"{work_root}/objectx-data-scannet-{short}-must3r"
    pi3x_env = dict(PI3X_ENV)
    pi3x_env["OBJECTX_PI3X_EXTERNAL_POSE_DIR"] = (
        f"{must3r_root}/scenes_sam2_must3r/{scene_id}/sequence"
    )

    profile = {
        "name": name,
        "dataset": "scannet",
        "scene_id": scene_id,
        "split": "val",
        "input_variant": "legacy",
        "mask_source": "gt_projection",
        "roots": {
            "baseline": "${SCANNET_ROOT}",
            "reconstruction": recon_root,
        },
        "must3r": {
            "config": "preprocessing/segmentation/pipeline.yaml",
            "run_must3r": True,
            "run_sam2": False,
            "run_registry": False,
            "input_root": "${SCANNET_ROOT}",
            "output_root": must3r_root,
            "mask_dirname": "gt_projection",
            "objects_filename": "objects.json",
            "scenes_dirname": "scenes_sam2_must3r",
            "env": MUST3R_ENV,
            "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{short}_must3r.log",
        },
        "pi3x": {
            "config": "preprocessing/segmentation/pipeline.yaml",
            "run_must3r": True,
            "run_sam2": False,
            "run_registry": False,
            "input_root": "${SCANNET_ROOT}",
            "output_root": recon_root,
            "mask_dirname": "gt_projection",
            "objects_filename": "objects.json",
            "scenes_dirname": "scenes_sam2_pi3x",
            "env": pi3x_env,
            "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{short}_pi3x.log",
        },
        "geometry_eval": {
            "dataset": "scannet",
            "method_name": name,
            "pred_input_mode": "sequence",
            "baseline_root": "${SCANNET_ROOT}",
            "pred_root": recon_root,
            "scenes_dirname": "scenes_sam2_pi3x",
            "write_debug_html": True,
            "debug_html_max_points": 300000,
            "visible_max_frames": min(frame_count, 160),
            "rgbd_max_frames": min(frame_count, 160),
            "group": "${GEOMETRY_EVAL_GROUP}",
            "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{short}_geometry_eval.log",
        },
    }
    sequence_max_frames = os.environ.get("SCANNET_GEOMETRY_SEQUENCE_MAX_FRAMES")
    if sequence_max_frames:
        profile["geometry_eval"]["sequence_max_frames"] = int(sequence_max_frames)
    return profile


def main() -> None:
    args = parse_args()
    scenes = read_scene_table(args.scene_table, args.max_scenes)
    if not scenes:
        raise SystemExit(f"No scenes found in {args.scene_table}")

    if not args.dry_run:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_names: list[str] = []
    for scene_id, frame_count in scenes:
        name = profile_name(scene_id)
        manifest_names.append(name)
        path = args.profile_dir / f"{name}.json"
        if not args.dry_run:
            path.write_text(json.dumps(make_profile(scene_id, frame_count), indent=2) + "\n", encoding="utf-8")

    if not args.dry_run:
        args.manifest.write_text("\n".join(manifest_names) + "\n", encoding="utf-8")
        if args.prune_stale:
            keep = {f"{name}.json" for name in manifest_names}
            for path in args.profile_dir.glob("scannet_scene*_pi3x.json"):
                if path.name not in keep:
                    path.unlink()

    action = "would write" if args.dry_run else "wrote"
    print(f"[scannet-profiles] {action} {len(manifest_names)} profiles")
    print(f"[scannet-profiles] manifest={args.manifest}")


if __name__ == "__main__":
    main()
