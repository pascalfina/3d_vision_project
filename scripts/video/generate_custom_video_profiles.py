#!/usr/bin/env python3
"""Generate Object-X workflow profiles for prepared custom RGB videos."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="3RScan-like root produced by prepare_video_dataset.py.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("configs/workflows/scene_profiles"),
    )
    parser.add_argument(
        "--pi3x-template",
        type=Path,
        default=Path("configs/workflows/scene_profiles/scene_5341b7e3_sam2_pi3x.json"),
    )
    parser.add_argument(
        "--final-template",
        type=Path,
        default=Path("configs/workflows/scene_profiles/scene_8f0f144b_sam2_must3r.json"),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/work/courses/3dv/team35/pafina"),
    )
    parser.add_argument(
        "--samobject-root",
        type=Path,
        default=Path("/work/courses/3dv/team35/pafina/sam2object-custom-video"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/workflows/scene_profiles/custom_video_profiles.txt"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def set_path(mapping: dict, keys: tuple[str, ...], value) -> None:
    cursor = mapping
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            return
        cursor = child
    cursor[keys[-1]] = value


def scene_frame_count(dataset_root: Path, scene_id: str) -> int:
    seq = dataset_root / "scenes" / scene_id / "sequence"
    return len(list(seq.glob("frame-*.color.jpg")))


def discover_scene_ids(dataset_root: Path) -> list[str]:
    manifest = dataset_root / "custom_video_dataset_manifest.json"
    if manifest.exists():
        data = load_json(manifest)
        scene_ids = [item["scene_id"] for item in data.get("scenes", [])]
        if scene_ids:
            return scene_ids
    scenes = dataset_root / "scenes"
    return sorted(path.name for path in scenes.iterdir() if (path / "sequence").is_dir())


def merge_final_sections(profile: dict, final_template: dict) -> None:
    for key in ["slat", "u3dgs", "plot_voxelised"]:
        if key in final_template:
            profile[key] = deepcopy(final_template[key])


def make_profile(
    pi3x_template: dict,
    final_template: dict,
    dataset_root: Path,
    work_root: Path,
    samobject_root: Path,
    scene_id: str,
) -> tuple[str, dict]:
    profile_name = f"custom_{scene_id}_pi3x_samobject"
    recon_root = work_root / f"objectx-data-{scene_id}-pi3x-samobject"
    must3r_root = work_root / f"objectx-data-{scene_id}-pi3x-samobject-must3r"
    pred_ready_root = work_root / f"objectx-data-{scene_id}-predready-pi3x-samobject-v1"
    frame_count = scene_frame_count(dataset_root, scene_id)

    profile = deepcopy(pi3x_template)
    merge_final_sections(profile, final_template)

    profile["name"] = profile_name
    profile["scene_id"] = scene_id
    profile["split"] = "val"
    profile["input_variant"] = "legacy+sam+must3r"
    profile["mask_source"] = "sam2_projection"

    set_path(profile, ("roots", "baseline"), str(dataset_root))
    set_path(profile, ("roots", "reconstruction"), str(recon_root))
    set_path(profile, ("roots", "pred_ready"), str(pred_ready_root))
    set_path(
        profile,
        ("artifacts", "manifest"),
        f"${{OBJECTX_REPO_DEBUG_ROOT}}/{scene_id}_custom_pi3x_samobject_manifest.json",
    )
    set_path(
        profile,
        ("artifacts", "joint_ply"),
        f"${{OBJECTX_REPO_VIS_ROOT}}/{scene_id}_custom_pi3x_samobject_joint.ply",
    )

    for section in ["segment_inputs", "must3r", "pi3x"]:
        set_path(profile, (section, "input_root"), str(dataset_root))
    set_path(profile, ("segment_inputs", "output_root"), str(recon_root))
    set_path(
        profile,
        ("segment_inputs", "log"),
        f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/{scene_id}_custom_pi3x_segment_inputs.log",
    )
    set_path(profile, ("must3r", "output_root"), str(must3r_root))
    set_path(
        profile,
        ("must3r", "log"),
        f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/{scene_id}_custom_pi3x_must3r.log",
    )
    set_path(profile, ("pi3x", "output_root"), str(recon_root))
    set_path(
        profile,
        ("pi3x", "log"),
        f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/{scene_id}_custom_pi3x.log",
    )
    set_path(
        profile,
        ("pi3x", "env", "OBJECTX_PI3X_EXTERNAL_POSE_DIR"),
        f"{must3r_root}/scenes_sam2_must3r/{scene_id}/sequence",
    )

    set_path(profile, ("samobject", "root_dir"), str(samobject_root))
    set_path(profile, ("samobject", "baseline_root"), str(dataset_root))
    set_path(
        profile,
        ("samobject", "source_sequence_dir"),
        f"{recon_root}/scenes_sam2_pi3x/{scene_id}/sequence",
    )
    set_path(
        profile,
        ("samobject", "log"),
        f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/{scene_id}_custom_samobject.log",
    )
    set_path(profile, ("samobject", "env", "SAMOBJECT_ADAPTER"), "pi3x_full_samobject")
    set_path(profile, ("samobject", "env", "SAMOBJECT_USE_PI3X_SURFACE"), "1")

    set_path(profile, ("voxelise", "scene_source_dirname"), "scenes_sam2_pi3x")
    set_path(profile, ("voxelise", "mask_source"), "gt_projection")
    set_path(profile, ("voxelise", "objects_filename"), "objects_sam2.json")
    set_path(
        profile,
        ("voxelise", "log"),
        f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/{scene_id}_custom_voxelise.log",
    )
    set_path(profile, ("voxelise", "env", "OBJECTX_MASK_SOURCE"), "gt_projection")
    set_path(profile, ("voxelise", "env", "OBJECTX_VOXEL_OBJECT_SOURCE"), "hybrid_masks")
    set_path(profile, ("voxelise", "env", "OBJECTX_VOXEL_OBJECTS_FILENAME"), "objects_sam2.json")
    set_path(profile, ("voxelise", "env", "OBJECTX_VOXEL_TSDF_FALLBACK_TO_LIFTED"), "1")
    set_path(profile, ("voxelise", "env", "OBJECTX_VOXEL_REQUIRE_XYZ"), "")
    set_path(profile, ("voxelise", "env", "OBJECTX_VOXEL_MAX_VIEWS"), "100")
    set_path(profile, ("voxelise", "env", "OBJECTX_VOXEL_POSE_JUMP_MAX_TRANSLATION"), "1.5")
    set_path(profile, ("voxelise", "env", "OBJECTX_VOXEL_POSE_JUMP_MAX_Z_TRANSLATION"), "1.0")
    set_path(profile, ("voxelise", "env", "OBJECTX_VOXEL_POSE_JUMP_RELATIVE_FACTOR"), "10.0")

    set_path(profile, ("build_pred_ready", "baseline_root"), str(dataset_root))
    set_path(profile, ("build_pred_ready", "reconstruction_root"), str(recon_root))
    set_path(profile, ("build_pred_ready", "target_root"), str(pred_ready_root))
    set_path(profile, ("build_pred_ready", "reconstruction_scenes_dirname"), "scenes_sam2_pi3x")

    set_path(profile, ("render_bundle", "label"), profile_name)
    set_path(profile, ("render_bundle", "data_root"), str(recon_root))
    set_path(profile, ("render_bundle", "max_views"), frame_count)
    set_path(profile, ("render_bundle", "mask_source"), "gt_projection")
    set_path(profile, ("render_bundle", "env", "OBJECTX_VIS_SCENES_DIRNAME"), "scenes_sam2_pi3x")

    voxelised_vis_root = work_root / "objectx-vis" / "custom_video" / profile_name / "voxelised"
    voxelised_log = work_root / "logs" / f"{scene_id}_custom_plot_voxelised.log"
    profile["plot_voxelised"] = {
        "data_root": str(recon_root),
        "mask_root": str(recon_root),
        "replacement_root": str(pred_ready_root),
        "mask_source": "gt_projection",
        "background_remove_mode": "loaded",
        "pose_mode": "raw",
        "lift_coord_system": "pinhole",
        "frame_selection": "diverse_area",
        "max_views": frame_count,
        "max_bg_points": 3000000,
        "max_obj_points": 3000000,
        "label": f"{profile_name}_voxelised",
        "out_dir": str(voxelised_vis_root),
        "log": str(voxelised_log),
        "env": {
            "OBJECTX_VIS_SCENES_DIRNAME": "scenes_sam2_pi3x",
            "OBJECTX_VIS_DEPTH_SOURCE": "must3r_xyz_if_available",
            "OBJECTX_VIS_RAW_DEPTH_CONF_THR": "0.10",
            "OBJECTX_VIS_RAW_DEPTH_MAX": "5.0",
            "OBJECTX_VIS_BG_VOXEL_SIZE": "0.07",
            "OBJECTX_VIS_BG_MIN_POINTS_PER_VOXEL": "6",
            "OBJECTX_VIS_BG_MIN_VIEWS_PER_VOXEL": "3",
            "OBJECTX_VIS_BG_MIN_KEEP_FRACTION": "0.05",
        },
    }

    geom_debug_root = work_root / "objectx-vis" / "custom_video" / profile_name / "geom_debug"
    geom_debug_log = work_root / "logs" / f"{scene_id}_custom_geom_debug.log"
    profile["geom_debug"] = {
        "data_root": str(recon_root),
        "scenes_dirname": "scenes_sam2_pi3x",
        "mask_source": "none",
        "max_views": frame_count,
        "max_points": 3000000,
        "write_gt_html": False,
        "label": f"{profile_name}_geom_debug",
        "out_dir": str(geom_debug_root),
        "log": str(geom_debug_log),
    }

    for section in ["slat", "u3dgs"]:
        if section in profile:
            set_path(profile, (section, "data_root"), str(pred_ready_root))
            set_path(profile, (section, "mask_source"), "gt_projection")
            set_path(profile, (section, "env", "OBJECTX_MASK_SOURCE"), "gt_projection")
            set_path(profile, (section, "log"), f"${{OBJECTX_REPO_DEBUG_ROOT}}/{scene_id}_custom_{section}.log")

    profile.pop("geometry_eval", None)
    profile.pop("compare_arrangement", None)
    return profile_name, profile


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    scene_ids = discover_scene_ids(dataset_root)
    if not scene_ids:
        raise SystemExit(f"No scenes found in {dataset_root}")

    pi3x_template = load_json(args.pi3x_template)
    final_template = load_json(args.final_template)

    generated: list[str] = []
    rows: list[str] = []
    for scene_id in scene_ids:
        profile_name, profile = make_profile(
            pi3x_template=pi3x_template,
            final_template=final_template,
            dataset_root=dataset_root,
            work_root=args.work_root,
            samobject_root=args.samobject_root,
            scene_id=scene_id,
        )
        generated.append(profile_name)
        path = args.profile_dir / f"{profile_name}.json"
        rows.append(f"{profile_name}\t{scene_id}\t{scene_frame_count(dataset_root, scene_id)}\t{path}")
        if not args.dry_run:
            write_json(path, profile)

    if not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text("\n".join(generated) + "\n", encoding="utf-8")
        table_path = args.manifest.with_suffix(".tsv")
        table_path.write_text("profile\tscene_id\tframes\tpath\n" + "\n".join(rows) + "\n", encoding="utf-8")

    action = "would write" if args.dry_run else "wrote"
    print(f"[custom-video-profiles] {action} {len(generated)} profiles")
    print(f"[custom-video-profiles] manifest={args.manifest}")
    for profile_name in generated:
        print(f"  {profile_name}")


if __name__ == "__main__":
    main()
