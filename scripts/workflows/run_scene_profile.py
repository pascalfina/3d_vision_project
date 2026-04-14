#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_ACTIONS = [
    "segment-inputs",
    "voxelise",
    "build-pred-ready",
    "validate-pred-ready",
    "compare-arrangement",
    "slat",
    "u3dgs",
    "render",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the current Object-X scene workflows from a short profile name "
            "instead of long ad-hoc commands."
        )
    )
    parser.add_argument("profile", nargs="?", help="Profile name or JSON file path.")
    parser.add_argument(
        "action",
        nargs="?",
        choices=DEFAULT_ACTIONS,
        help="Workflow step to execute.",
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Optional profile directory override. Defaults to configs/workflows/scene_profiles.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit.",
    )
    return parser.parse_args()


def profile_dir(repo_root: Path, override: str) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return repo_root / "configs" / "workflows" / "scene_profiles"


def resolve_profile_path(repo_root: Path, profile_arg: str, override_dir: str) -> Path:
    candidate = Path(profile_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()

    base_dir = profile_dir(repo_root, override_dir)
    if candidate.suffix == ".json":
        resolved = base_dir / candidate.name
    else:
        resolved = base_dir / f"{profile_arg}.json"
    if resolved.exists():
        return resolved.resolve()
    raise FileNotFoundError(
        f"Could not resolve profile '{profile_arg}'. Checked {candidate} and {resolved}."
    )


def load_profile(path: Path) -> dict:
    return json.loads(path.read_text())


def input_variant_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "workflows" / "input_variants.json"


def load_input_variants(repo_root: Path) -> dict:
    path = input_variant_config_path(repo_root)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Input variant config must be a dict: {path}")
    return data


def input_variant_name(profile: dict) -> str:
    return profile.get("input_variant", "legacy")


def input_variant_settings(repo_root: Path, profile: dict) -> dict:
    variants = load_input_variants(repo_root)
    name = input_variant_name(profile)
    value = variants.get(name)
    if value is None:
        if name == "legacy":
            return {"mask_source": "gt_projection", "run_segment_inputs": False}
        raise KeyError(
            f"Unknown input variant '{name}'. Configure it in {input_variant_config_path(repo_root)}."
        )
    if not isinstance(value, dict):
        raise ValueError(f"Input variant '{name}' must map to a dict.")
    return value


def profile_mask_source(repo_root: Path, profile: dict) -> str:
    return profile.get(
        "mask_source",
        input_variant_settings(repo_root, profile).get("mask_source", "gt_projection"),
    )


def list_profiles(repo_root: Path, override_dir: str) -> int:
    base_dir = profile_dir(repo_root, override_dir)
    if not base_dir.exists():
        print(f"No profile directory found at {base_dir}")
        return 0
    for path in sorted(base_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            print(path.stem)
            continue
        name = data.get("name", path.stem)
        scene_id = data.get("scene_id", "?")
        variant = data.get("input_variant", "legacy")
        print(f"{name}  scene_id={scene_id}  variant={variant}")
    return 0


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def print_run_header(profile_path: Path, action: str, cmd, env_updates: dict, log_path: Path):
    print(f"[profile] {profile_path.stem}")
    print(f"[action] {action}")
    if env_updates:
        print("[env]")
        for key in sorted(env_updates):
            value = os.environ.get(key, env_updates[key])
            if value is None:
                print(f"  unset {key}")
            else:
                print(f"  {key}={value}")
    if log_path is not None:
        print(f"[log] {log_path}")
    print(f"[cmd] {shell_join(cmd)}")


def stream_command(cmd, cwd: Path, env_updates: dict, log_path: Path, dry_run: bool) -> int:
    print_run_header(
        profile_path=Path(env_updates.pop("__profile_path__")),
        action=env_updates.pop("__action__"),
        cmd=cmd,
        env_updates=env_updates,
        log_path=log_path,
    )
    if dry_run:
        return 0

    env = os.environ.copy()
    for key, value in env_updates.items():
        if key in os.environ:
            continue
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
        return process.wait()
    finally:
        if log_handle is not None:
            log_handle.close()


def shared_value(profile: dict, key: str, default=None):
    return profile.get(key, default)


def roots(profile: dict) -> dict:
    value = profile.get("roots", {})
    if not isinstance(value, dict):
        raise ValueError("Profile field 'roots' must be a dict.")
    return value


def artifacts(profile: dict) -> dict:
    value = profile.get("artifacts", {})
    if not isinstance(value, dict):
        raise ValueError("Profile field 'artifacts' must be a dict.")
    return value


def section(profile: dict, name: str) -> dict:
    value = profile.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Profile section '{name}' must be a dict.")
    return value


def add_cli_arg(parts, flag: str, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            parts.append(flag)
        return
    if isinstance(value, list):
        for item in value:
            parts.extend([flag, str(item)])
        return
    parts.extend([flag, str(value)])


def build_voxelise_action(repo_root: Path, profile: dict):
    sec = section(profile, "voxelise")
    env_updates = {
        "DATA_ROOT_DIR": sec.get("data_root", roots(profile).get("reconstruction")),
        "OBJECTX_BASELINE_ROOT": sec.get("baseline_root", roots(profile).get("baseline")),
        "OBJECTX_SCENE_ID": sec.get("scene_id", shared_value(profile, "scene_id")),
        "SPLIT": sec.get("split", shared_value(profile, "split", "val")),
        "OBJECTX_MASK_SOURCE": sec.get(
            "mask_source", profile_mask_source(repo_root, profile)
        ),
        "RESET_TMP": str(sec.get("reset_tmp", 1)),
        "MAX_SCANS": str(sec.get("max_scans", 0)),
    }
    scene_source_dirname = sec.get("scene_source_dirname")
    if scene_source_dirname:
        env_updates["OBJECTX_SCENE_SOURCE_DIRNAME"] = str(scene_source_dirname)
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
    cmd = [
        "bash",
        str(repo_root / "scripts" / "voxel_annotations" / "pipeline" / "voxelise_features_tmp.sh"),
    ]
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, env_updates, log_path


def build_segment_inputs_action(repo_root: Path, profile: dict):
    sec = section(profile, "segment_inputs")
    variant = input_variant_settings(repo_root, profile)
    if not sec and not variant.get("run_segment_inputs", False):
        raise ValueError(
            f"Profile '{profile.get('name', 'unknown')}' uses variant '{input_variant_name(profile)}' "
            "which does not define a segment-inputs stage."
        )

    env_updates = {
        "OBJECTX_SEG_INPUT_ROOT": sec.get("input_root", roots(profile).get("baseline")),
        "OBJECTX_SEG_OUTPUT_ROOT": sec.get("output_root", roots(profile).get("reconstruction")),
        "OBJECTX_SEG_MASK_DIRNAME": sec.get(
            "mask_dirname", variant.get("mask_output_dirname", "gt_projection_predicted")
        ),
        "OBJECTX_SEG_OBJECTS_FILENAME": sec.get(
            "objects_filename", variant.get("objects_filename", "objects_predicted.json")
        ),
        "OBJECTX_SEG_SCENES_DIRNAME": sec.get(
            "scenes_dirname", variant.get("scenes_dirname", "scenes_predicted")
        ),
        "OBJECTX_SEG_RUN_MUST3R": str(
            int(sec.get("run_must3r", variant.get("run_must3r", True)))
        ),
        "OBJECTX_SEG_RUN_SAM2": str(
            int(sec.get("run_sam2", variant.get("run_sam2", True)))
        ),
        "OBJECTX_SEG_RUN_REGISTRY": str(
            int(sec.get("run_registry", variant.get("run_registry", True)))
        ),
        "MUST3R_PATH": sec.get("must3r_path", str(repo_root / "dependencies" / "must3r")),
    }
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)

    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "preprocessing" / "segmentation" / "run_pipeline.py"),
        "--config",
        sec.get("config", "preprocessing/segmentation/pipeline.yaml"),
        "--scene",
        sec.get("scene_id", shared_value(profile, "scene_id")),
    ]
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, env_updates, log_path


def build_pred_ready_action(repo_root: Path, profile: dict):
    sec = section(profile, "build_pred_ready")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "segmentation" / "pipeline" / "build_pred_ready_scene_root.py"),
    ]
    add_cli_arg(cmd, "--baseline-root", sec.get("baseline_root", roots(profile).get("baseline")))
    add_cli_arg(
        cmd,
        "--reconstruction-root",
        sec.get("reconstruction_root", roots(profile).get("reconstruction")),
    )
    add_cli_arg(
        cmd,
        "--reconstruction-scenes-dirname",
        sec.get("reconstruction_scenes_dirname"),
    )
    add_cli_arg(cmd, "--target-root", sec.get("target_root", roots(profile).get("pred_ready")))
    add_cli_arg(cmd, "--scene-id", sec.get("scene_id", shared_value(profile, "scene_id")))
    add_cli_arg(cmd, "--split", sec.get("split", shared_value(profile, "split", "val")))
    add_cli_arg(cmd, "--knn", sec.get("knn", 4))
    add_cli_arg(cmd, "--min-voxels", sec.get("min_voxels", 32))
    point_counts = sec.get("point_counts", [64, 128, 256, 512])
    if point_counts:
        cmd.append("--point-counts")
        cmd.extend(str(x) for x in point_counts)
    add_cli_arg(cmd, "--overwrite", sec.get("overwrite", False))
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, {}, log_path


def build_validate_action(repo_root: Path, profile: dict):
    sec = section(profile, "validate_pred_ready")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "segmentation" / "validation" / "validate_pred_ready_scene_root.py"),
    ]
    add_cli_arg(cmd, "--root", sec.get("root", roots(profile).get("pred_ready")))
    add_cli_arg(cmd, "--scene-id", sec.get("scene_id", shared_value(profile, "scene_id")))
    add_cli_arg(cmd, "--split", sec.get("split", shared_value(profile, "split", "val")))
    add_cli_arg(cmd, "--config", sec.get("config", "configs/config.yaml"))
    add_cli_arg(cmd, "--skip-dataset-check", sec.get("skip_dataset_check", False))
    add_cli_arg(cmd, "--out-dir", sec.get("out_dir"))
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, {}, log_path


def build_compare_action(repo_root: Path, profile: dict):
    sec = section(profile, "compare_arrangement")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "segmentation" / "validation" / "compare_scene_arrangement.py"),
    ]
    add_cli_arg(cmd, "--pred-root", sec.get("pred_root", roots(profile).get("pred_ready")))
    add_cli_arg(cmd, "--gt-root", sec.get("gt_root", roots(profile).get("baseline")))
    add_cli_arg(cmd, "--scene-id", sec.get("scene_id", shared_value(profile, "scene_id")))
    add_cli_arg(cmd, "--point-level", sec.get("point_level"))
    add_cli_arg(cmd, "--anchor-obj-id", sec.get("anchor_obj_id"))
    add_cli_arg(cmd, "--max-points-per-set", sec.get("max_points_per_set", 60000))
    add_cli_arg(cmd, "--out-dir", sec.get("out_dir"))
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, {}, log_path


def build_slat_action(repo_root: Path, profile: dict):
    sec = section(profile, "slat")
    mask_source = sec.get("mask_source", profile_mask_source(repo_root, profile))
    env_updates = {
        "DATA_ROOT_DIR": sec.get("data_root", roots(profile).get("pred_ready")),
        "SPLIT": sec.get("split", shared_value(profile, "split", "val")),
        "SCENE_ID": sec.get("scene_id", shared_value(profile, "scene_id")),
        "OBJECTX_MASK_SOURCE": mask_source,
    }
    if mask_source != "gt_projection":
        env_updates["OBJECTX_MASK_ROOT"] = sec.get(
            "mask_root", roots(profile).get("reconstruction")
        )
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
    cmd = [
        "bash",
        str(repo_root / "scripts" / "inference" / "pipeline" / "run_pipeline_slat_tmp.sh"),
    ]
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, env_updates, log_path


def build_u3dgs_action(repo_root: Path, profile: dict):
    sec = section(profile, "u3dgs")
    mask_source = sec.get("mask_source", profile_mask_source(repo_root, profile))
    env_updates = {
        "DATA_ROOT_DIR": sec.get("data_root", roots(profile).get("pred_ready")),
        "SPLIT": sec.get("split", shared_value(profile, "split", "val")),
        "SCENE_ID": sec.get("scene_id", shared_value(profile, "scene_id")),
        "OBJECTX_MASK_SOURCE": mask_source,
        "OBJECTX_INFER_CPU_DENSE_DECODE_FALLBACK": str(
            sec.get("cpu_dense_decode_fallback", 1)
        ),
    }
    if mask_source != "gt_projection":
        env_updates["OBJECTX_MASK_ROOT"] = sec.get(
            "mask_root", roots(profile).get("reconstruction")
        )
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
    for key in sec.get("unset_env", []):
        env_updates[key] = None
    cmd = [
        "bash",
        str(repo_root / "scripts" / "inference" / "pipeline" / "run_pipeline_u3dgs_tmp.sh"),
    ]
    if sec.get("visualize", True):
        cmd.append("--visualize")
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, env_updates, log_path


def build_render_action(repo_root: Path, profile: dict):
    sec = section(profile, "render_bundle")
    mask_source = sec.get("mask_source", profile_mask_source(repo_root, profile))
    cmd = [
        "bash",
        str(
            repo_root
            / "scripts"
            / "segmentation"
            / "visualization"
            / "render_joint_depth_background_bundle.sh"
        ),
    ]
    add_cli_arg(cmd, "--scan-id", sec.get("scan_id", shared_value(profile, "scene_id")))
    add_cli_arg(
        cmd,
        "--replacement-root",
        sec.get("replacement_root", roots(profile).get("pred_ready")),
    )
    add_cli_arg(cmd, "--label", sec.get("label"))
    add_cli_arg(cmd, "--manifest", sec.get("manifest", artifacts(profile).get("manifest")))
    add_cli_arg(cmd, "--data-root", sec.get("data_root", roots(profile).get("baseline")))
    if mask_source != "gt_projection":
        add_cli_arg(
            cmd,
            "--mask-root",
            sec.get("mask_root", roots(profile).get("reconstruction")),
        )
    add_cli_arg(cmd, "--joint-ply", sec.get("joint_ply", artifacts(profile).get("joint_ply")))
    add_cli_arg(cmd, "--mask-source", mask_source)
    add_cli_arg(cmd, "--background-remove-mode", sec.get("background_remove_mode", "loaded"))
    add_cli_arg(cmd, "--pose-mode", sec.get("pose_mode", "raw"))
    add_cli_arg(cmd, "--lift-coord-system", sec.get("lift_coord_system", "pinhole"))
    add_cli_arg(cmd, "--frame-selection", sec.get("frame_selection", "all"))
    add_cli_arg(cmd, "--max-views", sec.get("max_views"))
    add_cli_arg(cmd, "--bg-max-points", sec.get("bg_max_points"))
    add_cli_arg(cmd, "--joint-max-points", sec.get("joint_max_points"))
    add_cli_arg(cmd, "--joint-opacity-min", sec.get("joint_opacity_min"))
    add_cli_arg(cmd, "--joint-opacity-quantile", sec.get("joint_opacity_quantile"))
    add_cli_arg(cmd, "--joint-scale-quantile", sec.get("joint_scale_quantile"))
    add_cli_arg(cmd, "--fit-mode", sec.get("fit_mode"))
    add_cli_arg(cmd, "--fit-margin", sec.get("fit_margin"))
    add_cli_arg(cmd, "--num-frames", sec.get("num_frames"))
    add_cli_arg(cmd, "--fps", sec.get("fps"))
    add_cli_arg(cmd, "--fig-width", sec.get("fig_width"))
    add_cli_arg(cmd, "--fig-height", sec.get("fig_height"))
    add_cli_arg(cmd, "--bg-point-size", sec.get("bg_point_size"))
    add_cli_arg(cmd, "--joint-point-size", sec.get("joint_point_size"))
    add_cli_arg(cmd, "--out-dir", sec.get("out_dir"))
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, {}, log_path


ACTION_BUILDERS = {
    "segment-inputs": build_segment_inputs_action,
    "voxelise": build_voxelise_action,
    "build-pred-ready": build_pred_ready_action,
    "validate-pred-ready": build_validate_action,
    "compare-arrangement": build_compare_action,
    "slat": build_slat_action,
    "u3dgs": build_u3dgs_action,
    "render": build_render_action,
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    if args.list_profiles:
        raise SystemExit(list_profiles(repo_root, args.profile_dir))

    if not args.profile or not args.action:
        raise SystemExit(
            "Provide both PROFILE and ACTION, or use --list-profiles. "
            "Example: bash scripts/workflows/run_scene_profile.sh cabinet_predready_v2_floorfix u3dgs"
        )

    profile_path = resolve_profile_path(repo_root, args.profile, args.profile_dir)
    profile = load_profile(profile_path)
    builder = ACTION_BUILDERS[args.action]
    cmd, env_updates, log_path = builder(repo_root, profile)
    env_updates["__profile_path__"] = str(profile_path)
    env_updates["__action__"] = args.action
    return_code = stream_command(
        cmd=cmd,
        cwd=repo_root,
        env_updates=env_updates,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
