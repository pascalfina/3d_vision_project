#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_ACTIONS = [
    "features3d",
    "must3r",
    "mast3r-sfm",
    "pi3x",
    "fuse",
    "segment-inputs",
    "voxelise",
    "build-pred-ready",
    "validate-pred-ready",
    "compare-arrangement",
    "slat",
    "u3dgs",
    "geom-debug",
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
        choices=list(ACTION_BUILDERS.keys()),
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


def _strip_shell_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_local_paths_env(repo_root: Path) -> None:
    """Load optional per-user path overrides before expanding profile JSON.

    The file intentionally supports only simple ``KEY=value`` / ``export KEY=value``
    lines. That keeps it safe to parse from Python and still easy for teammates to
    edit without touching every scene profile.
    """

    path = repo_root / "configs" / "workflows" / "local_paths.env"
    if not path.exists():
        return
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid line in {path}:{lineno}: expected KEY=value")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty key in {path}:{lineno}")
        value = _strip_shell_quotes(value)
        os.environ[key] = os.path.expanduser(os.path.expandvars(value))


def set_base_path_defaults(repo_root: Path) -> None:
    user = os.environ.get("USER", "pafina")
    os.environ.setdefault("OBJECTX_REPO_ROOT", str(repo_root))
    os.environ.setdefault("OBJECTX_USER_ROOT", f"/work/scratch/{user}")
    os.environ.setdefault("OBJECTX_TEAM_ROOT", f"/work/courses/3dv/team35/{user}")


def set_derived_path_defaults() -> None:
    user_root = os.environ["OBJECTX_USER_ROOT"]
    team_root = os.environ["OBJECTX_TEAM_ROOT"]
    os.environ.setdefault("OBJECTX_BASELINE_ROOT", f"{user_root}/objectx-data-baseline")
    os.environ.setdefault("OBJECTX_REPO_DEBUG_ROOT", f"{os.environ['OBJECTX_REPO_ROOT']}/debug")
    os.environ.setdefault("OBJECTX_REPO_VIS_ROOT", f"{os.environ['OBJECTX_REPO_ROOT']}/vis")
    os.environ.setdefault("OBJECTX_WORKFLOW_LOG_ROOT", f"{team_root}/logs")

    # Scene/profile roots. Override these in configs/workflows/local_paths.env
    # when running on another account or storage layout.
    os.environ.setdefault(
        "OBJECTX_CABINET_SAM2_RECON_ROOT",
        f"{user_root}/objectx-data-fullscene-cabinet-hybrid-sam2mask",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_SAM2_PREDREADY_ROOT",
        f"{user_root}/objectx-data-fullscene-cabinet-predready-sam2-v1",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_RECON_ROOT",
        f"{team_root}/objectx-data-fullscene-cabinet-hybrid-gtmask",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_MUST3R_PREDREADY_ROOT",
        f"{user_root}/objectx-data-fullscene-cabinet-predready-sam2-must3r-v1",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_PREDREADY_FLOORFIX_ROOT",
        f"{user_root}/objectx-data-fullscene-cabinet-predready-v2-floorfix",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_MAST3R_SFM_RECON_ROOT",
        f"{user_root}/objectx-data-fullscene-cabinet-hybrid-sam2mask-mast3r-sfm",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_MAST3R_SFM_PREDREADY_ROOT",
        f"{user_root}/objectx-data-fullscene-cabinet-predready-sam2-mast3r-sfm-v1",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_MAST3R_SFM_MUST3R_PREDREADY_ROOT",
        f"{team_root}/objectx-data/objectx-data-fullscene-cabinet-predready-sam2-mast3r-sfm-must3r-v1",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_LIFTED_RECON_ROOT",
        f"{user_root}/objectx-data-fullscene-cabinet-lifted-gtmask",
    )
    os.environ.setdefault(
        "OBJECTX_CABINET_LIFTED_PREDREADY_ROOT",
        f"{user_root}/objectx-data-fullscene-cabinet-predready-v2-liftedonly",
    )
    os.environ.setdefault(
        "OBJECTX_OVEN_RECON_ROOT",
        f"{user_root}/objectx-data-fullscene-oven-hybrid-gtmask",
    )
    os.environ.setdefault(
        "OBJECTX_OVEN_PREDREADY_ROOT",
        f"{user_root}/objectx-data-fullscene-oven-predready-v1",
    )
    os.environ.setdefault(
        "OBJECTX_OVEN_SAM2_MUST3R_RECON_ROOT",
        f"{user_root}/objectx-data-fullscene-oven-hybrid-sam2mask-must3r",
    )
    os.environ.setdefault(
        "OBJECTX_OVEN_SAM2_MUST3R_PREDREADY_ROOT",
        f"{user_root}/objectx-data-fullscene-oven-predready-sam2-must3r-v1",
    )
    os.environ.setdefault(
        "OBJECTX_MAST3R_SFM_CACHE_DIR",
        f"{team_root}/objectx-cache/mast3r_sfm_cabinet",
    )


def expand_profile_values(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_profile_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_profile_values(item) for key, item in value.items()}
    return value


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
    return expand_profile_values(json.loads(path.read_text()))


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
            value = env_updates[key]
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
    variant = input_variant_settings(repo_root, profile)
    env_updates = {
        "DATA_ROOT_DIR": sec.get("data_root", roots(profile).get("reconstruction")),
        "OBJECTX_BASELINE_ROOT": sec.get("baseline_root", roots(profile).get("baseline")),
        "OBJECTX_SCENE_ID": sec.get("scene_id", shared_value(profile, "scene_id")),
        "SPLIT": sec.get("split", shared_value(profile, "split", "val")),
        "OBJECTX_MASK_SOURCE": sec.get(
            "mask_source", profile_mask_source(repo_root, profile)
        ),
        "OBJECTX_VOXEL_OBJECTS_FILENAME": sec.get(
            "objects_filename", variant.get("objects_filename", "objects.json")
        ),
        "RESET_TMP": str(sec.get("reset_tmp", 1)),
        "MAX_SCANS": str(sec.get("max_scans", 0)),
    }
    scene_source_dirname = sec.get("scene_source_dirname")
    if scene_source_dirname:
        env_updates["OBJECTX_SCENE_SOURCE_DIRNAME"] = str(scene_source_dirname)
    if sec.get("override", False):
        env_updates["OBJECTX_VOXEL_OVERRIDE"] = "1"
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


def build_must3r_action(repo_root: Path, profile: dict):
    sec = section(profile, "must3r")
    variant = input_variant_settings(repo_root, profile)
    if not sec and not variant.get("run_must3r", False):
        raise ValueError(
            f"Profile '{profile.get('name', 'unknown')}' uses variant '{input_variant_name(profile)}' "
            "which does not define a MUSt3R stage."
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
        "OBJECTX_SEG_RUN_MUST3R": str(int(sec.get("run_must3r", True))),
        "OBJECTX_SEG_RUN_SAM2": str(int(sec.get("run_sam2", False))),
        "OBJECTX_SEG_RUN_REGISTRY": str(int(sec.get("run_registry", False))),
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


def build_mast3r_sfm_action(repo_root: Path, profile: dict):
    """Runs the segmentation pipeline's pose+depth step via the MASt3R-SfM backend.

    Mirrors build_must3r_action but reads the ``mast3r_sfm`` profile section
    and hard-forces ``OBJECTX_POSE_DEPTH_BACKEND=mast3r_sfm`` so a single
    profile can orchestrate a MUSt3R run AND a MASt3R-SfM run with distinct
    output scenes_dirnames.
    """
    sec = section(profile, "mast3r_sfm")
    variant = input_variant_settings(repo_root, profile)
    if not sec:
        raise ValueError(
            f"Profile '{profile.get('name', 'unknown')}' has no 'mast3r_sfm' section."
        )

    env_updates = {
        "OBJECTX_SEG_INPUT_ROOT": sec.get("input_root", roots(profile).get("baseline")),
        "OBJECTX_SEG_OUTPUT_ROOT": sec.get(
            "output_root", roots(profile).get("reconstruction")
        ),
        "OBJECTX_SEG_MASK_DIRNAME": sec.get(
            "mask_dirname", variant.get("mask_output_dirname", "gt_projection_predicted")
        ),
        "OBJECTX_SEG_OBJECTS_FILENAME": sec.get(
            "objects_filename", variant.get("objects_filename", "objects_predicted.json")
        ),
        "OBJECTX_SEG_SCENES_DIRNAME": sec.get(
            "scenes_dirname", "scenes_sam2_mast3r_sfm"
        ),
        "OBJECTX_SEG_RUN_MUST3R": str(int(sec.get("run_must3r", True))),
        "OBJECTX_SEG_RUN_SAM2": str(int(sec.get("run_sam2", False))),
        "OBJECTX_SEG_RUN_REGISTRY": str(int(sec.get("run_registry", False))),
        "OBJECTX_POSE_DEPTH_BACKEND": "mast3r_sfm",
        "MUST3R_PATH": sec.get("must3r_path", str(repo_root / "dependencies" / "must3r")),
    }
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
    # Ensure the backend routing cannot be silently overridden by an inherited
    # env var from a prior must3r step.
    env_updates["OBJECTX_POSE_DEPTH_BACKEND"] = "mast3r_sfm"

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


def build_pi3x_action(repo_root: Path, profile: dict):
    """Runs the segmentation pipeline's pose+depth step via the Pi3X backend.

    Mirrors build_must3r_action / build_mast3r_sfm_action but reads the
    ``pi3x`` profile section and hard-forces ``OBJECTX_POSE_DEPTH_BACKEND=pi3x``.
    """
    sec = section(profile, "pi3x")
    variant = input_variant_settings(repo_root, profile)
    if not sec:
        raise ValueError(
            f"Profile '{profile.get('name', 'unknown')}' has no 'pi3x' section."
        )

    env_updates = {
        "OBJECTX_SEG_INPUT_ROOT": sec.get("input_root", roots(profile).get("baseline")),
        "OBJECTX_SEG_OUTPUT_ROOT": sec.get(
            "output_root", roots(profile).get("reconstruction")
        ),
        "OBJECTX_SEG_MASK_DIRNAME": sec.get(
            "mask_dirname", variant.get("mask_output_dirname", "gt_projection_predicted")
        ),
        "OBJECTX_SEG_OBJECTS_FILENAME": sec.get(
            "objects_filename", variant.get("objects_filename", "objects_predicted.json")
        ),
        "OBJECTX_SEG_SCENES_DIRNAME": sec.get(
            "scenes_dirname", "scenes_sam2_pi3x"
        ),
        "OBJECTX_SEG_RUN_MUST3R": str(int(sec.get("run_must3r", True))),
        "OBJECTX_SEG_RUN_SAM2": str(int(sec.get("run_sam2", False))),
        "OBJECTX_SEG_RUN_REGISTRY": str(int(sec.get("run_registry", False))),
        "OBJECTX_POSE_DEPTH_BACKEND": "pi3x",
        "MUST3R_PATH": sec.get("must3r_path", str(repo_root / "dependencies" / "must3r")),
        "PI3_PATH": sec.get("pi3_path", str(repo_root / "dependencies" / "pi3")),
    }
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
    env_updates["OBJECTX_POSE_DEPTH_BACKEND"] = "pi3x"

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


def build_fuse_action(repo_root: Path, profile: dict):
    """Fuse MASt3R-SfM poses with MUSt3R depth/xyz into a hybrid scene dir."""
    sec = section(profile, "fuse")
    reconstruction_root = sec.get(
        "reconstruction_root", roots(profile).get("reconstruction")
    )
    if not reconstruction_root:
        raise ValueError("fuse action requires 'reconstruction_root' or roots.reconstruction")
    cmd = [
        sys.executable,
        "-u",
        str(
            repo_root
            / "preprocessing"
            / "segmentation"
            / "fuse_must3r_pose_mast3r_sfm_depth.py"
        ),
        "--reconstruction-root",
        str(reconstruction_root),
        "--scene-id",
        sec.get("scene_id", shared_value(profile, "scene_id")),
        "--must3r-scenes-dirname",
        sec.get("must3r_scenes_dirname", "scenes_sam2_must3r"),
        "--mast3r-sfm-scenes-dirname",
        sec.get("mast3r_sfm_scenes_dirname", "scenes_sam2_mast3r_sfm"),
        "--output-scenes-dirname",
        sec.get("output_scenes_dirname", "scenes_sam2_mast3r_sfm_must3r"),
    ]
    baseline_root = sec.get("baseline_root", roots(profile).get("baseline"))
    if baseline_root:
        cmd += ["--baseline-root", str(baseline_root)]
    add_cli_arg(cmd, "--min-conf-thr", sec.get("min_conf_thr"))
    add_cli_arg(cmd, "--min-alignment-frames", sec.get("min_alignment_frames"))
    env_updates = {}
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
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
    add_cli_arg(cmd, "--max-scale", sec.get("max_scale"))
    add_cli_arg(cmd, "--max-extent", sec.get("max_extent"))
    add_cli_arg(cmd, "--max-center-norm", sec.get("max_center_norm"))
    add_cli_arg(cmd, "--allow-stale-geometry", sec.get("allow_stale_geometry", False))
    point_counts = sec.get("point_counts", [64, 128, 256, 512])
    if point_counts:
        cmd.append("--point-counts")
        cmd.extend(str(x) for x in point_counts)
    add_cli_arg(cmd, "--overwrite", sec.get("overwrite", False))
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, {}, log_path


def build_features3d_action(repo_root: Path, profile: dict):
    sec = section(profile, "features3d")
    voxel_sec = section(profile, "voxelise")
    default_scene_source_dirname = voxel_sec.get("scene_source_dirname", "scenes")
    env_updates = {
        "DATA_ROOT_DIR": sec.get("data_root", roots(profile).get("reconstruction")),
        "SPLIT": sec.get("split", shared_value(profile, "split", "val")),
        "SCENE_ID": sec.get("scene_id", shared_value(profile, "scene_id")),
        "MAX_SCANS": str(sec.get("max_scans", 0)),
        "RESET_TMP": str(sec.get("reset_tmp", 0)),
        "OBJECTX_SCENE_SOURCE_DIRNAME": sec.get(
            "scene_source_dirname", default_scene_source_dirname
        ),
        "OBJECTX_MASK_SOURCE": sec.get(
            "mask_source", profile_mask_source(repo_root, profile)
        ),
    }
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
    cmd = [
        "bash",
        str(repo_root / "scripts" / "features3D" / "obj_visual_embeddings_tmp.sh"),
    ]
    log_value = sec.get(
        "log",
        f"{os.environ['OBJECTX_REPO_DEBUG_ROOT']}/debug_{profile.get('name', 'scene')}_features3d.log",
    )
    log_path = Path(log_value).expanduser() if log_value else None
    return cmd, env_updates, log_path


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


def build_geom_debug_action(repo_root: Path, profile: dict):
    sec = section(profile, "geom_debug")
    render_sec = section(profile, "render_bundle")
    voxel_sec = section(profile, "voxelise")
    pred_ready_sec = section(profile, "build_pred_ready")

    data_root = sec.get("data_root", render_sec.get("data_root", roots(profile).get("reconstruction")))
    if not data_root:
        raise ValueError(
            "geom-debug action requires a reconstruction root via geom_debug.data_root, "
            "render_bundle.data_root, or roots.reconstruction."
        )

    scan_id = sec.get("scan_id", sec.get("scene_id", shared_value(profile, "scene_id")))
    if not scan_id:
        raise ValueError(
            "geom-debug action requires a scan/scene id via geom_debug.scan_id, "
            "geom_debug.scene_id, or profile.scene_id."
        )

    label = sec.get("label", f"{profile.get('name', 'scene')}_geom_debug")
    out_dir = sec.get(
        "out_dir",
        f"{os.environ['OBJECTX_REPO_VIS_ROOT']}/interactive_reconstruction_geometry/{label}",
    )

    render_env = {key: str(value) for key, value in render_sec.get("env", {}).items()}
    voxel_env = voxel_sec.get("env", {})
    env_updates = dict(render_env)

    scenes_dirname = sec.get(
        "scenes_dirname",
        render_env.get("OBJECTX_VIS_SCENES_DIRNAME")
        or voxel_sec.get("scene_source_dirname")
        or pred_ready_sec.get("reconstruction_scenes_dirname"),
    )
    if scenes_dirname:
        env_updates["OBJECTX_VIS_SCENES_DIRNAME"] = str(scenes_dirname)

    depth_source = sec.get(
        "depth_source",
        render_env.get("OBJECTX_VIS_DEPTH_SOURCE")
        or voxel_env.get("OBJECTX_VOXEL_DEPTH_SOURCE"),
    )
    if depth_source:
        env_updates["OBJECTX_VIS_DEPTH_SOURCE"] = str(depth_source)

    raw_depth_conf_thr = sec.get(
        "raw_depth_conf_thr",
        render_env.get("OBJECTX_VIS_RAW_DEPTH_CONF_THR")
        or voxel_env.get("OBJECTX_VOXEL_RAW_DEPTH_CONF_THR"),
    )
    if raw_depth_conf_thr is not None:
        env_updates["OBJECTX_VIS_RAW_DEPTH_CONF_THR"] = str(raw_depth_conf_thr)

    raw_depth_max = sec.get(
        "raw_depth_max",
        render_env.get("OBJECTX_VIS_RAW_DEPTH_MAX"),
    )
    if raw_depth_max is not None:
        env_updates["OBJECTX_VIS_RAW_DEPTH_MAX"] = str(raw_depth_max)

    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
    for key in sec.get("unset_env", []):
        env_updates[key] = None

    remove_obj_ids = sec.get("remove_obj_ids", [])
    mask_source = sec.get("mask_source")
    if mask_source is None:
        mask_source = profile_mask_source(repo_root, profile) if remove_obj_ids else "none"

    cmd = [
        sys.executable,
        str(
            repo_root
            / "scripts"
            / "segmentation"
            / "visualization"
            / "export_reconstruction_geometry_interactive.py"
        ),
    ]
    add_cli_arg(cmd, "--data-root", data_root)
    add_cli_arg(cmd, "--scan-id", scan_id)
    if remove_obj_ids:
        add_cli_arg(
            cmd,
            "--mask-root",
            sec.get("mask_root", roots(profile).get("reconstruction")),
        )
    add_cli_arg(cmd, "--mask-source", mask_source)
    add_cli_arg(cmd, "--remove-obj-id", remove_obj_ids)
    add_cli_arg(cmd, "--pose-mode", sec.get("pose_mode", render_sec.get("pose_mode", "raw")))
    add_cli_arg(
        cmd,
        "--lift-coord-system",
        sec.get("lift_coord_system", render_sec.get("lift_coord_system", "pinhole")),
    )
    add_cli_arg(
        cmd,
        "--frame-selection",
        sec.get("frame_selection", render_sec.get("frame_selection", "diverse_area")),
    )
    add_cli_arg(cmd, "--max-views", sec.get("max_views", render_sec.get("max_views", 96)))
    add_cli_arg(cmd, "--mask-erode-px", sec.get("mask_erode_px", 2))
    add_cli_arg(
        cmd,
        "--max-points",
        sec.get("max_points", render_sec.get("bg_max_points", 250000)),
    )
    add_cli_arg(cmd, "--label", label)
    add_cli_arg(cmd, "--out-dir", out_dir)
    add_cli_arg(cmd, "--skip-cameras", sec.get("skip_cameras", False))
    log_value = sec.get(
        "log",
        f"{os.environ['OBJECTX_WORKFLOW_LOG_ROOT']}/debug_{profile.get('name', 'scene')}_geom_debug.log",
    )
    log_path = Path(log_value).expanduser() if log_value else None
    return cmd, env_updates, log_path


def build_render_action(repo_root: Path, profile: dict):
    sec = section(profile, "render_bundle")
    mask_source = sec.get("mask_source", profile_mask_source(repo_root, profile))
    env_updates = {}
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)
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
    return cmd, env_updates, log_path


def build_plot_voxelised_action(repo_root: Path, profile: dict):
    sec = section(profile, "plot_voxelised")
    render_sec = section(profile, "render_bundle")

    scan_id = sec.get("scan_id", sec.get("scene_id", shared_value(profile, "scene_id")))
    data_root = sec.get("data_root", render_sec.get("data_root", roots(profile).get("reconstruction")))
    pred_ready_root = sec.get("replacement_root", roots(profile).get("pred_ready"))

    label = sec.get("label", f"{profile.get('name', 'scene')}_voxelised")
    out_dir = sec.get(
        "out_dir",
        f"{os.environ['OBJECTX_REPO_VIS_ROOT']}/interactive_depth_views/{label}",
    )

    render_env = {key: str(value) for key, value in render_sec.get("env", {}).items()}
    env_updates = dict(render_env)
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)

    # Load obj_ids from pred-ready objects.json; fall back to manifest
    obj_ids = sec.get("obj_ids", [])
    if not obj_ids and pred_ready_root:
        objects_json = Path(pred_ready_root) / "files" / "objects.json"
        if objects_json.exists():
            d = json.loads(objects_json.read_text())
            scans = d.get("scans", [])
            scan = scans[0] if isinstance(scans, list) else next(iter(scans.values()), {})
            obj_ids = [o["id"] for o in scan.get("objects", [])]

    manifest = sec.get("manifest", artifacts(profile).get("manifest")) if not obj_ids else None

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "segmentation" / "visualization" / "export_depth_background_interactive.py"),
    ]
    add_cli_arg(cmd, "--data-root", data_root)
    add_cli_arg(cmd, "--mask-root", sec.get("mask_root", data_root))
    add_cli_arg(cmd, "--replacement-root", pred_ready_root)
    add_cli_arg(cmd, "--scan-id", scan_id)
    add_cli_arg(cmd, "--obj-id", obj_ids)
    add_cli_arg(cmd, "--manifest", manifest)
    add_cli_arg(cmd, "--mask-source", sec.get("mask_source", render_sec.get("mask_source", "sam2_projection")))
    add_cli_arg(cmd, "--background-remove-mode", sec.get("background_remove_mode", render_sec.get("background_remove_mode", "loaded")))
    add_cli_arg(cmd, "--pose-mode", sec.get("pose_mode", render_sec.get("pose_mode", "raw")))
    add_cli_arg(cmd, "--lift-coord-system", sec.get("lift_coord_system", render_sec.get("lift_coord_system", "pinhole")))
    add_cli_arg(cmd, "--frame-selection", sec.get("frame_selection", render_sec.get("frame_selection", "diverse_area")))
    add_cli_arg(cmd, "--max-views", sec.get("max_views", render_sec.get("max_views", 277)))
    add_cli_arg(cmd, "--max-bg-points", sec.get("max_bg_points", render_sec.get("bg_max_points", 3000000)))
    add_cli_arg(cmd, "--max-obj-points", sec.get("max_obj_points", render_sec.get("joint_max_points", 3000000)))
    add_cli_arg(cmd, "--label", label)
    add_cli_arg(cmd, "--out-dir", out_dir)
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, env_updates, log_path


def build_samobject_action(repo_root: Path, profile: dict):
    sec = section(profile, "samobject")
    scan_id = sec.get("scan_id", sec.get("scene_id", shared_value(profile, "scene_id")))
    root_dir = sec.get("root_dir", roots(profile).get("reconstruction"))
    baseline_root = sec.get("baseline_root", roots(profile).get("baseline"))
    mesh_path = sec.get(
        "mesh_path",
        f"{baseline_root}/scenes/{scan_id}/mesh.refined.v2.obj" if baseline_root and scan_id else None,
    )

    images_dir = sec.get("images_dir", f"{root_dir}/color_images_cluster" if root_dir else None)
    source_sequence_dir = sec.get("source_sequence_dir")

    env_updates = {
        "SAMOBJECT_DIR": sec.get(
            "samobject_dir", str(repo_root / "dependencies" / "SAM2Object")
        ),
        "SAMOBJECT_VENV": sec.get("samobject_venv"),
        "SAMOBJECT_DATA_ROOT": root_dir,
        "SAMOBJECT_SCAN_ID": scan_id,
        "SAMOBJECT_MESH_PATH": mesh_path,
        "SAMOBJECT_IMAGES_DIR": images_dir,
        "SAMOBJECT_SOURCE_SEQUENCE_DIR": source_sequence_dir,
        "OBJECTX_REPO_ROOT": str(repo_root),
        "SAMOBJECT_PROJECTION_DILATION": str(sec.get("projection_dilation", 2)),
    }
    if sec.get("frame_skip") is not None:
        env_updates["SAMOBJECT_FRAME_SKIP"] = str(sec["frame_skip"])
    for key, value in sec.get("env", {}).items():
        env_updates[key] = str(value)

    cmd = ["bash", str(repo_root / "scripts" / "segmentation" / "run_samobject_pipeline.sh")]
    log_path = Path(sec["log"]).expanduser() if sec.get("log") else None
    return cmd, env_updates, log_path


ACTION_BUILDERS = {
    "features3d": build_features3d_action,
    "must3r": build_must3r_action,
    "mast3r-sfm": build_mast3r_sfm_action,
    "pi3x": build_pi3x_action,
    "fuse": build_fuse_action,
    "segment-inputs": build_segment_inputs_action,
    "voxelise": build_voxelise_action,
    "build-pred-ready": build_pred_ready_action,
    "validate-pred-ready": build_validate_action,
    "compare-arrangement": build_compare_action,
    "slat": build_slat_action,
    "u3dgs": build_u3dgs_action,
    "geom-debug": build_geom_debug_action,
    "render": build_render_action,
    "plot-voxelised": build_plot_voxelised_action,
    "samobject": build_samobject_action,
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    set_base_path_defaults(repo_root)
    load_local_paths_env(repo_root)
    set_derived_path_defaults()

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
