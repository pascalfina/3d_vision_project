#!/usr/bin/env python3
"""Generate SAM2/Pi3X workflow profiles for all short baseline sequences."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create scene_<slug>_sam2_pi3x.json profiles for every baseline "
            "scene with sequence.zip and fewer than --max-frames RGB frames."
        )
    )
    parser.add_argument(
        "--baseline-scenes-root",
        type=Path,
        default=Path("/work/scratch/pafina/objectx-data-baseline/scenes"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("configs/workflows/scene_profiles"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/workflows/scene_profiles/scene_5341b7e3_sam2_pi3x.json"),
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/workflows/scene_profiles/pi3x_under300_profiles.txt"),
    )
    parser.add_argument(
        "--scene-table",
        type=Path,
        default=Path("configs/workflows/scene_profiles/pi3x_under300_scenes.tsv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_color_frames(sequence_zip: Path) -> int:
    with ZipFile(sequence_zip) as zf:
        return sum(
            1
            for name in zf.namelist()
            if Path(name).name.startswith("frame-") and Path(name).name.endswith(".color.jpg")
        )


def discover_scenes(root: Path, max_frames: int) -> list[tuple[str, int]]:
    scenes: list[tuple[str, int]] = []
    for sequence_zip in sorted(root.glob("*/sequence.zip")):
        scene_id = sequence_zip.parent.name
        frame_count = count_color_frames(sequence_zip)
        if frame_count >= max_frames:
            continue
        mesh = sequence_zip.parent / "mesh.refined.v2.obj"
        if not mesh.exists():
            continue
        scenes.append((scene_id, frame_count))
    return scenes


def scene_slug(scene_id: str) -> str:
    return scene_id.split("-", 1)[0]


def set_if_present(mapping: dict, path: tuple[str, ...], value) -> None:
    cursor = mapping
    for key in path[:-1]:
        cursor = cursor.get(key, {})
        if not isinstance(cursor, dict):
            return
    if path[-1] in cursor:
        cursor[path[-1]] = value


def profile_for_scene(template: dict, scene_id: str, frame_count: int) -> tuple[str, dict]:
    slug = scene_slug(scene_id)
    profile_name = f"scene_{slug}_sam2_pi3x"
    recon_root = f"/work/scratch/pafina/objectx-data-fullscene-{slug}-pi3x"
    must3r_root = f"/work/scratch/pafina/objectx-data-fullscene-{slug}-hybrid-sam2mask-must3r"
    pred_ready_root = f"/work/scratch/pafina/objectx-data-fullscene-{slug}-predready-sam2-pi3x-v1"

    profile = deepcopy(template)
    profile["name"] = profile_name
    profile["scene_id"] = scene_id

    set_if_present(profile, ("roots", "reconstruction"), recon_root)
    set_if_present(profile, ("roots", "pred_ready"), pred_ready_root)
    set_if_present(profile, ("artifacts", "manifest"), f"${{OBJECTX_REPO_DEBUG_ROOT}}/{slug}_fullscene_pi3x_manifest.json")
    set_if_present(profile, ("artifacts", "joint_ply"), f"${{OBJECTX_REPO_VIS_ROOT}}/{scene_id}_joint_pi3x.ply")

    set_if_present(profile, ("segment_inputs", "output_root"), recon_root)
    set_if_present(profile, ("segment_inputs", "log"), f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/debug_{slug}_pi3x_segment_inputs.log")

    set_if_present(profile, ("must3r", "output_root"), must3r_root)
    set_if_present(profile, ("must3r", "log"), f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/debug_{slug}_pi3x_must3r.log")

    set_if_present(profile, ("pi3x", "output_root"), recon_root)
    set_if_present(profile, ("pi3x", "log"), f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/debug_{slug}_pi3x.log")
    set_if_present(
        profile,
        ("pi3x", "env", "OBJECTX_PI3X_EXTERNAL_POSE_DIR"),
        f"{must3r_root}/scenes_sam2_must3r/{scene_id}/sequence",
    )

    set_if_present(profile, ("voxelise", "log"), f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/debug_{slug}_2_5_pi3x.log")
    set_if_present(profile, ("validate_pred_ready", "out_dir"), f"${{OBJECTX_REPO_VIS_ROOT}}/pred_ready_validation/{profile_name}")
    set_if_present(profile, ("compare_arrangement", "out_dir"), f"${{OBJECTX_REPO_VIS_ROOT}}/arrangement_compare/{profile_name}_vs_gt")

    set_if_present(profile, ("render_bundle", "label"), profile_name)
    set_if_present(profile, ("render_bundle", "data_root"), recon_root)
    set_if_present(profile, ("render_bundle", "max_views"), frame_count)

    set_if_present(profile, ("samobject", "source_sequence_dir"), f"{recon_root}/scenes_sam2_pi3x/{scene_id}/sequence")
    set_if_present(profile, ("samobject", "log"), f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/samobject_pipeline_{slug}.log")

    set_if_present(profile, ("geometry_eval", "method_name"), profile_name)
    set_if_present(profile, ("geometry_eval", "pred_input_mode"), "sequence")
    set_if_present(profile, ("geometry_eval", "scenes_dirname"), "scenes_sam2_pi3x")
    set_if_present(profile, ("geometry_eval", "write_debug_html"), True)

    return profile_name, profile


def main() -> None:
    args = parse_args()
    template = json.loads(args.template.read_text())
    scenes = discover_scenes(args.baseline_scenes_root, args.max_frames)

    slugs = [scene_slug(scene_id) for scene_id, _ in scenes]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("Scene slug collision detected; refusing to overwrite profiles.")

    generated: list[tuple[str, str, int, Path]] = []
    for scene_id, frame_count in scenes:
        profile_name, profile = profile_for_scene(template, scene_id, frame_count)
        path = args.profile_dir / f"{profile_name}.json"
        generated.append((profile_name, scene_id, frame_count, path))
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(profile, indent=2) + "\n")

    if not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text("\n".join(name for name, _, _, _ in generated) + "\n")
        args.scene_table.parent.mkdir(parents=True, exist_ok=True)
        args.scene_table.write_text(
            "profile\tscene_id\tframes\tpath\n"
            + "\n".join(
                f"{name}\t{scene_id}\t{frames}\t{path}"
                for name, scene_id, frames, path in generated
            )
            + "\n"
        )

    action = "would write" if args.dry_run else "wrote"
    print(f"[pi3x-profiles] {action} {len(generated)} profiles")
    print(f"[pi3x-profiles] manifest={args.manifest}")
    print(f"[pi3x-profiles] scene_table={args.scene_table}")


if __name__ == "__main__":
    main()
