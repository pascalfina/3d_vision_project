#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--replacement-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--obj-id", required=True, type=int)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def ensure_symlink(src: Path, dst: Path) -> None:
    safe_remove(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def write_split_files(files_dir: Path, split: str, scene_id: str) -> None:
    for name in ["train", "val", "test"]:
        payload = f"{scene_id}\n" if name == split else ""
        (files_dir / f"{name}_resplit_scans.txt").write_text(payload)
        (files_dir / f"{name}_scans.txt").write_text(payload)


def main():
    args = parse_args()
    baseline_root = Path(args.baseline_root)
    replacement_root = Path(args.replacement_root)
    target_root = Path(args.target_root)
    scene_id = args.scene_id
    obj_id = str(args.obj_id)

    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target root already exists: {target_root}")
        shutil.rmtree(target_root)

    files_dir = target_root / "files"
    scenes_dir = target_root / "scenes"
    files_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # Full scene and scene-level metadata stay from baseline.
    ensure_symlink(baseline_root / "scenes" / scene_id, scenes_dir / scene_id)
    for name in [
        "3RScan.json",
        "objects.json",
        "scannet40_classes.txt",
        "orig",
        "Features3D",
        "gt_projection",
        "pred_projection",
    ]:
        src = baseline_root / "files" / name
        if src.exists():
            ensure_symlink(src, files_dir / name)

    # Scene-level gs_annotations: baseline for every object, except one override.
    target_scene_gs = files_dir / "gs_annotations" / scene_id
    target_scene_gs.mkdir(parents=True, exist_ok=True)
    baseline_scene_gs = baseline_root / "files" / "gs_annotations" / scene_id
    replacement_obj_dir = (
        replacement_root / "files" / "gs_annotations" / scene_id / obj_id
    )
    if not replacement_obj_dir.exists():
        raise FileNotFoundError(f"Missing replacement object dir: {replacement_obj_dir}")

    for obj_dir in sorted(baseline_scene_gs.iterdir(), key=lambda p: p.name):
        if not obj_dir.is_dir():
            continue
        ensure_symlink(obj_dir, target_scene_gs / obj_dir.name)
    ensure_symlink(replacement_obj_dir, target_scene_gs / obj_id)

    (files_dir / "gs_embeddings").mkdir(parents=True, exist_ok=True)
    write_split_files(files_dir, args.split, scene_id)

    manifest = {
        "baseline_root": str(baseline_root),
        "replacement_root": str(replacement_root),
        "target_root": str(target_root),
        "scene_id": scene_id,
        "obj_id": int(args.obj_id),
        "split": args.split,
    }
    (files_dir / "mixed_scene_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[built] {target_root}")
    print(f"[scene] {scene_id}")
    print(f"[replaced obj] {obj_id}")
    print(f"[baseline gs scene] {baseline_scene_gs}")
    print(f"[replacement obj dir] {replacement_obj_dir}")


if __name__ == "__main__":
    main()
