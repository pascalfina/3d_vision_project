#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--tmp-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--max-scans", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--densify-until-iter", type=int, default=15000)
    parser.add_argument("--override", action="store_true")
    return parser.parse_args()


def load_scan_ids(root: Path, split: str) -> list[str]:
    path = root / "files" / f"{split}_resplit_scans.txt"
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def all_outputs_exist(scratch_root: Path, obj_data: dict, iterations: int) -> bool:
    scan_id = obj_data["scan"]
    for obj in obj_data["objects"]:
        obj_id = str(obj["id"])
        ply_path = (
            scratch_root
            / "files"
            / "gs_annotations"
            / scan_id
            / obj_id
            / "point_cloud"
            / f"iteration_{iterations}"
            / "point_cloud.ply"
        )
        if not ply_path.exists():
            return False
    return True


def stage_scan(src_scan_dir: Path, tmp_scan_dir: Path):
    if tmp_scan_dir.exists():
        shutil.rmtree(tmp_scan_dir)
    tmp_scan_dir.mkdir(parents=True, exist_ok=True)

    for item in src_scan_dir.iterdir():
        if item.name == "sequence.zip":
            continue
        target = tmp_scan_dir / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(item, target)

    (tmp_scan_dir / "sequence").mkdir(exist_ok=True)
    subprocess.run(
        ["unzip", "-qo", str(src_scan_dir / "sequence.zip"), "-d", str(tmp_scan_dir / "sequence")],
        check=True,
    )


def cleanup_scan(tmp_root: Path, scan_id: str):
    tmp_scan_dir = tmp_root / "scenes" / scan_id
    if tmp_scan_dir.exists():
        shutil.rmtree(tmp_scan_dir)

    tmp_gs_scan_dir = tmp_root / "files" / "gs_annotations" / scan_id
    if tmp_gs_scan_dir.exists():
        shutil.rmtree(tmp_gs_scan_dir)


def prepare_tmp_files(tmp_root: Path, scratch_root: Path, split: str, scan_id: str):
    files_dir = tmp_root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "gs_annotations").mkdir(parents=True, exist_ok=True)

    for name in ["3RScan.json", "objects.json"]:
        src = scratch_root / "files" / name
        dst = files_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    split_map = {
        "train_resplit_scans.txt": [],
        "val_resplit_scans.txt": [],
        "test_resplit_scans.txt": [],
    }
    split_map[f"{split}_resplit_scans.txt"] = [scan_id]
    for name, scan_ids in split_map.items():
        (files_dir / name).write_text("\n".join(scan_ids) + ("\n" if scan_ids else ""))


def run_cmd(cmd: list[str], env: dict[str, str]):
    subprocess.run(cmd, check=True, cwd=env["REPO_ROOT"], env=env)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    scratch_root = Path(args.scratch_root)
    tmp_root = Path(args.tmp_root)

    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / "scenes").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files" / "gs_annotations").mkdir(parents=True, exist_ok=True)

    all_obj_info = json.load(open(scratch_root / "files" / "objects.json"))["scans"]
    obj_lookup = {entry["scan"]: entry for entry in all_obj_info}
    scan_ids = load_scan_ids(scratch_root, args.split)
    if args.max_scans > 0:
        scan_ids = scan_ids[: args.max_scans]

    total = len(scan_ids)
    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo_root)
    env["DATA_ROOT_DIR"] = str(tmp_root)

    for idx, scan_id in enumerate(scan_ids, start=1):
        obj_data = obj_lookup[scan_id]
        if all_outputs_exist(scratch_root, obj_data, args.iterations) and not args.override:
            if idx % 10 == 0 or idx == total:
                print(f"[2.6] {idx}/{total} (skip existing) {scan_id}", flush=True)
            continue

        src_scan_dir = scratch_root / "scenes" / scan_id
        tmp_scan_dir = tmp_root / "scenes" / scan_id

        print(f"[2.6-map] {idx}/{total} {args.split} {scan_id}", flush=True)
        prepare_tmp_files(tmp_root, scratch_root, args.split, scan_id)
        stage_scan(src_scan_dir, tmp_scan_dir)
        try:
            run_cmd(
                [
                    sys.executable,
                    "preprocessing/gs_anno/map_to_colmap.py",
                    "--config",
                    args.config,
                    "--split",
                    args.split,
                    "--num_workers",
                    "1",
                    "--output_dir",
                    str(tmp_root / "files" / "gs_annotations"),
                ],
                env,
            )

            print(f"[2.6-gs] {idx}/{total} {args.split} {scan_id}", flush=True)
            run_cmd(
                [
                    sys.executable,
                    "preprocessing/gs_anno/annotate_gaussians.py",
                    "--densify_until_iter",
                    str(args.densify_until_iter),
                    "--iterations",
                    str(args.iterations),
                    "--save_iterations",
                    str(args.iterations),
                    "--test_iterations",
                    str(args.iterations),
                    "--config",
                    args.config,
                    "--source_dir",
                    str(tmp_root),
                    "--model_dir",
                    str(scratch_root),
                    "--split",
                    args.split,
                ],
                env,
            )
        finally:
            cleanup_scan(tmp_root, scan_id)

    print(f"[done] tmp root kept at {tmp_root}", flush=True)


if __name__ == "__main__":
    main()
