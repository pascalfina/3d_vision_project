#!/usr/bin/env python3
import argparse
import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--vlsG-space", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--tmp-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--max-scans", type=int, default=0)
    parser.add_argument("--override", action="store_true")
    return parser.parse_args()


def safe_unlink(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def load_scan_ids(root: Path, split: str) -> list[str]:
    preferred = root / "files" / f"{split}_resplit_scans.txt"
    fallback = root / "files" / f"{split}_scans.txt"
    path = preferred if preferred.exists() else fallback
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def output_exists(scratch_root: Path, scan_id: str) -> bool:
    out_dir = scratch_root / "files" / "Features3D" / "obj_dinov2_top10_l3"
    plain = out_dir / f"{scan_id}.pkl"
    gz = out_dir / f"{scan_id}.pkl.gz"
    for candidate in (plain, gz):
        if candidate.exists() and candidate.stat().st_size > 128:
            return True
    return False


def stage_scan(src_scan_dir: Path, tmp_scan_dir: Path):
    if tmp_scan_dir.exists():
        shutil.rmtree(tmp_scan_dir)
    tmp_scan_dir.mkdir(parents=True, exist_ok=True)

    for item in src_scan_dir.iterdir():
        if item.name == "sequence.zip":
            continue
        target = tmp_scan_dir / item.name
        safe_unlink(target)
        os.symlink(item, target)

    (tmp_scan_dir / "sequence").mkdir(exist_ok=True)
    subprocess.run(
        ["unzip", "-qo", str(src_scan_dir / "sequence.zip"), "-d", str(tmp_scan_dir / "sequence")],
        check=True,
    )


def cleanup_scan(tmp_scan_dir: Path):
    if tmp_scan_dir.exists():
        shutil.rmtree(tmp_scan_dir)


def prepare_tmp_layout(scratch_root: Path, tmp_root: Path):
    (tmp_root / "scenes").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files" / "orig").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files" / "Features3D" / "obj_dinov2_top10_l3").mkdir(
        parents=True, exist_ok=True
    )

    for name in ["3RScan.json"]:
        src = scratch_root / "files" / name
        dst = tmp_root / "files" / name
        safe_unlink(dst)
        os.symlink(src, dst)

    src = scratch_root / "files" / "gt_projection"
    dst = tmp_root / "files" / "gt_projection"
    safe_unlink(dst)
    os.symlink(src, dst)


def write_single_scan_split(tmp_root: Path, split: str, scan_id: str):
    split_file = tmp_root / "files" / "orig" / f"{split}_resplit_scans.txt"
    split_file.write_text(scan_id + "\n")


def compress_to_scratch(tmp_file: Path, scratch_root: Path, scan_id: str):
    out_dir = scratch_root / "files" / "Features3D" / "obj_dinov2_top10_l3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{scan_id}.pkl.gz"
    tmp_out = out_dir / f"{scan_id}.pkl.gz.tmp"
    with open(tmp_file, "rb") as f_in, gzip.open(tmp_out, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.replace(tmp_out, out_file)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    vlsG_space = Path(args.vlsG_space)
    scratch_root = Path(args.scratch_root)
    tmp_root = Path(args.tmp_root)

    prepare_tmp_layout(scratch_root, tmp_root)

    scan_ids = load_scan_ids(scratch_root, args.split)
    if args.max_scans > 0:
        scan_ids = scan_ids[: args.max_scans]

    total = len(scan_ids)
    for idx, scan_id in enumerate(scan_ids, start=1):
        if output_exists(scratch_root, scan_id) and not args.override:
            if idx % 10 == 0 or idx == total:
                print(f"[feat3d] {idx}/{total} (skip existing) {scan_id}", flush=True)
            continue

        src_scan_dir = scratch_root / "scenes" / scan_id
        tmp_scan_dir = tmp_root / "scenes" / scan_id
        tmp_out_file = (
            tmp_root / "files" / "Features3D" / "obj_dinov2_top10_l3" / f"{scan_id}.pkl"
        )

        print(f"[feat3d] {idx}/{total} {args.split} {scan_id}", flush=True)
        write_single_scan_split(tmp_root, args.split, scan_id)
        stage_scan(src_scan_dir, tmp_scan_dir)

        env = os.environ.copy()
        env["Data_ROOT_DIR"] = str(tmp_root)
        env["VLSG_SPACE"] = str(vlsG_space)

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(
                        vlsG_space
                        / "preprocessing/sg_features/obj_visual_embeddings/Dinov2/gen_obj_visual_embeddings.py"
                    ),
                    "--config",
                    str(Path(args.config)),
                    "--split",
                    args.split,
                ],
                cwd=str(vlsG_space),
                env=env,
                check=True,
            )
            compress_to_scratch(tmp_out_file, scratch_root, scan_id)
        finally:
            cleanup_scan(tmp_scan_dir)
            if tmp_out_file.exists():
                tmp_out_file.unlink()

    print(f"[done] tmp root kept at {tmp_root}", flush=True)


if __name__ == "__main__":
    main()
