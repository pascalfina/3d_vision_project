#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--tmp-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--max-scans", type=int, default=0)
    return parser.parse_args()


def load_scan_ids(root: Path, split: str) -> list[str]:
    path = root / "files" / f"{split}_resplit_scans.txt"
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def all_outputs_exist(scratch_root: Path, obj_data: dict) -> bool:
    scan_id = obj_data["scan"]
    for obj in obj_data["objects"]:
        obj_id = str(obj["id"])
        voxel_path = (
            scratch_root
            / "files"
            / "gs_annotations"
            / scan_id
            / obj_id
            / "voxel_output_dense.npz"
        )
        mean_scale_path = (
            scratch_root
            / "files"
            / "gs_annotations"
            / scan_id
            / obj_id
            / "mean_scale_dense.npz"
        )
        if not voxel_path.exists() or not mean_scale_path.exists():
            return False
    return True


def stage_scan(src_scan_dir: Path, tmp_scan_dir: Path):
    if tmp_scan_dir.exists():
        shutil.rmtree(tmp_scan_dir)
    tmp_scan_dir.mkdir(parents=True, exist_ok=True)

    for item in src_scan_dir.iterdir():
        if item.name in {"sequence.zip", "sequence"}:
            continue
        target = tmp_scan_dir / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(item, target)

    src_zip = src_scan_dir / "sequence.zip"
    src_seq = src_scan_dir / "sequence"
    dst_seq = tmp_scan_dir / "sequence"
    if src_zip.exists():
        dst_seq.mkdir(exist_ok=True)
        subprocess.run(
            ["unzip", "-qo", str(src_zip), "-d", str(dst_seq)],
            check=True,
        )
    elif src_seq.exists():
        os.symlink(src_seq, dst_seq)
    else:
        raise FileNotFoundError(f"Neither sequence.zip nor sequence/ found in {src_scan_dir}")


def cleanup_scan(tmp_scan_dir: Path):
    if tmp_scan_dir.exists():
        shutil.rmtree(tmp_scan_dir)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    scratch_root = Path(args.scratch_root)
    tmp_root = Path(args.tmp_root)

    sys.path.insert(0, str(repo_root))
    from configs import update_configs
    import preprocessing.voxel_anno.voxelise_features as vf

    cfg = update_configs(args.config, [], do_ensure_dir=False)

    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / "scenes").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files").mkdir(parents=True, exist_ok=True)

    for name in [
        "3RScan.json",
        "objects.json",
        "train_resplit_scans.txt",
        "val_resplit_scans.txt",
        "test_resplit_scans.txt",
        "train_scans.txt",
        "val_scans.txt",
        "test_scans.txt",
    ]:
        src = scratch_root / "files" / name
        dst = tmp_root / "files" / name
        if src.exists():
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)

    for name in ["gt_projection"]:
        src = scratch_root / "files" / name
        dst = tmp_root / "files" / name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        os.symlink(src, dst)

    os.environ["DATA_ROOT_DIR"] = str(tmp_root)
    vf.args = argparse.Namespace(
        config=args.config,
        split=args.split,
        model_dir=str(scratch_root),
        model=args.model,
        visualize=False,
        vis_dir=str(repo_root / "vis"),
        dry_run=False,
        override=args.override,
    )
    vf.root_dir = str(tmp_root)
    vf.model = vf._load_dino_model(args.model)
    vf.model.eval().cuda()
    vf.transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    all_obj_info = json.load(open(scratch_root / "files" / "objects.json"))["scans"]
    obj_lookup = {entry["scan"]: entry for entry in all_obj_info}
    scan_ids = load_scan_ids(scratch_root, args.split)
    if args.max_scans > 0:
        scan_ids = scan_ids[: args.max_scans]

    total = len(scan_ids)
    for idx, scan_id in enumerate(scan_ids, start=1):
        obj_data = obj_lookup[scan_id]
        if all_outputs_exist(scratch_root, obj_data) and not args.override:
            if idx % 10 == 0 or idx == total:
                print(f"[2.5] {idx}/{total} (skip existing) {scan_id}", flush=True)
            continue

        src_scan_dir = scratch_root / "scenes" / scan_id
        tmp_scan_dir = tmp_root / "scenes" / scan_id

        print(f"[2.5] {idx}/{total} {args.split} {scan_id}", flush=True)
        stage_scan(src_scan_dir, tmp_scan_dir)
        try:
            vf.voxelise_features(obj_data=obj_data, scan_id=scan_id, mode="gs_annotations")
        finally:
            cleanup_scan(tmp_scan_dir)

    print(f"[done] tmp root kept at {tmp_root}", flush=True)


if __name__ == "__main__":
    main()
