#!/usr/bin/env python3
import argparse
import json
import logging
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


def resolve_mask_source() -> str:
    source = (os.environ.get("OBJECTX_MASK_SOURCE") or "gt_projection").strip()
    aliases = {
        "gt": "gt_projection",
        "gt_projection": "gt_projection",
        "pred": "pred_projection",
        "pred_projection": "pred_projection",
    }
    return aliases.get(source, source)


def ensure_cache_env() -> None:
    cache_root = os.environ.get("OBJECTX_CACHE_ROOT") or "/work/scratch/pafina/objectx-cache"
    torch_home = os.environ.get("TORCH_HOME") or str(Path(cache_root) / "torch")
    xdg_cache = os.environ.get("XDG_CACHE_HOME") or str(Path(cache_root) / "xdg")
    mpl_cache = os.environ.get("MPLCONFIGDIR") or str(Path(cache_root) / "matplotlib")
    dinov2_hub_dir = os.environ.get("OBJECTX_DINOV2_HUB_DIR") or str(
        Path(torch_home) / "hub" / "facebookresearch_dinov2_main"
    )

    os.environ.setdefault("OBJECTX_CACHE_ROOT", cache_root)
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache)
    os.environ.setdefault("OBJECTX_DINOV2_HUB_DIR", dinov2_hub_dir)

    Path(torch_home).mkdir(parents=True, exist_ok=True)
    Path(xdg_cache).mkdir(parents=True, exist_ok=True)
    Path(mpl_cache).mkdir(parents=True, exist_ok=True)


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
    ensure_cache_env()

    sys.path.insert(0, str(repo_root))
    from configs import update_configs
    from utils import common
    import preprocessing.voxel_anno.voxelise_features as vf

    common.init_log(level=logging.INFO)

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

    mask_dirname = resolve_mask_source()
    src = scratch_root / "files" / mask_dirname
    if not src.exists():
        raise FileNotFoundError(
            f"Mask source directory does not exist: {src} "
            f"(OBJECTX_MASK_SOURCE={mask_dirname})"
        )

    actual_dst = tmp_root / "files" / mask_dirname
    if actual_dst.exists() or actual_dst.is_symlink():
        if actual_dst.is_dir() and not actual_dst.is_symlink():
            shutil.rmtree(actual_dst)
        else:
            actual_dst.unlink()
    os.symlink(src, actual_dst)

    alias_dst = tmp_root / "files" / "gt_projection"
    if alias_dst != actual_dst:
        if alias_dst.exists() or alias_dst.is_symlink():
            if alias_dst.is_dir() and not alias_dst.is_symlink():
                shutil.rmtree(alias_dst)
            else:
                alias_dst.unlink()
        os.symlink(src, alias_dst)
    print(f"[2.5] using mask source {mask_dirname}", flush=True)
    print(
        f"[2.5] using cache root {os.environ['OBJECTX_CACHE_ROOT']} "
        f"(hub={os.environ['OBJECTX_DINOV2_HUB_DIR']})",
        flush=True,
    )

    os.environ["DATA_ROOT_DIR"] = str(tmp_root)
    visualize = os.environ.get("OBJECTX_VOXEL_VISUALIZE", "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    vf.args = argparse.Namespace(
        config=args.config,
        split=args.split,
        model_dir=str(scratch_root),
        model=args.model,
        visualize=visualize,
        vis_dir=str(repo_root / "vis"),
        dry_run=False,
        override=args.override,
        mask_source=mask_dirname,
        object_source=os.environ.get("OBJECTX_VOXEL_OBJECT_SOURCE"),
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
