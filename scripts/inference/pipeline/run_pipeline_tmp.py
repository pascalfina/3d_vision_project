#!/usr/bin/env python3
import argparse
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
    parser.add_argument("--mode", required=True, choices=["slat", "u3dgs"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
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


def resolve_mask_source() -> str:
    source = (os.environ.get("OBJECTX_MASK_SOURCE") or "gt_projection").strip()
    aliases = {
        "gt": "gt_projection",
        "gt_projection": "gt_projection",
        "pred": "pred_projection",
        "pred_projection": "pred_projection",
        "sam2": "sam2_projection",
        "sam2_projection": "sam2_projection",
        "sam3": "sam3_projection",
        "sam3_projection": "sam3_projection",
    }
    return aliases.get(source, source)


def stage_scan(src_scan_dir: Path, tmp_scan_dir: Path):
    if tmp_scan_dir.exists():
        shutil.rmtree(tmp_scan_dir)
    tmp_scan_dir.mkdir(parents=True, exist_ok=True)

    for item in src_scan_dir.iterdir():
        if item.name in {"sequence.zip", "sequence"}:
            continue
        target = tmp_scan_dir / item.name
        safe_unlink(target)
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


def prepare_tmp_root(scratch_root: Path, tmp_root: Path, split: str, scan_ids: list[str]):
    (tmp_root / "scenes").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files").mkdir(parents=True, exist_ok=True)

    # Link persistent metadata and preprocessed artifacts.
    mask_dirname = resolve_mask_source()
    link_names = [
        "3RScan.json",
        "objects.json",
        "scannet40_classes.txt",
        "orig",
        "gs_annotations",
        "Features3D",
    ]
    for name in link_names:
        src = scratch_root / "files" / name
        dst = tmp_root / "files" / name
        safe_unlink(dst)
        if src.exists():
            os.symlink(src, dst)

    mask_src = scratch_root / "files" / mask_dirname
    if mask_src.exists():
        actual_dst = tmp_root / "files" / mask_dirname
        safe_unlink(actual_dst)
        os.symlink(mask_src, actual_dst)

        alias_dst = tmp_root / "files" / "gt_projection"
        if alias_dst != actual_dst:
            safe_unlink(alias_dst)
            os.symlink(mask_src, alias_dst)
    print(f"[infer] using mask source {mask_dirname}", flush=True)

    # Write one-line split file for targeted inference, or mirror the requested split.
    split_file = tmp_root / "files" / f"{split}_resplit_scans.txt"
    split_file.write_text("\n".join(scan_ids) + "\n")

    # Keep the other split files present to avoid surprises in downstream code.
    for other in ["train", "val", "test"]:
        path = tmp_root / "files" / f"{other}_resplit_scans.txt"
        if other == split:
            continue
        src = scratch_root / "files" / f"{other}_resplit_scans.txt"
        if src.exists():
            safe_unlink(path)
            os.symlink(src, path)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    scratch_root = Path(args.scratch_root)
    tmp_root = Path(args.tmp_root)

    scan_ids = [args.scene_id] if args.scene_id else load_scan_ids(scratch_root, args.split)
    prepare_tmp_root(scratch_root, tmp_root, args.split, scan_ids)

    for scan_id in scan_ids:
        print(f"[infer] staging {scan_id}", flush=True)
        stage_scan(scratch_root / "scenes" / scan_id, tmp_root / "scenes" / scan_id)

    script = (
        repo_root / "src" / "inference" / ("structured_latent_inference.py" if args.mode == "slat" else "unstructured_latent_inference.py")
    )
    env = os.environ.copy()
    env["DATA_ROOT_DIR"] = str(tmp_root)

    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(repo_root / "configs" / "config.yaml"),
        "--split",
        args.split,
    ]
    if args.visualize:
        cmd.append("--visualize")
    if args.scene_id:
        cmd.extend(["--scene_id", args.scene_id])
    if args.extra:
        extra = args.extra
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

    # Persist embeddings on scratch even though DATA_ROOT_DIR points to tmp.
    cmd.append(f"inference.output_dir={scratch_root / 'files' / 'gs_embeddings'}")

    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    print(f"[done] tmp root kept at {tmp_root}", flush=True)


if __name__ == "__main__":
    main()
