#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap a predicted-mask slot from existing gt_projection masks."
    )
    parser.add_argument("--data-root", required=True, help="Path to the dataset root.")
    parser.add_argument(
        "--source-dirname",
        default="gt_projection",
        help="Mask directory under files/ to copy from.",
    )
    parser.add_argument(
        "--target-dirname",
        default="pred_projection",
        help="Mask directory under files/ to create.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Whether to symlink or copy mask files into the target slot.",
    )
    parser.add_argument(
        "--scan-file",
        default=None,
        help="Optional text file with one scan id per line.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace existing target files.",
    )
    return parser.parse_args()


def iter_scan_ids(scan_file: Optional[Path], source_dir: Path):
    if scan_file is not None:
        for line in scan_file.read_text().splitlines():
            scan_id = line.strip()
            if scan_id:
                yield scan_id
        return

    for path in sorted(source_dir.iterdir()):
        if path.name.endswith(".pkl.gz"):
            yield path.name[:-7]
        elif path.name.endswith(".pkl"):
            yield path.name[:-4]


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_dir = data_root / "files" / args.source_dirname / "obj_id_pkl"
    target_dir = data_root / "files" / args.target_dirname / "obj_id_pkl"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source mask directory does not exist: {source_dir}")

    scan_file = Path(args.scan_file) if args.scan_file else None
    count = 0
    for scan_id in iter_scan_ids(scan_file, source_dir):
        source_file = None
        for candidate in (
            source_dir / f"{scan_id}.pkl.gz",
            source_dir / f"{scan_id}.pkl",
        ):
            if candidate.exists():
                source_file = candidate
                break
        if source_file is None:
            raise FileNotFoundError(
                f"Could not find mask file for scan {scan_id} in {source_dir}"
            )

        target_file = target_dir / source_file.name
        if target_file.exists() or target_file.is_symlink():
            if not args.override:
                print(f"[skip existing] {scan_id}")
                continue
            target_file.unlink()

        if args.mode == "symlink":
            os.symlink(source_file, target_file)
        else:
            shutil.copy2(source_file, target_file)

        count += 1
        print(f"[{args.mode}] {scan_id} -> {target_file}")

    print(
        f"[done] bootstrapped {count} mask files into "
        f"{data_root / 'files' / args.target_dirname}"
    )


if __name__ == "__main__":
    main()
