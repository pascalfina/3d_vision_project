#!/usr/bin/env python3
"""Download a balanced 3RScan reference-scan subset for Object-X."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "user")
    default_root = Path(f"/work/scratch/{user}/objectx-data-baseline")
    default_toolkit = Path(f"/work/scratch/{user}/3RScan-toolkit")

    parser = argparse.ArgumentParser(
        description="Download a balanced 3RScan subset using the official toolkit downloader."
    )
    parser.add_argument(
        "--toolkit-dir",
        type=Path,
        default=default_toolkit,
        help="Path to the 3RScan toolkit containing download.py and splits/.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=default_root,
        help="Baseline dataset root. scenes/ and subset/ will be created under this path.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Total number of reference scans to select if split counts are not given.",
    )
    parser.add_argument("--train", type=int, default=None, help="Number of train scans.")
    parser.add_argument("--val", type=int, default=None, help="Number of val scans.")
    parser.add_argument("--test", type=int, default=None, help="Number of test scans.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing subset lists and downloaded subset scans before starting.",
    )
    parser.add_argument(
        "--assume-tos",
        action="store_true",
        help="Skip the interactive Terms-of-Use confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create subset lists and print what would be downloaded.",
    )
    return parser.parse_args()


def load_download_module(toolkit_dir: Path):
    module_path = toolkit_dir / "download.py"
    if not module_path.exists():
        raise FileNotFoundError(f"download.py not found at {module_path}")

    spec = importlib.util.spec_from_file_location("scan3r_download", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def allocate_counts(total: int, split_sizes: dict[str, int]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("--total must be positive")

    total_available = sum(split_sizes.values())
    if total > total_available:
        raise ValueError(
            f"Requested total {total} exceeds available reference scans {total_available}"
        )

    raw = {
        split: total * size / total_available for split, size in split_sizes.items()
    }
    counts = {split: int(value) for split, value in raw.items()}

    remainder = total - sum(counts.values())
    for split, _ in sorted(
        raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    ):
        if remainder == 0:
            break
        counts[split] += 1
        remainder -= 1

    return counts


def pick_evenly(ids: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if count >= len(ids):
        return ids[:]
    if count == 1:
        return [ids[len(ids) // 2]]

    positions = [round(i * (len(ids) - 1) / (count - 1)) for i in range(count)]
    chosen: list[str] = []
    seen: set[str] = set()
    for pos in positions:
        scan_id = ids[pos]
        if scan_id not in seen:
            chosen.append(scan_id)
            seen.add(scan_id)

    if len(chosen) < count:
        for scan_id in ids:
            if scan_id not in seen:
                chosen.append(scan_id)
                seen.add(scan_id)
                if len(chosen) == count:
                    break

    return chosen


def ensure_tos_confirmed(url: str, assume_tos: bool) -> None:
    if assume_tos:
        return
    print("By continuing you confirm that you agreed to the 3RScan Terms of Use:")
    print(url)
    print("***")
    input("Press Enter to continue, or CTRL-C to abort. ")


def main() -> int:
    args = parse_args()

    toolkit_dir = args.toolkit_dir.expanduser().resolve()
    root_dir = args.root_dir.expanduser().resolve()
    scenes_dir = root_dir / "scenes"
    subset_dir = root_dir / "subset"
    splits_dir = toolkit_dir / "splits"

    downloader = load_download_module(toolkit_dir)

    split_files = {split: splits_dir / f"{split}.txt" for split in ("train", "val", "test")}
    missing = [str(path) for path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing}")

    split_ids = {split: read_ids(path) for split, path in split_files.items()}
    split_sizes = {split: len(ids) for split, ids in split_ids.items()}

    if args.train is None and args.val is None and args.test is None:
        counts = allocate_counts(args.total, split_sizes)
    else:
        counts = {
            "train": args.train or 0,
            "val": args.val or 0,
            "test": args.test or 0,
        }

    for split, count in counts.items():
        if count > split_sizes[split]:
            raise ValueError(
                f"Requested {count} {split} scans, but only {split_sizes[split]} are available"
            )

    if args.reset:
        shutil.rmtree(scenes_dir, ignore_errors=True)
        shutil.rmtree(subset_dir, ignore_errors=True)

    scenes_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    selected = {
        split: pick_evenly(split_ids[split], counts[split]) for split in ("train", "val", "test")
    }
    all_ids = selected["train"] + selected["val"] + selected["test"]

    for split, ids in selected.items():
        (subset_dir / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    (subset_dir / "all.txt").write_text("\n".join(all_ids) + ("\n" if all_ids else ""))

    print("Selected subset:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(selected[split])}")
    print(f"  total: {len(all_ids)}")
    print(f"Subset lists written to: {subset_dir}")

    if args.dry_run:
        return 0

    ensure_tos_confirmed(downloader.TOS_URL, args.assume_tos)

    release = set(downloader.get_scans(downloader.BASE_URL + downloader.RELEASE))
    hidden = set(downloader.get_scans(downloader.BASE_URL + downloader.HIDDEN_RELEASE))

    for idx, scan_id in enumerate(all_ids, 1):
        print(f"[{idx}/{len(all_ids)}] {scan_id}")
        if scan_id in release:
            file_types = downloader.FILETYPES
        elif scan_id in hidden:
            file_types = downloader.TEST_FILETYPES
        else:
            print(f"Skipping unknown scan id: {scan_id}")
            continue
        downloader.download_scan(scan_id, str(scenes_dir / scan_id), file_types)

    print("Subset download finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
