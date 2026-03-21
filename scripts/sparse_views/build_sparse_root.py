#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import pickle
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sparse-view Object-X root from an existing full-view root."
    )
    parser.add_argument(
        "--full-root",
        default="/work/scratch/pafina/objectx-data-baseline",
        help="Existing full-view dataset root.",
    )
    parser.add_argument(
        "--target-root",
        default="/work/scratch/pafina/objectx-data-sparse10-v8",
        help="Sparse-view dataset root to create.",
    )
    parser.add_argument(
        "--benchmark-prefix",
        default="sparse_benchmark_10",
        help="Prefix for benchmark split files under <full-root>/files.",
    )
    parser.add_argument(
        "--views-per-scene",
        type=int,
        default=0,
        help="Number of frames to keep per scene. Mutually exclusive with --keep-ratio.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.0,
        help="Fraction of frames to keep per scene, e.g. 0.75. Mutually exclusive with --views-per-scene.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the target root before creating it.",
    )
    args = parser.parse_args()
    if bool(args.views_per_scene > 0) == bool(args.keep_ratio > 0):
        parser.error("Specify exactly one of --views-per-scene or --keep-ratio.")
    if args.keep_ratio <= 0 and args.views_per_scene <= 0:
        parser.error("A positive sparse selection must be specified.")
    if args.keep_ratio > 1:
        parser.error("--keep-ratio must be in (0, 1].")
    return args


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def choose_evenly_spaced(items: list[str], k: int) -> list[str]:
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[0]]

    chosen_indices = []
    for i in range(k):
        idx = round(i * (len(items) - 1) / (k - 1))
        if idx not in chosen_indices:
            chosen_indices.append(idx)

    chosen = [items[idx] for idx in chosen_indices]
    # In case rounding still produced duplicates, pad deterministically.
    if len(chosen) < k:
        seen = set(chosen_indices)
        for idx, item in enumerate(items):
            if idx in seen:
                continue
            chosen.append(item)
            if len(chosen) == k:
                break
    return chosen


def choose_keep_count(num_items: int, views_per_scene: int, keep_ratio: float) -> int:
    if num_items <= 0:
        return 0
    if keep_ratio > 0:
        return max(1, min(num_items, round(num_items * keep_ratio)))
    return max(1, min(num_items, views_per_scene))


def output_tag(args: argparse.Namespace) -> str:
    if args.keep_ratio > 0:
        return f"keep{round(args.keep_ratio * 100):02d}pct"
    return f"views{args.views_per_scene}"


def load_obj_masks(mask_file: Path) -> dict:
    with gzip.open(mask_file, "rb") as handle:
        return pickle.load(handle)


def write_obj_masks(mask_dict: dict, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_file, "wb") as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    os.symlink(src, dst)


def extract_sparse_sequence(
    src_zip: Path,
    dst_sequence_dir: Path,
    selected_frames: list[str],
) -> None:
    dst_sequence_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_zip, "r") as zf:
        members = set(zf.namelist())
        required = ["_info.txt", "sequence/_info.txt"]
        for frame_id in selected_frames:
            required.extend(
                [
                    f"frame-{frame_id}.color.jpg",
                    f"frame-{frame_id}.pose.txt",
                    f"frame-{frame_id}.depth.pgm",
                    f"sequence/frame-{frame_id}.color.jpg",
                    f"sequence/frame-{frame_id}.pose.txt",
                    f"sequence/frame-{frame_id}.depth.pgm",
                ]
            )

        for member in required:
            if member in members:
                out_name = Path(member).name
                with zf.open(member, "r") as src, open(dst_sequence_dir / out_name, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def subset_3rscan_json(src: Path, dst: Path, keep_scans: set[str]) -> None:
    data = json.loads(src.read_text())
    subset = []
    for entry in data:
        if entry["reference"] in keep_scans:
            subset.append(
                {
                    "ambiguity": entry.get("ambiguity", []),
                    "reference": entry["reference"],
                    "type": entry.get("type", "train"),
                    "scans": entry.get("scans", []),
                }
            )
    dst.write_text(json.dumps(subset))


def write_split_files(files_dir: Path, prefix: str, train: list[str], val: list[str], test: list[str]) -> None:
    split_map = {"train": train, "val": val, "test": test}
    for split, scan_ids in split_map.items():
        (files_dir / f"{split}_resplit_scans.txt").write_text("\n".join(scan_ids) + "\n")
        (files_dir / f"{split}_scans.txt").write_text("\n".join(scan_ids) + "\n")


def main() -> None:
    args = parse_args()
    full_root = Path(args.full_root)
    target_root = Path(args.target_root)
    full_files = full_root / "files"
    target_files = target_root / "files"

    train = load_lines(full_files / f"{args.benchmark_prefix}_train.txt")
    val = load_lines(full_files / f"{args.benchmark_prefix}_val.txt")
    test = load_lines(full_files / f"{args.benchmark_prefix}_test.txt")
    benchmark_scans = train + val + test

    if args.reset and target_root.exists():
        shutil.rmtree(target_root)

    (target_root / "scenes").mkdir(parents=True, exist_ok=True)
    target_files.mkdir(parents=True, exist_ok=True)

    # Share non-view-dependent metadata/artifacts.
    for name in ["orig", "objects.json", "scannet40_classes.txt", "patch_anno"]:
        src = full_files / name
        if src.exists():
            safe_symlink(src, target_files / name)

    subset_3rscan_json(full_files / "3RScan.json", target_files / "3RScan.json", set(benchmark_scans))
    write_split_files(target_files, args.benchmark_prefix, train, val, test)

    gt_out_dir = target_files / "gt_projection" / "obj_id_pkl"
    gt_out_dir.mkdir(parents=True, exist_ok=True)

    selected_frames_by_scan = {}
    stats_by_scan = {}
    total_full_frames = 0
    total_kept_frames = 0
    for scan_id in benchmark_scans:
        src_scene = full_root / "scenes" / scan_id
        dst_scene = target_root / "scenes" / scan_id
        dst_scene.mkdir(parents=True, exist_ok=True)

        # Link static scene assets.
        for item in src_scene.iterdir():
            if item.name in {"sequence", "sequence.zip"}:
                continue
            safe_symlink(item, dst_scene / item.name)

        full_mask_file = full_files / "gt_projection" / "obj_id_pkl" / f"{scan_id}.pkl.gz"
        full_masks = load_obj_masks(full_mask_file)
        all_frame_ids = sorted(full_masks.keys())
        keep_count = choose_keep_count(
            len(all_frame_ids), args.views_per_scene, args.keep_ratio
        )
        selected_frames = choose_evenly_spaced(all_frame_ids, keep_count)
        selected_frames_by_scan[scan_id] = selected_frames
        full_frame_count = len(all_frame_ids)
        kept_frame_count = len(selected_frames)
        kept_pct = (kept_frame_count / full_frame_count * 100.0) if full_frame_count else 0.0
        removed_pct = 100.0 - kept_pct if full_frame_count else 0.0
        stats_by_scan[scan_id] = {
            "full_frames": full_frame_count,
            "kept_frames": kept_frame_count,
            "kept_pct": round(kept_pct, 2),
            "removed_pct": round(removed_pct, 2),
        }
        total_full_frames += full_frame_count
        total_kept_frames += kept_frame_count

        # Extract only the selected frame files.
        extract_sparse_sequence(src_scene / "sequence.zip", dst_scene / "sequence", selected_frames)

        # Keep only the selected frame masks.
        sparse_masks = {frame_id: full_masks[frame_id] for frame_id in selected_frames}
        write_obj_masks(sparse_masks, gt_out_dir / f"{scan_id}.pkl.gz")

    tag = output_tag(args)
    views_json = target_files / f"{args.benchmark_prefix}_{tag}.json"
    views_json.write_text(json.dumps(selected_frames_by_scan, indent=2))
    stats_json = target_files / f"{args.benchmark_prefix}_{tag}_stats.json"
    stats_json.write_text(json.dumps(stats_by_scan, indent=2))

    overall_kept_pct = (total_kept_frames / total_full_frames * 100.0) if total_full_frames else 0.0
    overall_removed_pct = 100.0 - overall_kept_pct if total_full_frames else 0.0

    print(f"Created sparse root: {target_root}")
    print(f"Scenes: {len(benchmark_scans)}")
    if args.keep_ratio > 0:
        print(f"Keep ratio: {args.keep_ratio:.2f}")
    else:
        print(f"Views per scene: {args.views_per_scene}")
    print(f"View selection written to: {views_json}")
    print(f"View stats written to: {stats_json}")
    print(
        f"Overall frames kept: {total_kept_frames}/{total_full_frames} "
        f"({overall_kept_pct:.2f}% kept, {overall_removed_pct:.2f}% removed)"
    )
    for scan_id in benchmark_scans:
        stats = stats_by_scan[scan_id]
        print(
            f"[{scan_id}] kept {stats['kept_frames']}/{stats['full_frames']} frames "
            f"({stats['kept_pct']:.2f}% kept, {stats['removed_pct']:.2f}% removed)"
        )


if __name__ == "__main__":
    main()
