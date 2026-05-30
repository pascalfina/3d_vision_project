#!/usr/bin/env python3
"""Select ScanNet scenes with at most N color frames.

The ScanNet downloader can fetch whole scans, but selecting short scenes first
is much cheaper if we only download the tiny per-scene metadata files.  This
script writes a stable TSV manifest that downstream benchmark scripts can reuse.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://kaldir.vc.in.tum.de/scannet/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select short ScanNet v2 scenes.")
    parser.add_argument("--scannet-root", type=Path, default=Path("/work/scratch/pafina/scannet_under300_data"))
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("strict", "cap"),
        default="strict",
        help=(
            "strict: require raw numColorFrames <= max-frames. "
            "cap: select scenes with at least min-frames and report exported "
            "frame count as min(raw, max-frames)."
        ),
    )
    parser.add_argument(
        "--order",
        choices=("release", "shortest"),
        default="release",
        help="release: keep ScanNet release order. shortest: choose scenes with the fewest raw frames.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/workflows/scene_profiles/scannet_under300_scenes.tsv"),
    )
    parser.add_argument(
        "--release-url",
        default=f"{BASE_URL}/scans.txt",
        help="URL containing one ScanNet scan id per line.",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Re-download existing .txt metadata files.",
    )
    parser.add_argument(
        "--include-existing-prepared",
        action="store_true",
        default=True,
        help="Keep already prepared scenes even if metadata download fails.",
    )
    return parser.parse_args()


def url_read_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as handle:
        return handle.read().decode("utf-8")


def load_scan_ids(args: argparse.Namespace) -> list[str]:
    cache_path = args.scannet_root / "scannetv2_scans.txt"
    try:
        text = url_read_text(args.release_url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[scannet-select] warning: release list unavailable, using cache/local metadata: {exc}", file=sys.stderr)
    if cache_path.exists():
        return [line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scan_dirs = sorted(path.parent.name for path in (args.scannet_root / "scans").glob("scene*/scene*.txt"))
    if scan_dirs:
        return scan_dirs
    raise SystemExit(
        f"Could not load ScanNet release list from {args.release_url}, cache {cache_path}, "
        "or local metadata."
    )


def download_file(url: str, path: Path, *, refresh: bool = False) -> bool:
    if path.exists() and not refresh:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as handle:
            tmp.write_bytes(handle.read())
        tmp.replace(path)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"[scannet-select] warning: failed to download {url}: {exc}", file=sys.stderr)
        return False


def parse_num_color_frames(text: str) -> int | None:
    for line in text.splitlines():
        if "numColorFrames" not in line:
            continue
        match = re.search(r"numColorFrames\s*=\s*(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def prepared_frame_count(scan_dir: Path) -> int | None:
    color_dir = scan_dir / "data" / "color"
    if not color_dir.is_dir():
        return None
    return sum(1 for path in color_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main() -> None:
    args = parse_args()
    scans_root = args.scannet_root / "scans"
    scan_ids = load_scan_ids(args)

    candidates: list[tuple[str, int, int, str]] = []
    checked = 0
    for scan_id in scan_ids:
        checked += 1
        scan_dir = scans_root / scan_id
        txt_path = scan_dir / f"{scan_id}.txt"
        url = f"{BASE_URL}/scans/{scan_id}/{scan_id}.txt"
        if not download_file(url, txt_path, refresh=args.refresh_metadata):
            if args.include_existing_prepared:
                existing = prepared_frame_count(scan_dir)
                if existing is not None and existing <= args.max_frames and existing >= args.min_frames:
                    candidates.append((scan_id, existing, existing, "prepared"))
            continue
        frame_count = parse_num_color_frames(txt_path.read_text(errors="replace"))
        if frame_count is None:
            print(f"[scannet-select] warning: no numColorFrames in {txt_path}", file=sys.stderr)
            continue
        if frame_count < args.min_frames:
            continue
        if args.mode == "strict":
            if frame_count <= args.max_frames:
                candidates.append((scan_id, frame_count, frame_count, "metadata_strict"))
        else:
            candidates.append((scan_id, min(frame_count, args.max_frames), frame_count, "metadata_capped"))
        if args.order == "release" and len(candidates) >= args.target:
            break

    if args.order == "shortest":
        candidates.sort(key=lambda row: (row[2], row[0]))
    selected = [
        (rank, scan_id, frames, raw_frames, source)
        for rank, (scan_id, frames, raw_frames, source) in enumerate(candidates[: args.target], start=1)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["rank\tscene_id\tframes\traw_frames\tsource"]
    lines.extend(
        f"{rank}\t{scan_id}\t{frames}\t{raw_frames}\t{source}"
        for rank, scan_id, frames, raw_frames, source in selected
    )
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    profile_list = args.out.with_name("scannet_under300_profiles.txt")
    profile_lines = [f"scannet_{scan_id}_pi3x" for _, scan_id, _, _, _ in selected]
    profile_list.write_text("\n".join(profile_lines) + ("\n" if profile_lines else ""), encoding="utf-8")

    print(
        f"[scannet-select] selected={len(selected)} target={args.target} "
        f"max_frames={args.max_frames} mode={args.mode} order={args.order} checked={checked}"
    )
    print(f"[scannet-select] scenes={args.out}")
    print(f"[scannet-select] profiles={profile_list}")
    if len(selected) < args.target:
        raise SystemExit(
            f"Only selected {len(selected)} scenes with <= {args.max_frames} frames; "
            "increase the max frame threshold or inspect ScanNet metadata."
        )


if __name__ == "__main__":
    main()
