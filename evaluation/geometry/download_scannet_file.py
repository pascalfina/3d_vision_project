#!/usr/bin/env python3
"""Direct ScanNet file downloader used by batch jobs.

The official ScanDownloader.py is fine interactively, but it re-fetches release
lists and prints the TOS prompt for every file type.  In long Slurm loops that
can stall before the actual file transfer.  This helper downloads the exact
known URL directly and keeps a temp file until the transfer completes.
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://kaldir.vc.in.tum.de/scannet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one ScanNet scan file.")
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--file-type", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--chunk-mb", type=int, default=8)
    return parser.parse_args()


def url_for(scan_id: str, file_type: str) -> str:
    release = "v1/scans" if file_type == ".sens" else "v2/scans"
    return f"{BASE_URL}/{release}/{scan_id}/{scan_id}{file_type}"


def download(url: str, dst: Path, *, timeout: int, chunk_mb: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists():
        tmp.unlink()

    chunk_size = max(1, int(chunk_mb)) * 1024 * 1024
    downloaded = 0
    next_report = 64 * 1024 * 1024
    started = time.time()
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                elapsed = max(time.time() - started, 1e-6)
                mb = downloaded / (1024 * 1024)
                print(f"[scannet-download] {dst.name}: {mb:.1f} MB at {mb / elapsed:.1f} MB/s", flush=True)
                next_report += 64 * 1024 * 1024
    tmp.replace(dst)


def main() -> None:
    args = parse_args()
    dst = args.out_dir / f"{args.scan_id}{args.file_type}"
    if dst.exists():
        print(f"[scannet-download] skip existing {dst}")
        return

    url = url_for(args.scan_id, args.file_type)
    last_error: Exception | None = None
    for attempt in range(1, max(1, args.retries) + 1):
        print(f"[scannet-download] attempt {attempt}/{args.retries}: {url} -> {dst}", flush=True)
        try:
            download(url, dst, timeout=args.timeout, chunk_mb=args.chunk_mb)
            print(f"[scannet-download] wrote {dst}", flush=True)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            part = dst.with_name(dst.name + ".part")
            if part.exists():
                part.unlink()
            print(f"[scannet-download] failed attempt {attempt}: {exc}", flush=True)
            time.sleep(min(30, 2 * attempt))

    raise SystemExit(f"Failed to download {url}: {last_error}")


if __name__ == "__main__":
    main()
