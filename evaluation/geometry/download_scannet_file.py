#!/usr/bin/env python3
"""Direct ScanNet file downloader used by batch jobs.

The official ScanDownloader.py is fine interactively, but it re-fetches release
lists and prints the TOS prompt for every file type.  In long Slurm loops that
can stall before the actual file transfer.  This helper downloads the exact
known URL directly and keeps a temp file until the transfer completes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
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
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "urllib", "wget", "curl"),
        help="Download backend. auto prefers wget/curl so interrupted .part files can resume.",
    )
    return parser.parse_args()


def url_for(scan_id: str, file_type: str) -> str:
    release = "v1/scans" if file_type == ".sens" else "v2/scans"
    return f"{BASE_URL}/{release}/{scan_id}/{scan_id}{file_type}"


def download_urllib(url: str, dst: Path, *, timeout: int, chunk_mb: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    resume_at = tmp.stat().st_size if tmp.exists() else 0

    chunk_size = max(1, int(chunk_mb)) * 1024 * 1024
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
    request = urllib.request.Request(url, headers=headers)
    downloaded = resume_at
    next_report = max(64 * 1024 * 1024, ((downloaded // (64 * 1024 * 1024)) + 1) * 64 * 1024 * 1024)
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        append = resume_at > 0 and getattr(response, "status", None) == 206
        if resume_at and not append:
            print(f"[scannet-download] server did not resume {dst.name}; restarting partial file", flush=True)
            downloaded = 0
        with tmp.open("ab" if append else "wb") as handle:
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


def download_with_wget(url: str, dst: Path, *, timeout: int, retries: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    cmd = [
        "wget",
        "-c",
        "--tries",
        str(max(1, retries)),
        "--timeout",
        str(timeout),
        "--read-timeout",
        str(timeout),
        "--progress=dot:giga",
        "-O",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def download_with_curl(url: str, dst: Path, *, timeout: int, retries: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(max(0, retries - 1)),
        "--retry-delay",
        "5",
        "--connect-timeout",
        str(timeout),
        "--continue-at",
        "-",
        "--output",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if shutil.which("wget"):
        return "wget"
    if shutil.which("curl"):
        return "curl"
    return "urllib"


def download(url: str, dst: Path, *, timeout: int, retries: int, chunk_mb: int, backend: str) -> None:
    backend = resolve_backend(backend)
    print(f"[scannet-download] backend={backend}", flush=True)
    if backend == "wget":
        download_with_wget(url, dst, timeout=timeout, retries=retries)
    elif backend == "curl":
        download_with_curl(url, dst, timeout=timeout, retries=retries)
    else:
        download_urllib(url, dst, timeout=timeout, chunk_mb=chunk_mb)


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
            download(
                url,
                dst,
                timeout=args.timeout,
                retries=args.retries,
                chunk_mb=args.chunk_mb,
                backend=args.backend,
            )
            print(f"[scannet-download] wrote {dst}", flush=True)
            return
        except (urllib.error.URLError, TimeoutError, OSError, subprocess.CalledProcessError) as exc:
            last_error = exc
            print(f"[scannet-download] failed attempt {attempt}: {exc}; keeping .part for resume", flush=True)
            time.sleep(min(30, 2 * attempt))

    raise SystemExit(f"Failed to download {url}: {last_error}")


if __name__ == "__main__":
    main()
