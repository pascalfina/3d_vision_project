#!/usr/bin/env python3

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.2f}MB"


def recompress(path: Path) -> tuple[str, int, int]:
    before = path.stat().st_size

    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}

    tmp_path = path.with_name(path.name + ".recompress_tmp.npz")

    try:
        np.savez_compressed(tmp_path, **data)
        after = tmp_path.stat().st_size

        if after < before:
            os.replace(tmp_path, path)
            status = "compressed"
        else:
            tmp_path.unlink(missing_ok=True)
            status = "kept"
        return status, before, after
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        data.clear()
        gc.collect()


def main() -> int:
    if len(sys.argv) not in {2, 4}:
        print(
            "usage: recompress_npz.py <path> [index total]",
            file=sys.stderr,
        )
        return 2

    path = Path(sys.argv[1])
    prefix = ""
    if len(sys.argv) == 4:
        prefix = f"[{sys.argv[2]}/{sys.argv[3]}] "

    status, before, after = recompress(path)
    print(
        f"{prefix}{status:<10} {path}  {_fmt_mb(before)} -> {_fmt_mb(after)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
