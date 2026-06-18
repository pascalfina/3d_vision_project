#!/usr/bin/env python3
import argparse
import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--vlsG-space", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--tmp-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--scene-source-dirname", default="scenes")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--max-scans", type=int, default=0)
    parser.add_argument("--override", action="store_true")
    return parser.parse_args()


def safe_unlink(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _clean_pythonpath_for_vlsg(env: dict, repo_root: Path, vlsg_root: Path) -> None:
    """Ensure VLSG imports resolve to the dependency checkout, not this repo.

    The Object-X activation script prepends the repo root to PYTHONPATH. That
    causes top-level imports like ``configs`` or ``utils`` inside the VLSG
    feature-generation scripts to resolve to Object-X modules first. For the
    subprocess that runs VLSG code we instead want:
    1. VLSG workspace
    2. VLSG src
    3. the remaining non-Object-X paths
    """

    repo_real = os.path.realpath(str(repo_root))
    repo_src_real = os.path.realpath(str(repo_root / "src"))
    existing = [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p
    ]
    cleaned = []
    seen = set()
    for path in existing:
        real = os.path.realpath(path)
        if real in {repo_real, repo_src_real}:
            continue
        if real in seen:
            continue
        seen.add(real)
        cleaned.append(path)

    preferred = [str(vlsg_root), str(vlsg_root / "src")]
    env["PYTHONPATH"] = os.pathsep.join(preferred + cleaned)


def _write_numpy_pickle_compat(tmp_root: Path) -> Path:
    """Provide NumPy-2 pickle aliases when the runtime still has NumPy 1.x.

    Some SAMObject/Object-X mask pickle files are written by environments whose
    NumPy serializes arrays via ``numpy._core``.  The VLSG Feature3D subprocess
    currently runs with an older NumPy that exposes the same implementation as
    ``numpy.core``.  A tiny sitecustomize module is the least invasive place to
    bridge that import name before pickle.load() runs inside the dependency.
    """

    compat_dir = tmp_root / "_python_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    (compat_dir / "sitecustomize.py").write_text(
        """
import importlib
import sys

try:
    import numpy as _np

    try:
        import numpy._core  # noqa: F401
    except Exception:
        _core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", _core)
        setattr(_np, "_core", _core)
        for _name in (
            "multiarray",
            "_multiarray_umath",
            "numeric",
            "fromnumeric",
            "umath",
            "shape_base",
            "_methods",
            "records",
            "overrides",
            "function_base",
        ):
            try:
                _mod = importlib.import_module(f"numpy.core.{_name}")
                sys.modules.setdefault(f"numpy._core.{_name}", _mod)
            except Exception:
                pass
except Exception:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    return compat_dir


def load_scan_ids(root: Path, split: str) -> list[str]:
    preferred = root / "files" / f"{split}_resplit_scans.txt"
    fallback = root / "files" / f"{split}_scans.txt"
    path = preferred if preferred.exists() else fallback
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def output_exists(scratch_root: Path, scan_id: str) -> bool:
    out_dir = scratch_root / "files" / "Features3D" / "obj_dinov2_top10_l3"
    plain = out_dir / f"{scan_id}.pkl"
    gz = out_dir / f"{scan_id}.pkl.gz"
    for candidate in (plain, gz):
        if candidate.exists() and candidate.stat().st_size > 128:
            return True
    return False


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


def cleanup_scan(tmp_scan_dir: Path):
    if tmp_scan_dir.exists():
        shutil.rmtree(tmp_scan_dir)


def prepare_tmp_layout(scratch_root: Path, tmp_root: Path):
    (tmp_root / "scenes").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files" / "orig").mkdir(parents=True, exist_ok=True)
    (tmp_root / "files" / "Features3D" / "obj_dinov2_top10_l3").mkdir(
        parents=True, exist_ok=True
    )

    for name in ["3RScan.json"]:
        src = scratch_root / "files" / name
        dst = tmp_root / "files" / name
        safe_unlink(dst)
        os.symlink(src, dst)

    mask_dirname = resolve_mask_source()
    src = scratch_root / "files" / mask_dirname
    if not src.exists():
        raise FileNotFoundError(
            f"Mask source directory does not exist: {src} "
            f"(OBJECTX_MASK_SOURCE={mask_dirname})"
        )

    actual_dst = tmp_root / "files" / mask_dirname
    safe_unlink(actual_dst)
    os.symlink(src, actual_dst)

    alias_dst = tmp_root / "files" / "gt_projection"
    if alias_dst != actual_dst:
        safe_unlink(alias_dst)
        os.symlink(src, alias_dst)
    print(f"[feat3d] using mask source {mask_dirname}", flush=True)


def write_single_scan_split(tmp_root: Path, split: str, scan_id: str):
    split_file = tmp_root / "files" / "orig" / f"{split}_resplit_scans.txt"
    split_file.write_text(scan_id + "\n")


def compress_to_scratch(tmp_file: Path, scratch_root: Path, scan_id: str):
    out_dir = scratch_root / "files" / "Features3D" / "obj_dinov2_top10_l3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{scan_id}.pkl.gz"
    tmp_out = out_dir / f"{scan_id}.pkl.gz.tmp"
    with open(tmp_file, "rb") as f_in, gzip.open(tmp_out, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.replace(tmp_out, out_file)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    vlsG_space = Path(args.vlsG_space)
    scratch_root = Path(args.scratch_root)
    tmp_root = Path(args.tmp_root)

    prepare_tmp_layout(scratch_root, tmp_root)

    if args.scene_id:
        scan_ids = [args.scene_id]
    else:
        scan_ids = load_scan_ids(scratch_root, args.split)
        if args.max_scans > 0:
            scan_ids = scan_ids[: args.max_scans]

    total = len(scan_ids)
    for idx, scan_id in enumerate(scan_ids, start=1):
        if output_exists(scratch_root, scan_id) and not args.override:
            if idx % 10 == 0 or idx == total:
                print(f"[feat3d] {idx}/{total} (skip existing) {scan_id}", flush=True)
            continue

        src_scan_dir = scratch_root / args.scene_source_dirname / scan_id
        tmp_scan_dir = tmp_root / "scenes" / scan_id
        tmp_out_file = (
            tmp_root / "files" / "Features3D" / "obj_dinov2_top10_l3" / f"{scan_id}.pkl"
        )

        print(f"[feat3d] {idx}/{total} {args.split} {scan_id}", flush=True)
        write_single_scan_split(tmp_root, args.split, scan_id)
        stage_scan(src_scan_dir, tmp_scan_dir)

        env = os.environ.copy()
        env["Data_ROOT_DIR"] = str(tmp_root)
        env["VLSG_SPACE"] = str(vlsG_space)
        _clean_pythonpath_for_vlsg(env, repo_root, vlsG_space)
        compat_dir = _write_numpy_pickle_compat(tmp_root)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(compat_dir), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(
                        vlsG_space
                        / "preprocessing/sg_features/obj_visual_embeddings/Dinov2/gen_obj_visual_embeddings.py"
                    ),
                    "--config",
                    str(Path(args.config)),
                    "--split",
                    args.split,
                ],
                cwd=str(vlsG_space),
                env=env,
                check=True,
            )
            compress_to_scratch(tmp_out_file, scratch_root, scan_id)
        finally:
            cleanup_scan(tmp_scan_dir)
            if tmp_out_file.exists():
                tmp_out_file.unlink()

    print(f"[done] tmp root kept at {tmp_root}", flush=True)


if __name__ == "__main__":
    main()
