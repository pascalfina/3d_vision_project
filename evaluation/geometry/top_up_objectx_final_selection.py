#!/usr/bin/env python3
"""Top up an Object-X final benchmark selection with the next best source scenes.

This keeps already completed scenes, drops incomplete entries from the active
selection, and fills the selection back to the requested target size using the
best remaining Pi3X-vs-GT source-geometry runs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "bucket",
    "bucket_rank",
    "score",
    "score_scope",
    "scene_id",
    "profile",
    "source_metrics",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--source-metrics-glob", required=True)
    parser.add_argument(
        "--status-file",
        default=None,
        help="Optional benchmark status.tsv; failed scene_ids in it are excluded from top-up candidates.",
    )
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument(
        "--open-candidates",
        type=int,
        default=None,
        help=(
            "If set, keep all completed scenes and add this many fresh unfinished "
            "candidates, instead of filling to --target total rows."
        ),
    )
    parser.add_argument("--topup-bucket", default="best")
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep currently selected incomplete rows and add only the missing buffer candidates.",
    )
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def preferred_scope(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    scopes = metrics.get("scopes", {})
    for key in ("visible_gt", "pred_bbox_gt", "full_gt"):
        if key in scopes:
            return key, scopes[key]
    if scopes:
        key = sorted(scopes)[0]
        return key, scopes[key]
    return "none", {}


def score_metrics(metrics: dict[str, Any]) -> tuple[float, str]:
    scope_name, scope = preferred_scope(metrics)
    pred = metrics.get("pred_to_gt", {})
    comp = scope.get("gt_to_pred", {})
    chamfer = scope.get("chamfer_l1_mean")
    if isinstance(chamfer, (int, float)) and math.isfinite(float(chamfer)):
        return float(chamfer), scope_name
    values = [
        float(value)
        for value in (pred.get("mean"), comp.get("mean"))
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not values:
        return float("inf"), scope_name
    return sum(values) / len(values), scope_name


def read_profiles(profile_dir: Path) -> tuple[dict[str, str], set[str]]:
    profiles_by_scene: dict[str, str] = {}
    profile_names: set[str] = set()
    for path in profile_dir.glob("*.json"):
        profile = read_json(path)
        if not profile:
            continue
        name = profile.get("name") or path.stem
        scene_id = profile.get("scene_id")
        profile_names.add(name)
        if scene_id:
            profiles_by_scene[scene_id] = name
    return profiles_by_scene, profile_names


def read_selection(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def is_complete(out_root: Path, row: dict[str, str]) -> bool:
    return (out_root / row["profile"] / row["scene_id"] / "report_summary.md").exists()


def source_rows(
    *,
    metrics_glob: str,
    profile_dir: Path,
) -> list[dict[str, str]]:
    profiles_by_scene, profile_names = read_profiles(profile_dir)
    rows_by_scene: dict[str, dict[str, Any]] = {}
    for raw_path in glob.glob(metrics_glob, recursive=True):
        path = Path(raw_path)
        metrics = read_json(path)
        if not metrics:
            continue
        scene_id = metrics.get("scene_id") or path.parent.name
        method = metrics.get("method_name") or path.parent.parent.name
        profile = method if method in profile_names else profiles_by_scene.get(scene_id)
        if not profile:
            continue
        value, scope_name = score_metrics(metrics)
        if not math.isfinite(value):
            continue
        candidate = {
            "bucket": "candidate",
            "bucket_rank": "0",
            "score": f"{value:.17g}",
            "score_scope": scope_name,
            "scene_id": scene_id,
            "profile": profile,
            "source_metrics": str(path),
            "_score": value,
        }
        current = rows_by_scene.get(scene_id)
        if current is None or value < current["_score"]:
            rows_by_scene[scene_id] = candidate
    rows = list(rows_by_scene.values())
    rows.sort(key=lambda row: (row["_score"], row["scene_id"]))
    for row in rows:
        row.pop("_score", None)
    return rows


def write_selection(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in FIELDNAMES} for row in rows])


def failed_scene_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    failed: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") == "failed" and row.get("scene_id"):
                failed.add(row["scene_id"])
    return failed


def next_bucket_rank(rows: list[dict[str, str]], bucket: str) -> int:
    ranks = []
    for row in rows:
        if row.get("bucket") != bucket:
            continue
        try:
            ranks.append(int(row.get("bucket_rank", "0")))
        except ValueError:
            pass
    return max(ranks, default=0) + 1


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection_file)
    out_root = Path(args.out_root)
    profile_dir = Path(args.profile_dir)
    status_path = Path(args.status_file) if args.status_file else selection_path.parent / "status.tsv"
    failed_ids = failed_scene_ids(status_path)

    old_rows = read_selection(selection_path)
    completed_rows = [row for row in old_rows if is_complete(out_root, row)]
    incomplete_rows = [
        row
        for row in old_rows
        if not is_complete(out_root, row) and row.get("scene_id") not in failed_ids
    ]
    failed_incomplete_count = sum(
        1
        for row in old_rows
        if not is_complete(out_root, row) and row.get("scene_id") in failed_ids
    )
    kept_rows = completed_rows + (incomplete_rows if args.keep_incomplete else [])
    kept_scene_ids = {row["scene_id"] for row in kept_rows}
    old_scene_ids = {row["scene_id"] for row in old_rows}

    if args.open_candidates is not None:
        already_open = len(incomplete_rows) if args.keep_incomplete else 0
        needed = max(0, args.open_candidates - already_open)
        target_rows = len(kept_rows) + needed
    else:
        needed = max(0, args.target - len(kept_rows))
        target_rows = args.target
    candidates = source_rows(metrics_glob=args.source_metrics_glob, profile_dir=profile_dir)
    topups: list[dict[str, str]] = []
    rank = next_bucket_rank(kept_rows, args.topup_bucket)
    for candidate in candidates:
        scene_id = candidate["scene_id"]
        if scene_id in kept_scene_ids:
            continue
        if scene_id in failed_ids:
            continue
        if not args.keep_incomplete and scene_id in old_scene_ids:
            continue
        candidate = dict(candidate)
        candidate["bucket"] = args.topup_bucket
        candidate["bucket_rank"] = str(rank)
        topups.append(candidate)
        rank += 1
        if len(topups) >= needed:
            break

    new_rows = kept_rows + topups
    print(
        f"[topup] previous_selection={len(old_rows)} complete_kept={len(completed_rows)} "
        f"incomplete_kept={len(incomplete_rows) if args.keep_incomplete else 0} "
        f"failed_incomplete_dropped={failed_incomplete_count}"
    )
    print(f"[topup] excluded_failed_from_status={len(failed_ids)} status_file={status_path}")
    print(f"[topup] target_rows={target_rows} needed={needed} added={len(topups)}")
    if len(new_rows) < target_rows:
        raise SystemExit(
            f"Only found {len(topups)} top-up candidates; selection would have {len(new_rows)}/{target_rows}"
        )

    for row in topups[:10]:
        print(
            "[topup] add "
            f"bucket={row['bucket']} rank={row['bucket_rank']} "
            f"score={row['score']} scene={row['scene_id']} profile={row['profile']}"
        )
    if len(topups) > 10:
        print(f"[topup] ... {len(topups) - 10} more")

    if args.dry_run:
        return

    if selection_path.exists() and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = selection_path.with_name(f"{selection_path.name}.backup-{stamp}")
        shutil.copy2(selection_path, backup)
        print(f"[topup] backup={backup}")
    write_selection(selection_path, new_rows[:target_rows])
    print(f"[topup] wrote={selection_path}")


if __name__ == "__main__":
    main()
