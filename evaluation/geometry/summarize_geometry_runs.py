#!/usr/bin/env python3
"""Aggregate per-scene geometry metrics into report-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-glob",
        default="evaluation/outputs/geometry/**/metrics.json",
        help="Glob for per-run metrics.json files.",
    )
    parser.add_argument(
        "--out-dir",
        default="evaluation/outputs/geometry",
        help="Directory for aggregate geometry_summary.* outputs.",
    )
    parser.add_argument(
        "--scope",
        default="visible_gt",
        help="Preferred GT scope for aggregate rows; falls back to pred_bbox_gt/full_gt.",
    )
    parser.add_argument("--report-threshold", type=float, default=0.05)
    return parser.parse_args()


def threshold_key(threshold: float) -> str:
    return f"{float(threshold):.3f}m"


def preferred_scope(metrics: dict, requested: str) -> str:
    scopes = metrics.get("scopes", {})
    for candidate in (requested, "visible_gt", "pred_bbox_gt", "full_gt"):
        if candidate in scopes:
            return candidate
    if not scopes:
        raise ValueError("metrics file has no scopes")
    return sorted(scopes)[0]


def primary_threshold_payload(thresholds: dict, report_threshold: float) -> tuple[str, dict]:
    preferred = threshold_key(report_threshold)
    if preferred in thresholds:
        return preferred, thresholds[preferred]
    if not thresholds:
        return preferred, {"precision": None, "recall": None, "fscore": None}
    return min(
        thresholds.items(),
        key=lambda item: abs(float(item[1].get("threshold_m", report_threshold)) - report_threshold),
    )


def load_row(path: Path, *, requested_scope: str, report_threshold: float) -> dict:
    metrics = json.loads(path.read_text())
    scope = preferred_scope(metrics, requested_scope)
    payload = metrics["scopes"][scope]
    threshold_name, threshold = primary_threshold_payload(payload.get("thresholds", {}), report_threshold)
    alignment = metrics.get("alignment", {})
    rgbd_corr = alignment.get("rgbd_correspondences", {})
    rgbd_fit = alignment.get("rgbd_fit", {})
    pred = metrics["pred_to_gt"]
    comp = payload["gt_to_pred"]
    return {
        "metrics_path": str(path),
        "scene_id": metrics.get("scene_id", path.parent.name),
        "method_name": metrics.get("method_name", path.parent.parent.name),
        "scope": scope,
        "accuracy_mean_m": pred.get("mean"),
        "accuracy_median_m": pred.get("median"),
        "accuracy_p95_m": pred.get("p95"),
        "completeness_mean_m": comp.get("mean"),
        "completeness_median_m": comp.get("median"),
        "completeness_p95_m": comp.get("p95"),
        "chamfer_l1_mean_m": payload.get("chamfer_l1_mean"),
        f"precision_at_{threshold_name}": threshold.get("precision"),
        f"recall_at_{threshold_name}": threshold.get("recall"),
        f"fscore_at_{threshold_name}": threshold.get("fscore"),
        "report_threshold_m": threshold.get("threshold_m", report_threshold),
        "pred_points_eval": metrics.get("counts", {}).get("pred_points_eval"),
        "gt_points_scope": payload.get("gt_points"),
        "alignment_mode": alignment.get("mode"),
        "alignment_scale": alignment.get("effective_uniform_scale", rgbd_fit.get("scale")),
        "alignment_translation_norm_m": alignment.get("translation_norm"),
        "rgbd_frames_used": rgbd_corr.get("frames_used"),
        "rgbd_correspondences_raw": rgbd_corr.get("correspondences_raw"),
        "rgbd_fit_residual_median_m": rgbd_fit.get("residual_median"),
        "rgbd_fit_residual_p95_m": rgbd_fit.get("residual_p95"),
    }


def numeric_mean(rows: list[dict], key: str):
    values = [row.get(key) for row in rows if isinstance(row.get(key), int | float)]
    return mean(values) if values else None


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], *, report_threshold: float) -> None:
    threshold_name = threshold_key(report_threshold)
    threshold_label = f"@{int(round(report_threshold * 100))}cm"
    f_key = f"fscore_at_{threshold_name}"
    p_key = f"precision_at_{threshold_name}"
    r_key = f"recall_at_{threshold_name}"
    lines = [
        "# Geometry Evaluation Aggregate",
        "",
        f"- Runs: `{len(rows)}`",
        f"- Primary threshold: `{threshold_name}`",
        "",
        f"| Scene | Method | Scope | Acc mean | Acc med | Compl mean | Compl med | P{threshold_label} | R{threshold_label} | F1{threshold_label} |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scene_id")),
                    str(row.get("method_name")),
                    str(row.get("scope")),
                    fmt(row.get("accuracy_mean_m")),
                    fmt(row.get("accuracy_median_m")),
                    fmt(row.get("completeness_mean_m")),
                    fmt(row.get("completeness_median_m")),
                    fmt(row.get(p_key), digits=3),
                    fmt(row.get(r_key), digits=3),
                    fmt(row.get(f_key), digits=3),
                ]
            )
            + " |"
        )
    if rows:
        lines.extend(
            [
                "",
                "## Mean Across Runs",
                "",
                f"- Accuracy mean: `{fmt(numeric_mean(rows, 'accuracy_mean_m'))} m`",
                f"- Accuracy median: `{fmt(numeric_mean(rows, 'accuracy_median_m'))} m`",
                f"- Completeness mean: `{fmt(numeric_mean(rows, 'completeness_mean_m'))} m`",
                f"- Completeness median: `{fmt(numeric_mean(rows, 'completeness_median_m'))} m`",
                f"- Precision{threshold_label}: `{fmt(numeric_mean(rows, p_key), digits=3)}`",
                f"- Recall{threshold_label}: `{fmt(numeric_mean(rows, r_key), digits=3)}`",
                f"- F1{threshold_label}: `{fmt(numeric_mean(rows, f_key), digits=3)}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = sorted(Path().glob(args.metrics_glob))
    rows = [
        load_row(path, requested_scope=args.scope, report_threshold=args.report_threshold)
        for path in paths
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "geometry_summary.csv", rows)
    write_markdown(out_dir / "geometry_summary.md", rows, report_threshold=args.report_threshold)
    (out_dir / "geometry_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[geometry-summary] runs={len(rows)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
