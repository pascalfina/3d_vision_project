#!/usr/bin/env python3
"""Summarize the 100-scene Object-X final-geometry benchmark.

The benchmark writes three metric folders per scene:

* final_vs_gt
* final_vs_objectx_input
* final_vs_raw_pi3x

This script collects all available runs and writes report-friendly aggregate
tables across the full 100-scene selection and across the best/middle/worst
Pi3X source-quality buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


GT_CATEGORY = "final_vs_gt"
INPUT_CATEGORY = "final_vs_objectx_input"
RAW_CATEGORY = "final_vs_raw_pi3x"
CATEGORIES = (GT_CATEGORY, INPUT_CATEGORY, RAW_CATEGORY)
GROUPS = ("overall", "best", "middle", "worst")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--report-threshold", type=float, default=0.05)
    parser.add_argument("--secondary-threshold", type=float, default=0.10)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def threshold_key(value: float) -> str:
    return f"{float(value):.3f}m"


def threshold(metrics: dict[str, Any] | None, value: float) -> dict[str, Any]:
    if not metrics:
        return {}
    thresholds = metrics.get("thresholds", {})
    key = threshold_key(value)
    if key in thresholds:
        return thresholds[key]
    return next(iter(thresholds.values())) if thresholds else {}


def gt_scope(metrics: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if not metrics:
        return "", {}
    scopes = metrics.get("scopes", {})
    for key in ("visible_gt", "pred_bbox_gt", "full_gt"):
        if key in scopes:
            return key, scopes[key]
    if scopes:
        key = sorted(scopes)[0]
        return key, scopes[key]
    return "", {}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def get_nested(data: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def add_float(row: dict[str, Any], key: str, value: Any) -> None:
    number = as_float(value)
    row[key] = number


def flatten_scene_metrics(
    selected: dict[str, str],
    *,
    out_root: Path,
    report_threshold: float,
    secondary_threshold: float,
) -> dict[str, Any]:
    profile = selected["profile"]
    scene_id = selected["scene_id"]
    base = out_root / profile / scene_id
    gt = read_json(base / GT_CATEGORY / "metrics.json")
    obj_input = read_json(base / INPUT_CATEGORY / "metrics.json")
    raw = read_json(base / RAW_CATEGORY / "metrics.json")

    scope_name, scope = gt_scope(gt)
    gt_t = threshold(scope, report_threshold)
    gt_t2 = threshold(scope, secondary_threshold)
    input_t = threshold(obj_input, report_threshold)
    input_t2 = threshold(obj_input, secondary_threshold)
    raw_t = threshold(raw, report_threshold)
    raw_t2 = threshold(raw, secondary_threshold)

    row: dict[str, Any] = {
        "bucket": selected["bucket"],
        "bucket_rank": int(selected["bucket_rank"]),
        "source_score": as_float(selected["score"]),
        "source_score_scope": selected["score_scope"],
        "scene_id": scene_id,
        "profile": profile,
        "source_metrics": selected["source_metrics"],
        "complete": bool(gt and obj_input and raw),
        "gt_scope": scope_name,
    }

    add_float(row, "final_gt_accuracy_mean_m", get_nested(gt, "pred_to_gt", "mean"))
    add_float(row, "final_gt_accuracy_median_m", get_nested(gt, "pred_to_gt", "median"))
    add_float(row, "final_gt_accuracy_p95_m", get_nested(gt, "pred_to_gt", "p95"))
    add_float(row, "final_gt_completeness_mean_m", get_nested(scope, "gt_to_pred", "mean"))
    add_float(row, "final_gt_completeness_median_m", get_nested(scope, "gt_to_pred", "median"))
    add_float(row, "final_gt_completeness_p95_m", get_nested(scope, "gt_to_pred", "p95"))
    add_float(row, "final_gt_chamfer_l1_mean_m", scope.get("chamfer_l1_mean"))
    add_float(row, "final_gt_precision_5cm", gt_t.get("precision"))
    add_float(row, "final_gt_recall_5cm", gt_t.get("recall"))
    add_float(row, "final_gt_f1_5cm", gt_t.get("fscore"))
    add_float(row, "final_gt_precision_10cm", gt_t2.get("precision"))
    add_float(row, "final_gt_recall_10cm", gt_t2.get("recall"))
    add_float(row, "final_gt_f1_10cm", gt_t2.get("fscore"))

    for prefix, metrics, t5, t10 in (
        ("final_input", obj_input, input_t, input_t2),
        ("final_raw", raw, raw_t, raw_t2),
    ):
        add_float(row, f"{prefix}_final_to_ref_mean_m", get_nested(metrics, "final_to_reference", "mean"))
        add_float(row, f"{prefix}_final_to_ref_median_m", get_nested(metrics, "final_to_reference", "median"))
        add_float(row, f"{prefix}_final_to_ref_p95_m", get_nested(metrics, "final_to_reference", "p95"))
        add_float(row, f"{prefix}_ref_to_final_mean_m", get_nested(metrics, "reference_to_final", "mean"))
        add_float(row, f"{prefix}_ref_to_final_median_m", get_nested(metrics, "reference_to_final", "median"))
        add_float(row, f"{prefix}_ref_to_final_p95_m", get_nested(metrics, "reference_to_final", "p95"))
        add_float(row, f"{prefix}_chamfer_l1_mean_m", get_nested(metrics, "chamfer_l1_mean"))
        add_float(row, f"{prefix}_precision_5cm", t5.get("precision"))
        add_float(row, f"{prefix}_recall_5cm", t5.get("recall"))
        add_float(row, f"{prefix}_f1_5cm", t5.get("fscore"))
        add_float(row, f"{prefix}_precision_10cm", t10.get("precision"))
        add_float(row, f"{prefix}_recall_10cm", t10.get("recall"))
        add_float(row, f"{prefix}_f1_10cm", t10.get("fscore"))

    return row


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def category_keys(category: str) -> tuple[str, str, str, str, str, str, str, str]:
    if category == GT_CATEGORY:
        return (
            "final_gt_accuracy_mean_m",
            "final_gt_accuracy_median_m",
            "final_gt_completeness_mean_m",
            "final_gt_completeness_median_m",
            "final_gt_chamfer_l1_mean_m",
            "final_gt_precision_5cm",
            "final_gt_recall_5cm",
            "final_gt_f1_5cm",
        )
    if category == INPUT_CATEGORY:
        return (
            "final_input_final_to_ref_mean_m",
            "final_input_final_to_ref_median_m",
            "final_input_ref_to_final_mean_m",
            "final_input_ref_to_final_median_m",
            "final_input_chamfer_l1_mean_m",
            "final_input_precision_5cm",
            "final_input_recall_5cm",
            "final_input_f1_5cm",
        )
    if category == RAW_CATEGORY:
        return (
            "final_raw_final_to_ref_mean_m",
            "final_raw_final_to_ref_median_m",
            "final_raw_ref_to_final_mean_m",
            "final_raw_ref_to_final_median_m",
            "final_raw_chamfer_l1_mean_m",
            "final_raw_precision_5cm",
            "final_raw_recall_5cm",
            "final_raw_f1_5cm",
        )
    raise ValueError(category)


def category_label(category: str) -> tuple[str, str]:
    if category == GT_CATEGORY:
        return "Accuracy", "Completeness"
    if category == INPUT_CATEGORY:
        return "Final->input", "Input->final"
    if category == RAW_CATEGORY:
        return "Final->raw", "Raw->final"
    raise ValueError(category)


def category_summary_rows(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        if group == "overall":
            group_rows = [row for row in scene_rows if row.get("complete")]
        else:
            group_rows = [
                row for row in scene_rows if row.get("complete") and row.get("bucket") == group
            ]
        for category in CATEGORIES:
            (
                a_mean_key,
                a_median_key,
                b_mean_key,
                b_median_key,
                chamfer_key,
                precision_key,
                recall_key,
                f1_key,
            ) = category_keys(category)
            a_label, b_label = category_label(category)
            row = {
                "group": group,
                "category": category,
                "runs": len(group_rows),
                "metric_a_label": a_label,
                "metric_b_label": b_label,
                "metric_a_mean_of_means_m": mean(numeric_values(group_rows, a_mean_key)),
                "metric_a_median_of_means_m": median(numeric_values(group_rows, a_mean_key)),
                "metric_a_mean_of_medians_m": mean(numeric_values(group_rows, a_median_key)),
                "metric_b_mean_of_means_m": mean(numeric_values(group_rows, b_mean_key)),
                "metric_b_median_of_means_m": median(numeric_values(group_rows, b_mean_key)),
                "metric_b_mean_of_medians_m": mean(numeric_values(group_rows, b_median_key)),
                "chamfer_l1_mean_m": mean(numeric_values(group_rows, chamfer_key)),
                "precision_5cm_mean": mean(numeric_values(group_rows, precision_key)),
                "recall_5cm_mean": mean(numeric_values(group_rows, recall_key)),
                "f1_5cm_mean": mean(numeric_values(group_rows, f1_key)),
            }
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_number(value: Any, digits: int = 4) -> str:
    text = fmt(value, digits=digits)
    return text if text else "n/a"


def write_markdown(
    path: Path,
    *,
    scene_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    complete_rows = [row for row in scene_rows if row.get("complete")]
    lines = [
        "# Object-X Final Geometry 100-Scene Benchmark",
        "",
        f"- Selected scenes: `{len(scene_rows)}`",
        f"- Complete runs: `{len(complete_rows)}`",
        "",
        "## Category Aggregates",
        "",
        "| Group | Category | Runs | A metric | A mean | A median | B metric | B mean | B median | Chamfer | P@5cm | R@5cm | F1@5cm |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["category"]),
                    str(row["runs"]),
                    str(row["metric_a_label"]),
                    md_number(row["metric_a_mean_of_means_m"]),
                    md_number(row["metric_a_median_of_means_m"]),
                    str(row["metric_b_label"]),
                    md_number(row["metric_b_mean_of_means_m"]),
                    md_number(row["metric_b_median_of_means_m"]),
                    md_number(row["chamfer_l1_mean_m"]),
                    md_number(row["precision_5cm_mean"], 3),
                    md_number(row["recall_5cm_mean"], 3),
                    md_number(row["f1_5cm_mean"], 3),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Scene Summary",
            "",
            "| Bucket | Scene | Profile | Complete | Source score | GT F1@5cm | Input F1@5cm | Raw F1@5cm | GT acc mean | GT compl mean |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in scene_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["bucket"]),
                    str(row["scene_id"]),
                    str(row["profile"]),
                    "yes" if row.get("complete") else "no",
                    md_number(row.get("source_score")),
                    md_number(row.get("final_gt_f1_5cm"), 3),
                    md_number(row.get("final_input_f1_5cm"), 3),
                    md_number(row.get("final_raw_f1_5cm"), 3),
                    md_number(row.get("final_gt_accuracy_mean_m")),
                    md_number(row.get("final_gt_completeness_mean_m")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection_file)
    out_root = Path(args.out_root)
    selected_rows: list[dict[str, str]] = []
    with selection_path.open() as f:
        selected_rows = list(csv.DictReader(f, delimiter="\t"))

    scene_rows = [
        flatten_scene_metrics(
            selected,
            out_root=out_root,
            report_threshold=args.report_threshold,
            secondary_threshold=args.secondary_threshold,
        )
        for selected in selected_rows
    ]
    summary_rows = category_summary_rows(scene_rows)

    write_csv(out_root / "objectx_final_100_summary.csv", scene_rows)
    write_csv(out_root / "objectx_final_100_category_summary.csv", summary_rows)
    (out_root / "objectx_final_100_summary.json").write_text(
        json.dumps({"scenes": scene_rows, "category_summary": summary_rows}, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        out_root / "objectx_final_100_summary.md",
        scene_rows=scene_rows,
        summary_rows=summary_rows,
    )
    print(f"[aggregate] wrote {out_root / 'objectx_final_100_summary.md'}")
    print(f"[aggregate] complete={sum(1 for row in scene_rows if row.get('complete'))}/{len(scene_rows)}")


if __name__ == "__main__":
    main()
