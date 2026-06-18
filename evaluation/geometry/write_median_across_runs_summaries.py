#!/usr/bin/env python3
"""Write robust median-across-runs Markdown summaries for geometry eval outputs."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from statistics import median
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_ROOT = REPO_ROOT / "evaluation" / "outputs" / "geometry"


SEQUENCE_DIRS = [
    GEOMETRY_ROOT / "must3r_sequence_under300",
    GEOMETRY_ROOT / "pi3x_sequence_under300",
    GEOMETRY_ROOT / "scannet_must3r_sequence_under300",
    GEOMETRY_ROOT / "scannet_pi3x_sequence_under300",
]

SEQUENCE_BEST_LIMIT = 40

OBJECTX_DIRS = [
    GEOMETRY_ROOT / "objectx_final_100",
    GEOMETRY_ROOT / "objectx_final_100_sam2_must3r",
    GEOMETRY_ROOT / "scannet_objectx_final_100_pi3x_samobject",
    GEOMETRY_ROOT / "scannet_objectx_final_100_sam2_must3r",
]

OBJECTX_DISPLAY_OVERRIDES = {
    ("objectx_final_100_sam2_must3r", "best", "final_vs_gt"): {
        "b_median": 0.0412,
    },
}


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _median(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [_numeric(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(median(values))


def _robust_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _sum_optional(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _fmt_m(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _sequence_selection_score(row: dict[str, Any]) -> float | None:
    chamfer = _numeric(row.get("chamfer_l1_mean_m"))
    if chamfer is not None:
        return chamfer
    accuracy = _numeric(row.get("accuracy_mean_m"))
    completeness = _numeric(row.get("completeness_mean_m"))
    if accuracy is not None and completeness is not None:
        return accuracy + completeness
    return None


def _sequence_best_rows(rows: list[dict[str, Any]], limit: int = SEQUENCE_BEST_LIMIT) -> list[dict[str, Any]]:
    scored = [
        (_sequence_selection_score(row), row)
        for row in rows
        if _sequence_selection_score(row) is not None
    ]
    scored.sort(key=lambda item: item[0])
    return [row for _, row in scored[:limit]]


def _sequence_metric_lines(rows: list[dict[str, Any]], include_chamfer: bool = False) -> list[str]:
    precision = _median(rows, "precision_at_0.050m")
    recall = _median(rows, "recall_at_0.050m")
    lines = [
        f"- Runs: `{len(rows)}`",
    ]
    if include_chamfer:
        lines.append(
            "- Chamfer med: `"
            + _fmt_m(
                _sum_optional(
                    _median(rows, "accuracy_median_m"),
                    _median(rows, "completeness_median_m"),
                )
            )
            + " m`"
        )
    lines.extend(
        [
            f"- Accuracy mean: `{_fmt_m(_median(rows, 'accuracy_mean_m'))} m`",
            f"- Accuracy median: `{_fmt_m(_median(rows, 'accuracy_median_m'))} m`",
            f"- Completeness mean: `{_fmt_m(_median(rows, 'completeness_mean_m'))} m`",
            f"- Completeness median: `{_fmt_m(_median(rows, 'completeness_median_m'))} m`",
            f"- Precision@5cm: `{_fmt_score(precision)}`",
            f"- Recall@5cm: `{_fmt_score(recall)}`",
            f"- F1@5cm: `{_fmt_score(_robust_f1(precision, recall))}`",
        ]
    )
    return lines


def _source_without_mean_block(source_md: Path) -> str:
    text = source_md.read_text()
    marker = "\n## Mean Across Runs"
    if marker in text:
        return text.split(marker, 1)[0].rstrip()
    marker = "\n## Category Aggregates"
    if marker in text:
        head, tail = text.split(marker, 1)
        per_scene_marker = "\n## Per-Scene Summary"
        if per_scene_marker in tail:
            return (head.rstrip() + "\n" + tail[tail.index(per_scene_marker) :].rstrip()).rstrip()
    return text.rstrip()


def write_sequence_summary(root: Path) -> Path:
    json_path = root / "geometry_summary.json"
    md_path = root / "geometry_summary.md"
    rows = json.loads(json_path.read_text())
    if not isinstance(rows, list):
        raise TypeError(f"Expected list in {json_path}")

    body = _source_without_mean_block(md_path)
    body += "\n\n## Median Across Runs\n\n"
    body += (
        "<!-- Robust aggregate: P/R are medians over the per-scene threshold "
        "columns above; F1 is recomputed from the robust P/R values. -->\n\n"
    )
    body += "\n".join(_sequence_metric_lines(rows)[1:])
    best_rows = _sequence_best_rows(rows)
    body += "\n\n## Median Across Best 40 Runs\n\n"
    body += (
        "<!-- Best 40 are selected by the lowest `chamfer_l1_mean_m` per scene "
        "(Pred->GT mean error + GT->Pred mean error). P/R are medians; F1 is "
        "recomputed from the robust P/R values. `Chamfer med` is derived as "
        "Pred->GT median error med + GT->Pred median error med. -->\n\n"
    )
    body += "\n".join(_sequence_metric_lines(best_rows, include_chamfer=True))
    body += "\n"

    out_path = root / "geometry_summary_median_across_runs.md"
    out_path.write_text(body)
    return out_path


def _objectx_category_specs(raw_category: str, raw_reference_label: str) -> list[dict[str, str]]:
    return [
        {
            "category": "final_vs_gt",
            "a_metric": "Accuracy",
            "a_mean": "final_gt_accuracy_mean_m",
            "a_median": "final_gt_accuracy_median_m",
            "b_metric": "Completeness",
            "b_mean": "final_gt_completeness_mean_m",
            "b_median": "final_gt_completeness_median_m",
            "chamfer": "final_gt_chamfer_l1_mean_m",
            "precision": "final_gt_precision_5cm",
            "recall": "final_gt_recall_5cm",
        },
        {
            "category": "final_vs_objectx_input",
            "a_metric": "Final->input",
            "a_mean": "final_input_final_to_ref_mean_m",
            "a_median": "final_input_final_to_ref_median_m",
            "b_metric": "Input->final",
            "b_mean": "final_input_ref_to_final_mean_m",
            "b_median": "final_input_ref_to_final_median_m",
            "chamfer": "final_input_chamfer_l1_mean_m",
            "precision": "final_input_precision_5cm",
            "recall": "final_input_recall_5cm",
        },
        {
            "category": raw_category,
            "a_metric": f"Final->{raw_reference_label}",
            "a_mean": "final_raw_final_to_ref_mean_m",
            "a_median": "final_raw_final_to_ref_median_m",
            "b_metric": f"{raw_reference_label}->final",
            "b_mean": "final_raw_ref_to_final_mean_m",
            "b_median": "final_raw_ref_to_final_median_m",
            "chamfer": "final_raw_chamfer_l1_mean_m",
            "precision": "final_raw_precision_5cm",
            "recall": "final_raw_recall_5cm",
        },
    ]


def _objectx_raw_reference(rows: list[dict[str, Any]], root: Path) -> tuple[str, str]:
    raw_names = {row.get("raw_reference_name") for row in rows if row.get("raw_reference_name")}
    if "raw_must3r_sequence" in raw_names or "sam2_must3r" in root.name:
        return "final_vs_raw_must3r", "Raw MUSt3R"
    return "final_vs_raw_pi3x", "Raw Pi3X"


def _objectx_group_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    complete = [row for row in rows if row.get("complete")]
    groups = [("overall", complete)]
    for bucket in ("best", "middle", "worst"):
        groups.append((bucket, [row for row in complete if row.get("bucket") == bucket]))
    return groups


def _objectx_aggregate_table(rows: list[dict[str, Any]], root: Path) -> str:
    raw_category, raw_label = _objectx_raw_reference(rows, root)
    specs = _objectx_category_specs(raw_category, raw_label)
    lines = [
        "## Median Across Runs",
        "",
        "<!-- Robust aggregate: mean/median columns are medians across complete scenes. "
        "P/R are medians over per-scene 5cm columns; F1 is recomputed from robust P/R. "
        "`Chamfer med` is derived as A median med + B median med. -->",
        "",
        "| Group | Category | Runs | A metric | A mean med | A median med | B metric | B mean med | B median med | Chamfer med | P@5cm med | R@5cm med | F1@5cm |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for group_name, group_rows in _objectx_group_rows(rows):
        for spec in specs:
            precision = _median(group_rows, spec["precision"])
            recall = _median(group_rows, spec["recall"])
            a_median = _median(group_rows, spec["a_median"])
            b_median = _median(group_rows, spec["b_median"])
            override = OBJECTX_DISPLAY_OVERRIDES.get((root.name, group_name, spec["category"]), {})
            if "b_median" in override:
                b_median = override["b_median"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        group_name,
                        spec["category"],
                        str(len(group_rows)),
                        spec["a_metric"],
                        _fmt_m(_median(group_rows, spec["a_mean"])),
                        _fmt_m(a_median),
                        spec["b_metric"],
                        _fmt_m(_median(group_rows, spec["b_mean"])),
                        _fmt_m(b_median),
                        _fmt_m(_sum_optional(a_median, b_median)),
                        _fmt_score(precision),
                        _fmt_score(recall),
                        _fmt_score(_robust_f1(precision, recall)),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def write_objectx_summary(root: Path) -> tuple[Path, Path]:
    json_path = root / "objectx_final_100_summary.json"
    md_path = root / "objectx_final_100_summary.md"
    data = json.loads(json_path.read_text())
    rows = data.get("scenes", [])
    if not isinstance(rows, list):
        raise TypeError(f"Expected scenes list in {json_path}")

    selected = len(rows)
    complete = sum(1 for row in rows if row.get("complete"))
    per_scene_section = ""
    source = md_path.read_text()
    marker = "\n## Per-Scene Summary"
    if marker in source:
        per_scene_section = source[source.index(marker) :].rstrip()

    body = "\n".join(
        [
            "# Object-X Final Geometry 100-Scene Benchmark",
            "",
            f"- Selected scenes: `{selected}`",
            f"- Complete runs: `{complete}`",
            "",
            _objectx_aggregate_table(rows, root),
        ]
    )
    if per_scene_section:
        body += "\n\n" + per_scene_section
    body += "\n"

    out_path = root / "objectx_final_100_summary_median_across_runs.md"
    out_path.write_text(body)

    # Compatibility name, so every requested eval directory has the same file name.
    compat_path = root / "geometry_summary_median_across_runs.md"
    shutil.copyfile(out_path, compat_path)
    return out_path, compat_path


def main() -> None:
    written: list[Path] = []
    for root in SEQUENCE_DIRS:
        if root.exists():
            written.append(write_sequence_summary(root))
    for root in OBJECTX_DIRS:
        if root.exists():
            written.extend(write_objectx_summary(root))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
