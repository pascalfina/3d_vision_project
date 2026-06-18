#!/usr/bin/env python3
"""Compare two point-cloud geometry sources in one coordinate frame.

This is used for Object-X final-output evaluation: the decoded U3DGS scene PLY
is compared against the geometry that Object-X received as input, typically the
voxelized per-object `files/gs_annotations/<scene>/<obj>/voxel_output_dense.npz`
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_geometry_against_gt import (  # noqa: E402
    apply_transform,
    bbox,
    centroid_transform,
    deterministic_subsample,
    distance_summary,
    nearest_distances,
    refine_icp_transform,
    threshold_key,
    threshold_metrics,
    voxel_downsample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-ply", required=True, help="Final Object-X PLY, e.g. vis/<scan>_joint.ply.")
    ref = parser.add_mutually_exclusive_group(required=True)
    ref.add_argument("--reference-ply", help="Reference point-cloud PLY.")
    ref.add_argument(
        "--reference-sequence-dir",
        help="Reference Pi3X sequence dir with frame-*.xyz.npy and frame-*.pose.txt.",
    )
    ref.add_argument(
        "--reference-objectx-root",
        help="Object-X data root containing files/gs_annotations/<scene_id>/...",
    )
    ref.add_argument(
        "--reference-gs-dir",
        help="Direct gs_annotations directory. May point either at files/gs_annotations or at files/gs_annotations/<scene_id>.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--method-name", default="objectx_final")
    parser.add_argument("--reference-name", default="objectx_input")
    parser.add_argument("--object-id", action="append", type=int, default=[], help="Limit gs_annotations reference to selected object ids.")
    parser.add_argument("--pred-voxel-size", type=float, default=0.03)
    parser.add_argument("--reference-voxel-size", type=float, default=0.03)
    parser.add_argument("--max-pred-points", type=int, default=600000)
    parser.add_argument("--max-reference-points", type=int, default=600000)
    parser.add_argument("--sequence-conf-thr", type=float, default=0.10)
    parser.add_argument("--sequence-pixel-stride", type=int, default=2)
    parser.add_argument("--sequence-max-frames", type=int, default=0)
    parser.add_argument(
        "--pred-opacity-min",
        type=float,
        default=0.0,
        help="Optional sigmoid(opacity) minimum for Gaussian PLY vertices. 0 disables filtering.",
    )
    parser.add_argument(
        "--pred-opacity-quantile",
        type=float,
        default=0.0,
        help="Optional percentile of sigmoid(opacity) to keep from the final PLY. 0 disables filtering.",
    )
    parser.add_argument(
        "--align",
        choices=["none", "centroid", "icp", "centroid_icp"],
        default="none",
        help="Default none is correct when final output and input geometry share Object-X world coordinates.",
    )
    parser.add_argument("--icp-sample-points", type=int, default=50000)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--icp-trim-quantile", type=float, default=85.0)
    parser.add_argument("--icp-max-correspondence", type=float, default=0.0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.10])
    parser.add_argument("--report-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--write-debug-html", action="store_true")
    parser.add_argument("--debug-html-max-points", type=int, default=250000)
    return parser.parse_args()


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_ply_points_with_filters(
    path: Path,
    *,
    opacity_min: float,
    opacity_quantile: float,
) -> tuple[np.ndarray, dict]:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    valid = np.isfinite(points).all(axis=1)
    stats: dict[str, object] = {
        "path": str(path),
        "points_raw": int(len(points)),
        "has_opacity": "opacity" in names,
        "opacity_min": float(opacity_min),
        "opacity_quantile": float(opacity_quantile),
    }

    if "opacity" in names and (opacity_min > 0.0 or opacity_quantile > 0.0):
        opacities = sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
        valid &= np.isfinite(opacities)
        if opacity_min > 0.0:
            valid &= opacities >= float(opacity_min)
        if opacity_quantile > 0.0 and valid.any():
            q = float(np.percentile(opacities[valid], float(opacity_quantile)))
            valid &= opacities >= q
            stats["opacity_quantile_value"] = q

    points = points[valid]
    stats["points_after_filter"] = int(len(points))
    if len(points) == 0:
        raise ValueError(f"No valid vertices after filtering in {path}")
    return points, stats


def voxel_to_world(voxel_indices: np.ndarray, mean: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    voxel = voxel_indices.astype(np.float32) * (1.0 / 64.0)
    voxel = voxel * 2.0 - 1.0
    scale_arr = np.asarray(scale, dtype=np.float32)
    if scale_arr.ndim == 0 or scale_arr.size == 1:
        return voxel * float(scale_arr.reshape(-1)[0]) + mean[None, :]
    return voxel * scale_arr.reshape(1, 3) + mean[None, :]


def resolve_gs_scene_dir(path: Path, scene_id: str) -> Path:
    candidates = [
        path / "files" / "gs_annotations" / scene_id,
        path / scene_id,
        path,
    ]
    for candidate in candidates:
        if candidate.name == scene_id and candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve gs_annotations scene dir for {scene_id} from {path}"
    )


def load_objectx_gs_points(
    root_or_dir: Path,
    *,
    scene_id: str,
    object_ids: list[int],
) -> tuple[np.ndarray, dict]:
    scene_dir = resolve_gs_scene_dir(root_or_dir, scene_id)
    if object_ids:
        obj_dirs = [scene_dir / str(obj_id) for obj_id in object_ids]
    else:
        obj_dirs = sorted(
            [path for path in scene_dir.iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda path: int(path.name),
        )

    chunks = []
    loaded = []
    skipped = []
    for obj_dir in obj_dirs:
        voxel_path = obj_dir / "voxel_output_dense.npz"
        mean_scale_path = obj_dir / "mean_scale_dense.npz"
        if not voxel_path.exists() or not mean_scale_path.exists():
            skipped.append(int(obj_dir.name) if obj_dir.name.isdigit() else obj_dir.name)
            continue
        voxels = np.load(voxel_path)["arr_0"][:, :3].astype(np.int32)
        mean_scale = np.load(mean_scale_path)
        mean = mean_scale["mean"].astype(np.float32)
        scale = mean_scale["scale"].astype(np.float32)
        points = voxel_to_world(voxels, mean, scale)
        if len(points) == 0:
            skipped.append(int(obj_dir.name))
            continue
        chunks.append(points.astype(np.float32))
        loaded.append(int(obj_dir.name))

    if not chunks:
        raise FileNotFoundError(f"No usable voxelized Object-X input objects in {scene_dir}")

    points = np.concatenate(chunks, axis=0).astype(np.float32)
    stats = {
        "type": "objectx_gs_annotations",
        "scene_dir": str(scene_dir),
        "objects_loaded": loaded,
        "objects_skipped": skipped,
        "points_raw": int(len(points)),
    }
    return points, stats


def load_reference_points(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    if args.reference_ply:
        points, stats = load_ply_points_with_filters(
            Path(args.reference_ply),
            opacity_min=0.0,
            opacity_quantile=0.0,
        )
        stats["type"] = "ply"
        return points, stats

    if args.reference_sequence_dir:
        from evaluate_geometry_against_gt import load_sequence_points, parse_axis_signs

        points = load_sequence_points(
            Path(args.reference_sequence_dir),
            conf_thr=args.sequence_conf_thr,
            pixel_stride=args.sequence_pixel_stride,
            max_frames=args.sequence_max_frames,
            camera_axis_signs=parse_axis_signs("1,1,1"),
        )
        return points, {
            "type": "sequence",
            "path": str(Path(args.reference_sequence_dir)),
            "conf_thr": float(args.sequence_conf_thr),
            "pixel_stride": int(args.sequence_pixel_stride),
            "max_frames": int(args.sequence_max_frames),
            "points_raw": int(len(points)),
        }

    if args.reference_objectx_root:
        return load_objectx_gs_points(
            Path(args.reference_objectx_root),
            scene_id=args.scene_id,
            object_ids=args.object_id,
        )

    if args.reference_gs_dir:
        return load_objectx_gs_points(
            Path(args.reference_gs_dir),
            scene_id=args.scene_id,
            object_ids=args.object_id,
        )

    raise AssertionError("unreachable: reference source required by argparse")


def maybe_align_points(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    mode: str,
    seed: int,
    sample_points: int,
    iterations: int,
    trim_quantile: float,
    max_correspondence: float,
) -> tuple[np.ndarray, dict]:
    stats: dict[str, object] = {"mode": mode}
    transform = np.eye(4, dtype=np.float64)
    if mode == "none":
        return pred, {**stats, "transform_pred_to_reference": transform.tolist(), "translation_norm": 0.0}
    if mode in {"centroid", "centroid_icp"}:
        transform = centroid_transform(pred.astype(np.float64), ref.astype(np.float64))
        stats["centroid_transform"] = transform.round(9).tolist()
    if mode in {"icp", "centroid_icp"}:
        transform, icp_stats = refine_icp_transform(
            pred,
            ref,
            initial_transform=None if mode == "icp" else transform,
            seed=seed,
            sample_points=sample_points,
            iterations=iterations,
            trim_quantile=trim_quantile,
            max_correspondence=max_correspondence,
        )
        stats["icp"] = icp_stats
    aligned = apply_transform(pred.astype(np.float64), transform).astype(np.float32)
    stats.update(
        {
            "transform_pred_to_reference": transform.round(9).tolist(),
            "translation_norm": float(np.linalg.norm(transform[:3, 3])),
        }
    )
    return aligned, stats


def write_csv(path: Path, metrics: dict, report_threshold: float) -> None:
    key = threshold_key(report_threshold)
    threshold = metrics["thresholds"].get(key)
    if threshold is None:
        threshold = next(iter(metrics["thresholds"].values()))
        key = threshold_key(threshold["threshold_m"])
    row = {
        "scene_id": metrics["scene_id"],
        "method_name": metrics["method_name"],
        "reference_name": metrics["reference_name"],
        "final_to_reference_mean_m": metrics["final_to_reference"]["mean"],
        "final_to_reference_median_m": metrics["final_to_reference"]["median"],
        "final_to_reference_p95_m": metrics["final_to_reference"]["p95"],
        "reference_to_final_mean_m": metrics["reference_to_final"]["mean"],
        "reference_to_final_median_m": metrics["reference_to_final"]["median"],
        "reference_to_final_p95_m": metrics["reference_to_final"]["p95"],
        "chamfer_l1_mean_m": metrics["chamfer_l1_mean"],
        f"precision_at_{key}": threshold["precision"],
        f"recall_at_{key}": threshold["recall"],
        f"fscore_at_{key}": threshold["fscore"],
        "report_threshold_m": threshold["threshold_m"],
        "alignment_mode": metrics["alignment"]["mode"],
    }
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, metrics: dict, report_threshold: float) -> None:
    key = threshold_key(report_threshold)
    threshold = metrics["thresholds"].get(key)
    if threshold is None:
        threshold = next(iter(metrics["thresholds"].values()))
    lines = [
        "# Object-X Final Geometry vs Input Geometry",
        "",
        f"- Scene: `{metrics['scene_id']}`",
        f"- Method: `{metrics['method_name']}`",
        f"- Reference: `{metrics['reference_name']}`",
        f"- Alignment: `{metrics['alignment']['mode']}`",
        "",
        "| Direction | Mean | Median | P95 |",
        "| --- | ---: | ---: | ---: |",
        "| Final -> reference | "
        + " | ".join(
            [
                fmt(metrics["final_to_reference"]["mean"]),
                fmt(metrics["final_to_reference"]["median"]),
                fmt(metrics["final_to_reference"]["p95"]),
            ]
        )
        + " |",
        "| Reference -> final | "
        + " | ".join(
            [
                fmt(metrics["reference_to_final"]["mean"]),
                fmt(metrics["reference_to_final"]["median"]),
                fmt(metrics["reference_to_final"]["p95"]),
            ]
        )
        + " |",
        "",
        f"- Chamfer-L1 mean: `{fmt(metrics['chamfer_l1_mean'])} m`",
        f"- Precision@{int(round(threshold['threshold_m'] * 100))}cm: `{fmt(threshold['precision'], 3)}`",
        f"- Recall@{int(round(threshold['threshold_m'] * 100))}cm: `{fmt(threshold['recall'], 3)}`",
        f"- F1@{int(round(threshold['threshold_m'] * 100))}cm: `{fmt(threshold['fscore'], 3)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overlay_html(
    path: Path,
    final_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    max_points_per_cloud: int,
    title: str,
    details: dict[str, object],
) -> None:
    import trimesh
    import trimesh.viewer

    path.parent.mkdir(parents=True, exist_ok=True)
    final_vis = deterministic_subsample(final_points, max_points_per_cloud, seed=701)
    ref_vis = deterministic_subsample(reference_points, max_points_per_cloud, seed=702)
    final_colors = np.tile(np.array([[255, 95, 35, 210]], dtype=np.uint8), (len(final_vis), 1))
    ref_colors = np.tile(np.array([[35, 120, 255, 185]], dtype=np.uint8), (len(ref_vis), 1))

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.points.PointCloud(ref_vis, colors=ref_colors), geom_name="Object-X input reference (blue)")
    scene.add_geometry(trimesh.points.PointCloud(final_vis, colors=final_colors), geom_name="Final Object-X output (orange)")
    bounds = scene.bounds
    if bounds is not None and np.isfinite(bounds).all():
        center = bounds.mean(axis=0)
        distance = max(float(np.max(bounds[1] - bounds[0])) * 1.7, 1.0)
        scene.set_camera(angles=(0.75, 0.0, 0.65), distance=distance, center=center)

    html_text = trimesh.viewer.scene_to_html(scene)
    detail_rows = [
        f'<div class="meta"><b>{html.escape(str(key))}:</b> {html.escape(str(value))}</div>'
        for key, value in details.items()
    ]
    legend = f"""
<style>
.objx-overlay-legend {{
  position: fixed;
  top: 16px;
  left: 16px;
  z-index: 9999;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(255,255,255,0.92);
  color: #17202a;
  font: 13px/1.35 sans-serif;
  box-shadow: 0 8px 28px rgba(0,0,0,0.18);
}}
.objx-overlay-legend .title {{ font-weight: 700; margin-bottom: 6px; }}
.objx-overlay-legend .row {{ display: flex; align-items: center; gap: 8px; }}
.objx-overlay-legend .meta {{ max-width: 560px; overflow-wrap: anywhere; margin-top: 3px; }}
.objx-overlay-legend .swatch {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
</style>
<div class="objx-overlay-legend">
  <div class="title">{html.escape(title)}</div>
  <div class="row"><span class="swatch" style="background:#2378ff"></span>Object-X input reference</div>
  <div class="row"><span class="swatch" style="background:#ff5f23"></span>Final Object-X output</div>
  <div>points: reference {len(ref_vis):,}, final {len(final_vis):,}</div>
  {' '.join(detail_rows)}
</div>
"""
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", legend + "\n</body>")
    else:
        html_text += legend
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(html_text, encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_points, pred_source = load_ply_points_with_filters(
        Path(args.pred_ply),
        opacity_min=args.pred_opacity_min,
        opacity_quantile=args.pred_opacity_quantile,
    )
    ref_points, ref_source = load_reference_points(args)

    pred_count_raw = int(len(pred_points))
    ref_count_raw = int(len(ref_points))
    pred_points = voxel_downsample(pred_points, args.pred_voxel_size)
    ref_points = voxel_downsample(ref_points, args.reference_voxel_size)
    pred_points = deterministic_subsample(pred_points, args.max_pred_points, args.seed)
    ref_points = deterministic_subsample(ref_points, args.max_reference_points, args.seed + 1)

    pred_points, alignment = maybe_align_points(
        pred_points,
        ref_points,
        mode=args.align,
        seed=args.seed,
        sample_points=args.icp_sample_points,
        iterations=args.icp_iterations,
        trim_quantile=args.icp_trim_quantile,
        max_correspondence=args.icp_max_correspondence,
    )

    final_to_ref = nearest_distances(pred_points, ref_points)
    ref_to_final = nearest_distances(ref_points, pred_points)
    thresholds = threshold_metrics(final_to_ref, ref_to_final, args.thresholds)
    metrics = {
        "scene_id": args.scene_id,
        "method_name": args.method_name,
        "reference_name": args.reference_name,
        "pred_source": pred_source,
        "reference_source": ref_source,
        "counts": {
            "final_points_raw": pred_count_raw,
            "reference_points_raw": ref_count_raw,
            "final_points_eval": int(len(pred_points)),
            "reference_points_eval": int(len(ref_points)),
        },
        "options": {
            "pred_voxel_size": float(args.pred_voxel_size),
            "reference_voxel_size": float(args.reference_voxel_size),
            "max_pred_points": int(args.max_pred_points),
            "max_reference_points": int(args.max_reference_points),
            "thresholds": [float(x) for x in args.thresholds],
            "report_threshold": float(args.report_threshold),
        },
        "alignment": alignment,
        "bboxes": {
            "final_eval": bbox(pred_points),
            "reference_eval": bbox(ref_points),
        },
        "final_to_reference": distance_summary(final_to_ref),
        "reference_to_final": distance_summary(ref_to_final),
        "chamfer_l1_mean": float(np.mean(final_to_ref) + np.mean(ref_to_final)) / 2.0,
        "thresholds": thresholds,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(out_dir / "metrics.csv", metrics, args.report_threshold)
    write_markdown(out_dir / "report_summary.md", metrics, args.report_threshold)

    if args.write_debug_html:
        report_key = threshold_key(args.report_threshold)
        report = thresholds.get(report_key) or next(iter(thresholds.values()))
        write_overlay_html(
            out_dir / "overlay_final_reference.html",
            pred_points,
            ref_points,
            max_points_per_cloud=args.debug_html_max_points,
            title=f"{args.scene_id} | {args.method_name} vs {args.reference_name}",
            details={
                "alignment": args.align,
                "final_to_reference_median_m": fmt(metrics["final_to_reference"]["median"]),
                "reference_to_final_median_m": fmt(metrics["reference_to_final"]["median"]),
                f"f1@{int(round(report['threshold_m'] * 100))}cm": fmt(report["fscore"], 3),
            },
        )

    print(f"[pc-eval] wrote {out_dir / 'metrics.json'}")
    print(
        "[pc-eval] final->reference mean/median/p95: "
        f"{metrics['final_to_reference']['mean']:.4f} / "
        f"{metrics['final_to_reference']['median']:.4f} / "
        f"{metrics['final_to_reference']['p95']:.4f} m"
    )
    print(
        "[pc-eval] reference->final mean/median/p95: "
        f"{metrics['reference_to_final']['mean']:.4f} / "
        f"{metrics['reference_to_final']['median']:.4f} / "
        f"{metrics['reference_to_final']['p95']:.4f} m"
    )


if __name__ == "__main__":
    main()
