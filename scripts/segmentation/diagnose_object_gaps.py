#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.voxel_anno import voxelise_features as vf
from utils import scan3r


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether object-level gaps are mainly caused by missing "
            "visible information, poor lifting, or frame fusion loss."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Predseg/object-level dataset root to analyze.",
    )
    parser.add_argument(
        "--baseline-root",
        required=True,
        help="Baseline root that provides GT gs_annotations/mean_scale.",
    )
    parser.add_argument(
        "--selection-file",
        default=str(
            REPO_ROOT / "scripts" / "segmentation" / "object_level_pilot_selection.json"
        ),
        help="JSON file listing objects as scan_id/obj_id pairs.",
    )
    parser.add_argument(
        "--mask-source",
        default="pred_projection_clean",
        help="Mask source under files/, e.g. pred_projection_clean.",
    )
    parser.add_argument(
        "--frame-selection",
        default="diverse_area",
        choices=["all", "top_area", "diverse_area"],
        help="Frame selection mode used before lifting/fusion.",
    )
    parser.add_argument(
        "--object-source",
        default="tsdf_masks",
        choices=["lifted_masks", "tsdf_masks", "hybrid_masks"],
        help="Which object-construction path to evaluate as the fused result.",
    )
    parser.add_argument(
        "--k",
        action="append",
        type=int,
        dest="ks",
        default=None,
        help="Number of frames to keep. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for metrics and plots.",
    )
    return parser.parse_args()


def ensure_list(values: Optional[Iterable[int]], default: list[int]) -> list[int]:
    if not values:
        return default
    uniq = []
    seen = set()
    for value in values:
        if value not in seen:
            uniq.append(int(value))
            seen.add(int(value))
    return uniq


def load_objects(selection_file: Path) -> list[dict]:
    payload = json.loads(selection_file.read_text())
    return payload["objects"]


def safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def voxel_set(voxels: np.ndarray) -> set[tuple[int, int, int]]:
    if voxels.size == 0:
        return set()
    return {tuple(map(int, row[:3])) for row in np.asarray(voxels)}


def summarize_cause(metric: dict) -> str:
    visible = metric["gt_visible_ratio"]
    pre_visible = metric["pre_visible_recall"]
    fused_visible = metric["fused_visible_recall"]
    if visible < 0.35:
        return "missing_information"
    if pre_visible < 0.45:
        return "lifting_or_mask_depth"
    if fused_visible + 0.15 < pre_visible:
        return "fusion_loss"
    return "mixed_or_downstream"


@contextmanager
def temp_env(overrides: dict[str, str]):
    old = {}
    sentinel = object()
    for key, value in overrides.items():
        old[key] = os.environ.get(key, sentinel)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def normalize_with_reference(points: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    normalized = (points - mean[None, :]) * (1.0 / (2.0 * scale))
    return np.clip(normalized, -0.5 + 1e-6, 0.5 - 1e-6).astype(np.float32)


def load_gt_geometry(baseline_root: Path, scan_id: str, obj_id: int):
    mean_scale = np.load(
        baseline_root
        / "files"
        / "gs_annotations"
        / scan_id
        / str(obj_id)
        / "mean_scale_dense.npz"
    )
    gt_vox = np.load(
        baseline_root
        / "files"
        / "gs_annotations"
        / scan_id
        / str(obj_id)
        / "voxel_output_dense.npz"
    )["arr_0"][:, :3].astype(np.int32)
    return mean_scale["mean"].astype(np.float32), float(mean_scale["scale"]), gt_vox


def build_visible_frames(
    data_root: Path,
    scan_id: str,
    obj_id: int,
    masks: dict,
) -> tuple[list[str], list[np.ndarray]]:
    frame_ids = scan3r.load_frame_idxs(data_dir=str(data_root / "scenes"), scan_id=scan_id)
    selected_frame_ids = []
    selected_masks = []
    for frame_id in frame_ids:
        obj_mask = np.where(masks[frame_id] == int(obj_id), 1, 0)
        if obj_mask.sum() > 0:
            selected_frame_ids.append(frame_id)
            selected_masks.append(obj_mask.astype(np.uint8))
    return selected_frame_ids, selected_masks


def stage_scan(src_scan_dir: Path, dst_scan_dir: Path) -> None:
    if dst_scan_dir.exists():
        shutil.rmtree(dst_scan_dir)
    dst_scan_dir.mkdir(parents=True, exist_ok=True)

    for item in src_scan_dir.iterdir():
        if item.name in {"sequence.zip", "sequence"}:
            continue
        os.symlink(item, dst_scan_dir / item.name)

    src_zip = src_scan_dir / "sequence.zip"
    src_seq = src_scan_dir / "sequence"
    dst_seq = dst_scan_dir / "sequence"
    if src_zip.exists():
        dst_seq.mkdir(exist_ok=True)
        subprocess.run(["unzip", "-qo", str(src_zip), "-d", str(dst_seq)], check=True)
    elif src_seq.exists():
        os.symlink(src_seq, dst_seq)


def resolve_scene_source_root(
    data_root: Path, baseline_root: Path, scan_id: str, stage_root: Path
) -> Path:
    candidates = [data_root, baseline_root]
    for root in candidates:
        info_path = root / "scenes" / scan_id / "sequence" / "_info.txt"
        if info_path.exists():
            return root
    for root in candidates:
        scan_dir = root / "scenes" / scan_id
        if (scan_dir / "sequence.zip").exists() or (scan_dir / "sequence").exists():
            staged_scan_dir = stage_root / "scenes" / scan_id
            if not (staged_scan_dir / "sequence" / "_info.txt").exists():
                stage_scan(scan_dir, staged_scan_dir)
            return stage_root
    return baseline_root


def compute_metrics_for_object(
    data_root: Path,
    baseline_root: Path,
    stage_root: Path,
    scan_id: str,
    obj_id: int,
    label: str,
    mask_source: str,
    frame_selection: str,
    object_source: str,
    ks: list[int],
) -> list[dict]:
    scene_source_root = resolve_scene_source_root(
        data_root, baseline_root, scan_id, stage_root
    )
    scenes_dir = scene_source_root / "scenes"
    masks = scan3r.load_masks(str(data_root), scan_id, mask_source=mask_source)
    extrinsics = scan3r.load_frame_poses(
        data_dir=str(scene_source_root),
        scan_id=scan_id,
        frame_idxs=scan3r.load_frame_idxs(data_dir=str(scenes_dir), scan_id=scan_id),
    )
    intrinsics = scan3r.load_intrinsics(data_dir=str(scenes_dir), scan_id=scan_id)
    depth_intrinsics = scan3r.load_intrinsics(data_dir=str(scenes_dir), scan_id=scan_id, type="depth")
    depth_shift = vf._load_depth_shift(str(scenes_dir), scan_id)

    frame_ids, obj_masks = build_visible_frames(scene_source_root, scan_id, obj_id, masks)
    frame_ids, obj_masks = vf._filter_selected_masks(
        frame_ids, obj_masks, object_source=object_source
    )

    mean, scale, gt_voxels = load_gt_geometry(baseline_root, scan_id, obj_id)
    gt_set = voxel_set(gt_voxels)

    if not frame_ids:
        return []

    metrics = []
    depth_cache = {}
    for k in ks:
        with temp_env({"OBJECTX_VOXEL_FRAME_SELECTION": frame_selection}):
            selected_frame_ids, selected_masks = vf._select_object_frames(
                frame_ids,
                obj_masks,
                extrinsics=extrinsics,
                max_views=k,
            )

        selected_depths = []
        for frame_id in selected_frame_ids:
            if frame_id not in depth_cache:
                depth_cache[frame_id] = scan3r.load_depth_map(
                    str(scenes_dir / scan_id / "sequence" / f"frame-{frame_id}.depth.pgm"),
                    depth_shift,
                )
            selected_depths.append(depth_cache[frame_id])
        pose_camera_to_world = vf._resolve_pose_camera_to_world(
            extrinsics, selected_frame_ids
        )
        pose_world_to_camera = vf._invert_pose_list(pose_camera_to_world)

        projection_color, linear_depth = vf._project_to_image(
            torch.tensor(gt_voxels, dtype=torch.float32),
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor([scale], dtype=torch.float32),
            torch.from_numpy(np.stack(pose_world_to_camera)),
            torch.from_numpy(intrinsics["intrinsic_mat"]),
        )
        projection_depth, _ = vf._project_to_image(
            torch.tensor(gt_voxels, dtype=torch.float32),
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor([scale], dtype=torch.float32),
            torch.from_numpy(np.stack(pose_world_to_camera)),
            torch.from_numpy(depth_intrinsics["intrinsic_mat"]),
        )
        observed_views = vf._compute_voxel_observations(
            projection_color=projection_color,
            projection_depth=projection_depth,
            linear_depth=linear_depth,
            selected_masks=selected_masks,
            selected_depths=selected_depths,
            color_size=(intrinsics["width"], intrinsics["height"]),
            depth_size=(depth_intrinsics["width"], depth_intrinsics["height"]),
            depth_abs_tol=float(os.getenv("OBJECTX_VOXEL_DEPTH_ABS_TOL", "0.05")),
            depth_rel_tol=float(os.getenv("OBJECTX_VOXEL_DEPTH_REL_TOL", "0.02")),
        )
        gt_visible = observed_views.any(axis=0)
        gt_visible_set = voxel_set(gt_voxels[gt_visible])

        lifted_points = vf._lift_masked_points(
            selected_masks=selected_masks,
            selected_depths=selected_depths,
            pose_camera_to_world=pose_camera_to_world,
            depth_intrinsics=depth_intrinsics,
        )
        normalized_points = normalize_with_reference(lifted_points, mean, scale)
        pre_voxels = vf._voxelize_normalized_points(normalized_points, dilate_iters=1)
        pre_voxels = vf._keep_largest_voxel_component(pre_voxels)
        pre_set = voxel_set(pre_voxels)

        env = {
            "OBJECTX_VOXEL_REFERENCE_MEAN_SCALE_ROOT": str(baseline_root),
            "OBJECTX_VOXEL_MIN_LIFTED_POINTS": "0",
            "OBJECTX_VOXEL_MIN_VOXELS": "0",
            "OBJECTX_VOXEL_TSDF_FALLBACK_TO_LIFTED": "1",
        }
        vf.args = argparse.Namespace(visualize=False)
        with temp_env(env):
            if object_source == "lifted_masks":
                fused_voxels, _, _ = vf._build_lifted_object_voxel_grid(
                    scan_id=scan_id,
                    obj_id=obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )
            elif object_source == "hybrid_masks":
                fused_voxels, _, _ = vf._build_hybrid_object_voxel_grid(
                    scan_id=scan_id,
                    obj_id=obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )
            else:
                fused_voxels, _, _ = vf._build_tsdf_object_voxel_grid(
                    scan_id=scan_id,
                    obj_id=obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )
        fused_set = voxel_set(fused_voxels)

        metric = {
            "scan_id": scan_id,
            "obj_id": int(obj_id),
            "label": label,
            "object_source": object_source,
            "k": int(k),
            "selected_frames": len(selected_frame_ids),
            "frame_ids": selected_frame_ids,
            "mask_area_min": int(min(mask.sum() for mask in selected_masks)),
            "mask_area_mean": float(np.mean([mask.sum() for mask in selected_masks])),
            "mask_area_max": int(max(mask.sum() for mask in selected_masks)),
            "gt_voxel_count": int(len(gt_set)),
            "gt_visible_voxel_count": int(len(gt_visible_set)),
            "gt_visible_ratio": safe_ratio(len(gt_visible_set), len(gt_set)),
            "lifted_points_count": int(lifted_points.shape[0]),
            "pre_voxel_count": int(len(pre_set)),
            "fused_voxel_count": int(len(fused_set)),
            "pre_gt_recall": safe_ratio(len(pre_set & gt_set), len(gt_set)),
            "fused_gt_recall": safe_ratio(len(fused_set & gt_set), len(gt_set)),
            "pre_visible_recall": safe_ratio(len(pre_set & gt_visible_set), len(gt_visible_set)),
            "fused_visible_recall": safe_ratio(len(fused_set & gt_visible_set), len(gt_visible_set)),
        }
        metric["diagnosis"] = summarize_cause(metric)
        metrics.append(metric)
    return metrics


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = [
        "scan_id",
        "obj_id",
        "label",
        "k",
        "selected_frames",
        "mask_area_min",
        "mask_area_mean",
        "mask_area_max",
        "gt_voxel_count",
        "gt_visible_voxel_count",
        "gt_visible_ratio",
        "lifted_points_count",
        "pre_voxel_count",
        "fused_voxel_count",
        "pre_gt_recall",
        "fused_gt_recall",
        "pre_visible_recall",
        "fused_visible_recall",
        "diagnosis",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def plot_object_metrics(rows: list[dict], out_file: Path) -> None:
    rows = sorted(rows, key=lambda x: x["k"])
    ks = [row["k"] for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, [row["gt_visible_ratio"] for row in rows], marker="o", label="GT visible ratio")
    plt.plot(ks, [row["pre_gt_recall"] for row in rows], marker="o", label="Lifted voxel recall")
    plt.plot(ks, [row["fused_gt_recall"] for row in rows], marker="o", label="Fused voxel recall")
    plt.plot(
        ks,
        [row["fused_visible_recall"] for row in rows],
        marker="o",
        label="Fused recall on visible GT",
        linestyle="--",
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel("k selected frames")
    plt.ylabel("ratio")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_aggregate(rows: list[dict], out_file: Path) -> None:
    if not rows:
        return
    by_k = {}
    for row in rows:
        by_k.setdefault(row["k"], []).append(row)
    ks = sorted(by_k)
    agg = {
        "gt_visible_ratio": [float(np.mean([r["gt_visible_ratio"] for r in by_k[k]])) for k in ks],
        "pre_gt_recall": [float(np.mean([r["pre_gt_recall"] for r in by_k[k]])) for k in ks],
        "fused_gt_recall": [float(np.mean([r["fused_gt_recall"] for r in by_k[k]])) for k in ks],
        "pre_visible_recall": [float(np.mean([r["pre_visible_recall"] for r in by_k[k]])) for k in ks],
        "fused_visible_recall": [float(np.mean([r["fused_visible_recall"] for r in by_k[k]])) for k in ks],
    }

    plt.figure(figsize=(7, 4.5))
    for key, label in [
        ("gt_visible_ratio", "GT visible ratio"),
        ("pre_gt_recall", "Lifted voxel recall"),
        ("fused_gt_recall", "Fused voxel recall"),
        ("fused_visible_recall", "Fused recall on visible GT"),
    ]:
        plt.plot(ks, agg[key], marker="o", label=label)
    plt.ylim(0.0, 1.0)
    plt.xlabel("k selected frames")
    plt.ylabel("mean ratio across objects")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    baseline_root = Path(args.baseline_root)
    selection_file = Path(args.selection_file)
    ks = ensure_list(args.ks, [1, 2, 4, 8, 12])
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "vis" / "diagnostics" / f"{data_root.name}_{args.frame_selection}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = load_objects(selection_file)
    all_rows = []
    with tempfile.TemporaryDirectory(prefix="objectx-gapdiag-") as tmpdir:
        stage_root = Path(tmpdir)
        for obj in objects:
            rows = compute_metrics_for_object(
                data_root=data_root,
                baseline_root=baseline_root,
                stage_root=stage_root,
                scan_id=obj["scan_id"],
                obj_id=int(obj["obj_id"]),
                label=obj.get("label", "unknown"),
                mask_source=args.mask_source,
                frame_selection=args.frame_selection,
                object_source=args.object_source,
                ks=ks,
            )
            all_rows.extend(rows)
            if rows:
                stem = f"{obj['scan_id']}_{obj['obj_id']}_{obj.get('label', 'unknown').replace(' ', '_')}"
                plot_object_metrics(rows, out_dir / f"{stem}_metrics.png")

    save_csv(all_rows, out_dir / "summary.csv")
    (out_dir / "summary.json").write_text(json.dumps(all_rows, indent=2))
    plot_aggregate(all_rows, out_dir / "aggregate_metrics.png")

    diagnosis = {}
    for row in all_rows:
        diagnosis.setdefault((row["scan_id"], row["obj_id"], row["label"]), []).append(row)
    diagnosis_summary = []
    for key, rows in diagnosis.items():
        rows = sorted(rows, key=lambda x: x["k"])
        diagnosis_summary.append(
            {
                "scan_id": key[0],
                "obj_id": key[1],
                "label": key[2],
                "best_k": max(rows, key=lambda x: x["fused_gt_recall"])["k"],
                "final_k": rows[-1]["k"],
                "final_diagnosis": rows[-1]["diagnosis"],
                "final_gt_visible_ratio": rows[-1]["gt_visible_ratio"],
                "final_pre_visible_recall": rows[-1]["pre_visible_recall"],
                "final_fused_visible_recall": rows[-1]["fused_visible_recall"],
                "final_fused_gt_recall": rows[-1]["fused_gt_recall"],
            }
        )
    (out_dir / "diagnosis_summary.json").write_text(
        json.dumps(diagnosis_summary, indent=2)
    )
    print(f"Saved diagnosis outputs to {out_dir}")


if __name__ == "__main__":
    main()
