#!/usr/bin/env python3
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--point-level", type=int, default=None)
    parser.add_argument("--anchor-obj-id", type=int, default=None)
    parser.add_argument("--max-points-per-set", type=int, default=60000)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def install_numpy_pickle_compat() -> None:
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    for name in ["multiarray", "numeric", "umath", "_multiarray_umath", "_dtype"]:
        module = getattr(np.core, name, None)
        if module is not None:
            sys.modules.setdefault(f"numpy._core.{name}", module)


def load_pkl(path: Path):
    install_numpy_pickle_compat()
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_scene_graph(root: Path, scene_id: str) -> dict:
    files_dir = root / "files" / "orig" / "data"
    gz_path = files_dir / f"{scene_id}.pkl.gz"
    raw_path = files_dir / f"{scene_id}.pkl"
    if gz_path.exists():
        return load_pkl(gz_path)
    if raw_path.exists():
        return load_pkl(raw_path)
    raise FileNotFoundError(f"Missing scene graph for {scene_id} in {root}")


def choose_point_level(pred_graph: dict, gt_graph: dict, requested: Optional[int]) -> int:
    pred_levels = {int(k) for k in pred_graph["obj_points"].keys()}
    gt_levels = {int(k) for k in gt_graph["obj_points"].keys()}
    common = sorted(pred_levels & gt_levels)
    if not common:
        raise ValueError("Pred and GT scene graphs do not share any obj_points level.")
    if requested is None:
        return common[-1]
    if requested not in common:
        raise ValueError(f"Requested point level {requested} not available in both roots: {common}")
    return requested


def build_points_by_id(scene_graph: dict, point_level: int) -> dict[int, np.ndarray]:
    object_ids = [int(x) for x in scene_graph["objects_id"].tolist()]
    points = scene_graph["obj_points"][point_level]
    return {obj_id: points[idx].astype(np.float32) for idx, obj_id in enumerate(object_ids)}


def center(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0).astype(np.float32)


def subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[idx]


def build_overlay_html(gt_points_by_id: dict[int, np.ndarray], pred_points_by_id: dict[int, np.ndarray], shared_ids: list[int], out_path: Path, max_points_per_set: int) -> None:
    import trimesh
    import trimesh.viewer

    gt_points = np.concatenate([gt_points_by_id[obj_id] for obj_id in shared_ids], axis=0)
    pred_points = np.concatenate([pred_points_by_id[obj_id] for obj_id in shared_ids], axis=0)

    gt_points = subsample(gt_points, max_points_per_set)
    pred_points = subsample(pred_points, max_points_per_set)

    gt_colors = np.tile(np.array([[60, 180, 75]], dtype=np.uint8), (gt_points.shape[0], 1))
    pred_colors = np.tile(np.array([[220, 50, 47]], dtype=np.uint8), (pred_points.shape[0], 1))

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.points.PointCloud(gt_points, colors=gt_colors), geom_name="gt")
    scene.add_geometry(trimesh.points.PointCloud(pred_points, colors=pred_colors), geom_name="pred")
    bounds = scene.bounds
    scene.set_camera(
        angles=(0.7, 0.0, 0.6),
        distance=max(float(np.max(bounds[1] - bounds[0])) * 1.6, 1.0),
        center=bounds.mean(axis=0),
    )
    out_path.write_text(trimesh.viewer.scene_to_html(scene), encoding="utf-8")


def build_center_overlay_png(gt_centers: np.ndarray, pred_centers: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], c="#3cb44b", s=22, label="GT centers")
    ax.scatter(pred_centers[:, 0], pred_centers[:, 1], pred_centers[:, 2], c="#e31a1c", s=22, label="Pred centers")
    for gt_center, pred_center in zip(gt_centers, pred_centers):
        ax.plot(
            [gt_center[0], pred_center[0]],
            [gt_center[1], pred_center[1]],
            [gt_center[2], pred_center[2]],
            color="gray",
            linewidth=0.6,
            alpha=0.6,
        )
    ax.set_title("Arrangement centers: GT vs Pred")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_graph = load_scene_graph(pred_root, args.scene_id)
    gt_graph = load_scene_graph(gt_root, args.scene_id)
    point_level = choose_point_level(pred_graph, gt_graph, args.point_level)

    pred_points_by_id = build_points_by_id(pred_graph, point_level)
    gt_points_by_id = build_points_by_id(gt_graph, point_level)

    pred_ids = sorted(pred_points_by_id.keys())
    gt_ids = sorted(gt_points_by_id.keys())
    shared_ids = sorted(set(pred_ids) & set(gt_ids))
    pred_only_ids = sorted(set(pred_ids) - set(gt_ids))
    gt_only_ids = sorted(set(gt_ids) - set(pred_ids))
    if not shared_ids:
        raise ValueError("No shared object ids between pred and GT roots.")

    gt_centers = np.stack([center(gt_points_by_id[obj_id]) for obj_id in shared_ids], axis=0)
    pred_centers = np.stack([center(pred_points_by_id[obj_id]) for obj_id in shared_ids], axis=0)

    abs_center_errors = np.linalg.norm(pred_centers - gt_centers, axis=1)

    if args.anchor_obj_id is not None and args.anchor_obj_id in shared_ids:
        anchor_obj_id = int(args.anchor_obj_id)
    elif int(pred_graph["root_obj_id"]) in shared_ids:
        anchor_obj_id = int(pred_graph["root_obj_id"])
    else:
        anchor_obj_id = int(shared_ids[0])
    anchor_idx = shared_ids.index(anchor_obj_id)

    gt_rel = gt_centers - gt_centers[anchor_idx][None, :]
    pred_rel = pred_centers - pred_centers[anchor_idx][None, :]
    relative_center_errors = np.linalg.norm(pred_rel - gt_rel, axis=1)

    gt_pairwise = np.linalg.norm(gt_centers[:, None, :] - gt_centers[None, :, :], axis=-1)
    pred_pairwise = np.linalg.norm(pred_centers[:, None, :] - pred_centers[None, :, :], axis=-1)
    upper = np.triu_indices(len(shared_ids), k=1)
    pairwise_diff = pred_pairwise[upper] - gt_pairwise[upper]

    per_object = []
    for idx, obj_id in enumerate(shared_ids):
        per_object.append(
            {
                "obj_id": int(obj_id),
                "gt_center": [float(x) for x in gt_centers[idx]],
                "pred_center": [float(x) for x in pred_centers[idx]],
                "abs_center_error": float(abs_center_errors[idx]),
                "relative_center_error_to_anchor": float(relative_center_errors[idx]),
            }
        )

    html_path = out_dir / f"{args.scene_id}_arrangement_overlay.html"
    png_path = out_dir / f"{args.scene_id}_arrangement_centers.png"
    build_overlay_html(
        gt_points_by_id=gt_points_by_id,
        pred_points_by_id=pred_points_by_id,
        shared_ids=shared_ids,
        out_path=html_path,
        max_points_per_set=args.max_points_per_set,
    )
    build_center_overlay_png(gt_centers=gt_centers, pred_centers=pred_centers, out_path=png_path)

    summary = {
        "scene_id": args.scene_id,
        "point_level": int(point_level),
        "anchor_obj_id": int(anchor_obj_id),
        "shared_object_count": int(len(shared_ids)),
        "pred_object_count": int(len(pred_ids)),
        "gt_object_count": int(len(gt_ids)),
        "pred_only_ids": pred_only_ids,
        "gt_only_ids": gt_only_ids,
        "mean_abs_center_error": float(abs_center_errors.mean()),
        "median_abs_center_error": float(np.median(abs_center_errors)),
        "max_abs_center_error": float(abs_center_errors.max()),
        "mean_relative_center_error": float(relative_center_errors.mean()),
        "median_relative_center_error": float(np.median(relative_center_errors)),
        "max_relative_center_error": float(relative_center_errors.max()),
        "pairwise_distance_mae": float(np.mean(np.abs(pairwise_diff))) if pairwise_diff.size else 0.0,
        "pairwise_distance_rmse": float(np.sqrt(np.mean(pairwise_diff ** 2))) if pairwise_diff.size else 0.0,
        "html": str(html_path),
        "png": str(png_path),
        "per_object": per_object,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
