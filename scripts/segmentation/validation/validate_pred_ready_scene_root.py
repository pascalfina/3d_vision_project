#!/usr/bin/env python3
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-dataset-check", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


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


def load_scene_graph(files_dir: Path, scene_id: str) -> dict:
    gz_path = files_dir / "orig" / "data" / f"{scene_id}.pkl.gz"
    raw_path = files_dir / "orig" / "data" / f"{scene_id}.pkl"
    if gz_path.exists():
        return load_pkl(gz_path)
    if raw_path.exists():
        return load_pkl(raw_path)
    raise FileNotFoundError(f"Missing scene graph for {scene_id}")


def validate_schema(root: Path, scene_id: str) -> dict:
    files_dir = root / "files"
    scene_dir = root / "scenes" / scene_id
    objects_payload = load_json(files_dir / "objects.json")
    scene_graph = load_scene_graph(files_dir, scene_id)

    scan_entry = next((item for item in objects_payload["scans"] if item["scan"] == scene_id), None)
    if scan_entry is None:
        raise ValueError(f"Scene {scene_id} missing from objects.json")

    expected_keys = {
        "scan_id",
        "objects_id",
        "global_objects_id",
        "objects_cat",
        "objects_count",
        "object_id2idx",
        "obj_points",
        "root_obj_id",
        "rel_trans",
        "edges",
        "edges_count",
        "pairs",
        "triples",
        "edges_cat",
        "object_attributes",
        "bow_vec_object_attr_feats",
        "bow_vec_object_edge_feats",
    }
    missing = sorted(expected_keys - set(scene_graph.keys()))
    if missing:
        raise ValueError(f"Scene graph missing keys: {missing}")

    object_ids_json = [int(obj["id"]) for obj in scan_entry["objects"]]
    object_ids_graph = [int(x) for x in scene_graph["objects_id"].tolist()]
    if object_ids_json != object_ids_graph:
        raise ValueError("objects.json order does not match scene graph objects_id")

    obj_count = int(scene_graph["objects_count"])
    if obj_count != len(object_ids_json):
        raise ValueError(f"objects_count mismatch: {obj_count} vs {len(object_ids_json)}")

    obj_points = scene_graph["obj_points"]
    point_levels = sorted(int(k) for k in obj_points.keys())
    for level in point_levels:
        arr = obj_points[level]
        if arr.shape != (obj_count, level, 3):
            raise ValueError(f"obj_points[{level}] has wrong shape: {arr.shape}")

    rel_trans = scene_graph["rel_trans"]
    if rel_trans.shape != (obj_count, 3):
        raise ValueError(f"rel_trans has wrong shape: {rel_trans.shape}")
    if not np.allclose(rel_trans[0], 0.0):
        raise ValueError(f"root rel_trans is not zero: {rel_trans[0]}")

    edges = scene_graph["edges"]
    if edges.shape[0] != int(scene_graph["edges_count"]):
        raise ValueError("edges_count does not match edges.shape[0]")

    attr = scene_graph["bow_vec_object_attr_feats"]
    edge_attr = scene_graph["bow_vec_object_edge_feats"]
    if attr.shape[0] != obj_count:
        raise ValueError("bow_vec_object_attr_feats row count mismatch")
    if edge_attr.shape[0] != obj_count:
        raise ValueError("bow_vec_object_edge_feats row count mismatch")

    required_scene_files = [
        scene_dir / "data.npy",
        scene_dir / "sequence" / "_info.txt",
    ]
    missing_scene_files = [str(path) for path in required_scene_files if not path.exists()]
    if missing_scene_files:
        raise ValueError(f"Missing staged scene files: {missing_scene_files}")

    summary = {
        "root": str(root),
        "scene_id": scene_id,
        "objects_count": obj_count,
        "object_ids": object_ids_graph,
        "root_obj_id": int(scene_graph["root_obj_id"]),
        "point_levels": point_levels,
        "edges_count": int(scene_graph["edges_count"]),
        "attr_shape": list(attr.shape),
        "edge_attr_shape": list(edge_attr.shape),
        "required_keys_ok": True,
        "scene_files_ok": True,
    }
    return summary


def dataset_check(root: Path, split: str, config_path: str, scene_id: str) -> dict:
    from configs import update_configs
    from src.datasets.scan3r_scene import Scan3RSceneGraphDataset

    cfg = update_configs(
        config_path,
        [f"data.root_dir={root}"],
        do_ensure_dir=False,
    )
    dataset = Scan3RSceneGraphDataset(cfg, split)
    item = dataset[0]
    batch = dataset.collate_fn([item])
    scene_graphs = batch["scene_graphs"]
    return {
        "dataset_len": int(len(dataset)),
        "dataset_scan_id": item["scan_id"],
        "graph_per_obj_count": scene_graphs["graph_per_obj_count"].tolist(),
        "tot_obj_pts_shape": list(scene_graphs["tot_obj_pts"].shape),
        "edges_shape": list(scene_graphs["edges"].shape),
        "attr_shape": list(scene_graphs["tot_bow_vec_object_attr_feats"].shape),
        "edge_attr_shape": list(scene_graphs["tot_bow_vec_object_edge_feats"].shape),
    }


def export_html(root: Path, scene_id: str, out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    import trimesh
    import trimesh.viewer

    scene_graph = load_scene_graph(root / "files", scene_id)
    point_level = max(int(k) for k in scene_graph["obj_points"].keys())
    obj_points = scene_graph["obj_points"][point_level]
    obj_ids = [int(x) for x in scene_graph["objects_id"].tolist()]

    palette = plt.get_cmap("tab20")
    pts_all = []
    cols_all = []
    for idx, points in enumerate(obj_points):
        color = np.array(palette(idx % 20)[:3], dtype=np.float32)
        pts_all.append(points)
        cols_all.append(np.tile(color[None, :], (points.shape[0], 1)))

    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(cols_all, axis=0)
    cloud = trimesh.points.PointCloud(pts, colors=(255.0 * cols).astype(np.uint8))
    scene = trimesh.Scene()
    scene.add_geometry(cloud, geom_name="pred_ready_obj_points")
    bounds = scene.bounds
    scene.set_camera(
        angles=(0.7, 0.0, 0.6),
        distance=max(float(np.max(bounds[1] - bounds[0])) * 1.6, 1.0),
        center=bounds.mean(axis=0),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{scene_id}_pred_ready_obj_points.html"
    html_path.write_text(trimesh.viewer.scene_to_html(scene), encoding="utf-8")
    return {
        "html": str(html_path),
        "html_point_level": int(point_level),
        "html_object_ids": obj_ids,
    }


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else None

    summary = validate_schema(root=root, scene_id=args.scene_id)
    if not args.skip_dataset_check:
        summary["dataset_check"] = dataset_check(
            root=root,
            split=args.split,
            config_path=args.config,
            scene_id=args.scene_id,
        )
    if out_dir is not None:
        summary["export"] = export_html(root=root, scene_id=args.scene_id, out_dir=out_dir)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
