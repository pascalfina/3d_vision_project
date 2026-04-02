#!/usr/bin/env python3
import argparse
import copy
import gzip
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--replacement-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--obj-id", required=True, type=int)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def ensure_symlink(src: Path, dst: Path) -> None:
    safe_remove(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_pkl(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pkl_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def filter_scene_graph(scene_graph: dict, keep_obj_ids: list[int]) -> dict:
    keep_obj_ids = [int(x) for x in keep_obj_ids]
    old_ids = [int(x) for x in scene_graph["objects_id"].tolist()]
    old_id2idx = {obj_id: idx for idx, obj_id in enumerate(old_ids)}
    keep_indices = [old_id2idx[obj_id] for obj_id in keep_obj_ids]

    filtered = {
        "scan_id": scene_graph["scan_id"],
        "objects_id": scene_graph["objects_id"][keep_indices],
        "global_objects_id": scene_graph["global_objects_id"][keep_indices],
        "objects_cat": scene_graph["objects_cat"][keep_indices],
        "objects_count": len(keep_indices),
        "object_id2idx": {obj_id: idx for idx, obj_id in enumerate(keep_obj_ids)},
        "object_attributes": [scene_graph["object_attributes"][idx] for idx in keep_indices],
        "rel_trans": scene_graph["rel_trans"][keep_indices],
        "root_obj_id": keep_obj_ids[0],
        "bow_vec_object_attr_feats": scene_graph["bow_vec_object_attr_feats"][keep_indices],
    }

    filtered["obj_points"] = {
        res: points[keep_indices] for res, points in scene_graph["obj_points"].items()
    }

    keep_set = set(keep_obj_ids)
    kept_pairs = []
    kept_triples = []
    kept_edges_cat = []
    kept_edge_feats = []

    edge_feats = scene_graph["bow_vec_object_edge_feats"]
    for idx, pair in enumerate(scene_graph["pairs"]):
        src_id, dst_id = int(pair[0]), int(pair[1])
        if src_id not in keep_set or dst_id not in keep_set:
            continue
        kept_pairs.append([src_id, dst_id])
        if idx < len(scene_graph["triples"]):
            kept_triples.append(scene_graph["triples"][idx])
        if idx < len(scene_graph["edges_cat"]):
            kept_edges_cat.append(scene_graph["edges_cat"][idx])
        if idx < edge_feats.shape[0]:
            kept_edge_feats.append(edge_feats[idx])

    filtered["pairs"] = kept_pairs
    filtered["triples"] = kept_triples
    filtered["edges_cat"] = kept_edges_cat
    if kept_pairs:
        filtered["edges"] = np.array(
            [
                [filtered["object_id2idx"][int(src)], filtered["object_id2idx"][int(dst)]]
                for src, dst in kept_pairs
            ],
            dtype=scene_graph["edges"].dtype,
        )
        filtered["bow_vec_object_edge_feats"] = np.stack(kept_edge_feats, axis=0)
    else:
        filtered["edges"] = np.zeros((0, 2), dtype=scene_graph["edges"].dtype)
        feat_dim = edge_feats.shape[1] if edge_feats.ndim == 2 else 41
        filtered["bow_vec_object_edge_feats"] = np.zeros(
            (0, feat_dim), dtype=edge_feats.dtype
        )
    filtered["edges_count"] = int(filtered["edges"].shape[0])
    return filtered


def load_mean(root: Path, scene_id: str, obj_id: int) -> np.ndarray:
    payload = np.load(
        root / "files" / "gs_annotations" / scene_id / str(obj_id) / "mean_scale_dense.npz"
    )
    return payload["mean"].astype(np.float32)


def choose_context_objects(
    baseline_root: Path, scene_id: str, obj_id: int, max_objects: int
) -> list[int]:
    objects_cfg = load_json(baseline_root / "files" / "objects.json")["scans"]
    scan_entry = next(item for item in objects_cfg if item["scan"] == scene_id)
    scene_graph = load_pkl(baseline_root / "files" / "orig" / "data" / f"{scene_id}.pkl.gz")
    valid_ids = set(int(x) for x in scene_graph["objects_id"].tolist())
    all_ids = [int(obj["id"]) for obj in scan_entry["objects"] if int(obj["id"]) in valid_ids]
    if obj_id not in all_ids:
        raise ValueError(f"Object {obj_id} not found in scene {scene_id}")
    target_mean = load_mean(baseline_root, scene_id, obj_id)
    scored = []
    for other in all_ids:
        mean = load_mean(baseline_root, scene_id, other)
        dist = float(np.linalg.norm(mean - target_mean))
        scored.append((dist, other))
    scored.sort(key=lambda x: x[0])
    selected = [oid for _, oid in scored[:max_objects]]
    if obj_id not in selected:
        selected = [obj_id] + selected[:-1]
    return selected


def write_split_files(files_dir: Path, split: str, scene_id: str) -> None:
    for name in ["train", "val", "test"]:
        payload = f"{scene_id}\n" if name == split else ""
        (files_dir / f"{name}_resplit_scans.txt").write_text(payload)
        (files_dir / f"{name}_scans.txt").write_text(payload)


def main():
    args = parse_args()
    baseline_root = Path(args.baseline_root)
    replacement_root = Path(args.replacement_root)
    target_root = Path(args.target_root)
    scene_id = args.scene_id
    obj_id = int(args.obj_id)

    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target root already exists: {target_root}")
        shutil.rmtree(target_root)

    selected_obj_ids = choose_context_objects(
        baseline_root, scene_id, obj_id, args.max_objects
    )

    files_dir = target_root / "files"
    scenes_dir = target_root / "scenes"
    files_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    ensure_symlink(baseline_root / "scenes" / scene_id, scenes_dir / scene_id)

    for name in [
        "3RScan.json",
        "scannet40_classes.txt",
        "Features3D",
        "gt_projection",
        "pred_projection",
    ]:
        src = baseline_root / "files" / name
        if src.exists():
            ensure_symlink(src, files_dir / name)

    replacement_clean = replacement_root / "files" / "pred_projection_clean"
    if replacement_clean.exists():
        ensure_symlink(replacement_clean, files_dir / "pred_projection_clean")

    objects_cfg = load_json(baseline_root / "files" / "objects.json")["scans"]
    scan_entry = next(copy.deepcopy(item) for item in objects_cfg if item["scan"] == scene_id)
    scan_entry["objects"] = [
        obj for obj in scan_entry["objects"] if int(obj["id"]) in set(selected_obj_ids)
    ]
    write_json(files_dir / "objects.json", {"scans": [scan_entry]})

    scene_graph = load_pkl(baseline_root / "files" / "orig" / "data" / f"{scene_id}.pkl.gz")
    filtered_graph = filter_scene_graph(scene_graph, selected_obj_ids)
    write_pkl_gz(files_dir / "orig" / "data" / f"{scene_id}.pkl.gz", filtered_graph)

    target_scene_gs = files_dir / "gs_annotations" / scene_id
    target_scene_gs.mkdir(parents=True, exist_ok=True)
    for selected_obj_id in selected_obj_ids:
        src_dir = baseline_root / "files" / "gs_annotations" / scene_id / str(selected_obj_id)
        if selected_obj_id == obj_id:
            candidate = replacement_root / "files" / "gs_annotations" / scene_id / str(selected_obj_id)
            if candidate.exists():
                src_dir = candidate
        ensure_symlink(src_dir, target_scene_gs / str(selected_obj_id))

    (files_dir / "gs_embeddings").mkdir(parents=True, exist_ok=True)
    write_split_files(files_dir, args.split, scene_id)

    manifest = {
        "baseline_root": str(baseline_root),
        "replacement_root": str(replacement_root),
        "target_root": str(target_root),
        "scene_id": scene_id,
        "obj_id": obj_id,
        "split": args.split,
        "max_objects": args.max_objects,
        "selected_obj_ids": selected_obj_ids,
    }
    write_json(files_dir / "mixed_scene_manifest.json", manifest)

    print(f"[built] {target_root}")
    print(f"[scene] {scene_id}")
    print(f"[replaced obj] {obj_id}")
    print(f"[max_objects] {args.max_objects}")
    print(f"[selected] {selected_obj_ids}")


if __name__ == "__main__":
    main()
