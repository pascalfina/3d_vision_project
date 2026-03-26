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
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--selection-json", required=True)
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    selection = load_json(Path(args.selection_json))
    chosen = selection["objects"]
    default_split = selection.get("default_split", "val")

    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target root already exists: {target_root}")
        shutil.rmtree(target_root)

    (target_root / "files").mkdir(parents=True, exist_ok=True)
    (target_root / "scenes").mkdir(parents=True, exist_ok=True)
    (target_root / "files" / "orig" / "data").mkdir(parents=True, exist_ok=True)

    objects_cfg = load_json(source_root / "files" / "objects.json")["scans"]
    objects_by_scan = {entry["scan"]: entry for entry in objects_cfg}

    filtered_objects = []
    split_map = {"train": [], "val": [], "test": []}
    manifest = {
        "name": selection.get("name", "object_level_pilot"),
        "description": selection.get("description", ""),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "objects": [],
    }

    for item in chosen:
        scan_id = item["scan_id"]
        obj_id = int(item["obj_id"])
        split = item.get("split", default_split)
        if split not in split_map:
            raise ValueError(f"Invalid split {split} for {scan_id}/{obj_id}")

        src_scan = source_root / "scenes" / scan_id
        if not src_scan.exists():
            raise FileNotFoundError(f"Missing scene directory: {src_scan}")
        ensure_symlink(src_scan, target_root / "scenes" / scan_id)

        scan_entry = copy.deepcopy(objects_by_scan[scan_id])
        keep_objects = [obj for obj in scan_entry["objects"] if int(obj["id"]) == obj_id]
        if len(keep_objects) != 1:
            raise ValueError(f"Could not find unique object {obj_id} in {scan_id}")
        scan_entry["objects"] = keep_objects
        filtered_objects.append(scan_entry)
        split_map[split].append(scan_id)

        src_orig = source_root / "files" / "orig" / "data" / f"{scan_id}.pkl.gz"
        scene_graph = load_pkl(src_orig)
        filtered_graph = filter_scene_graph(scene_graph, [obj_id])
        write_pkl_gz(
            target_root / "files" / "orig" / "data" / f"{scan_id}.pkl.gz",
            filtered_graph,
        )

        manifest["objects"].append(
            {
                "scan_id": scan_id,
                "obj_id": obj_id,
                "label": keep_objects[0].get("label", item.get("label")),
                "split": split,
            }
        )

    write_json(target_root / "files" / "objects.json", {"scans": filtered_objects})
    write_json(target_root / "files" / "object_subset_manifest.json", manifest)

    for name in ["3RScan.json", "scannet40_classes.txt"]:
        src = source_root / "files" / name
        if src.exists():
            ensure_symlink(src, target_root / "files" / name)

    pred_src = source_root / "files" / "pred_projection"
    if pred_src.exists():
        ensure_symlink(pred_src, target_root / "files" / "pred_projection")

    for split, scan_ids in split_map.items():
        (target_root / "files" / f"{split}_resplit_scans.txt").write_text(
            ("\n".join(scan_ids) + "\n") if scan_ids else ""
        )
        (target_root / "files" / f"{split}_scans.txt").write_text(
            ("\n".join(scan_ids) + "\n") if scan_ids else ""
        )

    print(f"[built] {target_root}")
    print(f"[objects] {len(manifest['objects'])}")
    for item in manifest["objects"]:
        print(
            f"  - {item['split']} {item['scan_id']} obj {item['obj_id']} ({item['label']})"
        )


if __name__ == "__main__":
    main()
