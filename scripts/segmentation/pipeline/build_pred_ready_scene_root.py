#!/usr/bin/env python3
import argparse
import copy
import gzip
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

import numpy as np


FALLBACK_NYU40 = 40
FALLBACK_EIGEN13 = 13
FALLBACK_RIO27 = 27
FALLBACK_PLY_COLOR = "#808080"
MANIFEST_NAME = "pred_ready_scene_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--reconstruction-root", required=True)
    parser.add_argument("--reconstruction-scenes-dirname", default="scenes")
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--min-voxels", type=int, default=32)
    parser.add_argument(
        "--point-counts",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
    )
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


def read_scene_graph(root: Path, scene_id: str, name: str = "data") -> dict:
    files_dir = root / "files" / "orig" / name
    gz_path = files_dir / f"{scene_id}.pkl.gz"
    raw_path = files_dir / f"{scene_id}.pkl"
    if gz_path.exists():
        return load_pkl(gz_path)
    if raw_path.exists():
        return load_pkl(raw_path)
    raise FileNotFoundError(f"Missing scene graph for {scene_id}: {gz_path} / {raw_path}")


def voxel_to_world(voxel_indices: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    voxel = voxel_indices.astype(np.float32) * (1.0 / 64.0)
    voxel = voxel * 2.0 - 1.0
    return voxel * float(scale) + mean[None, :]


def flatten_attributes_dict(attributes: dict) -> list[str]:
    if not isinstance(attributes, dict):
        return []
    words = []
    for values in attributes.values():
        if isinstance(values, list):
            words.extend(str(value) for value in values)
    return words


def deterministic_sample(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = points[order]
    if points.shape[0] >= count:
        idx = np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)
        return points[idx].astype(np.float32)
    reps = int(np.ceil(count / points.shape[0]))
    tiled = np.tile(points, (reps, 1))
    return tiled[:count].astype(np.float32)


def build_target_3rscan(scan_data: list[dict], scene_id: str) -> list[dict]:
    for item in scan_data:
        if item["reference"] == scene_id or any(
            scan["reference"] == scene_id for scan in item["scans"]
        ):
            return [
                {
                    "reference": scene_id,
                    "scans": [],
                    "type": item.get("type", "validation"),
                    "ambiguity": [],
                }
            ]
    raise ValueError(f"Scene {scene_id} not found in baseline 3RScan.json")


def write_split_files(files_dir: Path, split: str, scene_id: str) -> None:
    for name in ["train", "val", "test"]:
        payload = f"{scene_id}\n" if name == split else ""
        (files_dir / f"{name}_resplit_scans.txt").write_text(payload)
        (files_dir / f"{name}_scans.txt").write_text(payload)


def pick_existing_dir(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def stage_scene(src_scan_dir: Path, dst_scan_dir: Path) -> None:
    safe_remove(dst_scan_dir)
    dst_scan_dir.mkdir(parents=True, exist_ok=True)

    for item in src_scan_dir.iterdir():
        if item.name in {"sequence.zip", "sequence"}:
            continue
        ensure_symlink(item, dst_scan_dir / item.name)

    src_zip = src_scan_dir / "sequence.zip"
    src_seq = src_scan_dir / "sequence"
    dst_seq = dst_scan_dir / "sequence"
    if src_zip.exists():
        dst_seq.mkdir(exist_ok=True)
        shutil.unpack_archive(str(src_zip), str(dst_seq), format="zip")
    elif src_seq.exists():
        ensure_symlink(src_seq, dst_seq)
    else:
        raise FileNotFoundError(
            f"Neither sequence.zip nor sequence/ found in scene directory {src_scan_dir}"
        )


def link_compatibility_inputs(
    baseline_root: Path,
    reconstruction_root: Path,
    target_root: Path,
    scene_id: str,
    split: str,
    reconstruction_scenes_dirname: str = "scenes",
) -> None:
    files_dir = target_root / "files"
    scenes_dir = target_root / "scenes"
    files_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    scene_src = reconstruction_root / reconstruction_scenes_dirname / scene_id
    if (
        reconstruction_scenes_dirname != "scenes"
        and not scene_src.exists()
    ):
        raise FileNotFoundError(
            f"Missing reconstruction scene directory for {scene_id}: {scene_src}"
        )
    if not scene_src.exists():
        scene_src = baseline_root / "scenes" / scene_id
    if not scene_src.exists():
        raise FileNotFoundError(f"Missing scene directory for {scene_id}")
    stage_scene(scene_src, scenes_dir / scene_id)

    scannet_classes = baseline_root / "files" / "scannet40_classes.txt"
    if scannet_classes.exists():
        ensure_symlink(scannet_classes, files_dir / "scannet40_classes.txt")

    features_dir = pick_existing_dir(
        reconstruction_root / "files" / "Features3D",
        baseline_root / "files" / "Features3D",
    )
    if features_dir is not None:
        ensure_symlink(features_dir, files_dir / "Features3D")

    for name in ["gt_projection", "pred_projection", "pred_projection_clean"]:
        src = pick_existing_dir(
            reconstruction_root / "files" / name,
            baseline_root / "files" / name,
        )
        if src is not None:
            ensure_symlink(src, files_dir / name)

    scan_data = load_json(baseline_root / "files" / "3RScan.json")
    write_json(files_dir / "3RScan.json", build_target_3rscan(scan_data, scene_id))
    write_split_files(files_dir, split, scene_id)
    (files_dir / "gs_embeddings").mkdir(parents=True, exist_ok=True)


def discover_reconstructed_objects(
    reconstruction_root: Path,
    scene_id: str,
    min_voxels: int,
) -> list[dict]:
    scene_root = reconstruction_root / "files" / "gs_annotations" / scene_id
    if not scene_root.exists():
        raise FileNotFoundError(f"Missing reconstruction scene annotations: {scene_root}")

    objects = []
    for obj_dir in sorted(scene_root.iterdir(), key=lambda p: int(p.name)):
        if not obj_dir.is_dir():
            continue
        voxel_path = obj_dir / "voxel_output_dense.npz"
        mean_scale_path = obj_dir / "mean_scale_dense.npz"
        if not voxel_path.exists() or not mean_scale_path.exists():
            continue

        voxels = np.load(voxel_path)["arr_0"][:, :3].astype(np.int32)
        if voxels.shape[0] < min_voxels:
            continue

        mean_scale = np.load(mean_scale_path)
        mean = mean_scale["mean"].astype(np.float32)
        scale = float(mean_scale["scale"])
        points_world = voxel_to_world(voxels, mean, scale)
        center = points_world.mean(axis=0).astype(np.float32)
        extent = (points_world.max(axis=0) - points_world.min(axis=0)).astype(np.float32)

        objects.append(
            {
                "obj_id": int(obj_dir.name),
                "obj_dir": obj_dir,
                "voxel_count": int(voxels.shape[0]),
                "voxels": voxels,
                "points_world": points_world,
                "center": center,
                "extent": extent,
                "mean": mean,
                "scale": scale,
            }
        )

    if not objects:
        raise ValueError(f"No reconstructed objects passed min_voxels for {scene_id}")

    root_object = min(objects, key=lambda item: (-item["voxel_count"], item["obj_id"]))
    root_obj_id = int(root_object["obj_id"])
    ordered = [root_object] + sorted(
        [item for item in objects if int(item["obj_id"]) != root_obj_id],
        key=lambda item: int(item["obj_id"]),
    )
    return ordered


def make_fallback_object_entry(obj_id: int) -> dict:
    fallback_global_id = 1_000_000 + int(obj_id)
    return {
        "ply_color": FALLBACK_PLY_COLOR,
        "nyu40": str(FALLBACK_NYU40),
        "eigen13": str(FALLBACK_EIGEN13),
        "label": "reconstructed_object",
        "rio27": str(FALLBACK_RIO27),
        "affordances": [],
        "id": str(obj_id),
        "global_id": str(fallback_global_id),
        "attributes": {
            "state": ["reconstructed"],
            "shape": ["object"],
            "lexical": ["generated"],
            "color": ["unknown"],
        },
        "generated": True,
    }


def build_objects_json(
    baseline_root: Path,
    scene_id: str,
    ordered_objects: list[dict],
) -> tuple[dict, dict[int, dict], list[str]]:
    all_scans = load_json(baseline_root / "files" / "objects.json")["scans"]
    baseline_scan = next((item for item in all_scans if item["scan"] == scene_id), None)
    if baseline_scan is None:
        raise ValueError(f"Scene {scene_id} not found in baseline objects.json")

    baseline_objects = {
        int(obj["id"]): copy.deepcopy(obj) for obj in baseline_scan["objects"]
    }
    generated_objects = []
    used_semantic_fields = [
        "id",
        "label",
        "nyu40",
        "global_id",
        "rio27",
        "eigen13",
        "attributes",
        "affordances",
        "ply_color",
    ]

    for item in ordered_objects:
        obj_id = int(item["obj_id"])
        if obj_id in baseline_objects:
            obj_entry = baseline_objects[obj_id]
            obj_entry["id"] = str(obj_id)
        else:
            obj_entry = make_fallback_object_entry(obj_id)
        generated_objects.append(obj_entry)

    payload = {"scans": [{"scan": scene_id, "objects": generated_objects}]}
    return payload, baseline_objects, used_semantic_fields


def build_reference_maps(scene_graph: dict) -> dict[str, dict]:
    object_ids = [int(x) for x in scene_graph["objects_id"].tolist()]
    id_to_idx = {obj_id: idx for idx, obj_id in enumerate(object_ids)}

    attr_feats = scene_graph.get("bow_vec_object_attr_feats")
    edge_feats = scene_graph.get("bow_vec_object_edge_feats")
    object_attributes = scene_graph.get("object_attributes", [])

    return {
        "id_to_idx": id_to_idx,
        "attr_dim": int(attr_feats.shape[1]) if attr_feats is not None and attr_feats.ndim == 2 else 0,
        "edge_dim": int(edge_feats.shape[1]) if edge_feats is not None and edge_feats.ndim == 2 else 0,
        "attr_dtype": attr_feats.dtype if attr_feats is not None else np.float32,
        "edge_dtype": edge_feats.dtype if edge_feats is not None else np.float32,
        "attr_feats": attr_feats,
        "object_attributes": object_attributes,
    }


def build_scene_graph(
    baseline_scene_graph: dict,
    objects_payload: dict,
    baseline_objects_by_id: dict[int, dict],
    ordered_objects: list[dict],
    point_counts: list[int],
    knn: int,
) -> dict:
    reference = build_reference_maps(baseline_scene_graph)
    ordered_ids = [int(item["obj_id"]) for item in ordered_objects]
    object_id2idx = {obj_id: idx for idx, obj_id in enumerate(ordered_ids)}

    objects_json_entries = {
        int(obj["id"]): obj for obj in objects_payload["scans"][0]["objects"]
    }

    centers = np.stack([item["center"] for item in ordered_objects], axis=0).astype(np.float32)
    root_center = centers[0]
    rel_trans = (root_center[None, :] - centers).astype(np.float32)

    obj_points = {}
    for count in point_counts:
        obj_points[int(count)] = np.stack(
            [deterministic_sample(item["points_world"], count) for item in ordered_objects],
            axis=0,
        ).astype(np.float32)

    object_attributes = []
    attr_rows = []
    edge_feat_dim = max(reference["edge_dim"], 41)
    attr_feat_dim = reference["attr_dim"]
    edge_feat_rows = np.zeros(
        (len(ordered_objects), edge_feat_dim),
        dtype=reference["edge_dtype"] if reference["edge_dim"] else np.float32,
    )

    global_objects_id = []
    objects_cat = []
    for item in ordered_objects:
        obj_id = int(item["obj_id"])
        object_entry = objects_json_entries[obj_id]
        baseline_object = baseline_objects_by_id.get(obj_id)
        baseline_idx = reference["id_to_idx"].get(obj_id)

        if baseline_idx is not None and baseline_idx < len(reference["object_attributes"]):
            attrs = [str(word) for word in reference["object_attributes"][baseline_idx]]
        elif baseline_object is not None:
            attrs = flatten_attributes_dict(baseline_object.get("attributes", {}))
        else:
            attrs = ["reconstructed", "object"]
        object_attributes.append(attrs)

        if attr_feat_dim > 0 and baseline_idx is not None:
            attr_rows.append(reference["attr_feats"][baseline_idx])
        elif attr_feat_dim > 0:
            attr_rows.append(np.zeros(attr_feat_dim, dtype=reference["attr_dtype"]))

        global_id = int(object_entry.get("global_id", 1_000_000 + obj_id))
        global_objects_id.append(global_id)
        objects_cat.append(global_id)

    if attr_feat_dim > 0:
        bow_vec_object_attr_feats = np.stack(attr_rows, axis=0).astype(reference["attr_dtype"])
    else:
        bow_vec_object_attr_feats = np.zeros((len(ordered_objects), 0), dtype=np.float32)

    num_objects = len(ordered_objects)
    edges = []
    pairs = []
    triples = []
    edges_cat = []
    if num_objects > 1 and knn > 0:
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        edge_k = min(knn, num_objects - 1)
        for src_idx in range(num_objects):
            neighbor_order = np.argsort(distances[src_idx], kind="stable")[:edge_k]
            src_obj_id = ordered_ids[src_idx]
            for dst_idx in neighbor_order.tolist():
                dst_obj_id = ordered_ids[dst_idx]
                edges.append([src_idx, dst_idx])
                pairs.append([src_obj_id, dst_obj_id])
                triples.append([src_obj_id, dst_obj_id, 0])
                edges_cat.append(0)

    scene_graph = {
        "scan_id": objects_payload["scans"][0]["scan"],
        "objects_id": np.array(ordered_ids, dtype=np.int64),
        "global_objects_id": np.array(global_objects_id, dtype=np.int64),
        "objects_cat": np.array(objects_cat, dtype=np.int64),
        "objects_count": int(num_objects),
        "object_id2idx": object_id2idx,
        "obj_points": obj_points,
        "root_obj_id": int(ordered_ids[0]),
        "rel_trans": rel_trans,
        "edges": np.array(edges, dtype=np.int64).reshape((-1, 2))
        if edges
        else np.zeros((0, 2), dtype=np.int64),
        "edges_count": int(len(edges)),
        "pairs": pairs,
        "triples": triples,
        "edges_cat": edges_cat,
        "object_attributes": object_attributes,
        "bow_vec_object_attr_feats": bow_vec_object_attr_feats,
        "bow_vec_object_edge_feats": edge_feat_rows,
    }
    return scene_graph


def link_reconstructed_gs_annotations(
    reconstruction_root: Path,
    target_root: Path,
    scene_id: str,
    ordered_objects: list[dict],
) -> None:
    target_scene_root = target_root / "files" / "gs_annotations" / scene_id
    target_scene_root.mkdir(parents=True, exist_ok=True)
    for item in ordered_objects:
        obj_id = int(item["obj_id"])
        src = reconstruction_root / "files" / "gs_annotations" / scene_id / str(obj_id)
        ensure_symlink(src, target_scene_root / str(obj_id))


def build_manifest(
    baseline_root: Path,
    reconstruction_root: Path,
    target_root: Path,
    scene_id: str,
    split: str,
    knn: int,
    point_counts: list[int],
    ordered_objects: list[dict],
    copied_semantic_fields: list[str],
) -> dict:
    return {
        "baseline_root": str(baseline_root),
        "reconstruction_root": str(reconstruction_root),
        "target_root": str(target_root),
        "scene_id": scene_id,
        "split": split,
        "selected_obj_ids": [int(item["obj_id"]) for item in ordered_objects],
        "root_obj_id": int(ordered_objects[0]["obj_id"]),
        "knn": int(knn),
        "point_counts": [int(x) for x in point_counts],
        "copied_semantic_fields": copied_semantic_fields,
        "objects": [
            {
                "obj_id": int(item["obj_id"]),
                "voxel_count": int(item["voxel_count"]),
                "center": [float(x) for x in item["center"]],
                "extent": [float(x) for x in item["extent"]],
                "mean": [float(x) for x in item["mean"]],
                "scale": float(item["scale"]),
            }
            for item in ordered_objects
        ],
    }


def main():
    args = parse_args()
    baseline_root = Path(args.baseline_root)
    reconstruction_root = Path(args.reconstruction_root)
    target_root = Path(args.target_root)
    scene_id = args.scene_id
    point_counts = sorted({int(x) for x in args.point_counts})

    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target root already exists: {target_root}")
        shutil.rmtree(target_root)

    link_compatibility_inputs(
        baseline_root=baseline_root,
        reconstruction_root=reconstruction_root,
        target_root=target_root,
        scene_id=scene_id,
        split=args.split,
        reconstruction_scenes_dirname=args.reconstruction_scenes_dirname,
    )

    ordered_objects = discover_reconstructed_objects(
        reconstruction_root=reconstruction_root,
        scene_id=scene_id,
        min_voxels=args.min_voxels,
    )
    objects_payload, baseline_objects_by_id, copied_semantic_fields = build_objects_json(
        baseline_root=baseline_root,
        scene_id=scene_id,
        ordered_objects=ordered_objects,
    )
    write_json(target_root / "files" / "objects.json", objects_payload)

    baseline_scene_graph = read_scene_graph(baseline_root, scene_id, name="data")
    generated_scene_graph = build_scene_graph(
        baseline_scene_graph=baseline_scene_graph,
        objects_payload=objects_payload,
        baseline_objects_by_id=baseline_objects_by_id,
        ordered_objects=ordered_objects,
        point_counts=point_counts,
        knn=args.knn,
    )
    write_pkl_gz(
        target_root / "files" / "orig" / "data" / f"{scene_id}.pkl.gz",
        generated_scene_graph,
    )

    link_reconstructed_gs_annotations(
        reconstruction_root=reconstruction_root,
        target_root=target_root,
        scene_id=scene_id,
        ordered_objects=ordered_objects,
    )

    manifest = build_manifest(
        baseline_root=baseline_root,
        reconstruction_root=reconstruction_root,
        target_root=target_root,
        scene_id=scene_id,
        split=args.split,
        knn=args.knn,
        point_counts=point_counts,
        ordered_objects=ordered_objects,
        copied_semantic_fields=copied_semantic_fields,
    )
    write_json(target_root / "files" / MANIFEST_NAME, manifest)

    print(f"[built] {target_root}")
    print(f"[scene] {scene_id}")
    print(f"[objects] {len(ordered_objects)}")
    print(f"[root] {ordered_objects[0]['obj_id']}")
    print(f"[ids] {[int(item['obj_id']) for item in ordered_objects]}")


if __name__ == "__main__":
    main()
