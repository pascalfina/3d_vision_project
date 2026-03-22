#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-root",
        default="/work/scratch/pafina/objectx-data-baseline",
    )
    parser.add_argument(
        "--sparse-root",
        default="/work/scratch/pafina/objectx-data-sparse10-keep75pct",
    )
    parser.add_argument(
        "--benchmark-file",
        default="/work/scratch/pafina/objectx-data-baseline/files/sparse_benchmark_10.txt",
    )
    parser.add_argument(
        "--pair-dir",
        default="/work/scratch/pafina/objectx-data-sparse10-keep75pct/files/step2_slat_pairs_keep75pct",
    )
    parser.add_argument(
        "--output",
        default="/work/scratch/pafina/objectx-data-sparse10-keep75pct/files/step2_slat_pairs_keep75pct_manifest.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scene_splits(root: Path) -> dict[str, str]:
    out = {}
    for split in ["train", "val", "test"]:
        path = root / "files" / f"{split}_resplit_scans.txt"
        if not path.exists():
            path = root / "files" / f"{split}_scans.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            scene = line.strip()
            if scene:
                out[scene] = split
    return out


def load_objects_lookup(root: Path) -> dict[str, dict]:
    obj = json.loads((root / "files" / "objects.json").read_text())["scans"]
    return {entry["scan"]: entry for entry in obj}


def load_scene_slat(path: Path) -> dict[int, dict]:
    file = np.load(path, allow_pickle=True)
    coords = file["coords"]
    feats = file["feats"]
    means = file["mean"]
    scales = file["scale"]
    obj_ids = list(map(int, file["obj_id"].tolist()))

    out = {}
    batch_index = coords[:, 0].astype(np.int32)
    for idx, obj_id in enumerate(obj_ids):
        mask = batch_index == idx
        out[obj_id] = {
            "coords": coords[mask, 1:].astype(np.int16, copy=False),
            "feats": feats[mask].astype(np.float32, copy=False),
            "mean": np.asarray(means[idx], dtype=np.float32),
            "scale": np.asarray(scales[idx], dtype=np.float32),
        }
    return out


def build_object_pair(
    scene_id: str,
    split: str,
    obj_id: int,
    full_obj: dict,
    sparse_obj: dict,
    output_path: Path,
    overwrite: bool,
) -> dict:
    full_coords = np.asarray(full_obj["coords"], dtype=np.int16)
    full_feats = np.asarray(full_obj["feats"], dtype=np.float32)
    sparse_coords = np.asarray(sparse_obj["coords"], dtype=np.int16)
    sparse_feats = np.asarray(sparse_obj["feats"], dtype=np.float32)

    coord_to_index = {
        tuple(map(int, coord.tolist())): idx for idx, coord in enumerate(full_coords)
    }
    observed_mask = np.zeros(full_coords.shape[0], dtype=np.uint8)
    sparse_feats_on_full = np.zeros_like(full_feats, dtype=np.float32)

    matched_sparse = 0
    extra_sparse_coords = []
    for coord, feat in zip(sparse_coords, sparse_feats):
        key = tuple(map(int, coord.tolist()))
        idx = coord_to_index.get(key)
        if idx is None:
            extra_sparse_coords.append(list(key))
            continue
        observed_mask[idx] = 1
        sparse_feats_on_full[idx] = feat
        matched_sparse += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not output_path.exists():
        np.savez_compressed(
            output_path,
            scene_id=np.array(scene_id),
            split=np.array(split),
            obj_id=np.int32(obj_id),
            full_coords=full_coords,
            full_feats=full_feats,
            sparse_coords=sparse_coords,
            sparse_feats=sparse_feats,
            sparse_feats_on_full=sparse_feats_on_full,
            observed_mask=observed_mask,
            full_mean=np.asarray(full_obj["mean"], dtype=np.float32),
            full_scale=np.asarray(full_obj["scale"], dtype=np.float32),
            sparse_mean=np.asarray(sparse_obj["mean"], dtype=np.float32),
            sparse_scale=np.asarray(sparse_obj["scale"], dtype=np.float32),
        )

    return {
        "obj_id": obj_id,
        "path": str(output_path),
        "full_voxels": int(full_coords.shape[0]),
        "sparse_voxels": int(sparse_coords.shape[0]),
        "observed_voxels": int(observed_mask.sum()),
        "observed_ratio": round(
            float(observed_mask.sum()) / float(max(full_coords.shape[0], 1)), 6
        ),
        "subset_ok": len(extra_sparse_coords) == 0,
        "extra_sparse_coords": extra_sparse_coords[:8],
        "mean_close": bool(
            np.allclose(
                np.asarray(full_obj["mean"], dtype=np.float32),
                np.asarray(sparse_obj["mean"], dtype=np.float32),
            )
        ),
        "scale_close": bool(
            np.allclose(
                np.asarray(full_obj["scale"], dtype=np.float32),
                np.asarray(sparse_obj["scale"], dtype=np.float32),
            )
        ),
        "matched_sparse_voxels": matched_sparse,
    }


def build_scene_entry(
    scene: str,
    split: str,
    full_root: Path,
    sparse_root: Path,
    pair_dir: Path,
    objects_lookup: dict[str, dict],
    overwrite: bool,
) -> dict:
    full_emb = full_root / "files" / "gs_embeddings"
    sparse_emb = sparse_root / "files" / "gs_embeddings"
    sparse_feat3d = sparse_root / "files" / "Features3D" / "obj_dinov2_top10_l3"

    entry = {
        "scene_id": scene,
        "split": split,
        "full": {
            "slat": str(full_emb / f"{scene}_slat.npz"),
            "ulat": str(full_emb / f"{scene}_ulat.npz"),
        },
        "sparse": {
            "slat": str(sparse_emb / f"{scene}_slat.npz"),
            "ulat": str(sparse_emb / f"{scene}_ulat.npz"),
            "features3d": str(sparse_feat3d / f"{scene}.pkl.gz"),
        },
        "available": {
            "full_slat": (full_emb / f"{scene}_slat.npz").exists(),
            "full_ulat": (full_emb / f"{scene}_ulat.npz").exists(),
            "sparse_slat": (sparse_emb / f"{scene}_slat.npz").exists(),
            "sparse_ulat": (sparse_emb / f"{scene}_ulat.npz").exists(),
            "sparse_features3d": (sparse_feat3d / f"{scene}.pkl.gz").exists(),
        },
        "expected_object_ids": [
            int(obj["id"]) for obj in objects_lookup[scene]["objects"]
        ],
    }

    if not (entry["available"]["full_slat"] and entry["available"]["sparse_slat"]):
        entry["full_slat_obj_ids"] = []
        entry["sparse_slat_obj_ids"] = []
        entry["paired_object_ids"] = []
        entry["missing_in_full_slat"] = []
        entry["missing_in_sparse_slat"] = []
        entry["object_pairs"] = []
        return entry

    full_scene = load_scene_slat(full_emb / f"{scene}_slat.npz")
    sparse_scene = load_scene_slat(sparse_emb / f"{scene}_slat.npz")
    full_obj_ids = list(full_scene.keys())
    sparse_obj_ids = list(sparse_scene.keys())
    paired_obj_ids = [obj_id for obj_id in full_obj_ids if obj_id in sparse_scene]

    entry["full_slat_obj_ids"] = full_obj_ids
    entry["sparse_slat_obj_ids"] = sparse_obj_ids
    entry["paired_object_ids"] = paired_obj_ids
    entry["missing_in_full_slat"] = sorted(
        set(entry["expected_object_ids"]) - set(full_obj_ids)
    )
    entry["missing_in_sparse_slat"] = sorted(
        set(entry["expected_object_ids"]) - set(sparse_obj_ids)
    )
    entry["missing_sparse_voxel_objects"] = []
    for obj_id in entry["expected_object_ids"]:
        d = sparse_root / "files" / "gs_annotations" / scene / str(obj_id)
        if not (d / "voxel_output_dense.npz").exists():
            entry["missing_sparse_voxel_objects"].append(obj_id)

    object_pairs = []
    for obj_id in paired_obj_ids:
        output_path = pair_dir / split / scene / f"obj_{obj_id}.npz"
        pair_meta = build_object_pair(
            scene_id=scene,
            split=split,
            obj_id=obj_id,
            full_obj=full_scene[obj_id],
            sparse_obj=sparse_scene[obj_id],
            output_path=output_path,
            overwrite=overwrite,
        )
        object_pairs.append(pair_meta)

    entry["object_pairs"] = object_pairs
    return entry


def main():
    args = parse_args()
    full_root = Path(args.full_root)
    sparse_root = Path(args.sparse_root)
    benchmark_file = Path(args.benchmark_file)
    pair_dir = Path(args.pair_dir)
    output = Path(args.output)

    split_lookup = load_scene_splits(full_root)
    objects_lookup = load_objects_lookup(sparse_root)
    scenes = [x.strip() for x in benchmark_file.read_text().splitlines() if x.strip()]

    entries = [
        build_scene_entry(
            scene=scene,
            split=split_lookup.get(scene, "unknown"),
            full_root=full_root,
            sparse_root=sparse_root,
            pair_dir=pair_dir,
            objects_lookup=objects_lookup,
            overwrite=args.overwrite,
        )
        for scene in scenes
    ]

    all_object_pairs = [pair for entry in entries for pair in entry["object_pairs"]]
    summary = {
        "scene_count": len(entries),
        "complete_slat_pairs": sum(
            1
            for e in entries
            if e["available"]["full_slat"] and e["available"]["sparse_slat"]
        ),
        "complete_ulat_pairs": sum(
            1
            for e in entries
            if e["available"]["full_ulat"] and e["available"]["sparse_ulat"]
        ),
        "complete_features3d_pairs": sum(
            1 for e in entries if e["available"]["sparse_features3d"]
        ),
        "object_pair_count": len(all_object_pairs),
        "subset_ok_object_pairs": sum(1 for p in all_object_pairs if p["subset_ok"]),
        "non_subset_object_pairs": sum(1 for p in all_object_pairs if not p["subset_ok"]),
        "scene_problem_count": sum(
            1
            for e in entries
            if e["missing_in_full_slat"]
            or e["missing_in_sparse_slat"]
            or e["missing_sparse_voxel_objects"]
        ),
    }

    payload = {
        "full_root": str(full_root),
        "sparse_root": str(sparse_root),
        "benchmark_file": str(benchmark_file),
        "pair_dir": str(pair_dir),
        "summary": summary,
        "entries": entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
