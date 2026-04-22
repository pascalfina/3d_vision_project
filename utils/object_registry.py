# utils/object_registry.py

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def _validate_registry(registry: dict) -> dict:
    scans = registry.get("scans")
    if not isinstance(scans, list) or len(scans) != 1:
        raise ValueError("Predicted registry must contain exactly one scan entry")

    scan_entry = scans[0]
    if not isinstance(scan_entry, dict):
        raise ValueError("Predicted scan entry must be a dict")
    if "scan" not in scan_entry or "objects" not in scan_entry:
        raise ValueError("Predicted scan entry must contain 'scan' and 'objects'")
    if not isinstance(scan_entry["objects"], list):
        raise ValueError("Predicted scan entry field 'objects' must be a list")
    return scan_entry


def build_objects_predicted(
    merged_tracks: Dict[int, Dict[int, np.ndarray]],
    scene_id: str,
) -> dict:
    """
    Builds a predicted object registry with the same outer structure as
    3RScan's objects.json: {"scans": [{"scan": ..., "objects": [...]}]}.

    Predicted objects only include fields that are available or useful for the
    current pipeline, instead of trying to emulate every GT-only annotation
    field such as labels or affordances.

    Args:
        obj_id_imgs: {scan_fidx → (H, W) int32}  — one ID map per frame
        scene_id:    scan identifier string
        depths:      optional list of (H, W) float32 depth maps, indexed by
                     position in frame_paths (not scan_fidx)

    Schema:
    {
      "scans": [{
        "scan": scene_id,
        "objects": [{
          "id": str,             # predicted obj_id (1-indexed, 0 = background)
          "n_frames": int,       # number of frames where the object appears
          "first_frame": int,    # first scan frame index
          "last_frame": int,     # last scan frame index
          "area_mean": float,    # mean pixel area across frames
          "area_max": float      # max pixel area across frames
        }]
      }]
    }
    """
    objects = [] # merged_tracks: {..., obj_id: # (540, 960), ...}
    for obj_id, frame_dict in merged_tracks.items():
        frame_idxs = sorted(frame_dict.keys())
        areas = [frame_dict[f].sum() for f in frame_idxs]

        obj_entry = {
            "id": str(obj_id),
            "n_frames": len(frame_idxs),
            "first_frame": int(frame_idxs[0]),
            "last_frame": int(frame_idxs[-1]),
            "area_mean": float(np.mean(areas)),
            "area_max": float(np.max(areas)),
        }

        objects.append(obj_entry)

    return {
        "scans": [{
            "scan": scene_id,
            "objects": sorted(objects, key=lambda x: int(x["id"]))
        }]
    }


def save_objects_predicted(registry: dict, out_path: str):
    """
    Creates or updates objects_predicted.json.

    If the file already exists, the scan entry in ``registry`` replaces the
    existing entry with the same scan id and all other scan entries are kept.

    Args:
        registry: output of build_objects_predicted()
        out_path: file path where objects_predicted.json will be written
    Returns:
        absolute path to the written file
    """
    scan_entry = _validate_registry(registry)
    scan_id = scan_entry["scan"]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged = {"scans": []}
    if Path(out_path).exists():
        with open(out_path) as f:
            existing = json.load(f)
        if not isinstance(existing, dict) or not isinstance(existing.get("scans"), list):
            raise ValueError(f"Existing predicted registry has invalid format: {out_path}")
        merged["scans"] = [
            entry for entry in existing["scans"]
            if isinstance(entry, dict) and entry.get("scan") != scan_id
        ]

    merged["scans"].append(scan_entry)
    merged["scans"].sort(key=lambda entry: entry["scan"])

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[Registry] Stored scan {scan_id}: {out_path}")
    return out_path


def load_gt_objects(objects_json_path: str, scene_id: str) -> List[dict]:
    """
    Load GT objects from 3RScan's objects.json for a given scene.
    Useful for comparing predicted IDs against ground truth.
    """
    with open(objects_json_path) as f:
        data = json.load(f)
    for scan in data["scans"]:
        if scan["scan"] == scene_id:
            return scan["objects"]
    return []
