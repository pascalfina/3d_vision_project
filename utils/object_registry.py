# utils/object_registry.py

import json
import numpy as np
from pathlib import Path
from typing import List


def build_objects_predicted(
    object_source,
    scene_id: str,
) -> dict:
    """
    Builds a dict with the same schema as 3RScan's objects.json,
    but with objects predicted by SAM2.

    Args:
        object_source:
            either {obj_id -> {"n_frames", "first_frame", ...}}
            or the older {obj_id -> {frame_idx -> binary mask}} format.
        scene_id: scan identifier string

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
          "area_max": float,     # max pixel area across frames
          "median_depth": float  # (optional) median depth at first frame
        }]
      }]
    }
    """
    objects = []

    for obj_id, value in object_source.items():
        if isinstance(value, dict) and "n_frames" in value:
            obj_entry = {
                "id": str(value.get("id", obj_id)),
                "n_frames": int(value["n_frames"]),
                "first_frame": int(value["first_frame"]),
                "last_frame": int(value["last_frame"]),
                "area_mean": float(value["area_mean"]),
                "area_max": float(value["area_max"]),
            }
        else:
            frame_dict = value
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
    Appends or creates objects_predicted.json in out_dir.
    If the file already exists, the new scan entry is merged in.

    Args:
        registry: output of build_objects_predicted()
        out_dir:  directory where objects_predicted.json will be written
        scene_id: used only for the log message
    Returns:
        absolute path to the written file
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[Registry] Stored: {out_path}")
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
