# utils/object_registry.py

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def build_objects_predicted(
    merged_tracks: Dict[int, Dict[int, np.ndarray]],
    scene_id: str,
    depths: List[np.ndarray] = None,
) -> dict:
    """
    Genera un dict con el mismo schema que objects.json de 3RScan,
    pero con objetos predichos por SAM2.

    Schema:
    {
      "scans": [{
        "scan": scene_id,
        "objects": [{
          "id": str,            # obj_id predicho (0-indexed)
          "n_frames": int,      # en cuántos frames aparece
          "first_frame": int,   # primer frame donde aparece
          "area_mean": float,   # área media en píxeles
          "median_depth": float # depth mediano (si depths disponibles)
        }]
      }]
    }
    """
    objects = []
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

        # Depth mediano del objeto en su primer frame
        if depths is not None and frame_idxs[0] < len(depths):
            depth = depths[frame_idxs[0]]
            mask = frame_dict[frame_idxs[0]]
            vals = depth[mask]
            vals = vals[vals > 0]
            if len(vals) > 0:
                obj_entry["median_depth"] = float(np.median(vals))

        objects.append(obj_entry)

    return {
        "scans": [{
            "scan": scene_id,
            "objects": sorted(objects, key=lambda x: int(x["id"]))
        }]
    }


def save_objects_predicted(registry: dict, out_dir: str, scene_id: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / f"{scene_id}_objects_predicted.json")
    with open(out_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[Registry] Guardado: {out_path}")
    return out_path


def load_gt_objects(objects_json_path: str, scene_id: str) -> List[dict]:
    """
    Carga los objetos GT de objects.json de 3RScan para una escena.
    Útil para comparar IDs predichos vs GT.
    """
    with open(objects_json_path) as f:
        data = json.load(f)
    for scan in data["scans"]:
        if scan["scan"] == scene_id:
            return scan["objects"]
    return []