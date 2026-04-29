"""
GT-geometry pipeline: SAM2 2D hypotheses -> 3D consolidation -> grouped 2D labels.
Usage: python run_pipeline.py --config configs/pipeline.yaml
"""
import argparse, os, time, torch, yaml, pickle
from pathlib import Path
from utils.io_utils import load_frames_from_scene, select_keyframes
from segment_sam2 import ensure_sam2_postprocess_ready, segment_keyframes, propagate_masks
from utils.object_registry import build_objects_predicted, save_objects_predicted
from sam2_helpers import parse_scan_frame_idx, save_obj_id_and_color_frames
from sam2_3d_track_merging import (
    GeometryProvider,
    merge_sam2_tracks_in_3d,
    compose_grouped_2d_labels,
)

def resolve_model_path(path):
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = Path(__file__).parent.parent.parent / path
    return candidate if candidate.exists() else p

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _validate_geometry_flags(cfg):
    gsrc = cfg.get("sam3d_merge", {}).get("geometry_source", {})
    use_gt_depth = bool(gsrc.get("use_gt_depth", False))
    use_pred_depth = bool(gsrc.get("use_pred_depth", False))
    use_gt_pose = bool(gsrc.get("use_gt_pose", False))
    use_pred_pose = bool(gsrc.get("use_pred_pose", False))
    if not (use_gt_depth and use_gt_pose):
        raise ValueError(
            "This clean pipeline is GT-geometry only. Set use_gt_depth=true and use_gt_pose=true."
        )
    if use_pred_depth or use_pred_pose:
        raise ValueError(
            "This clean pipeline is GT-geometry only. Set use_pred_depth=false and use_pred_pose=false."
        )

def get_scene_dirs(cfg):
    root = Path(cfg["dataset"]["root"])
    scene_id = cfg["dataset"]["scene_id"]
    subdir = cfg["dataset"].get("image_subdir", "sequence")
    scene_root = root / "scenes"
    if scene_id == "all":
        return sorted([
            (d.name, str(d / subdir))
            for d in scene_root.iterdir()
            if d.is_dir() and (d / subdir).exists()
        ])
    return [(scene_id, str(scene_root / scene_id / subdir))]

def make_output_dirs(cfg, scene_id):
    data_root = Path(cfg["dataset"]["root"])
    dirs = {
        "masks": str(data_root / "files" / "gt_projection_predicted"),
        "objects": str(data_root / "files"),
        "vis": str(data_root / "files" / "sam3d_debug" / scene_id)
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    return dirs

def run_scene(scene_id, scene_dir, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}\n Scene: {scene_id}  |  device: {device}\n{'='*60}")
    t0 = time.time()
    dirs = make_output_dirs(cfg, scene_id)

    scene_path = Path(scene_dir)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_path}")

    resize = cfg["dataset"].get("resize")
    if isinstance(resize, list): resize = tuple(resize)
    frames, frame_paths = load_frames_from_scene(
        scene_dir, cfg["dataset"].get("image_ext", ".color.jpg"), resize)
    if not frame_paths:
        raise ValueError(
            f"No frames found in {scene_path} matching '*{cfg['dataset'].get('image_ext', '.color.jpg')}'"
        )
    if len(frames) != len(frame_paths):
        raise ValueError(
            f"Loaded {len(frames)} frames but found {len(frame_paths)} frame paths for scene {scene_id}"
        )
    if any(frame is None for frame in frames):
        raise ValueError(f"Failed to decode one or more frames for scene {scene_id}")

    kf_cfg = cfg["keyframes"]
    keyframe_strategy = kf_cfg.get("strategy", "stride")
    keyframe_stride = kf_cfg.get("stride", 10)
    keyframe_count = kf_cfg.get("n_keyframes", 20)
    refine_keyframes = kf_cfg.get("refine_keyframes", keyframe_strategy == "heuristic")
    preview_candidates = kf_cfg.get("preview_candidates", 8)
    keyframe_idxs = select_keyframes(
        frame_paths,
        keyframe_strategy,
        keyframe_stride,
        keyframe_count,
        frames=frames,
        sam2_cfg=cfg.get("sam2"),
        device=device,
        preview_candidates=preview_candidates,
        refine_keyframes=refine_keyframes,
        scene_id=scene_id,
    )
    if not keyframe_idxs:
        raise ValueError(f"No keyframes selected for scene {scene_id}")
    if any(idx < 0 or idx >= len(frames) for idx in keyframe_idxs):
        raise ValueError(
            f"Keyframe indices out of range for scene {scene_id}: {keyframe_idxs}"
        )

    # STEP 1: Geometry mode (GT-only for clean pipeline)
    _validate_geometry_flags(cfg)
    print("[Geometry] Using GT depth + GT pose", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    # STEP 2: SAM2 grid → masks on keyframes
    sc = cfg["sam2"]
    require_postprocess = str(os.environ.get("OBJECTX_SAM2_REQUIRE_POSTPROCESS", "0")).strip().lower() not in {"0", "false", "no", "off", ""}
    if require_postprocess:
        ensure_sam2_postprocess_ready(device=device)
    keyframe_masks = segment_keyframes(frames, keyframe_idxs, sc, device)

    # STEP 3: SAM2 VideoPredictor → propagate to all frames (raw track hypotheses)
    tracks = propagate_masks(
        frame_paths=frame_paths,
        keyframe_masks=keyframe_masks,
        cfg=sc,
        device=device,
    )

    # STEP 4: 3D consolidation using GT geometry by default
    frame_ids = [f"{parse_scan_frame_idx(fp):06d}" for fp in frame_paths]
    gp = GeometryProvider(
        scene_id=scene_id,
        frame_ids=frame_ids,
        cfg=cfg,
        dataset_root=cfg["dataset"]["root"],
    )
    final_objects = merge_sam2_tracks_in_3d(
        tracks=tracks,
        frame_ids=frame_ids,
        scene_id=scene_id,
        cfg=cfg,
        geometry_provider=gp,
        debug_dir=dirs["vis"],
    )

    if not final_objects:
        raise ValueError(f"3D merger produced no objects for scene {scene_id}")

    h, w = frames[0].shape[:2]
    obj_id_imgs = compose_grouped_2d_labels(
        final_objects=final_objects,
        tracks=tracks,
        frame_ids=frame_ids,
        image_shape=(h, w),
        prob_thresh=cfg["sam3d_merge"]["masks"]["soft_prob_thresh"],
    )

    pkl_dir = Path(dirs["masks"]) / "obj_id_pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    save_obj_id_and_color_frames(obj_id_imgs, scan_id=scene_id, output_dir=dirs["masks"])
    with open(pkl_dir / f"{scene_id}.pkl", "wb") as f:
        pickle.dump({f"{int(k):06d}": v for k, v in obj_id_imgs.items()}, f)

    merged_tracks = {}
    for obj in final_objects:
        merged_tracks[int(obj.object_id)] = {}
        for fid in frame_ids:
            fid_int = int(fid)
            merged_tracks[int(obj.object_id)][fid_int] = (obj_id_imgs[fid_int] == int(obj.object_id))

    # STEP 5: Build and save object registry
    registry = build_objects_predicted(merged_tracks, scene_id)
    save_objects_predicted(registry, str(Path(dirs["objects"]) / "objects_predicted.json"))
    # data_root/3RScan/files/objects_predicted.json

    print(f"\n✓ {scene_id} completed in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="preprocessing/segmentation/pipeline.yaml")
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.scene: cfg["dataset"]["scene_id"] = args.scene
    
    cfg["sam2"]["checkpoint"] = str(resolve_model_path(cfg["sam2"]["checkpoint"]))
    for scene_id, scene_dir in get_scene_dirs(cfg):
        try: run_scene(scene_id, scene_dir, cfg)
        except Exception as e:
            print(f"[ERROR] {scene_id}: {e}")
            raise

if __name__ == "__main__":
    main()
