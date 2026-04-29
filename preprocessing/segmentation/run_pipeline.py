"""
Full pipeline: SAM2 (masks) + MUSt3R (poses + depth).
Usage: python run_pipeline.py --config configs/pipeline.yaml
"""
import argparse, os, time, torch, yaml
from pathlib import Path
from utils.io_utils import load_frames_from_scene, select_keyframes
from segment_sam2 import ensure_sam2_postprocess_ready, segment_keyframes, propagate_masks
from depth_pose_must3r import run_must3r_on_scene
from utils.object_registry import build_objects_predicted, save_objects_predicted

def resolve_model_path(path):
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = Path(__file__).parent.parent.parent / path
    return candidate if candidate.exists() else p

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

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
        "depth": str(data_root / "scenes_predicted" / scene_id / "sequence"),
        "poses": str(data_root / "scenes_predicted" / scene_id / "sequence"),
        "pointmaps": str(data_root / "scenes_predicted" / scene_id / "sequence" / "pointmaps"),
        "masks": str(data_root / "files" / "gt_projection_predicted"),
        "objects": str(data_root / "files"),
        "vis": str(data_root / "scenes_predicted" / scene_id / "sequence" / "vis")
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

    # STEP 1: MUSt3R → poses + depth (run first to free VRAM before SAM2)
    mc = cfg["must3r"]
    _, depths = run_must3r_on_scene(
        frame_paths, mc["checkpoint"], dirs["depth"], dirs["poses"],
        dirs["pointmaps"] if mc.get("output_pointmaps") else None,
        scene_id, mc.get("resolution", 512), mc.get("min_conf_thr",1.5), device)
    if not depths:
        raise ValueError(f"MUSt3R returned no depth maps for scene {scene_id}")
    if len(depths) != len(frame_paths):
        raise ValueError(
            f"MUSt3R returned {len(depths)} depth maps for {len(frame_paths)} input frames in scene {scene_id}"
        )

    if device == "cuda":
        torch.cuda.empty_cache()

    # STEP 2: SAM2 grid → masks on keyframes
    sc = cfg["sam2"]
    require_postprocess = str(os.environ.get("OBJECTX_SAM2_REQUIRE_POSTPROCESS", "0")).strip().lower() not in {"0", "false", "no", "off", ""}
    if require_postprocess:
        ensure_sam2_postprocess_ready(device=device)
    keyframe_masks = segment_keyframes(frames, keyframe_idxs, sc, device)

    # STEP 3: SAM2 VideoPredictor → propagate to all frames
    _, merged_tracks = propagate_masks(frame_paths=frame_paths, keyframe_masks=keyframe_masks, cfg=sc, output_dir=dirs["masks"], scan_id=scene_id, device=device)

    # STEP 4: Build and save object registry
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
    cfg["must3r"]["checkpoint"] = str(resolve_model_path(cfg["must3r"]["checkpoint"]))

    for scene_id, scene_dir in get_scene_dirs(cfg):
        try: run_scene(scene_id, scene_dir, cfg)
        except Exception as e:
            print(f"[ERROR] {scene_id}: {e}")
            raise

if __name__ == "__main__":
    main()
