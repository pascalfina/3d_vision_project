"""
Full pipeline: SAM2 (masks) + MUSt3R (poses + depth).
Usage: python run_pipeline.py --config configs/pipeline.yaml
"""
import argparse, os, time, torch, yaml
from pathlib import Path
from utils.io_utils import load_frames_from_scene, select_keyframes
from segment_sam2 import segment_keyframes, propagate_masks
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
    print(f"\n{'='*60}\n Escena: {scene_id}  |  device: {device}\n{'='*60}")
    t0 = time.time()
    dirs = make_output_dirs(cfg, scene_id)

    resize = cfg["dataset"].get("resize")
    if isinstance(resize, list): resize = tuple(resize)
    frames, frame_paths = load_frames_from_scene(
        scene_dir, cfg["dataset"].get("image_ext", ".color.jpg"), resize)

    kf_cfg = cfg["keyframes"]
    keyframe_idxs = select_keyframes(
        frame_paths, kf_cfg.get("strategy","stride"),
        kf_cfg.get("stride",10), kf_cfg.get("n_keyframes",20))

    # STEP 1: MUSt3R → poses + depth (run first to free VRAM before SAM2)
    mc = cfg["must3r"]
    _, depths = run_must3r_on_scene(
        frame_paths, mc["checkpoint"], dirs["depth"], dirs["poses"],
        dirs["pointmaps"] if mc.get("output_pointmaps") else None,
        scene_id, mc.get("resolution", 512), mc.get("min_conf_thr",1.5), device)
    
    print(frames[0].shape[:2])
    print(depths[0].shape)

    if device == "cuda":
        torch.cuda.empty_cache()

    # STEP 2: SAM2 grid → masks on keyframes
    sc = cfg["sam2"]
    keyframe_masks = segment_keyframes(frames, keyframe_idxs, sc, device)

    # STEP 3: SAM2 VideoPredictor → propagate to all frames
    _, merged_tracks = propagate_masks(frame_paths=frame_paths, keyframe_masks=keyframe_masks, cfg=sc, output_dir=dirs["masks"], scan_id=scene_id, device=device)

    # STEP 4: Build and save object registry
    registry = build_objects_predicted(merged_tracks, scene_id)
    save_objects_predicted(registry, str(Path(dirs["objects"]) / "objects_predicted.json"))
    # data_root/3RScan/files/objects_predicted.json

    print(f"\n✓ {scene_id} completada en {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="preprocessing/segmentation/pipeline.yaml")
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.scene: cfg["dataset"]["scene_id"] = args.scene
    
    # Resolve model paths relative to repo root if needed
    cfg["sam2"]["checkpoint"] = str(resolve_model_path(cfg["sam2"]["checkpoint"]))
    cfg["sam2"]["model_cfg"] = str(resolve_model_path(cfg["sam2"]["model_cfg"]))
    cfg["must3r"]["checkpoint"] = str(resolve_model_path(cfg["must3r"]["checkpoint"]))

    for scene_id, scene_dir in get_scene_dirs(cfg):
        try: run_scene(scene_id, scene_dir, cfg)
        except Exception as e:
            print(f"[ERROR] {scene_id}: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()