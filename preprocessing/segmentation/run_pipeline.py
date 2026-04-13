"""
Full pipeline: SAM2 (masks) + MUSt3R (poses + depth).
Usage: python run_pipeline.py --config configs/pipeline.yaml
"""
import argparse, os, time, zipfile, torch, yaml
from pathlib import Path
from utils.io_utils import load_frames_from_scene, select_keyframes, select_keyframe_candidate_groups
from segment_sam2 import segment_keyframes, propagate_masks
from keyframe_selection import refine_keyframes_with_mask_preview
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

def parse_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

def source_root_from_cfg(cfg):
    env_root = os.environ.get("OBJECTX_SEG_INPUT_ROOT")
    return Path(env_root) if env_root else Path(cfg["dataset"]["root"])

def output_root_from_cfg(cfg):
    output_cfg = cfg.get("output", {})
    env_root = os.environ.get("OBJECTX_SEG_OUTPUT_ROOT")
    if env_root:
        return Path(env_root)
    output_root = output_cfg.get("root")
    if output_root:
        return Path(output_root)
    return Path(cfg["dataset"]["root"])

def output_name(cfg, env_key, cfg_key, default):
    return os.environ.get(env_key) or cfg.get("output", {}).get(cfg_key, default)

def step_enabled(cfg, env_key, cfg_key, default):
    return parse_bool(os.environ.get(env_key), cfg.get("steps", {}).get(cfg_key, default))


def env_or_default(name, default, cast=str):
    value = os.environ.get(name)
    if value is None:
        return default
    return cast(value)


def summarize_keyframes(indices, max_show=12):
    if len(indices) <= max_show:
        return str(indices)
    half = max_show // 2
    head = ", ".join(str(x) for x in indices[:half])
    tail = ", ".join(str(x) for x in indices[-half:])
    return f"[{head}, ..., {tail}]"

def get_scene_dirs(cfg):
    root = source_root_from_cfg(cfg)
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
    data_root = output_root_from_cfg(cfg)
    scenes_dirname = output_name(cfg, "OBJECTX_SEG_SCENES_DIRNAME", "scenes_dirname", "scenes_predicted")
    mask_dirname = output_name(cfg, "OBJECTX_SEG_MASK_DIRNAME", "mask_dirname", "gt_projection_predicted")
    vis_dirname = output_name(cfg, "OBJECTX_SEG_VIS_DIRNAME", "vis_dirname", "vis")
    dirs = {
        "depth": str(data_root / scenes_dirname / scene_id / "sequence"),
        "poses": str(data_root / scenes_dirname / scene_id / "sequence"),
        "pointmaps": str(data_root / scenes_dirname / scene_id / "sequence" / "pointmaps"),
        "masks": str(data_root / "files" / mask_dirname),
        "objects": str(data_root / "files"),
        "vis": str(data_root / scenes_dirname / scene_id / "sequence" / vis_dirname)
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    return dirs


def ensure_sequence_input(cfg, scene_id, dirs):
    source_root = source_root_from_cfg(cfg)
    image_subdir = cfg["dataset"].get("image_subdir", "sequence")
    image_ext = cfg["dataset"].get("image_ext", ".color.jpg")

    scene_root = source_root / "scenes" / scene_id
    src_seq = scene_root / image_subdir
    if src_seq.exists():
        return str(src_seq)

    src_zip = scene_root / f"{image_subdir}.zip"
    dst_seq = Path(dirs["depth"])
    if list(dst_seq.glob(f"*{image_ext}")):
        return str(dst_seq)

    if src_zip.exists():
        with zipfile.ZipFile(src_zip, "r") as zf:
            zf.extractall(dst_seq)
        return str(dst_seq)

    raise FileNotFoundError(
        f"Could not find {src_seq} or {src_zip} for scene {scene_id}"
    )

def run_scene(scene_id, scene_dir, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}\n Escena: {scene_id}  |  device: {device}\n{'='*60}")
    t0 = time.time()
    dirs = make_output_dirs(cfg, scene_id)
    scene_dir = ensure_sequence_input(cfg, scene_id, dirs)

    resize = cfg["dataset"].get("resize")
    if isinstance(resize, list): resize = tuple(resize)
    frames, frame_paths = load_frames_from_scene(
        scene_dir, cfg["dataset"].get("image_ext", ".color.jpg"), resize)

    kf_cfg = cfg["keyframes"]
    keyframe_strategy = env_or_default(
        "OBJECTX_SEG_KEYFRAME_STRATEGY", kf_cfg.get("strategy", "stride"), str
    )
    keyframe_stride = env_or_default(
        "OBJECTX_SEG_KEYFRAME_STRIDE", kf_cfg.get("stride", 10), int
    )
    keyframe_count = env_or_default(
        "OBJECTX_SEG_N_KEYFRAMES", kf_cfg.get("n_keyframes", 20), int
    )
    refine_keyframes = parse_bool(
        os.environ.get("OBJECTX_SEG_REFINE_KEYFRAMES"),
        kf_cfg.get("refine_keyframes", keyframe_strategy.startswith("quality_")),
    )
    preview_candidates = env_or_default(
        "OBJECTX_SEG_KEYFRAME_PREVIEW_CANDIDATES",
        kf_cfg.get("preview_candidates", 8),
        int,
    )
    candidate_groups = None
    if refine_keyframes and keyframe_strategy.startswith("quality_"):
        candidate_groups = select_keyframe_candidate_groups(
            frame_paths,
            keyframe_strategy,
            keyframe_stride,
            keyframe_count,
            frames=frames,
            top_k=preview_candidates,
        )
        if keyframe_strategy == "quality_global" and candidate_groups:
            keyframe_idxs = sorted(int(idx) for idx in candidate_groups[0][:keyframe_count])
        else:
            keyframe_idxs = [int(group[0]) for group in candidate_groups if group]
    else:
        keyframe_idxs = select_keyframes(
            frame_paths, keyframe_strategy, keyframe_stride, keyframe_count, frames=frames)
    print(
        "[Keyframes] "
        f"strategy={keyframe_strategy} stride={keyframe_stride} "
        f"n_keyframes={keyframe_count} selected={len(keyframe_idxs)} "
        f"indices={summarize_keyframes(keyframe_idxs)}"
    )

    run_must3r = step_enabled(cfg, "OBJECTX_SEG_RUN_MUST3R", "run_must3r", True)
    run_sam2 = step_enabled(cfg, "OBJECTX_SEG_RUN_SAM2", "run_sam2", True)
    run_registry = step_enabled(cfg, "OBJECTX_SEG_RUN_REGISTRY", "run_registry", True)

    if run_must3r:
        # STEP 1: MUSt3R → poses + depth (run first to free VRAM before SAM2)
        mc = cfg["must3r"]
        _, depths = run_must3r_on_scene(
            frame_paths, mc["checkpoint"], dirs["depth"], dirs["poses"],
            dirs["pointmaps"] if mc.get("output_pointmaps") else None,
            scene_id, mc.get("resolution", 512), mc.get("min_conf_thr",1.5), device)
        print(frames[0].shape[:2])
        print(depths[0].shape)
    else:
        print("[MUSt3R] skipped by configuration")

    if device == "cuda":
        torch.cuda.empty_cache()

    object_stats = {}
    if run_sam2:
        # STEP 2: SAM2 grid → masks on keyframes
        sc = cfg["sam2"]
        if refine_keyframes and keyframe_strategy.startswith("quality_"):
            print(
                "[Keyframes] preview candidates="
                + (
                    f"pool_size={len(candidate_groups[0])} "
                    f"head={candidate_groups[0][:max(16, preview_candidates * 4)]}"
                    if keyframe_strategy == "quality_global" and len(candidate_groups) == 1
                    else f"{[group[:preview_candidates] for group in candidate_groups]}"
                )
            )
            keyframe_idxs = refine_keyframes_with_mask_preview(
                frames,
                candidate_groups,
                sc,
                device,
                target_count=keyframe_count,
                scene_id=scene_id,
                diagnostics_root=os.environ.get(
                    "OBJECTX_SEG_KEYFRAME_DIAG_ROOT",
                    str(Path(__file__).resolve().parents[2] / "debug" / "keyframe_selection"),
                ),
            )
            print(
                "[Keyframes] refined "
                f"indices={summarize_keyframes(keyframe_idxs)}"
            )
        keyframe_masks = segment_keyframes(frames, keyframe_idxs, sc, device)

        # STEP 3: SAM2 VideoPredictor → propagate to all frames
        _, object_stats = propagate_masks(frame_paths=frame_paths, keyframe_masks=keyframe_masks, cfg=sc, output_dir=dirs["masks"], scan_id=scene_id, device=device)
    else:
        print("[SAM2] skipped by configuration")

    # STEP 4: Build and save object registry
    if run_registry and object_stats:
        objects_filename = output_name(
            cfg, "OBJECTX_SEG_OBJECTS_FILENAME", "objects_filename", "objects_predicted.json"
        )
        registry = build_objects_predicted(object_stats, scene_id)
        save_objects_predicted(registry, str(Path(dirs["objects"]) / objects_filename))
    elif run_registry:
        print("[Registry] skipped because no SAM2 tracks were produced")
    else:
        print("[Registry] skipped by configuration")

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
    # Keep the SAM2 Hydra config key unchanged; build_sam2 resolves it inside the sam2 package.
    cfg["must3r"]["checkpoint"] = str(resolve_model_path(cfg["must3r"]["checkpoint"]))

    for scene_id, scene_dir in get_scene_dirs(cfg):
        try: run_scene(scene_id, scene_dir, cfg)
        except Exception as e:
            print(f"[ERROR] {scene_id}: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
