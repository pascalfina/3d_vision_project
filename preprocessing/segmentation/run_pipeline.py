"""
Full pipeline: SAM2 (masks) + MUSt3R (poses + depth).
Usage: python run_pipeline.py --config configs/pipeline.yaml
"""
import argparse, os, time, zipfile, torch, yaml
from pathlib import Path
from utils.io_utils import load_frames_from_scene, select_keyframes, select_keyframe_candidate_groups
from segment_sam2 import ensure_sam2_postprocess_ready, segment_keyframes, propagate_masks
from keyframe_selection import refine_keyframes_with_mask_preview
from depth_pose_must3r import run_must3r_on_scene
from depth_pose_mast3r_sfm import run_mast3r_sfm_on_scene
from depth_pose_pi3x import run_pi3x_on_scene
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
        # STEP 1: pose+depth backend — MUSt3R (default) or MASt3R-SfM
        backend = os.environ.get(
            "OBJECTX_POSE_DEPTH_BACKEND", cfg.get("pose_depth_backend", "must3r")
        ).lower()
        if backend == "mast3r_sfm":
            mc = cfg.get("mast3r_sfm", {})
            mast3r_kwargs = {
                "resolution": env_or_default(
                    "OBJECTX_MAST3R_SFM_RESOLUTION", mc.get("resolution", 512), int
                ),
                "min_conf_thr": env_or_default(
                    "OBJECTX_MAST3R_SFM_MIN_CONF_THR", mc.get("min_conf_thr", 0.5), float
                ),
                "scene_graph": env_or_default(
                    "OBJECTX_MAST3R_SFM_SCENE_GRAPH",
                    mc.get("scene_graph", "swin-15"),
                    str,
                ),
                "shared_intrinsics": parse_bool(
                    os.environ.get("OBJECTX_MAST3R_SFM_SHARED_INTRINSICS"),
                    mc.get("shared_intrinsics", True),
                ),
                "lr1": env_or_default(
                    "OBJECTX_MAST3R_SFM_LR1", mc.get("lr1", 0.07), float
                ),
                "niter1": env_or_default(
                    "OBJECTX_MAST3R_SFM_NITER1", mc.get("niter1", 300), int
                ),
                "lr2": env_or_default(
                    "OBJECTX_MAST3R_SFM_LR2", mc.get("lr2", 0.01), float
                ),
                "niter2": env_or_default(
                    "OBJECTX_MAST3R_SFM_NITER2", mc.get("niter2", 300), int
                ),
                "opt_depth": parse_bool(
                    os.environ.get("OBJECTX_MAST3R_SFM_OPT_DEPTH"),
                    mc.get("opt_depth", True),
                ),
                "matching_conf_thr": env_or_default(
                    "OBJECTX_MAST3R_SFM_MATCHING_CONF_THR",
                    mc.get("matching_conf_thr", 0.0),
                    float,
                ),
                "loss_dust3r_w": env_or_default(
                    "OBJECTX_MAST3R_SFM_LOSS_DUST3R_W",
                    mc.get("loss_dust3r_w", 0.01),
                    float,
                ),
                "subsample": env_or_default(
                    "OBJECTX_MAST3R_SFM_SUBSAMPLE", mc.get("subsample", 8), int
                ),
                "depth_mask_mode": env_or_default(
                    "OBJECTX_MAST3R_SFM_DEPTH_MASK_MODE",
                    mc.get("depth_mask_mode", "hard"),
                    str,
                ),
                "conf_smooth_sigma": env_or_default(
                    "OBJECTX_MAST3R_SFM_CONF_SMOOTH_SIGMA",
                    mc.get("conf_smooth_sigma", 0.0),
                    float,
                ),
                "conf_close_px": env_or_default(
                    "OBJECTX_MAST3R_SFM_CONF_CLOSE_PX",
                    mc.get("conf_close_px", 0),
                    int,
                ),
                "save_raw_depth": parse_bool(
                    os.environ.get("OBJECTX_MAST3R_SFM_SAVE_RAW_DEPTH"),
                    mc.get("save_raw_depth", False),
                ),
                "save_confidence_maps": parse_bool(
                    os.environ.get("OBJECTX_MAST3R_SFM_SAVE_CONFIDENCE"),
                    mc.get("save_confidence_maps", False),
                ),
                "pose_jump_max_translation": env_or_default(
                    "OBJECTX_MAST3R_SFM_POSE_JUMP_MAX_TRANSLATION",
                    mc.get("pose_jump_max_translation", 0.0),
                    float,
                ),
                "pose_jump_max_z_translation": env_or_default(
                    "OBJECTX_MAST3R_SFM_POSE_JUMP_MAX_Z_TRANSLATION",
                    mc.get("pose_jump_max_z_translation", 0.0),
                    float,
                ),
                "pose_jump_max_rotation_deg": env_or_default(
                    "OBJECTX_MAST3R_SFM_POSE_JUMP_MAX_ROTATION_DEG",
                    mc.get("pose_jump_max_rotation_deg", 0.0),
                    float,
                ),
                "pose_jump_relative_factor": env_or_default(
                    "OBJECTX_MAST3R_SFM_POSE_JUMP_RELATIVE_FACTOR",
                    mc.get("pose_jump_relative_factor", 0.0),
                    float,
                ),
                "zero_invalid_pose_depths": parse_bool(
                    os.environ.get("OBJECTX_MAST3R_SFM_ZERO_INVALID_POSE_DEPTHS"),
                    mc.get("zero_invalid_pose_depths", False),
                ),
                "cache_dir": os.environ.get(
                    "OBJECTX_MAST3R_SFM_CACHE_DIR", mc.get("cache_dir")
                ),
                "loss3d_backward_chunk_pairs": env_or_default(
                    "OBJECTX_MAST3R_SFM_LOSS3D_BACKWARD_CHUNK_PAIRS",
                    mc.get("loss3d_backward_chunk_pairs", 64),
                    int,
                ),
                "loss2d_backward_chunk_pairs": env_or_default(
                    "OBJECTX_MAST3R_SFM_LOSS2D_BACKWARD_CHUNK_PAIRS",
                    mc.get("loss2d_backward_chunk_pairs", 64),
                    int,
                ),
                "loss_dust3r_backward_chunk_pairs": env_or_default(
                    "OBJECTX_MAST3R_SFM_LOSS_DUST3R_BACKWARD_CHUNK_PAIRS",
                    mc.get("loss_dust3r_backward_chunk_pairs", 32),
                    int,
                ),
                "streaming_condense": parse_bool(
                    os.environ.get("OBJECTX_MAST3R_SFM_STREAMING_CONDENSE"),
                    mc.get("streaming_condense", True),
                ),
                "fast_loss_path": parse_bool(
                    os.environ.get("OBJECTX_MAST3R_SFM_FAST_LOSS_PATH"),
                    mc.get("fast_loss_path", False),
                ),
            }
            _, depths = run_mast3r_sfm_on_scene(
                frame_paths, mc["checkpoint"], dirs["depth"], dirs["poses"],
                dirs["pointmaps"] if mc.get("output_pointmaps") else None,
                scene_id, device=device, **mast3r_kwargs)
        elif backend == "pi3x":
            mc = cfg.get("pi3x", {})
            pi3x_kwargs = {
                "pixel_limit": env_or_default(
                    "OBJECTX_PI3X_PIXEL_LIMIT", mc.get("pixel_limit", 255000), int
                ),
                "chunk_size": env_or_default(
                    "OBJECTX_PI3X_CHUNK_SIZE", mc.get("chunk_size", 16), int
                ),
                "overlap": env_or_default(
                    "OBJECTX_PI3X_OVERLAP", mc.get("overlap", 6), int
                ),
                "conf_thr": env_or_default(
                    "OBJECTX_PI3X_CONF_THR", mc.get("conf_thr", 0.05), float
                ),
                "align_mode": str(
                    os.environ.get(
                        "OBJECTX_PI3X_ALIGN_MODE",
                        mc.get("align_mode", "se3"),
                    )
                ).strip().lower(),
                "save_raw_depth": parse_bool(
                    os.environ.get("OBJECTX_PI3X_SAVE_RAW_DEPTH"),
                    mc.get("save_raw_depth", True),
                ),
                "save_confidence_maps": parse_bool(
                    os.environ.get("OBJECTX_PI3X_SAVE_CONFIDENCE"),
                    mc.get("save_confidence_maps", mc.get("save_confidence", True)),
                ),
                "pose_jump_max_translation": env_or_default(
                    "OBJECTX_PI3X_POSE_JUMP_MAX_TRANSLATION",
                    mc.get("pose_jump_max_translation", 0.0),
                    float,
                ),
                "pose_jump_max_z_translation": env_or_default(
                    "OBJECTX_PI3X_POSE_JUMP_MAX_Z_TRANSLATION",
                    mc.get("pose_jump_max_z_translation", 0.0),
                    float,
                ),
                "pose_jump_max_rotation_deg": env_or_default(
                    "OBJECTX_PI3X_POSE_JUMP_MAX_ROTATION_DEG",
                    mc.get("pose_jump_max_rotation_deg", 0.0),
                    float,
                ),
                "pose_jump_relative_factor": env_or_default(
                    "OBJECTX_PI3X_POSE_JUMP_RELATIVE_FACTOR",
                    mc.get("pose_jump_relative_factor", 0.0),
                    float,
                ),
                "zero_invalid_pose_depths": parse_bool(
                    os.environ.get("OBJECTX_PI3X_ZERO_INVALID_POSE_DEPTHS"),
                    mc.get("zero_invalid_pose_depths", False),
                ),
                "anchor_count": env_or_default(
                    "OBJECTX_PI3X_ANCHOR_COUNT",
                    mc.get("anchor_count", 0),
                    int,
                ),
                "use_intrinsics": parse_bool(
                    os.environ.get("OBJECTX_PI3X_USE_INTRINSICS"),
                    mc.get("use_intrinsics", False),
                ),
                "local_pixel_limit": env_or_default(
                    "OBJECTX_PI3X_LOCAL_PIXEL_LIMIT",
                    mc.get("local_pixel_limit", 0),
                    int,
                ),
                "local_chunk_size": env_or_default(
                    "OBJECTX_PI3X_LOCAL_CHUNK_SIZE",
                    mc.get("local_chunk_size", 25),
                    int,
                ),
                "local_overlap": env_or_default(
                    "OBJECTX_PI3X_LOCAL_OVERLAP",
                    mc.get("local_overlap", 8),
                    int,
                ),
                "local_scale_max_ratio": env_or_default(
                    "OBJECTX_PI3X_LOCAL_SCALE_MAX_RATIO",
                    mc.get("local_scale_max_ratio", 1.35),
                    float,
                ),
                "output_native_resolution": parse_bool(
                    os.environ.get("OBJECTX_PI3X_OUTPUT_NATIVE_RES"),
                    mc.get("output_native_resolution", False),
                ),
                "frame_stride": env_or_default(
                    "OBJECTX_PI3X_FRAME_STRIDE",
                    mc.get("frame_stride", 1),
                    int,
                ),
            }
            _, depths = run_pi3x_on_scene(
                frame_paths,
                mc.get("checkpoint"),
                dirs["depth"],
                dirs["poses"],
                dirs["pointmaps"] if mc.get("output_pointmaps") else None,
                scene_id,
                device=device,
                **pi3x_kwargs,
            )
        else:
            mc = cfg["must3r"]
            must3r_kwargs = {
            "resolution": env_or_default(
                "OBJECTX_MUST3R_RESOLUTION", mc.get("resolution", 512), int
            ),
            "min_conf_thr": env_or_default(
                "OBJECTX_MUST3R_MIN_CONF_THR", mc.get("min_conf_thr", 1.5), float
            ),
            "execution_mode": env_or_default(
                "OBJECTX_MUST3R_EXECUTION_MODE",
                mc.get("execution_mode", "linseq"),
                str,
            ),
            "num_mem_images": env_or_default(
                "OBJECTX_MUST3R_NUM_MEM_IMAGES", mc.get("num_mem_images", 50), int
            ),
            "num_refinements_iterations": env_or_default(
                "OBJECTX_MUST3R_NUM_REFINEMENTS_ITERATIONS",
                mc.get("num_refinements_iterations", 0),
                int,
            ),
            "vidseq_local_context_size": env_or_default(
                "OBJECTX_MUST3R_VIDSEQ_LOCAL_CONTEXT_SIZE",
                mc.get("vidseq_local_context_size", 0),
                int,
            ),
            "keyframe_interval": env_or_default(
                "OBJECTX_MUST3R_KEYFRAME_INTERVAL",
                mc.get("keyframe_interval", 3),
                int,
            ),
            "slam_local_context_size": env_or_default(
                "OBJECTX_MUST3R_SLAM_LOCAL_CONTEXT_SIZE",
                mc.get("slam_local_context_size", 0),
                int,
            ),
            "subsample": env_or_default(
                "OBJECTX_MUST3R_SUBSAMPLE", mc.get("subsample", 2), int
            ),
            "min_conf_keyframe": env_or_default(
                "OBJECTX_MUST3R_MIN_CONF_KEYFRAME",
                mc.get("min_conf_keyframe", 1.5),
                float,
            ),
            "keyframe_overlap_thr": env_or_default(
                "OBJECTX_MUST3R_KEYFRAME_OVERLAP_THR",
                mc.get("keyframe_overlap_thr", 0.05),
                float,
            ),
            "overlap_percentile": env_or_default(
                "OBJECTX_MUST3R_OVERLAP_PERCENTILE",
                mc.get("overlap_percentile", 85),
                float,
            ),
            "depth_mask_mode": env_or_default(
                "OBJECTX_MUST3R_DEPTH_MASK_MODE",
                mc.get("depth_mask_mode", "hard"),
                str,
            ),
            "save_raw_depth": parse_bool(
                os.environ.get("OBJECTX_MUST3R_SAVE_RAW_DEPTH"),
                mc.get("save_raw_depth", False),
            ),
            "save_confidence_maps": parse_bool(
                os.environ.get("OBJECTX_MUST3R_SAVE_CONFIDENCE"),
                mc.get("save_confidence_maps", False),
            ),
            "pose_jump_max_translation": env_or_default(
                "OBJECTX_MUST3R_POSE_JUMP_MAX_TRANSLATION",
                mc.get("pose_jump_max_translation", 0.0),
                float,
            ),
            "pose_jump_max_z_translation": env_or_default(
                "OBJECTX_MUST3R_POSE_JUMP_MAX_Z_TRANSLATION",
                mc.get("pose_jump_max_z_translation", 0.0),
                float,
            ),
            "pose_jump_max_rotation_deg": env_or_default(
                "OBJECTX_MUST3R_POSE_JUMP_MAX_ROTATION_DEG",
                mc.get("pose_jump_max_rotation_deg", 0.0),
                float,
            ),
            "pose_jump_relative_factor": env_or_default(
                "OBJECTX_MUST3R_POSE_JUMP_RELATIVE_FACTOR",
                mc.get("pose_jump_relative_factor", 0.0),
                float,
            ),
            "zero_invalid_pose_depths": parse_bool(
                os.environ.get("OBJECTX_MUST3R_ZERO_INVALID_POSE_DEPTHS"),
                mc.get("zero_invalid_pose_depths", False),
            ),
        }
            _, depths = run_must3r_on_scene(
                frame_paths, mc["checkpoint"], dirs["depth"], dirs["poses"],
                dirs["pointmaps"] if mc.get("output_pointmaps") else None,
                scene_id, device=device, **must3r_kwargs)
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
        require_postprocess = parse_bool(
            os.environ.get("OBJECTX_SAM2_REQUIRE_POSTPROCESS"),
            True,
        )
        if require_postprocess:
            ensure_sam2_postprocess_ready(device=sc.get("device", device))
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
