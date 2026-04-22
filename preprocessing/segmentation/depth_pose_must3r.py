"""
MUSt3R: predicts poses (N,4,4) and depth maps (H,W) from RGB only.
Depth is extracted from the Z channel of the pointmaps (pts3d[...,2]).
"""
import os, sys, numpy as np, torch, cv2
from pathlib import Path
from typing import List, Optional, Tuple
from utils.io_utils import save_depth, save_poses, save_sequence_info


def _parse_scan_frame_idx(frame_path):
    stem = Path(frame_path).stem
    return int(stem.split(".")[0].split("-")[-1])


def _image_size(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image for size extraction: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _as_scalar(value):
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return float(array.reshape(-1)[0])


def _intrinsic_from_focal(focal, width, height):
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return np.array(
        [[float(focal), 0.0, cx], [0.0, float(focal), cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def run_must3r_on_scene(
    frame_paths, checkpoint, output_depth_dir, output_poses_dir,
    output_pointmaps_dir, scene_id,
    resolution=512, min_conf_thr=1.5, device="cuda"
):
    """
    Runs MUSt3R in multi-view mode over the entire scene.
    Returns: poses (N,4,4), depths list of (H,W).
    """
    print(f"\n[MUSt3R] Predicting poses+depth for {len(frame_paths)} frames")
    frame_idxs = [_parse_scan_frame_idx(fp) for fp in frame_paths]
    must3r_path = os.environ.get("MUST3R_PATH", "must3r")
    if must3r_path not in sys.path:
        sys.path.insert(0, must3r_path)

    from must3r.model import load_model
    from must3r.demo.gradio import get_reconstructed_scene
    
    # Load checkpoint and create model with args
    model = load_model(
        checkpoint,
        device=device,
        img_size=resolution,
        encoder=None,
        decoder=None,
        memory_mode=None,
    )

    with torch.no_grad():
        scene, _ = get_reconstructed_scene(
            outdir=output_pointmaps_dir or "/tmp/must3r_tmp",
            viser_server=None,
            should_save_glb=False,
            model=model,
            retrieval=None,
            device=device,
            verbose=True,
            image_size=resolution,
            amp=False,
            filelist=frame_paths,
            min_conf_thr=min_conf_thr,
            as_pointcloud=True,
            transparent_cams=False,
            local_pointmaps=False,
            cam_size=0.05,
            num_mem_images=min(50, len(frame_paths)),
            max_bs=1,
            render_once=False,
            camera_conf_thr=0.0,
            num_refinements_iterations=0,
            execution_mode="linseq",
            vidseq_local_context_size=0,
            keyframe_interval=3,
            slam_local_context_size=0,
            subsample=2,
            min_conf_keyframe=1.5,
            keyframe_overlap_thr=0.05,
            overlap_percentile=85,
        ) # dict_keys(['x_out', 'imgs', 'true_shape', 'focals', 'cams2world', 'image_list'])
    
    # Extract poses, pts3d, conf_masks from the scene object
    pts3d = [o["pts3d"] for o in scene.x_out] # 3D pointmap of each frame in global coords
    conf_masks = [o["conf"] > min_conf_thr for o in scene.x_out]
    poses_c2w = torch.stack([o["c2w"] for o in scene.x_out], dim=0) # Shape: (N, 4, 4)
    focal_values = []
    if hasattr(scene, "focals") and scene.focals is not None:
        focal_values = [_as_scalar(focal) for focal in scene.focals]
    elif scene.x_out and "focal" in scene.x_out[0]:
        focal_values = [_as_scalar(o["focal"]) for o in scene.x_out]
    
    # o["pts3d"]       : 3D pointmap of the frame in global coordinates
    # o["pts3d_local"] : pointmap in local camera coordinates
    # o["conf"]        : per-pixel/point confidence
    # o["focal"]       : estimated focal length
    # o["c2w"]         : camera-to-world pose

    # Depth → Z channel of each local pointmap
    depths = []
    pts3d_local = [o["pts3d_local"] for o in scene.x_out]

    for frame_idx, (pts_local, mask) in zip(frame_idxs, zip(pts3d_local, conf_masks)):
        pts_local_np = pts_local.cpu().numpy() if hasattr(pts_local, "cpu") else pts_local
        mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else mask

        depth = pts_local_np[..., 2].astype(np.float32)
        depth[~mask_np] = 0.0
        
        #depth = cv2.resize(depth, (224, 172), interpolation=cv2.INTER_NEAREST)
        depths.append(depth)
        save_depth(depth, output_depth_dir, frame_idx=frame_idx)

    save_poses(poses_c2w, output_poses_dir, scene_id, frame_idxs=frame_idxs)
    depth_width, depth_height = int(depths[0].shape[1]), int(depths[0].shape[0])
    color_width, color_height = _image_size(frame_paths[0])
    if focal_values:
        focal_depth = float(np.median(focal_values))
        intrinsics_source = "must3r_focal_estimates"
    else:
        # Fallback: keep writing Object-X compatible camera files even when
        # MUSt3R does not expose focal estimates for this run.
        focal_depth = float(max(depth_width, depth_height))
        intrinsics_source = "fallback_image_extent"

    scale_x = color_width / max(depth_width, 1)
    scale_y = color_height / max(depth_height, 1)
    focal_color_x = focal_depth * scale_x
    focal_color_y = focal_depth * scale_y
    depth_intrinsic = _intrinsic_from_focal(focal_depth, depth_width, depth_height)
    color_intrinsic = np.array(
        [
            [focal_color_x, 0.0, (color_width - 1.0) * 0.5],
            [0.0, focal_color_y, (color_height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    save_sequence_info(
        output_dir=output_poses_dir,
        color_size=(color_width, color_height),
        depth_size=(depth_width, depth_height),
        color_intrinsic=color_intrinsic,
        depth_intrinsic=depth_intrinsic,
        n_frames=len(frame_paths),
        depth_shift=1000.0,
        metadata={
            "scene_id": scene_id,
            "intrinsics_source": intrinsics_source,
            "frame_ids": [int(frame_idx) for frame_idx in frame_idxs],
            "focal_depth_median": focal_depth,
            "focal_depth_per_frame": [float(value) for value in focal_values],
            "depth_size": {"width": depth_width, "height": depth_height},
            "color_size": {"width": color_width, "height": color_height},
        },
    )
    print(f"[MUSt3R] Poses: {poses_c2w.shape} | Depth shape: {depths[0].shape}")
    del model; torch.cuda.empty_cache()
    return poses_c2w, depths


def depth_per_instance(depth_map, masks, p_low=5.0, p_high=95.0):
    """
    Robust per-instance depth, filtering border outliers
    via percentile clipping before computing statistics.
    """
    results = []
    for i, mask in enumerate(masks):
        vals = depth_map[mask]
        vals = vals[vals > 0] # discard zero/invalid depth
        if len(vals) < 10:
            results.append({"obj_id": i, "median_depth": 0.0}); continue
        lo, hi = np.percentile(vals, [p_low, p_high])
        filt = vals[(vals >= lo) & (vals <= hi)]
        results.append({
            "obj_id": i, "median_depth": float(np.median(filt)),
            "mean_depth": float(np.mean(filt)), "std_depth": float(np.std(filt)),
            "n_pixels": int(mask.sum())
        })
    return results
