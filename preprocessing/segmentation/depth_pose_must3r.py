"""
MUSt3R: predice poses (N,4,4) y depth maps (H,W) desde solo RGB.
El depth se extrae del canal Z de los pointmaps (pts3d[...,2]).
"""
import os, sys, numpy as np, torch, cv2
from typing import List, Optional, Tuple
from utils.io_utils import save_depth, save_poses


def run_must3r_on_scene(
    frame_paths, checkpoint, output_depth_dir, output_poses_dir,
    output_pointmaps_dir, scene_id,
    resolution=512, min_conf_thr=1.5, device="cuda"
):
    """
    Ejecuta MUSt3R en modo multi-view sobre toda la escena.
    Retorna: poses (N,4,4), depths lista de (H,W).
    """
    print(f"\n[MUSt3R] Prediciendo poses+depth para {len(frame_paths)} frames")
    must3r_path = os.environ.get("MUST3R_PATH", "must3r")
    if must3r_path not in sys.path:
        sys.path.insert(0, must3r_path)

    from must3r.model import load_model
    from must3r.demo.gradio import get_reconstructed_scene
    from dust3r.utils.image import load_images
    
    # Cargar checkpoint y crear modelo con args
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
    
    # Extraer poses, pts3d, conf_masks del scene object
    pts3d = [o["pts3d"] for o in scene.x_out] # pointmap 3D del frame en coordenades globals
    conf_masks = [o["conf"] > min_conf_thr for o in scene.x_out]
    poses_c2w = torch.stack([o["c2w"] for o in scene.x_out], dim=0) # Shape: (N, 4, 4)
    poses_w2c = torch.linalg.inv(poses_c2w)
    # o["pts3d"]: pointmap 3D del frame en coordenades globals o["pts3d_local"]: pointmap en coordenades locals de càmera o["conf"]: confiança per píxel/punt o["focal"]: focal estimada o["c2w"]: pose camera-to-world

    # Depth → canal Z de cada pointmap
    depths = []
    pts3d_local = [o["pts3d_local"] for o in scene.x_out]

    for i, (pts_local, mask) in enumerate(zip(pts3d_local, conf_masks)):
        pts_local_np = pts_local.cpu().numpy() if hasattr(pts_local, "cpu") else pts_local
        mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else mask

        depth = pts_local_np[..., 2].astype(np.float32)
        depth[~mask_np] = 0.0
        
        depth = cv2.resize(depth, (224, 172), interpolation=cv2.INTER_NEAREST)
        depths.append(depth)
        save_depth(depth, output_depth_dir, frame_idx=i)

    save_poses(poses_w2c, output_poses_dir, scene_id)
    print(f"[MUSt3R] Poses: {poses_w2c.shape} | Depth shape: {depths[0].shape}")
    del model; torch.cuda.empty_cache()
    return poses_w2c, depths


def depth_per_instance(depth_map, masks, p_low=5.0, p_high=95.0):
    """Depth robusto por instancia, filtrando outliers en los bordes."""
    results = []
    for i, mask in enumerate(masks):
        vals = depth_map[mask]
        vals = vals[vals > 0]
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