"""Unit-test the MASt3R-SfM wrapper on 50 cabinet frames."""
import os, sys, tempfile, shutil, time

REPO = "/work/scratch/pafina/object-x"
sys.path.insert(0, os.path.join(REPO, "preprocessing", "segmentation"))
sys.path.insert(0, REPO)

from depth_pose_mast3r_sfm import run_mast3r_sfm_on_scene

SCENE_ID = "e61b0e04-bada-2f31-82d6-72831a602ba7"
SEQ_DIR = f"/work/scratch/pafina/objectx-data-fullscene-cabinet-hybrid-sam2mask-must3r/scenes_sam2_must3r/{SCENE_ID}/sequence"
CKPT = "/work/scratch/pafina/object-x/models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

N = 50
STRIDE = 6
idxs = list(range(0, N * STRIDE, STRIDE))  # 0, 6, 12, ... — cover wider span than contiguous
frame_paths = [os.path.join(SEQ_DIR, f"frame-{i:06d}.color.jpg") for i in idxs]
frame_paths = [p for p in frame_paths if os.path.exists(p)]
print(f"[unit] {len(frame_paths)} frames (stride={STRIDE})")

tmp = tempfile.mkdtemp(prefix="mast3r_sfm_unit_")
depth_dir = os.path.join(tmp, "depth")
poses_dir = os.path.join(tmp, "poses")
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(poses_dir, exist_ok=True)

t0 = time.time()
poses_w2c, depths = run_mast3r_sfm_on_scene(
    frame_paths=frame_paths,
    checkpoint=CKPT,
    output_depth_dir=depth_dir,
    output_poses_dir=poses_dir,
    output_pointmaps_dir=None,
    scene_id=SCENE_ID,
    resolution=512,
    min_conf_thr=1.0,
    device="cuda",
    scene_graph="swin-10",
    shared_intrinsics=True,
    niter1=300, niter2=300,
    opt_depth=True,
    matching_conf_thr=0.0,
    subsample=8,
    depth_mask_mode="hard",
    save_raw_depth=True,
    save_confidence_maps=True,
    pose_jump_max_translation=0.0,
    pose_jump_max_z_translation=0.0,
    pose_jump_max_rotation_deg=0.0,
    pose_jump_relative_factor=5.0,
    zero_invalid_pose_depths=True,
)
elapsed = time.time() - t0
print(f"[unit] total runtime: {elapsed:.1f}s")
print(f"[unit] poses_w2c shape: {tuple(poses_w2c.shape)}")
print(f"[unit] depth shape: {depths[0].shape} len={len(depths)}")

# Summary stats
import numpy as np
trans = poses_w2c.numpy()[:, :3, 3]
steps = np.linalg.norm(np.diff(trans, axis=0), axis=1)
print(f"[unit] inter-frame translation w2c: median={np.median(steps):.3f}m max={steps.max():.3f}m")
coverages = [(d > 0).mean() for d in depths]
print(f"[unit] depth masked coverage: mean={np.mean(coverages):.3f} median={np.median(coverages):.3f}")

# Confirm on-disk artifacts
import glob
pgms = sorted(glob.glob(os.path.join(depth_dir, "*.depth.pgm")))
raws = sorted(glob.glob(os.path.join(depth_dir, "*.depth_raw.npy")))
conf_files = sorted(glob.glob(os.path.join(depth_dir, "*.conf.npy")))
pose_files = sorted(glob.glob(os.path.join(poses_dir, "*.pose.txt")))
print(f"[unit] on-disk: pgms={len(pgms)} raws={len(raws)} confs={len(conf_files)} poses={len(pose_files)}")
print(f"[unit] tmp_dir={tmp} (delete manually if needed)")
print("[unit] DONE")
