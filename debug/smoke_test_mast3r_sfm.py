"""Smoke test MASt3R-SfM on 10 cabinet frames."""
import os, sys, tempfile, time

MAST3R_ROOT = "/work/scratch/pafina/object-x/dependencies/mast3r"
sys.path.insert(0, MAST3R_ROOT)
sys.path.insert(0, os.path.join(MAST3R_ROOT, "dust3r"))

import torch
import numpy as np
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images

DEVICE = "cuda"
WEIGHTS = "/work/scratch/pafina/object-x/models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
SEQ_DIR = "/work/scratch/pafina/objectx-data-fullscene-cabinet-hybrid-sam2mask-must3r/scenes_sam2_must3r/e61b0e04-bada-2f31-82d6-72831a602ba7/sequence"

filelist = [os.path.join(SEQ_DIR, f"frame-{i:06d}.color.jpg") for i in range(10)]
print(f"[smoke] {len(filelist)} frames")

print("[smoke] loading model ...")
t0 = time.time()
model = AsymmetricMASt3R.from_pretrained(WEIGHTS).to(DEVICE)
print(f"[smoke] model loaded in {time.time()-t0:.1f}s")

print("[smoke] loading images ...")
imgs = load_images(filelist, size=512, verbose=False)
print(f"[smoke] images loaded: {len(imgs)}")

print("[smoke] building pairs (complete graph) ...")
pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
print(f"[smoke] pairs: {len(pairs)}")

cache_dir = tempfile.mkdtemp(prefix="mast3r_smoke_")
print(f"[smoke] cache: {cache_dir}")

print("[smoke] running sparse_global_alignment ...")
t0 = time.time()
scene = sparse_global_alignment(
    filelist, pairs, cache_dir, model,
    lr1=0.07, niter1=200, lr2=0.01, niter2=200,
    device=DEVICE, opt_depth=True, shared_intrinsics=True,
    matching_conf_thr=0.0, verbose=True,
)
print(f"[smoke] BA done in {time.time()-t0:.1f}s")

poses = scene.get_im_poses().detach().cpu().numpy()
focals = scene.get_focals().detach().cpu().numpy()
depths = scene.get_depthmaps()
print(f"[smoke] poses shape: {poses.shape}")
print(f"[smoke] focals: {focals.reshape(-1)}")
print(f"[smoke] depth per frame:")
for i, d in enumerate(depths):
    d = d.detach().cpu().numpy() if torch.is_tensor(d) else np.asarray(d)
    print(f"  frame {i}: depth shape={d.shape} min={d.min():.3f} max={d.max():.3f} mean={d.mean():.3f}")

trans = poses[:, :3, 3]
steps = np.linalg.norm(np.diff(trans, axis=0), axis=1)
print(f"[smoke] inter-frame translation: median={np.median(steps):.3f}m max={steps.max():.3f}m")
print("[smoke] DONE")
