import os
import sys

import numpy as np
from PIL import Image

# Allow `python dataprocess/get_posed_images.py` (cwd=segtrack) to import the sibling helper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_source import list_frames, read_intrinsics, link_or_copy

opj = os.path.join

DATASET           = os.environ.get("DATASET", "3RScan")
DATA_ROOT_DIR     = os.environ.get("DATA_ROOT_DIR", "/cluster/scratch/ealegret/sam2object")
SCAN_IDS_ENV      = os.environ.get("SCAN_IDS", "")
SCAN_IDS          = [s.strip() for s in SCAN_IDS_ENV.split(",") if s.strip()]
SCANNET_POSED_SRC = os.environ.get("SCANNET_POSED_SRC", "/cluster/project/cvg/data/scannet/posed_images")

scenes_dir = opj(DATA_ROOT_DIR, "scenes")          # 3RScan source (symlinked to GT scenes)
save_path  = opj(DATA_ROOT_DIR, "posed_images")
os.makedirs(save_path, exist_ok=True)


def write_depth(src_depth, dst_depth):
    """ScanNet depth is already .png (symlink it); 3RScan depth is .pgm (convert to .png)."""
    if src_depth.endswith(".png"):
        link_or_copy(src_depth, dst_depth)
    else:  # .pgm -> .png
        Image.open(src_depth).save(dst_depth)


# Both datasets share one re-index-and-write step: frame at position `i` in the
# natsorted list is written as {i:06d}.{jpg,png,txt}. Intrinsics are written under the
# names sam2object's get_scannet_color_and_depth_intrinsic expects. See frame_source.py.
for scene in SCAN_IDS:
    frames = list_frames(DATASET, scene, scannet_src=SCANNET_POSED_SRC, scenes_dir=scenes_dir)
    if not frames:
        continue
    out_scene = opj(save_path, scene)
    os.makedirs(out_scene, exist_ok=True)

    k_color, k_depth = read_intrinsics(DATASET, scene, scannet_src=SCANNET_POSED_SRC, scenes_dir=scenes_dir)
    np.savetxt(opj(out_scene, "intrinsics_color.txt"), k_color)
    np.savetxt(opj(out_scene, "intrinsics_depth.txt"), k_depth)

    copied = 0
    for i, f in enumerate(frames):
        if os.path.exists(f.color_path):
            link_or_copy(f.color_path, opj(out_scene, f"{i:06d}.jpg"))
        if os.path.exists(f.depth_path):
            write_depth(f.depth_path, opj(out_scene, f"{i:06d}.png"))
        if os.path.exists(f.pose_path):
            link_or_copy(f.pose_path, opj(out_scene, f"{i:06d}.txt"))
        copied += 1
    print(f"[OK] {DATASET} {scene}: staged {copied} frames -> posed_images")
