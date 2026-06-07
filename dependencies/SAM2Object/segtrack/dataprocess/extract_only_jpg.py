import os
import sys

# Allow `python dataprocess/extract_only_jpg.py` (cwd=segtrack) to import the sibling helper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_source import list_frames, link_or_copy

opj = os.path.join

DATASET           = os.environ.get("DATASET", "3RScan")
DATA_ROOT_DIR     = os.environ.get("DATA_ROOT_DIR", "/cluster/scratch/ealegret/sam2object")
SCAN_IDS_ENV      = os.environ.get("SCAN_IDS", "")
SCAN_IDS          = [s.strip() for s in SCAN_IDS_ENV.split(",") if s.strip()]
SCANNET_POSED_SRC = os.environ.get("SCANNET_POSED_SRC", "/cluster/project/cvg/data/scannet/posed_images")

scenes_dir = opj(DATA_ROOT_DIR, "scenes")          # 3RScan source (symlinked to GT scenes)
save_path  = opj(DATA_ROOT_DIR, "color_images_cluster")
os.makedirs(save_path, exist_ok=True)

# Both datasets share one re-index-and-write step: frame at position `i` in the
# natsorted list is written as {i:06d}.jpg. This keeps the SAM2-tracking positional
# index aligned with posed_images / mask naming (see frame_source.py).
for scene in SCAN_IDS:
    frames = list_frames(DATASET, scene, scannet_src=SCANNET_POSED_SRC, scenes_dir=scenes_dir)
    if not frames:
        continue
    out_scene = opj(save_path, scene)
    os.makedirs(out_scene, exist_ok=True)
    copied = 0
    for i, f in enumerate(frames):
        if not os.path.exists(f.color_path):
            continue
        link_or_copy(f.color_path, opj(out_scene, f"{i:06d}.jpg"))
        copied += 1
    print(f"[OK] {DATASET} {scene}: linked {copied} color frames -> color_images_cluster")
