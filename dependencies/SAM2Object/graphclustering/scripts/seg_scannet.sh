set -euo pipefail   # fail fast: a sam2object.py crash must propagate (not a silent [ok])

VIEW_FREQ=3    # sample frequency of the frames
THRES_CONNECT="0.9,0.3,5"   # dynamic threshold for region growing
# THRES_CONNECT="0.9,0.5,5"   # dynamic threshold for region growing
MAX_NEIGHBOR_DISTANCE=2    # farthest distance to take neighbors into account
THRES_MERGE=200 #400           # merge small groups with less than THRES_MERGE points during post-processing
DIS_DECAY=0.5              # decay rate of the distance weight
SIMILAR_METRIC="2-norm"    # metric for similarity measurement
MASK_NAME="semantic-sam"   # mask name for loading mask
ALIAS_MASK_NAME="semantic-sam"   # mask name for saving results


# DATASET (3RScan|ScanNet) and DATA_ROOT_DIR come from the batch script's env;
# defaults preserve standalone 3RScan behaviour.
export DATASET="${DATASET:-3RScan}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/cluster/scratch/ealegret/sam2object}"
TEXT_HEAD="${DATASET}"
TEXT="${TEXT_HEAD}_${VIEW_FREQ}view"
HEAD="${TEXT_HEAD}_${VIEW_FREQ}view_merge${THRES_MERGE}_${SIMILAR_METRIC}_${ALIAS_MASK_NAME}_connect${THRES_CONNECT}_depth${MAX_NEIGHBOR_DISTANCE}"
EVAL_DIR="${DATA_ROOT_DIR}/results/${HEAD}"   # directory to export results

python sam2object.py \
 --thres_merge=$THRES_MERGE \
 --similar_metric=$SIMILAR_METRIC \
 --thres_connect=$THRES_CONNECT \
 --mask_name=$MASK_NAME \
 --text=$TEXT \
 --max_neighbor_distance=$MAX_NEIGHBOR_DISTANCE \
 --view_freq=$VIEW_FREQ \
 --dis_decay=$DIS_DECAY \
 --eval_dir=$EVAL_DIR \
 --base_dir="$DATA_ROOT_DIR" \
 --scene_id="${SCAN_IDS:?set SCAN_IDS (exported by the batch script)}"
 #--use_torch \


cd /cluster/home/ealegret/3d_vision_project/dependencies/SAM2Object/graphclustering

export EVAL_DIR DATA_ROOT_DIR
python - <<'PY'
import os
from os.path import dirname, join
from helpers.visualize import save_scannet_eval_format_to_mesh

dataset = os.environ.get("DATASET", "3RScan").lower()
eval_dir = os.environ["EVAL_DIR"]
data_root = os.environ.get("DATA_ROOT_DIR", "/cluster/scratch/ealegret/sam2object")
scenes_path = os.environ.get("SAM2OBJECT_DATA_PATH", "/cluster/project/cvg/data/3RScan/scenes")

# 3rscan: data_dir is the scenes dir (joins <scene>/labels.instances...).
# scannet: data_dir is the dataset root (visualize joins scans/<scene>/...).
data_dir = scenes_path if dataset == "3rscan" else dirname(scenes_path)
out_dir = join(data_root, "vis_mesh")

scans = [s.strip() for s in os.environ.get("SCAN_IDS", "").split(",") if s.strip()]
for scene_id in scans:
    try:
        save_scannet_eval_format_to_mesh(
            scene_id=scene_id, res_dir=eval_dir, data_dir=data_dir,
            dataset=dataset, out_dir=out_dir,
        )
    except Exception as e:
        print(f"[viz skip] {scene_id}: {e}")
PY
