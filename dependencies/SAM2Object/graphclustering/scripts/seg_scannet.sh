VIEW_FREQ=3    # sample frequency of the frames
THRES_CONNECT="0.9,0.3,5"   # dynamic threshold for region growing
# THRES_CONNECT="0.9,0.5,5"   # dynamic threshold for region growing
MAX_NEIGHBOR_DISTANCE=2    # farthest distance to take neighbors into account
THRES_MERGE=200 #400           # merge small groups with less than THRES_MERGE points during post-processing
DIS_DECAY=0.5              # decay rate of the distance weight
SIMILAR_METRIC="2-norm"    # metric for similarity measurement
MASK_NAME="semantic-sam"   # mask name for loading mask
ALIAS_MASK_NAME="semantic-sam"   # mask name for saving results


TEXT_HEAD="3RScan"
TEXT="${TEXT_HEAD}_${VIEW_FREQ}view"
HEAD="${TEXT_HEAD}_${VIEW_FREQ}view_merge${THRES_MERGE}_${SIMILAR_METRIC}_${ALIAS_MASK_NAME}_connect${THRES_CONNECT}_depth${MAX_NEIGHBOR_DISTANCE}"
# EVAL_DIR="data/ScanNet/results/${HEAD}"                     # directory to export results
EVAL_DIR="/cluster/scratch/ealegret/3Rscan/results/${HEAD}"   # directory to export results

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
 --scene_id='5341b7e3-8a66-2cdd-8709-66a2159f0017'
 #--use_torch \


cd /cluster/home/ealegret/3d_vision_project/dependencies/SAM2Object/graphclustering

python - <<'PY'
from helpers.visualize import save_scannet_eval_format_to_mesh

save_scannet_eval_format_to_mesh(
    scene_id="5341b7e3-8a66-2cdd-8709-66a2159f0017",
    res_dir="/cluster/scratch/ealegret/3Rscan/results/3RScan_3view_merge150_2-norm_semantic-sam_connect0.92,0.85,2_depth2",
    data_dir="/cluster/project/cvg/data/3RScan/scenes",
    dataset="3rscan",
    out_dir="/cluster/scratch/ealegret/3Rscan/vis_mesh"
)
PY
