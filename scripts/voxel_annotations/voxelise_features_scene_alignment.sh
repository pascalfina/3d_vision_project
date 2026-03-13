#!/bin/bash

args=("$@")

# Environment setup
source "$(dirname "${BASH_SOURCE[0]}")/../activate_objectx_env.sh"

python scene_graph_recon/preprocessing/voxel_anno/voxelise_features_scene_alignment.py \
    --config "preprocessing/voxel_anno/voxel_anno.yaml" \
    --model_dir "$DATA_ROOT_DIR" \
    ${args[@]}
