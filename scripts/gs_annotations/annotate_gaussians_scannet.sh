#!/bin/bash

args=("$@")

# Environment setup
source "$(dirname "${BASH_SOURCE[0]}")/../activate_objectx_env.sh"

iterations=7000
densify_until_iter=15_000


python preprocessing/gs_anno/annotate_gaussians_scannet.py \
    --densify_until_iter $densify_until_iter \
    --iterations $iterations \
    --save_iterations $iterations \
    --test_iterations $iterations \
    --config "$VLSG_SPACE/preprocessing/gs_anno/gs_anno_scannet.yaml" \
    --source_dir "$DATA_ROOT_DIR" \
    --model_dir "$DATA_ROOT_DIR" \
    --split "test" \
    ${args[@]}
