#!/bin/bash

args=("$@")

# Environment setup
source "$(dirname "${BASH_SOURCE[0]}")/../activate_objectx_env.sh"


xvfb-run -a  python preprocessing/gs_anno/map_to_colmap_scannet.py \
    --config "$VLSG_SPACE/preprocessing/gs_anno/gs_anno_scannet.yaml" \
    --output_dir "$DATA_ROOT_DIR/files/gs_annotations_scannet" \
    --split "test" \
    ${args[@]}
