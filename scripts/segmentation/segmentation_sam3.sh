#!/bin/bash

args=("$@")

# Environment setup
source .venv/bin/activate

python preprocessing/segmentation/preprocess_sam3_projection.py \
    --model_dir "$DATA_ROOT_DIR" \
    ${args[@]}
