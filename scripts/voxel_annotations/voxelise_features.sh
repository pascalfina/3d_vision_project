#!/bin/bash

args=("$@")

# Environment setup
cd /cluster/home/ealegret/3d_vision_project
source /cluster/scratch/ealegret/.sam2object/bin/activate
source /cluster/scratch/ealegret/.venvv/bin/activate

# Global Variables
export VLSG_SPACE=$(pwd)
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"
export DATA_ROOT_DIR="/cluster/scratch/ealegret/sam2object"

export TORCH_HOME="/cluster/scratch/ealegret/torch_cache"
export XDG_CACHE_HOME="/cluster/scratch/ealegret/.cache"

SCAN_ID="5341b7e3-8a66-2cdd-8709-66a2159f0017"

export TORCH_HOME="/cluster/scratch/ealegret/torch_cache"
export XDG_CACHE_HOME="/cluster/scratch/ealegret/.cache"

mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"
echo "$SCAN_ID" > "$DATA_ROOT_DIR/files/sam2object_scans.txt"

python preprocessing/voxel_anno/voxelise_features.py \
  --config preprocessing/voxel_anno/voxel_anno.yaml \
  --split sam2object \
  --model_dir "$DATA_ROOT_DIR" \
  "data.root_dir='$DATA_ROOT_DIR'"  \
  "autoencoder.encoder.scan_type='scan'" \
  ${args[@]}
