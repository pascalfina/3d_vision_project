#!/bin/bash

args=("$@")

# Environment setup
export VLSG_SPACE=$(pwd)
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting:$VLSG_SPACE/dependencies/sam2:$VLSG_SPACE/dependencies/must3r:$VLSG_SPACE/dependencies/must3r/dust3r"
export DATA_ROOT_DIR="/cluster/project/cvg/data/3RScan"
export MUST3R_PATH="$VLSG_SPACE/dependencies/must3r"

# Environment setup
source scripts/activate_objectx_env.sh

# Run experiments
cd preprocessing/segmentation
python run_pipeline.py --config pipeline.yaml
