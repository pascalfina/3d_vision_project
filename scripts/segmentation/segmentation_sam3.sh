#!/bin/bash

args=("$@")

# Environment setup
export VLSG_SPACE=$(pwd)
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"
export DATA_ROOT_DIR="/cluster/project/cvg/data/3RScan"

# Environment setup
source 3dv/bin/activate

# Run experiments
cd preprocessing/segmentation
python run_pipeline.py --config pipeline.yaml