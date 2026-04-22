#!/bin/bash

args=("$@")

# Environment setup
export VLSG_SPACE=$(pwd)
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"
export MUST3R_PATH=/cluster/home/ealegret/3d_vision_project/models/must3r

# Adapt to the specific CUDA version and architecture of the cluster
export CUDA_HOME=
export TORCH_CUDA_ARCH_LIST="" # P.e: "8.6" 

# Environment setup
source /must3r_311/bin/activate

# Run experiments
cd models/sam2
pip install -e ".[notebooks]"
python preprocessing/segmentation/run_pipeline.py --config preprocessing/segmentation/pipeline.yaml