#!/bin/bash

args=("$@")

# Environment setup
export VLSG_SPACE=$(pwd)
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"
export MUST3R_PATH=/cluster/home/ealegret/3d_vision_project/models/must3r

# Adapt to the specific CUDA version and architecture of the cluster
export CUDA_HOME=
export TORCH_CUDA_ARCH_LIST="" # P.e: "8.6" 
export CUDA_HOME=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-13.0.2-uf6cve7i5qqoa5b7pxrzmiybyt6iskfi
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Environment setup
source /must3r_311/bin/activate

# Run experiments
cd models/sam2
pip install -e ".[notebooks]"
python preprocessing/segmentation/run_pipeline.py --config preprocessing/segmentation/pipeline.yaml
python preprocessing/voxel_anno/voxelise_features.py --config preprocessing/voxel_anno/voxel_anno.yaml --model_dir /cluster/project/cvg/data/3RScan --objects-file objects_predicted.json --scene 0ad2d3a1-79e2-2212-9b99-a96495d9f7fe --object-source lifted_masks