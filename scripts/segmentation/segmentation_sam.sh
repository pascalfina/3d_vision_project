#!/bin/bash

args=("$@")

# Environment setup
export VLSG_SPACE=$(pwd)
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"
export MUST3R_PATH=/cluster/home/ealegret/3d_vision_project/models/must3r
export OBJECTX_SAM2_APPLY_POSTPROCESS=0
export OBJECTX_SAM2_REQUIRE_POSTPROCESS=0

# Adapt to the specific CUDA version and architecture of the cluster
export CUDA_HOME=
export TORCH_CUDA_ARCH_LIST="8.6" # P.e: "8.6" 
export CUDA_HOME=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-13.0.2-uf6cve7i5qqoa5b7pxrzmiybyt6iskfi
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Environment setup
source /cluster/scratch/ealegret/must3r_311/bin/activate

# Run experiments
cd /cluster/home/ealegret/3d_vision_project/models/sam2
pip install -e ".[notebooks]"

cd /cluster/home/ealegret/3d_vision_project/
/usr/bin/time -v python preprocessing/segmentation/run_pipeline.py --config preprocessing/segmentation/pipeline.yaml
python preprocessing/segmentation/run_pipeline.py --config preprocessing/segmentation/pipeline.yaml
python preprocessing/voxel_anno/voxelise_features.py --config preprocessing/voxel_anno/voxel_anno.yaml --model_dir /cluster/project/cvg/data/3RScan --objects-file objects_predicted.json --scene 5341b7e3-8a66-2cdd-8709-66a2159f0017 --object-source lifted_masks
