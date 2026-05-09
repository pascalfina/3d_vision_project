#!/bin/bash
#SBATCH --job-name=sam2object_full
#SBATCH --output=/cluster/scratch/ealegret/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/ealegret/logs/%x_%j.err
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00

set -euo pipefail

echo "========== JOB INFO =========="
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "Date: $(date)"

echo "Working dir: $(pwd)"
echo "=============================="


# 1. Environment
module load stack/.2024-06-silent gcc/12.2.0 cuda/12.4.1 python/3.10.13 || true

YOUR_ROOT_DIR="YOUR_ROOT_DIR"
YOUR_SCRATCH_DIR="YOUR_SCRATCH_DIR"
PROJECT_DIR="${YOUR_ROOT_DIR}/3d_vision_project"


source "${YOUR_ROOT_DIR}/.sam2object/bin/activate"

cd "$PROJECT_DIR"

export VLSG_SPACE="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH:$PROJECT_DIR/dependencies/gaussian-splatting"

export TORCH_HOME="${YOUR_SCRATCH_DIR}/torch_cache"
export XDG_CACHE_HOME="${YOUR_SCRATCH_DIR}/.cache"

mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"
mkdir -p $YOUR_SCRATCH_DIR/logs


# 2. Paths / scene config
DATA_ROOT_DIR="${YOUR_SCRATCH_DIR}/sam2object"
SCAN_ID="5341b7e3-8a66-2cdd-8709-66a2159f0017"
SCENE_DIR="$DATA_ROOT_DIR/scenes/$SCAN_ID"

echo "DATA_ROOT_DIR=$DATA_ROOT_DIR"
echo "SCAN_ID=$SCAN_ID"
echo "SCENE_DIR=$SCENE_DIR"



# 3. Optional: run SAM2Object 2D tracking
echo "========== STEP 1: SAM2Object 2D tracking =========="
cd "$PROJECT_DIR/dependencies/SAM2Object/segtrack"
python seg_tracking.py


# 4. Convert masks
# Uncomment if graphclustering expects masks under 2D_masks/<scene>/semantic-sam

echo "========== STEP 2: mask_convert =========="
cd "$PROJECT_DIR/dependencies/SAM2Object/segtrack"
python mask_convert.py


# 5. Run Graph Clustering 3D
echo "========== STEP 3: Graph Clustering 3D =========="
cd "$PROJECT_DIR/dependencies/SAM2Object/graphclustering"
bash scripts/seg_scannet.sh


# 6. Prepare SAM2Object output for Object-X
echo "========== STEP 4: Prepare SAM2Object outputs for Object-X =========="

cd "$PROJECT_DIR"

python preprocessing/segmentation/prepare_sam2object_for_objectx.py \
  --root_dir "$DATA_ROOT_DIR" \
  --scan_id "$SCAN_ID" \
  --mesh_path "$SCENE_DIR/mesh.refined.v2.obj" \
  --sam_points "$SCENE_DIR/results/${SCAN_ID}_points.npy" \
  --sam_labels "$SCENE_DIR/results/${SCAN_ID}_labels_fine_global.npy" \
  --projection_dilation 2



# 8. Run Object-X voxelise_features.py

echo "========== STEP 6: Object-X voxelise features =========="
cd "$PROJECT_DIR"
echo "$SCAN_ID" > "$DATA_ROOT_DIR/files/sam2object_scans.txt"

VIS_DIR="/cluster/scratch/ealegret/objectx_vis"
mkdir -p "$VIS_DIR"

python preprocessing/voxel_anno/voxelise_features.py \
  --config preprocessing/voxel_anno/voxel_anno.yaml \
  --split sam2object \
  --model_dir "$DATA_ROOT_DIR" \
  --vis_dir "$VIS_DIR" \
  "data.root_dir='$DATA_ROOT_DIR'" \
  "autoencoder.encoder.scan_type='scan'"

echo "========== DONE =========="
date
