#!/bin/bash

set -euo pipefail

# -------------------------
# Config
# -------------------------
YOUR_ROOT_DIR="/cluster/home/ealegret"
YOUR_SCRATCH_DIR="/cluster/scratch/ealegret"

PROJECT_DIR="$YOUR_ROOT_DIR/3d_vision_project"
DATA_ROOT_DIR="$YOUR_SCRATCH_DIR/sam2object"

SCAN_IDS=(
  "5341b7e3-8a66-2cdd-8709-66a2159f0017"
)



# Environment
source "$YOUR_SCRATCH_DIR/.sam2object/bin/activate"

cd "$PROJECT_DIR"

export VLSG_SPACE="$PROJECT_DIR"
export PYTHONPATH="$VLSG_SPACE:${PYTHONPATH:-}:$VLSG_SPACE/dependencies/gaussian-splatting"

export TORCH_HOME="$YOUR_SCRATCH_DIR/torch_cache"
export XDG_CACHE_HOME="$YOUR_SCRATCH_DIR/.cache"

mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$VIS_DIR"
mkdir -p "$DATA_ROOT_DIR/files"

echo "Project dir: $PROJECT_DIR"
echo "Data root: $DATA_ROOT_DIR"
echo "Python: $(which python)"


# 1. Prepare SAM2Object output for Object-X
echo "========== Prepare SAM2Object for Object-X =========="

for SCAN_ID in "${SCAN_IDS[@]}"; do
  echo "[RUN] prepare_sam2object_for_objectx.py for $SCAN_ID"

  python preprocessing/segmentation/prepare_sam2object_for_objectx.py \
    --root_dir "$DATA_ROOT_DIR" \
    --scan_id "$SCAN_ID" \
    --mesh_path "$DATA_ROOT_DIR/scenes/$SCAN_ID/mesh.refined.v2.obj" \
    --sam_points "$DATA_ROOT_DIR/scenes/$SCAN_ID/results/${SCAN_ID}_points.npy" \
    --sam_labels "$DATA_ROOT_DIR/scenes/$SCAN_ID/results/${SCAN_ID}_labels_fine_global.npy" \
    --projection_dilation 2
done


# 2. Create split file for voxelise_features.py
echo "========== Create Object-X split file =========="

rm -f "$DATA_ROOT_DIR/files/sam2object_scans.txt"
rm -f "$DATA_ROOT_DIR/files/sam2object_resplit_scans.txt"

for SCAN_ID in "${SCAN_IDS[@]}"; do
  echo "$SCAN_ID" >> "$DATA_ROOT_DIR/files/sam2object_scans.txt"
  echo "$SCAN_ID" >> "$DATA_ROOT_DIR/files/sam2object_resplit_scans.txt"
done

echo "[OK] sam2object_scans.txt:"
cat "$DATA_ROOT_DIR/files/sam2object_scans.txt"

echo "[OK] sam2object_resplit_scans.txt:"
cat "$DATA_ROOT_DIR/files/sam2object_resplit_scans.txt"


# 3. Run Object-X voxelise_features.py
echo "========== Run voxelise_features.py =========="

python preprocessing/voxel_anno/voxelise_features.py \
  --config preprocessing/voxel_anno/voxel_anno.yaml \
  --split sam2object \
  --model_dir "$DATA_ROOT_DIR" \
  --vis_dir "$VIS_DIR" \
  --override \
  "data.root_dir='$DATA_ROOT_DIR'" \
  "autoencoder.encoder.scan_type='scan'"

echo "========== DONE =========="
echo "Voxel outputs should be in:"
echo "$DATA_ROOT_DIR/files/gs_annotations/"