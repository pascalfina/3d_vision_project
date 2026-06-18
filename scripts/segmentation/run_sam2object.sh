#!/bin/bash
## Give access: chmod +x scripts/run_sam2object_data_prep.sh
set -euo pipefail

# -------------------------
# Config
# -------------------------
YOUR_ROOT_DIR="/cluster/home/ealegret"
YOUR_SCRATCH_DIR="/cluster/scratch/ealegret"

PROJECT_DIR="$YOUR_ROOT_DIR/3d_vision_project"
SAM2OBJECT_DIR="$PROJECT_DIR/dependencies/SAM2Object"

DATA_ROOT_DIR="$YOUR_SCRATCH_DIR/sam2object"
ORIGINAL_3RSCAN_ROOT="/cluster/project/cvg/data/3RScan"

DATASET="3RScan"
SPLIT='val'
SCAN_IDS=(
  "5341b7e3-8a66-2cdd-8709-66a2159f0017"
)

# -------------------------
# Environment
# -------------------------
source "$YOUR_SCRATCH_DIR/.sam2object/bin/activate"

cd "$PROJECT_DIR"

export VLSG_SPACE="$PROJECT_DIR"
export PYTHONPATH="$VLSG_SPACE:${PYTHONPATH:-}:$VLSG_SPACE/dependencies/gaussian-splatting"

export DATASET
export FRAME_STEP
export DATA_ROOT_DIR
export ORIGINAL_3RSCAN_ROOT
export SCAN_IDS="$(IFS=,; echo "${SCAN_IDS[*]}")"

export TORCH_HOME="$YOUR_SCRATCH_DIR/torch_cache"
export XDG_CACHE_HOME="$YOUR_SCRATCH_DIR/.cache"

mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"
mkdir -p "$DATA_ROOT_DIR/files"

echo "Project dir: $PROJECT_DIR"
echo "SAM2Object dir: $SAM2OBJECT_DIR"
echo "Data root: $DATA_ROOT_DIR"
echo "Scan IDs: $SCAN_IDS"
echo "Python: $(which python)"

# -------------------------
# Split file
# -------------------------
SPLIT_FILE="$DATA_ROOT_DIR/files/sam2object_resplit_scans.txt"

rm -f "$SPLIT_FILE"

for SCAN_ID in "${SCAN_IDS[@]}"; do
  echo "$SCAN_ID" >> "$SPLIT_FILE"
done

echo "[OK] Wrote split file:"
cat "$SPLIT_FILE"

# -------------------------
# SAM2Object data preparation
# -------------------------
cd "$SAM2OBJECT_DIR/segtrack"

echo "[RUN] extract_only_jpg.py"
python dataprocess/extract_only_jpg.py
echo "[RUN] get_posed_images.py"
python dataprocess/get_posed_images.py
echo "[OK] SAM2Object data preparation finished."


echo "[RUN] seg_tracking.py"
python seg_tracking.py
echo "[OK] SAM2Object seg_tracking finished."


echo "[RUN] mask_convert.py"
python mask_convert.py
echo "[OK] SAM2Object mask_convert finished."

echo "[RUN] seg_scannet.py"
cd graphclustering
bash scripts/seg_scannet.sh
echo "[OK] SAM2Object seg_scannet finished."
