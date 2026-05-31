#!/bin/bash
# Batch: evaluate SAM2Object vs 3RScan GT AND export per-scan coloured point clouds
# for every scan id in a .txt list. Scans without a prediction are skipped.
# Give access: chmod +x scripts/segmentation/run_eval_export_batch.sh
set -euo pipefail

# -------------------------
# Config  (edit these)
# -------------------------
YOUR_ROOT_DIR="/cluster/home/ealegret"
YOUR_SCRATCH_DIR="/cluster/scratch/ealegret"

PROJECT_DIR="$YOUR_ROOT_DIR/3d_vision_project"
DATA_ROOT_DIR="$YOUR_SCRATCH_DIR/sam2object"

# 3RScan GT (per-vertex objectId in labels.instances.annotated.v2.ply)
GT_ROOT="/cluster/project/cvg/data/3RScan/scenes"

# Newline-delimited list of scan ids to process.
LIST_FILE="$DATA_ROOT_DIR/files/sam2object_scans.txt"
# e.g. the full validation split:
# LIST_FILE="/cluster/project/cvg/data/3RScan/files/val_resplit_scans.txt"

# Where SAM2Object writes per-scan results. Switch 'scans' -> 'scenes' if needed.
PRED_NPY_PATTERN="$DATA_ROOT_DIR/scans/{scan}/results/{scan}_labels_fine_global.npy"

# Outputs
OUT_DIR="$DATA_ROOT_DIR/eval_batch"
MODE="both"          # both | objects-only | all   (metrics)
IOU="0.25"           # IoU threshold for the colour-matching in the point clouds

# -------------------------
# Environment
# -------------------------
source "$YOUR_SCRATCH_DIR/.venvv/bin/activate"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR/plys"

echo "GT root:   $GT_ROOT"
echo "List file: $LIST_FILE  ($(grep -cve '^[[:space:]]*$' "$LIST_FILE") scans)"
echo "Pred npy:  $PRED_NPY_PATTERN"
echo "Out dir:   $OUT_DIR"
echo "Python:    $(which python)"

# -------------------------
# 1. Metrics (combined report, skips scans with no prediction)
# -------------------------
echo "========== Metrics =========="
python src/evaluation/evaluate_sam2object_3d.py \
  --gt_root "$GT_ROOT" \
  --split_file "$LIST_FILE" \
  --pred_npy "$PRED_NPY_PATTERN" \
  --mode "$MODE" \
  --out "$OUT_DIR/report.json"

# -------------------------
# 2. Per-scan coloured POINT CLOUDS (skips scans with no prediction)
# -------------------------
echo "========== Point clouds =========="
python src/evaluation/export_segmented_ply.py \
  --gt_root "$GT_ROOT" \
  --split_file "$LIST_FILE" \
  --pred_npy "$PRED_NPY_PATTERN" \
  --out_dir "$OUT_DIR/plys" \
  --mode objects-only \
  --iou "$IOU" \
  --points_only

echo "========== DONE =========="
echo "Metrics:      $OUT_DIR/report.json"
echo "Point clouds: $OUT_DIR/plys/<scan>_{gt,pred}.ply"
