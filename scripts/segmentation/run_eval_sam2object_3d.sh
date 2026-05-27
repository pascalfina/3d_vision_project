#!/bin/bash
# Evaluate SAM2Object 3D class-agnostic instance segmentation against 3RScan GT.
# Give access: chmod +x scripts/segmentation/run_eval_sam2object_3d.sh
set -euo pipefail

# -------------------------
# Config
# -------------------------
YOUR_ROOT_DIR="/cluster/home/ealegret"
YOUR_SCRATCH_DIR="/cluster/scratch/ealegret"

PROJECT_DIR="$YOUR_ROOT_DIR/3d_vision_project"
DATA_ROOT_DIR="$YOUR_SCRATCH_DIR/sam2object"

# Original 3RScan GT (per-vertex objectId lives in labels.instances.annotated.v2.ply)
GT_ROOT="/cluster/project/cvg/data/3RScan/scenes"

# SAM2Object writes per-scene results to <base_dir>/scans/<scan>/results/.
# If your layout uses 'scenes' instead of 'scans', change the path below.
PRED_NPY_PATTERN="$DATA_ROOT_DIR/scans/{scan}/results/{scan}_labels_fine_global.npy"

OUT_JSON="$DATA_ROOT_DIR/eval/sam2object_3d_report.json"

SCAN_IDS=(
  "5341b7e3-8a66-2cdd-8709-66a2159f0017"
)

# -------------------------
# Environment
# -------------------------
source "$YOUR_SCRATCH_DIR/.venvv/bin/activate"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "GT root:   $GT_ROOT"
echo "Pred npy:  $PRED_NPY_PATTERN"
echo "Scans:     ${SCAN_IDS[*]}"
echo "Python:    $(which python)"

python src/evaluation/evaluate_sam2object_3d.py \
  --gt_root "$GT_ROOT" \
  --scans "${SCAN_IDS[@]}" \
  --pred_npy "$PRED_NPY_PATTERN" \
  --mode both \
  --out "$OUT_JSON"

echo "[OK] report written to $OUT_JSON"
