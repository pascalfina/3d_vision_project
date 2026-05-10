#!/usr/bin/env bash
# Full SAM2Object segmentation pipeline for one scene, followed by ObjectX preparation.
#
# Required env vars (set by run_scene_profile.py):
#   SAMOBJECT_DIR          - path to SAM2Object repo (dependencies/SAM2Object)
#   SAMOBJECT_VENV         - path to SAM2Object Python venv (activate script)
#   SAMOBJECT_DATA_ROOT    - root dir for SAM2Object data (read + write)
#   SAMOBJECT_SCAN_ID      - scene/scan ID to process
#   SAMOBJECT_MESH_PATH    - path to mesh.refined.v2.obj for this scene
#   SAMOBJECT_IMAGES_DIR   - dir containing color_images_cluster/<scan_id>/*.jpg
#   OBJECTX_REPO_ROOT      - repo root (set by run_scene_profile.py)
#
# Optional env vars:
#   SAMOBJECT_CHECKPOINT           (default: looks for sam2.1_hiera_large.pt in models/sam2ckpt)
#   SAMOBJECT_MODEL_CFG            (default: sam2.1_hiera_l.yaml for SAM2.1)
#   SAMOBJECT_PROJECTION_DILATION  (default: 2)
#   SAMOBJECT_FRAME_SKIP           (default: unset)
#   VLSG_SPACE                     (default: OBJECTX_REPO_ROOT)
#   TORCH_HOME
#   XDG_CACHE_HOME

set -euo pipefail

: "${SAMOBJECT_DIR:?Need SAMOBJECT_DIR}"
: "${SAMOBJECT_VENV:?Need SAMOBJECT_VENV}"
: "${SAMOBJECT_DATA_ROOT:?Need SAMOBJECT_DATA_ROOT}"
: "${SAMOBJECT_SCAN_ID:?Need SAMOBJECT_SCAN_ID}"
: "${SAMOBJECT_MESH_PATH:?Need SAMOBJECT_MESH_PATH}"
: "${SAMOBJECT_IMAGES_DIR:?Need SAMOBJECT_IMAGES_DIR}"
: "${OBJECTX_REPO_ROOT:?Need OBJECTX_REPO_ROOT}"

PROJECTION_DILATION="${SAMOBJECT_PROJECTION_DILATION:-2}"

# sam2object.py saves to base_dir/scans/scan_id/results/
SAM_RESULTS_DIR="$SAMOBJECT_DATA_ROOT/scans/$SAMOBJECT_SCAN_ID/results"

echo "========== SAM2Object pipeline =========="
echo "  scan_id   : $SAMOBJECT_SCAN_ID"
echo "  data_root : $SAMOBJECT_DATA_ROOT"
echo "  images_dir: $SAMOBJECT_IMAGES_DIR"
echo "  sam2obj   : $SAMOBJECT_DIR"
echo "========================================="

# Activate SAM2Object venv
# shellcheck source=/dev/null
source "$SAMOBJECT_VENV"

export VLSG_SPACE="${VLSG_SPACE:-$OBJECTX_REPO_ROOT}"
export PYTHONPATH="$VLSG_SPACE:${PYTHONPATH:-}:$VLSG_SPACE/dependencies/gaussian-splatting"

# Derive checkpoint and model config
_DEFAULT_CKPT="$OBJECTX_REPO_ROOT/models/sam2ckpt/sam2.1_hiera_large.pt"
export SAMOBJECT_CHECKPOINT="${SAMOBJECT_CHECKPOINT:-$_DEFAULT_CKPT}"
export SAMOBJECT_MODEL_CFG="${SAMOBJECT_MODEL_CFG:-sam2.1_hiera_l.yaml}"

# Dirs used by seg_tracking.py and mask_convert.py
export SAMOBJECT_SEGTRACK_OUTPUT="$SAMOBJECT_DATA_ROOT/segtrack_outputs"
export SAMOBJECT_2D_MASKS_DIR="$SAMOBJECT_DATA_ROOT/2D_masks"
mkdir -p "$SAMOBJECT_SEGTRACK_OUTPUT" "$SAMOBJECT_2D_MASKS_DIR"

# Write scene ID file for seg_tracking.py
export SAMOBJECT_SCENE_IDS_FILE="$SAMOBJECT_DATA_ROOT/files/sam2object_resplit_scans.txt"
mkdir -p "$SAMOBJECT_DATA_ROOT/files"
echo "$SAMOBJECT_SCAN_ID" > "$SAMOBJECT_SCENE_IDS_FILE"

# ── STEP 0: Prepare color_images_cluster from existing sequence (symlinks) ───
echo "========== STEP 0: Prepare image directory =========="
IMAGES_SCAN_DIR="$SAMOBJECT_IMAGES_DIR/$SAMOBJECT_SCAN_ID"
mkdir -p "$IMAGES_SCAN_DIR"
if [[ -z "$(ls -A "$IMAGES_SCAN_DIR" 2>/dev/null)" ]]; then
  SOURCE_SEQ="${SAMOBJECT_SOURCE_SEQUENCE_DIR:-}"
  if [[ -z "$SOURCE_SEQ" ]]; then
    echo "ERROR: IMAGES_SCAN_DIR is empty and SAMOBJECT_SOURCE_SEQUENCE_DIR is not set."
    exit 1
  fi
  echo "  Symlinking from $SOURCE_SEQ -> $IMAGES_SCAN_DIR"
  idx=0
  while IFS= read -r img; do
    ln -sf "$img" "$IMAGES_SCAN_DIR/${idx}.jpg"
    idx=$((idx + 1))
  done < <(ls "$SOURCE_SEQ"/frame-*.color.jpg 2>/dev/null | sort)
  echo "  Created $idx image symlinks."
else
  echo "  $IMAGES_SCAN_DIR already populated, skipping."
fi

# ── STEP 1: 2D tracking ──────────────────────────────────────────────────────
echo "========== STEP 1: SAM2Object 2D tracking =========="
cd "$SAMOBJECT_DIR/segtrack"
python seg_tracking.py

# ── STEP 2: Mask conversion ──────────────────────────────────────────────────
echo "========== STEP 2: mask_convert =========="
cd "$SAMOBJECT_DIR/segtrack"
python mask_convert.py

# ── STEP 3: Graph clustering 3D ──────────────────────────────────────────────
echo "========== STEP 3: Graph Clustering 3D =========="
cd "$SAMOBJECT_DIR/graphclustering"
bash scripts/seg_scannet.sh

# ── STEP 4: Prepare for ObjectX ──────────────────────────────────────────────
echo "========== STEP 4: Prepare SAM2Object output for ObjectX =========="
cd "$OBJECTX_REPO_ROOT"

PREPARE_ARGS=(
  --root_dir "$SAMOBJECT_DATA_ROOT"
  --scan_id  "$SAMOBJECT_SCAN_ID"
  --mesh_path "$SAMOBJECT_MESH_PATH"
  --sam_points "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}_points.npy"
  --sam_labels "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}_labels_fine_global.npy"
  --projection_dilation "$PROJECTION_DILATION"
)
if [[ -n "${SAMOBJECT_FRAME_SKIP:-}" ]]; then
  PREPARE_ARGS+=(--frame_skip "$SAMOBJECT_FRAME_SKIP")
fi

python preprocessing/segmentation/prepare_sam2object_for_objectx.py "${PREPARE_ARGS[@]}"

echo "========== DONE =========="
