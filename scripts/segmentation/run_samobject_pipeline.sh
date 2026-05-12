#!/usr/bin/env bash
# Full SAM2Object segmentation pipeline for one scene, followed by ObjectX preparation.
#
# Required env vars (set by run_scene_profile.py):
#   SAMOBJECT_DIR          - path to SAM2Object repo (dependencies/SAM2Object)
#   SAMOBJECT_VENV         - path to SAM2Object Python venv (activate script)
#   SAMOBJECT_DATA_ROOT    - root dir for SAM2Object data (intermediate processing)
#   SAMOBJECT_SCAN_ID      - scene/scan ID to process
#   SAMOBJECT_MESH_PATH    - path to mesh.refined.v2.obj for this scene
#   SAMOBJECT_BASELINE_ROOT - baseline data root (scenes/<scan_id>/sequence lives here)
#   OBJECTX_REPO_ROOT      - repo root (set by run_scene_profile.py)
#
# Optional env vars:
#   SAMOBJECT_CHECKPOINT           (default: $SAMOBJECT_DIR/segtrack/checkpoints/sam2_hiera_large.pt)
#   SAMOBJECT_MODEL_CFG            (default: sam2_hiera_l.yaml)
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
: "${SAMOBJECT_BASELINE_ROOT:?Need SAMOBJECT_BASELINE_ROOT}"
: "${OBJECTX_REPO_ROOT:?Need OBJECTX_REPO_ROOT}"
# SAMOBJECT_OUTPUT_ROOT: where prepare writes masks/objects.json (defaults to SAMOBJECT_DATA_ROOT)
SAMOBJECT_OUTPUT_ROOT="${SAMOBJECT_OUTPUT_ROOT:-$SAMOBJECT_DATA_ROOT}"

PROJECTION_DILATION="${SAMOBJECT_PROJECTION_DILATION:-2}"
GRAPH_VIEW_FREQ="${SAMOBJECT_VIEW_FREQ:-3}"
GRAPH_THRES_MERGE="${SAMOBJECT_THRES_MERGE:-200}"
GRAPH_THRES_CONNECT="${SAMOBJECT_THRES_CONNECT:-0.9,0.3,5}"
GRAPH_MAX_NEIGHBOR_DISTANCE="${SAMOBJECT_MAX_NEIGHBOR_DISTANCE:-2}"
GRAPH_SIMILAR_METRIC="${SAMOBJECT_SIMILAR_METRIC:-2-norm}"
GRAPH_DIS_DECAY="${SAMOBJECT_DIS_DECAY:-0.5}"

# Paths used across all steps
BASELINE_SCENE_DIR="$SAMOBJECT_BASELINE_ROOT/scenes/$SAMOBJECT_SCAN_ID"
SAM_RESULTS_DIR="$SAMOBJECT_DATA_ROOT/scans/$SAMOBJECT_SCAN_ID/results"

echo "========== SAM2Object pipeline =========="
echo "  scan_id      : $SAMOBJECT_SCAN_ID"
echo "  data_root    : $SAMOBJECT_DATA_ROOT"
echo "  baseline     : $SAMOBJECT_BASELINE_ROOT"
echo "  sam2obj repo : $SAMOBJECT_DIR"
echo "  checkpoint   : ${SAMOBJECT_CHECKPOINT:-$OBJECTX_REPO_ROOT/models/sam2ckpt/sam2_hiera_large.pt}"
echo "  model_cfg    : ${SAMOBJECT_MODEL_CFG:-sam2_hiera_l.yaml}"
echo "========================================="

# Activate SAM2Object venv
# shellcheck source=/dev/null
source "$SAMOBJECT_VENV"

export VLSG_SPACE="${VLSG_SPACE:-$OBJECTX_REPO_ROOT}"
export PYTHONPATH="$VLSG_SPACE:${PYTHONPATH:-}:$VLSG_SPACE/dependencies/gaussian-splatting"

# Make torch CUDA libs findable (needed for SAM2 _C extension)
_TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="/cluster/data/cuda/12.8.0/lib64:${_TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Env vars that all SAM2Object Python scripts read
export DATASET="3RScan"
export SCAN_IDS="$SAMOBJECT_SCAN_ID"
export DATA_ROOT_DIR="$SAMOBJECT_DATA_ROOT"

# Checkpoint: SAM2Object repo uses SAM2 (not SAM2.1) configs
export SAMOBJECT_CHECKPOINT="${SAMOBJECT_CHECKPOINT:-$OBJECTX_REPO_ROOT/models/sam2ckpt/sam2_hiera_large.pt}"
export SAMOBJECT_MODEL_CFG="${SAMOBJECT_MODEL_CFG:-sam2_hiera_l.yaml}"

# Where sam2object.py should look for its 3D scene points.
# Default: baseline/GT 3RScan scene files. Pi3X path below can override this.
export SAMOBJECT_3RSCAN_SCENES_DIR="$SAMOBJECT_BASELINE_ROOT/scenes"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$SAMOBJECT_DATA_ROOT/files"

# ── STEP 0a: Symlink sequence so SAM2Object scripts can find it ───────────────
echo "========== STEP 0a: Setup scene data symlinks =========="
SCENE_DATA_DIR="$SAMOBJECT_DATA_ROOT/scenes/$SAMOBJECT_SCAN_ID"
mkdir -p "$SCENE_DATA_DIR"

# Resolve the sequence source: prefer SAMOBJECT_SOURCE_SEQUENCE_DIR, then baseline
_SEQ_SRC="${SAMOBJECT_SOURCE_SEQUENCE_DIR:-}"
if [[ -z "$_SEQ_SRC" ]]; then
  _SEQ_SRC="$BASELINE_SCENE_DIR/sequence"
fi

if [[ ! -e "$SCENE_DATA_DIR/sequence" ]]; then
  if [[ ! -d "$_SEQ_SRC" ]]; then
    echo "ERROR: sequence source not found: $_SEQ_SRC"
    echo "  Set source_sequence_dir in your scene profile samobject section."
    exit 1
  fi
  ln -s "$_SEQ_SRC" "$SCENE_DATA_DIR/sequence"
  echo "  Linked sequence: $_SEQ_SRC"
else
  if [[ -L "$SCENE_DATA_DIR/sequence" ]]; then
    _CUR_SEQ_TARGET="$(readlink -f "$SCENE_DATA_DIR/sequence")"
    _DESIRED_SEQ_TARGET="$(readlink -f "$_SEQ_SRC")"
    if [[ "$_CUR_SEQ_TARGET" != "$_DESIRED_SEQ_TARGET" ]]; then
      rm -f "$SCENE_DATA_DIR/sequence"
      ln -s "$_SEQ_SRC" "$SCENE_DATA_DIR/sequence"
      echo "  Relinked sequence: $_SEQ_SRC"
    else
      echo "  sequence dir already present."
    fi
  else
    echo "  sequence dir already present."
  fi
fi

# Write scene ID list file for scripts that need it
SCENE_IDS_FILE="$SAMOBJECT_DATA_ROOT/files/sam2object_resplit_scans.txt"
echo "$SAMOBJECT_SCAN_ID" > "$SCENE_IDS_FILE"

# ── STEP 0b-clean: Reset stale segtrack outputs for this scene ────────────────
SEGTRACK_OUTPUT_SCENE_DIR="$SAMOBJECT_DIR/segtrack/outputs/$SAMOBJECT_SCAN_ID"
if [[ "${SAMOBJECT_CLEAN_SEGTRACK_OUTPUTS:-1}" != "0" && -d "$SEGTRACK_OUTPUT_SCENE_DIR" ]]; then
  echo "========== STEP 0b-clean: Reset segtrack outputs =========="
  echo "  Removing stale segtrack outputs: $SEGTRACK_OUTPUT_SCENE_DIR"
  rm -rf "$SEGTRACK_OUTPUT_SCENE_DIR"
fi

# ── STEP 0b: Prepare posed_images and color_images_cluster ────────────────────
echo "========== STEP 0b: Prepare image data =========="

cd "$SAMOBJECT_DIR/segtrack"

POSED_SCENE_DIR="$SAMOBJECT_DATA_ROOT/posed_images/$SAMOBJECT_SCAN_ID"
COLOR_CLUSTER_SCENE_DIR="$SAMOBJECT_DATA_ROOT/color_images_cluster/$SAMOBJECT_SCAN_ID"
MASK2D_SCENE_DIR="$SAMOBJECT_DATA_ROOT/2D_masks/$SAMOBJECT_SCAN_ID/semantic-sam"

if [[ "${SAMOBJECT_REFRESH_POSED_IMAGES:-1}" != "0" && -d "$POSED_SCENE_DIR" ]]; then
  echo "  Refreshing posed_images scene dir: $POSED_SCENE_DIR"
  rm -rf "$POSED_SCENE_DIR"
fi
echo "  Running get_posed_images.py..."
python dataprocess/get_posed_images.py

if [[ "${SAMOBJECT_REFRESH_COLOR_CLUSTER:-1}" != "0" && -d "$COLOR_CLUSTER_SCENE_DIR" ]]; then
  echo "  Refreshing color_images_cluster scene dir: $COLOR_CLUSTER_SCENE_DIR"
  rm -rf "$COLOR_CLUSTER_SCENE_DIR"
fi
echo "  Running extract_only_jpg.py..."
python dataprocess/extract_only_jpg.py

# ── STEP 0c: Create superpoints from 3RScan segs.json ─────────────────────────
echo "========== STEP 0c: Create superpoints =========="
SUPERPOINTS_DIR="$SAMOBJECT_DATA_ROOT/superpoints/$SAMOBJECT_SCAN_ID"
SUPERPOINTS_FILE="$SUPERPOINTS_DIR/superpoint.pts"
SEGS_JSON="$BASELINE_SCENE_DIR/mesh.refined.0.010000.segs.v2.json"

if [[ -n "${SAMOBJECT_USE_PI3X_MESH:-}" && "${SAMOBJECT_USE_PI3X_MESH}" != "0" ]]; then
  echo "  Pi3X mode active: building Pi3X scene point cloud for SAM2Object."
  _PI3X_SEQ_DIR="${SAMOBJECT_PI3X_SEQ_DIR:-${SAMOBJECT_SOURCE_SEQUENCE_DIR:-}}"
  if [[ -z "$_PI3X_SEQ_DIR" ]]; then
    echo "ERROR: SAMOBJECT_PI3X_SEQ_DIR or SAMOBJECT_SOURCE_SEQUENCE_DIR must be set for Pi3X SAMObject mode."
    exit 1
  fi
  PI3X_SCENE_DIR="$SAMOBJECT_DATA_ROOT/scenes/$SAMOBJECT_SCAN_ID"
  mkdir -p "$PI3X_SCENE_DIR"
  python "$OBJECTX_REPO_ROOT/preprocessing/segmentation/create_pi3x_scene_ply_for_samobject.py" \
    --sequence-dir "$_PI3X_SEQ_DIR" \
    --out-ply "$PI3X_SCENE_DIR/labels.instances.annotated.v2.ply" \
    --conf-thr "${SAMOBJECT_PI3X_CONF_THR:-0.10}" \
    --pixel-stride "${SAMOBJECT_PI3X_PIXEL_STRIDE:-2}" \
    --voxel-dedup-size "${SAMOBJECT_PI3X_VOXEL_DEDUP_SIZE:-0.10}" \
    --superpoint-json-out "$PI3X_SCENE_DIR/mesh.refined.0.010000.segs.v2.json" \
    --superpoint-voxel-size "${SAMOBJECT_PI3X_SUPERPOINT_VOXEL_SIZE:-0.25}"
  export SAMOBJECT_3RSCAN_SCENES_DIR="$SAMOBJECT_DATA_ROOT/scenes"
  rm -f "$SAMOBJECT_DATA_ROOT/scans/$SAMOBJECT_SCAN_ID/points.pts"
  rm -f "$SAMOBJECT_DATA_ROOT/scans/$SAMOBJECT_SCAN_ID/results/${SAMOBJECT_SCAN_ID}_points.npy"
  rm -f "$SAMOBJECT_DATA_ROOT/scans/$SAMOBJECT_SCAN_ID/results/${SAMOBJECT_SCAN_ID}_labels_fine_global.npy"
else
  if [[ ! -f "$SUPERPOINTS_FILE" ]]; then
    if [[ ! -f "$SEGS_JSON" ]]; then
      echo "ERROR: segs.json not found at $SEGS_JSON"
      exit 1
    fi
    python "$OBJECTX_REPO_ROOT/preprocessing/segmentation/create_3rscan_superpoints.py" \
      --segs_json "$SEGS_JSON" \
      --out_dir   "$SUPERPOINTS_DIR"
  else
    echo "  superpoints already exist, skipping."
  fi
fi

# ── STEP 1: 2D tracking ──────────────────────────────────────────────────────
echo "========== STEP 1: SAM2Object 2D tracking =========="
# seg_tracking.py reads:
#   SAM2OBJECT_DIR -> PROJECT_DIR (repo dir, used for checkpoints + output path)
#   DATA_ROOT_DIR -> OUTPUT_PATH (base for color_images_cluster)
#   SCAN_IDS, DATASET (set above)
export SAM2OBJECT_DIR="$SAMOBJECT_DIR"
cd "$SAMOBJECT_DIR/segtrack"
python seg_tracking.py

# ── STEP 2: Mask conversion ──────────────────────────────────────────────────
echo "========== STEP 2: mask_convert =========="
# mask_convert.py reads:
#   SAM2OBJECT_DIR -> base_dir (segtrack outputs, used to READ masks)
#   DATA_ROOT_DIR -> DATA_PATH (destination for 2D_masks)
export SAM2OBJECT_DIR="$SAMOBJECT_DIR/segtrack/outputs"
cd "$SAMOBJECT_DIR/segtrack"
if [[ "${SAMOBJECT_REFRESH_2D_MASKS:-1}" != "0" && -d "$MASK2D_SCENE_DIR" ]]; then
  echo "  Refreshing 2D mask dir: $MASK2D_SCENE_DIR"
  rm -rf "$MASK2D_SCENE_DIR"
fi
python mask_convert.py

# ── STEP 3: Graph clustering 3D ──────────────────────────────────────────────
echo "========== STEP 3: Graph Clustering 3D =========="
cd "$SAMOBJECT_DIR/graphclustering"

if [[ -d "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}_pred_mask" ]]; then
  rm -rf "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}_pred_mask"
fi
rm -f \
  "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}.txt" \
  "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}_points.npy" \
  "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}_labels_fine_global.npy"

GRAPH_ARGS=(
  sam2object.py
  --base_dir "$SAMOBJECT_DATA_ROOT"
  --scene_id "$SAMOBJECT_SCAN_ID"
  --mask_name "semantic-sam"
  --view_freq "$GRAPH_VIEW_FREQ"
  --thres_merge "$GRAPH_THRES_MERGE"
  --thres_connect "$GRAPH_THRES_CONNECT"
  --max_neighbor_distance "$GRAPH_MAX_NEIGHBOR_DISTANCE"
  --similar_metric "$GRAPH_SIMILAR_METRIC"
  --dis_decay "$GRAPH_DIS_DECAY"
)
if [[ -n "${SAMOBJECT_FROM_POINTS_THR:-}" && ( -z "${SAMOBJECT_USE_PI3X_MESH:-}" || "${SAMOBJECT_USE_PI3X_MESH}" == "0" ) ]]; then
  GRAPH_ARGS+=(--from_points_thres "$SAMOBJECT_FROM_POINTS_THR")
fi
if [[ -n "${SAMOBJECT_GRAPH_PROCESS_NUM:-}" ]]; then
  GRAPH_ARGS+=(--process_num "$SAMOBJECT_GRAPH_PROCESS_NUM")
elif [[ -n "${SAMOBJECT_USE_PI3X_MESH:-}" && "${SAMOBJECT_USE_PI3X_MESH}" != "0" ]]; then
  GRAPH_ARGS+=(--process_num 1)
fi
python "${GRAPH_ARGS[@]}"

# ── STEP 4: Prepare for ObjectX ──────────────────────────────────────────────
echo "========== STEP 4: Prepare SAM2Object output for ObjectX =========="
cd "$OBJECTX_REPO_ROOT"

PREPARE_ARGS=(
  --root_dir        "$SAMOBJECT_DATA_ROOT"
  --output_root_dir "$SAMOBJECT_OUTPUT_ROOT"
  --scan_id         "$SAMOBJECT_SCAN_ID"
  --sam_points      "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}_points.npy"
  --sam_labels      "$SAM_RESULTS_DIR/${SAMOBJECT_SCAN_ID}_labels_fine_global.npy"
  --projection_dilation "$PROJECTION_DILATION"
)
if [[ -n "${SAMOBJECT_USE_PI3X_MESH:-}" && "${SAMOBJECT_USE_PI3X_MESH}" != "0" ]]; then
  _PI3X_SEQ_DIR="${SAMOBJECT_PI3X_SEQ_DIR:-${SAMOBJECT_SOURCE_SEQUENCE_DIR:-}}"
  if [[ -z "$_PI3X_SEQ_DIR" ]]; then
    echo "ERROR: SAMOBJECT_PI3X_SEQ_DIR or SAMOBJECT_SOURCE_SEQUENCE_DIR must be set for --use_pi3x_mesh"
    exit 1
  fi
  PREPARE_ARGS+=(--use_pi3x_mesh --pi3x_seq_dir "$_PI3X_SEQ_DIR")
  if [[ -n "${SAMOBJECT_PI3X_CONF_THR:-}" ]]; then
    PREPARE_ARGS+=(--pi3x_conf_thr "$SAMOBJECT_PI3X_CONF_THR")
  fi
  if [[ -n "${SAMOBJECT_PI3X_PIXEL_STRIDE:-}" ]]; then
    PREPARE_ARGS+=(--pi3x_pixel_stride "$SAMOBJECT_PI3X_PIXEL_STRIDE")
  fi
  if [[ -n "${SAMOBJECT_PI3X_VOXEL_DEDUP_SIZE:-}" ]]; then
    PREPARE_ARGS+=(--pi3x_voxel_dedup_size "$SAMOBJECT_PI3X_VOXEL_DEDUP_SIZE")
  fi
else
  PREPARE_ARGS+=(--mesh_path "$SAMOBJECT_MESH_PATH")
fi
if [[ -n "${SAMOBJECT_FRAME_SKIP:-}" ]]; then
  PREPARE_ARGS+=(--frame_skip "$SAMOBJECT_FRAME_SKIP")
fi

python preprocessing/segmentation/prepare_sam2object_for_objectx.py "${PREPARE_ARGS[@]}"

echo "========== DONE =========="
