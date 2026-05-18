#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"

SCAN_ID="${SCAN_ID:-5341b7e3-8a66-2cdd-8709-66a2159f0017}"
PRED_ROOT="${PRED_ROOT:-/work/scratch/pafina/objectx-data-fullscene-oven-pi3x}"
BASELINE_ROOT="${BASELINE_ROOT:-/work/scratch/pafina/objectx-data-baseline}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/evaluation/outputs/geometry/oven_pi3x}"

PRED_PLY="${PRED_PLY:-$PRED_ROOT/scenes/$SCAN_ID/labels.instances.annotated.v2.ply}"
GT_MESH="${GT_MESH:-$BASELINE_ROOT/scenes/$SCAN_ID/mesh.refined.v2.obj}"
PRED_SEQUENCE_DIR="${PRED_SEQUENCE_DIR:-${VISIBLE_SEQUENCE_DIR:-$PRED_ROOT/scenes_sam2_pi3x/$SCAN_ID/sequence}}"
GT_SEQUENCE_ZIP="${GT_SEQUENCE_ZIP:-$BASELINE_ROOT/scenes/$SCAN_ID/sequence.zip}"
PRED_INPUT_MODE="${PRED_INPUT_MODE:-ply}"

PRED_ARGS=()
DEFAULT_ALIGN="none"
DEFAULT_PRED_VOXEL_SIZE="0.03"
if [[ "$PRED_INPUT_MODE" == "ply" ]]; then
  PRED_ARGS=(--pred-ply "$PRED_PLY")
  DEFAULT_ALIGN="rgbd_correspondence"
  DEFAULT_PRED_VOXEL_SIZE="0.0"
elif [[ "$PRED_INPUT_MODE" == "sequence_gt_pose" ]]; then
  PRED_ARGS=(
    --pred-sequence-dir "$PRED_SEQUENCE_DIR"
    --sequence-pose-zip "$GT_SEQUENCE_ZIP"
    --sequence-camera-axis-signs "${SEQUENCE_CAMERA_AXIS_SIGNS:-1,1,1}"
    --sequence-conf-thr "${SEQUENCE_CONF_THR:-0.10}"
    --sequence-pixel-stride "${SEQUENCE_PIXEL_STRIDE:-2}"
    --sequence-max-frames "${SEQUENCE_MAX_FRAMES:-0}"
  )
else
  echo "Unknown PRED_INPUT_MODE=$PRED_INPUT_MODE (expected: sequence_gt_pose or ply)" >&2
  exit 2
fi

"$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/evaluate_geometry_against_gt.py" \
  "${PRED_ARGS[@]}" \
  --gt-mesh "$GT_MESH" \
  --visible-gt-sequence-zip "$GT_SEQUENCE_ZIP" \
  --pose-align-pred-sequence-dir "$PRED_SEQUENCE_DIR" \
  --pose-align-gt-sequence-zip "$GT_SEQUENCE_ZIP" \
  --rgbd-pred-sequence-dir "$PRED_SEQUENCE_DIR" \
  --rgbd-gt-sequence-zip "$GT_SEQUENCE_ZIP" \
  --rgbd-frame-stride "${RGBD_FRAME_STRIDE:-4}" \
  --rgbd-max-frames "${RGBD_MAX_FRAMES:-96}" \
  --rgbd-pixel-stride "${RGBD_PIXEL_STRIDE:-6}" \
  --rgbd-conf-thr "${RGBD_CONF_THR:-0.10}" \
  --rgbd-max-correspondences "${RGBD_MAX_CORRESPONDENCES:-50000}" \
  --rgbd-trim-quantile "${RGBD_TRIM_QUANTILE:-65.0}" \
  --rgbd-fit-iterations "${RGBD_FIT_ITERATIONS:-8}" \
  --out-dir "$OUT_DIR" \
  --sample-gt-points "${SAMPLE_GT_POINTS:-400000}" \
  --visible-max-frames "${VISIBLE_MAX_FRAMES:-96}" \
  --visible-frame-stride "${VISIBLE_FRAME_STRIDE:-1}" \
  --pred-voxel-size "${PRED_VOXEL_SIZE:-$DEFAULT_PRED_VOXEL_SIZE}" \
  --max-pred-points "${MAX_PRED_POINTS:-600000}" \
  --align "${ALIGN:-$DEFAULT_ALIGN}" \
  --pose-align-frame-stride "${POSE_ALIGN_FRAME_STRIDE:-1}" \
  --pose-align-max-frames "${POSE_ALIGN_MAX_FRAMES:-0}" \
  --axis-bbox-refine-top-k "${AXIS_BBOX_REFINE_TOP_K:-8}" \
  --coord-bbox-low-q "${COORD_BBOX_LOW_Q:-0.0}" \
  --coord-bbox-high-q "${COORD_BBOX_HIGH_Q:-100.0}" \
  --icp-sample-points "${ICP_SAMPLE_POINTS:-50000}" \
  --icp-iterations "${ICP_ITERATIONS:-30}" \
  --thresholds ${THRESHOLDS:-0.02 0.05 0.10} \
  --debug-html-max-points "${DEBUG_HTML_MAX_POINTS:-300000}" \
  ${WRITE_DEBUG_PLY:+--write-debug-ply} \
  ${WRITE_DEBUG_PLY:+--write-debug-html} \
  ${WRITE_DEBUG_HTML:+--write-debug-html}
