#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"

SCAN_ID="${SCAN_ID:?Set SCAN_ID to the scene id to evaluate}"
METHOD_NAME="${METHOD_NAME:-pi3x}"
PRED_ROOT="${PRED_ROOT:-}"
BASELINE_ROOT="${BASELINE_ROOT:-${GT_ROOT:-}}"
GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-final}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/$METHOD_NAME/$SCAN_ID}"
WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"

if [[ -z "$PRED_ROOT" && -z "${PRED_PLY:-}" ]]; then
  echo "Set PRED_ROOT or PRED_PLY" >&2
  exit 2
fi
if [[ -z "$BASELINE_ROOT" && -z "${GT_MESH:-}" ]]; then
  echo "Set BASELINE_ROOT/GT_ROOT or GT_MESH" >&2
  exit 2
fi

PRED_PLY="${PRED_PLY:-$PRED_ROOT/scenes/$SCAN_ID/labels.instances.annotated.v2.ply}"
GT_MESH="${GT_MESH:-$BASELINE_ROOT/scenes/$SCAN_ID/mesh.refined.v2.obj}"
PRED_SEQUENCE_DIR="${PRED_SEQUENCE_DIR:-${VISIBLE_SEQUENCE_DIR:-$PRED_ROOT/scenes_sam2_pi3x/$SCAN_ID/sequence}}"
GT_SEQUENCE_ZIP="${GT_SEQUENCE_ZIP:-$BASELINE_ROOT/scenes/$SCAN_ID/sequence.zip}"
PRED_INPUT_MODE="${PRED_INPUT_MODE:-auto}"
if [[ "$PRED_INPUT_MODE" == "auto" ]]; then
  if [[ -f "$PRED_PLY" ]]; then
    PRED_INPUT_MODE="ply"
  elif [[ -d "$PRED_SEQUENCE_DIR" ]]; then
    PRED_INPUT_MODE="sequence"
  else
    echo "Could not auto-resolve prediction input." >&2
    echo "Missing PLY: $PRED_PLY" >&2
    echo "Missing sequence dir: $PRED_SEQUENCE_DIR" >&2
    exit 2
  fi
fi

PRED_ARGS=()
DEFAULT_ALIGN="none"
DEFAULT_PRED_VOXEL_SIZE="0.03"
if [[ "$PRED_INPUT_MODE" == "ply" ]]; then
  PRED_ARGS=(--pred-ply "$PRED_PLY")
  DEFAULT_ALIGN="rgbd_correspondence"
  DEFAULT_PRED_VOXEL_SIZE="0.0"
elif [[ "$PRED_INPUT_MODE" == "sequence" ]]; then
  PRED_ARGS=(
    --pred-sequence-dir "$PRED_SEQUENCE_DIR"
    --sequence-conf-thr "${SEQUENCE_CONF_THR:-0.10}"
    --sequence-pixel-stride "${SEQUENCE_PIXEL_STRIDE:-2}"
    --sequence-max-frames "${SEQUENCE_MAX_FRAMES:-0}"
  )
  DEFAULT_ALIGN="rgbd_correspondence"
  DEFAULT_PRED_VOXEL_SIZE="0.03"
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
  echo "Unknown PRED_INPUT_MODE=$PRED_INPUT_MODE (expected: auto, ply, sequence, or sequence_gt_pose)" >&2
  exit 2
fi

"$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/evaluate_geometry_against_gt.py" \
  "${PRED_ARGS[@]}" \
  --scene-id "$SCAN_ID" \
  --method-name "$METHOD_NAME" \
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
  --report-threshold "${REPORT_THRESHOLD:-0.05}" \
  --debug-html-max-points "${DEBUG_HTML_MAX_POINTS:-300000}" \
  ${WRITE_DEBUG_PLY:+--write-debug-ply} \
  ${WRITE_DEBUG_PLY:+--write-debug-html} \
  ${WRITE_DEBUG_HTML:+--write-debug-html}
