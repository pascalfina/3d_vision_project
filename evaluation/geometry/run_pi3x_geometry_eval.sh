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
STRICT_SCENE_GUARD="${STRICT_SCENE_GUARD:-1}"
STRICT_SEQUENCE_COLOR_GUARD="${STRICT_SEQUENCE_COLOR_GUARD:-1}"
GEOMETRY_DATASET="${GEOMETRY_DATASET:-${DATASET:-3rscan}}"
GT_SEQUENCE_ZIP="${GT_SEQUENCE_ZIP:-}"
GT_SEQUENCE_DIR="${GT_SEQUENCE_DIR:-}"

if [[ -z "$PRED_ROOT" && -z "${PRED_PLY:-}" ]]; then
  echo "Set PRED_ROOT or PRED_PLY" >&2
  exit 2
fi
if [[ -z "$BASELINE_ROOT" && -z "${GT_MESH:-}" ]]; then
  echo "Set BASELINE_ROOT/GT_ROOT or GT_MESH" >&2
  exit 2
fi

source "$REPO_ROOT/evaluation/geometry/geometry_dataset_paths.sh"
geometry_resolve_gt_paths

PRED_PLY="${PRED_PLY:-$PRED_ROOT/scenes/$SCAN_ID/labels.instances.annotated.v2.ply}"
PRED_SEQUENCE_DIR="${PRED_SEQUENCE_DIR:-${VISIBLE_SEQUENCE_DIR:-$PRED_ROOT/scenes_sam2_pi3x/$SCAN_ID/sequence}}"

if [[ -z "${GT_MESH:-}" || ! -f "$GT_MESH" ]]; then
  echo "[geometry-eval] missing GT mesh for GEOMETRY_DATASET=$GEOMETRY_DATASET:" >&2
  echo "  GT_MESH=${GT_MESH:-}" >&2
  exit 2
fi
if [[ -z "${GT_SEQUENCE_ZIP:-}" && -z "${GT_SEQUENCE_DIR:-}" ]]; then
  echo "[geometry-eval] missing GT RGB-D sequence for GEOMETRY_DATASET=$GEOMETRY_DATASET." >&2
  echo "Set GT_SEQUENCE_ZIP for 3RScan or GT_SEQUENCE_DIR for ScanNet." >&2
  exit 2
fi

require_scene_path() {
  local label="$1"
  local path="$2"
  if [[ "$STRICT_SCENE_GUARD" != "1" ]]; then
    return 0
  fi
  local resolved
  resolved="$(readlink -f "$path" 2>/dev/null || printf '%s' "$path")"
  if [[ "$resolved" != *"/$SCAN_ID/"* && "$resolved" != *"/$SCAN_ID."* && "$resolved" != *"/${SCAN_ID}_"* ]]; then
    echo "[geometry-eval] refusing suspicious $label path for SCAN_ID=$SCAN_ID:" >&2
    echo "  $resolved" >&2
    echo "Set STRICT_SCENE_GUARD=0 only if this is intentional." >&2
    exit 2
  fi
}

require_sequence_color_match() {
  local pred_sequence="$1"
  local gt_sequence_zip="$2"
  local gt_sequence_dir="$3"
  local dataset="$4"
  if [[ "$STRICT_SEQUENCE_COLOR_GUARD" != "1" ]]; then
    return 0
  fi
  "$PYTHON_BIN" - "$pred_sequence" "$gt_sequence_zip" "$gt_sequence_dir" "$dataset" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

pred_sequence = Path(sys.argv[1])
gt_sequence_zip = Path(sys.argv[2]) if sys.argv[2] else None
gt_sequence_dir = Path(sys.argv[3]) if sys.argv[3] else None
dataset = sys.argv[4]

pred_colors = sorted(pred_sequence.glob("frame-*.color.jpg"))
if not pred_colors:
    raise SystemExit(f"[geometry-eval] no frame-*.color.jpg files in: {pred_sequence}")

idxs = [0, 1, len(pred_colors) // 2, max(0, len(pred_colors) // 2 + 1), max(0, len(pred_colors) - 2), max(0, len(pred_colors) - 1)]
sample = []
for idx in idxs:
    if idx < len(pred_colors) and pred_colors[idx] not in sample:
        sample.append(pred_colors[idx])

mismatches = []
checked = 0

def image_content_matches(pred_path: Path, gt_path: Path) -> bool:
    pred = Image.open(pred_path).convert("RGB")
    gt = Image.open(gt_path).convert("RGB")
    if pred.size != gt.size:
        gt = gt.resize(pred.size, Image.Resampling.BILINEAR)
    # Downsample before comparing so harmless JPEG/resizing differences do not
    # look like a scene mismatch.
    pred = pred.resize((64, 64), Image.Resampling.BILINEAR)
    gt = gt.resize((64, 64), Image.Resampling.BILINEAR)
    diff = np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32))
    return float(diff.mean()) <= 8.0

def pred_frame_id(path: Path) -> str:
    name = path.name
    return name.removeprefix("frame-").removesuffix(".color.jpg").removesuffix(".color.png")

def scannet_color_path(root: Path, frame_id: str) -> Path | None:
    if (root / "data" / "color").is_dir():
        root = root / "data"
    variants = [frame_id]
    try:
        number = int(frame_id)
    except ValueError:
        pass
    else:
        variants.extend([str(number), f"{number:06d}"])
    seen = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        for suffix in (".jpg", ".png"):
            path = root / "color" / f"{variant}{suffix}"
            if path.exists():
                return path
    return None

if dataset == "scannet":
    if gt_sequence_dir is None or not gt_sequence_dir.exists():
        raise SystemExit(f"[geometry-eval] missing ScanNet GT sequence dir: {gt_sequence_dir}")
    for pred_path in sample:
        frame_id = pred_frame_id(pred_path)
        gt_path = scannet_color_path(gt_sequence_dir, frame_id)
        if gt_path is None:
            mismatches.append(f"{pred_path.name}: missing in ScanNet GT color")
            continue
        pred_hash = hashlib.sha256(pred_path.read_bytes()).hexdigest()
        gt_hash = hashlib.sha256(gt_path.read_bytes()).hexdigest()
        checked += 1
        if pred_hash != gt_hash:
            if not image_content_matches(pred_path, gt_path):
                mismatches.append(f"{pred_path.name}: pred={pred_hash[:12]} gt={gt_hash[:12]}")
else:
    if gt_sequence_zip is None or not gt_sequence_zip.exists():
        raise SystemExit(f"[geometry-eval] missing GT sequence zip: {gt_sequence_zip}")
    with ZipFile(gt_sequence_zip) as zf:
        gt_names = set(zf.namelist())
        for pred_path in sample:
            name = pred_path.name
            if name not in gt_names:
                mismatches.append(f"{name}: missing in GT")
                continue
            pred_hash = hashlib.sha256(pred_path.read_bytes()).hexdigest()
            gt_hash = hashlib.sha256(zf.read(name)).hexdigest()
            checked += 1
            if pred_hash != gt_hash:
                mismatches.append(f"{name}: pred={pred_hash[:12]} gt={gt_hash[:12]}")

if mismatches:
    raise SystemExit(
        "[geometry-eval] prediction color frames do not match this scan's GT sequence.\n"
        f"  pred_sequence={pred_sequence}\n"
        f"  gt_sequence_zip={gt_sequence_zip}\n"
        f"  gt_sequence_dir={gt_sequence_dir}\n"
        "  mismatches:\n  " + "\n  ".join(mismatches[:8]) + "\n"
        "Set STRICT_SEQUENCE_COLOR_GUARD=0 only if this is intentional."
    )
print(f"[geometry-eval] color guard passed: {checked}/{len(sample)} sampled frames match GT", file=sys.stderr)
PY
}

PRED_INPUT_MODE="${PRED_INPUT_MODE:-auto}"
if [[ "$PRED_INPUT_MODE" == "auto" ]]; then
  if [[ -d "$PRED_SEQUENCE_DIR" ]] && compgen -G "$PRED_SEQUENCE_DIR/frame-*.xyz.npy" >/dev/null; then
    PRED_INPUT_MODE="sequence"
  elif [[ -f "$PRED_PLY" ]]; then
    PRED_INPUT_MODE="ply"
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
  require_scene_path "prediction PLY" "$PRED_PLY"
  if [[ -d "$PRED_SEQUENCE_DIR" && ( -f "${GT_SEQUENCE_ZIP:-}" || -d "${GT_SEQUENCE_DIR:-}" ) ]]; then
    require_sequence_color_match "$PRED_SEQUENCE_DIR" "${GT_SEQUENCE_ZIP:-}" "${GT_SEQUENCE_DIR:-}" "$GEOMETRY_DATASET"
  fi
  PRED_ARGS=(--pred-ply "$PRED_PLY")
  DEFAULT_ALIGN="rgbd_correspondence"
  DEFAULT_PRED_VOXEL_SIZE="0.0"
elif [[ "$PRED_INPUT_MODE" == "sequence" ]]; then
  require_scene_path "prediction sequence" "$PRED_SEQUENCE_DIR"
  if ! compgen -G "$PRED_SEQUENCE_DIR/frame-*.xyz.npy" >/dev/null; then
    echo "[geometry-eval] no frame-*.xyz.npy files in prediction sequence:" >&2
    echo "  $PRED_SEQUENCE_DIR" >&2
    exit 2
  fi
  require_sequence_color_match "$PRED_SEQUENCE_DIR" "${GT_SEQUENCE_ZIP:-}" "${GT_SEQUENCE_DIR:-}" "$GEOMETRY_DATASET"
  PRED_ARGS=(
    --pred-sequence-dir "$PRED_SEQUENCE_DIR"
    --sequence-conf-thr "${SEQUENCE_CONF_THR:-0.10}"
    --sequence-pixel-stride "${SEQUENCE_PIXEL_STRIDE:-2}"
    --sequence-max-frames "${SEQUENCE_MAX_FRAMES:-0}"
  )
  DEFAULT_ALIGN="rgbd_correspondence"
  DEFAULT_PRED_VOXEL_SIZE="0.03"
elif [[ "$PRED_INPUT_MODE" == "sequence_gt_pose" ]]; then
  require_scene_path "prediction sequence" "$PRED_SEQUENCE_DIR"
  if ! compgen -G "$PRED_SEQUENCE_DIR/frame-*.xyz.npy" >/dev/null; then
    echo "[geometry-eval] no frame-*.xyz.npy files in prediction sequence:" >&2
    echo "  $PRED_SEQUENCE_DIR" >&2
    exit 2
  fi
  require_sequence_color_match "$PRED_SEQUENCE_DIR" "${GT_SEQUENCE_ZIP:-}" "${GT_SEQUENCE_DIR:-}" "$GEOMETRY_DATASET"
  SEQUENCE_POSE_ARGS=()
  if [[ -n "${GT_SEQUENCE_ZIP:-}" ]]; then
    SEQUENCE_POSE_ARGS=(--sequence-pose-zip "$GT_SEQUENCE_ZIP")
  else
    SEQUENCE_POSE_ARGS=(--sequence-pose-dir "$GT_SEQUENCE_DIR")
  fi
  PRED_ARGS=(
    --pred-sequence-dir "$PRED_SEQUENCE_DIR"
    "${SEQUENCE_POSE_ARGS[@]}"
    --sequence-camera-axis-signs "${SEQUENCE_CAMERA_AXIS_SIGNS:-1,1,1}"
    --sequence-conf-thr "${SEQUENCE_CONF_THR:-0.10}"
    --sequence-pixel-stride "${SEQUENCE_PIXEL_STRIDE:-2}"
    --sequence-max-frames "${SEQUENCE_MAX_FRAMES:-0}"
  )
else
  echo "Unknown PRED_INPUT_MODE=$PRED_INPUT_MODE (expected: auto, ply, sequence, or sequence_gt_pose)" >&2
  exit 2
fi

require_scene_path "GT mesh" "$GT_MESH"
if [[ -n "${GT_SEQUENCE_ZIP:-}" ]]; then
  require_scene_path "GT sequence zip" "$GT_SEQUENCE_ZIP"
else
  require_scene_path "GT sequence dir" "$GT_SEQUENCE_DIR"
fi
GT_SEQUENCE_ARGS=()
if [[ -n "${GT_SEQUENCE_ZIP:-}" ]]; then
  GT_SEQUENCE_ARGS=(
    --visible-gt-sequence-zip "$GT_SEQUENCE_ZIP"
    --pose-align-gt-sequence-zip "$GT_SEQUENCE_ZIP"
    --rgbd-gt-sequence-zip "$GT_SEQUENCE_ZIP"
  )
else
  GT_SEQUENCE_ARGS=(
    --visible-gt-sequence-dir "$GT_SEQUENCE_DIR"
    --pose-align-gt-sequence-dir "$GT_SEQUENCE_DIR"
    --rgbd-gt-sequence-dir "$GT_SEQUENCE_DIR"
  )
fi
echo "[geometry-eval] dataset=$GEOMETRY_DATASET" >&2
echo "[geometry-eval] scan=$SCAN_ID method=$METHOD_NAME input=$PRED_INPUT_MODE" >&2
echo "[geometry-eval] pred_sequence=$PRED_SEQUENCE_DIR" >&2
echo "[geometry-eval] pred_ply=$PRED_PLY" >&2
echo "[geometry-eval] gt_mesh=$GT_MESH" >&2
echo "[geometry-eval] gt_sequence_zip=${GT_SEQUENCE_ZIP:-}" >&2
echo "[geometry-eval] gt_sequence_dir=${GT_SEQUENCE_DIR:-}" >&2

"$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/evaluate_geometry_against_gt.py" \
  "${PRED_ARGS[@]}" \
  --scene-id "$SCAN_ID" \
  --method-name "$METHOD_NAME" \
  --gt-mesh "$GT_MESH" \
  --pose-align-pred-sequence-dir "$PRED_SEQUENCE_DIR" \
  --rgbd-pred-sequence-dir "$PRED_SEQUENCE_DIR" \
  "${GT_SEQUENCE_ARGS[@]}" \
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

if [[ "${AUTO_SUMMARIZE_GEOMETRY:-1}" != "0" ]]; then
  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/summarize_geometry_runs.py" \
      --metrics-glob "${GEOMETRY_SUMMARY_GLOB:-evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/**/metrics.json}" \
      --out-dir "${GEOMETRY_SUMMARY_OUT_DIR:-evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP}"
  )
fi
