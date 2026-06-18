#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"

SCAN_ID="${SCAN_ID:?Set SCAN_ID to the scene id to evaluate}"
METHOD_NAME="${METHOD_NAME:-objectx_final}"
GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-objectx_final}"

PRED_ROOT="${PRED_ROOT:-}"
PRED_READY_ROOT="${PRED_READY_ROOT:-${INPUT_ROOT:-}}"
BASELINE_ROOT="${BASELINE_ROOT:-${GT_ROOT:-}}"
SCENES_DIRNAME="${SCENES_DIRNAME:-scenes_sam2_pi3x}"
GEOMETRY_DATASET="${GEOMETRY_DATASET:-${DATASET:-3rscan}}"
GT_SEQUENCE_ZIP="${GT_SEQUENCE_ZIP:-}"
GT_SEQUENCE_DIR="${GT_SEQUENCE_DIR:-}"

FINAL_PLY="${FINAL_PLY:-$REPO_ROOT/vis/${SCAN_ID}_joint.ply}"
PRED_SEQUENCE_DIR="${PRED_SEQUENCE_DIR:-${PRED_ROOT:+$PRED_ROOT/$SCENES_DIRNAME/$SCAN_ID/sequence}}"

source "$REPO_ROOT/evaluation/geometry/geometry_dataset_paths.sh"
geometry_resolve_gt_paths

OUT_DIR="${OUT_DIR:-$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/$METHOD_NAME/$SCAN_ID}"
FINAL_VS_GT_OUT_DIR="${FINAL_VS_GT_OUT_DIR:-$OUT_DIR/final_vs_gt}"
FINAL_VS_INPUT_OUT_DIR="${FINAL_VS_INPUT_OUT_DIR:-$OUT_DIR/final_vs_objectx_input}"
FINAL_VS_RAW_PI3X_OUT_DIR="${FINAL_VS_RAW_PI3X_OUT_DIR:-$OUT_DIR/final_vs_raw_pi3x}"
WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"
RAW_REFERENCE_NAME="${RAW_REFERENCE_NAME:-raw_pi3x_sequence}"
RAW_REFERENCE_LABEL="${RAW_REFERENCE_LABEL:-Raw Pi3X}"

if [[ "${RUN_OBJECTX:-0}" == "1" ]]; then
  PROFILE="${PROFILE:?Set PROFILE when RUN_OBJECTX=1 so slat/u3dgs can be run first}"
  bash "$REPO_ROOT/scripts/workflows/run_scene_profile.sh" "$PROFILE" slat
  bash "$REPO_ROOT/scripts/workflows/run_scene_profile.sh" "$PROFILE" u3dgs
fi

if [[ ! -f "$FINAL_PLY" ]]; then
  echo "[objectx-final-eval] missing final Object-X PLY:" >&2
  echo "  $FINAL_PLY" >&2
  echo "Run the profile's u3dgs action first, or set FINAL_PLY explicitly." >&2
  exit 2
fi
if [[ -z "$PRED_ROOT" || -z "$PRED_SEQUENCE_DIR" || ! -d "$PRED_SEQUENCE_DIR" ]]; then
  echo "[objectx-final-eval] missing Pi3X prediction sequence." >&2
  echo "  PRED_ROOT=$PRED_ROOT" >&2
  echo "  PRED_SEQUENCE_DIR=$PRED_SEQUENCE_DIR" >&2
  exit 2
fi
if [[ -z "$PRED_READY_ROOT" || ! -d "$PRED_READY_ROOT/files/gs_annotations/$SCAN_ID" ]]; then
  echo "[objectx-final-eval] missing Object-X input gs_annotations." >&2
  echo "  PRED_READY_ROOT=$PRED_READY_ROOT" >&2
  echo "Expected: $PRED_READY_ROOT/files/gs_annotations/$SCAN_ID" >&2
  exit 2
fi
if [[ -z "$BASELINE_ROOT" && -z "$GT_MESH" ]]; then
  echo "[objectx-final-eval] set BASELINE_ROOT or GT_MESH for final-vs-GT." >&2
  exit 2
fi

mkdir -p "$FINAL_VS_GT_OUT_DIR" "$FINAL_VS_INPUT_OUT_DIR" "$FINAL_VS_RAW_PI3X_OUT_DIR"

echo "[objectx-final-eval] scan=$SCAN_ID method=$METHOD_NAME" >&2
echo "[objectx-final-eval] final_ply=$FINAL_PLY" >&2
echo "[objectx-final-eval] objectx_input=$PRED_READY_ROOT/files/gs_annotations/$SCAN_ID" >&2
echo "[objectx-final-eval] raw_sequence=$PRED_SEQUENCE_DIR reference=$RAW_REFERENCE_NAME" >&2

AUTO_SUMMARIZE_GEOMETRY=0 \
SCAN_ID="$SCAN_ID" \
METHOD_NAME="${METHOD_NAME}_final_vs_gt" \
PRED_INPUT_MODE="ply" \
PRED_PLY="$FINAL_PLY" \
PRED_ROOT="$PRED_ROOT" \
PRED_SEQUENCE_DIR="$PRED_SEQUENCE_DIR" \
BASELINE_ROOT="$BASELINE_ROOT" \
GT_MESH="$GT_MESH" \
GT_SEQUENCE_ZIP="$GT_SEQUENCE_ZIP" \
GT_SEQUENCE_DIR="$GT_SEQUENCE_DIR" \
GEOMETRY_DATASET="$GEOMETRY_DATASET" \
OUT_DIR="$FINAL_VS_GT_OUT_DIR" \
WRITE_DEBUG_HTML="$WRITE_DEBUG_HTML" \
STRICT_SCENE_GUARD="${STRICT_SCENE_GUARD:-1}" \
STRICT_SEQUENCE_COLOR_GUARD="${STRICT_SEQUENCE_COLOR_GUARD:-1}" \
MAX_PRED_POINTS="${FINAL_VS_GT_MAX_PRED_POINTS:-${MAX_PRED_POINTS:-600000}}" \
PRED_VOXEL_SIZE="${FINAL_VS_GT_PRED_VOXEL_SIZE:-${PRED_VOXEL_SIZE:-0.03}}" \
ALIGN="${FINAL_VS_GT_ALIGN:-${ALIGN:-rgbd_correspondence}}" \
bash "$REPO_ROOT/evaluation/geometry/run_pi3x_geometry_eval.sh"

PC_ARGS=(
  --pred-ply "$FINAL_PLY"
  --reference-objectx-root "$PRED_READY_ROOT"
  --scene-id "$SCAN_ID"
  --method-name "$METHOD_NAME"
  --reference-name "objectx_input_gs_annotations"
  --out-dir "$FINAL_VS_INPUT_OUT_DIR"
  --pred-voxel-size "${FINAL_VS_INPUT_PRED_VOXEL_SIZE:-0.03}"
  --reference-voxel-size "${FINAL_VS_INPUT_REFERENCE_VOXEL_SIZE:-0.03}"
  --max-pred-points "${FINAL_VS_INPUT_MAX_PRED_POINTS:-600000}"
  --max-reference-points "${FINAL_VS_INPUT_MAX_REFERENCE_POINTS:-600000}"
  --align "${FINAL_VS_INPUT_ALIGN:-none}"
  --thresholds ${THRESHOLDS:-0.02 0.05 0.10}
  --report-threshold "${REPORT_THRESHOLD:-0.05}"
  --debug-html-max-points "${DEBUG_HTML_MAX_POINTS:-250000}"
)
if [[ -n "${FINAL_PLY_OPACITY_MIN:-}" ]]; then
  PC_ARGS+=(--pred-opacity-min "$FINAL_PLY_OPACITY_MIN")
fi
if [[ -n "${FINAL_PLY_OPACITY_QUANTILE:-}" ]]; then
  PC_ARGS+=(--pred-opacity-quantile "$FINAL_PLY_OPACITY_QUANTILE")
fi
if [[ "$WRITE_DEBUG_HTML" == "1" ]]; then
  PC_ARGS+=(--write-debug-html)
fi

"$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/evaluate_pointcloud_against_pointcloud.py" "${PC_ARGS[@]}"

RAW_ARGS=(
  --pred-ply "$FINAL_PLY"
  --reference-sequence-dir "$PRED_SEQUENCE_DIR"
  --scene-id "$SCAN_ID"
  --method-name "$METHOD_NAME"
  --reference-name "$RAW_REFERENCE_NAME"
  --out-dir "$FINAL_VS_RAW_PI3X_OUT_DIR"
  --sequence-conf-thr "${FINAL_VS_RAW_SEQUENCE_CONF_THR:-${SEQUENCE_CONF_THR:-0.10}}"
  --sequence-pixel-stride "${FINAL_VS_RAW_SEQUENCE_PIXEL_STRIDE:-${SEQUENCE_PIXEL_STRIDE:-2}}"
  --sequence-max-frames "${FINAL_VS_RAW_SEQUENCE_MAX_FRAMES:-${SEQUENCE_MAX_FRAMES:-0}}"
  --pred-voxel-size "${FINAL_VS_RAW_PRED_VOXEL_SIZE:-0.03}"
  --reference-voxel-size "${FINAL_VS_RAW_REFERENCE_VOXEL_SIZE:-0.03}"
  --max-pred-points "${FINAL_VS_RAW_MAX_PRED_POINTS:-600000}"
  --max-reference-points "${FINAL_VS_RAW_MAX_REFERENCE_POINTS:-600000}"
  --align "${FINAL_VS_RAW_ALIGN:-none}"
  --thresholds ${THRESHOLDS:-0.02 0.05 0.10}
  --report-threshold "${REPORT_THRESHOLD:-0.05}"
  --debug-html-max-points "${DEBUG_HTML_MAX_POINTS:-250000}"
)
if [[ -n "${FINAL_PLY_OPACITY_MIN:-}" ]]; then
  RAW_ARGS+=(--pred-opacity-min "$FINAL_PLY_OPACITY_MIN")
fi
if [[ -n "${FINAL_PLY_OPACITY_QUANTILE:-}" ]]; then
  RAW_ARGS+=(--pred-opacity-quantile "$FINAL_PLY_OPACITY_QUANTILE")
fi
if [[ "$WRITE_DEBUG_HTML" == "1" ]]; then
  RAW_ARGS+=(--write-debug-html)
fi

"$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/evaluate_pointcloud_against_pointcloud.py" "${RAW_ARGS[@]}"

"$PYTHON_BIN" - "$OUT_DIR" "$FINAL_VS_GT_OUT_DIR/metrics.json" "$FINAL_VS_INPUT_OUT_DIR/metrics.json" "$FINAL_VS_RAW_PI3X_OUT_DIR/metrics.json" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
gt_path = Path(sys.argv[2])
input_path = Path(sys.argv[3])
raw_path = Path(sys.argv[4])
gt = json.loads(gt_path.read_text())
inp = json.loads(input_path.read_text())
raw = json.loads(raw_path.read_text())
raw_label = os.environ.get("RAW_REFERENCE_LABEL", "Raw Pi3X")

def fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"

def threshold(metrics, key="0.050m"):
    payload = metrics.get("thresholds", {}).get(key)
    if payload is not None:
        return payload
    thresholds = metrics.get("thresholds", {})
    return next(iter(thresholds.values())) if thresholds else {}

scope = "visible_gt"
if scope not in gt.get("scopes", {}):
    scope = next(iter(gt.get("scopes", {})), None)
gt_scope = gt["scopes"][scope] if scope else {}
gt_t = threshold(gt_scope, "0.050m")
inp_t = threshold(inp, "0.050m")
raw_t = threshold(raw, "0.050m")

lines = [
    "# Object-X Final Geometry Evaluation",
    "",
    f"- Scene: `{gt.get('scene_id', inp.get('scene_id'))}`",
    f"- Method: `{inp.get('method_name')}`",
    f"- Final PLY: `{gt.get('pred_source', {}).get('path', 'n/a')}`",
    "",
    "## Final Output vs GT",
    "",
    f"- Scope: `{scope}`",
    f"- Accuracy mean/median/p95: `{fmt(gt.get('pred_to_gt', {}).get('mean'))}` / `{fmt(gt.get('pred_to_gt', {}).get('median'))}` / `{fmt(gt.get('pred_to_gt', {}).get('p95'))}` m",
    f"- Completeness mean/median/p95: `{fmt(gt_scope.get('gt_to_pred', {}).get('mean'))}` / `{fmt(gt_scope.get('gt_to_pred', {}).get('median'))}` / `{fmt(gt_scope.get('gt_to_pred', {}).get('p95'))}` m",
    f"- F1@5cm: `{fmt(gt_t.get('fscore'), 3)}`",
    "",
    "## Final Output vs Object-X Input",
    "",
    f"- Final -> input mean/median/p95: `{fmt(inp.get('final_to_reference', {}).get('mean'))}` / `{fmt(inp.get('final_to_reference', {}).get('median'))}` / `{fmt(inp.get('final_to_reference', {}).get('p95'))}` m",
    f"- Input -> final mean/median/p95: `{fmt(inp.get('reference_to_final', {}).get('mean'))}` / `{fmt(inp.get('reference_to_final', {}).get('median'))}` / `{fmt(inp.get('reference_to_final', {}).get('p95'))}` m",
    f"- F1@5cm: `{fmt(inp_t.get('fscore'), 3)}`",
    "",
    f"## Final Output vs {raw_label}",
    "",
    f"- Final -> {raw_label} mean/median/p95: `{fmt(raw.get('final_to_reference', {}).get('mean'))}` / `{fmt(raw.get('final_to_reference', {}).get('median'))}` / `{fmt(raw.get('final_to_reference', {}).get('p95'))}` m",
    f"- {raw_label} -> final mean/median/p95: `{fmt(raw.get('reference_to_final', {}).get('mean'))}` / `{fmt(raw.get('reference_to_final', {}).get('median'))}` / `{fmt(raw.get('reference_to_final', {}).get('p95'))}` m",
    f"- F1@5cm: `{fmt(raw_t.get('fscore'), 3)}`",
    "",
    "## Artifacts",
    "",
    f"- GT overlay: `{gt_path.parent / 'overlay_pred_gt.html'}`",
    f"- Input overlay: `{input_path.parent / 'overlay_final_reference.html'}`",
    f"- {raw_label} overlay: `{raw_path.parent / 'overlay_final_reference.html'}`",
]
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "report_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[objectx-final-eval] wrote {out_dir / 'report_summary.md'}")
PY
