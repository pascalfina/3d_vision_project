#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/segmentation/visualization/render_joint_depth_background_bundle.sh \
    --scan-id <scan_id> \
    --replacement-root <replacement_root> \
    --label <label> \
    [--manifest <manifest.json>] \
    [--data-root <data_root>] \
    [--mask-root <mask_root>] \
    [--joint-ply <joint_ply>] \
    [--mask-source <mask_source>] \
    [--background-remove-mode <loaded|all>] \
    [--pose-mode <raw|invert>] \
    [--lift-coord-system <pinhole|scan3r>] \
    [--frame-selection <all|top_area|diverse_area>] \
    [--max-views <int>] \
    [--bg-max-points <int>] \
    [--joint-max-points <int>] \
    [--joint-opacity-min <float>] \
    [--joint-opacity-quantile <float>] \
    [--joint-scale-quantile <float>] \
    [--fit-mode <all|objects>] \
    [--fit-margin <float>] \
    [--num-frames <int>] \
    [--fps <int>] \
    [--fig-width <float>] \
    [--fig-height <float>] \
    [--bg-point-size <float>] \
    [--joint-point-size <float>] \
    [--out-dir <path>]

This wrapper activates the Object-X environment and produces both:
  - MP4 orbit render
  - interactive HTML point cloud
EOF
}

SCAN_ID=""
REPLACEMENT_ROOT=""
LABEL=""
MANIFEST=""
DATA_ROOT="/work/scratch/${USER}/objectx-data-baseline"
MASK_ROOT=""
JOINT_PLY=""
MASK_SOURCE="gt_projection"
BACKGROUND_REMOVE_MODE="loaded"
POSE_MODE="raw"
LIFT_COORD_SYSTEM="pinhole"
FRAME_SELECTION="all"
MAX_VIEWS="9999"
BG_MAX_POINTS="250000"
JOINT_MAX_POINTS="250000"
JOINT_OPACITY_MIN="0.0"
JOINT_OPACITY_QUANTILE="0.8"
JOINT_SCALE_QUANTILE="0.95"
FIT_MODE="all"
FIT_MARGIN="1.08"
NUM_FRAMES="48"
FPS="24"
FIG_WIDTH="12.0"
FIG_HEIGHT="7.0"
BG_POINT_SIZE="0.35"
JOINT_POINT_SIZE="1.0"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scan-id)
            SCAN_ID="$2"
            shift 2
            ;;
        --replacement-root)
            REPLACEMENT_ROOT="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --mask-root)
            MASK_ROOT="$2"
            shift 2
            ;;
        --joint-ply)
            JOINT_PLY="$2"
            shift 2
            ;;
        --mask-source)
            MASK_SOURCE="$2"
            shift 2
            ;;
        --background-remove-mode)
            BACKGROUND_REMOVE_MODE="$2"
            shift 2
            ;;
        --pose-mode)
            POSE_MODE="$2"
            shift 2
            ;;
        --lift-coord-system)
            LIFT_COORD_SYSTEM="$2"
            shift 2
            ;;
        --frame-selection)
            FRAME_SELECTION="$2"
            shift 2
            ;;
        --max-views)
            MAX_VIEWS="$2"
            shift 2
            ;;
        --bg-max-points)
            BG_MAX_POINTS="$2"
            shift 2
            ;;
        --joint-max-points)
            JOINT_MAX_POINTS="$2"
            shift 2
            ;;
        --joint-opacity-min)
            JOINT_OPACITY_MIN="$2"
            shift 2
            ;;
        --joint-opacity-quantile)
            JOINT_OPACITY_QUANTILE="$2"
            shift 2
            ;;
        --joint-scale-quantile)
            JOINT_SCALE_QUANTILE="$2"
            shift 2
            ;;
        --fit-mode)
            FIT_MODE="$2"
            shift 2
            ;;
        --fit-margin)
            FIT_MARGIN="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --fig-width)
            FIG_WIDTH="$2"
            shift 2
            ;;
        --fig-height)
            FIG_HEIGHT="$2"
            shift 2
            ;;
        --bg-point-size)
            BG_POINT_SIZE="$2"
            shift 2
            ;;
        --joint-point-size)
            JOINT_POINT_SIZE="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$SCAN_ID" || -z "$REPLACEMENT_ROOT" || -z "$LABEL" ]]; then
    usage >&2
    exit 1
fi

if [[ -z "$JOINT_PLY" ]]; then
    JOINT_PLY="$REPO_ROOT/vis/${SCAN_ID}_joint.ply"
fi

if [[ -z "$MANIFEST" ]]; then
    MANIFEST="$(python - <<PY
import json
from pathlib import Path

scan_id = ${SCAN_ID@Q}
debug_dir = Path(${REPO_ROOT@Q}) / "debug"
candidates = []
for path in sorted(debug_dir.glob("*manifest*.json")):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    if data.get("scene_id") == scan_id:
        candidates.append(path)
print(candidates[0] if candidates else "")
PY
)"
fi

if [[ -z "$MANIFEST" ]]; then
    echo "Could not auto-resolve a manifest for scan_id=$SCAN_ID. Please pass --manifest." >&2
    exit 1
fi

if [[ ! -f "$JOINT_PLY" ]]; then
    echo "Joint PLY not found: $JOINT_PLY" >&2
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

source "$REPO_ROOT/scripts/activate_objectx_env.sh"

cmd=(
    python "$REPO_ROOT/scripts/segmentation/visualization/render_joint_depth_background.py"
    --data-root "$DATA_ROOT"
    --replacement-root "$REPLACEMENT_ROOT"
    --joint-ply "$JOINT_PLY"
    --scan-id "$SCAN_ID"
    --manifest "$MANIFEST"
    --mask-source "$MASK_SOURCE"
    --background-remove-mode "$BACKGROUND_REMOVE_MODE"
    --pose-mode "$POSE_MODE"
    --lift-coord-system "$LIFT_COORD_SYSTEM"
    --frame-selection "$FRAME_SELECTION"
    --max-views "$MAX_VIEWS"
    --bg-max-points "$BG_MAX_POINTS"
    --joint-max-points "$JOINT_MAX_POINTS"
    --joint-opacity-min "$JOINT_OPACITY_MIN"
    --joint-opacity-quantile "$JOINT_OPACITY_QUANTILE"
    --joint-scale-quantile "$JOINT_SCALE_QUANTILE"
    --fit-mode "$FIT_MODE"
    --fit-margin "$FIT_MARGIN"
    --num-frames "$NUM_FRAMES"
    --fps "$FPS"
    --fig-width "$FIG_WIDTH"
    --fig-height "$FIG_HEIGHT"
    --bg-point-size "$BG_POINT_SIZE"
    --joint-point-size "$JOINT_POINT_SIZE"
    --label "$LABEL"
)

if [[ -n "$MASK_ROOT" ]]; then
    cmd=( "${cmd[@]:0:2}" --mask-root "$MASK_ROOT" "${cmd[@]:2}" )
fi

if [[ -n "$OUT_DIR" ]]; then
    cmd+=(--out-dir "$OUT_DIR")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
