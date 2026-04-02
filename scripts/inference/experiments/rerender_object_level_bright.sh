#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT_DIR:-/work/scratch/pafina/objectx-data-predseg-objects5}"
SPLIT="${SPLIT:-val}"

cd "$REPO_ROOT"
source scripts/activate_objectx_env.sh

export DATA_ROOT_DIR="$DATA_ROOT"
export OBJECTX_MASK_SOURCE="${OBJECTX_MASK_SOURCE:-pred_projection_clean}"
export OBJECTX_VIS_SKIP_GS="${OBJECTX_VIS_SKIP_GS:-1}"
export OBJECTX_VIS_ORBIT_MODE="${OBJECTX_VIS_ORBIT_MODE:-object}"
export OBJECTX_VIS_FIT_MARGIN="${OBJECTX_VIS_FIT_MARGIN:-1.25}"
export OBJECTX_VIS_MIN_RADIUS="${OBJECTX_VIS_MIN_RADIUS:-0.18}"
export OBJECTX_VIS_MIN_HEIGHT="${OBJECTX_VIS_MIN_HEIGHT:-0.04}"
export OBJECTX_VIS_VERTICAL_LIFT="${OBJECTX_VIS_VERTICAL_LIFT:-0.06}"
export OBJECTX_VIS_RENDER_SCALE="${OBJECTX_VIS_RENDER_SCALE:-1.0}"
export OBJECTX_VIS_BG_COLOR="${OBJECTX_VIS_BG_COLOR:-0.95,0.95,0.95}"
export OBJECTX_VIS_EXPOSURE="${OBJECTX_VIS_EXPOSURE:-1.6}"
export OBJECTX_VIS_GAMMA="${OBJECTX_VIS_GAMMA:-0.85}"
export OBJECTX_VIS_BLACK_FLOOR="${OBJECTX_VIS_BLACK_FLOOR:-0.0}"
export RESET_TMP="${RESET_TMP:-1}"
export SPLIT

bash scripts/inference/pipeline/run_pipeline_u3dgs_tmp.sh --visualize

mapfile -t SCAN_IDS < <(grep -v '^\s*$' "$DATA_ROOT/files/${SPLIT}_resplit_scans.txt" || true)
for scan_id in "${SCAN_IDS[@]}"; do
  src="vis/rendered/${scan_id}_orbit_rendered.mp4"
  dst="vis/rendered/${scan_id}_orbit_rendered_objectcentric_bright.mp4"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "Saved bright render to $dst"
  fi
done
