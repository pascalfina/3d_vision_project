#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH_ROOT="${DATA_ROOT_DIR:-/work/scratch/pafina/objectx-data-baseline}"
TMP_INFER_ROOT="${TMP_INFER_ROOT:-/tmp/${USER}-objectx-infer}"
RESET_TMP="${RESET_TMP:-1}"
SPLIT="${SPLIT:-val}"
SCENE_ID="${SCENE_ID:-}"

source "$REPO_ROOT/scripts/activate_objectx_env.sh"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OBJECTX_VIS_EXPORT_MESH="${OBJECTX_VIS_EXPORT_MESH:-0}"
export OBJECTX_VIS_SKIP_GS="${OBJECTX_VIS_SKIP_GS:-1}"
export OBJECTX_VIS_RENDER_SCALE="${OBJECTX_VIS_RENDER_SCALE:-0.5}"
export OBJECTX_VIS_NUM_FRAMES="${OBJECTX_VIS_NUM_FRAMES:-72}"

if [[ "$RESET_TMP" == "1" ]]; then
  rm -rf "$TMP_INFER_ROOT"
fi

args=("$@")
extra=()
scene_args=()
if [[ ${#args[@]} -gt 0 ]]; then
  extra+=(-- "${args[@]}")
fi
if [[ -n "$SCENE_ID" ]]; then
  scene_args+=(--scene-id "$SCENE_ID")
fi

python -u "$REPO_ROOT/scripts/inference/run_pipeline_tmp.py" \
  --repo-root "$REPO_ROOT" \
  --scratch-root "$SCRATCH_ROOT" \
  --tmp-root "$TMP_INFER_ROOT" \
  --mode u3dgs \
  --split "$SPLIT" \
  "${scene_args[@]}" \
  "${extra[@]}"
