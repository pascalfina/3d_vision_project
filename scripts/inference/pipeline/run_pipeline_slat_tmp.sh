#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
USER_ROOT="${OBJECTX_USER_ROOT:-/work/scratch/${USER:-$(id -un)}}"
SCRATCH_ROOT="${DATA_ROOT_DIR:-${OBJECTX_BASELINE_ROOT:-$USER_ROOT/objectx-data-baseline}}"
TMP_INFER_ROOT="${TMP_INFER_ROOT:-/tmp/${USER}-objectx-infer}"
RESET_TMP="${RESET_TMP:-1}"
SPLIT="${SPLIT:-val}"
SCENE_ID="${SCENE_ID:-}"

source "$REPO_ROOT/scripts/activate_objectx_env.sh"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

python -u "$REPO_ROOT/scripts/inference/pipeline/run_pipeline_tmp.py" \
  --repo-root "$REPO_ROOT" \
  --scratch-root "$SCRATCH_ROOT" \
  --tmp-root "$TMP_INFER_ROOT" \
  --mode slat \
  --split "$SPLIT" \
  "${scene_args[@]}" \
  "${extra[@]}"
