#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH_ROOT="${DATA_ROOT_DIR:-/work/scratch/pafina/objectx-data-baseline}"
TMP_VOX_ROOT="${TMP_VOX_ROOT:-/tmp/${USER}-objectx-voxelise}"
RESET_TMP="${RESET_TMP:-0}"
SPLIT="${SPLIT:-train}"
MAX_SCANS="${MAX_SCANS:-0}"
CACHE_ROOT="${OBJECTX_CACHE_ROOT:-/work/scratch/pafina/objectx-cache}"

if [[ ! -d "$REPO_ROOT/.venv_objx" ]]; then
  echo "Missing venv at $REPO_ROOT/.venv_objx" >&2
  exit 1
fi

source "$REPO_ROOT/.venv_objx/bin/activate"

if [[ "$RESET_TMP" == "1" ]]; then
  rm -rf "$TMP_VOX_ROOT"
fi

mkdir -p "$TMP_VOX_ROOT/scenes" "$TMP_VOX_ROOT/files"
mkdir -p "$SCRATCH_ROOT/files/gs_annotations"
mkdir -p "$CACHE_ROOT/torch/hub" "$CACHE_ROOT/xdg" "$CACHE_ROOT/matplotlib"

export OBJECTX_CACHE_ROOT="$CACHE_ROOT"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export OBJECTX_DINOV2_HUB_DIR="$TORCH_HOME/hub/facebookresearch_dinov2_main"

cd "$REPO_ROOT"
python -u scripts/voxel_annotations/run_scanwise_voxelise_tmp.py \
  --repo-root "$REPO_ROOT" \
  --scratch-root "$SCRATCH_ROOT" \
  --tmp-root "$TMP_VOX_ROOT" \
  --config "$REPO_ROOT/preprocessing/voxel_anno/voxel_anno.yaml" \
  --split "$SPLIT" \
  --max-scans "$MAX_SCANS"
