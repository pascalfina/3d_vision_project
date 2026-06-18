#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRATCH_ROOT="${DATA_ROOT_DIR:-/work/scratch/pafina/objectx-data-baseline}"
TMP_VOX_ROOT="${TMP_VOX_ROOT:-/tmp/${USER}-objectx-voxelise}"
RESET_TMP="${RESET_TMP:-0}"
SPLIT="${SPLIT:-train}"
MAX_SCANS="${MAX_SCANS:-0}"
CACHE_ROOT="${OBJECTX_CACHE_ROOT:-/work/scratch/pafina/objectx-cache}"
source "$REPO_ROOT/scripts/activate_objectx_env.sh"

if [[ "$RESET_TMP" == "1" ]]; then
  rm -rf "$TMP_VOX_ROOT"
fi

# Older runs left /work/scratch/.../objectx-cache as a symlink into a deleted
# cache tree. Replace that broken link with a real directory so cache setup and
# downstream torch/matplotlib caches work again.
if [[ -L "$CACHE_ROOT" && ! -e "$CACHE_ROOT" ]]; then
  rm -f "$CACHE_ROOT"
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
VOXELISE_EXTRA_ARGS=()
if [[ "${OBJECTX_VOXEL_OVERRIDE:-0}" == "1" ]]; then
  VOXELISE_EXTRA_ARGS+=(--override)
fi
python -u scripts/voxel_annotations/pipeline/run_scanwise_voxelise_tmp.py \
  --repo-root "$REPO_ROOT" \
  --scratch-root "$SCRATCH_ROOT" \
  --tmp-root "$TMP_VOX_ROOT" \
  --config "$REPO_ROOT/preprocessing/voxel_anno/voxel_anno.yaml" \
  --split "$SPLIT" \
  --max-scans "$MAX_SCANS" \
  "${VOXELISE_EXTRA_ARGS[@]}"
