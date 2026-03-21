#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLSG_SPACE="${VLSG_SPACE:-$REPO_ROOT/dependencies/VLSG}"
SCRATCH_ROOT="${Data_ROOT_DIR:-/work/scratch/pafina/objectx-data-baseline}"
TMP_FEAT3D_ROOT="${TMP_FEAT3D_ROOT:-/tmp/${USER}-objectx-feat3d}"
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
  rm -rf "$TMP_FEAT3D_ROOT"
fi

mkdir -p "$TMP_FEAT3D_ROOT/scenes" "$TMP_FEAT3D_ROOT/files/orig"
mkdir -p "$SCRATCH_ROOT/files/Features3D/obj_dinov2_top10_l3"
mkdir -p "$CACHE_ROOT/torch/hub" "$CACHE_ROOT/xdg" "$CACHE_ROOT/matplotlib"

export VLSG_SPACE
export OBJECTX_CACHE_ROOT="$CACHE_ROOT"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export OBJECTX_DINOV2_HUB_DIR="$TORCH_HOME/hub/facebookresearch_dinov2_main"

cd "$REPO_ROOT"
python -u scripts/features3D/run_scanwise_obj_visual_embeddings_tmp.py \
  --repo-root "$REPO_ROOT" \
  --vlsG-space "$VLSG_SPACE" \
  --scratch-root "$SCRATCH_ROOT" \
  --tmp-root "$TMP_FEAT3D_ROOT" \
  --config "$VLSG_SPACE/preprocessing/sg_features/obj_visual_embeddings/Dinov2/obj_visual_embeddings.yaml" \
  --split "$SPLIT" \
  --max-scans "$MAX_SCANS"
