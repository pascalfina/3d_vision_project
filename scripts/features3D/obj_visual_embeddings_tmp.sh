#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLSG_DEP_SPACE="${VLSG_DEP_SPACE:-$REPO_ROOT/dependencies/VLSG}"
USER_ROOT="${OBJECTX_USER_ROOT:-/work/scratch/${USER:-$(id -un)}}"
SCRATCH_ROOT="${DATA_ROOT_DIR:-${Data_ROOT_DIR:-${OBJECTX_BASELINE_ROOT:-$USER_ROOT/objectx-data-baseline}}}"
TMP_FEAT3D_ROOT="${TMP_FEAT3D_ROOT:-/tmp/${USER}-objectx-feat3d}"
RESET_TMP="${RESET_TMP:-0}"
SPLIT="${SPLIT:-train}"
MAX_SCANS="${MAX_SCANS:-0}"
SCENE_ID="${SCENE_ID:-}"
OBJECTX_SCENE_SOURCE_DIRNAME="${OBJECTX_SCENE_SOURCE_DIRNAME:-scenes}"
CACHE_ROOT="${OBJECTX_CACHE_ROOT:-$USER_ROOT/objectx-cache}"
source "$REPO_ROOT/scripts/activate_objectx_env.sh"
export VLSG_SPACE="$VLSG_DEP_SPACE"

if [[ "$RESET_TMP" == "1" ]]; then
  rm -rf "$TMP_FEAT3D_ROOT"
fi

if [[ "${OBJECTX_FEATURES3D_FORCE_LOCAL_OUTPUT:-0}" == "1" && -L "$SCRATCH_ROOT/files/Features3D" ]]; then
  rm "$SCRATCH_ROOT/files/Features3D"
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
  --scene-source-dirname "$OBJECTX_SCENE_SOURCE_DIRNAME" \
  ${SCENE_ID:+--scene-id "$SCENE_ID"} \
  --max-scans "$MAX_SCANS" \
  ${OBJECTX_FEATURES3D_OVERRIDE:+--override}
