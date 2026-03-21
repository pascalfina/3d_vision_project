#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH_ROOT="${DATA_ROOT_DIR:-/work/scratch/pafina/objectx-data-baseline}"
TMP_GS_ROOT="${TMP_GS_ROOT:-/tmp/${USER}-objectx-gs}"
RESET_TMP="${RESET_TMP:-0}"
SPLIT="${SPLIT:-train}"
MAX_SCANS="${MAX_SCANS:-0}"

source "$REPO_ROOT/scripts/activate_objectx_env.sh"

if [[ "$RESET_TMP" == "1" ]]; then
  rm -rf "$TMP_GS_ROOT"
fi

mkdir -p "$TMP_GS_ROOT/scenes" "$TMP_GS_ROOT/files" "$TMP_GS_ROOT/files/gs_annotations"

cd "$REPO_ROOT"
python -u scripts/gs_annotations/run_scanwise_gaussians_tmp.py \
  --repo-root "$REPO_ROOT" \
  --scratch-root "$SCRATCH_ROOT" \
  --tmp-root "$TMP_GS_ROOT" \
  --config "$REPO_ROOT/preprocessing/gs_anno/gs_anno.yaml" \
  --split "$SPLIT" \
  --max-scans "$MAX_SCANS"
