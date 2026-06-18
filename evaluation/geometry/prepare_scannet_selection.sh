#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SELECTION_FILE="${SELECTION_FILE:-$REPO_ROOT/evaluation/outputs/geometry/scannet_objectx_final_100_pi3x_samobject/selected_100.tsv}"
SCANNET_ROOT="${SCANNET_ROOT:-/work/courses/3dv/team35/pafina/scannet_under300_data}"
SCANNET_DOWNLOAD_TYPES="${SCANNET_DOWNLOAD_TYPES:-.txt _vh_clean_2.ply .sens}"
SCANNET_DOWNLOAD_BACKEND="${SCANNET_DOWNLOAD_BACKEND:-auto}"
SCANNET_DELETE_SENS_AFTER_EXPORT="${SCANNET_DELETE_SENS_AFTER_EXPORT:-1}"
SCANNET_EXPORT_MAX_FRAMES="${SCANNET_EXPORT_MAX_FRAMES:-300}"
SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-1}"
SCANNET_AUTO_FRAME_SKIP_FOR_MAX="${SCANNET_AUTO_FRAME_SKIP_FOR_MAX:-1}"
SCANNET_DIRECT_DOWNLOAD="${SCANNET_DIRECT_DOWNLOAD:-1}"
SKIP_PREPARED="${SKIP_PREPARED:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
START_AT="${START_AT:-}"
LIMIT="${LIMIT:-0}"

is_prepared() {
  local scene_id="$1"
  local scan_dir="$SCANNET_ROOT/scans/$scene_id"
  [[ -f "$scan_dir/${scene_id}_vh_clean_2.ply" ]] || return 1
  [[ -d "$scan_dir/data/color" ]] || return 1
  [[ -d "$scan_dir/data/depth" ]] || return 1
  [[ -d "$scan_dir/data/pose" ]] || return 1
  find "$scan_dir/data/color" -maxdepth 1 -type f | grep -q .
}

echo "[prepare-selection] selection=$SELECTION_FILE"
echo "[prepare-selection] root=$SCANNET_ROOT"
echo "[prepare-selection] types=$SCANNET_DOWNLOAD_TYPES backend=$SCANNET_DOWNLOAD_BACKEND"
echo "[prepare-selection] max_frames=$SCANNET_EXPORT_MAX_FRAMES delete_sens=$SCANNET_DELETE_SENS_AFTER_EXPORT"
echo "[prepare-selection] start_at=${START_AT:-<first>} limit=$LIMIT skip_prepared=$SKIP_PREPARED continue=$CONTINUE_ON_ERROR"

if [[ ! -f "$SELECTION_FILE" ]]; then
  echo "[prepare-selection] missing selection file: $SELECTION_FILE" >&2
  exit 2
fi

count=0
seen_start=0
tail -n +2 "$SELECTION_FILE" | while IFS=$'\t' read -r bucket bucket_rank score score_scope scene_id profile source_metrics; do
  [[ -n "${scene_id:-}" ]] || continue
  if [[ -n "$START_AT" && "$seen_start" != "1" ]]; then
    if [[ "$scene_id" == "$START_AT" ]]; then
      seen_start=1
    else
      continue
    fi
  fi
  if [[ "$LIMIT" != "0" && "$count" -ge "$LIMIT" ]]; then
    break
  fi
  count=$((count + 1))
  echo "========== [prepare $count] $bucket scene=$scene_id profile=$profile =========="
  if [[ "$SKIP_PREPARED" == "1" ]] && is_prepared "$scene_id"; then
    echo "[prepare-selection] skip prepared scene=$scene_id"
    continue
  fi
  if ! SCAN_ID="$scene_id" \
    SCANNET_ROOT="$SCANNET_ROOT" \
    SCANNET_DOWNLOAD_TYPES="$SCANNET_DOWNLOAD_TYPES" \
    SCANNET_DOWNLOAD_BACKEND="$SCANNET_DOWNLOAD_BACKEND" \
    SCANNET_DELETE_SENS_AFTER_EXPORT="$SCANNET_DELETE_SENS_AFTER_EXPORT" \
    SCANNET_MAX_FRAMES="$SCANNET_EXPORT_MAX_FRAMES" \
    SCANNET_FRAME_SKIP="$SCANNET_FRAME_SKIP" \
    SCANNET_AUTO_FRAME_SKIP_FOR_MAX="$SCANNET_AUTO_FRAME_SKIP_FOR_MAX" \
    SCANNET_DIRECT_DOWNLOAD="$SCANNET_DIRECT_DOWNLOAD" \
    bash "$REPO_ROOT/evaluation/geometry/prepare_scannet_scene.sh"; then
    echo "[prepare-selection] failed scene=$scene_id" >&2
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
  fi
done

echo "[prepare-selection] done"
