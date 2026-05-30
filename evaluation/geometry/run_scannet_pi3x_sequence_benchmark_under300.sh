#!/usr/bin/env bash
set -euo pipefail

# ScanNet geometry benchmark matching the 3RScan Pi3X sequence benchmark.
#
# Per scene:
#   1. select/download/export ScanNet scenes with <= SCANNET_SELECTION_MAX_FRAMES
#   2. run MUSt3R for poses
#   3. run Pi3X with MUSt3R poses as prior
#   4. evaluate Pi3X sequence geometry against ScanNet GT mesh
#   5. clean generated Object-X artifacts, keeping ScanNet data + eval outputs

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"
RUN_PROFILE="$REPO_ROOT/scripts/workflows/run_scene_profile.sh"

export OBJECTX_REPO_ROOT="$REPO_ROOT"
if [[ -f "$REPO_ROOT/configs/workflows/local_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/configs/workflows/local_paths.env"
fi

SCANNET_ROOT="${SCANNET_ROOT:-/work/courses/3dv/team35/pafina/scannet_under300_data}"
SCANNET_WORK_ROOT="${SCANNET_WORK_ROOT:-/work/courses/3dv/team35/pafina}"
SCANNET_TARGET_SCENES="${SCANNET_TARGET_SCENES:-300}"
SCANNET_SELECTION_MAX_FRAMES="${SCANNET_SELECTION_MAX_FRAMES:-300}"
SCANNET_SELECTION_MODE="${SCANNET_SELECTION_MODE:-cap}"
SCANNET_SELECTION_ORDER="${SCANNET_SELECTION_ORDER:-shortest}"
SCANNET_EXPORT_MAX_FRAMES="${SCANNET_EXPORT_MAX_FRAMES:-$SCANNET_SELECTION_MAX_FRAMES}"
SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-1}"
SCANNET_AUTO_FRAME_SKIP_FOR_MAX="${SCANNET_AUTO_FRAME_SKIP_FOR_MAX:-1}"
SCANNET_SCENE_TABLE="${SCANNET_SCENE_TABLE:-$REPO_ROOT/configs/workflows/scene_profiles/scannet_under300_scenes.tsv}"
PROFILE_LIST="${PROFILE_LIST:-$REPO_ROOT/configs/workflows/scene_profiles/scannet_under300_profiles.txt}"
GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-scannet_pi3x_sequence_under300}"
SCANNET_REUSE_SELECTION="${SCANNET_REUSE_SELECTION:-1}"
SCANNET_PRUNE_STALE_PROFILES="${SCANNET_PRUNE_STALE_PROFILES:-1}"
FORCE="${FORCE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
KEEP_FAILED_ARTIFACTS="${KEEP_FAILED_ARTIFACTS:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
START_AT="${START_AT:-1}"
LIMIT="${LIMIT:-0}"
WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"
SCANNET_PREPARE="${SCANNET_PREPARE:-1}"
SCANNET_DELETE_SENS_AFTER_EXPORT="${SCANNET_DELETE_SENS_AFTER_EXPORT:-1}"
SCANNET_DOWNLOAD_TYPES="${SCANNET_DOWNLOAD_TYPES:-.txt _vh_clean_2.ply .sens}"
SCANNET_DIRECT_DOWNLOAD="${SCANNET_DIRECT_DOWNLOAD:-1}"
SCANNET_PURGE_PREPARED_AFTER_SCENE="${SCANNET_PURGE_PREPARED_AFTER_SCENE:-0}"
SCANNET_PURGE_PREPARED_AFTER_FAILED="${SCANNET_PURGE_PREPARED_AFTER_FAILED:-$SCANNET_PURGE_PREPARED_AFTER_SCENE}"
SCANNET_PURGE_PREPARED_ON_SKIP="${SCANNET_PURGE_PREPARED_ON_SKIP:-0}"

export SCANNET_ROOT
export SCANNET_WORK_ROOT
export GEOMETRY_EVAL_GROUP

echo "[scannet-benchmark] root=$SCANNET_ROOT"
echo "[scannet-benchmark] work_root=$SCANNET_WORK_ROOT"
echo "[scannet-benchmark] target=$SCANNET_TARGET_SCENES max_frames=$SCANNET_SELECTION_MAX_FRAMES mode=$SCANNET_SELECTION_MODE order=$SCANNET_SELECTION_ORDER group=$GEOMETRY_EVAL_GROUP"
echo "[scannet-benchmark] force=$FORCE skip_completed=$SKIP_COMPLETED cleanup=$CLEANUP_AFTER_SCENE continue=$CONTINUE_ON_ERROR dry_run=$DRY_RUN reuse_selection=$SCANNET_REUSE_SELECTION prune_profiles=$SCANNET_PRUNE_STALE_PROFILES"
echo "[scannet-benchmark] purge_prepared_after_scene=$SCANNET_PURGE_PREPARED_AFTER_SCENE purge_failed=$SCANNET_PURGE_PREPARED_AFTER_FAILED purge_on_skip=$SCANNET_PURGE_PREPARED_ON_SKIP"

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "$SCANNET_REUSE_SELECTION" == "1" && -f "$SCANNET_SCENE_TABLE" ]]; then
    echo "[scannet-select] reusing existing scene table: $SCANNET_SCENE_TABLE"
  else
    "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/select_scannet_under300.py" \
      --scannet-root "$SCANNET_ROOT" \
      --target "$SCANNET_TARGET_SCENES" \
      --max-frames "$SCANNET_SELECTION_MAX_FRAMES" \
      --mode "$SCANNET_SELECTION_MODE" \
      --order "$SCANNET_SELECTION_ORDER" \
      --out "$SCANNET_SCENE_TABLE"
  fi
  generate_args=(
    "$REPO_ROOT/evaluation/geometry/generate_scannet_pi3x_profiles.py"
    --scene-table "$SCANNET_SCENE_TABLE" \
    --manifest "$PROFILE_LIST" \
    --max-scenes "$SCANNET_TARGET_SCENES"
  )
  if [[ "$SCANNET_PRUNE_STALE_PROFILES" == "1" ]]; then
    generate_args+=(--prune-stale)
  fi
  "$PYTHON_BIN" "${generate_args[@]}"
else
  echo "[scannet-benchmark:dry-run] would select scenes and generate profiles"
fi

if [[ ! -f "$PROFILE_LIST" ]]; then
  echo "[scannet-benchmark] missing profile list: $PROFILE_LIST" >&2
  exit 2
fi

mapfile -t PROFILES < <(grep -Ev '^\s*(#|$)' "$PROFILE_LIST" | head -n "$SCANNET_TARGET_SCENES")

profile_info() {
  local profile="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$profile" <<'PY'
import json
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
profile_name = sys.argv[2]
path = repo_root / "configs" / "workflows" / "scene_profiles" / f"{profile_name}.json"
data = json.loads(path.read_text())

def expand(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value

data = expand(data)
scene_id = data["scene_id"]
roots = data.get("roots", {})
must3r = data.get("must3r", {})
geom = data.get("geometry_eval", {})
print(scene_id)
print(data.get("name", profile_name))
print(roots.get("baseline", ""))
print(roots.get("reconstruction", ""))
print(must3r.get("output_root", ""))
print(geom.get("method_name", data.get("name", profile_name)))
PY
}

metrics_path_for_profile() {
  local profile="$1"
  local scene_id profile_name baseline_root recon_root must3r_root method_name
  mapfile -t info < <(profile_info "$profile")
  scene_id="${info[0]}"
  method_name="${info[5]}"
  printf '%s/evaluation/outputs/geometry/%s/%s/%s/metrics.json\n' \
    "$REPO_ROOT" "$GEOMETRY_EVAL_GROUP" "$method_name" "$scene_id"
}

scan_is_prepared() {
  local scene_id="$1"
  local seq="$SCANNET_ROOT/scenes/$scene_id/sequence"
  [[ -f "$seq/_info.txt" ]] || return 1
  compgen -G "$seq/frame-*.color.jpg" >/dev/null || return 1
  compgen -G "$seq/frame-*.pose.txt" >/dev/null || return 1
  [[ -f "$SCANNET_ROOT/scenes/$scene_id/${scene_id}_vh_clean_2.ply" || -f "$SCANNET_ROOT/scenes/$scene_id/${scene_id}_vh_clean.ply" ]]
}

prepare_scan() {
  local scene_id="$1"
  if scan_is_prepared "$scene_id"; then
    echo "[scannet-prepare] already prepared: $scene_id"
    return 0
  fi
  echo "[scannet-prepare] preparing: $scene_id"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "SCAN_ID=$scene_id SCANNET_ROOT=$SCANNET_ROOT bash evaluation/geometry/prepare_scannet_scene.sh"
    return 0
  fi
  SCAN_ID="$scene_id" \
  SCANNET_ROOT="$SCANNET_ROOT" \
  SCANNET_MAX_FRAMES="$SCANNET_EXPORT_MAX_FRAMES" \
  SCANNET_FRAME_SKIP="$SCANNET_FRAME_SKIP" \
  SCANNET_AUTO_FRAME_SKIP_FOR_MAX="$SCANNET_AUTO_FRAME_SKIP_FOR_MAX" \
  SCANNET_DOWNLOAD_TYPES="$SCANNET_DOWNLOAD_TYPES" \
  SCANNET_DIRECT_DOWNLOAD="$SCANNET_DIRECT_DOWNLOAD" \
  SCANNET_DELETE_SENS_AFTER_EXPORT="$SCANNET_DELETE_SENS_AFTER_EXPORT" \
  bash "$REPO_ROOT/evaluation/geometry/prepare_scannet_scene.sh"
}

link_sequence_inputs_for_eval() {
  local scene_id="$1"
  local recon_root="$2"
  local src_seq="$SCANNET_ROOT/scenes/$scene_id/sequence"
  local dst_seq="$recon_root/scenes_sam2_pi3x/$scene_id/sequence"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[sequence-links:dry-run] would link color/_info from $src_seq -> $dst_seq"
    return 0
  fi
  if [[ ! -d "$src_seq" || ! -d "$dst_seq" ]]; then
    echo "[sequence-links] missing source or destination sequence: src=$src_seq dst=$dst_seq" >&2
    return 1
  fi
  if [[ -f "$src_seq/_info.txt" && ! -e "$dst_seq/_info.txt" ]]; then
    ln -sfn "$(realpath --relative-to="$dst_seq" "$src_seq/_info.txt")" "$dst_seq/_info.txt"
  fi
  local linked=0
  shopt -s nullglob
  for src in "$src_seq"/frame-*.color.jpg; do
    local dst="$dst_seq/$(basename "$src")"
    if [[ ! -e "$dst" ]]; then
      ln -s "$(realpath --relative-to="$dst_seq" "$src")" "$dst"
      linked=$((linked + 1))
    fi
  done
  shopt -u nullglob
  echo "[sequence-links] linked color frames for eval: scene=$scene_id linked=$linked"
}

safe_rm_generated_root() {
  local path="$1"
  [[ -n "$path" ]] || return 0
  case "$path" in
    /work/scratch/*/objectx-data-scannet-*|/work/scratch/*/objectx-data-fullscene-*|/work/courses/3dv/team35/pafina/objectx-data-scannet-*)
      if [[ -e "$path" || -L "$path" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
          echo "[cleanup:dry-run] would remove $path"
        else
          rm -rf -- "$path"
          echo "[cleanup] removed $path"
        fi
      fi
      ;;
    *)
      echo "[cleanup] skip suspicious path: $path" >&2
      ;;
  esac
}

cleanup_profile_artifacts() {
  local profile="$1"
  local scene_id profile_name baseline_root recon_root must3r_root method_name
  mapfile -t info < <(profile_info "$profile")
  recon_root="${info[3]}"
  must3r_root="${info[4]}"
  safe_rm_generated_root "$recon_root"
  safe_rm_generated_root "$must3r_root"
}

purge_prepared_scan() {
  local scene_id="$1"
  if [[ ! "$scene_id" =~ ^scene[0-9]{4}_[0-9]{2}$ ]]; then
    echo "[scannet-purge] skip suspicious scene id: $scene_id" >&2
    return 0
  fi
  case "$SCANNET_ROOT" in
    /work/courses/3dv/team35/pafina/scannet_under300_data|/work/scratch/pafina/scannet_under300_data|/work/scratch/pafina/scannet_under300_data/*|/work/courses/3dv/team35/pafina/scannet_under300_data/*)
      ;;
    *)
      echo "[scannet-purge] skip suspicious SCANNET_ROOT: $SCANNET_ROOT" >&2
      return 0
      ;;
  esac
  local scan_dir="$SCANNET_ROOT/scans/$scene_id"
  local scene_link="$SCANNET_ROOT/scenes/$scene_id"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[scannet-purge:dry-run] would remove $scan_dir and $scene_link"
    return 0
  fi
  rm -rf -- "$scan_dir"
  rm -f -- "$scene_link"
  echo "[scannet-purge] removed prepared ScanNet data for $scene_id"
}

run_profile_action() {
  local profile="$1"
  local action="$2"
  echo "========== [$profile] $action =========="
  if [[ "$DRY_RUN" == "1" ]]; then
    bash "$RUN_PROFILE" "$profile" "$action" --dry-run
  else
    bash "$RUN_PROFILE" "$profile" "$action"
  fi
}

summarize_runs() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[scannet-benchmark:dry-run] would summarize $GEOMETRY_EVAL_GROUP"
    return 0
  fi
  "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/summarize_geometry_runs.py" \
    --metrics-glob "evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/**/metrics.json" \
    --out-dir "evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP"
}

failed=()
completed=0
skipped=0
seen=0
processed=0

for profile in "${PROFILES[@]}"; do
  seen=$((seen + 1))
  if (( seen < START_AT )); then
    continue
  fi
  if (( LIMIT > 0 && processed >= LIMIT )); then
    break
  fi
  processed=$((processed + 1))

  mapfile -t info < <(profile_info "$profile")
  scene_id="${info[0]}"
  metrics_path="$(metrics_path_for_profile "$profile")"

  if [[ "$FORCE" != "1" && "$SKIP_COMPLETED" == "1" && -f "$metrics_path" ]]; then
    echo "========== [$seen/${#PROFILES[@]}] [$profile] skip existing metrics =========="
    echo "$metrics_path"
    if [[ "$SCANNET_PURGE_PREPARED_ON_SKIP" == "1" ]]; then
      purge_prepared_scan "$scene_id"
    fi
    skipped=$((skipped + 1))
    continue
  fi

  echo "========== [$seen/${#PROFILES[@]}] scene=$scene_id profile=$profile =========="
  if [[ "$FORCE" == "1" && "$DRY_RUN" != "1" ]]; then
    rm -rf -- "$(dirname "$metrics_path")"
  fi
  cleanup_profile_artifacts "$profile"

  if (
    if [[ "$SCANNET_PREPARE" == "1" ]]; then
      prepare_scan "$scene_id"
    fi
    run_profile_action "$profile" must3r
    run_profile_action "$profile" pi3x
    link_sequence_inputs_for_eval "$scene_id" "${info[3]}"
    WRITE_DEBUG_HTML="$WRITE_DEBUG_HTML" run_profile_action "$profile" geometry-eval
    test "$DRY_RUN" == "1" || test -f "$metrics_path"
  ); then
    completed=$((completed + 1))
    echo "========== [$profile] done =========="
    if [[ "$CLEANUP_AFTER_SCENE" == "1" ]]; then
      cleanup_profile_artifacts "$profile"
    fi
    if [[ "$SCANNET_PURGE_PREPARED_AFTER_SCENE" == "1" ]]; then
      purge_prepared_scan "$scene_id"
    fi
  else
    failed+=("$profile")
    echo "========== [$profile] FAILED ==========" >&2
    if [[ "$KEEP_FAILED_ARTIFACTS" != "1" ]]; then
      cleanup_profile_artifacts "$profile"
    fi
    if [[ "$SCANNET_PURGE_PREPARED_AFTER_FAILED" == "1" ]]; then
      purge_prepared_scan "$scene_id"
    fi
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      break
    fi
  fi
  summarize_runs || true
done

summarize_runs || true

echo "[scannet-benchmark] completed=$completed skipped=$skipped failed=${#failed[@]}"
echo "[scannet-benchmark] summary=$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/geometry_summary.md"

if (( ${#failed[@]} > 0 )); then
  printf '[scannet-benchmark] failed profiles:\n' >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
