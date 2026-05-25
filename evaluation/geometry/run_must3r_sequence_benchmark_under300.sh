#!/usr/bin/env bash
set -euo pipefail

# Direct MUSt3R geometry benchmark over the same under-300-frame profile list
# used by the Pi3X geometry benchmark.
#
# Per scene:
#   1. run the profile's MUSt3R action
#   2. evaluate scenes_sam2_must3r/<scene>/sequence against the GT mesh
#   3. remove generated MUSt3R artifacts, keeping evaluation outputs
#
# Useful env knobs:
#   GEOMETRY_EVAL_GROUP=must3r_sequence_under300  output group
#   PROFILE_LIST=...                              newline-separated profile list
#   FORCE=1                                       rerun existing metrics
#   KEEP_ARTIFACTS=1                              keep generated MUSt3R roots
#   KEEP_FAILED_ARTIFACTS=1                       keep artifacts for failed scenes
#   CONTINUE_ON_ERROR=0                           stop at first failure
#   START_AT=1                                    1-based profile index
#   LIMIT=0                                       max profiles to process, 0 = all
#   DRY_RUN=1                                     print commands only

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"
RUN_PROFILE="$REPO_ROOT/scripts/workflows/run_scene_profile.sh"

export OBJECTX_REPO_ROOT="$REPO_ROOT"
if [[ -f "$REPO_ROOT/configs/workflows/local_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/configs/workflows/local_paths.env"
fi

PROFILE_LIST="${PROFILE_LIST:-$REPO_ROOT/configs/workflows/scene_profiles/pi3x_under300_profiles.txt}"
GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-must3r_sequence_under300}"
FORCE="${FORCE:-0}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
KEEP_FAILED_ARTIFACTS="${KEEP_FAILED_ARTIFACTS:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
START_AT="${START_AT:-1}"
LIMIT="${LIMIT:-0}"
WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"

mapfile -t PROFILES < <(grep -Ev '^\s*(#|$)' "$PROFILE_LIST")

profile_info() {
  local profile="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$profile" <<'PY'
import json
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
profile_name = sys.argv[2]
profile_path = repo_root / "configs" / "workflows" / "scene_profiles" / f"{profile_name}.json"
profile = json.loads(profile_path.read_text())

def expand(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value

profile = expand(profile)
roots = profile.get("roots", {})
must3r = profile.get("must3r", {})
scene_id = profile["scene_id"]
profile_method = profile.get("geometry_eval", {}).get("method_name") or profile.get("name") or profile_name
method_name = profile_method.replace("_sam2_pi3x", "_must3r")
if method_name == profile_method:
    method_name = f"{profile_method}_must3r"

print(scene_id)
print(method_name)
print(roots.get("baseline", ""))
print(must3r.get("output_root", ""))
print(must3r.get("scenes_dirname", "scenes_sam2_must3r"))
print(must3r.get("log", ""))
PY
}

safe_rm_generated_root() {
  local path="$1"
  [[ -n "$path" ]] || return 0
  case "$path" in
    /work/scratch/*/objectx-data-fullscene-*|/work/courses/3dv/team35/*/objectx-heavy/data/objectx-data-fullscene-*)
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
  local scene_id method_name baseline_root must3r_root scenes_dirname must3r_log
  mapfile -t info < <(profile_info "$profile")
  scene_id="${info[0]}"
  method_name="${info[1]}"
  baseline_root="${info[2]}"
  must3r_root="${info[3]}"
  scenes_dirname="${info[4]}"
  must3r_log="${info[5]}"

  safe_rm_generated_root "$must3r_root"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[cleanup:dry-run] would remove logs for $profile"
  else
    rm -f -- "$must3r_log" 2>/dev/null || true
    rm -f -- "$REPO_ROOT/logs"/*"${profile#scene_}"*.log 2>/dev/null || true
  fi
}

metrics_path_for_profile() {
  local profile="$1"
  local scene_id method_name baseline_root must3r_root scenes_dirname must3r_log
  mapfile -t info < <(profile_info "$profile")
  scene_id="${info[0]}"
  method_name="${info[1]}"
  printf '%s/evaluation/outputs/geometry/%s/%s/%s/metrics.json\n' \
    "$REPO_ROOT" "$GEOMETRY_EVAL_GROUP" "$method_name" "$scene_id"
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

run_eval() {
  local profile="$1"
  local scene_id method_name baseline_root must3r_root scenes_dirname must3r_log sequence_dir metrics_path
  mapfile -t info < <(profile_info "$profile")
  scene_id="${info[0]}"
  method_name="${info[1]}"
  baseline_root="${info[2]}"
  must3r_root="${info[3]}"
  scenes_dirname="${info[4]}"
  must3r_log="${info[5]}"
  sequence_dir="$must3r_root/$scenes_dirname/$scene_id/sequence"
  metrics_path="$(metrics_path_for_profile "$profile")"

  echo "========== [$profile] geometry-eval-must3r =========="
  echo "[must3r-eval] sequence=$sequence_dir"
  echo "[must3r-eval] metrics=$metrics_path"
  if [[ "$DRY_RUN" == "1" ]]; then
    cat <<EOF
SCAN_ID=$scene_id METHOD_NAME=$method_name PRED_INPUT_MODE=sequence \\
PRED_ROOT=$must3r_root PRED_SEQUENCE_DIR=$sequence_dir \\
BASELINE_ROOT=$baseline_root GEOMETRY_EVAL_GROUP=$GEOMETRY_EVAL_GROUP \\
WRITE_DEBUG_HTML=$WRITE_DEBUG_HTML bash evaluation/geometry/run_pi3x_geometry_eval.sh
EOF
    return 0
  fi

  SCAN_ID="$scene_id" \
  METHOD_NAME="$method_name" \
  PRED_INPUT_MODE=sequence \
  PRED_ROOT="$must3r_root" \
  PRED_SEQUENCE_DIR="$sequence_dir" \
  BASELINE_ROOT="$baseline_root" \
  GEOMETRY_EVAL_GROUP="$GEOMETRY_EVAL_GROUP" \
  WRITE_DEBUG_HTML="$WRITE_DEBUG_HTML" \
  bash "$REPO_ROOT/evaluation/geometry/run_pi3x_geometry_eval.sh"
}

echo "[must3r-benchmark] group=$GEOMETRY_EVAL_GROUP"
echo "[must3r-benchmark] profiles=${#PROFILES[@]} list=$PROFILE_LIST"
echo "[must3r-benchmark] force=$FORCE keep_artifacts=$KEEP_ARTIFACTS continue_on_error=$CONTINUE_ON_ERROR dry_run=$DRY_RUN start_at=$START_AT limit=$LIMIT"

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

  metrics_path="$(metrics_path_for_profile "$profile")"
  if [[ "$FORCE" != "1" && -f "$metrics_path" ]]; then
    echo "========== [$seen/${#PROFILES[@]}] [$profile] skip existing metrics =========="
    echo "$metrics_path"
    skipped=$((skipped + 1))
    continue
  fi

  echo "========== [$seen/${#PROFILES[@]}] [$profile] start =========="
  cleanup_profile_artifacts "$profile"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[cleanup:dry-run] would remove $(dirname "$metrics_path")"
  else
    rm -rf -- "$(dirname "$metrics_path")"
  fi

  if (
    run_profile_action "$profile" must3r
    run_eval "$profile"
    test "$DRY_RUN" == "1" || test -f "$metrics_path"
  ); then
    completed=$((completed + 1))
    if [[ "$KEEP_ARTIFACTS" != "1" ]]; then
      cleanup_profile_artifacts "$profile"
    fi
    echo "========== [$profile] done =========="
  else
    failed+=("$profile")
    echo "========== [$profile] FAILED ==========" >&2
    if [[ "$KEEP_FAILED_ARTIFACTS" != "1" ]]; then
      cleanup_profile_artifacts "$profile"
    fi
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      break
    fi
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[must3r-benchmark:dry-run] would summarize evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/**/metrics.json"
else
  "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/summarize_geometry_runs.py" \
    --metrics-glob "evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/**/metrics.json" \
    --out-dir "evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP"
fi

echo "[must3r-benchmark] completed=$completed skipped=$skipped failed=${#failed[@]}"
echo "[must3r-benchmark] summary=$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/geometry_summary.md"

if (( ${#failed[@]} > 0 )); then
  printf '[must3r-benchmark] failed profiles:\n' >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
