#!/usr/bin/env bash
set -euo pipefail

# Runs a direct Pi3X geometry benchmark over a fixed 15-scene set.
#
# Per scene:
#   1. must3r
#   2. pi3x
#   3. geometry-eval on the Pi3X sequence
#   4. cleanup generated reconstruction artifacts, keeping evaluation outputs
#
# Useful env knobs:
#   GEOMETRY_EVAL_GROUP=pi3x_sequence_15   output group under evaluation/outputs/geometry
#   PROFILE_LIST=path/to/profiles.txt       optional newline-separated profile list
#   FORCE=1                                rerun even if metrics.json already exists
#   KEEP_ARTIFACTS=1                       keep generated Pi3X/MUSt3R roots after successful scenes
#   KEEP_FAILED_ARTIFACTS=1                keep artifacts for failed scenes
#   CONTINUE_ON_ERROR=0                    stop at first failed scene
#   DRY_RUN=1                              print profile commands and cleanup targets only

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"
RUN_PROFILE="$REPO_ROOT/scripts/workflows/run_scene_profile.sh"
export OBJECTX_REPO_ROOT="$REPO_ROOT"
if [[ -f "$REPO_ROOT/configs/workflows/local_paths.env" ]]; then
  # Keep path expansion identical to run_scene_profile.py and the manual commands.
  # shellcheck source=/dev/null
  source "$REPO_ROOT/configs/workflows/local_paths.env"
fi

GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-pi3x_sequence_15}"
FORCE="${FORCE:-0}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
KEEP_FAILED_ARTIFACTS="${KEEP_FAILED_ARTIFACTS:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_PROFILES=(
  scene_5341b7e3_sam2_pi3x
  scene_4d3d829e_sam2_pi3x
  scene_4d3d82b0_sam2_pi3x
  scene_5630cfcf_sam2_pi3x
  scene_c92fb5b5_sam2_pi3x
  scene_dbeb4cee_sam2_pi3x
  scene_0cac766c_sam2_pi3x
  scene_75c25973_sam2_pi3x
  scene_c6707938_sam2_pi3x
  scene_13124cad_sam2_pi3x
  scene_5341b7db_sam2_pi3x
  scene_5630cfcd_sam2_pi3x
  scene_0cac7574_sam2_pi3x
  scene_8eabc45f_sam2_pi3x
  scene_0f2f271b_sam2_pi3x
)

if [[ -n "${PROFILE_LIST:-}" ]]; then
  mapfile -t PROFILES < <(grep -Ev '^\s*(#|$)' "$PROFILE_LIST")
else
  PROFILES=("${DEFAULT_PROFILES[@]}")
fi

profile_json_path() {
  local profile="$1"
  printf '%s/configs/workflows/scene_profiles/%s.json\n' "$REPO_ROOT" "$profile"
}

profile_info() {
  local profile="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$profile" <<'PY'
import json
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
profile = sys.argv[2]
path = repo_root / "configs" / "workflows" / "scene_profiles" / f"{profile}.json"
data = json.loads(path.read_text())

def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value

data = expand(data)
roots = data.get("roots", {})
must3r = data.get("must3r", {})
geom = data.get("geometry_eval", {})

print(data["scene_id"])
print(geom.get("method_name") or data.get("name") or profile)
print(roots.get("reconstruction", ""))
print(must3r.get("output_root", ""))
print(roots.get("pred_ready", ""))
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
  local scene_id method_name reconstruction_root must3r_root pred_ready_root
  mapfile -t info < <(profile_info "$profile")
  scene_id="${info[0]}"
  method_name="${info[1]}"
  reconstruction_root="${info[2]}"
  must3r_root="${info[3]}"
  pred_ready_root="${info[4]}"

  safe_rm_generated_root "$reconstruction_root"
  safe_rm_generated_root "$must3r_root"
  safe_rm_generated_root "$pred_ready_root"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[cleanup:dry-run] would remove matching logs for $profile"
  else
    rm -rf -- \
      "$REPO_ROOT/logs"/*"${profile#scene_}"*.log \
      "$REPO_ROOT/logs"/*"$method_name"*.log \
      2>/dev/null || true
  fi
}

metrics_path_for_profile() {
  local profile="$1"
  local scene_id method_name
  mapfile -t info < <(profile_info "$profile")
  scene_id="${info[0]}"
  method_name="${info[1]}"
  printf '%s/evaluation/outputs/geometry/%s/%s/%s/metrics.json\n' \
    "$REPO_ROOT" "$GEOMETRY_EVAL_GROUP" "$method_name" "$scene_id"
}

run_action() {
  local profile="$1"
  local action="$2"
  echo "========== [$profile] $action =========="
  if [[ "$DRY_RUN" == "1" ]]; then
    bash "$RUN_PROFILE" "$profile" "$action" --dry-run
  else
    bash "$RUN_PROFILE" "$profile" "$action"
  fi
}

echo "[benchmark] group=$GEOMETRY_EVAL_GROUP"
echo "[benchmark] profiles=${#PROFILES[@]}"
echo "[benchmark] force=$FORCE keep_artifacts=$KEEP_ARTIFACTS continue_on_error=$CONTINUE_ON_ERROR dry_run=$DRY_RUN"

failed=()
completed=0
skipped=0

for profile in "${PROFILES[@]}"; do
  metrics_path="$(metrics_path_for_profile "$profile")"
  if [[ "$FORCE" != "1" && -f "$metrics_path" ]]; then
    echo "========== [$profile] skip existing metrics =========="
    echo "$metrics_path"
    ((skipped += 1))
    continue
  fi

  echo "========== [$profile] start =========="
  cleanup_profile_artifacts "$profile"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[cleanup:dry-run] would remove $(dirname "$metrics_path")"
  else
    rm -rf -- "$(dirname "$metrics_path")"
  fi

  if (
    run_action "$profile" must3r
    run_action "$profile" pi3x
    GEOMETRY_EVAL_GROUP="$GEOMETRY_EVAL_GROUP" run_action "$profile" geometry-eval
  ); then
    ((completed += 1))
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
  echo "[benchmark:dry-run] would summarize evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/**/metrics.json"
else
  "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/summarize_geometry_runs.py" \
    --metrics-glob "evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/**/metrics.json" \
    --out-dir "evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP"
fi

echo "[benchmark] completed=$completed skipped=$skipped failed=${#failed[@]}"
echo "[benchmark] summary=$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/geometry_summary.md"

if (( ${#failed[@]} > 0 )); then
  printf '[benchmark] failed profiles:\n' >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
