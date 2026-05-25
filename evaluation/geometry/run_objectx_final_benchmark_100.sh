#!/usr/bin/env bash
set -euo pipefail

# Full Object-X final-geometry benchmark over a balanced 100-scene subset.
#
# This is intentionally a plain bash runner, not a Slurm/sbatch script.  It
# selects scenes from the existing Pi3X-vs-GT benchmark and then runs:
# must3r -> pi3x -> samobject -> voxelise -> build-pred-ready -> features3d
# -> slat -> u3dgs -> objectx-final-geometry-eval -> cleanup.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"
PROFILE_DIR="${PROFILE_DIR:-$REPO_ROOT/configs/workflows/scene_profiles}"

SOURCE_GROUP="${SOURCE_GROUP:-pi3x_sequence_under300}"
METRICS_GLOB="${METRICS_GLOB:-$REPO_ROOT/evaluation/outputs/geometry/$SOURCE_GROUP/**/metrics.json}"
GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-objectx_final_100}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP}"
SELECTION_FILE="${SELECTION_FILE:-$OUT_ROOT/selected_100.tsv}"
STATUS_FILE="${STATUS_FILE:-$OUT_ROOT/status.tsv}"

TOTAL_SCENES="${TOTAL_SCENES:-100}"
BEST_COUNT="${BEST_COUNT:-50}"
MID_COUNT="${MID_COUNT:-25}"
WORST_COUNT="${WORST_COUNT:-25}"
TARGET_COMPLETE_RUNS="${TARGET_COMPLETE_RUNS:-100}"

DRY_RUN="${DRY_RUN:-0}"
RESAMPLE="${RESAMPLE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
START_AT="${START_AT:-1}"
LIMIT="${LIMIT:-0}"

SAMOBJECT_DATA_ROOT_DEFAULT="${SAMOBJECT_DATA_ROOT_DEFAULT:-/work/scratch/pafina/sam2object}"

ACTIONS=(
  must3r
  pi3x
  samobject
  voxelise
  build-pred-ready
  features3d
  slat
  u3dgs
  objectx-final-geometry-eval
)

mkdir -p "$OUT_ROOT"

select_scenes() {
  "$PYTHON_BIN" - \
    "$METRICS_GLOB" \
    "$PROFILE_DIR" \
    "$SELECTION_FILE" \
    "$TOTAL_SCENES" \
    "$BEST_COUNT" \
    "$MID_COUNT" \
    "$WORST_COUNT" <<'PY'
import csv
import glob
import json
import sys
from pathlib import Path

metrics_glob = sys.argv[1]
profile_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])
total = int(sys.argv[4])
best_count = int(sys.argv[5])
mid_count = int(sys.argv[6])
worst_count = int(sys.argv[7])

if best_count + mid_count + worst_count != total:
    raise SystemExit("BEST_COUNT + MID_COUNT + WORST_COUNT must equal TOTAL_SCENES")

profiles_by_scene = {}
profiles_by_name = {}
for path in profile_dir.glob("*.json"):
    try:
        profile = json.loads(path.read_text())
    except Exception:
        continue
    name = profile.get("name") or path.stem
    scene_id = profile.get("scene_id")
    profiles_by_name[name] = name
    if scene_id:
        profiles_by_scene[scene_id] = name

def preferred_scope(metrics):
    scopes = metrics.get("scopes", {})
    for key in ("visible_gt", "pred_bbox_gt", "full_gt"):
        if key in scopes:
            return key, scopes[key]
    if scopes:
        key = sorted(scopes)[0]
        return key, scopes[key]
    return "none", {}

def score(metrics):
    scope_name, scope = preferred_scope(metrics)
    pred = metrics.get("pred_to_gt", {})
    comp = scope.get("gt_to_pred", {})
    chamfer = scope.get("chamfer_l1_mean")
    if isinstance(chamfer, (int, float)):
        return float(chamfer), scope_name
    pred_mean = pred.get("mean")
    comp_mean = comp.get("mean")
    values = [float(v) for v in (pred_mean, comp_mean) if isinstance(v, (int, float))]
    if not values:
        return float("inf"), scope_name
    return sum(values) / len(values), scope_name

rows_by_scene = {}
for raw_path in glob.glob(metrics_glob, recursive=True):
    path = Path(raw_path)
    try:
        metrics = json.loads(path.read_text())
    except Exception:
        continue
    scene_id = metrics.get("scene_id") or path.parent.name
    method = metrics.get("method_name") or path.parent.parent.name
    profile = method if method in profiles_by_name else profiles_by_scene.get(scene_id)
    if not profile:
        continue
    value, scope_name = score(metrics)
    if not (value < float("inf")):
        continue
    current = rows_by_scene.get(scene_id)
    row = {
        "scene_id": scene_id,
        "profile": profile,
        "score": value,
        "score_scope": scope_name,
        "source_metrics": str(path),
    }
    if current is None or value < current["score"]:
        rows_by_scene[scene_id] = row

rows = sorted(rows_by_scene.values(), key=lambda item: (item["score"], item["scene_id"]))
if len(rows) < total:
    raise SystemExit(f"Need at least {total} scored scenes, found {len(rows)}")

best = rows[:best_count]
worst = rows[-worst_count:] if worst_count else []
middle_pool = rows[best_count : len(rows) - worst_count]
if len(middle_pool) < mid_count:
    raise SystemExit("Not enough middle-pool scenes after best/worst split")
mid_start = max(0, (len(middle_pool) - mid_count) // 2)
middle = middle_pool[mid_start : mid_start + mid_count]

selected = []
for bucket, bucket_rows in (("best", best), ("middle", middle), ("worst", worst)):
    for local_rank, row in enumerate(bucket_rows, start=1):
        selected.append({"bucket": bucket, "bucket_rank": local_rank, **row})

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "bucket",
            "bucket_rank",
            "score",
            "score_scope",
            "scene_id",
            "profile",
            "source_metrics",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(selected)

print(f"[select] wrote {len(selected)} scenes to {out_path}")
print(f"[select] source scenes={len(rows)} best={len(best)} middle={len(middle)} worst={len(worst)}")
PY
}

profile_path_for() {
  local profile="$1"
  if [[ -f "$profile" ]]; then
    printf '%s\n' "$profile"
  else
    printf '%s\n' "$PROFILE_DIR/$profile.json"
  fi
}

safe_rm() {
  local path="$1"
  [[ -n "$path" ]] || return 0
  [[ -e "$path" || -L "$path" ]] || return 0
  local resolved
  resolved="$(readlink -f "$path" 2>/dev/null || printf '%s' "$path")"
  case "$resolved" in
    /work/scratch/pafina/objectx-data-fullscene-*|\
    /work/scratch/pafina/sam2object/*|\
    "$REPO_ROOT"/vis/*|\
    "$REPO_ROOT"/dependencies/SAM2Object/segtrack/outputs/*|\
    "$REPO_ROOT"/logs/*|\
    /tmp/"$USER"-objectx-feat3d*|\
    /work/courses/3dv/team35/pafina/logs/*)
      rm -rf "$path"
      ;;
    *)
      echo "[cleanup] refusing unsafe rm: $resolved" >&2
      ;;
  esac
}

cleanup_scene() {
  local profile="$1"
  local scene_id="$2"
  local profile_path
  profile_path="$(profile_path_for "$profile")"

  if [[ "$CLEANUP_AFTER_SCENE" != "1" || "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  echo "[cleanup] scene=$scene_id profile=$profile" >&2

  "$PYTHON_BIN" - "$profile_path" <<'PY' | while IFS= read -r path; do safe_rm "$path"; done
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
profile = json.loads(path.read_text())
roots = profile.get("roots", {})
for key in ("reconstruction", "pred_ready"):
    value = roots.get(key)
    if value:
        print(os.path.expandvars(value))

def walk(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)
    elif isinstance(value, str):
        yield os.path.expandvars(value)

for value in sorted(set(walk(profile))):
    if value.startswith("/work/scratch/pafina/objectx-data-fullscene-"):
        print(value)
PY

  local samroot="$SAMOBJECT_DATA_ROOT_DEFAULT"
  for path in \
    "$samroot/posed_images/$scene_id" \
    "$samroot/color_images_cluster/$scene_id" \
    "$samroot/scenes/$scene_id" \
    "$samroot/scans/$scene_id" \
    "$samroot/2D_masks/$scene_id" \
    "$REPO_ROOT/dependencies/SAM2Object/segtrack/outputs/$scene_id" \
    "$REPO_ROOT/vis/${scene_id}_joint.ply" \
    "$REPO_ROOT/vis/${scene_id}_slat.ply" \
    "/tmp/${USER}-objectx-feat3d"; do
    safe_rm "$path"
  done

  for path in \
    "$REPO_ROOT"/vis/rendered/"${scene_id}"_* \
    "$REPO_ROOT"/vis/rendered_gs/"${scene_id}"_* \
    "$REPO_ROOT"/logs/*"${scene_id}"* \
    "$REPO_ROOT"/logs/*"${profile}"* \
    /work/courses/3dv/team35/pafina/logs/*"${scene_id}"* \
    /work/courses/3dv/team35/pafina/logs/*"${profile}"*; do
    if [[ -e "$path" || -L "$path" ]]; then
      safe_rm "$path"
    fi
  done

  return 0
}

run_action() {
  local profile="$1"
  local action="$2"
  local scene_id="$3"
  local action_log_dir="$OUT_ROOT/$profile/$scene_id/action_logs"
  local action_log="$action_log_dir/${action}.log"
  echo "[run] $profile $action" >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  mkdir -p "$action_log_dir"
  echo "[run] $profile $action" > "$action_log"
  set +e
  GEOMETRY_EVAL_GROUP="$GEOMETRY_EVAL_GROUP" \
  WRITE_DEBUG_HTML=1 \
  DEBUG_HTML_MAX_POINTS="${DEBUG_HTML_MAX_POINTS:-150000}" \
  OBJECTX_VIS_NUM_FRAMES="${OBJECTX_FINAL_BENCH_VIS_NUM_FRAMES:-12}" \
  OBJECTX_VIS_RENDER_SCALE="${OBJECTX_FINAL_BENCH_VIS_RENDER_SCALE:-0.25}" \
  OBJECTX_VIS_SKIP_GS=1 \
  OBJECTX_VIS_EXPORT_MESH=0 \
  OBJECTX_FEATURES3D_RESET_TMP=1 \
  bash "$REPO_ROOT/scripts/workflows/run_scene_profile.sh" "$profile" "$action" 2>&1 | tee -a "$action_log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" == "0" && "${OBJECTX_BENCH_FAIL_ON_LOG_ERROR:-1}" == "1" ]]; then
    if grep -Eq '\[ERROR\]|Traceback \(most recent call last\)|AssertionError|CUDA initialization: CUDA unknown error|CUDA is not available' "$action_log"; then
      echo "[run] detected error pattern in log=$action_log" >&2
      rc=98
    fi
  fi
  if [[ "$rc" != "0" ]]; then
    echo "[run] failed rc=$rc log=$action_log" >&2
  fi
  return "$rc"
}

append_status() {
  local status="$1"
  local bucket="$2"
  local profile="$3"
  local scene_id="$4"
  local detail="$5"
  if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'status\tbucket\tprofile\tscene_id\tdetail\n' > "$STATUS_FILE"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$status" "$bucket" "$profile" "$scene_id" "$detail" >> "$STATUS_FILE"
}

write_aggregate() {
  "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/summarize_objectx_final_benchmark.py" \
    --selection-file "$SELECTION_FILE" \
    --out-root "$OUT_ROOT" \
    --report-threshold "${REPORT_THRESHOLD:-0.05}" \
    --secondary-threshold "${SECONDARY_REPORT_THRESHOLD:-0.10}"
}

if [[ "$RESAMPLE" == "1" || ! -f "$SELECTION_FILE" ]]; then
  select_scenes
else
  echo "[select] reusing $SELECTION_FILE" >&2
fi

completed=0
complete_success=0
seen=0
while IFS=$'\t' read -r bucket bucket_rank score score_scope scene_id profile source_metrics; do
  [[ "$bucket" == "bucket" ]] && continue
  if (( TARGET_COMPLETE_RUNS > 0 && complete_success >= TARGET_COMPLETE_RUNS )); then
    echo "[done] reached TARGET_COMPLETE_RUNS=$TARGET_COMPLETE_RUNS" >&2
    break
  fi
  seen=$((seen + 1))
  if (( seen < START_AT )); then
    continue
  fi
  if (( LIMIT > 0 && completed >= LIMIT )); then
    break
  fi

  scene_out="$OUT_ROOT/$profile/$scene_id"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$scene_out/report_summary.md" ]]; then
    echo "[skip] complete scene=$scene_id profile=$profile" >&2
    complete_success=$((complete_success + 1))
    completed=$((completed + 1))
    continue
  fi

  echo "========== [$seen] $bucket scene=$scene_id profile=$profile score=$score complete=$complete_success/$TARGET_COMPLETE_RUNS =========="
  ok=1
  failed_action=""
  for action in "${ACTIONS[@]}"; do
    if ! run_action "$profile" "$action" "$scene_id"; then
      ok=0
      failed_action="$action"
      break
    fi
  done

  if [[ "$ok" == "1" ]]; then
    append_status "done" "$bucket" "$profile" "$scene_id" "ok"
    complete_success=$((complete_success + 1))
  else
    append_status "failed" "$bucket" "$profile" "$scene_id" "$failed_action log=$scene_out/action_logs/${failed_action}.log"
  fi

  cleanup_scene "$profile" "$scene_id"
  write_aggregate

  completed=$((completed + 1))
  if [[ "$ok" != "1" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit 1
  fi
done < "$SELECTION_FILE"

write_aggregate
echo "[done] selection=$SELECTION_FILE"
echo "[done] outputs=$OUT_ROOT"
