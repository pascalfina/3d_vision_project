#!/usr/bin/env bash
set -euo pipefail

# Full ScanNet Object-X final-geometry benchmark for Pi3X + SAMObject.
#
# Per scene:
#   prepare-scannet -> must3r -> pi3x -> samobject -> voxelise
#   -> build-pred-ready -> features3d -> slat -> u3dgs
#   -> objectx-final-geometry-eval -> cleanup
#
# The runner intentionally keeps only evaluation outputs and optional overlay
# HTMLs. All generated ScanNet/Object-X/SAMObject intermediates are cleaned per
# scene so long jobs do not creep into quota trouble.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"
PROFILE_DIR="${PROFILE_DIR:-$REPO_ROOT/configs/workflows/scene_profiles}"

export OBJECTX_REPO_ROOT="$REPO_ROOT"
if [[ -f "$REPO_ROOT/configs/workflows/local_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/configs/workflows/local_paths.env"
fi

SCANNET_ROOT="${SCANNET_ROOT:-/work/courses/3dv/team35/pafina/scannet_under300_data}"
SCANNET_WORK_ROOT="${SCANNET_WORK_ROOT:-/work/courses/3dv/team35/pafina}"
SCANNET_SAMOBJECT_ROOT="${SCANNET_SAMOBJECT_ROOT:-/work/scratch/pafina/sam2object_scannet}"
OBJECTX_WORKFLOW_LOG_ROOT="${OBJECTX_WORKFLOW_LOG_ROOT:-$SCANNET_WORK_ROOT/logs}"

SOURCE_GROUP="${SOURCE_GROUP:-scannet_pi3x_sequence_under300}"
METRICS_GLOB="${METRICS_GLOB:-$REPO_ROOT/evaluation/outputs/geometry/$SOURCE_GROUP/**/metrics.json}"
SCANNET_TARGET_SCENES="${SCANNET_TARGET_SCENES:-300}"
SCANNET_SELECTION_MAX_FRAMES="${SCANNET_SELECTION_MAX_FRAMES:-300}"
SCANNET_SELECTION_MODE="${SCANNET_SELECTION_MODE:-cap}"
SCANNET_SELECTION_ORDER="${SCANNET_SELECTION_ORDER:-shortest}"
SCANNET_EXPORT_MAX_FRAMES="${SCANNET_EXPORT_MAX_FRAMES:-$SCANNET_SELECTION_MAX_FRAMES}"
SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-1}"
SCANNET_AUTO_FRAME_SKIP_FOR_MAX="${SCANNET_AUTO_FRAME_SKIP_FOR_MAX:-1}"
SCANNET_SCENE_TABLE="${SCANNET_SCENE_TABLE:-$PROFILE_DIR/scannet_under300_scenes.tsv}"
SCANNET_BASE_PROFILE_LIST="${SCANNET_BASE_PROFILE_LIST:-$PROFILE_DIR/scannet_under300_profiles.txt}"
SCANNET_REUSE_SELECTION="${SCANNET_REUSE_SELECTION:-1}"
SCANNET_PRUNE_STALE_PROFILES="${SCANNET_PRUNE_STALE_PROFILES:-1}"
SCANNET_PREPARE="${SCANNET_PREPARE:-1}"
SCANNET_DELETE_SENS_AFTER_EXPORT="${SCANNET_DELETE_SENS_AFTER_EXPORT:-1}"
SCANNET_DOWNLOAD_TYPES="${SCANNET_DOWNLOAD_TYPES:-.txt _vh_clean_2.ply .sens}"
SCANNET_DIRECT_DOWNLOAD="${SCANNET_DIRECT_DOWNLOAD:-1}"
SCANNET_DOWNLOAD_BACKEND="${SCANNET_DOWNLOAD_BACKEND:-auto}"
SCANNET_PREFETCH_AHEAD="${SCANNET_PREFETCH_AHEAD:-1}"
SCANNET_PURGE_PREPARED_AFTER_SCENE="${SCANNET_PURGE_PREPARED_AFTER_SCENE:-1}"
SCANNET_PURGE_PREPARED_AFTER_FAILED="${SCANNET_PURGE_PREPARED_AFTER_FAILED:-1}"
SCANNET_PURGE_PREPARED_ON_SKIP="${SCANNET_PURGE_PREPARED_ON_SKIP:-1}"

GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-scannet_objectx_final_100_pi3x_samobject}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP}"
SELECTION_FILE="${SELECTION_FILE:-$OUT_ROOT/selected_100.tsv}"
STATUS_FILE="${STATUS_FILE:-$OUT_ROOT/status.tsv}"
GENERATED_PROFILE_DIR="${GENERATED_PROFILE_DIR:-$OUT_ROOT/generated_profiles}"

RUN_SAM2_MUST3R_AFTER="${RUN_SAM2_MUST3R_AFTER:-1}"
SAM2_MUST3R_GEOMETRY_EVAL_GROUP="${SAM2_MUST3R_GEOMETRY_EVAL_GROUP:-scannet_objectx_final_100_sam2_must3r}"
SAM2_MUST3R_OUT_ROOT="${SAM2_MUST3R_OUT_ROOT:-$REPO_ROOT/evaluation/outputs/geometry/$SAM2_MUST3R_GEOMETRY_EVAL_GROUP}"
SAM2_MUST3R_SELECTION_FILE="${SAM2_MUST3R_SELECTION_FILE:-$SAM2_MUST3R_OUT_ROOT/selected_100.tsv}"
SAM2_MUST3R_STATUS_FILE="${SAM2_MUST3R_STATUS_FILE:-$SAM2_MUST3R_OUT_ROOT/status.tsv}"
SAM2_MUST3R_GENERATED_PROFILE_DIR="${SAM2_MUST3R_GENERATED_PROFILE_DIR:-$SAM2_MUST3R_OUT_ROOT/generated_profiles}"

SELECTION_TOTAL="${SELECTION_TOTAL:-130}"
BEST_COUNT="${BEST_COUNT:-65}"
MID_COUNT="${MID_COUNT:-32}"
WORST_COUNT="${WORST_COUNT:-33}"
TARGET_COMPLETE_RUNS="${TARGET_COMPLETE_RUNS:-100}"
ALLOW_SCANNET_TABLE_FALLBACK="${ALLOW_SCANNET_TABLE_FALLBACK:-1}"

DRY_RUN="${DRY_RUN:-0}"
RESAMPLE="${RESAMPLE:-0}"
FORCE="${FORCE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
KEEP_FAILED_ARTIFACTS="${KEEP_FAILED_ARTIFACTS:-0}"
KEEP_ACTION_LOGS="${KEEP_ACTION_LOGS:-0}"
ABORT_ON_CUDA_UNHEALTHY="${ABORT_ON_CUDA_UNHEALTHY:-1}"
COUNT_GLOBAL_COMPLETED="${COUNT_GLOBAL_COMPLETED:-1}"
START_AT="${START_AT:-1}"
LIMIT="${LIMIT:-0}"
WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"
DEBUG_HTML_MAX_POINTS="${DEBUG_HTML_MAX_POINTS:-150000}"

START_AT_RAW="$START_AT"
START_AT_MODE="index"
START_AT_INDEX=1
START_AT_SCENE=""
if [[ -z "$START_AT_RAW" ]]; then
  START_AT_RAW="1"
fi
if [[ "$START_AT_RAW" =~ ^[0-9]+$ ]]; then
  START_AT_INDEX=$((10#$START_AT_RAW))
  if (( START_AT_INDEX < 1 )); then
    START_AT_INDEX=1
  fi
else
  START_AT_MODE="scene"
  START_AT_SCENE="$START_AT_RAW"
fi

ACTIONS=(
  must3r
  pi3x
  link-sequence-inputs
  samobject
  verify-samobject-output
  ensure-scannet-recon-compat
  voxelise
  build-pred-ready
  verify-pred-ready-inference-inputs
  features3d
  slat
  u3dgs
  objectx-final-geometry-eval
)

SAM2_MUST3R_ACTIONS=(
  segment-inputs
  verify-sam2-masks
  ensure-scannet-recon-compat
  link-sequence-inputs
  voxelise
  build-pred-ready
  ensure-predready-sam2-alias
  verify-pred-ready-inference-inputs
  features3d
  slat
  u3dgs
  objectx-final-geometry-eval
)

export SCANNET_ROOT SCANNET_WORK_ROOT SCANNET_SAMOBJECT_ROOT OBJECTX_WORKFLOW_LOG_ROOT
export GEOMETRY_EVAL_GROUP

mkdir -p "$OUT_ROOT" "$GENERATED_PROFILE_DIR" "$OBJECTX_WORKFLOW_LOG_ROOT"
if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
  mkdir -p "$SAM2_MUST3R_OUT_ROOT" "$SAM2_MUST3R_GENERATED_PROFILE_DIR"
fi

echo "[scannet-pi3x-samobject] root=$SCANNET_ROOT"
echo "[scannet-pi3x-samobject] work_root=$SCANNET_WORK_ROOT samobject_root=$SCANNET_SAMOBJECT_ROOT"
echo "[scannet-pi3x-samobject] source_group=$SOURCE_GROUP group=$GEOMETRY_EVAL_GROUP"
echo "[scannet-pi3x-samobject] run_sam2_must3r_after=$RUN_SAM2_MUST3R_AFTER sam2_group=$SAM2_MUST3R_GEOMETRY_EVAL_GROUP"
echo "[scannet-pi3x-samobject] selection_total=$SELECTION_TOTAL target_complete=$TARGET_COMPLETE_RUNS"
echo "[scannet-pi3x-samobject] cleanup=$CLEANUP_AFTER_SCENE continue=$CONTINUE_ON_ERROR skip_completed=$SKIP_COMPLETED"
echo "[scannet-pi3x-samobject] purge_prepared_after_scene=$SCANNET_PURGE_PREPARED_AFTER_SCENE purge_failed=$SCANNET_PURGE_PREPARED_AFTER_FAILED purge_on_skip=$SCANNET_PURGE_PREPARED_ON_SKIP"
echo "[scannet-pi3x-samobject] download_backend=$SCANNET_DOWNLOAD_BACKEND prefetch_ahead=$SCANNET_PREFETCH_AHEAD"
echo "[scannet-pi3x-samobject] start_at=$START_AT_RAW mode=$START_AT_MODE index=$START_AT_INDEX scene=${START_AT_SCENE:-}"
echo "[scannet-pi3x-samobject] abort_on_cuda_unhealthy=$ABORT_ON_CUDA_UNHEALTHY"
echo "[scannet-pi3x-samobject] count_global_completed=$COUNT_GLOBAL_COMPLETED"

safe_rm() {
  local path="$1"
  [[ -n "$path" ]] || return 0
  [[ -e "$path" || -L "$path" ]] || return 0
  local resolved
  resolved="$(readlink -f "$path" 2>/dev/null || printf '%s' "$path")"
  case "$resolved" in
    "$SCANNET_WORK_ROOT"/objectx-data-scannet-*|\
    /work/scratch/pafina/objectx-data-scannet-*|\
    /work/courses/3dv/team35/pafina/objectx-data-scannet-*|\
    "$SCANNET_SAMOBJECT_ROOT"/*|\
    "$REPO_ROOT"/dependencies/SAM2Object/segtrack/outputs/*|\
    "$REPO_ROOT"/vis/*|\
    "$REPO_ROOT"/logs/*|\
    "$REPO_ROOT"/debug/*|\
    "$OUT_ROOT"/*/*/action_logs|\
    /tmp/"$USER"-objectx-feat3d*|\
    /tmp/"$USER"-objectx-infer*|\
    /tmp/"$USER"-objectx-voxelise*|\
    /work/courses/3dv/team35/pafina/logs/*)
      rm -rf -- "$path"
      ;;
    *)
      echo "[cleanup] refusing unsafe rm: $resolved" >&2
      ;;
  esac
}

scan_is_prepared() {
  local scene_id="$1"
  local seq="$SCANNET_ROOT/scenes/$scene_id/sequence"
  [[ -f "$seq/_info.txt" ]] || return 1
  compgen -G "$seq/frame-*.color.jpg" >/dev/null || return 1
  compgen -G "$seq/frame-*.pose.txt" >/dev/null || return 1
  [[ -f "$SCANNET_ROOT/scenes/$scene_id/${scene_id}_vh_clean_2.ply" || -f "$SCANNET_ROOT/scenes/$scene_id/${scene_id}_vh_clean.ply" ]]
}

declare -A SCANNET_PREFETCH_PIDS=()

prefetch_scan_async() {
  local scene_id="$1"
  if [[ "$DRY_RUN" == "1" || "$SCANNET_PREPARE" != "1" || "$SCANNET_PREFETCH_AHEAD" == "0" ]]; then
    return 0
  fi
  if [[ -z "$scene_id" || "$scene_id" == "scene_id" ]]; then
    return 0
  fi
  if scan_is_prepared "$scene_id"; then
    return 0
  fi
  local existing_pid="${SCANNET_PREFETCH_PIDS[$scene_id]:-}"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    return 0
  fi

  local prefetch_log="$OUT_ROOT/prefetch_logs/${scene_id}.log"
  mkdir -p "$(dirname "$prefetch_log")"
  (
    SCAN_ID="$scene_id" \
    SCANNET_ROOT="$SCANNET_ROOT" \
    SCANNET_MAX_FRAMES="$SCANNET_EXPORT_MAX_FRAMES" \
    SCANNET_FRAME_SKIP="$SCANNET_FRAME_SKIP" \
    SCANNET_AUTO_FRAME_SKIP_FOR_MAX="$SCANNET_AUTO_FRAME_SKIP_FOR_MAX" \
    SCANNET_DOWNLOAD_TYPES="$SCANNET_DOWNLOAD_TYPES" \
    SCANNET_DIRECT_DOWNLOAD="$SCANNET_DIRECT_DOWNLOAD" \
    SCANNET_DOWNLOAD_BACKEND="$SCANNET_DOWNLOAD_BACKEND" \
    SCANNET_DELETE_SENS_AFTER_EXPORT="$SCANNET_DELETE_SENS_AFTER_EXPORT" \
    bash "$REPO_ROOT/evaluation/geometry/prepare_scannet_scene.sh"
  ) >"$prefetch_log" 2>&1 &
  SCANNET_PREFETCH_PIDS["$scene_id"]=$!
  echo "[scannet-prefetch] started scene=$scene_id pid=${SCANNET_PREFETCH_PIDS[$scene_id]} log=$prefetch_log" >&2
}

wait_prefetch_scan() {
  local scene_id="$1"
  local pid="${SCANNET_PREFETCH_PIDS[$scene_id]:-}"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  echo "[scannet-prefetch] waiting scene=$scene_id pid=$pid" >&2
  if wait "$pid"; then
    unset 'SCANNET_PREFETCH_PIDS[$scene_id]'
    if scan_is_prepared "$scene_id"; then
      echo "[scannet-prefetch] ready scene=$scene_id" >&2
      return 0
    fi
    echo "[scannet-prefetch] finished but scene is not prepared: $scene_id" >&2
    return 2
  fi
  unset 'SCANNET_PREFETCH_PIDS[$scene_id]'
  echo "[scannet-prefetch] failed scene=$scene_id; falling back to synchronous prepare" >&2
  return 2
}

prefetch_upcoming_scans() {
  local current_seen="$1"
  local ahead="$SCANNET_PREFETCH_AHEAD"
  if [[ "$ahead" == "0" || "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  local offset scene
  for ((offset = 1; offset <= ahead; offset++)); do
    scene="$(
      awk -F $'\t' -v row="$((current_seen + 1 + offset))" 'NR == row {print $5}' "$SELECTION_FILE"
    )"
    if [[ -n "$scene" ]]; then
      prefetch_scan_async "$scene"
    fi
  done
}

prepare_scan() {
  local scene_id="$1"
  if scan_is_prepared "$scene_id"; then
    echo "[scannet-prepare] already prepared: $scene_id"
    return 0
  fi
  if wait_prefetch_scan "$scene_id"; then
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
  SCANNET_DOWNLOAD_BACKEND="$SCANNET_DOWNLOAD_BACKEND" \
  SCANNET_DELETE_SENS_AFTER_EXPORT="$SCANNET_DELETE_SENS_AFTER_EXPORT" \
  bash "$REPO_ROOT/evaluation/geometry/prepare_scannet_scene.sh"
  scan_is_prepared "$scene_id"
}

purge_prepared_scan() {
  local scene_id="$1"
  if [[ ! "$scene_id" =~ ^scene[0-9]{4}_[0-9]{2}$ ]]; then
    echo "[scannet-purge] skip suspicious scene id: $scene_id" >&2
    return 0
  fi
  local root_real
  root_real="$(readlink -f "$SCANNET_ROOT" 2>/dev/null || printf '%s' "$SCANNET_ROOT")"
  case "$root_real" in
    /work/courses/3dv/team35/pafina/scannet_under300_data|/work/scratch/pafina/scannet_under300_data)
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

cleanup_stale_download_parts() {
  # Keep *.part files so interrupted ScanNet downloads can resume.
  return 0
}

ensure_base_profiles() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[scannet-profiles:dry-run] would ensure ScanNet base Pi3X profiles"
    return 0
  fi
  if [[ "$SCANNET_REUSE_SELECTION" != "1" || ! -f "$SCANNET_SCENE_TABLE" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/select_scannet_under300.py" \
      --scannet-root "$SCANNET_ROOT" \
      --target "$SCANNET_TARGET_SCENES" \
      --max-frames "$SCANNET_SELECTION_MAX_FRAMES" \
      --mode "$SCANNET_SELECTION_MODE" \
      --order "$SCANNET_SELECTION_ORDER" \
      --out "$SCANNET_SCENE_TABLE"
  fi
  local generate_args=(
    "$REPO_ROOT/evaluation/geometry/generate_scannet_pi3x_profiles.py"
    --scene-table "$SCANNET_SCENE_TABLE"
    --manifest "$SCANNET_BASE_PROFILE_LIST"
    --max-scenes "$SCANNET_TARGET_SCENES"
  )
  if [[ "$SCANNET_PRUNE_STALE_PROFILES" == "1" ]]; then
    generate_args+=(--prune-stale)
  fi
  "$PYTHON_BIN" "${generate_args[@]}"
}

prepare_selection() {
  if [[ "$RESAMPLE" != "1" && -f "$SELECTION_FILE" ]]; then
    echo "[select] reusing $SELECTION_FILE" >&2
    return 0
  fi
  ensure_base_profiles
  "$PYTHON_BIN" - \
    "$METRICS_GLOB" \
    "$PROFILE_DIR" \
    "$SCANNET_SCENE_TABLE" \
    "$SELECTION_FILE" \
    "$SELECTION_TOTAL" \
    "$BEST_COUNT" \
    "$MID_COUNT" \
    "$WORST_COUNT" \
    "$ALLOW_SCANNET_TABLE_FALLBACK" <<'PY'
import csv
import glob
import json
import math
import sys
from pathlib import Path

metrics_glob = sys.argv[1]
profile_dir = Path(sys.argv[2])
scene_table = Path(sys.argv[3])
out_path = Path(sys.argv[4])
total = int(sys.argv[5])
best_count = int(sys.argv[6])
mid_count = int(sys.argv[7])
worst_count = int(sys.argv[8])
allow_fallback = sys.argv[9] == "1"

if best_count + mid_count + worst_count != total:
    raise SystemExit("BEST_COUNT + MID_COUNT + WORST_COUNT must equal SELECTION_TOTAL")

def final_profile_name(scene_id: str) -> str:
    return f"scannet_{scene_id}_pi3x_samobject"

profiles_by_scene = {}
for path in profile_dir.glob("scannet_scene*_pi3x.json"):
    try:
        profile = json.loads(path.read_text())
    except Exception:
        continue
    scene_id = profile.get("scene_id")
    if scene_id:
        profiles_by_scene[scene_id] = path.stem

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
    if isinstance(chamfer, (int, float)) and math.isfinite(float(chamfer)):
        return float(chamfer), scope_name
    values = []
    for value in (pred.get("mean"), comp.get("mean")):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
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
    if scene_id not in profiles_by_scene:
        continue
    value, scope_name = score(metrics)
    if not math.isfinite(value):
        continue
    row = {
        "scene_id": scene_id,
        "profile": final_profile_name(scene_id),
        "score": value,
        "score_scope": scope_name,
        "source_metrics": str(path),
    }
    current = rows_by_scene.get(scene_id)
    if current is None or value < current["score"]:
        rows_by_scene[scene_id] = row

rows = sorted(rows_by_scene.values(), key=lambda item: (item["score"], item["scene_id"]))

selected = []
if len(rows) >= total:
    best = rows[:best_count]
    worst = rows[-worst_count:] if worst_count else []
    middle_pool = rows[best_count : len(rows) - worst_count]
    if len(middle_pool) < mid_count:
        raise SystemExit("Not enough middle-pool scenes after best/worst split")
    mid_start = max(0, (len(middle_pool) - mid_count) // 2)
    middle = middle_pool[mid_start : mid_start + mid_count]
    buckets = (("best", best), ("middle", middle), ("worst", worst))
else:
    if not allow_fallback:
        raise SystemExit(f"Need {total} scored scenes, found {len(rows)}")
    third = max(1, len(rows) // 4) if rows else 0
    best = rows[: max(0, min(len(rows), len(rows) // 2))]
    remaining = rows[len(best):]
    worst = remaining[-third:] if third else []
    middle = remaining[: max(0, len(remaining) - len(worst))]
    buckets = (("best", best), ("middle", middle), ("worst", worst))

for bucket, bucket_rows in buckets:
    for local_rank, row in enumerate(bucket_rows, start=1):
        selected.append({"bucket": bucket, "bucket_rank": local_rank, **row})

selected_scene_ids = {row["scene_id"] for row in selected}
fallback_rows = []
if len(selected) < total and allow_fallback:
    if not scene_table.exists():
        raise SystemExit(f"Missing scene table for fallback selection: {scene_table}")
    for line in scene_table.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("rank\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        scene_id = parts[1]
        if scene_id in selected_scene_ids or scene_id not in profiles_by_scene:
            continue
        fallback_rows.append(
            {
                "bucket": "fallback",
                "bucket_rank": len(fallback_rows) + 1,
                "score": "",
                "score_scope": "unscored_scene_table",
                "scene_id": scene_id,
                "profile": final_profile_name(scene_id),
                "source_metrics": "fallback:scannet_under300_scenes.tsv",
            }
        )
        selected_scene_ids.add(scene_id)
        if len(selected) + len(fallback_rows) >= total:
            break
selected.extend(fallback_rows)

if len(selected) < total:
    raise SystemExit(f"Need {total} selected scenes, found {len(selected)}")

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
print(f"[select] scored={len(rows)} fallback={len(fallback_rows)}")
PY
}

prepare_sam2_must3r_selection() {
  if [[ "$RUN_SAM2_MUST3R_AFTER" != "1" ]]; then
    return 0
  fi
  if [[ "$RESAMPLE" != "1" && -f "$SAM2_MUST3R_SELECTION_FILE" ]]; then
    echo "[select:sam2-must3r] reusing $SAM2_MUST3R_SELECTION_FILE" >&2
    return 0
  fi
  "$PYTHON_BIN" - "$SELECTION_FILE" "$SAM2_MUST3R_SELECTION_FILE" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
with source.open(newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        scene_id = row["scene_id"]
        row = dict(row)
        row["profile"] = f"scannet_{scene_id}_sam2_must3r"
        rows.append(row)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
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
    writer.writerows(rows)
print(f"[select:sam2-must3r] wrote {len(rows)} scenes to {out}")
PY
}

generate_profile() {
  local requested_profile="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$PROFILE_DIR" "$GENERATED_PROFILE_DIR" "$requested_profile" "$GEOMETRY_EVAL_GROUP" <<'PY'
import copy
import json
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
profile_dir = Path(sys.argv[2])
generated_dir = Path(sys.argv[3])
requested_profile = sys.argv[4]
group = sys.argv[5]

source_name = requested_profile
if source_name.endswith("_pi3x_samobject"):
    source_name = source_name[: -len("_samobject")]

src_path = profile_dir / f"{source_name}.json"
if not src_path.exists():
    raise SystemExit(f"could not resolve source ScanNet Pi3X profile: {requested_profile}")

profile = json.loads(src_path.read_text())
scene_id = profile["scene_id"]
new_name = f"scannet_{scene_id}_pi3x_samobject"
work_root = "${SCANNET_WORK_ROOT}"
baseline_root = "${SCANNET_ROOT}"
samobject_root = "${SCANNET_SAMOBJECT_ROOT}"
must3r_root = f"{work_root}/objectx-data-scannet-{scene_id}-pi3x-samobject-must3r"
recon_root = f"{work_root}/objectx-data-scannet-{scene_id}-pi3x-samobject"
pred_ready_root = f"{work_root}/objectx-data-scannet-{scene_id}-predready-pi3x-samobject-v1"

template_pi3x = json.loads((profile_dir / "oven_legacy_sam2_pi3x.json").read_text())

out = copy.deepcopy(profile)
out["name"] = new_name
out["dataset"] = "scannet"
out["input_variant"] = "legacy"
out["mask_source"] = "gt_projection"
out["roots"] = {
    "baseline": baseline_root,
    "reconstruction": recon_root,
    "pred_ready": pred_ready_root,
}
out["artifacts"] = {
    "joint_ply": f"${{OBJECTX_REPO_VIS_ROOT}}/{scene_id}_joint.ply",
}
out.pop("segment_inputs", None)

must3r = copy.deepcopy(profile.get("must3r", {}))
must3r.update(
    {
        "run_must3r": True,
        "run_sam2": False,
        "run_registry": False,
        "input_root": baseline_root,
        "output_root": must3r_root,
        "mask_dirname": "gt_projection",
        "objects_filename": "objects.json",
        "scenes_dirname": "scenes_sam2_must3r",
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{scene_id}_pi3x_samobject_must3r.log",
    }
)
out["must3r"] = must3r

pi3x = copy.deepcopy(profile.get("pi3x", {}))
pi3x_env = dict(pi3x.get("env", {}))
pi3x_env["OBJECTX_PI3X_EXTERNAL_POSE_DIR"] = (
    f"{must3r_root}/scenes_sam2_must3r/{scene_id}/sequence"
)
pi3x.update(
    {
        "run_must3r": True,
        "run_sam2": False,
        "run_registry": False,
        "input_root": baseline_root,
        "output_root": recon_root,
        "mask_dirname": "gt_projection",
        "objects_filename": "objects.json",
        "scenes_dirname": "scenes_sam2_pi3x",
        "env": pi3x_env,
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{scene_id}_pi3x_samobject_pi3x.log",
    }
)
out["pi3x"] = pi3x

samobject = copy.deepcopy(template_pi3x.get("samobject", {}))
samobject.update(
    {
        "root_dir": samobject_root,
        "baseline_root": baseline_root,
        "output_root_dir": recon_root,
        "source_sequence_dir": f"{recon_root}/scenes_sam2_pi3x/{scene_id}/sequence",
        "mesh_path": f"{baseline_root}/scenes/{scene_id}/{scene_id}_vh_clean_2.ply",
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/samobject_scannet_{scene_id}_pi3x.log",
    }
)
sam_env = samobject.setdefault("env", {})
sam_env["SAMOBJECT_ADAPTER"] = "pi3x_full_samobject"
sam_env["SAMOBJECT_USE_PI3X_SURFACE"] = "1"
sam_env.setdefault("SAMOBJECT_GRAPH_PROCESS_NUM", "1")
out["samobject"] = samobject

voxelise = copy.deepcopy(template_pi3x.get("voxelise", profile.get("voxelise", {})))
voxelise.update(
    {
        "data_root": recon_root,
        "baseline_root": baseline_root,
        "scene_source_dirname": "scenes_sam2_pi3x",
        "mask_source": "gt_projection",
        "objects_filename": "objects_sam2.json",
        "override": True,
        "reset_tmp": 1,
        "max_scans": 0,
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{scene_id}_pi3x_samobject_voxelise.log",
    }
)
out["voxelise"] = voxelise

build_pred_ready = copy.deepcopy(template_pi3x.get("build_pred_ready", profile.get("build_pred_ready", {})))
build_pred_ready.update(
    {
        "baseline_root": baseline_root,
        "reconstruction_root": recon_root,
        "reconstruction_scenes_dirname": "scenes_sam2_pi3x",
        "target_root": pred_ready_root,
        "overwrite": True,
    }
)
out["build_pred_ready"] = build_pred_ready

features3d = copy.deepcopy(profile.get("features3d", {}))
features3d.update(
    {
        "data_root": pred_ready_root,
        "scene_source_dirname": "scenes",
        "mask_source": "gt_projection",
        "reset_tmp": 1,
        "log": f"${{OBJECTX_REPO_DEBUG_ROOT}}/debug_{new_name}_features3d.log",
    }
)
out["features3d"] = features3d

for name in ("slat", "u3dgs"):
    sec = copy.deepcopy(template_pi3x.get(name, profile.get(name, {})))
    sec["data_root"] = pred_ready_root
    sec["mask_source"] = "gt_projection"
    env = sec.setdefault("env", {})
    env["OBJECTX_MASK_SOURCE"] = "gt_projection"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("RESET_TMP", "1")
    if name == "u3dgs":
        env.setdefault("OBJECTX_VIS_SKIP_GS", "1")
        env.setdefault("OBJECTX_VIS_RENDER_SCALE", "0.25")
        env.setdefault("OBJECTX_VIS_NUM_FRAMES", "12")
        env.setdefault("OBJECTX_INFER_CPU_DENSE_DECODE_FALLBACK", "1")
    sec["log"] = f"${{OBJECTX_REPO_DEBUG_ROOT}}/debug_{new_name}_{name}.log"
    out[name] = sec

frame_cap = int(profile.get("geometry_eval", {}).get("visible_max_frames", 160) or 160)
out["objectx_final_geometry_eval"] = {
    "dataset": "scannet",
    "method_name": new_name,
    "group": group,
    "baseline_root": baseline_root,
    "pred_root": recon_root,
    "pred_ready_root": pred_ready_root,
    "scenes_dirname": "scenes_sam2_pi3x",
    "raw_reference_name": "raw_pi3x_sequence",
    "raw_reference_label": "Raw Pi3X",
    "strict_scene_guard": 1,
    "strict_sequence_color_guard": 1,
    "write_debug_html": 1,
    "debug_html_max_points": 150000,
    "final_vs_raw_sequence_max_frames": min(frame_cap, 160),
    "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{scene_id}_objectx_final_geometry_eval.log",
}

generated_dir.mkdir(parents=True, exist_ok=True)
out_path = generated_dir / f"{new_name}.json"
out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(new_name)
print(out_path)
print(scene_id)
print(os.path.expandvars(must3r_root))
print(os.path.expandvars(recon_root))
print(os.path.expandvars(pred_ready_root))
PY
}

generate_sam2_must3r_profile() {
  local scene_id="$1"
  local common_must3r_root="$2"
  "$PYTHON_BIN" - \
    "$REPO_ROOT" \
    "$PROFILE_DIR" \
    "$SAM2_MUST3R_GENERATED_PROFILE_DIR" \
    "$scene_id" \
    "$common_must3r_root" \
    "$SAM2_MUST3R_GEOMETRY_EVAL_GROUP" <<'PY'
import copy
import json
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
profile_dir = Path(sys.argv[2])
generated_dir = Path(sys.argv[3])
scene_id = sys.argv[4]
common_must3r_root = sys.argv[5]
group = sys.argv[6]

source_path = profile_dir / f"scannet_{scene_id}_pi3x.json"
if not source_path.exists():
    raise SystemExit(f"missing source ScanNet Pi3X profile: {source_path}")
profile = json.loads(source_path.read_text())
template_path = profile_dir / "oven_legacy_sam2_must3r_hybrid.json"
template = json.loads(template_path.read_text()) if template_path.exists() else {}

new_name = f"scannet_{scene_id}_sam2_must3r"
baseline_root = "${SCANNET_ROOT}"
work_root = "${SCANNET_WORK_ROOT}"
pred_ready_root = f"{work_root}/objectx-data-scannet-{scene_id}-predready-sam2-must3r-v1"

out = copy.deepcopy(profile)
out["name"] = new_name
out["dataset"] = "scannet"
out["input_variant"] = "legacy+sam+must3r"
out["mask_source"] = "sam2_projection"
out["roots"] = {
    "baseline": baseline_root,
    "reconstruction": common_must3r_root,
    "pred_ready": pred_ready_root,
}
out["artifacts"] = {
    "joint_ply": f"${{OBJECTX_REPO_VIS_ROOT}}/{scene_id}_joint.ply",
}
out.pop("pi3x", None)
out.pop("samobject", None)
out.pop("geometry_eval", None)

must3r = copy.deepcopy(profile.get("must3r", {}))
must3r.update(
    {
        "run_must3r": True,
        "run_sam2": False,
        "run_registry": False,
        "input_root": baseline_root,
        "output_root": common_must3r_root,
        "mask_dirname": "sam2_projection",
        "objects_filename": "objects_sam2.json",
        "scenes_dirname": "scenes_sam2_must3r",
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{scene_id}_dual_common_must3r.log",
    }
)
out["must3r"] = must3r

segment_inputs = copy.deepcopy(template.get("segment_inputs", {}))
segment_inputs.update(
    {
        "config": "preprocessing/segmentation/pipeline.yaml",
        "run_must3r": False,
        "run_sam2": True,
        "run_registry": True,
        "input_root": baseline_root,
        "output_root": common_must3r_root,
        "mask_dirname": "sam2_projection",
        "objects_filename": "objects_sam2.json",
        "scenes_dirname": "scenes_sam2_must3r",
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{scene_id}_sam2_must3r_segment_inputs.log",
    }
)
seg_env = segment_inputs.setdefault("env", {})
seg_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
seg_env.setdefault("OBJECTX_SAM2_OFFLOAD_VIDEO_TO_CPU", "1")
seg_env.setdefault("OBJECTX_SAM2_OFFLOAD_STATE_TO_CPU", "1")
seg_env.setdefault("OBJECTX_SAM2_ASYNC_LOADING_FRAMES", "0")
out["segment_inputs"] = segment_inputs

voxelise = copy.deepcopy(template.get("voxelise", profile.get("voxelise", {})))
voxelise.update(
    {
        "data_root": common_must3r_root,
        "baseline_root": baseline_root,
        "scene_source_dirname": "scenes_sam2_must3r",
        "mask_source": "sam2_projection",
        "objects_filename": "objects_sam2.json",
        "override": True,
        "reset_tmp": 1,
        "max_scans": 0,
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{scene_id}_sam2_must3r_voxelise.log",
    }
)
voxel_env = voxelise.setdefault("env", {})
voxel_env["OBJECTX_VOXEL_MAX_VIEWS"] = "100"
out["voxelise"] = voxelise

build_pred_ready = copy.deepcopy(template.get("build_pred_ready", profile.get("build_pred_ready", {})))
build_pred_ready.update(
    {
        "baseline_root": baseline_root,
        "reconstruction_root": common_must3r_root,
        "reconstruction_scenes_dirname": "scenes_sam2_must3r",
        "target_root": pred_ready_root,
        "overwrite": True,
    }
)
out["build_pred_ready"] = build_pred_ready

features3d = copy.deepcopy(profile.get("features3d", {}))
features3d.update(
    {
        "data_root": pred_ready_root,
        "scene_source_dirname": "scenes",
        "mask_source": "sam2_projection",
        "reset_tmp": 1,
        "log": f"${{OBJECTX_REPO_DEBUG_ROOT}}/debug_{new_name}_features3d.log",
    }
)
out["features3d"] = features3d

for name in ("slat", "u3dgs"):
    sec = copy.deepcopy(template.get(name, profile.get(name, {})))
    sec["data_root"] = pred_ready_root
    sec["mask_source"] = "sam2_projection"
    sec["mask_root"] = common_must3r_root
    env = sec.setdefault("env", {})
    env["OBJECTX_MASK_SOURCE"] = "sam2_projection"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("RESET_TMP", "1")
    if name == "u3dgs":
        env.setdefault("OBJECTX_VIS_SKIP_GS", "1")
        env.setdefault("OBJECTX_VIS_RENDER_SCALE", "0.25")
        env.setdefault("OBJECTX_VIS_NUM_FRAMES", "12")
        env.setdefault("OBJECTX_INFER_CPU_DENSE_DECODE_FALLBACK", "1")
    sec["log"] = f"${{OBJECTX_REPO_DEBUG_ROOT}}/debug_{new_name}_{name}.log"
    out[name] = sec

frame_cap = int(profile.get("geometry_eval", {}).get("visible_max_frames", 160) or 160)
out["objectx_final_geometry_eval"] = {
    "dataset": "scannet",
    "method_name": new_name,
    "group": group,
    "baseline_root": baseline_root,
    "pred_root": common_must3r_root,
    "pred_ready_root": pred_ready_root,
    "scenes_dirname": "scenes_sam2_must3r",
    "raw_reference_name": "raw_must3r_sequence",
    "raw_reference_label": "Raw MUSt3R",
    "strict_scene_guard": 1,
    "strict_sequence_color_guard": 1,
    "write_debug_html": 1,
    "debug_html_max_points": 150000,
    "final_vs_raw_sequence_max_frames": min(frame_cap, 160),
    "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/scannet_{scene_id}_sam2_must3r_objectx_final_geometry_eval.log",
}

generated_dir.mkdir(parents=True, exist_ok=True)
out_path = generated_dir / f"{new_name}.json"
out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(new_name)
print(out_path)
print(scene_id)
print(os.path.expandvars(common_must3r_root))
print(os.path.expandvars(pred_ready_root))
PY
}

link_sequence_inputs() {
  local scene_id="$1"
  local recon_root="$2"
  local scenes_dirname="${3:-scenes_sam2_pi3x}"
  local src_seq="$SCANNET_ROOT/scenes/$scene_id/sequence"
  local dst_seq="$recon_root/$scenes_dirname/$scene_id/sequence"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[sequence-links:dry-run] would link color/depth/_info from $src_seq -> $dst_seq"
    return 0
  fi
  if [[ ! -d "$src_seq" || ! -d "$dst_seq" ]]; then
    echo "[sequence-links] missing source or destination sequence: src=$src_seq dst=$dst_seq" >&2
    return 2
  fi
  if [[ -f "$src_seq/_info.txt" && ! -e "$dst_seq/_info.txt" ]]; then
    ln -sfn "$(realpath --relative-to="$dst_seq" "$src_seq/_info.txt")" "$dst_seq/_info.txt"
  fi
  local linked=0
  shopt -s nullglob
  for src in "$src_seq"/frame-*.color.jpg "$src_seq"/frame-*.depth.pgm; do
    local dst="$dst_seq/$(basename "$src")"
    if [[ ! -e "$dst" ]]; then
      ln -s "$(realpath --relative-to="$dst_seq" "$src")" "$dst"
      linked=$((linked + 1))
    fi
  done
  shopt -u nullglob
  if [[ ! -e "$dst_seq/_info.txt" ]]; then
    echo "[sequence-links] missing _info.txt after linking: $dst_seq/_info.txt" >&2
    return 2
  fi
  echo "[sequence-links] linked sequence inputs: scene=$scene_id scenes=$scenes_dirname linked=$linked"
}

verify_samobject_output() {
  local scene_id="$1"
  local recon_root="$2"
  local mask_dir="$recon_root/files/gt_projection/obj_id_pkl"
  if [[ ! -f "$mask_dir/$scene_id.pkl" && ! -f "$mask_dir/$scene_id.pkl.gz" ]]; then
    echo "[samobject-guard] missing SAMObject mask pkl in $mask_dir" >&2
    return 2
  fi
  if [[ ! -f "$recon_root/files/objects_sam2.json" ]]; then
    echo "[samobject-guard] missing objects_sam2.json: $recon_root/files/objects_sam2.json" >&2
    return 2
  fi
  if [[ ! -f "$recon_root/scenes_sam2_pi3x/$scene_id/labels.instances.annotated.v2.ply" ]]; then
    echo "[samobject-guard] missing SAMObject-segmented Pi3X PLY in scenes_sam2_pi3x" >&2
    return 2
  fi
  echo "[samobject-guard] verified SAMObject masks + objects_sam2 + segmented Pi3X PLY"
}

ensure_scannet_recon_compat() {
  local scene_id="$1"
  local recon_root="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[compat:dry-run] would write split files under $recon_root/files"
    return 0
  fi
  mkdir -p "$recon_root/files"
  for split in train val test; do
    local payload=""
    if [[ "$split" == "val" ]]; then
      payload="$scene_id"$'\n'
    fi
    printf '%s' "$payload" > "$recon_root/files/${split}_resplit_scans.txt"
    printf '%s' "$payload" > "$recon_root/files/${split}_scans.txt"
  done
  if [[ ! -f "$recon_root/files/objects.json" && -f "$recon_root/files/objects_sam2.json" ]]; then
    ln -sfn objects_sam2.json "$recon_root/files/objects.json"
  fi
  echo "[compat] wrote ScanNet split/object aliases for $scene_id"
}

verify_pred_ready_inference_inputs() {
  local scene_id="$1"
  local pred_ready_root="$2"
  local missing=0
  local files_dir="$pred_ready_root/files"
  local scene_dir="$pred_ready_root/scenes/$scene_id"

  for path in \
    "$files_dir/3RScan.json" \
    "$files_dir/objects.json" \
    "$files_dir/scannet40_classes.txt" \
    "$scene_dir/data.npy"; do
    if [[ ! -e "$path" ]]; then
      echo "[pred-ready-guard] missing $path" >&2
      missing=1
    fi
  done

  if [[ ! -f "$files_dir/orig/data/$scene_id.pkl" && ! -f "$files_dir/orig/data/$scene_id.pkl.gz" ]]; then
    echo "[pred-ready-guard] missing scene graph: $files_dir/orig/data/$scene_id.pkl[.gz]" >&2
    missing=1
  fi
  if [[ ! -f "$files_dir/gt_projection/obj_id_pkl/$scene_id.pkl" && ! -f "$files_dir/gt_projection/obj_id_pkl/$scene_id.pkl.gz" ]]; then
    echo "[pred-ready-guard] missing gt_projection mask pkl: $files_dir/gt_projection/obj_id_pkl/$scene_id.pkl[.gz]" >&2
    missing=1
  fi
  if [[ ! -d "$files_dir/gs_annotations/$scene_id" ]]; then
    echo "[pred-ready-guard] missing gs_annotations scene dir: $files_dir/gs_annotations/$scene_id" >&2
    missing=1
  fi

  if [[ "$missing" != "0" ]]; then
    return 2
  fi
  echo "[pred-ready-guard] verified inference inputs for $scene_id"
}

verify_sam2_masks() {
  local scene_id="$1"
  local recon_root="$2"
  local sam2_dir="$recon_root/files/sam2_projection/obj_id_pkl"
  local sam2_pkl="$sam2_dir/$scene_id.pkl"
  local sam2_pkl_gz="$sam2_dir/$scene_id.pkl.gz"
  local gt_pkl="$recon_root/files/gt_projection/obj_id_pkl/$scene_id.pkl"
  local gt_pkl_gz="$recon_root/files/gt_projection/obj_id_pkl/$scene_id.pkl.gz"

  if [[ ! -f "$sam2_pkl" && ! -f "$sam2_pkl_gz" ]]; then
    echo "[sam2-guard] missing SAM2 mask pkl in $sam2_dir" >&2
    return 2
  fi
  if [[ ! -f "$recon_root/files/objects_sam2.json" ]]; then
    echo "[sam2-guard] missing SAM2 object registry: $recon_root/files/objects_sam2.json" >&2
    return 2
  fi
  if [[ -f "$gt_pkl" || -f "$gt_pkl_gz" ]]; then
    echo "[sam2-guard] found GT mask pkl in reconstruction root, refusing SAM2-only path: $gt_pkl[.gz]" >&2
    return 2
  fi
  echo "[sam2-guard] verified SAM2 masks + objects_sam2 for $scene_id"
}

ensure_predready_sam2_alias() {
  local scene_id="$1"
  local recon_root="$2"
  local pred_ready_root="$3"
  local src="$recon_root/files/sam2_projection"
  local dst="$pred_ready_root/files/sam2_projection"
  local gt_alias="$pred_ready_root/files/gt_projection"

  if [[ ! -d "$src/obj_id_pkl" ]]; then
    echo "[sam2-guard] missing source SAM2 projection: $src" >&2
    return 2
  fi
  mkdir -p "$pred_ready_root/files"
  rm -rf -- "$dst" "$gt_alias"
  ln -s "$src" "$dst"
  ln -s "$src" "$gt_alias"
  "$PYTHON_BIN" - "$pred_ready_root/files/SAM2_MASK_SOURCE_GUARD.json" "$scene_id" "$src" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "scene_id": sys.argv[2],
    "mask_source": "sam2_projection",
    "gt_projection_alias_points_to": sys.argv[3],
    "note": "This pred-ready root intentionally aliases gt_projection to SAM2 masks for Object-X compatibility.",
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  if [[ ! -e "$gt_alias/obj_id_pkl/$scene_id.pkl" && ! -e "$gt_alias/obj_id_pkl/$scene_id.pkl.gz" ]]; then
    echo "[sam2-guard] pred-ready SAM2 alias did not expose mask pkl for $scene_id" >&2
    return 2
  fi
  echo "[sam2-guard] pred-ready gt_projection -> sam2_projection alias installed"
}

common_must3r_ready() {
  local scene_id="$1"
  local must3r_root="$2"
  local seq="$must3r_root/scenes_sam2_must3r/$scene_id/sequence"
  [[ -d "$seq" ]] || return 1
  compgen -G "$seq/frame-*.pose.txt" >/dev/null || return 1
  compgen -G "$seq/frame-*.depth.npy" >/dev/null || compgen -G "$seq/frame-*.xyz.npy" >/dev/null
}

cleanup_branch_artifacts() {
  local scene_id="$1"
  local recon_root="$2"
  local pred_ready_root="$3"
  if [[ "$CLEANUP_AFTER_SCENE" != "1" || "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  safe_rm "$recon_root"
  safe_rm "$pred_ready_root"
  safe_rm "$REPO_ROOT/vis/${scene_id}_joint.ply"
  safe_rm "$REPO_ROOT/vis/${scene_id}_slat.ply"
  safe_rm "/tmp/${USER}-objectx-feat3d"
  safe_rm "/tmp/${USER}-objectx-infer"
  safe_rm "/tmp/${USER}-objectx-voxelise"
}

gpu_action_requires_cuda() {
  case "$1" in
    must3r|pi3x|samobject|segment-inputs|voxelise|features3d|slat|u3dgs)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

cuda_healthcheck() {
  "$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch

    print("[cuda-guard] torch=", torch.__version__, "torch_cuda=", torch.version.cuda)
    print("[cuda-guard] cuda_available=", torch.cuda.is_available())
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    print("[cuda-guard] cuda_device=", torch.cuda.get_device_name(0))
    print("[cuda-guard] cuda_capability=", capability)
    print("[cuda-guard] bf16_supported=", torch.cuda.is_bf16_supported())
    if capability < (7, 0):
        sys.exit(f"GPU capability sm_{capability[0]}{capability[1]} is too old")
    if not torch.cuda.is_bf16_supported():
        sys.exit("bf16 is not supported")
    probe = torch.empty((1,), device="cuda")
    probe += 1
    torch.cuda.synchronize()
except Exception as exc:
    print(f"[cuda-guard] ERROR: {exc}", file=sys.stderr)
    sys.exit(97)
PY
}

run_profile_action() {
  local profile_name="$1"
  local profile_json="$2"
  local action="$3"
  local scene_id="$4"
  local action_log_root="${RUN_ACTION_OUT_ROOT:-$OUT_ROOT}"
  local action_log_dir="$action_log_root/$profile_name/$scene_id/action_logs"
  local action_log="$action_log_dir/${action}.log"
  echo "[run] $profile_name $action" >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    bash "$REPO_ROOT/scripts/workflows/run_scene_profile.sh" "$profile_json" "$action" --dry-run
    return 0
  fi
  mkdir -p "$action_log_dir"
  echo "[run] $profile_name $action" > "$action_log"
  if gpu_action_requires_cuda "$action"; then
    if ! cuda_healthcheck 2>&1 | tee -a "$action_log"; then
      echo "[cuda-guard] unhealthy before $profile_name $action log=$action_log" >&2
      if [[ "$ABORT_ON_CUDA_UNHEALTHY" == "1" ]]; then
        echo "[cuda-guard] aborting benchmark because CUDA is unhealthy; resubmit on a fresh GPU node" >&2
        exit 97
      fi
      return 97
    fi
  fi
  set +e
  GEOMETRY_EVAL_GROUP="$GEOMETRY_EVAL_GROUP" \
  WRITE_DEBUG_HTML="$WRITE_DEBUG_HTML" \
  DEBUG_HTML_MAX_POINTS="$DEBUG_HTML_MAX_POINTS" \
  OBJECTX_VIS_NUM_FRAMES="${OBJECTX_FINAL_BENCH_VIS_NUM_FRAMES:-12}" \
  OBJECTX_VIS_RENDER_SCALE="${OBJECTX_FINAL_BENCH_VIS_RENDER_SCALE:-0.25}" \
  OBJECTX_VIS_SKIP_ORBIT=1 \
  OBJECTX_VIS_SKIP_GS=1 \
  OBJECTX_VIS_EXPORT_MESH=0 \
  OBJECTX_FEATURES3D_RESET_TMP=1 \
  bash "$REPO_ROOT/scripts/workflows/run_scene_profile.sh" "$profile_json" "$action" 2>&1 | tee -a "$action_log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ! -f "$action_log" ]]; then
    echo "[run] action log disappeared; treating as failed action=$action expected_log=$action_log" >&2
    rc=99
  fi
  if [[ "$rc" == "0" && "${OBJECTX_BENCH_FAIL_ON_LOG_ERROR:-1}" == "1" ]]; then
    if grep -Eiq '\[ERROR\]|Traceback \(most recent call last\)|AssertionError|CUDA initialization: CUDA unknown error|CUDA is not available|CUDA out of memory|out of memory|Disk quota exceeded|No space left on device|Killed' "$action_log"; then
      echo "[run] detected error pattern in log=$action_log" >&2
      rc=98
    fi
  fi
  if [[ "$rc" != "0" ]]; then
    echo "[run] failed rc=$rc log=$action_log" >&2
  fi
  return "$rc"
}

append_status_file() {
  local status_file="$1"
  shift
  local status="$1"
  local bucket="$2"
  local profile="$3"
  local scene_id="$4"
  local detail="$5"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[status:dry-run] $status $bucket $profile $scene_id $detail" >&2
    return 0
  fi
  if [[ ! -f "$status_file" ]]; then
    mkdir -p "$(dirname "$status_file")"
    printf 'status\tbucket\tprofile\tscene_id\tdetail\n' > "$status_file"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$status" "$bucket" "$profile" "$scene_id" "$detail" >> "$status_file"
}

append_status() {
  append_status_file "$STATUS_FILE" "$@"
}

append_sam2_status() {
  append_status_file "$SAM2_MUST3R_STATUS_FILE" "$@"
}

cleanup_scene() {
  local profile_name="$1"
  local scene_id="$2"
  local must3r_root="$3"
  local recon_root="$4"
  local pred_ready_root="$5"
  if [[ "$CLEANUP_AFTER_SCENE" != "1" || "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  echo "[cleanup] scene=$scene_id profile=$profile_name" >&2
  safe_rm "$must3r_root"
  safe_rm "$recon_root"
  safe_rm "$pred_ready_root"
  for path in \
    "$SCANNET_SAMOBJECT_ROOT/posed_images/$scene_id" \
    "$SCANNET_SAMOBJECT_ROOT/color_images_cluster/$scene_id" \
    "$SCANNET_SAMOBJECT_ROOT/scenes/$scene_id" \
    "$SCANNET_SAMOBJECT_ROOT/scans/$scene_id" \
    "$SCANNET_SAMOBJECT_ROOT/2D_masks/$scene_id" \
    "$SCANNET_SAMOBJECT_ROOT/superpoints/$scene_id" \
    "$REPO_ROOT/dependencies/SAM2Object/segtrack/outputs/$scene_id" \
    "$REPO_ROOT/vis/${scene_id}_joint.ply" \
    "$REPO_ROOT/vis/${scene_id}_slat.ply" \
    "/tmp/${USER}-objectx-feat3d" \
    "/tmp/${USER}-objectx-infer" \
    "/tmp/${USER}-objectx-voxelise"; do
    safe_rm "$path"
  done
  for path in \
    "$REPO_ROOT"/vis/rendered/"${scene_id}"_* \
    "$REPO_ROOT"/vis/rendered_gs/"${scene_id}"_* \
    "$REPO_ROOT"/logs/*"${scene_id}"* \
    "$REPO_ROOT"/logs/*"${profile_name}"* \
    "$REPO_ROOT"/debug/*"${scene_id}"* \
    "$REPO_ROOT"/debug/*"${profile_name}"* \
    /work/courses/3dv/team35/pafina/logs/*"${scene_id}"* \
    /work/courses/3dv/team35/pafina/logs/*"${profile_name}"*; do
    if [[ -e "$path" || -L "$path" ]]; then
      safe_rm "$path"
    fi
  done
  if [[ "$KEEP_ACTION_LOGS" != "1" ]]; then
    local cleanup_out_root="${CLEANUP_OUT_ROOT:-$OUT_ROOT}"
    safe_rm "$cleanup_out_root/$profile_name/$scene_id/action_logs"
  fi
  cleanup_stale_download_parts
}

CURRENT_PROFILE_NAME=""
CURRENT_SCENE_ID=""
CURRENT_MUST3R_ROOT=""
CURRENT_RECON_ROOT=""
CURRENT_PRED_READY_ROOT=""
CURRENT_SAM2_PROFILE_NAME=""
CURRENT_SAM2_RECON_ROOT=""
CURRENT_SAM2_PRED_READY_ROOT=""

cleanup_current_scene_on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  for scene in "${!SCANNET_PREFETCH_PIDS[@]}"; do
    local pid="${SCANNET_PREFETCH_PIDS[$scene]}"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[scannet-prefetch] stopping scene=$scene pid=$pid" >&2
      kill "$pid" 2>/dev/null || true
    fi
  done
  if [[ -n "${CURRENT_SCENE_ID:-}" ]]; then
    echo "[cleanup] exit trap rc=$rc scene=$CURRENT_SCENE_ID profile=$CURRENT_PROFILE_NAME" >&2
    cleanup_scene "$CURRENT_PROFILE_NAME" "$CURRENT_SCENE_ID" "$CURRENT_MUST3R_ROOT" "$CURRENT_RECON_ROOT" "$CURRENT_PRED_READY_ROOT"
    if [[ -n "${CURRENT_SAM2_PROFILE_NAME:-}" ]]; then
      CLEANUP_OUT_ROOT="$SAM2_MUST3R_OUT_ROOT" cleanup_scene "$CURRENT_SAM2_PROFILE_NAME" "$CURRENT_SCENE_ID" "" "$CURRENT_SAM2_RECON_ROOT" "$CURRENT_SAM2_PRED_READY_ROOT"
    fi
  fi
  exit "$rc"
}

write_aggregate() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[aggregate:dry-run] would summarize $OUT_ROOT"
    return 0
  fi
  "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/summarize_objectx_final_benchmark.py" \
    --selection-file "$SELECTION_FILE" \
    --out-root "$OUT_ROOT" \
    --report-threshold "${REPORT_THRESHOLD:-0.05}" \
    --secondary-threshold "${SECONDARY_REPORT_THRESHOLD:-0.10}"
}

write_sam2_aggregate() {
  if [[ "$RUN_SAM2_MUST3R_AFTER" != "1" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[aggregate:dry-run] would summarize $SAM2_MUST3R_OUT_ROOT"
    return 0
  fi
  "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/summarize_objectx_final_benchmark.py" \
    --selection-file "$SAM2_MUST3R_SELECTION_FILE" \
    --out-root "$SAM2_MUST3R_OUT_ROOT" \
    --report-threshold "${REPORT_THRESHOLD:-0.05}" \
    --secondary-threshold "${SECONDARY_REPORT_THRESHOLD:-0.10}"
}

count_completed_reports() {
  local selection_file="$1"
  local out_root="$2"
  [[ -f "$selection_file" ]] || {
    printf '0\n'
    return 0
  }
  awk -F $'\t' -v out_root="$out_root" '
    NR > 1 {
      path = out_root "/" $6 "/" $5 "/report_summary.md"
      if ((getline _ < path) >= 0) {
        count += 1
        close(path)
      }
    }
    END { print count + 0 }
  ' "$selection_file"
}

prepare_selection
prepare_sam2_must3r_selection
trap cleanup_current_scene_on_exit EXIT INT TERM HUP

completed=0
complete_success=0
sam2_complete_success=0
if [[ "$COUNT_GLOBAL_COMPLETED" == "1" ]]; then
  complete_success="$(count_completed_reports "$SELECTION_FILE" "$OUT_ROOT")"
  if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
    sam2_complete_success="$(count_completed_reports "$SAM2_MUST3R_SELECTION_FILE" "$SAM2_MUST3R_OUT_ROOT")"
  fi
  echo "[progress] initial global complete pi3x=$complete_success/$TARGET_COMPLETE_RUNS sam2=$sam2_complete_success/$TARGET_COMPLETE_RUNS" >&2
fi
seen=0
start_at_scene_seen=0
while IFS=$'\t' read -r bucket bucket_rank score score_scope scene_id profile source_metrics; do
  [[ "$bucket" == "bucket" ]] && continue
  if (( TARGET_COMPLETE_RUNS > 0 )); then
    if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
      if (( complete_success >= TARGET_COMPLETE_RUNS && sam2_complete_success >= TARGET_COMPLETE_RUNS )); then
        echo "[done] reached TARGET_COMPLETE_RUNS=$TARGET_COMPLETE_RUNS for both branches" >&2
        break
      fi
    elif (( complete_success >= TARGET_COMPLETE_RUNS )); then
      echo "[done] reached TARGET_COMPLETE_RUNS=$TARGET_COMPLETE_RUNS" >&2
      break
    fi
  fi
  seen=$((seen + 1))
  if [[ "$START_AT_MODE" == "index" ]]; then
    if (( seen < START_AT_INDEX )); then
      continue
    fi
  elif (( start_at_scene_seen == 0 )); then
    if [[ "$scene_id" != "$START_AT_SCENE" && "$profile" != "$START_AT_SCENE" ]]; then
      continue
    fi
    start_at_scene_seen=1
    echo "[start-at] matched START_AT=$START_AT_SCENE at selection_index=$seen scene=$scene_id profile=$profile" >&2
  fi
  if (( LIMIT > 0 && completed >= LIMIT )); then
    break
  fi

  mapfile -t profile_info < <(generate_profile "$profile")
  profile_name="${profile_info[0]}"
  profile_json="${profile_info[1]}"
  scene_id="${profile_info[2]}"
  must3r_root="${profile_info[3]}"
  recon_root="${profile_info[4]}"
  pred_ready_root="${profile_info[5]}"
  scene_out="$OUT_ROOT/$profile_name/$scene_id"

  sam2_profile_name=""
  sam2_profile_json=""
  sam2_recon_root=""
  sam2_pred_ready_root=""
  sam2_scene_out=""
  if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
    mapfile -t sam2_profile_info < <(generate_sam2_must3r_profile "$scene_id" "$must3r_root")
    sam2_profile_name="${sam2_profile_info[0]}"
    sam2_profile_json="${sam2_profile_info[1]}"
    sam2_recon_root="${sam2_profile_info[3]}"
    sam2_pred_ready_root="${sam2_profile_info[4]}"
    sam2_scene_out="$SAM2_MUST3R_OUT_ROOT/$sam2_profile_name/$scene_id"
  fi

  CURRENT_PROFILE_NAME="$profile_name"
  CURRENT_SCENE_ID="$scene_id"
  CURRENT_MUST3R_ROOT="$must3r_root"
  CURRENT_RECON_ROOT="$recon_root"
  CURRENT_PRED_READY_ROOT="$pred_ready_root"
  CURRENT_SAM2_PROFILE_NAME="$sam2_profile_name"
  CURRENT_SAM2_RECON_ROOT="$sam2_recon_root"
  CURRENT_SAM2_PRED_READY_ROOT="$sam2_pred_ready_root"

  needs_pi3x=1
  needs_sam2=0
  if [[ "$SKIP_COMPLETED" == "1" && -f "$scene_out/report_summary.md" ]]; then
    echo "[skip] complete pi3x+samobject scene=$scene_id profile=$profile_name" >&2
    if [[ "$COUNT_GLOBAL_COMPLETED" != "1" ]]; then
      complete_success=$((complete_success + 1))
    fi
    needs_pi3x=0
  fi
  if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
    needs_sam2=1
    if [[ "$SKIP_COMPLETED" == "1" && -f "$sam2_scene_out/report_summary.md" ]]; then
      echo "[skip] complete sam2+must3r scene=$scene_id profile=$sam2_profile_name" >&2
      if [[ "$COUNT_GLOBAL_COMPLETED" != "1" ]]; then
        sam2_complete_success=$((sam2_complete_success + 1))
      fi
      needs_sam2=0
    fi
  fi

  if [[ "$needs_pi3x" == "0" && "$needs_sam2" == "0" ]]; then
    if [[ "$SCANNET_PURGE_PREPARED_ON_SKIP" == "1" ]]; then
      purge_prepared_scan "$scene_id"
    fi
    completed=$((completed + 1))
    CURRENT_PROFILE_NAME=""
    CURRENT_SCENE_ID=""
    CURRENT_MUST3R_ROOT=""
    CURRENT_RECON_ROOT=""
    CURRENT_PRED_READY_ROOT=""
    CURRENT_SAM2_PROFILE_NAME=""
    CURRENT_SAM2_RECON_ROOT=""
    CURRENT_SAM2_PRED_READY_ROOT=""
    continue
  fi

  echo "========== [$seen] $bucket scene=$scene_id score=${score:-n/a} pi3x=$complete_success/$TARGET_COMPLETE_RUNS sam2=$sam2_complete_success/$TARGET_COMPLETE_RUNS =========="
  cleanup_scene "$profile_name" "$scene_id" "$must3r_root" "$recon_root" "$pred_ready_root"
  if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
    CLEANUP_OUT_ROOT="$SAM2_MUST3R_OUT_ROOT" cleanup_scene "$sam2_profile_name" "$scene_id" "" "$sam2_recon_root" "$sam2_pred_ready_root"
  fi

  pi3x_ok=1
  sam2_ok=1
  pi3x_failed_action=""
  sam2_failed_action=""
  common_ready=0

  if [[ "$FORCE" == "1" && "$DRY_RUN" != "1" ]]; then
    rm -rf -- "$scene_out"
    if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
      rm -rf -- "$sam2_scene_out"
    fi
  fi

  if [[ "$SCANNET_PREPARE" == "1" ]]; then
    if ! prepare_scan "$scene_id"; then
      pi3x_ok=0
      sam2_ok=0
      pi3x_failed_action="prepare-scannet"
      sam2_failed_action="prepare-scannet"
    fi
  fi
  if [[ "$pi3x_ok" == "1" || "$sam2_ok" == "1" ]]; then
    prefetch_upcoming_scans "$seen"
  fi

  if [[ "$needs_pi3x" == "1" && "$pi3x_ok" == "1" ]]; then
    for action in "${ACTIONS[@]}"; do
      case "$action" in
        link-sequence-inputs)
          if ! link_sequence_inputs "$scene_id" "$recon_root"; then
            pi3x_ok=0
            pi3x_failed_action="$action"
            break
          fi
          ;;
        verify-samobject-output)
          if [[ "$DRY_RUN" == "1" ]]; then
            echo "[samobject-guard:dry-run] would verify SAMObject output"
          elif ! verify_samobject_output "$scene_id" "$recon_root"; then
            pi3x_ok=0
            pi3x_failed_action="$action"
            break
          fi
          ;;
        ensure-scannet-recon-compat)
          if ! ensure_scannet_recon_compat "$scene_id" "$recon_root"; then
            pi3x_ok=0
            pi3x_failed_action="$action"
            break
          fi
          ;;
        verify-pred-ready-inference-inputs)
          if [[ "$DRY_RUN" == "1" ]]; then
            echo "[pred-ready-guard:dry-run] would verify pred-ready inference inputs"
          elif ! verify_pred_ready_inference_inputs "$scene_id" "$pred_ready_root"; then
            pi3x_ok=0
            pi3x_failed_action="$action"
            break
          fi
          ;;
        *)
          if ! run_profile_action "$profile_name" "$profile_json" "$action" "$scene_id"; then
            pi3x_ok=0
            pi3x_failed_action="$action"
            break
          fi
          if [[ "$action" == "must3r" ]]; then
            if common_must3r_ready "$scene_id" "$must3r_root"; then
              common_ready=1
            else
              echo "[must3r-guard] MUSt3R action returned success but outputs are incomplete: $must3r_root" >&2
              pi3x_ok=0
              pi3x_failed_action="$action"
              break
            fi
          fi
          ;;
      esac
    done
  fi

  if [[ "$needs_pi3x" == "1" ]]; then
    if [[ "$pi3x_ok" == "1" ]]; then
      append_status "done" "$bucket" "$profile_name" "$scene_id" "ok"
      complete_success=$((complete_success + 1))
    else
      append_status "failed" "$bucket" "$profile_name" "$scene_id" "$pi3x_failed_action log=$scene_out/action_logs/${pi3x_failed_action}.log"
    fi
    if [[ "$pi3x_ok" == "1" || "$KEEP_FAILED_ARTIFACTS" != "1" ]]; then
      cleanup_branch_artifacts "$scene_id" "$recon_root" "$pred_ready_root"
    fi
  fi

  if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" && "$needs_sam2" == "1" ]]; then
    if [[ "$common_ready" != "1" ]] && ! common_must3r_ready "$scene_id" "$must3r_root"; then
      echo "[dual] common MUSt3R root missing; running MUSt3R once for SAM2 branch" >&2
      if run_profile_action "$profile_name" "$profile_json" "must3r" "$scene_id"; then
        common_ready=1
      else
        sam2_ok=0
        sam2_failed_action="must3r"
      fi
    fi

    if [[ "$sam2_ok" == "1" ]]; then
      for action in "${SAM2_MUST3R_ACTIONS[@]}"; do
        case "$action" in
          verify-sam2-masks)
            if [[ "$DRY_RUN" == "1" ]]; then
              echo "[sam2-guard:dry-run] would verify SAM2 masks in $sam2_recon_root/files/sam2_projection"
            elif ! verify_sam2_masks "$scene_id" "$sam2_recon_root"; then
              sam2_ok=0
              sam2_failed_action="$action"
              break
            fi
            ;;
          ensure-scannet-recon-compat)
            if ! ensure_scannet_recon_compat "$scene_id" "$sam2_recon_root"; then
              sam2_ok=0
              sam2_failed_action="$action"
              break
            fi
            ;;
          link-sequence-inputs)
            if ! link_sequence_inputs "$scene_id" "$sam2_recon_root" "scenes_sam2_must3r"; then
              sam2_ok=0
              sam2_failed_action="$action"
              break
            fi
            ;;
          ensure-predready-sam2-alias)
            if [[ "$DRY_RUN" == "1" ]]; then
              echo "[sam2-guard:dry-run] would alias pred-ready gt_projection to SAM2 masks"
            elif ! ensure_predready_sam2_alias "$scene_id" "$sam2_recon_root" "$sam2_pred_ready_root"; then
              sam2_ok=0
              sam2_failed_action="$action"
              break
            fi
            ;;
          verify-pred-ready-inference-inputs)
            if [[ "$DRY_RUN" == "1" ]]; then
              echo "[pred-ready-guard:dry-run] would verify SAM2 pred-ready inference inputs"
            elif ! verify_pred_ready_inference_inputs "$scene_id" "$sam2_pred_ready_root"; then
              sam2_ok=0
              sam2_failed_action="$action"
              break
            fi
            ;;
          *)
            if ! RUN_ACTION_OUT_ROOT="$SAM2_MUST3R_OUT_ROOT" run_profile_action "$sam2_profile_name" "$sam2_profile_json" "$action" "$scene_id"; then
              sam2_ok=0
              sam2_failed_action="$action"
              break
            fi
            ;;
        esac
      done
    fi

    if [[ "$sam2_ok" == "1" ]]; then
      append_sam2_status "done" "$bucket" "$sam2_profile_name" "$scene_id" "ok"
      sam2_complete_success=$((sam2_complete_success + 1))
    else
      append_sam2_status "failed" "$bucket" "$sam2_profile_name" "$scene_id" "$sam2_failed_action log=$sam2_scene_out/action_logs/${sam2_failed_action}.log"
    fi
  fi

  if [[ "$pi3x_ok" == "1" && "$sam2_ok" == "1" ]]; then
    if [[ "$SCANNET_PURGE_PREPARED_AFTER_SCENE" == "1" ]]; then
      purge_prepared_scan "$scene_id"
    fi
  else
    if [[ "$pi3x_failed_action" == "prepare-scannet" || "$sam2_failed_action" == "prepare-scannet" ]]; then
      echo "[scannet-purge] keeping partial prepared data for failed download resume: $scene_id" >&2
    elif [[ "$SCANNET_PURGE_PREPARED_AFTER_FAILED" == "1" ]]; then
      purge_prepared_scan "$scene_id"
    fi
  fi

  if [[ "$pi3x_ok" == "1" && "$sam2_ok" == "1" || "$KEEP_FAILED_ARTIFACTS" != "1" ]]; then
    cleanup_scene "$profile_name" "$scene_id" "$must3r_root" "$recon_root" "$pred_ready_root"
    if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
      CLEANUP_OUT_ROOT="$SAM2_MUST3R_OUT_ROOT" cleanup_scene "$sam2_profile_name" "$scene_id" "" "$sam2_recon_root" "$sam2_pred_ready_root"
    fi
  fi
  CURRENT_PROFILE_NAME=""
  CURRENT_SCENE_ID=""
  CURRENT_MUST3R_ROOT=""
  CURRENT_RECON_ROOT=""
  CURRENT_PRED_READY_ROOT=""
  CURRENT_SAM2_PROFILE_NAME=""
  CURRENT_SAM2_RECON_ROOT=""
  CURRENT_SAM2_PRED_READY_ROOT=""
  write_aggregate || true
  write_sam2_aggregate || true

  completed=$((completed + 1))
  if [[ "$pi3x_ok" != "1" || "$sam2_ok" != "1" ]] && [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit 1
  fi
done < "$SELECTION_FILE"

if [[ "$START_AT_MODE" == "scene" && "$start_at_scene_seen" == "0" ]]; then
  echo "[start-at] WARNING: START_AT scene/profile not found in selection: $START_AT_SCENE" >&2
fi

write_aggregate || true
write_sam2_aggregate || true
echo "[done] selection=$SELECTION_FILE"
echo "[done] outputs=$OUT_ROOT"
if [[ "$RUN_SAM2_MUST3R_AFTER" == "1" ]]; then
  echo "[done] sam2_selection=$SAM2_MUST3R_SELECTION_FILE"
  echo "[done] sam2_outputs=$SAM2_MUST3R_OUT_ROOT"
fi
