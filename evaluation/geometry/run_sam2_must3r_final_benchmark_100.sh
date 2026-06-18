#!/usr/bin/env bash
set -euo pipefail

# Full Object-X final-geometry benchmark for the SAM2 + MUSt3R baseline.
#
# This intentionally does NOT run SAMObject.  Per scene:
#   must3r -> segment-inputs(SAM2) -> SAM2 mask guard -> voxelise
#   -> build-pred-ready -> pred-ready SAM2 alias guard
#   -> features3d -> slat -> u3dgs -> objectx-final-geometry-eval -> cleanup

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"
PROFILE_DIR="${PROFILE_DIR:-$REPO_ROOT/configs/workflows/scene_profiles}"

SOURCE_SELECTION_FILE="${SOURCE_SELECTION_FILE:-$REPO_ROOT/evaluation/outputs/geometry/objectx_final_100/selected_100.tsv}"
GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-objectx_final_100_sam2_must3r}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP}"
SELECTION_FILE="${SELECTION_FILE:-$OUT_ROOT/selected_100.tsv}"
STATUS_FILE="${STATUS_FILE:-$OUT_ROOT/status.tsv}"
GENERATED_PROFILE_DIR="${GENERATED_PROFILE_DIR:-$OUT_ROOT/generated_profiles}"

TARGET_COMPLETE_RUNS="${TARGET_COMPLETE_RUNS:-100}"
DRY_RUN="${DRY_RUN:-0}"
RESAMPLE="${RESAMPLE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
START_AT="${START_AT:-1}"
LIMIT="${LIMIT:-0}"

ACTIONS=(
  must3r
  segment-inputs
  verify-sam2-masks
  voxelise
  build-pred-ready
  ensure-predready-sam2-alias
  features3d
  slat
  u3dgs
  objectx-final-geometry-eval
)

mkdir -p "$OUT_ROOT" "$GENERATED_PROFILE_DIR"

safe_rm() {
  local path="$1"
  [[ -n "$path" ]] || return 0
  [[ -e "$path" || -L "$path" ]] || return 0
  local resolved
  resolved="$(readlink -f "$path" 2>/dev/null || printf '%s' "$path")"
  case "$resolved" in
    /work/scratch/pafina/objectx-data-fullscene-*|\
    "$REPO_ROOT"/vis/*|\
    "$REPO_ROOT"/logs/*|\
    "$REPO_ROOT"/debug/*|\
    /tmp/"$USER"-objectx-feat3d*|\
    /tmp/"$USER"-objectx-infer*|\
    /work/courses/3dv/team35/pafina/logs/*)
      rm -rf "$path"
      ;;
    *)
      echo "[cleanup] refusing unsafe rm: $resolved" >&2
      ;;
  esac
}

CURRENT_PROFILE_NAME=""
CURRENT_SCENE_ID=""
CURRENT_RECON_ROOT=""
CURRENT_PRED_READY_ROOT=""

generate_profile() {
  local source_profile="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$PROFILE_DIR" "$GENERATED_PROFILE_DIR" "$source_profile" "$GEOMETRY_EVAL_GROUP" <<'PY'
import copy
import json
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
profile_dir = Path(sys.argv[2])
generated_dir = Path(sys.argv[3])
source_profile = sys.argv[4]
group = sys.argv[5]

src_path = Path(source_profile)
if not src_path.exists():
    src_path = profile_dir / f"{source_profile}.json"
if not src_path.exists() and source_profile.endswith("_sam2_must3r"):
    src_path = profile_dir / f"{source_profile.replace('_sam2_must3r', '_sam2_pi3x')}.json"
if not src_path.exists():
    raise SystemExit(f"could not resolve source profile for {source_profile}")
profile = json.loads(src_path.read_text())
scene_id = profile["scene_id"]
short = scene_id.split("-")[0]
old_name = profile.get("name") or src_path.stem
new_name = old_name.replace("_sam2_pi3x", "_sam2_must3r")
if new_name == old_name:
    new_name = f"scene_{short}_sam2_must3r"

must3r = copy.deepcopy(profile.get("must3r", {}))
if not must3r:
    raise SystemExit(f"source profile has no must3r section: {src_path}")

baseline_root = profile.get("roots", {}).get("baseline", "${OBJECTX_BASELINE_ROOT}")
recon_root = must3r.get(
    "output_root",
    f"/work/scratch/pafina/objectx-data-fullscene-{short}-hybrid-sam2mask-must3r",
)
pred_ready_root = f"/work/scratch/pafina/objectx-data-fullscene-{short}-predready-sam2-must3r-v1"

template_path = profile_dir / "oven_legacy_sam2_must3r_hybrid.json"
template = json.loads(template_path.read_text()) if template_path.exists() else {}

out = copy.deepcopy(profile)
out["name"] = new_name
out["input_variant"] = "legacy+sam+must3r"
out["mask_source"] = "sam2_projection"
out["roots"] = {
    "baseline": baseline_root,
    "reconstruction": recon_root,
    "pred_ready": pred_ready_root,
}
out["artifacts"] = {
    "joint_ply": f"${{OBJECTX_REPO_VIS_ROOT}}/{scene_id}_joint.ply",
}
out.pop("pi3x", None)
out.pop("samobject", None)

must3r.update(
    {
        "run_must3r": True,
        "run_sam2": False,
        "run_registry": False,
        "input_root": baseline_root,
        "output_root": recon_root,
        "mask_dirname": "sam2_projection",
        "objects_filename": "objects_sam2.json",
        "scenes_dirname": "scenes_sam2_must3r",
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/debug_{short}_sam2_must3r_must3r.log",
    }
)
out["must3r"] = must3r

segment_inputs = copy.deepcopy(profile.get("segment_inputs") or template.get("segment_inputs", {}))
segment_inputs.update(
    {
        "run_must3r": False,
        "run_sam2": True,
        "run_registry": True,
        "input_root": baseline_root,
        "output_root": recon_root,
        "mask_dirname": "sam2_projection",
        "objects_filename": "objects_sam2.json",
        "scenes_dirname": "scenes_sam2_must3r",
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/debug_{short}_sam2_must3r_segment_inputs.log",
    }
)
out["segment_inputs"] = segment_inputs

voxelise = copy.deepcopy(template.get("voxelise", profile.get("voxelise", {})))
voxelise.update(
    {
        "scene_source_dirname": "scenes_sam2_must3r",
        "override": True,
        "reset_tmp": 1,
        "max_scans": 0,
        "mask_source": "sam2_projection",
        "objects_filename": "objects_sam2.json",
        "log": f"${{OBJECTX_WORKFLOW_LOG_ROOT}}/debug_{short}_2_5_sam2_must3r.log",
    }
)
out["voxelise"] = voxelise

build_pred_ready = copy.deepcopy(template.get("build_pred_ready", profile.get("build_pred_ready", {})))
build_pred_ready.update(
    {
        "reconstruction_root": recon_root,
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
    sec["mask_root"] = recon_root
    env = sec.setdefault("env", {})
    env["OBJECTX_MASK_SOURCE"] = "sam2_projection"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("RESET_TMP", "1")
    sec["log"] = f"${{OBJECTX_REPO_DEBUG_ROOT}}/debug_{new_name}_{name}.log"
    out[name] = sec

out["objectx_final_geometry_eval"] = {
    "method_name": new_name,
    "group": group,
    "pred_root": recon_root,
    "pred_ready_root": pred_ready_root,
    "scenes_dirname": "scenes_sam2_must3r",
    "raw_reference_name": "raw_must3r_sequence",
    "raw_reference_label": "Raw MUSt3R",
    "strict_scene_guard": 1,
    "strict_sequence_color_guard": 1,
    "write_debug_html": 1,
}

generated_dir.mkdir(parents=True, exist_ok=True)
out_path = generated_dir / f"{new_name}.json"
out_path.write_text(json.dumps(out, indent=2) + "\n")
print(new_name)
print(out_path)
print(scene_id)
print(recon_root)
print(pred_ready_root)
PY
}

prepare_selection() {
  if [[ "$RESAMPLE" != "1" && -f "$SELECTION_FILE" ]]; then
    echo "[select] reusing $SELECTION_FILE" >&2
    return 0
  fi
  if [[ ! -f "$SOURCE_SELECTION_FILE" ]]; then
    echo "[select] missing source selection: $SOURCE_SELECTION_FILE" >&2
    exit 2
  fi
  "$PYTHON_BIN" - "$SOURCE_SELECTION_FILE" "$SELECTION_FILE" "$PROFILE_DIR" "$GENERATED_PROFILE_DIR" "$GEOMETRY_EVAL_GROUP" "$REPO_ROOT" <<'PY'
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
profile_dir = Path(sys.argv[3])
generated_dir = Path(sys.argv[4])
group = sys.argv[5]
repo_root = Path(sys.argv[6])

rows = []
with source.open(newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if not row:
            continue
        src_profile = row["profile"]
        src_path = profile_dir / f"{src_profile}.json"
        profile = json.loads(src_path.read_text())
        scene_id = profile["scene_id"]
        short = scene_id.split("-")[0]
        new_name = (profile.get("name") or src_profile).replace("_sam2_pi3x", "_sam2_must3r")
        if new_name == (profile.get("name") or src_profile):
            new_name = f"scene_{short}_sam2_must3r"
        row = dict(row)
        row["profile"] = new_name
        rows.append(row)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["bucket", "bucket_rank", "score", "score_scope", "scene_id", "profile", "source_metrics"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)
print(f"[select] wrote {len(rows)} transformed rows to {out}")
PY
}

verify_sam2_masks() {
  local scene_id="$1"
  local recon_root="$2"
  local sam2_dir="$recon_root/files/sam2_projection/obj_id_pkl"
  local sam2_pkl="$sam2_dir/$scene_id.pkl"
  local sam2_pkl_gz="$sam2_dir/$scene_id.pkl.gz"
  local gt_dir="$recon_root/files/gt_projection"

  if [[ ! -f "$sam2_pkl" && ! -f "$sam2_pkl_gz" ]]; then
    echo "[sam2-guard] missing SAM2 mask pkl in $sam2_dir" >&2
    return 2
  fi
  if [[ ! -f "$recon_root/files/objects_sam2.json" ]]; then
    echo "[sam2-guard] missing SAM2 object registry: $recon_root/files/objects_sam2.json" >&2
    return 2
  fi
  if [[ -e "$gt_dir" || -L "$gt_dir" ]]; then
    echo "[sam2-guard] reconstruction root unexpectedly contains gt_projection: $gt_dir" >&2
    echo "[sam2-guard] refusing to continue because this SAM2-only benchmark must not consume GT/SAMObject masks." >&2
    return 2
  fi
  echo "[sam2-guard] verified SAM2 masks: $sam2_pkl"
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
  rm -rf "$dst" "$gt_alias"
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
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
  if [[ ! -e "$gt_alias/obj_id_pkl/$scene_id.pkl" && ! -e "$gt_alias/obj_id_pkl/$scene_id.pkl.gz" ]]; then
    echo "[sam2-guard] pred-ready SAM2 alias did not expose mask pkl for $scene_id" >&2
    return 2
  fi
  echo "[sam2-guard] pred-ready gt_projection -> sam2_projection alias installed"
}

run_profile_action() {
  local profile_name="$1"
  local profile_json="$2"
  local action="$3"
  local scene_id="$4"
  local action_log_dir="$OUT_ROOT/$profile_name/$scene_id/action_logs"
  local action_log="$action_log_dir/${action}.log"
  echo "[run] $profile_name $action" >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    bash "$REPO_ROOT/scripts/workflows/run_scene_profile.sh" "$profile_json" "$action" --dry-run
    return 0
  fi
  mkdir -p "$action_log_dir"
  echo "[run] $profile_name $action" > "$action_log"
  set +e
  GEOMETRY_EVAL_GROUP="$GEOMETRY_EVAL_GROUP" \
  WRITE_DEBUG_HTML=1 \
  DEBUG_HTML_MAX_POINTS="${DEBUG_HTML_MAX_POINTS:-150000}" \
  OBJECTX_VIS_NUM_FRAMES="${OBJECTX_FINAL_BENCH_VIS_NUM_FRAMES:-12}" \
  OBJECTX_VIS_RENDER_SCALE="${OBJECTX_FINAL_BENCH_VIS_RENDER_SCALE:-0.25}" \
  OBJECTX_VIS_SKIP_GS=1 \
  OBJECTX_VIS_EXPORT_MESH=0 \
  OBJECTX_FEATURES3D_RESET_TMP=1 \
  bash "$REPO_ROOT/scripts/workflows/run_scene_profile.sh" "$profile_json" "$action" 2>&1 | tee -a "$action_log"
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

cleanup_scene() {
  local profile_name="$1"
  local scene_id="$2"
  local recon_root="$3"
  local pred_ready_root="$4"
  if [[ "$CLEANUP_AFTER_SCENE" != "1" || "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  echo "[cleanup] scene=$scene_id profile=$profile_name" >&2
  safe_rm "$recon_root"
  safe_rm "$pred_ready_root"
  safe_rm "$REPO_ROOT/vis/${scene_id}_joint.ply"
  safe_rm "$REPO_ROOT/vis/${scene_id}_slat.ply"
  safe_rm "/tmp/${USER}-objectx-feat3d"
  safe_rm "/tmp/${USER}-objectx-infer"
  safe_rm "/tmp/${USER}-objectx-voxelise"
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
}

cleanup_current_scene_on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${CURRENT_SCENE_ID:-}" ]]; then
    echo "[cleanup] exit trap rc=$rc scene=$CURRENT_SCENE_ID profile=$CURRENT_PROFILE_NAME" >&2
    cleanup_scene "$CURRENT_PROFILE_NAME" "$CURRENT_SCENE_ID" "$CURRENT_RECON_ROOT" "$CURRENT_PRED_READY_ROOT"
  fi
  exit "$rc"
}

write_aggregate() {
  "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/summarize_objectx_final_benchmark.py" \
    --selection-file "$SELECTION_FILE" \
    --out-root "$OUT_ROOT" \
    --report-threshold "${REPORT_THRESHOLD:-0.05}" \
    --secondary-threshold "${SECONDARY_REPORT_THRESHOLD:-0.10}"
}

prepare_selection
trap cleanup_current_scene_on_exit EXIT INT TERM HUP

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

  mapfile -t profile_info < <(generate_profile "$profile")
  profile_name="${profile_info[0]}"
  profile_json="${profile_info[1]}"
  scene_id="${profile_info[2]}"
  recon_root="${profile_info[3]}"
  pred_ready_root="${profile_info[4]}"
  scene_out="$OUT_ROOT/$profile_name/$scene_id"
  CURRENT_PROFILE_NAME="$profile_name"
  CURRENT_SCENE_ID="$scene_id"
  CURRENT_RECON_ROOT="$recon_root"
  CURRENT_PRED_READY_ROOT="$pred_ready_root"

  if [[ "$SKIP_COMPLETED" == "1" && -f "$scene_out/report_summary.md" ]]; then
    echo "[skip] complete scene=$scene_id profile=$profile_name" >&2
    complete_success=$((complete_success + 1))
    completed=$((completed + 1))
    continue
  fi

  echo "========== [$seen] $bucket scene=$scene_id profile=$profile_name score=$score complete=$complete_success/$TARGET_COMPLETE_RUNS =========="
  cleanup_scene "$profile_name" "$scene_id" "$recon_root" "$pred_ready_root"

  ok=1
  failed_action=""
  for action in "${ACTIONS[@]}"; do
    case "$action" in
      verify-sam2-masks)
        if [[ "$DRY_RUN" == "1" ]]; then
          echo "[sam2-guard:dry-run] would verify reconstruction SAM2 masks in $recon_root/files/sam2_projection"
        elif ! verify_sam2_masks "$scene_id" "$recon_root"; then
          ok=0
          failed_action="$action"
          break
        fi
        ;;
      ensure-predready-sam2-alias)
        if [[ "$DRY_RUN" == "1" ]]; then
          echo "[sam2-guard:dry-run] would alias pred-ready gt_projection to SAM2 masks"
        elif ! ensure_predready_sam2_alias "$scene_id" "$recon_root" "$pred_ready_root"; then
          ok=0
          failed_action="$action"
          break
        fi
        ;;
      *)
        if ! run_profile_action "$profile_name" "$profile_json" "$action" "$scene_id"; then
          ok=0
          failed_action="$action"
          break
        fi
        ;;
    esac
  done

  if [[ "$ok" == "1" ]]; then
    append_status "done" "$bucket" "$profile_name" "$scene_id" "ok"
    complete_success=$((complete_success + 1))
  else
    append_status "failed" "$bucket" "$profile_name" "$scene_id" "$failed_action log=$scene_out/action_logs/${failed_action}.log"
  fi

  cleanup_scene "$profile_name" "$scene_id" "$recon_root" "$pred_ready_root"
  CURRENT_PROFILE_NAME=""
  CURRENT_SCENE_ID=""
  CURRENT_RECON_ROOT=""
  CURRENT_PRED_READY_ROOT=""
  write_aggregate

  completed=$((completed + 1))
  if [[ "$ok" != "1" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit 1
  fi
done < "$SELECTION_FILE"

write_aggregate
echo "[done] selection=$SELECTION_FILE"
echo "[done] outputs=$OUT_ROOT"
