#!/usr/bin/env bash
set -euo pipefail

# Rerun the worst ScanNet Pi3X geometry scenes with a capped prediction frame
# count during evaluation.  This tests whether the large Acc/Compl gap comes
# from evaluating all predicted frames while alignment/visible-GT use fewer.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"

SOURCE_GROUP="${SOURCE_GROUP:-scannet_pi3x_sequence_under300}"
SOURCE_SUMMARY="${SOURCE_SUMMARY:-$REPO_ROOT/evaluation/outputs/geometry/$SOURCE_GROUP/geometry_summary.json}"
SOURCE_SCENE_TABLE="${SOURCE_SCENE_TABLE:-$REPO_ROOT/configs/workflows/scene_profiles/scannet_under300_scenes.tsv}"

WORST_N="${WORST_N:-10}"
WORST_SORT_KEY="${WORST_SORT_KEY:-acc_compl_ratio}"
FRAME_CAP="${FRAME_CAP:-160}"
TEST_NAME="${TEST_NAME:-worst${WORST_N}_seqcap${FRAME_CAP}}"

SCANNET_SCENE_TABLE="${SCANNET_SCENE_TABLE:-$REPO_ROOT/configs/workflows/scene_profiles/scannet_${TEST_NAME}_scenes.tsv}"
PROFILE_LIST="${PROFILE_LIST:-$REPO_ROOT/configs/workflows/scene_profiles/scannet_${TEST_NAME}_profiles.txt}"
GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-scannet_pi3x_sequence_under300_${TEST_NAME}}"

export SCANNET_ROOT="${SCANNET_ROOT:-/work/courses/3dv/team35/pafina/scannet_under300_data}"
export SCANNET_WORK_ROOT="${SCANNET_WORK_ROOT:-/work/courses/3dv/team35/pafina}"
export OBJECTX_WORKFLOW_LOG_ROOT="${OBJECTX_WORKFLOW_LOG_ROOT:-$SCANNET_WORK_ROOT/logs}"
export SCANNET_REUSE_SELECTION=1
export SCANNET_PRUNE_STALE_PROFILES=0
export SCANNET_SCENE_TABLE
export PROFILE_LIST
export SCANNET_TARGET_SCENES="$WORST_N"
export GEOMETRY_EVAL_GROUP
export SEQUENCE_MAX_FRAMES="$FRAME_CAP"
export SCANNET_GEOMETRY_SEQUENCE_MAX_FRAMES="$FRAME_CAP"
export CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"

echo "[seqcap-test] source_summary=$SOURCE_SUMMARY"
echo "[seqcap-test] source_scene_table=$SOURCE_SCENE_TABLE"
echo "[seqcap-test] selected_scene_table=$SCANNET_SCENE_TABLE"
echo "[seqcap-test] profile_list=$PROFILE_LIST"
echo "[seqcap-test] group=$GEOMETRY_EVAL_GROUP"
echo "[seqcap-test] worst_n=$WORST_N sort=$WORST_SORT_KEY sequence_max_frames=$SEQUENCE_MAX_FRAMES"

"$PYTHON_BIN" - "$SOURCE_SUMMARY" "$SOURCE_SCENE_TABLE" "$SCANNET_SCENE_TABLE" "$WORST_N" "$WORST_SORT_KEY" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
source_table_path = Path(sys.argv[2])
out_table_path = Path(sys.argv[3])
worst_n = int(sys.argv[4])
sort_key = sys.argv[5]

rows = json.loads(summary_path.read_text(encoding="utf-8"))
if not rows:
    raise SystemExit(f"No rows in {summary_path}")

scene_meta = {}
if source_table_path.exists():
    for line in source_table_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("rank\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        scene_id = parts[1]
        frames = parts[2]
        raw_frames = parts[3] if len(parts) > 3 else frames
        source = parts[4] if len(parts) > 4 else "summary"
        scene_meta[scene_id] = (frames, raw_frames, source)

def num(row, key, default=math.nan):
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else default

def score(row):
    acc = num(row, "accuracy_mean_m")
    comp = num(row, "completeness_mean_m")
    fscore = num(row, "fscore_at_0.050m")
    precision = num(row, "precision_at_0.050m")
    if sort_key == "accuracy_mean":
        return acc
    if sort_key == "low_f1":
        return -fscore
    if sort_key == "low_precision":
        return -precision
    if sort_key == "acc_compl_ratio":
        return acc / max(comp, 1e-9)
    raise SystemExit(
        "Unknown WORST_SORT_KEY="
        + sort_key
        + " (expected acc_compl_ratio, accuracy_mean, low_f1, low_precision)"
    )

ranked = sorted(rows, key=score, reverse=True)[:worst_n]
out_table_path.parent.mkdir(parents=True, exist_ok=True)
lines = ["rank\tscene_id\tframes\traw_frames\tsource\ttest_score\tacc_mean_m\tcompl_mean_m\tf1_005m"]
for rank, row in enumerate(ranked, start=1):
    scene_id = row.get("scene_id")
    frames, raw_frames, source = scene_meta.get(scene_id, ("160", "160", "summary_fallback"))
    lines.append(
        "\t".join(
            [
                str(rank),
                str(scene_id),
                str(frames),
                str(raw_frames),
                str(source),
                f"{score(row):.9f}",
                f"{num(row, 'accuracy_mean_m'):.9f}",
                f"{num(row, 'completeness_mean_m'):.9f}",
                f"{num(row, 'fscore_at_0.050m'):.9f}",
            ]
        )
    )

out_table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[seqcap-test] selected {len(ranked)} scenes")
for line in lines[1:]:
    print("[seqcap-test] " + line)
PY

"$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/generate_scannet_pi3x_profiles.py" \
  --scene-table "$SCANNET_SCENE_TABLE" \
  --manifest "$PROFILE_LIST" \
  --max-scenes "$WORST_N"

bash "$REPO_ROOT/evaluation/geometry/run_scannet_pi3x_sequence_benchmark_under300.sh"
