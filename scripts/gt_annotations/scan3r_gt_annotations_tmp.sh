#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLSG_SPACE="${VLSG_SPACE:-$REPO_ROOT/dependencies/VLSG}"
SCRATCH_ROOT="${Data_ROOT_DIR:-/work/scratch/pafina/objectx-data-baseline}"
TMP_GT_ROOT="${TMP_GT_ROOT:-/tmp/${USER}-objectx-gt-anno}"
RESET_TMP="${RESET_TMP:-0}"

if [[ ! -d "$REPO_ROOT/3dv" ]]; then
  echo "Missing venv at $REPO_ROOT/3dv" >&2
  exit 1
fi

source "$REPO_ROOT/3dv/bin/activate"

if [[ "$RESET_TMP" == "1" ]]; then
  rm -rf "$TMP_GT_ROOT"
fi

mkdir -p "$TMP_GT_ROOT/scenes" "$TMP_GT_ROOT/files"
mkdir -p "$TMP_GT_ROOT/files/gt_projection/obj_id_pkl"
mkdir -p "$TMP_GT_ROOT/files/patch_anno/patch_anno_16_9"

python - <<'PY' "$SCRATCH_ROOT" "$TMP_GT_ROOT"
import json
import os
import sys
from pathlib import Path

scratch_root = Path(sys.argv[1])
tmp_root = Path(sys.argv[2])

with open(scratch_root / "files" / "3RScan.json") as f:
    scan_cfg = json.load(f)

# The official 2.3 scripts process validation + train.
needed = []
filtered_cfg = []
for entry in scan_cfg:
    if entry.get("type") not in ("train", "validation"):
        continue
    filtered_cfg.append(entry)
    needed.append(entry["reference"])
    needed.extend(scan["reference"] for scan in entry.get("scans", []))

needed = sorted(set(needed))

with open(tmp_root / "files" / "3RScan.json", "w") as f:
    json.dump(filtered_cfg, f)

for name in ["objects.json"]:
    src = scratch_root / "files" / name
    dst = tmp_root / "files" / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

with open(tmp_root / "tasks.tsv", "w") as f:
    for entry in filtered_cfg:
        split = entry["type"]
        f.write(f"{split}\t{entry['reference']}\n")
        for scan in entry.get("scans", []):
            f.write(f"{split}\t{scan['reference']}\n")

print(len(needed))
PY

TOTAL_SCANS="$(wc -l < "$TMP_GT_ROOT/tasks.tsv")"
echo "[stage] scanwise tmp fallback for $TOTAL_SCANS train+validation scans"

cd "$VLSG_SPACE"
export VLSG_SPACE
export Data_ROOT_DIR="$TMP_GT_ROOT"

mkdir -p "$SCRATCH_ROOT/files/gt_projection/obj_id_pkl"
mkdir -p "$SCRATCH_ROOT/files/patch_anno/patch_anno_16_9"

COUNT=0
while IFS=$'\t' read -r SPLIT SCAN_ID; do
  COUNT=$((COUNT + 1))
  SCRATCH_OBJ_PKL_GZ="$SCRATCH_ROOT/files/gt_projection/obj_id_pkl/$SCAN_ID.pkl.gz"
  SCRATCH_PATCH_PKL="$SCRATCH_ROOT/files/patch_anno/patch_anno_16_9/$SCAN_ID.pkl"

  if [[ -f "$SCRATCH_PATCH_PKL" && -f "$SCRATCH_OBJ_PKL_GZ" ]]; then
    PATCH_SIZE="$(stat -c%s "$SCRATCH_PATCH_PKL" 2>/dev/null || echo 0)"
    if [[ "$PATCH_SIZE" -gt 5 ]]; then
      if (( COUNT % 25 == 0 || COUNT == TOTAL_SCANS )); then
        echo "[2.3] $COUNT/$TOTAL_SCANS (skip existing)"
      fi
      continue
    fi
  fi

  SRC_SCAN_DIR="$SCRATCH_ROOT/scenes/$SCAN_ID"
  TMP_SCAN_DIR="$TMP_GT_ROOT/scenes/$SCAN_ID"
  rm -rf "$TMP_SCAN_DIR"
  mkdir -p "$TMP_SCAN_DIR"

  ln -sfn "$SRC_SCAN_DIR/data.npy" "$TMP_SCAN_DIR/data.npy"
  ln -sfn "$SRC_SCAN_DIR/labels.instances.annotated.v2.ply" \
    "$TMP_SCAN_DIR/labels.instances.annotated.v2.ply"
  mkdir -p "$TMP_SCAN_DIR/sequence"
  unzip -qo "$SRC_SCAN_DIR/sequence.zip" -d "$TMP_SCAN_DIR/sequence"

  echo "[2.3] $COUNT/$TOTAL_SCANS $SPLIT $SCAN_ID"
  python "$REPO_ROOT/scripts/gt_annotations/run_single_scan_gt_anno.py" \
    --vlsG-space "$VLSG_SPACE" \
    --data-root "$TMP_GT_ROOT" \
    --split "$SPLIT" \
    --scan-id "$SCAN_ID" \
    --config "$VLSG_SPACE/preprocessing/gt_anno_2D/gt_anno.yaml"

  python - <<'PY' "$TMP_GT_ROOT/files/gt_projection/obj_id_pkl/$SCAN_ID.pkl" "$SCRATCH_OBJ_PKL_GZ"
import gzip
import pickle
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    data = pickle.load(f)

compressed = {}
for key, value in data.items():
    arr = np.asarray(value)
    if arr.dtype.kind in "iu" and arr.min() >= 0 and arr.max() <= np.iinfo(np.uint16).max:
        arr = arr.astype(np.uint16, copy=False)
    compressed[key] = arr

with gzip.open(dst, "wb", compresslevel=1) as f:
    pickle.dump(compressed, f, protocol=pickle.HIGHEST_PROTOCOL)
PY
  rsync -a "$TMP_GT_ROOT/files/patch_anno/patch_anno_16_9/$SCAN_ID.pkl" \
    "$SCRATCH_ROOT/files/patch_anno/patch_anno_16_9/" 2>/dev/null || true

  rm -rf "$TMP_GT_ROOT/scenes/$SCAN_ID/sequence"
  rm -rf "$TMP_GT_ROOT/files/gt_projection/color/$SCAN_ID"
  rm -rf "$TMP_GT_ROOT/files/gt_projection/obj_id/$SCAN_ID"
  rm -f "$TMP_GT_ROOT/files/gt_projection/obj_id_pkl/$SCAN_ID.pkl"
  rm -f "$TMP_GT_ROOT/files/patch_anno/patch_anno_16_9/$SCAN_ID.pkl"
done < "$TMP_GT_ROOT/tasks.tsv"

rm -rf "$TMP_GT_ROOT/files/gt_projection/color" "$TMP_GT_ROOT/files/gt_projection/obj_id"

echo "[done] tmp root kept at $TMP_GT_ROOT"
