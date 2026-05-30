#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_objx/bin/python}"

SCAN_ID="${SCAN_ID:?Set SCAN_ID, e.g. scene0000_00}"
SCANNET_ROOT="${SCANNET_ROOT:-${BASELINE_ROOT:-/work/scratch/pafina/scannet_data}}"
SCANNET_DOWNLOAD_ZIP="${SCANNET_DOWNLOAD_ZIP:-/work/courses/3dv/team35/pafina/ScanNetDownload.zip}"
SCANNET_TOOLS_DIR="${SCANNET_TOOLS_DIR:-/work/scratch/pafina/scannet_download_tools}"
SCANNET_DOWNLOAD_TYPES="${SCANNET_DOWNLOAD_TYPES:-.txt _vh_clean_2.ply .sens}"
SCANNET_DIRECT_DOWNLOAD="${SCANNET_DIRECT_DOWNLOAD:-1}"
SCANNET_DOWNLOAD_TIMEOUT="${SCANNET_DOWNLOAD_TIMEOUT:-180}"
SCANNET_DOWNLOAD_RETRIES="${SCANNET_DOWNLOAD_RETRIES:-3}"
SCANNET_CLEAN_STALE_TMP="${SCANNET_CLEAN_STALE_TMP:-1}"
SCANNET_EXPORT_SENS="${SCANNET_EXPORT_SENS:-1}"
SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-1}"
SCANNET_MAX_FRAMES="${SCANNET_MAX_FRAMES:-0}"
SCANNET_AUTO_FRAME_SKIP_FOR_MAX="${SCANNET_AUTO_FRAME_SKIP_FOR_MAX:-1}"
SCANNET_LINK_SCENES="${SCANNET_LINK_SCENES:-1}"
SCANNET_WRITE_COMPAT_SEQUENCE="${SCANNET_WRITE_COMPAT_SEQUENCE:-1}"
SCANNET_DELETE_SENS_AFTER_EXPORT="${SCANNET_DELETE_SENS_AFTER_EXPORT:-0}"

if [[ "$SCANNET_DIRECT_DOWNLOAD" != "1" && ! -f "$SCANNET_DOWNLOAD_ZIP" ]]; then
  echo "[scannet-prepare] missing ScanNet downloader zip: $SCANNET_DOWNLOAD_ZIP" >&2
  exit 2
fi

if [[ "$SCANNET_DIRECT_DOWNLOAD" != "1" && ! -f "$SCANNET_TOOLS_DIR/ScanNetDownload/ScanDownloader.py" ]]; then
  mkdir -p "$SCANNET_TOOLS_DIR"
  unzip -q -o "$SCANNET_DOWNLOAD_ZIP" -d "$SCANNET_TOOLS_DIR"
fi

DOWNLOADER="$SCANNET_TOOLS_DIR/ScanNetDownload/ScanDownloader.py"
SCAN_DIR="$SCANNET_ROOT/scans/$SCAN_ID"
mkdir -p "$SCANNET_ROOT/scans"
mkdir -p "$SCAN_DIR"

if [[ "$SCANNET_CLEAN_STALE_TMP" == "1" ]]; then
  find "$SCAN_DIR" -maxdepth 1 -type f \( -name 'tmp*' -o -name '*.part' \) -delete
fi

echo "[scannet-prepare] scan=$SCAN_ID root=$SCANNET_ROOT" >&2
for file_type in $SCANNET_DOWNLOAD_TYPES; do
  echo "[scannet-prepare] downloading type=$file_type" >&2
  if [[ "$SCANNET_DIRECT_DOWNLOAD" == "1" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/download_scannet_file.py" \
      --scan-id "$SCAN_ID" \
      --file-type "$file_type" \
      --out-dir "$SCAN_DIR" \
      --timeout "$SCANNET_DOWNLOAD_TIMEOUT" \
      --retries "$SCANNET_DOWNLOAD_RETRIES"
  else
    "$PYTHON_BIN" "$DOWNLOADER" -o "$SCANNET_ROOT" --id "$SCAN_ID" --type "$file_type"
  fi
done

if [[ "$SCANNET_EXPORT_SENS" == "1" ]]; then
  SENS_FILE="$SCAN_DIR/$SCAN_ID.sens"
  if [[ ! -f "$SENS_FILE" ]]; then
    echo "[scannet-prepare] missing .sens file after download: $SENS_FILE" >&2
    exit 2
  fi
  export_args=(
    "$PYTHON_BIN" "$REPO_ROOT/evaluation/geometry/export_scannet_sens.py"
    --filename "$SENS_FILE" \
    --output-path "$SCAN_DIR/data" \
    --frame-skip "$SCANNET_FRAME_SKIP" \
    --max-frames "$SCANNET_MAX_FRAMES"
  )
  if [[ "$SCANNET_WRITE_COMPAT_SEQUENCE" == "1" ]]; then
    export_args+=(--compat-sequence-path "$SCAN_DIR/sequence")
  fi
  if [[ "$SCANNET_AUTO_FRAME_SKIP_FOR_MAX" == "1" ]]; then
    export_args+=(--auto-frame-skip-for-max)
  fi
  "${export_args[@]}"
  if [[ "$SCANNET_DELETE_SENS_AFTER_EXPORT" == "1" ]]; then
    rm -f -- "$SENS_FILE"
    echo "[scannet-prepare] deleted exported .sens to save quota: $SENS_FILE" >&2
  fi
fi

if [[ "$SCANNET_LINK_SCENES" == "1" ]]; then
  mkdir -p "$SCANNET_ROOT/scenes"
  ln -sfn "../scans/$SCAN_ID" "$SCANNET_ROOT/scenes/$SCAN_ID"
fi

echo "[scannet-prepare] ready:" >&2
find "$SCAN_DIR" -maxdepth 3 -type f | sed -n '1,80p' >&2
