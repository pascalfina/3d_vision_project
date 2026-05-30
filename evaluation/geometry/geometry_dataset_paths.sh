#!/usr/bin/env bash
# Shared GT path resolver for geometry evaluation.
#
# Supported datasets:
#   GEOMETRY_DATASET=3rscan   baseline/scenes/<scan>/mesh.refined.v2.obj + sequence.zip
#   GEOMETRY_DATASET=scannet  ScanNet scan dirs with <scan>_vh_clean*.ply + data/{color,depth,pose,intrinsic}

geometry_dataset_normalize() {
  local dataset="${1:-3rscan}"
  dataset="${dataset,,}"
  case "$dataset" in
    3rscan|3rscan-v2|scan3r)
      printf '3rscan\n'
      ;;
    scannet|scannetv2|scannet-v2)
      printf 'scannet\n'
      ;;
    *)
      echo "[geometry-dataset] unknown GEOMETRY_DATASET=$dataset (expected 3rscan or scannet)" >&2
      return 2
      ;;
  esac
}

geometry_first_existing_file() {
  local path
  for path in "$@"; do
    if [[ -n "$path" && -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

geometry_first_existing_dir() {
  local path
  for path in "$@"; do
    if [[ -n "$path" && -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

geometry_scannet_scene_dirs() {
  local root="$1"
  local scan_id="$2"
  printf '%s\n' \
    "$root/scenes/$scan_id" \
    "$root/scans/$scan_id" \
    "$root/$scan_id"
}

geometry_resolve_gt_paths() {
  GEOMETRY_DATASET="$(geometry_dataset_normalize "${GEOMETRY_DATASET:-${DATASET:-3rscan}}")"
  export GEOMETRY_DATASET

  if [[ -z "${SCAN_ID:-}" ]]; then
    echo "[geometry-dataset] SCAN_ID is required" >&2
    return 2
  fi

  if [[ "$GEOMETRY_DATASET" == "3rscan" ]]; then
    if [[ -z "${GT_MESH:-}" && -n "${BASELINE_ROOT:-}" ]]; then
      GT_MESH="$BASELINE_ROOT/scenes/$SCAN_ID/mesh.refined.v2.obj"
    fi
    if [[ -z "${GT_SEQUENCE_ZIP:-}" && -n "${BASELINE_ROOT:-}" ]]; then
      GT_SEQUENCE_ZIP="$BASELINE_ROOT/scenes/$SCAN_ID/sequence.zip"
    fi
    export GT_MESH GT_SEQUENCE_ZIP
    return 0
  fi

  if [[ "$GEOMETRY_DATASET" == "scannet" ]]; then
    if [[ -n "${BASELINE_ROOT:-}" && -f "$BASELINE_ROOT" && "$BASELINE_ROOT" == *.zip ]]; then
      echo "[geometry-dataset] BASELINE_ROOT points to a zip file: $BASELINE_ROOT" >&2
      echo "[geometry-dataset] ScanNetDownload.zip contains the downloader/SDK, not scan geometry." >&2
      echo "[geometry-dataset] Download/export scans first, then set BASELINE_ROOT to the directory containing scans/<scan_id> or scenes/<scan_id>." >&2
      return 2
    fi

    if [[ -z "${BASELINE_ROOT:-}" && -z "${GT_MESH:-}" ]]; then
      echo "[geometry-dataset] ScanNet eval needs BASELINE_ROOT or explicit GT_MESH" >&2
      return 2
    fi

    if [[ -z "${GT_MESH:-}" ]]; then
      local scene_dir mesh
      while IFS= read -r scene_dir; do
        mesh="$(geometry_first_existing_file \
          "$scene_dir/${SCAN_ID}_vh_clean_2.ply" \
          "$scene_dir/${SCAN_ID}_vh_clean.ply" \
          "$scene_dir/${SCAN_ID}_vh_clean_2.labels.ply" \
          "$scene_dir/mesh.refined.v2.obj" \
          "$scene_dir/labels.instances.annotated.v2.ply" || true)"
        if [[ -n "$mesh" ]]; then
          GT_MESH="$mesh"
          break
        fi
      done < <(geometry_scannet_scene_dirs "$BASELINE_ROOT" "$SCAN_ID")
    fi

    if [[ -z "${GT_SEQUENCE_DIR:-}" && -z "${GT_SEQUENCE_ZIP:-}" ]]; then
      local scene_dir sequence_dir
      while IFS= read -r scene_dir; do
        sequence_dir="$(geometry_first_existing_dir \
          "$scene_dir/data" \
          "$scene_dir" || true)"
        if [[ -n "$sequence_dir" && -d "$sequence_dir/pose" && -d "$sequence_dir/intrinsic" ]]; then
          GT_SEQUENCE_DIR="$sequence_dir"
          break
        fi
      done < <(geometry_scannet_scene_dirs "$BASELINE_ROOT" "$SCAN_ID")
    fi

    export GT_MESH GT_SEQUENCE_DIR GT_SEQUENCE_ZIP
    return 0
  fi
}
