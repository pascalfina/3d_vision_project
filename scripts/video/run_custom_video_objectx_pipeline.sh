#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PROFILE_MANIFEST="${PROFILE_MANIFEST:-configs/workflows/scene_profiles/custom_video_horizontal_profiles.txt}"
ACTION_LIST="${ACTION_LIST:-must3r pi3x samobject voxelise build-pred-ready features3d slat u3dgs}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
START_AT="${START_AT:-}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

export SAMOBJECT_CHECKPOINT="${SAMOBJECT_CHECKPOINT:-$REPO_ROOT/models/sam2ckpt/sam2_hiera_base_plus.pt}"
export SAMOBJECT_MODEL_CFG="${SAMOBJECT_MODEL_CFG:-sam2_hiera_b+.yaml}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "$PROFILE_MANIFEST" ]]; then
  echo "Missing profile manifest: $PROFILE_MANIFEST" >&2
  exit 1
fi

mapfile -t PROFILES < <(grep -v '^[[:space:]]*$' "$PROFILE_MANIFEST")
if [[ ${#PROFILES[@]} -eq 0 ]]; then
  echo "No profiles found in $PROFILE_MANIFEST" >&2
  exit 1
fi

echo "[custom-video-pipeline] manifest=$PROFILE_MANIFEST"
echo "[custom-video-pipeline] actions=$ACTION_LIST"
echo "[custom-video-pipeline] continue_on_error=$CONTINUE_ON_ERROR start_at=$START_AT limit=$LIMIT dry_run=$DRY_RUN"

started=0
processed=0
for profile in "${PROFILES[@]}"; do
  if [[ -n "$START_AT" && "$started" == "0" ]]; then
    if [[ "$profile" != "$START_AT" ]]; then
      echo "[skip-before-start] $profile"
      continue
    fi
  fi
  started=1
  processed=$((processed + 1))
  if [[ -n "$LIMIT" && "$processed" -gt "$LIMIT" ]]; then
    break
  fi

  echo "========== [$processed/${#PROFILES[@]}] $profile =========="
  for action in $ACTION_LIST; do
    echo "[run] $profile $action"
    cmd=(bash scripts/workflows/run_scene_profile.sh "$profile" "$action")
    if [[ "$DRY_RUN" == "1" ]]; then
      cmd+=(--dry-run)
    fi
    if ! "${cmd[@]}"; then
      echo "[error] $profile $action failed" >&2
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit 1
      fi
      break
    fi
  done
done
