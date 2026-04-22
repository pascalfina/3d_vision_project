#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/cluster/home/ealegret/3d_vision_project"

source "$REPO_ROOT/scripts/activate_objectx_env.sh"

exec python scripts/workflows/run_scene_profile.py \
  --repo-root "$REPO_ROOT" \
  "$@"
