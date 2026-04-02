#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "$REPO_ROOT/scripts/activate_objectx_env.sh"

exec python "$REPO_ROOT/scripts/workflows/run_scene_profile.py" \
  --repo-root "$REPO_ROOT" \
  "$@"
