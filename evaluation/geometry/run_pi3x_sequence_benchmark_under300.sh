#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROFILE_LIST="${PROFILE_LIST:-$REPO_ROOT/configs/workflows/scene_profiles/pi3x_under300_profiles.txt}"
export GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-pi3x_sequence_under300}"

exec bash "$REPO_ROOT/evaluation/geometry/run_pi3x_sequence_benchmark_15.sh"
