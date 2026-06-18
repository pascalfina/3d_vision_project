#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export SCAN_ID="${SCAN_ID:-5341b7e3-8a66-2cdd-8709-66a2159f0017}"
export METHOD_NAME="${METHOD_NAME:-oven_pi3x}"
export PRED_ROOT="${PRED_ROOT:-/work/scratch/pafina/objectx-data-fullscene-oven-pi3x}"
export BASELINE_ROOT="${BASELINE_ROOT:-/work/scratch/pafina/objectx-data-baseline}"
export GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-final}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/evaluation/outputs/geometry/$GEOMETRY_EVAL_GROUP/oven_pi3x/$SCAN_ID}"

exec bash "$REPO_ROOT/evaluation/geometry/run_pi3x_geometry_eval.sh"
