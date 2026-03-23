#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${VLSG_TRAINING_OUT_DIR:-$REPO_ROOT/runs/step2_slat_completion_${TIMESTAMP}}"

source "$REPO_ROOT/scripts/activate_objectx_env.sh"

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/runs" "$OUTPUT_DIR"
python src/trainval/train_step2_slat_completion.py \
  --output-dir "$OUTPUT_DIR" \
  "$@"
