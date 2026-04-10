#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -J cabinet-sam2-seg
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-cabinet-sam2-seg-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"
PROFILE_NAME="${PROFILE_NAME:-cabinet_legacy_sam2_hybrid}"
ACTION="${ACTION:-segment-inputs}"
PIPELINE_LOG="${PIPELINE_LOG:-$REPO_DIR/debug/debug_cabinet_segment_inputs_sam2.log}"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug"
source scripts/activate_objectx_env.sh

export OBJECTX_SAM2_TMP_ROOT="${OBJECTX_SAM2_TMP_ROOT:-$TMPDIR/sam2-cache}"
export OBJECTX_SAM2_PROB_CACHE_DTYPE="${OBJECTX_SAM2_PROB_CACHE_DTYPE:-float32}"
mkdir -p "$OBJECTX_SAM2_TMP_ROOT"
rm -rf "$OBJECTX_SAM2_TMP_ROOT"/*

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "PROFILE_NAME=$PROFILE_NAME"
echo "ACTION=$ACTION"
echo "TMPDIR=$TMPDIR"
echo "OBJECTX_SAM2_TMP_ROOT=$OBJECTX_SAM2_TMP_ROOT"
echo "OBJECTX_SAM2_PROB_CACHE_DTYPE=$OBJECTX_SAM2_PROB_CACHE_DTYPE"
echo "PIPELINE_LOG=$PIPELINE_LOG"

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"

python scripts/workflows/run_scene_profile.py "$PROFILE_NAME" "$ACTION" \
  --repo-root "$REPO_DIR" \
  2>&1 | tee "$PIPELINE_LOG"
