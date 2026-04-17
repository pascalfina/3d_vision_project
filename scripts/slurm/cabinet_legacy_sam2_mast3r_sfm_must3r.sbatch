#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -J cabinet-mast3r
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-cabinet-mast3r-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"
PROFILE_NAME="${PROFILE_NAME:-cabinet_legacy_sam2_mast3r_sfm_hybrid}"
ACTION="${ACTION:-must3r}"
PIPELINE_LOG="${PIPELINE_LOG:-$REPO_DIR/debug/debug_cabinet_mast3r_sfm.log}"
MAST3R_CACHE_DIR="${OBJECTX_MAST3R_SFM_CACHE_DIR:-/work/courses/3dv/team35/pafina/objectx-cache/mast3r_sfm_cabinet}"
RESET_MAST3R_CACHE="${RESET_MAST3R_CACHE:-0}"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug"
source scripts/activate_objectx_env.sh

export OBJECTX_MAST3R_SFM_CACHE_DIR="$MAST3R_CACHE_DIR"
mkdir -p "$OBJECTX_MAST3R_SFM_CACHE_DIR"

if [[ "$RESET_MAST3R_CACHE" == "1" ]]; then
    rm -rf "$OBJECTX_MAST3R_SFM_CACHE_DIR"
    mkdir -p "$OBJECTX_MAST3R_SFM_CACHE_DIR"
fi

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "PROFILE_NAME=$PROFILE_NAME"
echo "ACTION=$ACTION"
echo "TMPDIR=$TMPDIR"
echo "OBJECTX_MAST3R_SFM_CACHE_DIR=$OBJECTX_MAST3R_SFM_CACHE_DIR"
echo "RESET_MAST3R_CACHE=$RESET_MAST3R_CACHE"
echo "PIPELINE_LOG=$PIPELINE_LOG"

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

python scripts/workflows/run_scene_profile.py "$PROFILE_NAME" "$ACTION" \
  --repo-root "$REPO_DIR" \
  2>&1 | tee "$PIPELINE_LOG"
