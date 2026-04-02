#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH -J objx-bright
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-objx-bright-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/work/scratch/pafina/objectx-data-predseg-objects5}"
SPLIT="${SPLIT:-val}"

cd "$REPO_DIR"
source scripts/activate_objectx_env.sh

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "DATA_ROOT_DIR=$DATA_ROOT_DIR"
echo "SPLIT=$SPLIT"

python -c "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"

bash scripts/inference/experiments/rerender_object_level_bright.sh
