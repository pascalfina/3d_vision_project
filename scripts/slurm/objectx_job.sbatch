#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -J objectx
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"
RUN_SCRIPT="${1:-scripts/train_val/train_latent_autoencoder.sh}"
if [[ $# -gt 0 ]]; then
    shift
fi
RUN_ARGS=("$@")

if [[ -z "${DATA_ROOT_DIR:-}" ]]; then
    echo "DATA_ROOT_DIR is not set." >&2
    echo "Example:" >&2
    echo "  sbatch --export=ALL,DATA_ROOT_DIR=/work/scratch/pafina/objectx-data,SCRATCH=/work/scratch/pafina scripts/slurm/objectx_job.sbatch" >&2
    echo "  sbatch --export=ALL,DATA_ROOT_DIR=/work/scratch/pafina/objectx-data,SCRATCH=/work/scratch/pafina scripts/slurm/objectx_job.sbatch scripts/train_val/train.sh" >&2
    exit 1
fi

export SCRATCH="${SCRATCH:-/work/scratch/pafina}"

cd "$REPO_DIR"
source scripts/activate_objectx_env.sh

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "DATA_ROOT_DIR=$DATA_ROOT_DIR"
echo "SCRATCH=$SCRATCH"
echo "RUN_SCRIPT=$RUN_SCRIPT"

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"

bash "$RUN_SCRIPT" "${RUN_ARGS[@]}"
