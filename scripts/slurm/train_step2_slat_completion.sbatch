#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH -J step2-slat
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-step2-slat-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/scripts/train_val/train_step2_slat_completion.json}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-/work/scratch/pafina/step2_slat_completion_${TIMESTAMP}}"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug" "$OUTPUT_DIR"
source scripts/activate_objectx_env.sh

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"

bash scripts/train_val/train_step2_slat_completion.sh \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
