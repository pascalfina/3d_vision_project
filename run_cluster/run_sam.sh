#!/bin/bash
#SBATCH --job-name=run_sam3
#SBATCH --output=/cluster/scratch/ealegret/3dv/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/ealegret/3dv/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1

set -euo pipefail

# Inicialize the modules
module purge
module load stack/.2024-06-silent
module load gcc/12.2.0
module load cuda/12.4.1
module load python/3.9.18

# Activate venv la venv
source /cluster/home/ealegret/3d_vision_project/3dv/bin/activate

# Go to workdir
cd /cluster/home/ealegret/3d_vision_project/
export Data_ROOT_DIR=/cluster/project/cvg/data/3RScan

# Run experiments
echo "=== Job started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Current dir: $(pwd)"
echo "Arguments passed to sbatch: $@"
echo "Submitting training with override if provided"

bash scripts/segmentation/segmentation_sam.sh "$@" 