#!/bin/bash
# Submit the full SAM2Object pipeline (preprocess -> segment -> evaluate) for all
# scans in the .txt configured inside scripts/segmentation/run_pipeline.sh.
#   sbatch run_cluster/run_pipeline.sbatch
# Resumable: re-submitting skips scenes whose prediction .npy already exists.
#SBATCH --job-name=sam2object_pipeline
#SBATCH --output=/cluster/scratch/ealegret/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/ealegret/logs/%x_%j.err
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
# NOTE: 100 scenes may exceed 24h (the model reloads per scene). If it times out,
# just re-submit -- finished scenes are skipped. Or split the .txt across jobs.
set -euo pipefail

echo "===== JOB $SLURM_JOB_ID on $(hostname) at $(date) ====="
module load stack/.2024-06-silent gcc/12.2.0 cuda/12.4.1 python/3.10.13 || true

PROJECT_DIR="/cluster/home/ealegret/3d_vision_project"
mkdir -p /cluster/scratch/ealegret/logs
cd "$PROJECT_DIR"

# Select ScanNet (run_pipeline.sh defaults to 3RScan when these are unset).
export DATASET=ScanNet
export SCAN_LIST="$PROJECT_DIR/objectx_complete_scans_scannet.txt"
# Optional override of the read-only pre-extracted frames source:
# export SCANNET_POSED_SRC=/cluster/project/cvg/data/scannet/posed_images

# run_pipeline.sh activates the venvs itself (.sam2object for seg, .venvv for eval).
bash scripts/segmentation/run_pipeline.sh

echo "===== DONE at $(date) ====="
