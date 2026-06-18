#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -J scannet-prepare-final100
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-scannet-prepare-final100-%j.out
#SBATCH -e /work/scratch/pafina/object-x/debug/slurm-scannet-prepare-final100-%j.err
#SBATCH --open-mode=append

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"
cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug"

export SELECTION_FILE="${SELECTION_FILE:-$REPO_DIR/evaluation/outputs/geometry/scannet_objectx_final_100_pi3x_samobject/selected_100.tsv}"
export SCANNET_ROOT="${SCANNET_ROOT:-/work/courses/3dv/team35/pafina/scannet_under300_data}"
export SCANNET_DOWNLOAD_TYPES="${SCANNET_DOWNLOAD_TYPES:-.txt _vh_clean_2.ply .sens}"
export SCANNET_DOWNLOAD_BACKEND="${SCANNET_DOWNLOAD_BACKEND:-auto}"
export SCANNET_DELETE_SENS_AFTER_EXPORT="${SCANNET_DELETE_SENS_AFTER_EXPORT:-1}"
export SCANNET_EXPORT_MAX_FRAMES="${SCANNET_EXPORT_MAX_FRAMES:-300}"
export SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-1}"
export SCANNET_AUTO_FRAME_SKIP_FOR_MAX="${SCANNET_AUTO_FRAME_SKIP_FOR_MAX:-1}"
export SCANNET_DIRECT_DOWNLOAD="${SCANNET_DIRECT_DOWNLOAD:-1}"
export SKIP_PREPARED="${SKIP_PREPARED:-1}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export LIMIT="${LIMIT:-3}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "SELECTION_FILE=$SELECTION_FILE"
echo "SCANNET_ROOT=$SCANNET_ROOT"
echo "SCANNET_DOWNLOAD_TYPES=$SCANNET_DOWNLOAD_TYPES"
echo "SCANNET_DOWNLOAD_BACKEND=$SCANNET_DOWNLOAD_BACKEND"
echo "SCANNET_DELETE_SENS_AFTER_EXPORT=$SCANNET_DELETE_SENS_AFTER_EXPORT"
echo "SCANNET_EXPORT_MAX_FRAMES=$SCANNET_EXPORT_MAX_FRAMES"
echo "START_AT=${START_AT:-}"
echo "LIMIT=${LIMIT:-}"

bash "$REPO_DIR/evaluation/geometry/prepare_scannet_selection.sh"
