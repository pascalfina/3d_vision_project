#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=64G
#SBATCH -J scannet-must3r-geom300
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-scannet-must3r-geom300-%j.out
#SBATCH -e /work/scratch/pafina/object-x/debug/slurm-scannet-must3r-geom300-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug"

# Do not inherit an old interactive SCANNET_ROOT accidentally. Use
# SCANNET_ROOT_OVERRIDE only when intentionally changing the persistent cache.
export SCANNET_ROOT="${SCANNET_ROOT_OVERRIDE:-/work/courses/3dv/team35/pafina/scannet_under300_data}"
export SCANNET_WORK_ROOT="${SCANNET_WORK_ROOT_OVERRIDE:-/work/courses/3dv/team35/pafina}"
export OBJECTX_WORKFLOW_LOG_ROOT="${OBJECTX_WORKFLOW_LOG_ROOT_OVERRIDE:-$SCANNET_WORK_ROOT/logs}"
export SCANNET_TARGET_SCENES="${SCANNET_TARGET_SCENES:-300}"
export SCANNET_SELECTION_MAX_FRAMES="${SCANNET_SELECTION_MAX_FRAMES:-300}"
export SCANNET_SELECTION_MODE="${SCANNET_SELECTION_MODE:-cap}"
export SCANNET_SELECTION_ORDER="${SCANNET_SELECTION_ORDER:-shortest}"
export SCANNET_EXPORT_MAX_FRAMES="${SCANNET_EXPORT_MAX_FRAMES:-$SCANNET_SELECTION_MAX_FRAMES}"
export SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-1}"
export SCANNET_AUTO_FRAME_SKIP_FOR_MAX="${SCANNET_AUTO_FRAME_SKIP_FOR_MAX:-1}"
export SCANNET_DELETE_SENS_AFTER_EXPORT="${SCANNET_DELETE_SENS_AFTER_EXPORT:-1}"
export SCANNET_DIRECT_DOWNLOAD="${SCANNET_DIRECT_DOWNLOAD:-1}"
export SCANNET_DOWNLOAD_TYPES="${SCANNET_DOWNLOAD_TYPES:-.txt _vh_clean_2.ply .sens}"
export SCANNET_PURGE_PREPARED_AFTER_SCENE="${SCANNET_PURGE_PREPARED_AFTER_SCENE:-1}"
export SCANNET_PURGE_PREPARED_AFTER_FAILED="${SCANNET_PURGE_PREPARED_AFTER_FAILED:-1}"
export SCANNET_PURGE_PREPARED_ON_SKIP="${SCANNET_PURGE_PREPARED_ON_SKIP:-1}"
export GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-scannet_must3r_sequence_under300}"
export SCANNET_REUSE_SELECTION="${SCANNET_REUSE_SELECTION:-1}"
export FORCE="${FORCE:-0}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "SCANNET_ROOT=$SCANNET_ROOT"
echo "SCANNET_WORK_ROOT=$SCANNET_WORK_ROOT"
echo "OBJECTX_WORKFLOW_LOG_ROOT=$OBJECTX_WORKFLOW_LOG_ROOT"
echo "SCANNET_TARGET_SCENES=$SCANNET_TARGET_SCENES"
echo "SCANNET_SELECTION_MAX_FRAMES=$SCANNET_SELECTION_MAX_FRAMES"
echo "SCANNET_SELECTION_MODE=$SCANNET_SELECTION_MODE"
echo "SCANNET_SELECTION_ORDER=$SCANNET_SELECTION_ORDER"
echo "SCANNET_EXPORT_MAX_FRAMES=$SCANNET_EXPORT_MAX_FRAMES"
echo "SCANNET_DIRECT_DOWNLOAD=$SCANNET_DIRECT_DOWNLOAD"
echo "SCANNET_DOWNLOAD_TYPES=$SCANNET_DOWNLOAD_TYPES"
echo "SCANNET_PURGE_PREPARED_AFTER_SCENE=$SCANNET_PURGE_PREPARED_AFTER_SCENE"
echo "SCANNET_PURGE_PREPARED_AFTER_FAILED=$SCANNET_PURGE_PREPARED_AFTER_FAILED"
echo "SCANNET_PURGE_PREPARED_ON_SKIP=$SCANNET_PURGE_PREPARED_ON_SKIP"
echo "GEOMETRY_EVAL_GROUP=$GEOMETRY_EVAL_GROUP"
echo "SCANNET_REUSE_SELECTION=$SCANNET_REUSE_SELECTION"
echo "FORCE=$FORCE"
echo "SKIP_COMPLETED=$SKIP_COMPLETED"
echo "CLEANUP_AFTER_SCENE=$CLEANUP_AFTER_SCENE"
echo "CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"
echo "WRITE_DEBUG_HTML=$WRITE_DEBUG_HTML"
echo "START_AT=${START_AT:-}"
echo "LIMIT=${LIMIT:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"

nvidia-smi || true
mkdir -p "$SCANNET_ROOT" "$OBJECTX_WORKFLOW_LOG_ROOT"

srun --ntasks=1 "$REPO_DIR/.venv_objx/bin/python" - <<'PY'
import os
import sys
import torch

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_JOB_GPUS=", os.environ.get("SLURM_JOB_GPUS"))
print("torch=", torch.__version__, "torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA is not available inside the Slurm step; aborting before benchmark.")
capability = torch.cuda.get_device_capability(0)
print("cuda_device=", torch.cuda.get_device_name(0))
print("cuda_capability=", capability)
if capability < (7, 0):
    sys.exit(
        "Assigned GPU is too old for this PyTorch build. "
        f"Need sm_70+, got sm_{capability[0]}{capability[1]}."
    )
bf16_supported = torch.cuda.is_bf16_supported()
print("bf16_supported=", bf16_supported)
if not bf16_supported:
    sys.exit("Assigned GPU does not support bf16; MUSt3R requires a bf16-capable GPU.")
PY

srun --ntasks=1 bash evaluation/geometry/run_scannet_must3r_sequence_benchmark_under300.sh
