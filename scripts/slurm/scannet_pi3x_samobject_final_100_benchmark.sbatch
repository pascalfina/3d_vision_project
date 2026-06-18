#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --exclude=studgpu-node09,studgpu-node25
#SBATCH --mem=64G
#SBATCH -J scannet-pi3x-samobj-final100
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-scannet-pi3x-samobject-final100-%j.out
#SBATCH -e /work/scratch/pafina/object-x/debug/slurm-scannet-pi3x-samobject-final100-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug"

# Use *_OVERRIDE only when intentionally moving the persistent ScanNet cache or
# work roots. This avoids accidentally inheriting stale interactive exports.
export SCANNET_ROOT="${SCANNET_ROOT_OVERRIDE:-/work/courses/3dv/team35/pafina/scannet_under300_data}"
export SCANNET_WORK_ROOT="${SCANNET_WORK_ROOT_OVERRIDE:-/work/courses/3dv/team35/pafina}"
export SCANNET_SAMOBJECT_ROOT="${SCANNET_SAMOBJECT_ROOT_OVERRIDE:-/work/scratch/pafina/sam2object_scannet}"
export OBJECTX_WORKFLOW_LOG_ROOT="${OBJECTX_WORKFLOW_LOG_ROOT_OVERRIDE:-$SCANNET_WORK_ROOT/logs}"

export SOURCE_GROUP="${SOURCE_GROUP:-scannet_pi3x_sequence_under300}"
export GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-scannet_objectx_final_100_pi3x_samobject}"
export RUN_SAM2_MUST3R_AFTER="${RUN_SAM2_MUST3R_AFTER:-1}"
export SAM2_MUST3R_GEOMETRY_EVAL_GROUP="${SAM2_MUST3R_GEOMETRY_EVAL_GROUP:-scannet_objectx_final_100_sam2_must3r}"
export SCANNET_TARGET_SCENES="${SCANNET_TARGET_SCENES:-300}"
export SCANNET_SELECTION_MAX_FRAMES="${SCANNET_SELECTION_MAX_FRAMES:-300}"
export SCANNET_SELECTION_MODE="${SCANNET_SELECTION_MODE:-cap}"
export SCANNET_SELECTION_ORDER="${SCANNET_SELECTION_ORDER:-shortest}"
export SCANNET_EXPORT_MAX_FRAMES="${SCANNET_EXPORT_MAX_FRAMES:-$SCANNET_SELECTION_MAX_FRAMES}"
export SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-1}"
export SCANNET_AUTO_FRAME_SKIP_FOR_MAX="${SCANNET_AUTO_FRAME_SKIP_FOR_MAX:-1}"
export SCANNET_DELETE_SENS_AFTER_EXPORT="${SCANNET_DELETE_SENS_AFTER_EXPORT:-1}"
export SCANNET_DIRECT_DOWNLOAD="${SCANNET_DIRECT_DOWNLOAD:-1}"
export SCANNET_DOWNLOAD_BACKEND="${SCANNET_DOWNLOAD_BACKEND:-auto}"
export SCANNET_DOWNLOAD_TYPES="${SCANNET_DOWNLOAD_TYPES:-.txt _vh_clean_2.ply .sens}"
export SCANNET_PREFETCH_AHEAD="${SCANNET_PREFETCH_AHEAD:-1}"
export SCANNET_PURGE_PREPARED_AFTER_SCENE="${SCANNET_PURGE_PREPARED_AFTER_SCENE:-1}"
export SCANNET_PURGE_PREPARED_AFTER_FAILED="${SCANNET_PURGE_PREPARED_AFTER_FAILED:-1}"
export SCANNET_PURGE_PREPARED_ON_SKIP="${SCANNET_PURGE_PREPARED_ON_SKIP:-1}"
export SCANNET_REUSE_SELECTION="${SCANNET_REUSE_SELECTION:-1}"

export SELECTION_TOTAL="${SELECTION_TOTAL:-130}"
export TARGET_COMPLETE_RUNS="${TARGET_COMPLETE_RUNS:-100}"
export BEST_COUNT="${BEST_COUNT:-65}"
export MID_COUNT="${MID_COUNT:-32}"
export WORST_COUNT="${WORST_COUNT:-33}"
export ALLOW_SCANNET_TABLE_FALLBACK="${ALLOW_SCANNET_TABLE_FALLBACK:-1}"

export FORCE="${FORCE:-0}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
export KEEP_FAILED_ARTIFACTS="${KEEP_FAILED_ARTIFACTS:-0}"
export KEEP_ACTION_LOGS="${KEEP_ACTION_LOGS:-0}"
export ABORT_ON_CUDA_UNHEALTHY="${ABORT_ON_CUDA_UNHEALTHY:-1}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"
export DEBUG_HTML_MAX_POINTS="${DEBUG_HTML_MAX_POINTS:-150000}"

export SAMOBJECT_CHECKPOINT="${SAMOBJECT_CHECKPOINT:-/work/scratch/pafina/object-x/models/sam2ckpt/sam2_hiera_base_plus.pt}"
export SAMOBJECT_MODEL_CFG="${SAMOBJECT_MODEL_CFG:-sam2_hiera_b+.yaml}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "SCANNET_ROOT=$SCANNET_ROOT"
echo "SCANNET_WORK_ROOT=$SCANNET_WORK_ROOT"
echo "SCANNET_SAMOBJECT_ROOT=$SCANNET_SAMOBJECT_ROOT"
echo "OBJECTX_WORKFLOW_LOG_ROOT=$OBJECTX_WORKFLOW_LOG_ROOT"
echo "SOURCE_GROUP=$SOURCE_GROUP"
echo "GEOMETRY_EVAL_GROUP=$GEOMETRY_EVAL_GROUP"
echo "RUN_SAM2_MUST3R_AFTER=$RUN_SAM2_MUST3R_AFTER"
echo "SAM2_MUST3R_GEOMETRY_EVAL_GROUP=$SAM2_MUST3R_GEOMETRY_EVAL_GROUP"
echo "SELECTION_TOTAL=$SELECTION_TOTAL"
echo "TARGET_COMPLETE_RUNS=$TARGET_COMPLETE_RUNS"
echo "SCANNET_TARGET_SCENES=$SCANNET_TARGET_SCENES"
echo "SCANNET_SELECTION_MAX_FRAMES=$SCANNET_SELECTION_MAX_FRAMES"
echo "SCANNET_EXPORT_MAX_FRAMES=$SCANNET_EXPORT_MAX_FRAMES"
echo "SCANNET_DOWNLOAD_TYPES=$SCANNET_DOWNLOAD_TYPES"
echo "SCANNET_DOWNLOAD_BACKEND=$SCANNET_DOWNLOAD_BACKEND"
echo "SCANNET_PREFETCH_AHEAD=$SCANNET_PREFETCH_AHEAD"
echo "SCANNET_PURGE_PREPARED_AFTER_SCENE=$SCANNET_PURGE_PREPARED_AFTER_SCENE"
echo "SCANNET_PURGE_PREPARED_AFTER_FAILED=$SCANNET_PURGE_PREPARED_AFTER_FAILED"
echo "FORCE=$FORCE"
echo "SKIP_COMPLETED=$SKIP_COMPLETED"
echo "CLEANUP_AFTER_SCENE=$CLEANUP_AFTER_SCENE"
echo "KEEP_ACTION_LOGS=$KEEP_ACTION_LOGS"
echo "ABORT_ON_CUDA_UNHEALTHY=$ABORT_ON_CUDA_UNHEALTHY"
echo "CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"
echo "WRITE_DEBUG_HTML=$WRITE_DEBUG_HTML"
echo "SAMOBJECT_CHECKPOINT=$SAMOBJECT_CHECKPOINT"
echo "SAMOBJECT_MODEL_CFG=$SAMOBJECT_MODEL_CFG"
echo "START_AT=${START_AT:-}"
echo "LIMIT=${LIMIT:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"

nvidia-smi || true
mkdir -p "$SCANNET_ROOT" "$SCANNET_WORK_ROOT" "$SCANNET_SAMOBJECT_ROOT" "$OBJECTX_WORKFLOW_LOG_ROOT"

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

srun --ntasks=1 bash evaluation/geometry/run_scannet_pi3x_samobject_final_benchmark_100.sh
