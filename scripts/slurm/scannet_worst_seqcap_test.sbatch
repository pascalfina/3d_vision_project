#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=64G
#SBATCH -J scannet-seqcap-test
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-scannet-seqcap-test-%j.out
#SBATCH -e /work/scratch/pafina/object-x/debug/slurm-scannet-seqcap-test-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"
cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug"

export SCANNET_ROOT="${SCANNET_ROOT_OVERRIDE:-/work/courses/3dv/team35/pafina/scannet_under300_data}"
export SCANNET_WORK_ROOT="${SCANNET_WORK_ROOT_OVERRIDE:-/work/courses/3dv/team35/pafina}"
export OBJECTX_WORKFLOW_LOG_ROOT="${OBJECTX_WORKFLOW_LOG_ROOT_OVERRIDE:-$SCANNET_WORK_ROOT/logs}"
export WORST_N="${WORST_N:-10}"
export WORST_SORT_KEY="${WORST_SORT_KEY:-acc_compl_ratio}"
export FRAME_CAP="${FRAME_CAP:-160}"
export CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "SCANNET_ROOT=$SCANNET_ROOT"
echo "SCANNET_WORK_ROOT=$SCANNET_WORK_ROOT"
echo "OBJECTX_WORKFLOW_LOG_ROOT=$OBJECTX_WORKFLOW_LOG_ROOT"
echo "WORST_N=$WORST_N"
echo "WORST_SORT_KEY=$WORST_SORT_KEY"
echo "FRAME_CAP=$FRAME_CAP"
echo "CLEANUP_AFTER_SCENE=$CLEANUP_AFTER_SCENE"
echo "CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"
echo "SKIP_COMPLETED=$SKIP_COMPLETED"
echo "WRITE_DEBUG_HTML=$WRITE_DEBUG_HTML"
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
if not torch.cuda.is_bf16_supported():
    sys.exit("Assigned GPU does not support bf16; MUSt3R requires a bf16-capable GPU.")
PY

srun --ntasks=1 bash evaluation/geometry/run_scannet_worst_sequence_cap_test.sh
