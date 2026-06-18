#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -J objx-final100
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-objx-final100-%j.out
#SBATCH -e /work/scratch/pafina/object-x/debug/slurm-objx-final100-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug"

# Keep the benchmark resumable and quota-safe by default.  Override any of
# these with sbatch --export=ALL,KEY=value if needed.
export GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-objectx_final_100}"
export CLEANUP_AFTER_SCENE="${CLEANUP_AFTER_SCENE:-1}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"
export TARGET_COMPLETE_RUNS="${TARGET_COMPLETE_RUNS:-100}"
export SAMOBJECT_CHECKPOINT="${SAMOBJECT_CHECKPOINT:-/work/scratch/pafina/object-x/models/sam2ckpt/sam2_hiera_base_plus.pt}"
export SAMOBJECT_MODEL_CFG="${SAMOBJECT_MODEL_CFG:-sam2_hiera_b+.yaml}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "GEOMETRY_EVAL_GROUP=$GEOMETRY_EVAL_GROUP"
echo "CLEANUP_AFTER_SCENE=$CLEANUP_AFTER_SCENE"
echo "CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"
echo "SKIP_COMPLETED=$SKIP_COMPLETED"
echo "WRITE_DEBUG_HTML=$WRITE_DEBUG_HTML"
echo "TARGET_COMPLETE_RUNS=$TARGET_COMPLETE_RUNS"
echo "SAMOBJECT_CHECKPOINT=$SAMOBJECT_CHECKPOINT"
echo "SAMOBJECT_MODEL_CFG=$SAMOBJECT_MODEL_CFG"
echo "START_AT=${START_AT:-}"
echo "LIMIT=${LIMIT:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"

nvidia-smi || true

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
print("cuda_device=", torch.cuda.get_device_name(0))
print("bf16_supported=", torch.cuda.is_bf16_supported())
PY

srun --ntasks=1 bash evaluation/geometry/run_objectx_final_benchmark_100.sh
