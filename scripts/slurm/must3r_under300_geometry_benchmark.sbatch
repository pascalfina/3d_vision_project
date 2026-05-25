#!/usr/bin/env bash
#SBATCH -A 3dv
#SBATCH --qos=3dv-team35
#SBATCH -p jobs
#SBATCH -t 24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=64G
#SBATCH -J must3r-geom329
#SBATCH -o /work/scratch/pafina/object-x/debug/slurm-must3r-geom329-%j.out
#SBATCH -e /work/scratch/pafina/object-x/debug/slurm-must3r-geom329-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/scratch/pafina/object-x}"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/debug"

export GEOMETRY_EVAL_GROUP="${GEOMETRY_EVAL_GROUP:-must3r_sequence_under300}"
export PROFILE_LIST="${PROFILE_LIST:-$REPO_DIR/configs/workflows/scene_profiles/pi3x_under300_profiles.txt}"
export FORCE="${FORCE:-0}"
export KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
export KEEP_FAILED_ARTIFACTS="${KEEP_FAILED_ARTIFACTS:-0}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export WRITE_DEBUG_HTML="${WRITE_DEBUG_HTML:-1}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "GEOMETRY_EVAL_GROUP=$GEOMETRY_EVAL_GROUP"
echo "PROFILE_LIST=$PROFILE_LIST"
echo "FORCE=$FORCE"
echo "KEEP_ARTIFACTS=$KEEP_ARTIFACTS"
echo "KEEP_FAILED_ARTIFACTS=$KEEP_FAILED_ARTIFACTS"
echo "CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"
echo "WRITE_DEBUG_HTML=$WRITE_DEBUG_HTML"
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

srun --ntasks=1 bash evaluation/geometry/run_must3r_sequence_benchmark_under300.sh
