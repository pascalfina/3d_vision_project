#!/usr/bin/env bash

#srun -A 3dv --qos=3dv-team35 -p jobs -t 24:00:00 --pty bash --login 

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO_DIR"
source scripts/activate_objectx_env.sh

export DATA_ROOT_DIR="${DATA_ROOT_DIR:-/work/scratch/pafina/objectx-data}"
export SCRATCH="${SCRATCH:-/work/scratch/pafina}"

./run.sh --action train_latent_autoencoder "$@"
