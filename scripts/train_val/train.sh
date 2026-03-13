#!/bin/bash

args=("$@")

# Set environment variables
source "$(dirname "${BASH_SOURCE[0]}")/../activate_objectx_env.sh"
# get output directory argument if it exists
for i in "$@"
do
case $i in
    -o=*|--output_dir=*)
    export VLSG_TRAINING_OUT_DIR="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# Set output directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export VLSG_TRAINING_OUT_DIR="$SCRATCH/training_recon_scene/$timestamp"

# Navigate to VLSG space
cd "$VLSG_SPACE" || { echo "Failed to change directory to $VLSG_SPACE"; exit 1; }

# Run training script
python src/trainval/train_reconstruction.py --config scripts/train_val/train.yaml --log_steps 1 output_dir=\"$VLSG_TRAINING_OUT_DIR\" ${args[@]}
