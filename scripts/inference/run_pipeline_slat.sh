#!/bin/bash

args=("$@")

# Set environment variables
source "$(dirname "${BASH_SOURCE[0]}")/../activate_objectx_env.sh"
export RESUME_DIR="$VLSG_TRAINING_OUT_DIR"

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
export VLSG_TRAINING_OUT_DIR="$SCRATCH/test_latent_autoencoder/$timestamp"

# Navigate to VLSG space
cd "$VLSG_SPACE" || { echo "Failed to change directory to $VLSG_SPACE"; exit 1; }

python src/inference/structured_latent_inference.py --config configs/config.yaml  ${args[@]}
