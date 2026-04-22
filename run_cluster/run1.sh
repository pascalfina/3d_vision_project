#!/usr/bin/env bash

set -euo pipefail

SCRIPT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_REPO_DIR"
CONFIG_PATH="$SCRIPT_REPO_DIR/configs/objectx_runner.env"
CLI_ACTION=""
CLI_MODE=""
CLI_SCRIPT=""

usage() {
    cat <<'EOF'
Usage:
  ./run.sh [--config PATH] [--action NAME] [--mode local|sbatch] [--script PATH] [--] [extra args...]

Examples:
  ./run.sh
  ./run.sh --action train_latent_autoencoder
  ./run.sh --mode sbatch
  ./run.sh --config configs/objectx_runner.env --action train_reconstruction -- autoencoder.encoder.voxel.channels=[16,32,64]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --action)
            CLI_ACTION="$2"
            shift 2
            ;;
        --mode)
            CLI_MODE="$2"
            shift 2
            ;;
        --script)
            CLI_SCRIPT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

EXTRA_ARGS=("$@")

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_PATH"

REPO_DIR="${REPO_DIR:-$SCRIPT_REPO_DIR}"

PARAMS_FILE="${PARAMS_FILE:-}"
if [[ -n "$PARAMS_FILE" ]]; then
    if [[ "$PARAMS_FILE" != /* ]]; then
        PARAMS_FILE="$REPO_DIR/$PARAMS_FILE"
    fi
    if [[ ! -f "$PARAMS_FILE" ]]; then
        echo "Params file not found: $PARAMS_FILE" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$PARAMS_FILE"
fi

ACTION="${CLI_ACTION:-${ACTION:-check_env}}"
LAUNCH_MODE="${CLI_MODE:-${LAUNCH_MODE:-local}}"

if [[ -n "$CLI_SCRIPT" ]]; then
    CUSTOM_SCRIPT="$CLI_SCRIPT"
    ACTION="custom"
fi

if [[ ! -d "$REPO_DIR" ]]; then
    echo "REPO_DIR does not exist: $REPO_DIR" >&2
    exit 1
fi

case "$ACTION" in
    check_env)
        TARGET_SCRIPT="scripts/check_objectx_env.sh"
        REQUIRES_DATA_ROOT=0
        ;;
    train_latent_autoencoder)
        TARGET_SCRIPT="scripts/train_val/train_latent_autoencoder.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    train_reconstruction)
        TARGET_SCRIPT="scripts/train_val/train.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    train_scene_graph_loc)
        TARGET_SCRIPT="scripts/train_val/train_scene_graph_loc.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    inference_slat)
        TARGET_SCRIPT="scripts/inference/pipeline/run_pipeline_slat_tmp.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    inference_u3dgs)
        TARGET_SCRIPT="scripts/inference/pipeline/run_pipeline_u3dgs_tmp.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    voxelise_features)
        TARGET_SCRIPT="scripts/voxel_annotations/pipeline/voxelise_features_tmp.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    voxelise_features_scannet)
        TARGET_SCRIPT="scripts/voxel_annotations/variants/voxelise_features_scannet.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    voxelise_features_scene_alignment)
        TARGET_SCRIPT="scripts/voxel_annotations/variants/voxelise_features_scene_alignment.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    annotate_gaussians)
        TARGET_SCRIPT="scripts/gs_annotations/annotate_gaussians.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    annotate_gaussians_scannet)
        TARGET_SCRIPT="scripts/gs_annotations/annotate_gaussians_scannet.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    map_to_colmap)
        TARGET_SCRIPT="scripts/gs_annotations/map_to_colmap.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    map_to_colmap_scannet)
        TARGET_SCRIPT="scripts/gs_annotations/map_to_colmap_scannet.sh"
        REQUIRES_DATA_ROOT=1
        ;;
    custom)
        TARGET_SCRIPT="${CUSTOM_SCRIPT:-}"
        REQUIRES_DATA_ROOT=1
        ;;
    *)
        echo "Unknown ACTION: $ACTION" >&2
        exit 1
        ;;
esac

if [[ -z "${TARGET_SCRIPT:-}" ]]; then
    echo "No script selected for ACTION=$ACTION" >&2
    exit 1
fi

if [[ ! -f "$REPO_DIR/$TARGET_SCRIPT" ]]; then
    echo "Target script not found: $REPO_DIR/$TARGET_SCRIPT" >&2
    exit 1
fi

if [[ "$REQUIRES_DATA_ROOT" -eq 1 ]]; then
    if [[ -z "${DATA_ROOT_DIR:-}" ]]; then
        echo "DATA_ROOT_DIR is empty in $CONFIG_PATH" >&2
        exit 1
    fi
    if [[ ! -d "$DATA_ROOT_DIR" ]]; then
        echo "DATA_ROOT_DIR does not exist: $DATA_ROOT_DIR" >&2
        exit 1
    fi
fi

if ! declare -p RUN_ARGS >/dev/null 2>&1; then
    RUN_ARGS=()
fi

if ! declare -p PARAM_RUN_ARGS >/dev/null 2>&1; then
    PARAM_RUN_ARGS=()
fi

PARAM_ARGS=()

append_override() {
    local key="$1"
    local value="${2:-}"
    if [[ -n "$value" ]]; then
        PARAM_ARGS+=("${key}=${value}")
    fi
}

append_override "train.optim.max_epoch" "${TRAIN_MAX_EPOCH:-}"
append_override "train.batch_size" "${TRAIN_BATCH_SIZE:-}"
append_override "val.batch_size" "${VAL_BATCH_SIZE:-}"
append_override "train.num_workers" "${TRAIN_NUM_WORKERS:-}"
append_override "val.num_workers" "${VAL_NUM_WORKERS:-}"
append_override "train.optim.lr" "${TRAIN_LR:-}"
append_override "train.snapshot_steps" "${TRAIN_SNAPSHOT_STEPS:-}"
append_override "train.visualize_steps" "${TRAIN_VISUALIZE_STEPS:-}"
append_override "train.val_steps" "${TRAIN_VAL_STEPS:-}"
append_override "train.log_steps" "${TRAIN_LOG_STEPS:-}"
append_override "train.freeze_encoder" "${TRAIN_FREEZE_ENCODER:-}"
append_override "train.train_decoder" "${TRAIN_DECODER:-}"
append_override "train.clip_grad" "${TRAIN_CLIP_GRAD:-}"

COMMON_ARGS=("${RUN_ARGS[@]}" "${PARAM_ARGS[@]}" "${PARAM_RUN_ARGS[@]}" "${EXTRA_ARGS[@]}")

echo "REPO_DIR=$REPO_DIR"
echo "ACTION=$ACTION"
echo "LAUNCH_MODE=$LAUNCH_MODE"
echo "TARGET_SCRIPT=$TARGET_SCRIPT"
if [[ "$REQUIRES_DATA_ROOT" -eq 1 ]]; then
    echo "DATA_ROOT_DIR=$DATA_ROOT_DIR"
fi
if [[ -n "$PARAMS_FILE" ]]; then
    echo "PARAMS_FILE=$PARAMS_FILE"
fi

case "$LAUNCH_MODE" in
    local)
        cd "$REPO_DIR"
        export DATA_ROOT_DIR="${DATA_ROOT_DIR:-}"
        export SCRATCH="${SCRATCH:-/work/scratch/$USER}"
        source scripts/activate_objectx_env.sh
        bash "$TARGET_SCRIPT" "${COMMON_ARGS[@]}"
        ;;
    sbatch)
        cd "$REPO_DIR"
        export DATA_ROOT_DIR="${DATA_ROOT_DIR:-}"
        export SCRATCH="${SCRATCH:-/work/scratch/$USER}"
        sbatch \
            -A "${SBATCH_ACCOUNT:-3dv}" \
            --qos="${SBATCH_QOS:-3dv-team35}" \
            -p "${SBATCH_PARTITION:-jobs}" \
            -t "${SBATCH_TIME:-24:00:00}" \
            --gpus="${SBATCH_GPUS:-1}" \
            --cpus-per-task="${SBATCH_CPUS:-2}" \
            --mem="${SBATCH_MEM:-24G}" \
            -J "${SBATCH_JOB_NAME:-objectx}" \
            -o "${SBATCH_OUTPUT:-slurm-%x-%j.out}" \
            --export=ALL,DATA_ROOT_DIR="$DATA_ROOT_DIR",SCRATCH="$SCRATCH",REPO_DIR="$REPO_DIR" \
            scripts/slurm/objectx_job.sbatch \
            "$TARGET_SCRIPT" \
            "${COMMON_ARGS[@]}"
        ;;
    *)
        echo "Unsupported LAUNCH_MODE: $LAUNCH_MODE" >&2
        exit 1
        ;;
esac
