#!/usr/bin/env bash

_objectx_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLSG_SPACE="$(cd "${_objectx_script_dir}/.." && pwd)"

if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/etc/profile.d/modules.sh
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi

if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    module load cuda/12.8 >/dev/null 2>&1 || true
fi

if [[ -f "$VLSG_SPACE/.venv_objx/bin/activate" ]]; then
    export VLSG_VENV_PATH="$VLSG_SPACE/.venv_objx"
elif [[ -f "$VLSG_SPACE/.venv/bin/activate" ]]; then
    export VLSG_VENV_PATH="$VLSG_SPACE/.venv"
else
    echo "Object-X environment not found under $VLSG_SPACE/.venv_objx or $VLSG_SPACE/.venv" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$VLSG_VENV_PATH/bin/activate"

unset LD_LIBRARY_PATH

export TMPDIR="${TMPDIR:-/tmp/${USER}-objectx}"
mkdir -p "$TMPDIR"

if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

export PIP_NO_CACHE_DIR=1
export PIP_CONFIG_FILE=/dev/null
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export SCRATCH="${SCRATCH:-/work/scratch/${USER}}"

case ":${PYTHONPATH:-}:" in
    *":$VLSG_SPACE:"*) ;;
    *) export PYTHONPATH="$VLSG_SPACE${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

case ":${PYTHONPATH:-}:" in
    *":$VLSG_SPACE/dependencies/gaussian-splatting:"*) ;;
    *) export PYTHONPATH="$VLSG_SPACE/dependencies/gaussian-splatting${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

for _objectx_dep in \
    "$VLSG_SPACE/dependencies/sam2" \
    "$VLSG_SPACE/dependencies/must3r" \
    "$VLSG_SPACE/dependencies/must3r/dust3r"
do
    if [[ -d "$_objectx_dep" ]]; then
        case ":${PYTHONPATH:-}:" in
            *":$_objectx_dep:"*) ;;
            *) export PYTHONPATH="$_objectx_dep${PYTHONPATH:+:$PYTHONPATH}" ;;
        esac
    fi
done

export OBJECTX_ENV_ACTIVATED=1

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Object-X environment activated in a child shell. Use 'source scripts/activate_objectx_env.sh' to persist it."
fi
