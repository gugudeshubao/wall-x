#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
WALL_X_MODEL_PATH="${WALL_X_MODEL_PATH:-/root/autodl-fs/models/wall-oss-flow}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "venv not found: $VENV_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export WALL_X_MODEL_PATH
export HF_ENDPOINT

echo "Activated wall-x environment"
echo "ROOT_DIR=$ROOT_DIR"
echo "VENV_PATH=$VENV_PATH"
echo "CUDA_HOME=$CUDA_HOME"
echo "WALL_X_MODEL_PATH=$WALL_X_MODEL_PATH"
