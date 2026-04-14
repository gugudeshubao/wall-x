#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"

CODE_DIR="${CODE_DIR:-/root/wall-x}"
CONFIG_PATH="${CONFIG_PATH:-$CODE_DIR/workspace/lerobot_example/config_qact_a800.yml}"
MASTER_PORT="${MASTER_PORT:-12345}"

export LAUNCHER="accelerate launch --num_processes=$NUM_GPUS --main_process_port=$MASTER_PORT"
export SCRIPT="$CODE_DIR/train_qact.py"
export SCRIPT_ARGS="--config $CONFIG_PATH --seed $MASTER_PORT"

echo "Running command: $LAUNCHER $SCRIPT $SCRIPT_ARGS"
$LAUNCHER $SCRIPT $SCRIPT_ARGS
