#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"

CODE_DIR="${CODE_DIR:-/root/wall-x}"
CONFIG_PATH="${CONFIG_PATH:-$CODE_DIR/workspace/lerobot_example/config_qact_a800_1h.yml}"
MASTER_PORT="${MASTER_PORT:-12348}"

cd "$CODE_DIR"
source scripts/autodl_env.sh

accelerate launch \
  --num_processes=1 \
  --main_process_port="$MASTER_PORT" \
  train_qact.py \
  --config "$CONFIG_PATH" \
  --seed "$MASTER_PORT"
