#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FT_MODEL_PATH="${FT_MODEL_PATH:-/root/autodl-tmp/outputs/wall-x-train-1h/0}"

cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/autodl_env.sh"

exec python "$ROOT_DIR/scripts/vqa_inference.py" \
  --model-path "$FT_MODEL_PATH" \
  --image-path "$ROOT_DIR/assets/cot_example_frame.png" \
  "$@"
