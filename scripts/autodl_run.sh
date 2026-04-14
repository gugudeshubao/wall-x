#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/scripts"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/autodl_env.sh"

MODE="${1:-smoke}"
shift || true

case "$MODE" in
  smoke|fake)
    exec python "$SCRIPT_DIR/fake_inference.py" --model-path "$WALL_X_MODEL_PATH" "$@"
    ;;
  vqa)
    exec python "$SCRIPT_DIR/vqa_inference.py" \
      --model-path "$WALL_X_MODEL_PATH" \
      --image-path "$ROOT_DIR/assets/cot_example_frame.png" \
      "$@"
    ;;
  bench)
    exec python "$SCRIPT_DIR/benchmark_vqa.py" \
      --model-path "$WALL_X_MODEL_PATH" \
      --image-path "$ROOT_DIR/assets/cot_example_frame.png" \
      "$@"
    ;;
  repl)
    exec python "$SCRIPT_DIR/vqa_repl.py" \
      --model-path "$WALL_X_MODEL_PATH" \
      --image-path "$ROOT_DIR/assets/cot_example_frame.png" \
      "$@"
    ;;
  shell)
    echo "Environment loaded. ROOT_DIR=$ROOT_DIR"
    exec "${SHELL:-/bin/bash}"
    ;;
  *)
    cat <<'EOF'
Usage:
  source scripts/autodl_env.sh
  bash scripts/autodl_run.sh [smoke|fake|vqa|bench|repl|shell] [extra args]

Examples:
  bash scripts/autodl_run.sh
  bash scripts/autodl_run.sh fake --seq-length 32
  bash scripts/autodl_run.sh vqa --question "What should you do next?"
  bash scripts/autodl_run.sh bench --runs 3 --max-new-tokens 128
  bash scripts/autodl_run.sh repl
  bash scripts/autodl_run.sh shell
EOF
    exit 1
    ;;
esac
