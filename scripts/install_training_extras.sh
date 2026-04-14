#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/autodl_env.sh"

echo "Installing training-only extras into $VENV_PATH"
echo "CUDA_HOME=$CUDA_HOME"

MAX_JOBS="${MAX_JOBS:-4}" pip install flash-attn==2.7.4.post1 --no-build-isolation

cat <<'EOF'

flash-attn installation finished.

If you plan to train, the next manual step is installing lerobot:

  git clone https://github.com/huggingface/lerobot.git
  cd lerobot
  git checkout c66cd401767e60baece16e1cf68da2824227e076
  pip install -e .

EOF
