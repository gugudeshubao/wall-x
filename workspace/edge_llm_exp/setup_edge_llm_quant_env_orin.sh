#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM}"
VENV_DIR="${2:-/data/wy/wall-x/workspace/edge_llm_exp/venv_edge_quant}"
CUDA_TORCH_WHEEL="${3:-/home/dog/work/wheels/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl}"
TMP_BASE="${TMP_BASE:-/data/wy/tmp/edge_llm_quant}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/wy/tmp/pip-cache-edge-llm}"
MODEL_OPT_VERSION="${MODEL_OPT_VERSION:-0.39.0}"

REQ_FILE="${REPO_ROOT}/experimental/quantization/requirements.txt"
TMP_REQ="$(mktemp /tmp/edge_llm_quant_requirements.XXXXXX.txt)"
trap 'rm -f "${TMP_REQ}"' EXIT

if [[ ! -f "${CUDA_TORCH_WHEEL}" ]]; then
  echo "missing torch wheel: ${CUDA_TORCH_WHEEL}" >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi

mkdir -p "${TMP_BASE}" "${PIP_CACHE_DIR}"
export TMPDIR="${TMP_BASE}"
export TMP="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install --no-deps "${CUDA_TORCH_WHEEL}"

# Install the quantization package stack without letting pip replace torch.
# Use the repo-pinned ModelOpt line instead of the newer standalone one because
# the Orin CUDA torch wheel we rely on is compatible with 0.39.x, not 0.42.x.
grep -vE '^(torch==|nvidia-modelopt==|numpy~=)' "${REQ_FILE}" > "${TMP_REQ}"
"${VENV_DIR}/bin/pip" install --no-deps "nvidia-modelopt==${MODEL_OPT_VERSION}"
"${VENV_DIR}/bin/pip" install -r "${TMP_REQ}" \
  "numpy<2" \
  "pydantic>=2.0" \
  "nvidia-ml-py>=12" \
  rich \
  pulp \
  scipy \
  regex \
  "torchprofile>=0.0.4"

echo "quant env ready: ${VENV_DIR}"
echo "python: $("${VENV_DIR}/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())')"
