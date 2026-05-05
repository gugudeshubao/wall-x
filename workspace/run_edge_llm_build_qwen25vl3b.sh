#!/bin/bash
set -euo pipefail

source /data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin/activate
cd /data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin

export EDGELLM_PLUGIN_PATH=/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so

exec ./examples/llm/llm_build \
  --onnxDir=/data/wy/wall-x/workspace/edge_llm_exp/work_onnx/qwen2.5-vl-3b \
  --engineDir=/data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen2.5-vl-3b \
  --maxBatchSize=1 \
  --maxInputLen=1024 \
  --maxKVCacheCapacity=4096
