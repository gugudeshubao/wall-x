#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wy/wall-x/workspace/edge_llm_wallx_flow
MODEL_PATH=/data/wy/models/wall-oss-flow
REF=/data/wy/wall-x/workspace/trt_spike/tmp/flow_dummy_1/flow_dummy_reference.safetensors
EDGE_LLM_ROOT=/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM
ONNX_DIR=/data/wy/wall-x/workspace/edge_llm_exp/work_onnx/wallx_flow_action_step
ENGINE_DIR=/data/wy/wall-x/workspace/edge_llm_exp/edge_engines/wallx_flow_action_step

echo "[1/3] export ONNX"
/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin/python \
  "$ROOT/export_wallx_action_onnx.py" \
  --model-path "$MODEL_PATH" \
  --ref "$REF" \
  --output-dir "$ONNX_DIR" \
  --num-layers 36

echo "[2/3] build action.engine"
export EDGELLM_PLUGIN_PATH="$EDGE_LLM_ROOT/build_orin/libNvInfer_edgellm_plugin.so"
/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin/python \
  "$ROOT/build_wallx_action_engine.py" \
  --edge-llm-root "$EDGE_LLM_ROOT" \
  --onnx-dir "$ONNX_DIR" \
  --engine-dir "$ENGINE_DIR" \
  --min-seq-len 456 \
  --max-seq-len 488 \
  --opt-seq-len 472

echo "[3/3] run smoke + euler loop"
/usr/bin/python3 \
  "$ROOT/run_wallx_action_engine.py" \
  --engine-dir "$ENGINE_DIR" \
  --ref "$REF" \
  --model-path "$MODEL_PATH"
