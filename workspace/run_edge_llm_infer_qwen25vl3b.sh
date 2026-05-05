#!/bin/bash
set -euo pipefail

mkdir -p /data/wy/wall-x/workspace/edge_llm_exp

cat > /data/wy/wall-x/workspace/edge_llm_exp/input_vlm_qwen25vl3b.json <<'JSON'
{
  "batch_size": 1,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 50,
  "max_generate_length": 32,
  "requests": [
    {
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": [
            {
              "type": "image",
              "image": "/data/wy/wall-x/test_images/fruits_on_table.png"
            },
            {
              "type": "text",
              "text": "Please describe the image."
            }
          ]
        }
      ]
    }
  ]
}
JSON

cd /data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin
export EDGELLM_PLUGIN_PATH=/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so

exec ./examples/llm/llm_inference \
  --engineDir /data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen2.5-vl-3b \
  --multimodalEngineDir /data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen2.5-vl-3b/visual \
  --inputFile /data/wy/wall-x/workspace/edge_llm_exp/input_vlm_qwen25vl3b.json \
  --outputFile /data/wy/wall-x/workspace/edge_llm_exp/output_vlm_qwen25vl3b.json
