# wall-x Flow Action on TensorRT-Edge-LLM

Goal:
- export the wall-x flow-action step to an Edge-LLM-compatible ONNX
- build it with Edge-LLM `action_build`
- run the action engine on Orin and compare against wall-x reference

Current status:
- bridge working on Orin
- target backend: `TensorRT-Edge-LLM action_build`
- target model: `wall-oss-flow`
- target host: Orin only

Files:
- `export_wallx_action_onnx.py`
- `quantize_wallx_action_onnx.py`
- `build_wallx_action_engine.py`
- `run_wallx_action_engine.py`
- `flow_policy_bridge.py`
- `run_wallx_action_orin.sh`

Results:
- see `RESULTS.md`

## What this is

This is a custom bridge for `wall-x Flow Action`.

It is not the stock Edge-LLM VLM path:
- not `llm_inference`
- not the standard `multimodalEngineDir` route

Instead, it does:

1. export a wall-x flow-action step to ONNX
2. build an `action.engine` with Edge-LLM `action_build`
3. run that engine with a custom runner
4. keep the outer Euler loop on host

It now also has a minimal serving adapter:

- `workspace/edge_llm_wallx_flow/flow_policy_bridge.py`
- `wall_x/serving/flow_policy.py`
- `wall_x/serving/launch_flow_serving.py`

It also has a first INT8 experiment path:

- `quantize_wallx_action_onnx.py`
  - post-export INT8 QDQ quantization of the custom wall-x action ONNX
  - this is **not** the official Edge-LLM `export_action` path
  - it is the first custom attempt to see whether `action_build` can consume an INT8-ish graph
  - current result:
    - quantized ONNX can be produced
    - but `action_build` rejects `DynamicQuantizeLinear` / `MatMulInteger`

It now also has a second, more promising custom INT8-SQ path:

- `export_wallx_action_int8sq_onnx.py`
  - custom INT8-SQ QDQ export using the same `trt::int8_sq_*` custom ops that Edge-LLM uses in `llm_loader`
  - this one **does** build with `action_build`
  - current result on Orin:
    - `action_step_ms_mean = 15.374`
    - `flow_total_ms_mean = 77.866`
    - but `flow_final_cosine = 0.27758002`, so accuracy is far too low for practical use

## Minimal websocket serving smoke

On Orin, the Flow bridge can now be launched through `wall_x.serving`:

```bash
cd /data/wy/wall-x
PYTHONPATH=/data/wy/wall-x:/home/dog/.local/lib/python3.10/site-packages:/data/wy/trtllm_test/TensorRT-LLM/build/lib:/data/wy/trtllm_test/TensorRT-LLM/cpp/build/tensorrt_llm \
/usr/bin/python3 -m wall_x.serving.launch_flow_serving \
  --engine-dir /data/wy/wall-x/workspace/edge_llm_exp/edge_engines/wallx_flow_action_step \
  --ref-path /data/wy/wall-x/workspace/trt_spike/tmp/flow_dummy_1/flow_dummy_reference.safetensors \
  --model-path /data/wy/models/wall-oss-flow \
  --host 127.0.0.1 \
  --port 8795
```

And a minimal websocket client request succeeds with:

- metadata returned from the server
- response keys:
  - `action`
  - `predict_action`
  - `flow`
  - `server_timing`

## Requirements on Orin

- wall-x model path:
  - `/data/wy/models/wall-oss-flow`
- flow reference:
  - `/data/wy/wall-x/workspace/trt_spike/tmp/flow_dummy_1/flow_dummy_reference.safetensors`
- Edge-LLM repo:
  - `/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM`
- Edge-LLM export env:
  - `/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin/python`
- TensorRT-LLM runtime import available from:
  - `/usr/bin/python3`

## One-shot run on Orin

```bash
cd /data/wy/wall-x/workspace/edge_llm_wallx_flow
bash run_wallx_action_orin.sh
```

## Manual steps on Orin

### 1. Export ONNX

```bash
/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin/python \
  export_wallx_action_onnx.py \
  --model-path /data/wy/models/wall-oss-flow \
  --ref /data/wy/wall-x/workspace/trt_spike/tmp/flow_dummy_1/flow_dummy_reference.safetensors \
  --output-dir /data/wy/wall-x/workspace/edge_llm_exp/work_onnx/wallx_flow_action_step \
  --num-layers 36
```

### 2. Build engine

```bash
export EDGELLM_PLUGIN_PATH=/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so

/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin/python \
  build_wallx_action_engine.py \
  --onnx-dir /data/wy/wall-x/workspace/edge_llm_exp/work_onnx/wallx_flow_action_step \
  --engine-dir /data/wy/wall-x/workspace/edge_llm_exp/edge_engines/wallx_flow_action_step \
  --min-seq-len 456 \
  --max-seq-len 488 \
  --opt-seq-len 472
```

### 3. Run smoke + full Euler loop

```bash
/usr/bin/python3 \
  run_wallx_action_engine.py \
  --engine-dir /data/wy/wall-x/workspace/edge_llm_exp/edge_engines/wallx_flow_action_step \
  --ref /data/wy/wall-x/workspace/trt_spike/tmp/flow_dummy_1/flow_dummy_reference.safetensors \
  --model-path /data/wy/models/wall-oss-flow
```

## Important compatibility note

The ONNX export must use opset 22.

Reason:
- opset 23 introduces native `RMSNormalization`
- current Edge-LLM action-build path on Orin does not accept that op directly
- opset 22 keeps the graph in standard primitive ops and builds successfully
