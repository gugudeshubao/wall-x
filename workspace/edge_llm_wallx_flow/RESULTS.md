# Results

## Orin action-flow bridge

Status: working.

Built with TensorRT-Edge-LLM `action_build`:
- `action.engine` generated successfully
- source ONNX exported from wall-x flow-action weights

Smoke on Orin:
- `action_step_ms = 31.435`
- `denoised_vs_ref cosine = 0.99999797`
- `denoised_vs_ref mean_abs = 1.22605660e-03`
- `denoised_vs_ref max_abs = 5.83672523e-03`

Full Euler loop on Orin:
- `flow_final_cosine = 0.98523772`
- `flow_final_mean_abs = 7.00874105e-02`
- `flow_final_max_abs = 2.99018741e-01`
- `flow_step_ms_mean = 31.347`
- `flow_step_ms_std = 0.077`

Warm benchmark on Orin (`--warmup 2 --iters 5`):
- `action_step_ms_mean = 31.329`
- `action_step_ms_std = 0.018`
- `flow_step_ms_mean = 31.335`
- `flow_step_ms_std = 0.032`
- `flow_total_ms_mean = 157.661`
- `flow_total_ms_std = 0.138`

## Comparison

### Full Flow Action (same wall-x benchmark family)

| Route | Mean latency | Relative to `cpp_infer` |
|---|---:|---:|
| `cpp_infer` | `290.7 ms` | `1.00x` |
| Hand TRT / TRT-LLM | `176.837 ms` | `1.65x faster` |
| Edge-LLM custom bridge | `157.661 ms` | `1.85x faster` |

Notes:
- The Edge-LLM bridge number is a custom action bridge with host-side Euler loop.
- It is not the stock `llm_inference` multimodal path.
- It is also not directly equivalent to a full VLM/VQA engine-path benchmark.

### What this means

- `cpp_infer` is still the cleanest full custom baseline.
- Hand TRT / TRT-LLM is already a clear win on full Flow Action.
- Edge-LLM custom bridge is the fastest of the three on the measured action-flow loop, but it is the least stock / least integrated path.

Notes:
- This is Orin-only.
- The bridge uses:
  - wall-x flow checkpoint
  - TensorRT-Edge-LLM `action_build`
  - a custom runner around the generated `action.engine`
- The action backend is not the stock `llm_inference` VLM path.

## Plugin / operator notes

No new custom TensorRT plugin was written for this bridge.

What we wrote:
- ONNX export bridge
- `action_build` wrapper
- custom runner

What we reused from Edge-LLM:
- `examples/multimodal/action_build.cpp`
- `cpp/builder/actionBuilder.cpp`
- built plugin/runtime library shipped by Edge-LLM

The exported ONNX stays on standard ops only. Main op types observed:
- `MatMul`
- `Add`
- `Mul`
- `Softmax`
- `ReduceMean`
- `Pow`
- `Sqrt`
- `Concat`
- `Slice`
- `Transpose`
- `Reshape`

The one export adjustment we had to make was:
- switch ONNX export from opset 23 to opset 22

Reason:
- opset 23 introduced native `RMSNormalization`
- current Edge-LLM action build path on Orin did not accept that op directly
- opset 22 kept the graph in standard primitive ops and built successfully

## Quantization status

This bridge is **not quantized**.

What is true today:
- the measured `157.661 ms` result is from the custom bridge without a new INT8/W8A8 pass
- the bridge uses the original wall-x weights exported into ONNX
- current `TensorRT-Edge-LLM` action export path is effectively **FP16-only**

Source boundary:
- `tensorrt_edgellm/scripts/export_action.py`
- `tensorrt_edgellm/onnx_export/action_export.py`

The current action export explicitly rejects other dtypes and only allows:
- `dtype = fp16`

So the practical conclusion is:

> **VQA/VLM in Edge-LLM has a real quantization story; the current action-expert path used by this wall-x Flow bridge does not.**

## One-glance summary

| Route | Flow Action latency | Note |
|---|---:|---|
| `cpp_infer` | `290.7 ms` | baseline |
| hand TRT / TRT-LLM | `176.837 ms` | stock TRT route, clearly faster |
| Edge-LLM custom bridge | `157.661 ms` | fastest, but custom bridge |

## Decision

- Keep `cpp_infer` as the clean baseline.
- Keep hand TRT / TRT-LLM as the current stock TRT path.
- Keep Edge-LLM custom bridge as the fastest Flow path, but treat it as a bridge, not the final stock runtime.

## Serving smoke

The Flow custom bridge is no longer only a benchmark script.

On Orin we also verified a serving-layer smoke with:

- `wall_x.serving.flow_policy.FlowPolicy`
- `wall_x.serving.launch_flow_serving`
- `wall_x.serving.websocket_policy_server.WebsocketPolicyServer`

Smoke result on Orin (`ws://127.0.0.1:8795`):

- metadata:
  - `backend = edge_flow_bridge`
  - `engine_dir = /data/wy/wall-x/workspace/edge_llm_exp/edge_engines/wallx_flow_action_step`
- one websocket request succeeded
- response keys:
  - `action`
  - `predict_action`
  - `flow`
  - `server_timing`
- returned action shape:
  - `(1, 32, 20)`
- returned timing:
  - `server_timing.infer_ms ≈ 214.29`
  - `flow.flow_total_ms ≈ 212.86`
  - `flow.flow_step_ms_mean ≈ 31.61`

This means:

> **The Edge-LLM Flow bridge has now reached wall_x.serving / websocket layer smoke-complete status on Orin.**
