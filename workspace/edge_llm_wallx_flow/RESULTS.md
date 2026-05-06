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

### First INT8 experiment status

We also tried a first custom INT8 path on top of the custom Flow bridge:

- post-export ONNX quantization with ONNX Runtime
- dynamic INT8, `MatMul`-only, QDQ-style graph
- output directory:
  - `/data/wy/wall-x/workspace/edge_llm_exp/work_onnx/wallx_flow_action_step_int8dyn`

Result:

- the quantized ONNX was generated successfully
- but `TensorRT-Edge-LLM action_build` rejected it

The concrete failure was:

- ONNX nodes such as:
  - `DynamicQuantizeLinear`
  - `MatMulInteger`
- `action_build` / TensorRT parser failed in:
  - `checkDynamicQuantizeLinear`
  - `checkMatMulInteger`

So the practical boundary is:

> **For the current wall-x Flow custom bridge, a first INT8 QDQ / MatMulInteger attempt is not accepted by Edge-LLM `action_build`.**

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

## First INT8-SQ experiment

We also attempted a first custom INT8-SQ path for the Flow bridge:

- export path:
  - `export_wallx_action_int8sq_onnx.py`
- quantized ONNX output:
  - `/data/wy/wall-x/workspace/edge_llm_exp/work_onnx/wallx_flow_action_step_int8sq/model.onnx`
- build path:
  - `TensorRT-Edge-LLM action_build`
- runtime:
  - existing custom runner from `run_wallx_action_engine.py`

Result on Orin:

- `action_step_ms = 15.473`
- `denoised_vs_ref cosine = 0.99081051`
- `denoised_vs_ref mean_abs = 1.65894449e-01`
- `denoised_vs_ref max_abs = 7.19319820e-01`
- `flow_final_cosine = 0.27758002`
- `flow_final_mean_abs = 6.16935730e-01`
- `flow_final_max_abs = 2.84748363e+00`
- `flow_total_ms = 89.122`
- `flow_step_ms_mean = 15.362`
- `flow_step_ms_std = 0.064`

Warm benchmark on Orin (`--warmup 0 --iters 1`):

- `action_step_ms_mean = 15.374`
- `flow_step_ms_mean = 15.360`
- `flow_total_ms_mean = 77.866`

This tells us:

> **The custom Flow INT8-SQ path is buildable and much faster, but the accuracy drops sharply enough that it is not yet a usable replacement for the FP16 bridge.**
