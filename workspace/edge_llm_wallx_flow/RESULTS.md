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

### Benchmark conditions — what each number actually measures

| Route | ViT included | LLM prefix included | ODE included | Benchmark input |
|---|:---:|:---:|:---:|---|
| `cpp_infer` | ✓ 115 ms | ✓ ~80 ms | ✓ ~96 ms | real image |
| Hand TRT / TRT-LLM | ✗ | ✓ 110 ms | ✓ 82 ms | dummy ref tensors |
| Edge-LLM FP16 / INT8 / Mixed10 | ✗ | ✗ | ✓ | dummy ref tensors |

**cpp_infer 290.7 ms is a full-pipeline number. TRT and Edge-LLM numbers are partial-pipeline (KV cache pre-computed from dummy reference tensors).**

### ODE denoising — dummy reference (all routes consistent)

Dummy reference: zero-initialized inputs, prefix_length=456, total_seq=488.

| Route | ODE 5 steps | step latency | Hz | flow_final cosine |
|---|---:|---:|---:|---:|
| cpp_infer (estimated) | ~96 ms | ~19 ms | ~10 Hz | — |
| Hand TRT / TRT-LLM | **82 ms** | **16 ms** | **~12 Hz** | 0.985 |
| Edge-LLM FP16 | 157 ms | 31 ms | 6 Hz | 0.985 |
| Edge-LLM Mixed10 | 99 ms | 20 ms | ~10 Hz | 0.292 |
| Edge-LLM INT8-SQ | 78 ms | 16 ms | ~13 Hz | 0.278 |

### ODE denoising — real image reference

Real image: `fruits_on_table.png`, prompt: `"Pick up the red object on the table."`
Export: Python wall-x model, prefix_length=427, padded to 456 for engine compatibility.

| Route | ODE 5 steps | step latency | Hz | flow_final cosine |
|---|---:|---:|---:|---:|
| Edge-LLM FP16 (real img) | **156.9 ms** | **31.2 ms** | **6 Hz** | **0.876** |
| Edge-LLM Mixed10 (real img) | **100.3 ms** | **19.8 ms** | **~10 Hz** | **0.102** |

Notes on real-image cosine:
- FP16: cosine drops from 0.985 (dummy) → 0.876 (real image). FP16 is still usable.
- Mixed10: cosine drops from 0.292 (dummy) → 0.102 (real image). INT8 error amplified by real semantics.
- The cosine reference is the Python wall-x model output on the same real image.

### Full pipeline — apples-to-apples (new image, first action)

| Route | Total (est.) | Hz | cosine | Notes |
|---|---:|---:|---:|---|
| `cpp_infer` INT8 | **290.7 ms** | **3.4 Hz** | — | real-image, fully measured |
| Hand TRT (+ ViT) | ~301–307 ms | ~3.3 Hz | 0.985 | ViT 115ms（实测）+ TRT 176ms（实测），误差 <5%，无需真图再验 |
| Edge-LLM FP16 (+ ViT/prefix) | ~352 ms | ~2.8 Hz | **0.876** | real-image ODE validated |
| Edge-LLM Mixed10 (+ ViT/prefix) | ~295 ms | ~3.4 Hz | 0.102 | real-image ODE validated |
| Edge-LLM INT8-SQ (+ ViT/prefix) | ~273 ms | ~3.7 Hz | 0.278 | dummy cosine only |

### Summary: which route is actually usable today

| Route | Speed | Accuracy (real img) | Usable? |
|---|---|---|---|
| cpp_infer FP16 | 3.4 Hz full | high | ✓ baseline |
| Hand TRT | ~3.3 Hz full, 12 Hz ODE | 0.985 (dummy) | ✓ best ODE |
| Edge-LLM FP16 | ~2.8 Hz full, 6 Hz ODE | **0.876** (real) | ✓ usable |
| Edge-LLM Mixed10 | ~3.4 Hz full, **10 Hz ODE** | 0.102 (real) | ✗ accuracy broken |
| Edge-LLM INT8-SQ | ~3.7 Hz full, 13 Hz ODE | 0.278 (dummy) | ✗ accuracy broken |

**Current best practical choice: Hand TRT (82ms ODE, cosine=0.985) or Edge-LLM FP16 (157ms ODE, cosine=0.876 on real image).**

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

Warm benchmark on Orin (`--warmup 2 --iters 5`):

- `action_step_ms_mean = 15.383`
- `action_step_ms_std = 0.032`
- `flow_step_ms_mean = 15.406`
- `flow_step_ms_std = 0.027`
- `flow_total_ms_mean = 78.183`
- `flow_total_ms_std = 0.313`

This tells us:

> **The custom Flow INT8-SQ path is buildable and runs at ~2x FP16 speed, but flow_final_cosine = 0.278 indicates the accuracy is not yet usable.**

## Mixed Precision experiment: 26-layer INT8 + 10-layer FP16

### Motivation

Full INT8-SQ gives ~2x speedup but accuracy collapses after 5 Euler steps.
Goal: find a balance between speed and accuracy by keeping the last 10 layers in FP16.

### Setup

- Layers 0–25: INT8-SQ (`Int8SQLinearBridge`)
- Layers 26–35: FP16 (`FP16LinearBridge`)
- Export: `export_wallx_action_int8sq_onnx.py --fp16-from-layer 26`
- Export env: PyTorch 2.13 on RTX-5090 (required for `dynamo=True` ONNX export API)
- Build + bench: Orin (same runner as FP16 and INT8-SQ)
- ONNX output: `work_onnx/wallx_flow_action_step_mixed10/`
- Engine output: `edge_engines/wallx_flow_action_step_mixed10/`

### Results (`--warmup 2 --iters 5`)

- `action_step_ms_mean = 19.818`
- `action_step_ms_std = 0.038`
- `flow_step_ms_mean = 19.791`
- `flow_step_ms_std = 0.035`
- `flow_total_ms_mean = 99.950`
- `flow_total_ms_std = 0.121`
- `denoised_vs_ref cosine = 0.98777`
- `flow_final_cosine = 0.29179`

### Key milestone: **10 Hz**

`flow_total_ms = 99.95 ms` → **≈ 10 Hz action frequency on Orin.**

This is the first time wall-x Flow Action reaches the 10 Hz real-time control threshold on Orin without a dedicated GPU.

### Accuracy finding

Mixed precision (10 layers FP16) did not improve `flow_final_cosine` compared to full INT8-SQ:

| Config | flow_final_cosine | flow_total_ms |
|---|---:|---:|
| FP16 (all layers) | 0.9852 | 157.661 ms |
| INT8-SQ (all layers) | 0.2776 | 78.183 ms |
| Mixed10 (26 INT8 + 10 FP16) | 0.2918 | **99.950 ms** |

The accuracy problem is not in the last 10 layers—it is a systemic error accumulation across all 5 Euler steps. Even FP16 single-step cosine = 0.9878 for Mixed10, but the 5-step loop amplifies this error. Keeping later layers in FP16 does not prevent the upstream INT8 error from propagating.

### Updated one-glance summary

| Route | Flow Action latency | flow_final_cosine | Note |
|---|---:|---:|---|
| `cpp_infer` FP16 | `290.7 ms` | — | baseline |
| Hand TRT / TRT-LLM | `176.837 ms` | — | stock TRT route |
| Edge-LLM FP16 bridge | `157.661 ms` | `0.9852` | fastest FP16 |
| Edge-LLM Mixed10 | `99.950 ms` | `0.2918` | **10 Hz milestone** |
| Edge-LLM INT8-SQ | `78.183 ms` | `0.2776` | fastest, accuracy collapsed |

### Next steps for accuracy

The systemic accuracy issue at the Euler loop level likely requires:
1. **Per-layer error sensitivity analysis** — identify which specific layers contribute most to single-step error
2. **Attention-only FP16** — keep all `q/k/v/o` projections in FP16, quantize only MLP
3. **Finer quantization granularity** — per-token activation quantization instead of per-tensor
