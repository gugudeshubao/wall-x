# Orin Runtime Notes

## Benchmarks

- `bf16` VQA (`fruits_on_table`, `max_new_tokens=20`, `benchmark=5`)
  - total `1539.98 ms`
  - ViT `551.303 ms`
  - prefill `137.704 ms`
  - decode `850.519 ms`
  - `12.99 tok/s`
- `INT8` VQA (`fruits_on_table`, `max_new_tokens=20`, `benchmark=5`)
  - total `1198.07 ms`
  - ViT `508.621 ms`
  - prefill `87.917 ms`
  - decode `601.061 ms`
  - `16.69 tok/s`
- `INT8` speedup vs `bf16`: `1.29x`

## Accuracy

- `C++ bf16 vs Python baseline`: `5/8` exact / text match
- `C++ INT8 vs Python INT8 baseline`: `1/8` exact / text match

### INT8 diagnosis

- `real_tabletop_2`
  - `19/20` token match
  - first divergence at step 2
  - Python `61891`
  - C++ `4419`
  - logit diff `0.250`
- `dual_arm_robot`
  - `14/20` token match
  - first divergence at step 2
  - Python `4933`
  - C++ `61891`
  - logit diff `0.250`
- `robot_gripper`
  - `3/20` token match
  - first divergence at step 2
  - Python `18689`
  - C++ `374`
  - logit diff `1.500`

### Vision isolation

- `robot_gripper`
  - Python `INT8 image_embeds` fed into `C++ INT8` gives exact token match
  - So the main error source is the C++ vision path, not decoder
- vision layer compare summary:
  - `after_reorder`: exact
  - `block0`: near-exact
  - `block7`: small drift starts
  - `pre_merger`: drift grows
  - `final image_embeds`: drift becomes visible in tokens

### Deeper vision compare (INT8)

- `robot_gripper` `block15` compare:
  - `vision_block15_q / k / v`: near-exact
  - `vision_block15_q_rot / k_rot`: small drift
  - `vision_block15_attn_out`: still close
  - `vision_block15_mlp_out`: drift grows more clearly
  - `vision_block15_output`: still close overall, but larger than `block7`
- Interpretation:
  - the INT8 error is still mostly a vision-side accumulation problem
  - the later vision MLP appears to contribute more error than attention at block15

### Block15 summary (robot_gripper)

- `vision_block15_q / k / v` are still near-identical between Python INT8 and C++ INT8
- `vision_block15_q_rot / k_rot` diverge a bit more than block7, but still not enough to explain the final token drift alone
- `vision_block15_mlp_out` is the first place where the drift starts to look visibly larger than the attention output
- `vision_block15_output` remains close in cosine, but the accumulated absolute error is now clearly larger than `block7`
- Overall conclusion:
  - later vision blocks continue to accumulate error
  - MLP is the more likely amplifier than attention in the later half of the vision stack

## Flow Action — INT8 Quantization Benchmarks

### INT8-SQ (all 36 layers quantized)

Engine: `edge_engines/wallx_flow_action_step_int8sq`
Warm benchmark (`--warmup 2 --iters 5`):

- `action_step_ms_mean = 15.383`
- `action_step_ms_std = 0.032`
- `flow_total_ms_mean = 78.183`
- `flow_total_ms_std = 0.313`
- `denoised_vs_ref cosine = 0.99081`
- `flow_final_cosine = 0.27758`

Speedup vs FP16 bridge (157.661 ms): **2.02x**
Accuracy: single-step near-exact but flow_final collapses after 5 Euler steps.

### Mixed Precision — 26 INT8 + 10 FP16 (layers 26–35 kept FP16)

Engine: `edge_engines/wallx_flow_action_step_mixed10`
Export: PyTorch 2.13 on 5090 → build on Orin
Warm benchmark (`--warmup 2 --iters 5`):

- `action_step_ms_mean = 19.818`
- `action_step_ms_std = 0.038`
- `flow_total_ms_mean = 99.950`
- `flow_total_ms_std = 0.121`
- `denoised_vs_ref cosine = 0.98777`
- `flow_final_cosine = 0.29179`

**Milestone: 99.95 ms ≈ 10 Hz on Orin.**

### Flow Action full comparison table

| Route | flow_total_ms | flow_final_cosine | Note |
|---|---:|---:|---|
| `cpp_infer` FP16 | `290.7 ms` | — | baseline |
| Hand TRT / TRT-LLM | `176.837 ms` | — | stock TRT |
| Edge-LLM FP16 bridge | `157.661 ms` | `0.9852` | fastest FP16 |
| Edge-LLM Mixed10 | `99.950 ms` | `0.2918` | **10 Hz** |
| Edge-LLM INT8-SQ | `78.183 ms` | `0.2776` | fastest, acc. collapsed |

---

### Block23 summary (robot_gripper)

- `vision_block23_q / k / v` remain near-identical
- `vision_block23_q_rot / k_rot` still show only small drift
- `vision_block23_attn_out` stays close to Python INT8
- `vision_block23_mlp_out` drifts more than `block15_mlp_out`
- `vision_block23_output` is still high-cosine overall, but the mean absolute error keeps growing
- Combined reading:
  - the error keeps accumulating through the later vision blocks
  - the later MLPs continue to look like the dominant amplifier, not attention
