# ThorU Runtime Notes

## Login / Environment

- Host: private, see external local context
- SSH port: private, see external local context
- Auth: private, see external local context
- Workdir: `/home/user/wy`
- GPU: `NVIDIA Thor`
- CUDA: `13.0`
- Python: `3.12.3`
- `wallx-venv`: `/home/user/wy/wallx-venv`

## Packages

- `torch 2.11.0+cu130`
- `torchvision 0.26.0+cu130`
- `transformers 4.57.6`
- `accelerate 1.10.1`
- `peft 0.17.1`
- `scipy 1.15.3`
- `torchdiffeq 0.2.5`
- `qwen_vl_utils 0.0.11`
- `safetensors`
- `sentencepiece`
- `diffusers 0.37.1`

## Build / Runtime

- `wall_x` editable install works
- `Qwen2_5_VLMoEForAction` import works
- `wallx_infer` builds with `CUDACXX=/usr/local/cuda-13.0/bin/nvcc` and `CUDA_HOME=/usr/local/cuda-13.0`

## Benchmarks

### cpp_infer BF16（旧 ThorU，未优化，benchmark=5）

- `bf16` VQA (`fruits_on_table`, `max_new_tokens=20`)
  - total `1464.87 ms`，ViT `526.508 ms`，prefill `93.038 ms`，decode `845.165 ms`，`13.65 tok/s`
- `INT8` VQA (`fruits_on_table`, `max_new_tokens=20`)
  - total `1280.05 ms`，ViT `509.712 ms`，prefill `61.638 ms`，decode `708.537 ms`，`15.62 tok/s`
- `INT8` speedup vs `bf16`: `1.14x`

---

### cpp_infer BF16（新 Thor，无 Article-5 优化，benchmark=5，2026-05-08 实测）

#### Flow Action（ViT + Prefill + ODE 5步，全链路）

| 阶段 | 均值 |
|---|---:|
| ViT | 76.6 ms |
| Prefill | 197.2 ms |
| ODE 5步 | 61.9 ms |
| **Flow Action 总计** | **336.9 ms（≈ 3.0 Hz）** |

- Run 1-5: 336.5 / 337.1 / 337.2 / 336.7 / 337.0 ms（极稳定，std < 0.3ms）
- CUDA Graph 关闭（`WALLX_DISABLE_FLOW_GRAPH=1`，PyTorch 2.11 兼容性限制）

#### VQA（max_new_tokens=20）

| 阶段 | 均值 |
|---|---:|
| ViT | 76.9 ms |
| Prefill | 65.6 ms |
| Decode（20 tok） | 786.5 ms |
| **VQA 总计** | **929.2 ms（≈ 1.08 Hz）** |

- `21.52 tok/s`

#### 与 Orin 对比（Action 全链路）

| 阶段 | Thor BF16 无优化 | Orin INT8+优化 | 对比 |
|---|---:|---:|---|
| ViT | 76.6 ms | 115 ms | Thor **1.5× 快** |
| Prefill | **197 ms** | **80 ms** | Thor **2.5× 慢** ← 主瓶颈 |
| ODE 5步 | 61.9 ms | 96 ms | Thor **1.55× 快** |
| **总计** | **337 ms（3.0 Hz）** | **291 ms（3.4 Hz）** | Thor 略慢 16% |

**关键发现**：Thor 的 Prefill 显著慢于 Orin（197ms vs 80ms），原因是 MoE token routing 在 Thor 上没有做 fusion 优化。ViT 和 ODE 上 Thor 均有明显提升，但被 Prefill 拖累。  
预估：若对 Thor 实施同样的 Article-5 fusion 优化，Prefill 应能降至 ~60ms，全链路 **~200ms ≈ 5 Hz**。

### TRT Edge LLM Flow Action（ODE only，2026-05-08 实测）

使用 TRT Edge LLM custom action bridge（同 Orin 路线），仅 ODE 段：

| | Thor TRT Edge-LLM FP16 | Orin TRT Edge-LLM FP16 | 对比 |
|---|---:|---:|---|
| action_step mean | 24.5 ms | 31.3 ms | **1.28× 快** |
| flow_total (5步) | **122.3 ms** | **157.7 ms** | **1.28× 快** |
| Hz（ODE 复用 KV cache） | **8.2 Hz** | **6.3 Hz** | +1.9 Hz |

- 与 Roofline 预测一致：带宽比 = 256/205 = 1.25×，实测 1.28×（memory-bound）
- 注：使用 default CUDA stream，非最优；non-default stream 可再改善 5-10ms

---

## TRT Edge LLM Benchmark (2026-05-07/08)

### Setup
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Image: `fruits_on_table.png`, prompt: `Describe what you see in this image.`
- TRT Edge LLM built for `jetson-thor`, SM 10.1 hardware (TRT reports SM=101)
- FMHA: sm101 cubins compiled in (required patching cmake + cpp/CMakeLists.txt)

### Key build fix
- `AARCH64_BUILD` was undefined → cmake defaulted to x86 arch list, sm101 was excluded
- Fix: `-DAARCH64_BUILD=TRUE -DCMAKE_CUDA_ARCHITECTURES=80` + patch `cpp/CMakeLists.txt` to force sm101 inclusion for jetson-thor

### Results (warmup=1, runs=3)

- `warmup_ms = 9302`
- `run_1_ms = 9348`, `run_2_ms = 9325`, `run_3_ms = 9337`
- **mean = 9337 ms**

### Comparison with Orin TRT Edge LLM

| | Thor TRT Edge LLM | Orin TRT Edge LLM |
|---|---:|---:|
| VQA warm run | **9337 ms** | **7664 ms** |

Thor is ~22% slower than Orin on TRT Edge LLM VQA. This aligns with Roofline analysis: VQA decode is memory-bound, Thor's bandwidth advantage (1.25×) is limited.

---

## Accuracy

- `C++ INT8 vs Python INT8 baseline`: `1/8` exact / text match
- sample-level step compare:
  - `real_tabletop_2`: first divergence at step 2
  - `robot_gripper`: first divergence at step 16

