# wall-x cpp_infer → FlashRT 演进路线

日期：2026-05-08

> 本文只回答一个问题：**`cpp_infer` 当前已经走到哪了？下一步是不是就走到 FlashRT 路径？**

## 1. 一句话结论

> **是的，`cpp_infer` 在架构上已经是"准 FlashRT"，已完成 FlashRT 路径上 ~70% 的工作；剩下的差距集中在 FP8 / NVFP4 + 手写 attention，预期再吃 ~3-4× 加速空间。**

## 2. FlashRT 是什么

`FlashRT`（`github.com/LiangSu8899/FlashRT`）是当前 NVIDIA 生态里 **VLA 端侧推理性能最强的开源 runtime**，专门针对小 batch、低延迟。它在 Thor U / RTX 5090 上把 `Pi0.5 / Pi0 / GROOT N1.6 / Pi0-FAST` 跑出了远超官方 PyTorch eager / JAX 路径的延迟。

### FlashRT 实测性能

| 模型 | Hardware | Latency | Throughput |
|---|---|---:|---:|
| **Pi0.5** | Jetson AGX Thor (SM110) | **44 ms** (FP8) / **39.78 ms** (NVFP4) | **23 / 25 Hz** |
| **Pi0** | Jetson AGX Thor (SM110) | **46 ms** | **22 Hz** |
| **GROOT N1.6** | Jetson AGX Thor (SM110) | **45 ms** (T=50) / **41 ms** (T=16) | **22 / 24 Hz** |
| **Pi0-FAST** | Jetson AGX Thor (SM110) | **8.1 ms/token** | **123 tok/s** |
| **Pi0.5** | RTX 5090 (SM120) | **17.58 ms** | **57 Hz** |

**vs openpi 官方 JAX baseline @ Thor**：`714 ms (1.4 Hz) → 44 ms (23 Hz)`，**~16-18× 加速，cosine ≥ 0.9996 零精度损失**。

### FlashRT 不在 Orin 上跑

- 已验证硬件：**RTX 5090 / 4090 / 4060 Ti / Jetson AGX Thor**
- Orin (SM87) **不在已验证清单**，作者在 README 主动征集 AGX Orin 数据
- 即使能跑，因为 SM87 无 FP8 / NVFP4 / FA2，理论估算 Orin Pi0.5 上限 **~3-4 Hz**，**跑不到 10 Hz**

## 3. FlashRT 的 9 项核心技术

| # | 技术点 | 作用 |
|---|---|---|
| 1 | **静态 CUDA Graph 全图捕获** | 整个 forward (ViT + prefix LLM + diffusion loop) 捕获成一张图，零 Python overhead |
| 2 | **手写 CUDA kernel** | norm / activation / RoPE / qkv-split / quant 全部手写，针对小 batch 极致优化 |
| 3 | **Kernel Fusion** | residual + RMSNorm + quant 单 kernel，geglu + 双 GEMM 融合 |
| 4 | **FP8 (E4M3) GEMM** | cuBLASLt FP8 / CUTLASS SM100 FP8，自动 per-tensor 校准 |
| 5 | **NVFP4 GEMM** | Blackwell block-scaled FP4 (block=16, UE4M3 scales)，encoder FFN |
| 6 | **AWQ 量化** | activation-aware weight quantization，保住 FP4 多层精度 |
| 7 | **Vendored Flash Attention 2** | RTX 卡走 FA2，Thor U 走 cuBLAS-decomposed FMHA |
| 8 | **safetensors / Orbax 直接加载** | 无 ONNX export、无 TRT engine 编译，首次 ~3s 后全是 graph replay |
| 9 | **C++ Runtime + 3 行 Python API** | 部署形态干净，绕开 Python eager dispatch |

## 4. wall-x cpp_infer 当前已做的事

通过审计 `cpp_infer/src/` 代码，已经实现的能力：

| FlashRT 技术 | cpp_infer 状态 | 证据 |
|---|---|---|
| **C++ Runtime** | ✅ **已做** | `main.cpp` (782 行) + `model.cpp` (983 行) 纯 C++，绕开 Python eager |
| **静态 CUDA Graph 全图捕获** | ✅ **已做（Flow Action）** | `model.cpp:134/254` 用 `at::cuda::CUDAGraph::capture_begin/end` 把整个 ODE loop 捕获成图 |
| **手写 Triton Fusion kernel** | ✅ **已做** | `triton_kernels/fused_add_rmsnorm.py`、`fused_silu_mul.py` |
| **手写 CUDA kernel** | ✅ **已做** | `cpp_infer/src/kernels/`：`norm_kernels.cu` / `activation_kernels.cu` / `int8_kernels.cu` |
| **CUTLASS 量化 GEMM** | ✅ **已做（INT8 W8A8）** | `cutlass_int8_gemm.cu` fused INT8 GEMM + dequant epilogue，per-token act + per-channel weight |
| **MoE / GroupedGEMM** | ✅ **已做** | `moe.cpp` + `csrc/dual_asym_grouped_gemm.cu` 双专家 MoE |
| **safetensors 直接加载** | ✅ **已做** | `weight_loader.cpp` 跳过 ONNX/engine 编译 |
| **任务专用 ODE/Euler runtime** | ✅ **已做** | `ode_solver.cpp` host 侧 Euler loop |

**`cpp_infer` 的 Flow Action 实测**：**290.7 ms**（当前最强可控基线）

**和 FlashRT 同硬件（Thor U）对比**：

| 路线 | Pi0.5 / Flow Action 延迟 | 帧率 |
|---|---|---|
| openpi JAX (官方 baseline) | 714 ms | 1.4 Hz |
| openpi PyTorch eager | 276 ms | 3.6 Hz |
| **wall-x `cpp_infer` (BF16+INT8)** | **290.7 ms** | **3.4 Hz** |
| 手工 TRT / TRT-LLM | 176.8 ms | 5.7 Hz |
| TensorRT-Edge-LLM custom bridge | 157.7 ms | 6.3 Hz |
| **FlashRT @ Thor U (FP8)** | **44 ms** | **23 Hz** |
| **FlashRT @ Thor U (NVFP4)** | **39.78 ms** | **25 Hz** |

`cpp_infer` 与 FlashRT 的差距：**~6.6×**。

## 5. cpp_infer 还差什么（优化路线图）

| 缺口 | 当前 cpp_infer | FlashRT 的做法 | 预期收益 | 工程量 |
|---|---|---|---|---|
| **数值精度** | BF16 + INT8 W8A8 | **FP8 E4M3 W8A8** | ~1.8-2.0× | 中（CUTLASS FP8 GEMM，已有 INT8 模板可复用） |
| **encoder FFN 精度** | BF16 | **NVFP4 + AWQ + P1 split-GU** | ~1.3-1.5× | 中（cuBLASLt `_scaled_mm` 直接调，已实证 Thor U 675 TFLOPS 可用） |
| **Attention** | `torch::scaled_dot_product_attention`（cuDNN/SDPA） | Thor 专用 cuBLAS-decomposed FMHA | ~1.2-1.5× | 中-高 |
| **VQA decode CUDA Graph** | scaffold 已写但 disabled (`model.cpp:174 return false`) | Per-step graph capture loop | ~1.2-1.5×（仅 VQA） | 低（修稳定性即可） |
| **Vision encoder graph capture** | 未捕获 | 整张 ViT 进 graph | ~1.1-1.3× | 中 |

## 6. 你的下一步（按 ROI 排序）

### Step 1：跑一遍 FlashRT 拿天花板基线（1 天）

**目的**：确定 Thor U 上 `Pi0.5` 的物理极限，知道 `cpp_infer` 还有多少空间可挖。

```bash
# Thor U 上（ARM64，不能用 docker image）
git clone https://github.com/LiangSu8899/FlashRT.git
cd FlashRT
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
pip install -e ".[torch]"
cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j$(nproc)

# 跑 Pi0.5 benchmark
python examples/quickstart.py \
  --checkpoint /path/to/pi05_checkpoint \
  --benchmark 20
# 期望：P50 ~44 ms (23 Hz)
```

**注意**：FlashRT 不是 wall-x 的模型，不能直接在 wall-x 上跑。这一步是**拿天花板基线 + 学 kernel 写法**，不是替换 cpp_infer。

### Step 2：用 nsys 拆 cpp_infer 当前 Flow Action 的瓶颈（0.5 天）

```bash
nsys profile --stats=true --output=cppinfer_flow.qdrep \
  ./build/wallx_infer --benchmark
```

预期看到：BF16 GEMM 占大头（attention QKV / MoE FFN），Python overhead 已经被 CUDA Graph 吃掉。

### Step 3：FP8 W8A8 替换 INT8 W8A8（2-4 周）

`cpp_infer/src/kernels/cutlass_int8_gemm.cu` 已经是完整的 CUTLASS GEMM + dequant epilogue 模板。改造成 FP8：

- `cutlass::int8_t` → `cutlass::float_e4m3_t`
- per-token symmetric INT8 scale → per-tensor FP8 scale
- `int8_quant::linear` → `fp8_quant::linear`
- 走 cuBLASLt 路径作为 fallback

参考：你已有的 `Thor_U_NVFP4_完整测试报告.md` §5 已经实证 Thor U FP8 cuBLASLt 中尺寸 (4K-8K) = **246-308 TFLOPS**，立即可用。

**预期**：`cpp_infer` 290 ms → **~145-160 ms**（5-7 Hz）

### Step 4：encoder FFN 上 NVFP4（2-3 周）

`Thor_U_NVFP4_完整测试报告.md` §0 已经实证：

- cuBLASLt 黑盒 NVFP4 = **675 TFLOPS @ 16K（65% 利用率）**，**V24 数值验证 8/8 全正确**，**生产首选**
- `torch._scaled_mm(..., scale_a, scale_b, dtype=float4_e2m1fn_x2)` 直接可调

只在 Vision encoder FFN（不在 attention QKV、不在 LM head）上替换：

**预期**：`cpp_infer` 145 ms → **~100-120 ms**（8-10 Hz）

### Step 5：启用 VQA decode CUDA Graph（1 周）

`model.cpp:174` 当前 `return false` 直接 disable。scaffold 已写完，主要解决 KV cache 增长 + graph 形状不变的边界。

**预期**：VQA 端到端 851 ms → **~600-700 ms**

### Step 6（可选，长期）：手写 Thor 专用 FMHA

只有在 Step 3-5 都吃完之后再考虑。FlashRT 在 Thor 上走的是 cuBLAS-decomposed FA，不是真 FA2，工程量很大、收益相对较小。

## 7. 路线对比图

```
当前 cpp_infer (290 ms / 3.4 Hz)
        ↓ Step 3: FP8 W8A8 替换 INT8
~145-160 ms / 6-7 Hz
        ↓ Step 4: encoder FFN NVFP4
~100-120 ms / 8-10 Hz   ← 自研路径的合理目标
        ↓ Step 5+: VQA decode graph + FMHA
~70-90 ms / 11-14 Hz    ← 自研路径的极限

—— FlashRT @ Thor U (44 ms / 23 Hz) ——   ← 物理极限参考
```

## 8. 决策建议

### 8.1 短期（1 个月内）

> **先跑 FlashRT 拿天花板基线，再回头吃 FP8 + NVFP4。**

- FlashRT @ Thor U 跑通：~1 天
- nsys 拆 cpp_infer 瓶颈：0.5 天
- FP8 W8A8 替换 INT8：2-4 周
- 单 Step 3 就能让 `cpp_infer` 进入 5-7 Hz，已经能覆盖大部分操控场景

### 8.2 中期（2-3 个月）

> **NVFP4 上 encoder FFN，把 cpp_infer 推到 8-10 Hz。**

- 这个目标已经能满足 wall-x 部署的实际控制需求
- 不必硬追 FlashRT 的 23 Hz（FlashRT 的 23 Hz 是 Pi0.5 不是 wall-x，wall-x 的 MoE / Flow Action 控制结构更复杂）

### 8.3 是否要直接换 FlashRT runtime

**不要直接换**，原因：

- FlashRT 不支持 wall-x 模型（只支持 Pi0/Pi0.5/GROOT/Pi0-FAST）
- wall-x 的 `Flow Action` 有 MoE、prefix/postfix、`moe_token_types`、`ActionProcessor.step()` 等控制结构，不是标准 VLA
- FlashRT 的价值是**当作 reference / 天花板基线 + kernel 写法参考**，不是当作直接替换的 runtime

**正确姿势**：FlashRT 是教科书，cpp_infer 是你自己的 runtime，吃 FlashRT 的优化思路（FP8/NVFP4/Fusion），保留 wall-x 的控制结构。

## 9. 最短结论

> **`cpp_infer` 在架构上已经走在 FlashRT 路径上（C++ runtime + 静态 CUDA Graph + 手写 Fusion + CUTLASS 量化 GEMM + safetensors 直载，6/9 项已对齐）。下一步不是"重新走一条 FlashRT 的路"，而是把 cpp_infer 的 BF16+INT8 升级到 FP8+NVFP4，预期 290 ms → 100-120 ms（8-10 Hz）。FlashRT 当作天花板参考和 kernel 写法教科书，不当作直接替换。**

## 10. 参考

- FlashRT 仓库：[LiangSu8899/FlashRT](https://github.com/LiangSu8899/FlashRT)
- cpp_infer Flow Action graph 捕获：`cpp_infer/src/model.cpp:134/254`
- cpp_infer CUTLASS INT8 GEMM：`cpp_infer/src/kernels/cutlass_int8_gemm.cu`
- Thor U FP8/NVFP4 算力实证：`vibecode/docs/4leg/Thor_U_NVFP4_完整测试报告.md`
- openpi vs cpp_infer 三方对比：`workspace/trt_spike/docs/analysis/edge_llm_vs_wallx_vs_cppinfer.md`
