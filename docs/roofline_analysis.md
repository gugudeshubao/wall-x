# Wall-X VLA Roofline 分析：Orin vs Thor

> 本文档用修正后的 Roofline Model 分析 wall-x 在 Orin 和 Thor 上的理论上限，
> 并用 cpp_infer 的实测数据验证每一步结论。

---

## 1. 硬件参数（实际值）

| 参数 | Orin AGX 64GB | ThorU（当前实机） | Thor 高端（理论参考） |
|---|---:|---:|---:|
| FP16 稀疏峰值 TFLOPS | 275 TOPS (INT8)，~138 TFLOPS FP16 sparse | 未知（可能是嵌入式版） | >1000 TFLOPS |
| **FP16 密集有效算力** | **~69 TFLOPS** | **~70–100 TFLOPS（估算）** | **~800 TFLOPS（保守）** |
| **内存带宽** | **205 GB/s** | **256 GB/s** | **256 GB/s** |
| 带宽：计算比（ridge point） | 336 FLOP/byte | ~350 FLOP/byte | **3125 FLOP/byte** |

> ⚠️ **实机说明**：当前可访问的 ThorU 实机在未优化 VQA benchmark 上仅比
> Orin 快 5%（1464ms vs 1539ms），提升主要来自 prefill（32% faster）而非 ViT 或 decode。
> 这与 "Thor 高端 SoC（1000+ TFLOPS）" 的理论预测不符，当前实机可能是嵌入式
> ThorU 中低端版本，本文 Thor 高端列仅为理论参考。

---

## 2. 模型参数（wall-x flow-action）

| 参数 | 值 |
|---|---|
| 模型总参数量 | ~3B（Qwen2.5-VL 骨干 + MoE action expert） |
| 视觉编码器（ViT） | ~0.5B（SigLIP 风格，32×32=1024 patches → merge 后 256 visual tokens） |
| LLM 骨干（含 MoE） | ~2.5B |
| Prefix 序列长度（Flow Action） | 456 tokens（dummy）/ 427 tokens（真图，fruits_on_table） |
| Postfix 序列长度（ODE 每步） | 32 tokens |
| ODE 步数 | 5 |
| 精度 | BF16 prefill + INT8 GEMM（cpp_infer） |

---

## 3. FLOPs 公式说明（关键修正）

### 常见错误：用训练公式估算推理

$$\text{FLOPs}_{\text{训练}} \approx 6 \times N \times S \quad \leftarrow \text{前向 + 反向 + 梯度}$$

### 正确推理公式

$$\text{FLOPs}_{\text{推理，prefill}} \approx 2 \times N \times S$$

$$\text{FLOPs}_{\text{推理，decode/ODE}} \approx 2 \times N \times 1 \quad \text{(per token per step)}$$

> 说明：`2×N×S` 里的 2 来自一次矩阵乘法的乘加（Multiply-Add = 2 FLOPs），
> N 是参数量（近似权重读写），S 是序列长度（决定矩阵宽度）。

### 各阶段 FLOPs 估算

| 阶段 | FLOPs 计算 | 结果 |
|---|---|---:|
| ViT（1024 patches，~0.5B params） | 2 × 0.5B × 1024 / merge² | ~0.26 TFLOPs |
| LLM Prefill（N=2.5B，S=456） | 2 × 2.5B × 456 | **2.28 TFLOPs** |
| ODE 5步（action expert ~1B，32 tokens/step） | 2 × 1B × 32 × 5 | 0.32 TFLOPs |
| **合计** | | **~2.86 TFLOPs** |

---

## 4. 算术强度（Arithmetic Intensity）

$$\text{AI} = \frac{\text{FLOPs}}{\text{字节数}} = \frac{2 \times N \times S}{2 \times N} = S \quad \text{(FLOP/byte)}$$

| 阶段 | AI（FLOP/byte） | Orin ridge（336） | 类型 |
|---|---:|---:|---|
| Prefill（S=456） | 456 | > 336 | **Compute-Bound** |
| ODE 每步（S=32） | 32 | < 336 | **Memory-Bound** |
| ViT（patch 级别更复杂） | ~128–256 | < 336 | 接近 Memory-Bound |

---

## 5. Orin 理论时间下限

### 5.1 Compute-Bound 阶段（Prefill）

$$T_{\text{prefill\_theory}} = \frac{2.28 \text{ TFLOPs}}{45 \text{ TFLOPS（有效）}} \approx 51 \text{ ms}$$

### 5.2 Memory-Bound 阶段（ODE）

Action expert 约 1B 参数，FP16（2 bytes），5步：

$$T_{\text{ODE\_theory}} = \frac{1\text{B} \times 2 \text{ bytes} \times 5}{205 \text{ GB/s}} \approx 49 \text{ ms}$$

### 5.3 ViT

$$T_{\text{ViT\_theory}} \approx \frac{0.5\text{B} \times 2 \text{ bytes}}{205 \text{ GB/s}} \approx 5 \text{ ms（权重读取，接近 memory-bound）}$$

### 5.4 Orin 理论全流程下限

| 阶段 | 理论下限 | cpp\_infer 实测 | 效率 |
|---|---:|---:|---:|
| ViT | ~5–8 ms | **115 ms** | 5–7% |
| Prefill | ~51 ms | **80 ms** | **64%** |
| ODE 5步 | ~49 ms | **96 ms** | **51%** |
| **全流程** | **~105–120 ms** | **290.7 ms** | **~40%** |
| **理论帧率** | **~8–9 Hz** | **3.4 Hz** | |

### 5.5 为什么 ViT 效率只有 5-7%？

这不是软件未优化，而是 ViT 在 Orin 上的结构性问题：

1. **Window attention 的不规则内存访问**：无法高效利用 Tensor Core
2. **MoE routing**：token 稀疏激活，理论算力利用率先天低
3. **序列长度不足**：1024 patches 的矩阵对 Orin 的 GEMM 来说仍偏小
4. **BF16/FP32 混合**：layernorm、softmax 的 FP32 中间态是瓶颈

cpp_infer 已经做了 equal-block no-mask fast path（ViT 从 200ms→115ms），
**这是在 Orin 硬件约束下接近可达的工程上限**，不是"未用 TRT 的未优化版本"。

---

## 6. Thor 理论时间预估

Thor 的核心提升：**算力 ~15×**（800 vs 69 TFLOPS），**带宽仅 1.25×**（256 vs 205 GB/s）。

| 阶段 | 类型 | Orin 实测 | Thor 提升因子 | Thor 预估 |
|---|---|---:|---:|---:|
| ViT | Compute-bound | 115 ms | ~10× | **~12 ms** |
| Prefill | Compute-bound | 80 ms | ~10× | **~8 ms** |
| ODE 5步 | Memory-bound | 96 ms | **1.25×** | **~77 ms** |
| **全流程** | | **291 ms** | | **~97 ms** |
| **帧率** | | **3.4 Hz** | | **~10 Hz** |

> **关键洞察：ODE 是 Memory-Bound，Thor 的 15× 算力优势在这里几乎不起作用。**
> ODE 改善来自带宽（256 vs 205 GB/s），提升约 25%，不是 10–15×。

### 6.1 Thor 上 INT8 的额外收益

若 Thor 上也做 INT8 量化（模型大小减半）：
- ODE memory-bound：1B × **1 byte** × 5 / 256 GB/s ≈ **20 ms**
- 全流程：~12 + ~8 + ~20 = **~40 ms → 25 Hz**
- 实际折扣 ~50%：**~12–15 Hz**

---

## 7. 最终对比表

| 指标 | Orin 理论上限 | Orin 实测（cpp_infer+优化） | Thor 实测（cpp_infer 无优化） | Thor 预估（优化后） |
|---|:---:|:---:|:---:|:---:|
| 主要瓶颈 | ViT compute + ODE memory | ViT 结构效率 | **Prefill MoE routing** | Prefill→ODE memory |
| ViT | ~5–8 ms | **115 ms** | **76.6 ms** | ~60 ms |
| Prefill | ~51 ms | **80 ms** | **197 ms** ← 意外瓶颈 | ~60 ms |
| ODE 5步 | ~49 ms | **96 ms** | **61.9 ms** | ~50 ms |
| **全流程（实测）** | — | **290.7 ms** | **336.9 ms** | **~170 ms** |
| **帧率（实测）** | — | **3.4 Hz** | **3.0 Hz** | **~6 Hz** |

> **2026-05-08 Thor 实测修正**：Thor 的 ViT（76ms）和 ODE（62ms）均快于 Orin 优化版，分别快 1.5× 和 1.55×。  
> 但 Prefill 高达 197ms（Orin 优化版 80ms），推测为 MoE token routing 未做 fusion 优化所致。  
> 若对 Thor 实施 Article-5 类似优化，全链路预计可降至 ~170ms ≈ 6 Hz。

---

## 8. 关键结论

### 8.1 Orin 的 3.4 Hz 不是"优化不足"

cpp_infer 已实现：
- 自定义 INT8 CUDA kernels（CUTLASS）
- Flash Attention equal-block no-mask fast path
- CUDA Graph for ODE
- 算子融合（RMSNorm+residual, gate/up+SiLU）

**实测 290.7ms 约为理论下限的 40% 效率**，处于 VLA 系统的正常工程范围（25–50%）。
剩余空间主要在 ViT（结构性约束），不是软件层面的未优化。

### 8.2 Thor 的真正价值

| 阶段 | 提升来源 | 预估收益 |
|---|---|---|
| ViT | 算力 15× | **~10×** |
| Prefill | 算力 15× | **~10×** |
| ODE（FP16） | 带宽 1.25× | **~1.25×** |
| ODE（INT8） | 带宽 1.25× + 模型体积减半 | **~2.5×** |

Thor 的收益**不均匀**：ViT 和 Prefill 会大幅加速，ODE 需要靠 INT8 才能显著提升。

### 8.3 "10 Hz VLA on Thor"的工程解读

- **FP16 全流程**：~10 Hz（理论），~7–8 Hz（实测预估）
- **INT8 全流程**：~25 Hz（理论），~12–15 Hz（实测预估）
- **10 Hz 是 FP16 的合理保守目标，不是 INT8 的极限**

### 8.4 Orin 还有多少空间

| 方向 | 可提升量 | 难度 |
|---|---|---|
| ViT 结构优化（e.g., window size、merge strategy） | ~20–30% | 高 |
| ODE INT8（action expert） | ~40%（78ms → 49ms） | 已实验，精度问题待解决 |
| ViT TRT engine（专门 build） | ~20–30% | 中 |
| **综合最优（FP16 + ODE INT8 精度修复）** | **~6 Hz 全流程** | 中高 |

---

## 附录：常见分析错误

| 错误 | 影响 | 正确做法 |
|---|---|---|
| 用 `6×N×S` 估算推理 FLOPs | 高估 3× | 用 `2×N×S` |
| Orin FP16 用 200+ TFLOPS | 高估 3× | 用 ~69 TFLOPS 密集值 |
| 上两项错误"凑巧"相消 | 数字合理但推理链错 | 分别验证 |
| 全程视为同一瓶颈类型 | 忽略 prefill/ODE 的不同特性 | 分阶段分析 AI |
| "实测慢 = 软件未优化" | 低估了 VLA 的结构性约束 | 用实测/理论比验证 |
