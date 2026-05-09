# 在 Orin 上给 wall-x 做 INT8：从 W8A8 落地到量化框架雏形

> 前三篇我们完成了：环境部署（第一篇）、Flash Attention 深挖（第二篇）、C++ 推理框架（第三篇，~557ms）。这一篇真正动刀量化后，故事并没有像“理论 2×”那样直线展开，而是分成了三个阶段：**第一阶段只量化 Attention / 部分 Vision Linear，端到端几乎不动；第二阶段补齐 ViT MLP 的 96 个漏网层，收益开始出现；第三阶段把 216 个 MoE expert projection 也拉进 INT8，Flow Action 从 568ms 压到 480ms，VQA（20 tokens）从 1316ms 压到 1013ms。** 这篇文章记录的，不再只是“INT8 为什么一开始几乎没有收益”，而是**量化覆盖率如何决定收益什么时候真正释放出来**。

**TL;DR**
- wall-x 3B 模型里，真正处在主推理热路径上的核心 Linear 层是 **522 个**：36 层 decoder 的 Attention 144 个 + MoE expert 216 个 + Vision 162 个
- 量化方案：**Per-channel 权重 INT8 + Per-token 动态激活 INT8**，Python 离线量化导出，C++ 在线推理
- **朴素 INT8 方案反而慢了 1.6 倍**（871ms vs `~560ms` bf16）——根因：INT32 输出带宽翻倍 + cuBLAS tile 选择不佳
- **CUTLASS 融合方案彻底解决**：用 EVT（Epilogue Visitor Tree）将 dequant 融合进 GEMM epilogue，INT32 不落地
- GEMM 微基准测试：CUTLASS INT8 比 bf16 快 **1.43-1.53 倍**（全矩阵尺寸）
- **第一阶段只量化 210 层时，端到端几乎持平**：Flow Action `568ms → 573ms`，VQA `1316ms → 1312ms`
- **补齐 ViT MLP 后覆盖率到 306 层，收益开始出现**：Flow Action `563ms`，VQA `1299ms`
- **MoE expert INT8 打通后覆盖率到 522 层，量化收益真正释放**：Flow Action `480ms (2.08 Hz)`，VQA `1013ms (19.75 tok/s)`
- **Amdahl 定律不是说“INT8 无效”，而是说“量化覆盖率不够时，收益会被 MoE 和逐元素 kernel 吃掉”**
- Orin SM 8.7 INT8 Tensor Core 的理论吞吐量是 bf16 的 **2 倍**（MMA 指令 m16n8k32 vs m16n8k16）
- **下一步优化**：CUDA Graph 消除 3.3 万次 kernel launch 开销、RMSNorm/Quant/GEMM/SiLU 融合、VQA decode 剩余同步开销

---

## 一、先算账：GEMM 到底有多少

wall-x 3B（wall-oss-flow）的模型结构：

```
Text Decoder: 36 层 Transformer
  ├── hidden_size = 2048
  ├── intermediate_size = 11008 (FFN)
  ├── num_attention_heads = 16, num_kv_heads = 2 (GQA)
  └── MoE: 2 experts (语言 expert intermediate=11008, 动作 expert intermediate=2048)

Vision Encoder: 32 层 ViT
  ├── hidden_size = 1280
  ├── intermediate_size = 3420 (MLP)
  └── PatchMerger: 1280×4 → 2048

Action Head: 小 MLP (w1, w2, w3)
  └── action_dim → 2048 → 2048
```

**每一层的 Linear 层清单（Decoder）**：

| Linear 层 | 维度 | 参数量 | GEMM 大小 |
|-----------|------|--------|-----------|
| q_proj | 2048 → 2048 | 4.2M | 中 |
| k_proj | 2048 → 256 | 0.5M | 小 |
| v_proj | 2048 → 256 | 0.5M | 小 |
| o_proj | 2048 → 2048 | 4.2M | 中 |
| moe.experts.0.gate_proj | 2048 → 11008 | 22.5M | **大** |
| moe.experts.0.up_proj | 2048 → 11008 | 22.5M | **大** |
| moe.experts.0.down_proj | 11008 → 2048 | 22.5M | **大** |
| moe.experts.1.gate_proj | 2048 → 2048 | 4.2M | 中 |
| moe.experts.1.up_proj | 2048 → 2048 | 4.2M | 中 |
| moe.experts.1.down_proj | 2048 → 2048 | 4.2M | 中 |

**36 层 Decoder 合计**：36 × (4 Attention + 6 Expert) = **360 个 Linear 层**。其中真正的大矩阵主体，不再是“共享 FFN”，而是 **MoE expert 0 的三组 2048×11008 / 11008×2048 GEMM**。

**32 层 ViT 合计**：32 × 5 = **160 个 Linear 层** + PatchMerger 2 个 = 162 个

**主推理热路径的核心 Linear 层总计**：360（Decoder）+ 162（ViT） = **522 个**

Action Head 还有 3 个小线性层，但它们不在这一轮量化目标里。也就是说，本文真正讨论的是：**怎样把这 522 个热路径 Linear 尽可能多地从 bf16 GEMM 变成 INT8 GEMM。**

---

## 二、核心问题：PyTorch、TensorRT、ONNX，选哪个？

这是量化方案的第一个分叉点。三条路线的本质区别是：

### 2.1 PyTorch eager 量化（torchao 风格 / 自定义脚本）

```
工作方式：
  1. 遍历模型的 nn.Linear / 热路径权重
  2. 计算权重 scale（可配合 Observer / 校准数据，也可直接 absmax）
  3. 保存量化后的权重表示（INT8 + scale + 可选 orig_shape）
  4. 推理时在 eager mode 下调用 INT8 GEMM（torch._int_mm / cublasLt / CUTLASS）

是否需要静态图：不需要
自定义算子影响：不受影响（只替换 Linear 层）
```

**优势**：
- **逐模块操作**，不碰模型整体结构，自定义 CUDA ops 原样保留
- 可以精确控制哪些层量化、哪些不量化（比如 Action Head 保持 bf16）
- 调试方便，可以逐层对比量化前后的输出差异
- 和第三篇的 libtorch C++ 框架无缝集成

**劣势**：
- PyTorch CUDA INT8 后端不如 TensorRT 调优深
- JetPack 定制版 PyTorch 的量化 API 支持可能有缺失

### 2.2 TensorRT 量化

```
工作方式：
  1. 导出模型为 ONNX 或通过 torch_tensorrt
  2. TensorRT 自动选择 INT8 kernel（SM 8.7 专用）
  3. 用 Calibrator 类 feed 校准数据
  4. 构建 INT8 engine

是否需要静态图：需要（ONNX 或 torch.export）
自定义算子影响：每个都要写 IPluginV2
```

**优势**：
- **Orin 上 INT8 性能最好**——NVIDIA 专门为 Jetson 调优
- 自动 kernel fusion + INT8，双重加速
- 内置校准器，使用简单

**劣势**：
- 6 个自定义算子要写 plugin（上一篇分析过：2-3 周工作量）
- 需要导出为静态图（ONNX 或 torch.export），wall-x 的 autograd function 导出不保证成功
- 黑盒优化，精度调试困难

### 2.3 ONNX Runtime 量化

```
工作方式：
  1. torch.onnx.export 导出 ONNX
  2. onnxruntime.quantization 做 PTQ
  3. 用 ONNX Runtime 的 CUDA EP 推理

是否需要静态图：需要（ONNX 格式本身就是静态图）
自定义算子影响：需要注册 ONNX 自定义 op
```

**优势**：
- 标准格式，工具链成熟
- 跨平台（不只 Orin）

**劣势**：
- ONNX Runtime 的 CUDA EP 在 aarch64 上测试不充分
- 自定义算子导出和注册都是额外工作
- INT8 Tensor Core 的利用率不如 TensorRT
- ONNX export 对 wall-x 的 autograd function 可能失败

### 2.4 结论

| 方案 | 需要静态图？ | 自定义算子处理 | Orin INT8 性能 | 实施难度 |
|------|-------------|---------------|---------------|----------|
| **PyTorch eager + 自定义离线导出** | **不需要** | **不受影响** | 良好 | **低** |
| TensorRT | 需要 | 要写 6 个 plugin | 最好 | 高 |
| ONNX Runtime | 需要 | 要注册 custom op | 一般 | 中高 |
| torch_tensorrt 混合 | 部分需要 | 标准层走 TRT | 好 | 中 |

**最终采用路线：保留 PyTorch eager 模型结构，但自己写离线量化导出脚本。**

也就是说，思路上和 `torchao` 很像：**不导出全图，不改控制流，只替换热路径 Linear 的权重表达和运行时实现**。但实际落地并没有直接依赖 `torchao` 模块，而是用了自定义 `safetensors` 导出脚本 + C++ `LinearOp`。原因很现实：

- `torchao` 在 JetPack / aarch64 上的可用性和行为不够稳定
- 我们还需要为 `vision_mlp` 这类不对齐矩阵保存 `weight_orig_shape`
- 后面 MoE expert 要切一条“bf16 dual_gemm / INT8 per-expert LinearOp”双路径，自己控权重格式更方便

这里需要额外强调一句：**这不是在“小路上自娱自乐”，而是在做第一阶段最务实的量化落地。** 从长期方向看，真正更完整的形态当然还是显式的图级量化表达，也就是让 `QuantizeLinear / DequantizeLinear` 这类边界直接出现在 IR 里，再交给 `TensorRT / TVM` 这类后端去做全局 pattern match、fusion 和 lowering；但在 `wall-x` 当前这类自定义算子很多、动态图控制流很重的模型上，先把热路径 `Linear` 的 `QDQ island` 手工打通，本身就是后面走向图级自动化之前必须跨过去的一步。

---

## 三、关键问题：需不需要展开为静态图？

**短答案：不需要。**

很多人一说"量化"就想到"先导出 ONNX/TensorRT engine"，这确实需要静态图。但 wall-x 的情况不一样：

### 3.1 为什么 wall-x 不适合全模型静态图导出

```python
# 试图导出整个模型：
exported = torch.export.export(model, example_inputs)

# 会在这些地方失败：
# 1. ops.permute() - 自定义 autograd function，不在 torch.export 的 op 集合里
# 2. ops.asym_dual_gmm() - 同上
# 3. ops.rot_pos_emb() - 同上
# 4. DynamicCache - 非 Tensor 对象
# 5. MoE if/else 分支 - 动态控制流
# 6. torchdiffeq.odeint() - 外部库调用
```

wall-x 的 6 个自定义算子、MoE 动态路由、ODE 积分——这些都是 `torch.export` 和 ONNX 导出的"雷区"。强行导出要么失败，要么需要大量 workaround。

### 3.2 逐模块量化：不需要静态图

正确的做法是 **只量化 Linear 层，不动模型整体结构**：

```python
# 不需要导出整个模型！只需要替换 Linear 层：
import torchao

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        if should_quantize(name):
            # 替换为 INT8 版本
            quantized = torchao.quantize_linear(module, method="int8_dynamic")
            set_module(model, name, quantized)
        # 不需要量化的层（Action Head）跳过
```

这个操作：
- **不改变模型的 forward 逻辑**——控制流、自定义算子、ODE 积分全部不受影响
- **只替换了 nn.Linear 的计算方式**——从 bf16 matmul 变成 INT8 matmul
- **不需要任何形式的图导出**——在 eager mode 下直接工作

### 3.3 什么时候才需要静态图

只有以下情况才需要：
1. **用 TensorRT 原生 INT8**——需要 ONNX 或 torch.export
2. **用 torch_tensorrt 编译子图**——需要 `torch.export` 部分子图
3. **用 ONNX Runtime 推理**——需要 ONNX 导出

这些都是 **可选的 Phase 2 优化**，不是量化的前提。

---

## 四、量化方法：先用纯 W8A8 动态量化，SmoothQuant 作为预案

### 4.1 为什么选 W8A8

Orin SM 8.7 的硬件约束直接决定了方法。这里先只看本文实际相关的 INT8 路径：

| 量化方式 | Orin 硬件支持 | 计算加速 | 内存节省 |
|----------|-------------|----------|----------|
| **W8A8** (INT8 权重 + INT8 激活) | INT8 Tensor Core | **2x** | 2x |
| W8A16 (INT8 权重 + FP16 激活) | dequant 到 FP16 | **无** | 接近 2x |

**W8A8 是 Orin 上唯一既省内存又加速计算的方案。** INT8 Tensor Core 吞吐量是 bf16 Tensor Core 的 2 倍。

### 4.2 SmoothQuant：解决激活值 outlier 的备选方案

直接对 Transformer 做 INT8 量化，最大的问题是 **激活值 outlier**——某些 channel 的激活值特别大（比其他 channel 大 10-100 倍），导致 INT8 动态范围不够。

SmoothQuant 的思路很优雅：**把激活值的 outlier "平滑"到权重上**。

```
原始计算：Y = X @ W

SmoothQuant 变换：
  s = max(|X|, axis=0) / max(|W|, axis=0)   # per-channel 平滑因子
  X_smooth = X / s
  W_smooth = W * s
  
  Y = X_smooth @ W_smooth = X @ W   # 数学上等价！

但量化友好度大幅提升：
  X_smooth 的 outlier 被压小了 → INT8 量化更准
  W_smooth 的范围变大了一点 → 但权重本来就比激活稳定
```

这个变换在数学上完全等价，但让激活值的 INT8 量化精度大幅提升。SmoothQuant 的论文报告 W8A8 精度损失在 1% 以内。

**但本文当前这版实测并没有启用 SmoothQuant。** 我们先用最简单的：

- 权重：per-channel 静态 INT8
- 激活：per-token 动态 INT8

先把 kernel、权重格式、C++ 运行时和端到端收益跑通。只有当某一批层的 cosine similarity 掉到不可接受，或者机器人任务成功率明显下降时，SmoothQuant 才作为下一层精度补救手段引入。

### 4.3 动态量化 vs 静态量化

| 维度 | 动态量化 | 静态量化 |
|------|----------|----------|
| 权重 | 提前量化为 INT8（静态 scale） | 同左 |
| 激活 | 推理时逐 token/逐 batch 计算 scale | 校准时确定固定 scale |
| 额外开销 | 每层多一次 `absmax()` 计算 scale | 无 |
| 精度 | **更好**（适应不同输入的激活范围） | 较差（固定 scale 可能截断） |
| 实现复杂度 | 中 | 高（需要精确校准） |

**当前实现采用动态量化**：权重静态 INT8（提前量化），激活逐 token 动态 INT8（推理时计算 scale）。多出的 `absmax()` 开销很小（几十微秒），但精度提升明显。

### 4.4 INT8 GEMM 在 Orin 上怎么跑

**朴素方案**（最初尝试）：

```
bf16 GEMM（基线）：
  cuBLAS → cublasGemmEx(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF)
  使用 bf16 Tensor Core

朴素 INT8 GEMM：
  cuBLAS → cublasLtMatmul(CUDA_R_8I, CUDA_R_8I, CUDA_R_32I)
  输出 INT32 → 写回 DRAM → 读回 → dequant → 写回 bf16
  ❌ 实测比 bf16 慢 1.6 倍（详见第十节）
```

**实际采用的 CUTLASS 融合方案**：

```
CUTLASS INT8 GEMM + EVT Epilogue：
  CUTLASS → GemmWithEpilogueVisitor
  输入 INT8 → MMA 指令（m16n8k32）→ INT32 累加器
  → 在寄存器内直接执行 dequant → 输出 bf16
  ✅ INT32 不落地，节省 2x 带宽
  ✅ 实测比 bf16 快 1.43-1.53 倍
```

核心是 CUTLASS 的 **EVT（Epilogue Visitor Tree）** 机制——把 `int32 * act_scale * weight_scale → bf16` 的 dequant 逻辑融合进 GEMM 的 epilogue 阶段，INT32 累加器直接在寄存器中完成类型转换，不经过全局内存。

### 4.5 Orin MMA 指令全景：INT8 天然有 2x 优势

Orin 的 SM 8.7 基于 Ampere 架构。下面只列出本文相关的 MMA 指令形状：

| 数据类型 | MMA 形状 | 每条指令运算量 | 相对吞吐量 |
|---------|----------|-------------|-----------|
| BF16 | m16n8k8 | 2,048 ops | 0.5× |
| **BF16** | **m16n8k16** | **4,096 ops** | **1×（基线）** |
| FP16 | m16n8k16 | 4,096 ops | 1× |
| **INT8** | m16n8k16 | 4,096 ops | 1× |
| **INT8** | **m16n8k32** | **8,192 ops** | **2×** |

**关键结论**：
- INT8 的 `m16n8k32` 指令每次处理 32 个 K 维元素（bf16 只处理 16 个），**同一条 MMA 指令的运算量翻倍**
- 这不是"多发几条指令"的假加速，而是**单条指令真正多算一倍**

**MMA 指令不是瓶颈**。INT8 在硬件层面确实有 2× 吞吐优势。真正的问题出在软件层面（朴素实现的 INT32 带宽、tile 选择），以及第一阶段 partial INT8 条件下的 Amdahl 限制。

---

## 五、分层量化策略：不是所有层都该量化

### 5.1 量化优先级

| 组件 | 量化建议 | 原因 |
|------|----------|------|
| **MoE 语言 Expert** | **INT8** | 最大 GEMM（2048×11008 / 11008×2048），收益最大 |
| **MoE 动作 Expert** | **INT8** | 虽然矩阵更小，但层数多；实测 layer-wise cosine similarity 仍然很高 |
| **Decoder Attention** (QKV, O) | **INT8** | 中等 GEMM，层数多 |
| **Vision Encoder** | **INT8** | 32 层 ViT，中等收益 |
| **Action Head** (w1/w2/w3) | **bf16 保持** | 小 MLP，ODE 积分会放大误差 |
| **Normalizer** | **不量化** | min/delta 统计量，必须保持精度 |
| **Embedding / LM Head** | **bf16 保持** | 量化收益小，但影响 token 预测 |

### 5.2 为什么 Action Head 不能量化

这是 wall-x 量化最关键的决策。Flow Action 的推理流程：

```
noise → [Euler step 1] → [Euler step 2] → ... → [Euler step 10] → 最终动作

每个 Euler step:
  1. ActionProcessor.step(timestep, noisy_action) → action_embed
     └── w1(noisy_action) → w2(concat(embed, time)) → w3(...)
     └── 这里的量化误差 ε₁
  
  2. Transformer forward → v_t
     └── 这里的量化误差 ε₂
  
  3. noisy_action += dt * v_t
     └── 误差累积：ε_total = Σ(ε₁ + ε₂) × dt × 10步
```

**误差放大机制**：
- 单步误差 ε = 0.1%（INT8 典型精度损失）
- 10 步 Euler 积分后：理论上最坏情况 ε_total ≈ 10 × 0.1% = 1%
- 但如果误差有系统性偏移（不是随机的），实际可能更大
- 对机器人控制来说，1% 的动作偏差可能导致抓取失败

**安全策略**：Decoder 内部的 Transformer 层（包括 MoE expert）可以量化，因为它们输出的是中间隐状态；但 Action Head 的 `w1/w2/w3` 仍然保持 bf16，因为它们直接参与 ODE 状态更新的 embedding 构造，误差路径更短、更敏感。

### 5.3 MoE 自定义 GEMM 的处理

wall-x 的 MoE 用 `asym_dual_gmm`（自定义 CUDA kernel）做双 expert GEMM。它最开始是本文里最大的障碍，但最后并没有靠“重写 kernel”解决，而是靠**双路径运行时**绕开：

**方案 A：修改 `asym_dual_gmm` 支持 INT8**
- 改 CUDA kernel，输入从 bf16 改为 INT8
- 工作量大（改 kernel + 测试）
- 性能最好

**方案 B：保留 permute/unpermute，但 expert 内部改走 `LinearOp`**
- bf16 权重时：继续走 `asym_dual_gmm` 的 dual-expert 并行路径
- INT8 权重时：保留 `moe_permute_topK_op` / `moe_recover_topK_op`
- 中间的 expert projection 改成 `gate_proj.forward()` / `up_proj.forward()` / `down_proj.forward()`
- 换句话说：**只把 expert 内部 GEMM 从“自定义 bf16 kernel”切到“每个 expert 单独的 INT8 LinearOp”**

本文最终采用的是 **方案 B**。它的性能上限不一定比“重写 INT8 dual_gmm kernel”高，但工程复杂度小很多，而且已经足够把 216 个 expert projection 拉进 INT8 路径。

---

## 六、量化变换的算子图和权重陷阱

量化不是"替换 Linear 层就完事"。wall-x 的自定义算子和模型结构会引入几个容易忽略的问题。

### 6.1 SmoothQuant 如果启用，会永久修改权重

虽然本文当前实测没启用 SmoothQuant，但如果后续引入，它仍然会带来一个重要工程后果：`W_smooth = W * s` 会永久改变权重值。后果：
- 原始 bf16 checkpoint 和量化后的权重 **对不上**
- 不能直接用原始 checkpoint 加载量化模型
- 模型重新训练后，SmoothQuant 的 scale `s` 要重新计算

**解决方案**：量化后另存一份 checkpoint（INT8 权重 + per-channel scales + smooth scales），原始 checkpoint 不动。

### 6.2 `asym_dual_gmm` 不支持 INT8——最大的坑，但不是死路

wall-x 的 MoE 层用自定义 CUDA kernel `asym_dual_gmm` 做双 expert GEMM。这个 kernel **只处理 bf16 张量**。如果你直接把 expert 权重量化为 INT8，再原样喂给它，结果要么报错，要么直接错。

但这里真正重要的结论不是“MoE 不能量化”，而是：

> **不能继续沿用原来的 dual-gemm bf16 调用方式。**

最后的解法是双路径：

- bf16 checkpoint：继续走 `asym_dual_gmm`
- INT8 checkpoint：保留 `permute / unpermute`，中间 expert projection 切到 `LinearOp`

也就是说，**MoE 的路由和 token 重排逻辑保留，只有 expert 内部的 GEMM 实现发生切换。**

### 6.3 自定义 RoPE 的 dtype 安全

量化后的 Q/K projection 输出应该是 bf16（dequant 后）。但如果量化实现有 bug，忘了 dequant 就传给 RoPE：

```python
q = quantized_q_proj(hidden_states)  # 如果返回 int32 而不是 bf16...
q = ops.rot_pos_emb(q, cos, sin)     # CUDA kernel 收到 int32 → crash
```

**防范**：在 attention forward 的关键位置加 dtype 断言：
```python
q = self.q_proj(hidden_states)
assert q.dtype == torch.bfloat16, f"q_proj output dtype mismatch: {q.dtype}"
```

### 6.4 state_dict key 变化

替换 `nn.Linear` 为量化版本后，`model.state_dict()` 的 key 会变（新增 `weight_scale`、`weight_zero_point` 等）。`model.load_state_dict(bf16_checkpoint)` 会报 key mismatch。

**解决流程**：先加载 bf16 checkpoint → 量化 → 保存新的量化 checkpoint。不要试图直接加载 bf16 权重到量化模型。

### 6.5 风险汇总

| 问题 | 严重性 | 防范措施 |
|------|--------|----------|
| `asym_dual_gmm` 不支持 INT8 | **高** | INT8 checkpoint 切换到 per-expert `LinearOp` 路径 |
| SmoothQuant 改了权重 | 中 | 另存量化 checkpoint |
| RoPE 收到错误 dtype | 中 | 加 assert 检查 |
| state_dict key 变化 | 中 | bf16 加载 → 量化 → 另存 |
| MoE permute/unpermute dtype | **低** | dequant 在 Linear 内部完成，外部 bf16 |

---

## 七、C++ 推理框架和量化的实际集成

第三篇的 C++ 替换和第四篇的 INT8 量化要叠加使用。核心挑战是：**torchao 的 Python 量化模块在 C++ libtorch 里不存在，且朴素 INT8 GEMM 在 Orin 上反而更慢。**

### 7.1 架构：Python 离线量化 + C++ 在线推理

```
阶段 1：Python 离线量化（在 5090 或 Orin 上，跑一次）
  ├── 加载 bf16 模型
  ├── 逐层计算 per-channel 权重 absmax
  ├── 权重量化 → round(W / scale * 127) → INT8
  └── 保存：
      model.layers.*.self_attn.*.{weight, weight_scale}
      model.layers.*.moe.experts.*.*.{weight, weight_scale}
      model.visual.blocks.*.{attn,mlp}.*.{weight, weight_scale}
      visual.*.mlp.*.weight_orig_shape   # 仅对 pad 后矩阵额外保存

阶段 2：C++ 在线推理（在 Orin 上运行）
  ├── 加载 INT8 权重 + weight_scale（普通 tensor）
  ├── 对每个 Linear 层调用 2-kernel 流水线：
  │     kernel 1: fused_quantize_activation() → act_int8 + act_scale
  │     kernel 2: cutlass_int8::gemm_dequant() → bf16 输出
  ├── 对 3420 这类不对齐层：运行时输入补零，输出再切回原始 shape
  └── 自定义算子（RoPE、MoE permute 等）照常 bf16；MoE expert 在 INT8 checkpoint 下改走 per-expert LinearOp
```

### 7.2 三条路线的探索历程

我们实际尝试了三种 INT8 GEMM 实现方式：

**路线 1：朴素方案（torch::_int_mm + 手动 dequant）**

```cpp
// 4 个 kernel：quantize_act + absmax + _int_mm + dequant
auto x_int8 = quantize(input);
auto out_int32 = torch::_int_mm(x_int8, weight_int8.t());  // cuBLAS INT8
auto out_bf16 = dequant(out_int32, act_scale, weight_scale);
// ❌ 结果：871ms（同时期 bf16 ≈ 560ms），慢 1.6 倍
```

慢的根因（nsys + ncu 确认）：
- INT32 输出带宽是 bf16 的 2 倍（4 bytes vs 2 bytes per element）
- cuBLAS 对 M=32 的 INT8 选了非最优 tile（128×64 而不是 256×128）
- splitK 策略引入额外 reduce kernel

**路线 2：fused kernels + cublasLt**

```cpp
// 2 个 kernel：fused_quantize + cublasLtMatmul(INT8→bf16)
auto [act_int8, act_scale] = int8_fused::quantize_activation(input);
auto out = CublasLtInt8Gemm::run(act_int8, weight, act_scale, weight_scale);
// ❌ GEMM 本身仍然比 bf16 慢（cublasLt tile 选择同样不佳）
```

Fused kernel 效果好（quantize 200us→16us），但 GEMM 部分没解决。

**路线 3：CUTLASS 融合 GEMM + EVT dequant（最终方案）✅**

```cpp
// 2 个 kernel：fused_quantize + CUTLASS GEMM with fused dequant
auto [act_int8, act_scale] = int8_fused::quantize_activation(input);
auto out = cutlass_int8::gemm_dequant(act_int8, weight_int8,
                                       act_scale, weight_scale);
// ✅ INT8 GEMM + dequant 融合为一个 kernel，INT32 不落地
```

### 7.3 CUTLASS EVT 融合 GEMM 的关键实现

用 CUTLASS 4.0 的 `GemmWithEpilogueVisitor` + `Sm80EVT`，把 dequant 逻辑编码为 Epilogue Visitor Tree：

```
EVT 树结构（寄存器内执行）：
  StoreD[bf16]
    └── Mul ← RowBroadcast[weight_scale]     // × per-channel weight scale
         └── Mul ← ColBroadcast[act_scale]    // × per-token activation scale
              └── AccFetch                     // INT32 累加器（寄存器）
```

CUTLASS 的 `VisitorCompute` 自动处理 INT32→float 的类型转换（通过 `NumericArrayConverter`），最终输出 bf16。整个 dequant 流程在 shared memory / registers 中完成，**INT32 累加器从不写回全局内存**。

```cpp
// 关键类型定义
using ElementA = int8_t;                    // RowMajor
using ElementB = int8_t;                    // ColumnMajor
using ElementAccumulator = int32_t;
using ElementOutput = cutlass::bfloat16_t;  // RowMajor

// GEMM 配置（为 Orin SM 8.7 调优）
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;  // INT8 MMA
constexpr int NumStages = 4;  // 4-stage pipeline

// EVT 定义
using EVT = cutlass::epilogue::threadblock::Sm80EVT<
    StoreD,           // output bf16
    MulWeightScale,   // × weight_scale[N]
    MulActScale,      // × act_scale[M]
    AccFetch          // INT32 accumulator
>;
```

### 7.4 C++ 侧的 INT8 Linear 集成

最终的 `LinearOp` 实现非常干净：

```cpp
struct LinearOp {
    torch::Tensor weight_int8_;    // [N, K] int8（原始 layout，CUTLASS ColumnMajor B）
    torch::Tensor weight_scale_;   // [N] float32
    torch::Tensor bias_;

    torch::Tensor forward(const torch::Tensor& input) {
        auto flat = input.reshape({-1, input.size(-1)});
        // kernel 1: fused quantize（per-token 动态量化）
        auto [act_int8, act_scale] = int8_fused::quantize_activation(flat);
        // kernel 2: CUTLASS fused INT8 GEMM + dequant
        auto out = cutlass_int8::gemm_dequant(
            act_int8, weight_int8_, act_scale, weight_scale_);
        if (bias_.defined()) out = out + bias_;
        return out.reshape(/*restore original shape*/);
    }
};
```

**只需 2 个 CUDA kernel**——fused quantize + CUTLASS fused GEMM，比朴素方案（4-5 个 kernel）减少了一半以上的 kernel launch 开销。

---

## 八、如果后续引入 SmoothQuant / QAT，校准数据怎么准备

先把边界说清楚：**本文最终跑通并得到 480ms / 1013ms 的这版实现，没有使用校准集。** 当前实装的是：

- 权重：per-channel absmax 静态 INT8
- 激活：per-token 动态 INT8
- 精度验证：逐层 cosine similarity + ODE chain simulation

下面这一节保留，是为了回答一个更长线的问题：**如果后面要继续叠 SmoothQuant、真正做 PTQ 校准，或者做 QAT，数据应该怎么准备。**

### 8.1 数据来源

wall-x 的训练数据是 LeRobot 格式：
```
每条样本包含：
  - image_inputs: 机器人视角图片（640×480 RGB）
  - text: 任务指令（"把红色方块放到篮子里"）
  - action: 动作轨迹 chunk（action_horizon × action_dim）
  - agent_pos: 机械臂关节状态（proprioception）
```

**校准数据只需要训练集的一个子集**：
- VQA 校准：128-256 条图文对（覆盖不同场景和指令类型）
- Flow Action 校准：128-256 条包含动作轨迹的完整样本
- **从训练集随机采样即可**，不需要额外标注

### 8.2 校准脚本框架

```python
# scripts/calibrate_int8.py（伪代码）
import torch
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight

# 1. 加载模型
model = load_wall_x_model(model_path)
model.eval().cuda().bfloat16()

# 2. 加载校准数据
calib_dataset = load_lerobot_subset(dataset_path, num_samples=256)

# 3. 定义哪些层需要量化
def filter_fn(module, name):
    # Action Head 不量化
    if "action_processor" in name:
        return False
    # Normalizer 不量化
    if "normalizer" in name:
        return False
    # Embedding 和 LM Head 不量化
    if "embed_tokens" in name or "lm_head" in name:
        return False
    # 其余 Linear 层量化
    return isinstance(module, torch.nn.Linear)

# 4. 应用 SmoothQuant + INT8 动态量化
quantize_(model, int8_dynamic_activation_int8_weight(), filter_fn=filter_fn)

# 5. 跑校准数据验证精度
for batch in calib_dataset:
    with torch.no_grad():
        output_quant = model(**batch)
        # 对比 bf16 baseline 的输出差异
```

### 8.3 精度验证指标

| 任务 | 指标 | 可接受阈值 |
|------|------|-----------|
| VQA | 生成文本 BLEU vs bf16 baseline | > 0.95 |
| VQA | Top-1 token accuracy | > 99% |
| Flow Action | 动作轨迹 MSE vs bf16 baseline | < 2x bf16 MSE |
| Flow Action | 任务成功率（如果有 eval 环境） | > 95% of bf16 |

**Flow Action 的精度验证比 VQA 重要得多**——VQA 的误差只影响文本质量，Flow Action 的误差直接影响机器人动作。

---

## 九、所需资源清单

这一节同样是 **“如果继续往 SmoothQuant / QAT 走”** 才需要的资源，不是复现本文当前 Stage 1/2/3 结果的必需条件。复现当前实现，实际上只需要：

- 现有的 PyTorch / safetensors 环境
- 离线量化导出脚本
- Orin 上的 C++ `wallx_infer`

### 9.1 硬件

| 阶段 | 硬件 | 你是否已有 | 用时 |
|------|------|-----------|------|
| PTQ 校准（SmoothQuant） | RTX 5090（单卡） | **已有** | 2-4 小时 |
| 精度验证（VQA） | RTX 5090 | **已有** | 1-2 小时 |
| 精度验证（Flow Action） | RTX 5090 + 仿真环境 | **需确认** | 半天 |
| INT8 部署 + Benchmark | Orin | **已有** | 半天 |
| QAT（如果 PTQ 不够） | 4-8 × A100/H100 | **需要租用** | 1-3 天训练 |

**PTQ 全流程只需要你现有的 RTX 5090 + Orin。** QAT 大概率不需要。

### 9.2 软件依赖

```
需要安装（在 5090 上）：
  pip install torchao           # PyTorch 官方量化库
  pip install smoothquant       # SmoothQuant 实现（可选，也可自己实现）

需要安装（在 Orin 上）：
  pip install torchao           # 需要确认 aarch64 兼容性
  # 如果 torchao 不支持 aarch64，回退到 torch.ao.quantization
```

### 9.3 校准数据

```
需要准备：
  - 256 条 LeRobot 格式样本（从训练集随机采样）
  - 写一个采样脚本（~50 行 Python）
  - 不需要额外标注或收集新数据
```

---

## 十、实战结果：从 1.6x 更慢，到真正破壁

### 10.0 测试用例说明

> **系列文章测试用例对比**——四篇文章使用了不同的测试场景和框架，数据不能直接横向对比：
>
> | | 第一篇（部署篇） | 第二篇（FA2 篇） | **第四篇（本文 INT8）** |
> |---|--------|--------|--------|
> | **测试脚本** | `test_vqa_bench.py` / `fake_inference.py` | `bench_fa2_vs_sdpa.py` | C++ `wallx_infer` |
> | **输入数据** | 8 张真实图片 640×480（VQA） / 50 token 随机 tensor（fake forward） | 1 张真实图片 640×480（单图单问） | 真实图片 VQA case / 固定 benchmark case（对齐真实 shape） |
> | **任务类型** | VQA 文本生成（128 tokens）/ fake forward（单次 Transformer） | VQA 文本生成（64 tokens） | **Flow Action + VQA staged INT8 benchmark** |
> | **推理方式** | Python `model.generate()` / `model()` | Python `model.generate()` | **C++ libtorch `generate_flow_action()` / `generate_text()`** |
> | **序列长度** | 420 tokens（VQA）/ 50 tokens（fake） | ~420 tokens | **488 tokens（Action） / 456+20 tokens（VQA）** |
> | **框架开销** | ~67% Python/HF 开销 | ~67% Python/HF 开销 | **无 Python 开销** |
> | **Orin 延迟** | 13,898ms（VQA）/ 115ms（fake forward） | 6,360ms（FA2 VQA） | **480ms（Action Stage 3） / 1013ms（VQA Stage 3, 20tok）** |
>
> ⚠️ 第一篇的 `fake_inference.py`（14.2ms / 115ms）是 50 token 随机 tensor 的单次 Transformer forward，**没有 ViT 编码、没有 ODE 积分、没有文本生成**，不代表任何真实任务的延迟。

第四篇的测试结果，实际上由**两类 case**组成：

- **真实图片 VQA case**：用于补充验证真实图推理链路上的延迟和答案一致性
- **固定 benchmark case**：用于把 Stage 1 / 2 / 3 的量化覆盖率变化放到同一套推理路径和张量尺寸下对比

其中，后者走的是和真实任务一致的推理链路，只是输入内容固定下来，便于专门观察 runtime 和量化收益。具体构成：

```
固定 benchmark case 构成（用于 staged latency 对比）：
  input_ids:       488 tokens = 200 文本(randint) + 256 图像(image_token_id) + 32 动作(action_token_id)
  pixel_values:    torch::randn, 1024 patches × (3×2×14×14), 对齐真实 vision patch 张量形状
  image_grid_thw:  [1, 32, 32] → 32×32 patches, merge 后 256 tokens
  moe_token_types: 前 456=0(文本/视觉), 后 32=1(动作)
  ODE timesteps:   5 步 Euler 积分
```

这组固定 benchmark case 回答的是：

- 在真实的 `generate_flow_action()` / `generate_text()` 路径上，INT8 覆盖率打到哪里，端到端延迟才开始下降
- `ViT / Prefill / Decode / ODE` 这些子阶段，哪一段真正吃到了量化收益
- 当前这套 `W8A8 + CUTLASS + LinearOp` 路线，在 Orin 上值不值得继续做 runtime 优化

它**不直接回答**的是：

- 真实图片和真实文本上的 VQA 生成质量有没有下降
- 真实机器人 / 仿真任务成功率有没有下降
- 更细粒度的任务指标（BLEU、token accuracy、TTFT、prefill tokens/s）有没有变化

所以更准确的说法是：**第四篇的主结论重点在 runtime 和量化收益评估；真实图片 case 则是补充验证，不和 staged benchmark 混为一谈。**

对 **延迟 benchmark** 来说，这组固定 case 是有效的——GEMM 耗时主要取决于矩阵尺寸 `(M×K×N)` 和执行路径，而不取决于具体数值；但它**不能替代真实图片和真实任务上的质量验证**。

当前已经完成的验证可以分成三类：

1. **数值精度检查**：Stage 3（522 层 INT8）在 `Decode-1 / Postfix-32 / Prefill-488` 上平均 `CosSim ≈ 0.999926`，ODE chain simulation 做完 10 次 INT8 GEMM 后仍有 `CosSim ≈ 0.99895`。
2. **端到端延迟检查**：C++ 引擎已经能跑完整的 `generate_text()` decode loop，`max_new_tokens=20` 下 Stage 3 VQA 实测 `1012.6ms / 19.75 tok/s`，并且能拆出 `ViT / Prefill / Decode` 三段时间。
3. **真实图片 VQA case 补充验证**：后续在 Orin 上我们也补过一组真实图片的 `C++ bf16 / C++ INT8` 对比。那组数据对应的是修完 `vision rotary` 之后的 `wallx_infer`，不和本文 Stage 1/2/3 的 staged benchmark 混在同一张表里，但结论方向是一致的：在 `fruits_on_table` 单图、`max_new_tokens=20`、`benchmark=5` 的条件下，`C++ INT8` 相对 `C++ bf16` 的真实图端到端加速约 **1.29x**。这组结果更适合作为“真实图推理链路的补充 case”，而不是本文主 benchmark 表的替代；它的作用是说明第四篇的主结果虽然主要建立在固定 benchmark case 上，但并不是完全脱离真实图推理链路做出来的“空中楼阁”。

还没补上的，是两类更“产品化”的验证：

- 真实机器人 / 仿真任务成功率：INT8 轨迹 vs bf16 轨迹 vs ground truth
- VQA 生成质量与更细粒度指标：BLEU、token accuracy、TTFT、prefill tokens/s、device-side sampling

从现有结果看，**VQA 的 INT8 收益确实比 Flow Action 更明显，但它并没有主要落在 ViT 上，而是落在 Prefill + Decode 一起减负。** Stage 3 的 `Decode 927.8ms → 682.7ms` 已经说明：当 MoE 和 Vision 两边都覆盖到位后，VQA 才真正进入“量化有效区间”。

### 10.1 GEMM 微基准测试

单层 GEMM 对比（K=N=2048，Orin 上测试）：

| M（batch tokens） | bf16 (μs) | CUTLASS INT8 (μs) | 加速比 |
|---|-----------|-------------------|---------|
| 32 | 55.7 | 37.7 | **1.47×** |
| 488 | 55.6 | 36.3 | **1.53×** |
| 1024 | 56.2 | 39.3 | **1.43×** |
| 4096 | 55.6 | 37.9 | **1.47×** |

**CUTLASS INT8 在所有矩阵尺寸下都比 bf16 快 43-53%。**

作为对比，朴素方案（torch::_int_mm）在相同尺寸下反而慢 50-60%。三条路线的 GEMM kernel 耗时：

| 方案 | M=32 (μs) | M=488 (μs) | 说明 |
|------|-----------|------------|------|
| bf16 baseline | 55.7 | 55.6 | cuBLAS bf16 |
| Route 1: _int_mm | ~90 | ~85 | INT32 输出 + dequant 分离 |
| Route 2: cublasLt | ~75 | ~70 | fused quant + cublasLt INT8 |
| **Route 3: CUTLASS** | **37.7** | **36.3** | **fused quant + CUTLASS EVT** |

### 10.2 端到端推理基准

真正有价值的 benchmark，不是“某一个 INT8 kernel 快多少”，而是**量化覆盖率扩到哪一步时，端到端收益才开始释放**。

先看 Flow Action（3 次取平均，Orin 上测试）：

| 模型配置 | 量化层数 | 总时间 (ms) | ViT (ms) | Prefill (ms) | ODE (ms) |
|---------|---------|-----------|---------|-------------|---------|
| bf16 baseline | 0 | 568.1 | 221.6 | 201.1 | 142.7 |
| Stage 1：Attention + VisionAttn + Merger | 210 | 573.3 | 225.1 | 201.6 | 143.8 |
| Stage 2：+ Vision MLP padding | 306 | 563.4 | 215.3 | 203.0 | 142.4 |
| **Stage 3：+ MoE Expert INT8** | **522** | **480.0** | **217.0** | **120.2** | **139.9** |
| 朴素 INT8（Route 1） | — | 871.0 | — | — | — |

再看 VQA（`max_new_tokens=20`）：

| 模型配置 | 量化层数 | 总时间 (ms) | ViT (ms) | Prefill (ms) | Decode (ms) | tok/s |
|---------|---------|-----------|---------|-------------|------------|------|
| bf16 baseline | 0 | 1316.4 | 222.2 | 165.1 | 927.8 | 15.19 |
| Stage 1：Attention + VisionAttn + Merger | 210 | 1312.1 | 223.1 | 165.3 | 922.4 | 15.24 |
| Stage 2：+ Vision MLP padding | 306 | 1298.7 | 213.4 | 163.6 | 920.7 | 15.40 |
| **Stage 3：+ MoE Expert INT8** | **522** | **1012.6** | **214.4** | **114.5** | **682.7** | **19.75** |

这组数据把第四篇真正的主线讲清楚了：

1. **第一阶段只量化 210 层时，端到端几乎不动。** 这就是本文最开始撞上的 Amdahl 墙。
2. **第二阶段补齐 96 个 `vision_mlp` 漏网层后，ViT 开始下降，但收益仍然有限。**
3. **第三阶段把 216 个 MoE expert projection 也拉进 INT8 后，收益才真正释放。** Flow Action 直接进入 `2.08 Hz`，VQA 也压到 `~1.0s`。

### 10.3 为什么第一阶段会撞上 Amdahl 墙

下面这组 nsys profiling，**对应的是第一阶段 210 层量化模型**，不是最终 522 层版本。它的价值在于解释：为什么“kernel 已经快了 1.5×”，端到端却一开始几乎不动。

nsys profiling 揭示了第一阶段的瓶颈分布（INT8 CUTLASS partial model）：

| 算子类别 | GPU 时间占比 | 绝对时间 (ms) | 备注 |
|---------|-----------|-------------|------|
| **MoE Grouped GEMM** | **28.0%** | **409** | `dual_asym_grouped_gemm`，bf16，未量化 |
| bf16 GEMM（杂项） | ~21% | ~307 | MoE 单 expert 回退 + Action Head |
| **逐元素操作** | **~20%** | **~293** | 数千个小 kernel（add、mul、cast） |
| CUTLASS INT8 GEMM | 6.5% | 95 | ← 我们优化的部分，已经很快了 |
| Softmax | 3.6% | 53 | |
| RMSNorm | 3.0% | 44 | |
| fused_quantize | 2.4% | 35 | |
| Flash Attention | 1.5% | 22 | |

**关键发现（对应第一阶段 partial INT8）**：

1. **量化后的 Linear GEMM 只占 6.5%**——它已经被优化到不再是瓶颈
2. **MoE Grouped GEMM 占 28%**——用自定义 `AsymmetricDualExpertGemm` CUTLASS kernel，始终 bf16，不走 LinearOp
3. **逐元素操作占 20%**——成千上万的小 kernel（add、mul、type cast），单个很快但 launch overhead 巨大
4. **33,552 次 cudaLaunchKernel 调用**，平均每次 21.7μs——光 CPU 端 launch 开销就吃掉了 ~728ms 的 wall clock time

### 10.4 覆盖率是怎么一点点补齐的

如果只看“有没有上 INT8”，第四篇的结论会很混乱。真正应该看的，是**每个阶段到底覆盖到了哪些层**：

| 阶段 | 新增覆盖 | 累计量化层数 | 端到端结果 | 说明 |
|------|---------|-------------|-----------|------|
| Stage 1 | Decoder Attention 144 + Vision Attention 64 + Merger 2 | 210 | Action `573ms` / VQA `1312ms` | 核心 GEMM kernel 更快，但 MoE 仍是 bf16 |
| Stage 2 | `vision_mlp` 96（通过 padding 补齐 3420→3424） | 306 | Action `563ms` / VQA `1299ms` | ViT 终于开始受益 |
| **Stage 3** | **MoE expert 216** | **522** | **Action `480ms` / VQA `1013ms`** | **量化覆盖率终于打到真正的大头** |

从这里也能看出一个比“Amdahl 定律”更具体的工程事实：

> **INT8 收益不是一个开关，而是一个覆盖率问题。**

当 MoE expert 还在 bf16 时，量化只是在热路径边缘打转；一旦把 expert projection 也拖进来，Prefill 阶段立刻出现断崖式下降：

- Flow Action Prefill：`201.1ms → 120.2ms`
- VQA Prefill：`165.1ms → 114.5ms`

而 Decode / ODE 的下降就没有这么大，这说明下一篇的主战场已经不再是“继续补量化覆盖率”，而是**launch overhead 和逐元素 kernel 融合**。

---

## 十一、实战教训和下一步优化路线

### 11.1 教训一：朴素 INT8 ≠ 自动加速

最大的教训：**不是所有 INT8 GEMM 都比 bf16 快**。PyTorch 的 `torch::_int_mm` 调用 cuBLAS INT8 GEMM，输出 INT32 到全局内存，再读回做 dequant。这个"标准"流程在 Orin 上反而更慢：

```
朴素 INT8 的隐形开销：
  1. INT32 输出写回 DRAM：4 bytes/elem vs bf16 的 2 bytes/elem → 带宽翻倍
  2. cuBLAS heuristic 对 M=32 选了 128×64 tile → 效率低
  3. splitK 策略额外引入 reduce kernel → 多一次 launch + 同步
  
  总计：GEMM 本身就慢了 1.5-1.6×，加上额外的 quant/dequant kernel → 端到端慢 1.6×
```

**解决方案是跳过 cuBLAS，直接用 CUTLASS 融合 dequant 进 epilogue。** 这不是"调参"能解决的——需要从 kernel 层面重新设计数据流。

### 11.2 教训二：Amdahl 定律比你想的更残酷

但这里必须加一个限定词：**这是对第一阶段 partial INT8 模型成立，不是对所有量化阶段都成立。**

当时量化前后真正变化的部分只占 ~25% GPU 时间（因为 MoE expert 仍然走 bf16 dual_gmm）。即使 GEMM 加速 1.5×：

```
理论加速 = 1 / (1 - 0.25 + 0.25/1.5) = 1 / (0.75 + 0.167) = 1.09×
```

**最多 9% 的端到端加速**——这就是为什么第一阶段实测几乎持平。

但一旦把 `vision_mlp` 和 `MoE expert` 也量化，Amdahl 的分母就变了：被优化的部分不再是 25%，而是热路径的大头。于是第四篇后半段的结果从“几乎不动”变成了：

- Flow Action：`568ms → 480ms`
- VQA：`1316ms → 1013ms`

所以正确结论不是“INT8 无效”，而是：

> **只量化一小块时，Amdahl 定律会把收益吃光；一旦覆盖率打到 MoE 主体，收益就会重新冒出来。**

### 11.3 教训三：nsys profiling 是唯一的真相

不做 profiling 之前，我们以为"GEMM 占 54-69%"（第二篇在 Python 推理中的结论）。换到 C++ 推理 + INT8 量化后，这个比例完全变了：

```
Python bf16 推理：GEMM 占 54-69%（Python 框架开销大，被包含在 GEMM 统计中）
C++ INT8 推理：CUTLASS INT8 GEMM 占 6.5%（Python 开销消除后，非 GEMM 算子暴露）
```

**永远在实际环境下 profile，不要用历史数据推断。** 第四篇里“先撞墙，再破壁”的过程，本质上就是 profiling 驱动的路线修正。

### 11.4 下一步优化路线图

现在基于最新的 522 层量化版本，优先级已经重排了。`MoE Expert INT8` 不再是待办项，而是新的 baseline。

| 优先级 | 优化手段 | 目标 | 预期收益 | 复杂度 |
|-------|---------|------|---------|-------|
| **P0** | CUDA Graph（ODE 循环） | 消除 3.3 万次 kernel launch 开销 | 10-20% | 中 |
| **P1** | VQA decode 去同步 / device-side sampling | 继续压缩 decode 的 682ms | 10-20% | 中高 |
| P1 | 算子融合（RMSNorm+Quant, GEMM+SiLU） | 减少逐元素 kernel 数量 | 10-15% | 中 |
| P2 | 减少 ODE 步数（5→3→2） | 减少 40-60% 的 ODE 计算 | 20-40% | 需要重训 |
| P2 | 更进一步的量化探索 | 继续压缩带宽和权重体积 | 5-15% | 中高 |
| P3 | INT8 KV Cache | 减少 KV cache 内存占用和带宽 | 3-5% | 低 |

**CUDA Graph 仍然是最高 ROI 的下一步**：ODE 循环中 M=32 的 fixed-shape 推理非常适合 graph capture；与此同时，VQA decode 的主瓶颈已经明显转向 `682ms` 的逐 token decode 路径，而不是纯 GEMM。

**如果把视野再放远一点，更后的方向其实已经很清楚：不是永远停留在“每个 `LinearOp` 自己做 quantize/GEMM/dequant”这层，而是把今天这些手工打通的量化边界，总结成显式的图级 `QDQ` pattern，让后端去判断哪些 `dequant` 可以后移、哪些 `requant` 可以省掉、哪些 `quantize + layout transform + matmul + bias + dequant` 值得整体 fuse 成一个 kernel。换句话说，第四篇解决的是“主热路径的低比特算子怎么先跑起来”，第五篇开始解决的是“这些 `QDQ island` 之间的数据流怎么进一步缩短”；再往后，才自然会逼近 `TensorRT / TVM` 那类图级自动化系统。**

**这些还没自动化接住的剩余工作，本身也正是我们和 NVIDIA 原生量化栈之间性能差距的重要来源之一。**

**把这条线继续做下去，其实就是在摸一个面向端侧 VLA 的量化方案 1.0。**

---

## 十二、落地思考

### 12.1 量化真的不是银弹

从第一篇到第四篇，优化路线走下来的实际数据：

```
第一篇：部署上去，跑通 baseline → Python bf16 ~1150ms（Flow Action）
第三篇：C++ 消除 Python 开销 → ~557ms
第四篇 Stage 1：先量化 210 层 → 573ms（几乎没收益）
第四篇 Stage 2：补齐 vision_mlp → 563ms
第四篇 Stage 3：补齐 MoE expert → 480ms
```

**真正的教训不是“INT8 无效”，而是“只量化了容易量化的那一小块时，收益会被覆盖率不足和 launch overhead 吃掉”。** 这篇文章前后半段的转折，恰恰来自这个事实。

### 12.2 从 480ms 还要往哪降？

wall-x 的动作控制频率要求：
- 基础操作（抓取/放置）：10-20 Hz → 50-100ms/step → **还差 5-10 倍**
- 高精度操作（插入/对齐）：30-50 Hz → 20-33ms/step → **差 17-28 倍**

这个差距靠单纯“继续补量化覆盖率”已经填不满了。下一阶段真正的路线是：
1. **CUDA Graph + 算子融合**：10-25% → `~430-450ms`
2. **减少 ODE 步数**（5→3→2）：30-50% → `~250-320ms`
3. **模型蒸馏**（3B → 1B）：再降 50% → `~120-160ms`
4. **异步流水线**（ViT / Transformer / ODE 重叠）：进一步压有效控制延迟

**具身智能的瓶颈不在 model，在 runtime。** 单次推理的"绝对延迟"能压到 100ms 已经很好了，但 10Hz 控制频率需要的是**流水线吞吐量**——多个推理请求重叠执行，摊平单步延迟。

#### 近期部署目标

上面是理论极限。工程上，我们先定两个**可验证的近期目标**：

| 任务 | 当前延迟 | 目标频率 | 目标延迟 | 当前状态 |
|------|---------|---------|---------|---------|
| **Flow Action** | **480 ms (2.08 Hz)** | **2-3 Hz** | **333-500 ms** | **已进入目标区间** |
| **VQA** | **~1013 ms (20tok)** | **~1 Hz** | **< 1.2 s** | **已基本达标** |

为什么 VQA 目标只要 ~1 Hz？因为 **VQA 不在机器人的控制回路中**。Flow Action 才是驱动机械臂的关键路径。VQA 的角色是训练基座（提供视觉理解能力）、量化精度评估的标尺、以及调试时的感知验证工具——秒级响应足够。VQA 限制 max_new_tokens=20（机器人场景下回答通常 10-15 tokens）后，当前 C++ 引擎已接近 1 Hz，**Flow Action 才是优化主战场**。

基于这两个目标，优化优先级也重新排序：

| 优化手段 | Flow Action 收益 | VQA 收益 | 优先级 |
|---------|-----------------|---------|--------|
| **CUDA Graph** | 高（ODE shape 固定） | 中（decode 每步需 GPU sync） | **最高** |
| **VQA decode 去同步** | 低 | **高**（当前 decode 682ms） | 高 |
| **Triton fused kernel / 算子融合** | 中（逐元素 kernel 还很多） | 中 | 高 |
| **更激进量化** | 中 | 中 | 中 |

### 12.3 本篇最重要的三句话

1. **"标准"INT8 GEMM 可能比 bf16 更慢**——不要相信理论吞吐量，在目标硬件上实测。cuBLAS 的 heuristic 不是万能的，CUTLASS 自定义 kernel 才是正道。

2. **Amdahl 定律是铁律，但它对应的是“当前覆盖率下的分母”**——优化 25% 的代码，即使快 2 倍，端到端也只快 14%；把覆盖率打到 MoE 主体，分母变了，收益也会重新出现。

3. **量化的真正价值是“带宽 + 内存 + 覆盖率”三件事一起成立。** 只谈 kernel 理论吞吐没有意义，真正决定端到端的是：你量到了哪一层、是不是还被 launch overhead 卡住、以及剩下的 bf16 大头是谁。

---

## 系列预告

本文是 **wall-x 机器人大模型部署系列** 的第四篇。

**第一篇**：把 3B 大模型塞进机器人：RTX 5090 与 Jetson Orin 边缘端产品部署踩坑全记录
- 环境搭建、CUDA 依赖链、profiling 数据、28 万+ kernel 分析

**第二篇**：当算子逼近硬件极限：一次 Orin Profiling 引发的具身智能实时系统思考
- FA2 编译全过程、FA2 vs SDPA benchmark、GEMM 带宽天花板证明、67% 框架空转发现

**第三篇**：用 C++ 替换 Python 推理：在 Orin 上把 wall-x 跑到当前精度的极限
- VQA 热路径分析、CUDA vs TensorRT 方案选型、libtorch + CUDA Graph 实施计划

**第四篇（本文）**：在 Orin 上给 wall-x 这个机器人 VLA 做 INT8：为什么理论 2× 加速一开始几乎没有收益
- 三条 INT8 GEMM 路线对比：朴素（慢 1.6×）→ cublasLt（仍慢）→ CUTLASS EVT（快 1.5×）
- `vision_mlp` padding 补齐 96 个漏网层：覆盖率 `210 → 306`
- `MoE expert INT8` 打通：覆盖率 `306 → 522`
- 端到端结果：Flow Action `568ms → 480ms`，VQA `1316ms → 1013ms`
- 新结论：Amdahl 墙不是“INT8 无效”，而是“覆盖率不够”

**第五篇（预告）**：量化之后还剩什么：在 Orin 上给 wall-x 做 CUDA Graph、算子融合和 CUTLASS
- ODE / postfix fixed-shape 推理的 graph capture
- `residual + rmsnorm`、`quantize + layout transform`、`GEMM + SiLU` 这类高频短链融合
- 基于 CUTLASS 把 GEMM 前后的数据流继续往主算子里收
- 从手工 fusion 走到 compiler pass，最后自然逼近 runtime / AI OS

---

*测试环境：wall-oss-flow 3B 模型。量化：Per-channel 权重 INT8 + Per-token 动态激活 INT8（无 SmoothQuant），Python 离线量化导出；阶段 2 额外补齐 `vision_mlp` padding，阶段 3 额外量化 216 个 MoE expert projection。推理：C++ libtorch + CUTLASS 4.0 融合 INT8 GEMM（EVT dequant epilogue），MoE 在 INT8 checkpoint 下切到 per-expert `LinearOp` 路径。测试平台：Jetson AGX Orin 64GB，JetPack 6.2.1，CUDA 12.6，PyTorch 2.5.0a0+872d972e41.nv24.08。Profiling 工具：nsys 2024.7.1 + ncu。2026 年 4 月。*
