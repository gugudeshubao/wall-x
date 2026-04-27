# INT8 量化实战：从理论 2× 加速到 Amdahl 定律的铁壁

> 前三篇我们完成了：环境部署（第一篇）、Flash Attention 深挖（第二篇）、C++ 推理框架（第三篇，557ms）。这一篇终于要动刀了：**把 bf16 GEMM 变成 INT8 GEMM，理论上直接砍一半计算量。** 但实战的结果出乎意料——朴素 INT8 反而慢了 1.6 倍，用 CUTLASS 融合方案修正后 GEMM 快了 1.5 倍，端到端却只从 557ms 降到 554ms。这篇完整记录三条技术路线的探索、CUTLASS EVT 的实现细节、nsys profiling 的深度分析，以及 Amdahl 定律给出的残酷答案。

**TL;DR**
- wall-x 3B 模型有 **~723 个 Linear 层**（36 层 decoder × 7 + 32 层 ViT × 5 + 杂项），GEMM 是绝对主体
- 量化方案：**Per-channel 权重 INT8 + Per-token 动态激活 INT8**，Python 离线量化导出，C++ 在线推理
- **朴素 INT8 方案反而慢了 1.6 倍**（871ms vs 554ms bf16）——根因：INT32 输出带宽翻倍 + cuBLAS tile 选择不佳
- **CUTLASS 融合方案彻底解决**：用 EVT（Epilogue Visitor Tree）将 dequant 融合进 GEMM epilogue，INT32 不落地
- GEMM 微基准测试：CUTLASS INT8 比 bf16 快 **1.43-1.53 倍**（全矩阵尺寸）
- 端到端结果：INT8 CUTLASS = 554ms，bf16 = 557ms——**持平，没有显著加速**
- **Amdahl 定律作祟**：量化后 Linear GEMM 仅占 GPU 时间 6.5%，非 GEMM 算子（MoE 28%、逐元素 20%）成为新瓶颈
- Orin SM 8.7 INT8 Tensor Core 的理论吞吐量是 bf16 的 **2 倍**（MMA 指令 m16n8k32 vs m16n8k16）
- **下一步优化**：CUDA Graph 消除 3.3 万次 kernel launch 开销、算子融合、MoE INT8 量化

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
| gate_proj | 2048 → 11008 | 22.5M | **大** |
| up_proj | 2048 → 11008 | 22.5M | **大** |
| down_proj | 11008 → 2048 | 22.5M | **大** |

**36 层 Decoder 合计**：36 × 7 = **252 个 Linear 层**，其中 FFN 的 gate/up/down 三个大矩阵占参数量主体。

**32 层 ViT 合计**：32 × 5 = **160 个 Linear 层** + PatchMerger 2 个 = 162 个

**全模型 Linear 层总计**：~560（Decoder）+ 162（ViT）+ 若干杂项 ≈ **~723 个 Linear 层**

这就是量化的目标：把这 723 个 Linear 层的 bf16 GEMM 尽可能多地变成 INT8 GEMM。

---

## 二、核心问题：PyTorch、TensorRT、ONNX，选哪个？

这是量化方案的第一个分叉点。三条路线的本质区别是：

### 2.1 PyTorch 原生量化（torchao / torch.ao.quantization）

```
工作方式：
  1. 遍历模型的 nn.Linear 层
  2. 插入 Observer，跑校准数据，收集激活值范围
  3. 替换 nn.Linear 为 QuantizedLinear
  4. 推理时用 INT8 GEMM（torch._int_mm 或 cuBLAS INT8）

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
| **PyTorch (torchao)** | **不需要** | **不受影响** | 良好 | **低** |
| TensorRT | 需要 | 要写 6 个 plugin | 最好 | 高 |
| ONNX Runtime | 需要 | 要注册 custom op | 一般 | 中高 |
| torch_tensorrt 混合 | 部分需要 | 标准层走 TRT | 好 | 中 |

**推荐路线：PyTorch 原生量化（torchao）。**

不需要展开为静态图，不需要碰自定义算子，逐模块替换 Linear 层即可。如果后续发现 PyTorch INT8 kernel 性能不够，再用 torch_tensorrt 把标准子图编译到 TensorRT——但这是可选的 Phase 2，不是必须。

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

## 四、量化方法：SmoothQuant + W8A8 动态量化

### 4.1 为什么选 W8A8

Orin SM 8.7 的硬件约束直接决定了方法：

| 量化方式 | Orin 硬件支持 | 计算加速 | 内存节省 |
|----------|-------------|----------|----------|
| **W8A8** (INT8 权重 + INT8 激活) | INT8 Tensor Core | **2x** | 2x |
| W4A16 (INT4 权重 + FP16 激活) | 无 INT4 TC，dequant 到 FP16 | **无** | 2x（仅内存） |
| W8A16 (INT8 权重 + FP16 激活) | dequant 到 FP16 | **无** | 接近 2x |
| W4A8 (INT4 权重 + INT8 激活) | 无 INT4 TC | 无 | 2x |

W4A16（GPTQ/AWQ 的典型方案）在桌面 GPU 上很流行，但 **在 Orin 上只省内存不加速**——因为 SM 8.7 没有 INT4 Tensor Core，运行时还是要 dequant 到 FP16 再做 GEMM。

**W8A8 是 Orin 上唯一既省内存又加速计算的方案。** INT8 Tensor Core 吞吐量是 bf16 Tensor Core 的 2 倍。

### 4.2 SmoothQuant：解决激活值 outlier 问题

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

### 4.3 动态量化 vs 静态量化

| 维度 | 动态量化 | 静态量化 |
|------|----------|----------|
| 权重 | 提前量化为 INT8（静态 scale） | 同左 |
| 激活 | 推理时逐 token/逐 batch 计算 scale | 校准时确定固定 scale |
| 额外开销 | 每层多一次 `absmax()` 计算 scale | 无 |
| 精度 | **更好**（适应不同输入的激活范围） | 较差（固定 scale 可能截断） |
| 实现复杂度 | 中 | 高（需要精确校准） |

**推荐动态量化**：权重静态 INT8（提前量化），激活逐 token 动态 INT8（推理时计算 scale）。多出的 `absmax()` 开销很小（几十微秒），但精度提升明显。

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

Orin 的 SM 8.7 基于 Ampere 架构。以下是 CUTLASS `mma_sm80.h` 中定义的所有 MMA 指令形状：

| 数据类型 | MMA 形状 | 每条指令运算量 | 相对吞吐量 |
|---------|----------|-------------|-----------|
| BF16 | m16n8k8 | 2,048 ops | 0.5× |
| **BF16** | **m16n8k16** | **4,096 ops** | **1×（基线）** |
| FP16 | m16n8k16 | 4,096 ops | 1× |
| **INT8** | m16n8k16 | 4,096 ops | 1× |
| **INT8** | **m16n8k32** | **8,192 ops** | **2×** |
| INT4 | m16n8k64 | 16,384 ops | 4× |
| TF32 | m16n8k4 | 1,024 ops | 0.25× |
| TF32 | m16n8k8 | 2,048 ops | 0.5× |

**关键结论**：
- INT8 的 `m16n8k32` 指令每次处理 32 个 K 维元素（bf16 只处理 16 个），**同一条 MMA 指令的运算量翻倍**
- 这不是"多发几条指令"的假加速，而是**单条指令真正多算一倍**
- INT4 更极端（m16n8k64，4× 吞吐），但 INT4 Tensor Core 在 Orin 上没有对应的 cuBLAS API，需要纯 CUTLASS 实现
- FP8 不可用——FP8 Tensor Core 需要 SM 8.9+（Ada Lovelace / Hopper），Orin SM 8.7 没有

**MMA 指令不是瓶颈**。INT8 在硬件层面确实有 2× 吞吐优势。真正的问题出在软件层面（朴素实现的 INT32 带宽、tile 选择），以及 Amdahl 定律（GEMM 只占总时间 25%）。

---

## 五、分层量化策略：不是所有层都该量化

### 5.1 量化优先级

| 组件 | 量化建议 | 原因 |
|------|----------|------|
| **Decoder FFN** (gate/up/down) | **INT8** | 最大 GEMM（2048×11008），收益最大 |
| **Decoder Attention** (QKV, O) | **INT8** | 中等 GEMM，层数多 |
| **Vision Encoder** | **INT8** | 32 层 ViT，中等收益 |
| **MoE 语言 Expert** | **INT8**（谨慎） | intermediate=11008，大矩阵，但路由逻辑可能放大误差 |
| **MoE 动作 Expert** | **bf16 保持** | intermediate=2048，小矩阵且影响动作预测精度 |
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

**安全策略**：Decoder 的 Transformer 层可以量化（误差只影响 v_t），但 Action Head 的 w1/w2/w3 保持 bf16（直接产生 action embedding，误差被 ODE 积分放大）。

### 5.3 MoE 自定义 GEMM 的处理

wall-x 的 MoE 用 `asym_dual_gmm`（自定义 CUDA kernel）做双 expert GEMM。如果要量化 MoE expert 的权重，有两条路：

**方案 A：修改 asym_dual_gmm 支持 INT8**
- 改 CUDA kernel，输入从 bf16 改为 INT8
- 工作量大（改 kernel + 测试）
- 性能最好

**方案 B：MoE 层回退标准 PyTorch**
- 不用 `asym_dual_gmm`，用标准的 quantized nn.Linear
- 先 permute → 分别跑两个 expert 的 quantized Linear → unpermute
- 损失一些 dual-expert 并行效率，但量化免费
- 用于验证量化精度，后续再决定是否改 kernel

**推荐先做方案 B**——先验证精度，再优化性能。

---

## 六、量化变换的算子图和权重陷阱

量化不是"替换 Linear 层就完事"。wall-x 的自定义算子和模型结构会引入几个容易忽略的问题。

### 6.1 SmoothQuant 永久修改权重

SmoothQuant 变换 `W_smooth = W * s` 会永久改变权重值。后果：
- 原始 bf16 checkpoint 和量化后的权重 **对不上**
- 不能直接用原始 checkpoint 加载量化模型
- 模型重新训练后，SmoothQuant 的 scale `s` 要重新计算

**解决方案**：量化后另存一份 checkpoint（INT8 权重 + per-channel scales + smooth scales），原始 checkpoint 不动。

### 6.2 `asym_dual_gmm` 不支持 INT8——最大的坑

wall-x 的 MoE 层用自定义 CUDA kernel `asym_dual_gmm` 做双 expert GEMM。这个 kernel **只处理 bf16 张量**。如果你把 expert 权重量化为 INT8，kernel 直接报错或产生垃圾输出。

这就是上面"MoE 层先不量化"的根本原因。要量化 MoE，必须先改 kernel 或拆开用标准 Linear。

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
| `asym_dual_gmm` 不支持 INT8 | **高** | MoE 层先不量化 |
| SmoothQuant 改了权重 | 中 | 另存量化 checkpoint |
| RoPE 收到错误 dtype | 中 | 加 assert 检查 |
| state_dict key 变化 | 中 | bf16 加载 → 量化 → 另存 |
| MoE permute/unpermute dtype | **低** | dequant 在 Linear 内部完成，外部 bf16 |

---

## 七、C++ 推理框架和量化的实际集成

第三篇的 C++ 替换和第四篇的 INT8 量化要叠加使用。核心挑战是：**torchao 的 Python 量化模块在 C++ libtorch 里不存在，且朴素 INT8 GEMM 在 Orin 上反而更慢。**

### 7.1 架构：Python 离线量化 + C++ 在线推理

```
阶段 1：Python 离线量化（在 5090 上，跑一次）
  ├── 加载 bf16 模型
  ├── 逐层计算 per-channel 权重 absmax
  ├── 权重量化 → round(W / scale * 127) → INT8
  └── 保存：
      model.layers.*.self_attn.{q,k,v,o}_proj.{weight_int8, weight_scale}
      model.layers.*.mlp.{gate,up,down}_proj.{weight_int8, weight_scale}
      model.visual.blocks.*.{attn, mlp}.*.{weight_int8, weight_scale}

阶段 2：C++ 在线推理（在 Orin 上运行）
  ├── 加载 INT8 权重 + weight_scale（普通 tensor）
  ├── 对每个 Linear 层调用 2-kernel 流水线：
  │     kernel 1: fused_quantize_activation() → act_int8 + act_scale
  │     kernel 2: cutlass_int8::gemm_dequant() → bf16 输出
  └── 自定义算子（RoPE、MoE permute 等）照常 bf16
```

### 7.2 三条路线的探索历程

我们实际尝试了三种 INT8 GEMM 实现方式：

**路线 1：朴素方案（torch::_int_mm + 手动 dequant）**

```cpp
// 4 个 kernel：quantize_act + absmax + _int_mm + dequant
auto x_int8 = quantize(input);
auto out_int32 = torch::_int_mm(x_int8, weight_int8.t());  // cuBLAS INT8
auto out_bf16 = dequant(out_int32, act_scale, weight_scale);
// ❌ 结果：871ms（bf16 = 554ms），慢 1.6 倍
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

## 八、校准数据准备

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
    # MoE 动作 expert 不量化
    if "expert.1" in name:  # expert 1 = action expert
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

## 十、实战结果：从 1.6x 更慢到持平

### 10.0 测试用例说明

> **系列文章测试用例对比**——四篇文章使用了不同的测试场景和框架，数据不能直接横向对比：
>
> | | 第一篇（部署篇） | 第二篇（FA2 篇） | **第四篇（本文 INT8）** |
> |---|--------|--------|--------|
> | **测试脚本** | `test_vqa_bench.py` / `fake_inference.py` | `bench_fa2_vs_sdpa.py` | C++ `wallx_infer` |
> | **输入数据** | 8 张真实图片 640×480 / 50 token 随机 tensor | 1 张真实图片 640×480 | Dummy 随机 tensor |
> | **任务类型** | VQA 文本生成（128 tokens）/ 单次 forward | VQA 文本生成（64 tokens） | **Flow Action ODE 积分** |
> | **推理方式** | Python `model.generate()` / `model()` | Python `model.generate()` | **C++ libtorch forward** |
> | **序列长度** | 420 tokens（VQA）/ 50 tokens（fake） | ~420 tokens | **488 tokens** |
> | **框架开销** | ~67% Python/HF 开销 | ~67% Python/HF 开销 | **无 Python 开销** |
> | **Orin 延迟** | 13,898ms（VQA）/ 115ms（fake forward） | 6,360ms（FA2 VQA） | **557ms（Flow Action）** |
>
> ⚠️ 第一篇的 `fake_inference.py`（14.2ms / 115ms）是 50 token 随机 tensor 的单次 Transformer forward，**没有 ViT 编码、没有 ODE 积分、没有文本生成**，不代表任何真实任务的延迟。

端到端 benchmark 使用的是 **Dummy 输入（随机数据）**，不是真实图片和文本。具体构成：

```
输入构成（模拟典型 VQA + Flow Action 场景）：
  input_ids:       488 tokens = 200 文本(randint) + 256 图像(image_token_id) + 32 动作(action_token_id)
  pixel_values:    torch::randn, 1024 patches × (3×2×14×14), 模拟 1 张 224×224 图
  image_grid_thw:  [1, 32, 32] → 32×32 patches, merge 后 256 tokens
  moe_token_types: 前 456=0(文本/视觉), 后 32=1(动作)
  ODE timesteps:   5 步 Euler 积分
```

对 **延迟 benchmark** 来说这是有效的——GEMM 耗时只取决于矩阵尺寸 (M×K×N)，不取决于具体数值。但 **不能用于精度验证**。

> **TODO：精度验证测试**
> - [ ] 准备真实测试集：从 LeRobot 训练集抽取 10-50 条真实样本（图片 + 文本指令 + ground truth 动作轨迹）
> - [ ] 对比 bf16 vs INT8 输出：逐层输出 cosine similarity、最终动作轨迹 MSE
> - [ ] Flow Action 端到端精度：INT8 预测轨迹 vs bf16 预测轨迹 vs ground truth 的 L2 距离
> - [ ] 确认 ODE 5 步积分是否放大了量化误差（对比单步 vs 多步的误差累积曲线）
>
> **TODO：VQA 文本生成基准测试**
> - [x] C++ 引擎添加 `generate_text()` 方法（autoregressive decode loop + lm_head + argmax）——**已完成，实测 3469ms / 18.45 tok/s（64 tokens），比 Python FA2 快 1.83x**
> - [x] 当前只有 `generate_flow_action()`，lm_head_weight_ 已加载但未使用——**已启用，支持 `--mode vqa`**
> - [ ] VQA decode 是 M=1 逐 token 生成（标准 LLM 模式），INT8 对 memory-bound decode 可能有更大收益
> - [ ] 测试指标：prefill tokens/s、decode tokens/s、首 token 延迟（TTFT）
> - [ ] 对比 bf16 vs INT8 的 VQA 生成质量（BLEU / token accuracy）
>
> **INT8 VQA 收益预估**：C++ VQA decode 每步 49.0ms（GPU kernel ~29.7ms + GPU→CPU 同步 ~19ms）。Decode 阶段的 GEMV 是纯 bandwidth-bound（第二篇已证明跑在 Orin 带宽极限的 72%），INT8 权重读取量减半，理论上每步 GEMV 能省 ~10ms → 63 步累计省 ~630ms → 从 3469ms 降到 ~2840ms（~22.5 tok/s）。这比 Flow Action 的 INT8 收益大得多（Flow Action 的 Linear GEMM 仅占 GPU 时间 6.5%，而 VQA decode 的 GEMV 占 GPU kernel 的 ~60%）。VQA 是 INT8 量化真正能发力的场景。

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

完整推理管线（3 次取平均，Orin 上测试）：

| 模型配置 | 总时间 (ms) | ViT (ms) | Prefill (ms) | ODE (ms) |
|---------|-----------|---------|-------------|---------|
| bf16 baseline | 556.7 | 221.1 | 198.2 | 134.9 |
| **INT8 CUTLASS** | **553.9** | **220.9** | **198.1** | **132.2** |
| INT8 朴素（Route 1） | 871.0 | — | — | — |

**结论：CUTLASS INT8 端到端与 bf16 持平（554ms vs 557ms），比朴素 INT8 快 1.6 倍。但没有实现预期的"砍一半"加速。**

### 10.3 nsys 深度分析：Amdahl 定律的铁律

nsys profiling 揭示了真正的瓶颈分布（INT8 CUTLASS 模型）：

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

**关键发现**：

1. **量化后的 Linear GEMM 只占 6.5%**——它已经被优化到不再是瓶颈
2. **MoE Grouped GEMM 占 28%**——用自定义 `AsymmetricDualExpertGemm` CUTLASS kernel，始终 bf16，不走 LinearOp
3. **逐元素操作占 20%**——成千上万的小 kernel（add、mul、type cast），单个很快但 launch overhead 巨大
4. **33,552 次 cudaLaunchKernel 调用**，平均每次 21.7μs——光 CPU 端 launch 开销就吃掉了 ~728ms 的 wall clock time

### 10.4 量化层覆盖情况

| 模型组件 | 层数 | 实际状态 | 原因 |
|---------|------|---------|------|
| Decoder Attention (Q/K/V/O) | 36 × 4 = 144 | ✅ INT8 | 通过 LinearOp |
| ViT Attention + MLP | 32 × 5 = 160 | ✅ INT8 | 通过 LinearOp |
| MoE Expert Projections | 36 × 6 = 216 | ❌ bf16 | 用 `torch::linear`，不走 LinearOp |
| Action Head (w1/w2/w3) | 3 | ❌ bf16 | 设计决策：ODE 误差放大 |
| Embedding / LM Head | 2 | ❌ bf16 | 量化收益小 |

**约 304 个 Linear 层已 INT8 量化，但 MoE 的 216 个 expert projection 仍是 bf16——这正是 nsys 中 28% MoE GEMM 的来源。**

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

量化前 GEMM 占 ~25% GPU 时间（因为 MoE 用的是独立的 CUTLASS kernel）。即使 GEMM 加速 1.5×：

```
理论加速 = 1 / (1 - 0.25 + 0.25/1.5) = 1 / (0.75 + 0.167) = 1.09×
```

**最多 9% 的端到端加速**——实测 ~0.5%（554ms vs 557ms）。剩下的差距来自 kernel launch overhead 的微小波动。

### 11.3 教训三：nsys profiling 是唯一的真相

不做 profiling 之前，我们以为"GEMM 占 54-69%"（第二篇在 Python 推理中的结论）。换到 C++ 推理 + INT8 量化后，这个比例完全变了：

```
Python bf16 推理：GEMM 占 54-69%（Python 框架开销大，被包含在 GEMM 统计中）
C++ INT8 推理：CUTLASS INT8 GEMM 占 6.5%（Python 开销消除后，非 GEMM 算子暴露）
```

**永远在实际环境下 profile，不要用历史数据推断。**

### 11.4 下一步优化路线图

基于 nsys 分析，按 ROI 排序：

| 优先级 | 优化手段 | 目标 | 预期收益 | 复杂度 |
|-------|---------|------|---------|-------|
| **P0** | CUDA Graph（ODE 循环） | 消除 3.3 万次 kernel launch 开销 | 10-20% | 中 |
| **P1** | MoE Expert INT8 量化 | 砍掉 28% 的 bf16 MoE GEMM | 15-25% | 高 |
| P1 | 减少 ODE 步数（5→3→2） | 减少 40-60% 的 ODE 计算 | 20-40% | 需要重训 |
| P2 | 算子融合（RMSNorm+Quant, GEMM+SiLU） | 减少逐元素 kernel 数量 | 10-15% | 中 |
| P2 | W4A16 量化 | 权重体积再砍一半（memory-bound 场景更快） | 5-10% | 中 |
| P3 | INT8 KV Cache | 减少 KV cache 内存占用和带宽 | 3-5% | 低 |

**CUDA Graph 是最高 ROI 的下一步**：ODE 循环中 M=32 的 fixed-shape 推理非常适合 graph capture，可以把 33,552 次 kernel launch 合并为 5 次（每个 ODE step 一次 graph replay）。

---

## 十二、落地思考

### 12.1 量化真的不是银弹

从第一篇到第四篇，优化路线走下来的实际数据：

```
第一篇：部署上去，跑通 baseline → Python bf16 ~1150ms（VQA pipeline）
第二篇：试 FA2 → 不行（短序列 + Orin heuristic 不匹配）
第三篇：C++ 消除 Python 开销 → 557ms（~2× 加速）
第四篇：INT8 量化 → 554ms（朴素方案 871ms，CUTLASS 修正后持平）
```

**INT8 量化在 kernel 层面确实快了 1.5×，但端到端几乎没有差距。** 原因很明确：

1. C++ 推理已经消除了 Python 开销，暴露了真正的 GPU 瓶颈
2. 量化只覆盖了 ~304 个 Linear 层（占 GPU 时间 ~25%），MoE 的 216 个 expert projection 仍是 bf16
3. 33,552 次 kernel launch 的 CPU 端开销占了大量 wall clock time
4. **优化已经进入"长尾"阶段——每一步的边际收益递减，但工程复杂度递增**

### 12.2 从 557ms 要降到多少才够？

wall-x 的动作控制频率要求：
- 基础操作（抓取/放置）：10-20 Hz → 50-100ms/step → **还差 5-10 倍**
- 高精度操作（插入/对齐）：30-50 Hz → 20-33ms/step → **差 17-28 倍**

这个差距靠单纯的推理优化已经填不满。真正的路线是：
1. **CUDA Graph + 算子融合**：10-30% → ~400ms
2. **MoE INT8 + 减少 ODE 步数**：30-50% → ~200-280ms
3. **模型蒸馏**（3B → 1B）：再降 50% → ~100-140ms
4. **异步流水线**（ViT / Transformer / ODE 重叠）：有效延迟再降

**具身智能的瓶颈不在 model，在 runtime。** 单次推理的"绝对延迟"能压到 100ms 已经很好了，但 10Hz 控制频率需要的是**流水线吞吐量**——多个推理请求重叠执行，摊平单步延迟。

#### 近期部署目标

上面是理论极限。工程上，我们先定两个**可验证的近期目标**：

| 任务 | 当前延迟 | 目标频率 | 目标延迟 | 约束 |
|------|---------|---------|---------|------|
| **Flow Action** | 554 ms (1.81 Hz) | **2-3 Hz** | **333-500 ms** | 控制回路硬约束 |
| **VQA** | ~1313 ms (20tok) | **~1 Hz** | **< 1.2 s** | max_new_tokens ≤ 20 |

为什么 VQA 目标只要 ~1 Hz？因为 **VQA 不在机器人的控制回路中**。Flow Action 才是驱动机械臂的关键路径。VQA 的角色是训练基座（提供视觉理解能力）、量化精度评估的标尺、以及调试时的感知验证工具——秒级响应足够。VQA 限制 max_new_tokens=20（机器人场景下回答通常 10-15 tokens）后，当前 C++ 引擎已接近 1 Hz，**Flow Action 才是优化主战场**。

基于这两个目标，INT8 的优先级重新排序：

| 优化手段 | Flow Action 收益 | VQA 收益 | 优先级 |
|---------|-----------------|---------|--------|
| **CUDA Graph** | 高（ODE shape 固定） | 中（decode 每步需 GPU sync） | **最高** |
| **INT8 VQA decode** | 低（GEMM 仅占 6.5%） | **高**（GEMV bandwidth-bound，每步省 ~10ms） | 高 |
| **MoE INT8** | 中（MoE 占 GPU 28%） | 中 | 中 |
| **Triton fused kernel** | 中（fused_add_rmsnorm 快 3.9x） | 中 | 中 |

### 12.3 本篇最重要的三句话

1. **"标准"INT8 GEMM 可能比 bf16 更慢**——不要相信理论吞吐量，在目标硬件上实测。cuBLAS 的 heuristic 不是万能的，CUTLASS 自定义 kernel 才是正道。

2. **Amdahl 定律是铁律**——优化 25% 的代码，即使快 2 倍，端到端也只快 14%。nsys profiling 是做任何优化之前的"第零步"。

3. **量化的真正价值是内存**——INT8 权重体积是 bf16 的一半。在 Orin 64GB 这种内存受限的边缘设备上，省下的内存可以用来跑更大的 batch、更长的 KV cache、或者并行部署更多模型。计算加速只是锦上添花。

---

## 系列预告

本文是 **wall-x 机器人大模型部署系列** 的第四篇。

**第一篇**：[把 3B 大模型塞进机器人：RTX 5090 与 Jetson Orin 边缘端产品部署踩坑全记录](#)
- 环境搭建、CUDA 依赖链、profiling 数据、28 万+ kernel 分析

**第二篇**：[当算子逼近硬件极限：一次 Orin Profiling 引发的具身智能实时系统思考](#)
- FA2 编译全过程、FA2 vs SDPA benchmark、GEMM 带宽天花板证明、67% 框架空转发现

**第三篇**：[用 C++ 替换 Python 推理：在 Orin 上把 wall-x 跑到当前精度的极限](#)
- VQA 热路径分析、CUDA vs TensorRT 方案选型、libtorch + CUDA Graph 实施计划

**第四篇（本文）**：INT8 量化实战——从理论 2× 加速到 Amdahl 定律的铁壁
- 三条 INT8 GEMM 路线对比：朴素（慢 1.6×）→ cublasLt（仍慢）→ CUTLASS EVT（快 1.5×）
- CUTLASS Epilogue Visitor Tree 融合 dequant 的实现细节
- Orin SM 8.7 MMA 指令全景（INT8 m16n8k32 = 2× bf16 吞吐）
- nsys 深度分析：MoE 28%、逐元素 20%、33K kernel launch
- 端到端 554ms vs 557ms：Amdahl 定律的残酷验证

**第五篇（预告）**：CUDA Graph + 算子融合——消灭 3.3 万次 kernel launch
- CUDA Graph 捕获 ODE 循环的 fixed-shape 推理
- RMSNorm + Quantize 算子融合
- MoE Grouped GEMM 的 INT8 改造
- 从 557ms 到 ~400ms 的最后一公里

---

*测试环境：wall-oss-flow 3B 模型。量化：Per-channel 权重 INT8 + Per-token 动态激活 INT8（无 SmoothQuant），Python 离线量化导出。推理：C++ libtorch + CUTLASS 4.0 融合 INT8 GEMM（EVT dequant epilogue）。测试平台：Jetson AGX Orin 64GB，JetPack 6.2.1，CUDA 12.6，PyTorch 2.5.0a0+872d972e41.nv24.08。Profiling 工具：nsys 2024.7.1 + ncu。2026 年 4 月。*
