# INT8 量化落地：在 Orin 上把 wall-x 的 GEMM 砍一半

> 前三篇我们完成了：环境部署（第一篇）、Flash Attention 深挖（第二篇）、C++ 推理框架设计（第三篇）。到现在为止，所有优化都没动过模型精度——bf16 GEMM 仍然占 GPU 时间的 54-69%，是推理延迟的绝对主体。这一篇终于要动刀了：**把 bf16 GEMM 变成 INT8 GEMM，理论上直接砍一半计算量。**

**TL;DR**
- wall-x 3B 模型有 **~723 个 Linear 层**（36 层 decoder × 7 + 32 层 ViT × 5 + 杂项），GEMM 是绝对主体
- **不需要把整个模型展开为静态图**——逐模块替换 Linear 层即可，自定义 CUDA 算子不受影响
- 量化框架选择：**PyTorch 原生（torchao）> TensorRT > ONNX**，原因是 6 个自定义算子
- 推荐方法：**SmoothQuant + W8A8 动态量化**——权重静态 INT8，激活逐 token 动态 INT8
- Orin SM 8.7 有 INT8 Tensor Core，cuBLAS 原生支持 INT8 GEMM，理论吞吐量是 bf16 的 **2 倍**
- **Flow Action 是量化的最大风险点**：ODE 10 步积分会放大量化误差，Action Head 建议保持 bf16
- 最大的坑：`asym_dual_gmm` 自定义 CUDA kernel 不支持 INT8，MoE 层暂不量化
- **C++ 推理和量化可以叠加**：Python 离线量化导出 INT8 权重 + scale，C++ 用 `torch::_int_mm` 推理，完全解耦
- 校准数据：从 LeRobot 训练集随机抽 256 条，**现有硬件（5090 + Orin）足够做 PTQ**

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

```
bf16 GEMM（当前）：
  cuBLAS → cublasGemmEx(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF)
  使用 bf16 Tensor Core

INT8 GEMM（目标）：
  cuBLAS → cublasLtMatmul(CUDA_R_8I, CUDA_R_8I, CUDA_R_32I)
  使用 INT8 Tensor Core
  输出 INT32 → dequant → bf16（和后续层对接）

PyTorch API：
  torch._int_mm(A_int8, B_int8)  → 返回 INT32 tensor
  然后手动 dequant：output_bf16 = output_int32 * scale_a * scale_b
```

PyTorch 2.5 的 `torch._int_mm` 底层调用 cuBLAS INT8 GEMM，直接使用 Orin 的 INT8 Tensor Core。这是一个成熟的 API。

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

## 七、C++ 推理框架和量化的兼容性

第三篇的 C++ 替换和第四篇的 INT8 量化要叠加使用。但这里有一个关键兼容性问题：**torchao 的 Python 量化模块在 C++ libtorch 里不存在。**

### 7.1 问题

```
torchao.QuantizedLinear      → Python 类，C++ 里不存在
torch.ao 的 quantized module → TorchScript 兼容，但老 API
torch.jit.trace(量化模型)    → 取决于量化方式，可能失败
```

### 7.2 解决方案：Python 量化 + C++ 推理分离

最干净的做法是把量化和推理完全解耦：

```
阶段 1：Python 离线量化（在 5090 上，跑一次）
  ├── 加载 bf16 模型
  ├── 跑 SmoothQuant → 计算 smooth scales
  ├── 跑校准数据 → 计算 per-channel weight scales
  ├── 权重量化 → bf16 round to INT8
  └── 保存三个文件：
      weights_int8.pt      # 所有 Linear 层的 INT8 权重
      weight_scales.pt     # per-channel scale (float32)
      smooth_scales.pt     # SmoothQuant 平滑因子 (float32)

阶段 2：C++ 在线推理（在 Orin 上运行）
  ├── 加载 INT8 权重 + scales（普通 tensor，不依赖任何量化框架）
  ├── 对每个 Linear 层：
  │     input_int8 = 动态量化(input_bf16)
  │     output_int32 = torch::_int_mm(input_int8, weight_int8)
  │     output_bf16 = dequant(output_int32, scale_a, scale_w)
  └── 自定义算子（RoPE、MoE permute 等）照常 bf16
```

### 7.3 C++ 侧的 INT8 Linear 实现

```cpp
struct QuantizedLinear {
    torch::Tensor weight_int8;    // [out, in], int8
    torch::Tensor weight_scale;   // [out], float32
    torch::Tensor smooth_scale;   // [in], float32

    torch::Tensor forward(torch::Tensor input_bf16) {
        // 1. SmoothQuant 平滑
        auto x = input_bf16 / smooth_scale;
        // 2. 动态量化激活
        auto abs_max = x.abs().amax(-1, true);
        auto input_scale = abs_max / 127.0;
        auto x_int8 = (x / input_scale).round().clamp(-128, 127).to(torch::kInt8);
        // 3. INT8 GEMM (cuBLAS Tensor Core)
        auto out_int32 = torch::_int_mm(x_int8, weight_int8.t());
        // 4. Dequantize
        return out_int32.to(torch::kBFloat16) * input_scale * weight_scale;
    }
};
```

**这段 C++ 完全不依赖任何 Python 量化框架**。它只需要三个 `.pt` 文件（普通 tensor），用 `torch::load()` 加载即可。

### 7.4 为什么这是最佳方案

| 优势 | 说明 |
|------|------|
| Python 生态丰富 | 校准、调参、精度验证全在 Python 做 |
| C++ 侧极简 | 只用 `torch::_int_mm`，一个 API |
| 完全解耦 | 模型更新 → Python 重新量化 → C++ 代码不改 |
| 无框架依赖 | C++ 不依赖 torchao、torch.ao、bitsandbytes |
| 自定义算子不受影响 | quantize/dequant 在 QuantizedLinear 内部，外部全是 bf16 |

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

## 十、完整实施路线

```
Phase 1: 校准 + PTQ（5090 上，1-2 天）
  ├── Step 1: 从训练集采样 256 条校准数据
  ├── Step 2: 跑 SmoothQuant 收集激活值统计 + 计算平滑因子
  ├── Step 3: 对 Decoder + ViT 的 Linear 层做 W8A8 动态量化
  ├── Step 4: VQA 精度验证（对比 bf16 baseline）
  └── Step 5: Flow Action 精度验证（对比轨迹 MSE）

Phase 2: Orin 部署（Orin 上，1 天）
  ├── Step 6: 把量化后的模型部署到 Orin
  ├── Step 7: 确认 INT8 Tensor Core 被正确使用（nsys 验证 kernel 类型）
  ├── Step 8: 完整 benchmark（INT8 vs bf16，VQA + Flow Action）
  └── Step 9: 和第三篇的 C++ 框架叠加（INT8 + C++ decode + CUDA Graph）

Phase 3: 精度修复（如果需要，1-3 天）
  ├── Step 10: 如果某些层精度下降严重，该层回退 bf16
  ├── Step 11: 尝试 per-channel 量化替代 per-tensor（更细粒度）
  └── Step 12: 如果 PTQ 整体不够，评估 QAT 成本
```

### 10.1 预期结果

| 配置 | 预期延迟 | 对比 |
|------|----------|------|
| bf16 baseline (Python) | ~115ms | 1x |
| bf16 + C++ decode + CUDA Graph（第三篇） | ~80-90ms（估） | 1.3-1.4x |
| **INT8 + C++ decode + CUDA Graph** | **~50-60ms（估）** | **~2x** |

如果 GEMM 占 GPU 时间的 60%，INT8 把 GEMM 砍一半 → 整体节省 ~30%。叠加第三篇的 C++ 框架优化（消除 Python 开销 ~20%），总加速 ~2x。

---

## 十一、量化方案的不确定性和备选路线

### 11.1 torchao 在 Orin aarch64 上的兼容性

torchao 是 PyTorch 2024 年推出的量化库，主要在 x86 + CUDA 上测试。JetPack 的 PyTorch 2.5 是定制版，torchao 不保证兼容。

**备选方案**：
- 回退到 `torch.ao.quantization`（PyTorch 内置，更老但更稳定）
- 手动实现：权重量化 + `torch._int_mm` 调用（最底层，最可控）
- 在 5090 上量化，把量化后的权重文件（INT8 weights + scales）传到 Orin，Orin 上直接用

### 11.2 如果 W8A8 精度不够

降级路线：
1. **W8A8 → W8A16**：只量化权重，激活保持 bf16（省内存但不加速 GEMM）
2. **W8A8 → 混合精度**：敏感层用 bf16，其他用 INT8（逐层搜索最优配置）
3. **PTQ → QAT**：如果 PTQ 精度损失 > 2%，做量化感知训练恢复精度

### 11.3 如果 cuBLAS INT8 性能不达预期

cuBLAS 的 INT8 GEMM 在小 batch size (batch=1) 下可能不如 bf16——因为 INT8 kernel 的 launch overhead 和 dequant 开销在 batch=1 时占比较大。

**验证方法**：先写一个最小 benchmark，单独测 INT8 vs bf16 GEMM 的延迟：

```python
# 最小 INT8 GEMM benchmark
import torch

M, K, N = 1, 2048, 11008  # batch=1, FFN gate_proj 维度
A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')

# INT8 GEMM
%timeit torch._int_mm(A, B)

# bf16 GEMM 对照
A_bf16 = A.to(torch.bfloat16)
B_bf16 = B.to(torch.bfloat16)
%timeit A_bf16 @ B_bf16
```

如果 INT8 在 batch=1 下没有加速（可能发生），那量化的收益就只有内存节省。这时候应该优先做 **batched inference**（攒多个请求一起推理）来让 INT8 Tensor Core 充分利用。

---

## 十二、落地思考

### 12.1 量化不是银弹

从第一篇到第四篇，优化路线走下来的逻辑是：

```
第一篇：部署上去，跑通 baseline → 115ms
第二篇：试 FA2 → 不行（短序列 + Orin heuristic 不匹配）
第三篇：C++ 消除 Python 开销 → 预计 ~80-90ms
第四篇：INT8 量化砍 GEMM → 预计 ~50-60ms
```

每一步都是在上一步的基础上叠加。**没有哪一步能单独解决问题**——C++ 只消除了 ~20% 的 Python 开销，量化只砍了 ~30% 的 GEMM 时间。叠加起来才有 ~2x 的整体加速。

### 12.2 从 115ms 到 50ms 够不够？

wall-x 的动作控制频率要求：
- 基础操作（抓取/放置）：10-20 Hz → 50-100ms/step → **够了**
- 高精度操作（插入/对齐）：30-50 Hz → 20-33ms/step → **还差一点**

如果还需要继续压，可能的方向：
- **Batched generation**：攒 2-4 步的 decode，一次 launch 多个 GEMM
- **模型蒸馏**：从 3B 蒸馏到 1B，GEMM 直接缩小
- **Speculative decoding**：小模型预测 + 大模型验证

但这些已经不是单纯的推理优化问题了——它们需要的是一个**端侧 AI 操作系统**。第五篇会正式展开这个方向。

### 12.3 最重要的一句话

**先跑 INT8 GEMM 的最小 benchmark（上面 9.3 的脚本），确认 Orin 上 batch=1 的 INT8 真的比 bf16 快。** 如果不快，整个量化方案的前提就不成立——这时候应该优先做 batch 攒批或模型蒸馏。

这是整个量化方案的"第零步"。不跑这个，后面都是空中楼阁。

---

## 系列预告

本文是 **wall-x 机器人大模型部署系列** 的第四篇。

**第一篇**：[把 3B 大模型塞进机器人：RTX 5090 与 Jetson Orin 边缘端产品部署踩坑全记录](#)
- 环境搭建、CUDA 依赖链、profiling 数据、28 万+ kernel 分析

**第二篇**：[当算子逼近硬件极限：一次 Orin Profiling 引发的具身智能实时系统思考](#)
- FA2 编译全过程、FA2 vs SDPA benchmark、GEMM 带宽天花板证明、67% 框架空转发现

**第三篇**：[用 C++ 替换 Python 推理：在 Orin 上把 wall-x 跑到当前精度的极限](#)
- VQA 热路径分析、CUDA vs TensorRT 方案选型、libtorch + CUDA Graph 实施计划

**第四篇（本文）**：INT8 量化落地
- PyTorch vs TensorRT vs ONNX 选型、分层量化策略、SmoothQuant + W8A8
- 量化变换的算子图和权重陷阱（asym_dual_gmm、RoPE dtype、state_dict）
- C++ 推理框架和量化的兼容性：Python 量化 + C++ 推理分离方案
- Flow Action ODE 积分的精度风险和应对策略

**第五篇（预告）**：端侧 AI OS —— 从推理优化到系统架构
- 前四篇的结论汇聚：**具身智能的瓶颈不在 model，而在 runtime**
- VLA 是有损压缩的物理模拟器，提高刷新率比提高单次精度更有价值
- 从 model set runtime 到端侧 AI OS：感知-决策-执行的实时流水线、硬件抽象、资源调度
- **附 1.0 版本 GitHub 地址**

---

*测试环境：wall-oss-flow 3B 模型，bfloat16 精度 → INT8 量化，batch_size=1。测试平台：校准在 RTX 5090 上完成，部署和 benchmark 在 Jetson AGX Orin 64GB 上完成。JetPack 6.2.1，CUDA 12.6，PyTorch 2.5.0a0+872d972e41.nv24.08。2026 年 4 月。*
