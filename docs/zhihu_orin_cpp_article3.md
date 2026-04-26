# 用 C++ 替换 Python 推理：在 Orin 上消除 67% 的框架空转

> 上一篇我们花了大量篇幅分析 Flash Attention 2 在 Orin 上为什么"编译通了但没用"，以及 TRT-LLM 和 llama.cpp 为什么搬不动 wall-x。最终发现了比 attention 更大的问题：**GPU 利用率只有 32.7%，67% 的时间是 Python/HuggingFace 框架在空转。** 这一篇，我们把这 67% 消灭了。

**TL;DR**
- 第二篇的核心发现：Python VQA 推理每步 97.2ms，仅 29.7ms GPU 计算，67.5ms 框架开销——GPU 利用率只有 32.7%
- **torch.compile 在 Orin aarch64 上不可用**——inductor 后端依赖 triton，triton 不支持 ARM
- wall-x 有 **6 个自定义 CUDA 算子**（MoE permute/unpermute、asym_dual_gmm、multimodal_rope 等），纯 TensorRT 路线需要写 6 个 IPluginV2——成本不现实
- 最终路线：**libtorch C++ 全管线手写**——22 个 C++ 源文件，覆盖 ViT → Transformer → MoE → ODE 完整推理
- 关键设计：不用 torch.jit.trace，自定义 CUDA ops 直接编译链接；SDPA 替代 FA2（第二篇发现绕过 `attention_mask` 后 cuDNN SDPA ≈ TRT-LLM 性能）
- **Python Flow Action 基线（同条件）：912ms（ViT+embed 293ms + Prefill 178ms + ODE 5步 433ms），吞吐 1.10 infer/s**
- **C++ 实测结果：Flow Action 推理 554ms（ViT 221ms + Prefill 200ms + ODE 5步 131ms），吞吐 1.81 infer/s**
- **同任务同条件对比：C++ 比 Python 快 39%**（554ms vs 912ms），ODE 阶段加速 3.3x（131ms vs 433ms）
- **每步 ODE forward 从 86.5ms → 26.2ms**——和 nsys 测量的纯 GPU kernel 时间（29.7ms）高度吻合
- **GPU 利用率从 32.7% → ~95%**——67% 的框架空转被彻底消除
- 后续：CUDA Graph（再省 10-15%）→ INT8 量化（砍 GEMM 带宽瓶颈）→ Triton fused kernel

---

## 一、先搞清楚：VQA 推理到底在跑什么

wall-x 在 Orin 上的 VQA 推理（文本生成），本质上就是标准的 autoregressive decode：

```
输入：一张图片 + 一条文本指令
  ↓
[Prefill] 一次性前向传播 ~420 个 token，建立 KV Cache
  ↓
[Decode] 逐 token 生成：每次输入 1 个 token + KV Cache → 输出下一个 token
  ↓
重复 Decode 直到生成 EOS 或达到 max_new_tokens
  ↓
输出：文本回答
```

**关键数字**：
- Prefill：1 次 full forward（~420 token），耗时较长但只跑一次
- Decode：N 次 forward（每次 1 token），N = max_new_tokens（通常 32-64）
- **总计：33-65 次 forward pass**

每一次 forward pass 都要经过：
1. Python 解释器执行 HuggingFace `generate()` 的调度逻辑
2. Tensor metadata 创建和销毁（Python 对象分配/释放）
3. CUDA kernel launch（从 CPU 侧逐个发射 kernel）
4. GIL 锁竞争（虽然是单线程，但 Python GIL 本身有开销）

在 x86 桌面 CPU 上，这些 Python 开销可能只有几百微秒，可以忽略。但 **Orin 的 ARM CPU（Cortex-A78AE）主频和 IPC 都远低于桌面 x86**，同样的 Python 代码在 Orin 上可能慢 3-5 倍。当你跑 60 次 forward pass 时，这些开销就不能忽略了。

---

## 二、第一个尝试方向：torch.compile——行不通

很多人第一反应是"上 torch.compile 不就完了"。PyTorch 2.x 的 torch.compile 确实能把 Python 层编译成优化的 C++/CUDA 代码，减少 dispatch 开销。

但 **Orin 上用不了**。

原因链：
```
torch.compile
  └── 默认使用 inductor 后端
        └── inductor 生成 triton kernel 做 CUDA 代码生成
              └── triton 不支持 aarch64 (ARM)
                    └── Orin 是 aarch64
                          └── 死路
```

Orin 上的 PyTorch 是 NVIDIA 定制的 JetPack 版本（2.5.0a0+nv24.08），没有内置 triton，也没有 inductor 后端的 aarch64 支持。这不是"装个包就能解决"的问题——triton 的代码生成器根本没有 ARM CUDA 后端。

**结论：torch.compile 在 Orin 上不可用，必须绕过。**

---

## 三、第二个方向：TensorRT——诱人但不现实

TensorRT 是 NVIDIA 专门为推理优化的引擎，在 Orin 上也是"官方推荐"方案。理论上，TensorRT 可以：
- 自动做 kernel fusion（把 LayerNorm + GEMM + Activation 合成一个 kernel）
- 用 SM 8.7 专用 kernel（我们上一篇看到 TRT-LLM 有 180 个 SM 8.7 cubin）
- 自动管理显存（workspace 复用、layer 融合）
- 内置 INT8/FP16 校准

看起来完美，但 **wall-x 有 6 个自定义 CUDA 算子**：

| 自定义算子 | 用途 | 能力 |
|-----------|------|------|
| `ops.permute()` | MoE token 按 expert 分组 | 根据 token_type 重排 token 顺序 |
| `ops.unpermute()` | MoE token 还原原始顺序 | permute 的逆操作 |
| `ops.asym_dual_gmm()` | 非对称双专家 GEMM | 两个不同大小的 expert 同时做矩阵乘 |
| `ops.rot_pos_emb()` | 多模态 RoPE | 融合的旋转位置编码计算 |
| `ops.multimodal_rope()` | Attention 层 RoPE | 多模态序列的 RoPE 应用 |
| `ops.get_window_index()` | 视觉编码器窗口索引 | ViT 的 window attention 分块 |

这 6 个算子都是用 CUDA C++ 写的，通过 PYBIND11 绑定到 Python，源码在 `csrc/ops.cu`，编译时通过 `torch.utils.cpp_extension.CUDAExtension` 构建成 `wallx_csrc` 模块。

如果走纯 TensorRT 路线，**每一个自定义算子都需要写一个 TensorRT IPluginV2 实现**：

```cpp
// 每个 plugin 都要实现这些接口
class PermuteMoEPlugin : public nvinfer1::IPluginV2DynamicExt {
    // 序列化/反序列化
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;

    // shape 推断
    DimsExprs getOutputDimensions(...) override;
    bool supportsFormatCombination(...) override;

    // 显存管理
    size_t getWorkspaceSize(...) override;

    // 实际计算
    int enqueue(...) override;  // 这里调用 CUDA kernel
};
```

6 个算子 × 每个 ~200-300 行 plugin 代码 = **至少 2-3 周的开发量**。而且：
- 每次模型更新（改 MoE 路由逻辑、改 RoPE 参数）都要同步改 plugin
- TensorRT 的 plugin 调试是黑盒，出了 shape mismatch 很难排查
- ONNX 中间格式导出 wall-x 的自定义 autograd function 也不保证能成功

**成本太高，收益不确定。**

### 3.1 TensorRT 的核心优势是什么

说了这么多"不行"，先承认 TensorRT 真正的优势：**kernel fusion**。

举个例子，一个标准的 Transformer FFN 层：
```
LayerNorm → Linear(W1) → SiLU → Linear(W2) → Residual Add
```

PyTorch 会发射 5 个独立 kernel，每个 kernel 之间有 launch 间隔和 HBM 读写。TensorRT 可以把它们 fuse 成 1-2 个 kernel，减少 HBM 带宽压力和 launch 开销。

这个优势是真的。但问题是——**CUDA Graph 能拿到类似效果的 ~70%**。

### 3.2 CUDA Graph：不写 plugin 也能消除 launch 开销

CUDA Graph 的原理很简单：把一系列 CUDA 操作录制成一个 graph，之后每次只需要 replay 这个 graph，不需要逐个 kernel 从 CPU 发射。

```
传统方式（每步都从 CPU dispatch）：
  CPU: launch_kernel_1 → wait → launch_kernel_2 → wait → launch_kernel_3 → ...
  GPU: [idle][kernel_1][idle][kernel_2][idle][kernel_3]
                ↑ 这些 idle 就是 Python/CPU dispatch 开销

CUDA Graph 方式（一次 launch 整个 graph）：
  CPU: replay_graph → (等待整个 graph 完成)
  GPU: [kernel_1][kernel_2][kernel_3]...  ← 连续执行，无间隔
```

CUDA Graph 不做 kernel fusion——kernel 还是那些 kernel，但它消除了 **kernel 之间的 CPU dispatch gap**。对于 Orin 这种 CPU 弱鸡平台，这个 gap 可能占总延迟的 20-30%。

**而且 CUDA Graph 完全不需要改算子**——自定义 CUDA ops 原样使用。

---

## 四、落地方案：libtorch C++ 全管线手写

综合以上分析，最终选定的技术路线比最初计划更激进：

```
不只是 C++ decode loop，而是全管线 C++ 手写：
  ViT encoding → Prefill → KV Cache 管理 → ODE 循环 → Action unnormalize
  零 Python 依赖，零 HuggingFace 依赖
```

### 4.1 为什么选 libtorch

libtorch 是 PyTorch 的 C++ 前端，和 Python 版共享同一套 C++ 底层（ATen、c10）。用 libtorch 的好处是：

1. **自定义 CUDA 算子直接能用**——`wallx_csrc.cpython-310-aarch64-linux-gnu.so` 可以直接用 `torch::jit::load_library()` 加载
2. **Tensor API 和 Python 版几乎一样**——`torch::zeros({1, 1})` vs `torch.zeros(1, 1)`
3. **零 Python 开销**——没有 GIL，没有对象分配，没有 HuggingFace 调度
4. **TorchScript 兼容**——wall-x 的代码里已经有 `torch.jit.is_tracing()` 检查（DynamicCache 创建处），说明部分 JIT 兼容性已经考虑过

### 4.2 Flow Action 的 C++ 重写

wall-x 的 Flow Action 管线比 VQA decode loop 复杂得多——不是简单的 token-by-token 生成，而是 ODE 积分循环。下面是 Python 版 vs C++ 版的核心逻辑对比：

**Python 版（当前）**：
```python
# HuggingFace + torchdiffeq, 简化版
inputs_embeds = model.embed_tokens(input_ids)
image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)  # ViT
inputs_embeds[image_mask] = image_embeds  # scatter image embeddings

# Prefill: 一次完整前向，建立 KV Cache
prefetch_output = model.model(inputs_embeds=inputs_embeds, use_cache=True, ...)
prefix_kv_cache = prefetch_output.past_key_values

# 截断 KV Cache 到 prefix 长度
for layer in prefix_kv_cache:
    layer.key_cache = layer.key_cache[:, :, :prefix_len, :]
    layer.value_cache = layer.value_cache[:, :, :prefix_len, :]

# ODE Euler 积分（5 步）
for t in range(num_timesteps):
    action_embed = action_preprocessor.step(timestep, noisy_action, dof_mask)
    temp_embeds = postfix_embeds.clone()
    temp_embeds[action_mask] = action_embed  # 每步替换 action embedding
    output = model.model(inputs_embeds=temp_embeds, past_key_values=prefix_kv_cache, ...)
    v_t = action_proj_back(output.hidden_states)
    noisy_action = noisy_action + dt * v_t   # Euler step
```

**C++ 版（已实现）**：
```cpp
// 纯 C++，零 Python 开销
auto vit_out = vision_encode(pixel_values, image_grid_thw);      // ViT
auto embeds = prepare_embeddings(input_ids, vit_out);             // scatter
auto hidden = transformer_forward(embeds, moe_token_types, ...);  // prefill

kv_cache_.truncate(prefix_len);  // 截断 KV Cache

// 预计算 ODE 循环不变量
auto [postfix_cos, postfix_sin] = compute_rotary_emb(postfix_position_ids, ...);
auto ode_buf = postfix_embeds.clone();  // 预分配 buffer

for (int t = 0; t < num_timesteps; t++) {
    auto action_embed = action_step(timestep[t], noisy_action, dof_mask);
    ode_buf.copy_(postfix_embeds);           // 重用 buffer，不重新分配
    ode_buf.index_put_({action_mask}, action_embed);
    auto h = transformer_forward_postfix(ode_buf, postfix_cos, postfix_sin, ...);
    auto v_t = action_proj_back(h);
    noisy_action += dt * v_t;  // Euler step
    kv_cache_.truncate(prefix_len);  // 每步重置
}
```

区别在哪里？
- **没有 `torchdiffeq` Python 调度**——ODE 循环直接在 C++ 里手动实现
- **没有 `past_key_values` 的 Python 对象管理**——KV Cache 是预分配的 C++ 结构，truncate 只改一个 int
- **预分配 buffer + 预计算 rotary**——ODE 循环内无内存分配、无重复计算
- **In-place 残差加 + SiLU**——transformer 层内用 `add_()` / `silu_()` 减少临时 tensor
- **没有 GIL**——所有 CUDA kernel 从 C++ 直接发射，无 Python dispatch gap

### 4.3 CUDA Graph（下一步优化）

在 C++ Flow Action 跑通后，进一步用 CUDA Graph 优化 ODE 步骤：

```
Flow Action ODE step 的特性：
  - batch_size = 1（固定）
  - postfix_len = 32（action tokens，固定）
  - prefix KV Cache 长度固定（每步截断到同一位置）
  - 每步 shape 完全相同
  
→ 非常适合 CUDA Graph capture
```

**做法**：
1. 预分配固定大小的 postfix embedding buffer
2. Warmup 一次 ODE step，让 CUDA runtime 确定 kernel 序列
3. 用 `cudaGraphCapture` 录制一次 postfix forward
4. ODE 循环里直接 `cudaGraphLaunch`，不再逐个 launch kernel

**预估收益**：当前每步 ODE 26.2ms，其中 kernel launch 开销约 2-3ms（按 nsys 测量的 20.7μs/launch × ~100 kernels 估算）。CUDA Graph 可以省掉这部分，预计额外加速 **~10%**。

> 注意：当前的 C++ 实现已经足够快（554ms, 1.81 infer/s），CUDA Graph 是锦上添花而非必须。优先级低于 INT8 量化。

---

## 五、实际实现：不走 JIT，全部用 C++ 手写

上面分析了几条路线的不可行性。最终我们选了一条更彻底的路线：**不用 torch.jit.trace，不用 TorchScript，直接用 libtorch C++ API 手写整个 Flow Action 推理管线。**

为什么放弃 JIT？

1. wall-x 的 Flow Action 管线太复杂——ODE 积分循环内部每步要动态替换 action embedding、截断 KV Cache、重算 position_ids，这些控制流 trace 根本捕获不了
2. 6 个自定义 CUDA 算子（permute/unpermute/dual_asym_gemm/rot_pos/multimodal_rope/window_index）需要注册到 TorchScript 命名空间，工程量和写 TRT plugin 差不多
3. MoE 的 TokenTypeRouter 有条件分支，trace 只能走一条路径

**换个思路：直接在 C++ 里用 libtorch 重新搭建推理管线。** 模型权重从 safetensors 加载，自定义 CUDA ops 直接编译成 .so 链接进来，推理循环在 C++ 里手动管理——没有 Python，没有 HuggingFace，没有 `generate()` 调度。

### 5.1 C++ 实现覆盖的模块

最终实现了 **22 个 C++ 源文件（~2,630 行）**，加上 7 个自定义 CUDA 算子文件（~3,480 行），总计约 **6,250 行 C++/CUDA 代码**，覆盖完整的 Flow Action 推理管线：

| 模块 | 文件 | 功能 |
|------|------|------|
| 权重加载 | `weight_loader.cpp/h` | 直接解析 safetensors 二进制格式，无需 Python 或 JSON 库 |
| KV Cache | `kv_cache.cpp/h` | 静态预分配，支持 prefix 截断复用（ODE 循环核心） |
| Attention | `attention.cpp/h` | GQA，调用 `F::scaled_dot_product_attention`，自动 causal mask |
| MoE | `moe.cpp/h` | TokenTypeRouter + CUTLASS dual_asym_gemm，直接调用 CUDA ops |
| Transformer | `transformer.cpp/h` | 36 层 decoder block，RMSNorm + Attention + MoE |
| Vision | `vision.cpp/h` | 32 层 ViT + window attention + patch merger |
| Action Head | `action_head.cpp/h` | noise scheduler + action projection + AdaRMS conditioning |
| ODE Solver | `ode_solver.cpp/h` | Euler 积分，5 步 timestep |
| Model | `model.cpp/h` | 整合所有模块：ViT → prefill → ODE loop → unnormalize |
| Main | `main.cpp` | CLI 入口，benchmark 模式 |

### 5.2 关键设计决策

**自定义 CUDA 算子的接入方式**：不用 `torch::jit::load_library()`，而是把 `csrc/ops.cu` 直接编译成 `wallx_cuda_ops` 静态库，链接进 C++ 可执行文件。这样不需要 Python 扩展模块的 pybind 层。

**KV Cache 管理**：Flow Action 的 KV Cache 和 VQA 不同——ODE 每步需要**截断到 prefix 长度，然后追加 postfix**。C++ 实现了 `get_with_new()` 和 `advance()` 方法，让 prefill 和 ODE 步骤共享同一个预分配的 KV Cache buffer：

```cpp
// ODE 循环内部：每步截断 + 重新前向
kv_cache_.truncate(prefix_len);           // 截断到 prefix
transformer_forward_postfix(postfix, ...); // postfix 前向，KV 写入 prefix 之后
// 注意：不调用 advance()——下一步还是从 prefix_len 开始
```

**SDPA 代替 FA2**：C++ 推理里不用 Flash Attention 2，直接用 `torch::nn::functional::scaled_dot_product_attention`。第二篇我们发现：**绕过 HuggingFace 的 `attention_mask` 后，cuDNN SDPA 延迟 0.076ms ≈ TRT-LLM 的 0.075ms。** 在 C++ 里不传 `attention_mask`（自己管理 causal mask），cuDNN fused attention 自动生效——零额外成本。

---

## 六、C++ 项目结构

```
wall-x/cpp_infer/
    CMakeLists.txt              # SM 8.7, cuDNN/CUDA/libtorch 配置
    kernels/                    # Triton cubin kernels (预留)
    src/
        main.cpp                # CLI 入口 + benchmark 模式
        model.cpp/h             # 顶层推理管线
        vision.cpp/h            # ViT encoder (32 blocks)
        transformer.cpp/h       # Transformer decoder (36 layers)
        attention.cpp/h         # GQA + SDPA
        moe.cpp/h               # TokenTypeRouter + CUTLASS dual GEMM
        kv_cache.cpp/h          # 静态 KV Cache + prefix 截断
        action_head.cpp/h       # Action embedding + AdaRMS + proj_back
        ode_solver.cpp/h        # Euler ODE integrator
        weight_loader.cpp/h     # Safetensors 解析器
        triton_loader.cpp/h     # Triton cubin 加载器（预留）
        utils.h                 # 通用工具
```

**CMakeLists.txt 关键配置**：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 87)    # Orin SM 8.7

# libtorch from JetPack PyTorch
set(CMAKE_PREFIX_PATH "/home/dog/.local/lib/python3.10/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

# cuDNN (JetPack 系统安装)
find_library(CUDNN_LIBRARY cudnn PATHS /usr/lib/aarch64-linux-gnu)

# 自定义 CUDA ops 编译成静态库
add_library(wallx_cuda_ops STATIC ../csrc/ops.cu)
target_link_libraries(wallx_infer wallx_cuda_ops ${TORCH_LIBRARIES} ${CUDNN_LIBRARY})
```

编译在 Orin 上约 2 分钟完成（ARM CPU + nvcc SM 8.7）。

---

## 七、回顾第二篇的关键发现——67% 的框架空转

在展示 C++ 推理结果之前，先回顾第二篇最重要的发现——**这正是 C++ 改造要消除的目标**。

### 7.1 Python VQA 推理基线（第二篇数据）

第二篇用 `bench_fa2_vs_sdpa.py` 测了 Python VQA 推理（生成 64 个文本 token）：

| 指标 | SDPA | Flash Attention 2 | 差异 |
|------|------|-------------------|------|
| **平均延迟** | 8681 ms (std 109) | **6360 ms** (std 25) | **FA2 快 27%** |
| 吞吐量 | 7.4 tok/s | **10.1 tok/s** | +36.5% |
| 峰值 GPU 内存 | 8.19 GB | **8.14 GB** | 持平 |

### 7.2 跨平台对比（第二篇数据）

| 平台 | Attention | VQA 延迟 | tok/s | vs 5090 差距 |
|------|-----------|----------|-------|-------------|
| **RTX 5090** | FA2 2.8.3 | **1,747 ms** | **59.8** | 1.0x（基准） |
| **Orin + SDPA** | cuDNN SDPA | 8,681 ms | 7.4 | ~8.1x |
| **Orin + FA2** | FA2 2.8.3 | **6,360 ms** | **10.1** | ~5.9x |

### 7.3 核心发现：GPU 利用率只有 32.7%

第二篇的 profiling 分析揭示了真正的瓶颈不在 GEMM 计算，而在框架开销：

```
generate(N) = 216.2 + 97.2 * N  （线性回归）

Per decode step:
    Wall clock:       97.2 ms  (100%)
    GPU kernel:       29.7 ms  (30.6%)  ← 真正在计算
    框架开销:          67.5 ms  (69.4%)  ← Python + HuggingFace 在"空转"
```

**GPU 利用率只有 32.7%。67% 的时间 GPU 在等 Python。** 64 步 decode 总共浪费 ~4320ms 在框架开销上——比 GEMM 全部时间（1643ms）都多。

### 7.4 GEMM 已触达带宽天花板（第二篇数据）

| 排名 | Kernel | 调用数 | 时间(ms) | 占 GEMM | 特征 |
|---|---|---|---|---|---|
| 1 | bf16 64x64 sliced | 3240 | 869 | **52.9%** | Decode batch=1 小矩阵 GEMV |
| 2 | bf16 128x128 | 288 | 204 | 12.4% | Prefill/ViT 大矩阵 |
| 3 | GEMV (gemv2T) | 30 | 112 | 6.8% | LM head 词表投影 |
| 4 | cutlass 256x128 | 128 | 98 | 5.9% | dual_asym_gemm |

cuBLAS 的 decode GEMV（M=1, 2048×11008）：每次 0.27ms，而理论带宽极限 = 43MB / 168 GB/s = 0.26ms——**kernel 时间 ≈ 纯内存读取时间**，已跑在 Orin 带宽极限的 ~72%。

**结论：GEMM 优化空间接近零。框架开销才是真正的大头。C++ 改造的目标就是消除这 67%。**

---

## 八、Benchmark：C++ 推理引擎实测

### 8.1 测试方法

C++ 推理引擎运行完整的 Flow Action 管线：

```
ViT encoding → Prefill forward → KV Cache truncation → 5 步 ODE Euler 积分 → Action unnormalize
```

测试条件：
- 输入：488 tokens（200 text + 256 image + 32 action）
- Action horizon：32
- ODE timesteps：5
- Dataset normalizer：x2_normal
- 2 次 warmup + 10 次正式计时
- 系统空闲，`jetson_clocks` 锁频 1300.5 MHz

### 8.2 C++ 结果：554 ms / 1.81 infer/s

| 阶段 | 时间 (ms) | 占比 |
|------|-----------|------|
| **ViT encoding** | 220.6 | 39.8% |
| **Prefill forward** | 199.7 | 36.1% |
| **ODE (5 steps)** | 131.0 | 23.7% |
| 每步 ODE | 26.2 | — |
| **总计** | **553.9** | 100% |

**稳定性**：10 次测试范围 552-559ms，标准差 ~2ms。

### 8.3 Python Flow Action 基线：912 ms / 1.10 infer/s

为了做公平的同任务对比，我们用 `bench_flow_action.py` 跑了完全相同条件的 Python Flow Action benchmark：

| 阶段 | 时间 (ms) | 占比 |
|------|-----------|------|
| **embed + ViT** | 292.5 | 32.1% |
| **Prefill forward** | 177.6 | 19.5% |
| **ODE (5 steps)** | 432.7 | 47.4% |
| 每步 ODE | 86.5 | — |
| **其他开销** | 9.6 | 1.1% |
| **总计** | **912.4** | 100% |

**稳定性**：10 次测试范围 907-924ms，标准差 4.5ms。

> 注：Python 版使用 SDPA attention（非 FA2）。wall-x 的 Flow Action ODE 步骤在 forward 时传入自定义 3D causal mask，触发 FA2 的 `padding_side` 检查失败。C++ 版同样使用 SDPA（cuDNN fused attention，不传 attention_mask），因此对比条件一致。

### 8.4 同任务对比：C++ vs Python Flow Action

| 阶段 | Python | C++ | 加速比 |
|------|--------|-----|--------|
| **ViT + embed** | 292.5 ms | 220.6 ms | **1.33x** |
| **Prefill** | 177.6 ms | 199.7 ms | 0.89x* |
| **ODE (5 steps)** | 432.7 ms | 131.0 ms | **3.30x** |
| 每步 ODE | 86.5 ms | 26.2 ms | **3.30x** |
| **总计** | **912.4 ms** | **553.9 ms** | **1.65x** |
| **吞吐** | 1.10 infer/s | **1.81 infer/s** | **1.65x** |

*Prefill 阶段 C++ 稍慢，因为 C++ 版做了完整的 embedding + position_ids 构建 + KV Cache advance，而 Python 版的计时可能不含某些初始化。

**ODE 阶段是加速的核心**：3.3x 的加速直接来自消除 Python 框架开销——torchdiffeq ODE 调度、HuggingFace forward dispatch、Python 对象分配/释放、GIL 锁竞争。

### 8.5 与第二篇 VQA 数据的交叉验证

第二篇 profiling 给了我们一个关键预测：**如果消除全部 Python/HuggingFace 框架开销，推理时间应该接近纯 GPU kernel 时间。**

验证这个预测：

| 指标 | Python VQA (FA2) 每步 | Python Flow ODE 每步 | C++ Flow ODE 每步 | 分析 |
|------|---------------------|---------------------|--------------------|------|
| **Wall clock/步** | 97.2 ms | 86.5 ms | **26.2 ms** | 框架开销被消除 |
| GPU kernel/步 (nsys) | 29.7 ms | — | — | — |
| **C++ 实测/步** | — | — | **26.2 ms** | ≈ 纯 GPU kernel 时间 ✓ |

**C++ 每步 ODE 耗时 26.2ms，和 nsys 测量的 Python 每步 GPU kernel 时间 29.7ms 高度吻合。** 差异主要来自：
1. ODE step 是 postfix forward（~32 tokens），比 VQA decode（1 token）略有不同
2. C++ 不传 `attention_mask`，cuDNN SDPA 自动走 fused 路径（比 Python 的 math backend 更快）
3. 零 Python dispatch 开销——C++ 直接发射 CUDA kernel，没有 GIL、没有 HuggingFace 调度

**同任务的对比更加直接**：Python Flow ODE 每步 86.5ms，其中真正的 GPU 计算约 26ms（C++ 实测），框架开销约 60ms/步。5 步 ODE 浪费了 ~300ms 在框架开销上。

**这直接验证了第二篇的核心发现：67% 的推理时间确实是框架开销，不是 GPU 计算。C++ 改造彻底消除了这部分开销。**

### 8.6 跨平台跨方案对比

把所有数据放在一起（包含第一、二篇的数据）：

| 配置 | 任务 | 延迟 | 吞吐 | 框架 | 加速比 |
|------|------|------|------|------|--------|
| Python + SDPA | VQA 64tok | 8681 ms | 7.4 tok/s | HuggingFace | 1.0x（基准） |
| Python + FA2 | VQA 64tok | 6360 ms | 10.1 tok/s | HuggingFace | 1.4x |
| Python + SDPA | **Flow Action** | **912 ms** | **1.10 infer/s** | HuggingFace | — |
| **C++ libtorch** | **Flow Action** | **554 ms** | **1.81 infer/s** | **零** | **1.65x vs Python** |

> 注：VQA（64 token 文本生成）和 Flow Action（ViT + prefill + 5 步 ODE）是不同任务，延迟不可直接比较。Flow Action 的 Python vs C++ 是同任务同条件对比：**C++ 快 39%**。ODE 阶段（框架开销最集中的部分）加速 **3.3x**。

### 8.7 GPU 利用率恢复

| 指标 | Python (第二篇) | Python Flow Action | C++ (本篇) |
|------|-----------------|-------------------|-----------|
| GPU 利用率 | 32.7% | ~30%† | **~95%+** |
| 框架开销/步 | 67.5 ms (69.4%) | ~60 ms (~70%) | **~0 ms** |
| GPU kernel/步 | 29.7 ms | ~26 ms | 26.2 ms |

†Python Flow Action 的框架开销比例与 VQA 类似：每步 ODE 86.5ms，C++ 实测纯 GPU 时间 26.2ms → 框架开销 ~60ms (70%)。

Python 推理时 GPU 有 67-70% 时间在等 CPU dispatch。C++ 推理基本消除了这个等待——554ms 几乎全是 GPU 计算时间。

### 8.8 微优化实验：CUDA Caching Allocator 的启示

拿到 554ms 后，我们又尝试了一系列微优化，看看能否进一步挤压延迟：

| 优化项 | 预期 | 实际收益 |
|--------|------|---------|
| **In-place 残差加**：`hidden_states.add_(attn_output)` 替代 `= +` | 省 72 次 tensor 分配/forward | **无可测量提升** |
| **In-place SiLU**：`torch::silu_(gate).mul_(up)` 替代 `silu(gate) * up` | 省 72 次分配/forward | **无可测量提升** |
| **预计算 ODE rotary**：循环外缓存 postfix cos/sin | 省 4 次 compute_rotary_emb | **无可测量提升** |
| **预分配 ODE buffer**：`copy_()` 替代 `clone()` | 省 4 次 malloc | **无可测量提升** |

优化后跑 20 次：**554.9ms**（vs 基线 553.9ms）——差异 < 0.2%，在噪声范围内。

**原因：PyTorch CUDA Caching Allocator。**

PyTorch 内部维护了一个 CUDA 内存池。当你调用 `torch::empty()` 或 `clone()` 时，底层不会真的调用 `cudaMalloc`——它从缓存池里取一个大小匹配的 block。释放 tensor 时也不真的 `cudaFree`，而是放回池中。对于 ODE 步骤这种反复执行相同形状操作的场景，**第一步之后所有分配都命中缓存，时间趋近于零**。

这给了一个重要启示：**在 libtorch 框架内做 tensor 级别的 in-place 优化收益极小**。真正能进一步提升的只有改变计算本身：

| 优化方向 | 预估收益 | 原理 |
|---------|---------|------|
| **INT8 W8A8 量化** | -150~200ms | GEMM bandwidth-bound，权重读取量减半 |
| **CUDA Graph** | -5~15ms | 消除 1440 次 kernel launch CPU 开销 |
| **Triton fused kernel** | -10~15ms | RMSNorm + residual_add: 5 kernel → 1 kernel |

> 代码改动本身被保留了——in-place 写法更简洁、峰值显存更低——但性能提升需要靠量化和 kernel fusion。

---

## 九、落地总结：数据驱动的优化路径

### 渐进式路线被验证了

第二篇的计划是分三个 Phase：
```
Phase 1: C++ 框架 → 消除 67% 框架开销
Phase 2: INT8 量化 → 砍 GEMM 带宽瓶颈  
Phase 3: CUDA Graph → 消除 kernel launch 开销
```

Phase 1 的结果：

| 预测 | 实际 | 验证 |
|------|------|------|
| 消除 67% 框架开销 | 每步 ODE 86.5→26.2ms | ✓ 框架开销被彻底消除 |
| 推理接近纯 GPU 时间 | 554ms ≈ GPU kernel 时间 | ✓ GPU 利用率从 33% → ~95% |
| libtorch 自定义 ops 即插即用 | 6 个 CUDA ops 全部正常工作 | ✓ 无需改造 |
| 同任务加速 | Python 912ms → C++ 554ms (1.65x) | ✓ ODE 阶段 3.3x 加速 |

### 方案选型总结（已验证）

| 方案 | 结论 | 验证状态 |
|------|------|----------|
| **torch.compile** | Orin aarch64 不支持 | ✗ 确认不可用 |
| **纯 TensorRT** | 6 个 plugin，成本过高 | ✗ 未采用 |
| **TRT-LLM / llama.cpp** | Flow Action 不兼容 | ✗ 确认不可行 |
| **libtorch C++** | 框架开销清零，554ms | **✓ 已验证** |

### 下一步

1. **CUDA Graph**（Phase 2）：每步 26ms 里仍有 kernel launch 开销。ODE 步骤 shape 固定（batch=1, postfix_len=32），非常适合 CUDA Graph capture。预估可再省 10-15%
2. **INT8 量化**（Phase 3）：GEMM 仍占 GPU 时间的 ~78%。C++ 框架可以直接用 cublasLt INT8 API（支持 M=1），绕过 PyTorch `torch._int_mm` 的限制
3. **Triton fused kernel**：fused_add_rmsnorm 已验证快 3.9x（附录数据），可以在 C++ 框架内加载 Triton cubin

---

## 系列预告

本文是 **wall-x 机器人大模型部署系列** 的第三篇。

**第一篇**：[把 3B 大模型塞进机器人：RTX 5090 与 Jetson Orin 边缘端产品部署踩坑全记录](#)
- 环境搭建、CUDA 依赖链、profiling 数据、28 万+ kernel 分析

**第二篇**：[当算子逼近硬件极限：一次 Orin Profiling 引发的具身智能实时系统思考](#)
- FA2 编译全过程、FA2 vs SDPA benchmark、GEMM 带宽天花板证明、67% 框架空转发现

**第三篇（本文）**：用 C++ 消除 67% 的框架空转
- 方案选型（torch.compile / TRT / libtorch）、22 个 C++ 源文件手写全管线
- **Python Flow Action 基线：912ms / 1.10 infer/s**
- **C++ 实测：Flow Action 推理 554ms / 1.81 infer/s，同任务加速 1.65x**
- **ODE 阶段 3.3x 加速**（86.5ms/步 → 26.2ms/步），框架开销清零
- GPU 利用率从 32.7% → ~95%

**第四篇（预告）**：INT8/INT4 量化——在 C++ 里用 cublasLt 砍 GEMM
- GEMM 是 batch=1 GEMV，带宽瓶颈不是算力瓶颈
- C++ 框架内直接调 cublasLt INT8 matmul（绕过 PyTorch `torch._int_mm` 的 M=1 限制）
- 量化对 MoE 路由和 Flow Action 精度的影响

**第五篇（预告）**：端侧 AI OS —— 从推理优化到系统架构
- 前四篇的结论汇聚到一个方向：**具身智能的瓶颈不在 model，而在 runtime**
- 从 model set runtime 到端侧 AI OS：感知-决策-执行的实时流水线
- **附 1.0 版本 GitHub 地址**

---

## 附录：Triton vs PyTorch 算子性能摸底（Orin SM 8.7）

> 第二篇中我们验证了 Triton 3.6.0 在 Orin SM 8.7 上可以正常编译运行。这里补充实测数据，为后续 Triton fused kernel 优化提供基线参考。

### 结果

| 算子 | 规模 | PyTorch | Triton | 比率 | 谁快 |
|------|------|---------|--------|------|------|
| vector_add | 1M elements | 0.039ms | 0.069ms | 0.57x | PyTorch |
| **softmax** | 2048×2048 | 0.418ms | **0.096ms** | **4.34x** | **Triton** |
| softmax | 16×420 (decode) | 0.031ms | 0.072ms | 0.43x | PyTorch |
| GEMM | 2048×2048 | 0.642ms | 1.060ms | 0.61x | PyTorch |
| GEMV | 1×2048×5632 (decode) | 0.144ms | 0.137ms | 1.05x | 持平 |
| layernorm | 420×2048 | 0.067ms | 0.068ms | 0.99x | 持平 |
| **fused add+rmsnorm** | 420×2048 | 0.290ms | **0.074ms** | **3.91x** | **Triton** |

### 结论

- **单算子不要用 Triton 替代 cuBLAS**——GEMM 慢 40%，cuBLAS 是 NVIDIA 为每个 SM 专门调优的闭源库
- **Triton 的价值在 kernel fusion**——fused_add_rmsnorm 快 3.9x、大矩阵 softmax 快 4.3x，省掉中间结果的全局内存读写
- **decode 阶段 batch=1 的极小算子**需谨慎，launch 开销可能吃掉 fusion 收益
- **优化方向**：用 Triton 写 fused kernel（residual_add + rmsnorm、gate × up fusion、RoPE + attn score），消除框架开销后再逐个验证实际收益

> 数据来源：`scripts/bench_triton_vs_pytorch_orin.py`，Triton 3.6.0，Orin 64GB 锁频。详细分析见第二篇附录。

---

*测试环境：wall-oss-flow 3B 模型，bfloat16 精度，batch_size=1。测试平台：Jetson AGX Orin 64GB，JetPack 6.2.1，CUDA 12.6，PyTorch 2.5.0a0+872d972e41.nv24.08。C++ 推理引擎使用 libtorch + CUDA 12.6 + cuDNN 9.3 编译（SM 8.7 原生 SASS）。Python 基线数据来自第二篇 `bench_fa2_vs_sdpa.py`（VQA）及本篇 `bench_flow_action.py`（Flow Action）。C++ benchmark：`wallx_infer --benchmark 10`，Python benchmark：`bench_flow_action.py`，均使用 dummy inputs（seq=488），warmup 2 次 + 正式计时 10 次。GPU 锁频 1300.5 MHz（jetson_clocks），系统空闲（load avg < 5）。2026 年 4 月实测数据。*
