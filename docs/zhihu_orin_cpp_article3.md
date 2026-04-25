# 用 C++ 替换 Python 推理：在 Orin 上把 wall-x 跑到当前精度的极限

> 上一篇我们花了大量篇幅分析 Flash Attention 2 在 Orin 上为什么"编译通了但没用"，以及 TRT-LLM 和 llama.cpp 为什么搬不动 wall-x。结论是：**推理引擎换不了，必须在现有 PyTorch 栈内优化。** 那这篇文章的问题就很直接了——PyTorch 推理到底慢在哪里？Python 层的开销有多大？能不能用 C++ 把它消掉？

**TL;DR**
- wall-x VQA 推理热路径：1 次 prefill + N 次 decode（N=32-64），每次 decode 都要经过 Python 解释器 + HuggingFace generate() 完整调度
- **torch.compile 在 Orin aarch64 上不可用**——inductor 后端依赖 triton，triton 不支持 ARM
- wall-x 有 **6 个自定义 CUDA 算子**（MoE permute/unpermute、asym_dual_gmm、multimodal_rope 等），纯 TensorRT 路线需要写 6 个 IPluginV2——成本不现实
- 推荐路线：**libtorch C++ decode 循环 + CUDA Graph**，自定义算子直接加载现有 .so，零改造
- CUDA Graph 能把 decode step 的所有 kernel 打包成一个 graph，一次 launch 全部执行——拿到 TensorRT "kernel fusion" ~70% 的效果，但不需要写 plugin
- 后续如果还不够，`torch_tensorrt` 混合编译可以让标准层走 TRT、自定义层回退 PyTorch

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

## 四、落地方案：libtorch C++ + CUDA Graph

综合以上分析，最终选定的技术路线是：

```
Phase 1: libtorch C++ decode loop     ← 消除 Python 开销
Phase 2: CUDA Graph for decode step   ← 消除 kernel launch 间隔
Phase 3: torch_tensorrt 混合编译      ← 可选，标准层走 TRT fusion
```

### 4.1 为什么选 libtorch

libtorch 是 PyTorch 的 C++ 前端，和 Python 版共享同一套 C++ 底层（ATen、c10）。用 libtorch 的好处是：

1. **自定义 CUDA 算子直接能用**——`wallx_csrc.cpython-310-aarch64-linux-gnu.so` 可以直接用 `torch::jit::load_library()` 加载
2. **Tensor API 和 Python 版几乎一样**——`torch::zeros({1, 1})` vs `torch.zeros(1, 1)`
3. **零 Python 开销**——没有 GIL，没有对象分配，没有 HuggingFace 调度
4. **TorchScript 兼容**——wall-x 的代码里已经有 `torch.jit.is_tracing()` 检查（DynamicCache 创建处），说明部分 JIT 兼容性已经考虑过

### 4.2 VQA Decode 循环的 C++ 重写

VQA 的 decode 循环本质上很简单，下面是 Python 版 vs C++ 版的核心逻辑：

**Python 版（当前）**：
```python
# HuggingFace generate() 内部，简化版
for step in range(max_new_tokens):
    # --- Python 开销区 ---
    model_inputs = self.prepare_inputs_for_generation(input_ids, past_key_values=past)
    # prepare_inputs_for_generation 里有大量 Python 逻辑：
    #   cache_position 计算、attention_mask 更新、position_ids 生成...
    
    # --- GPU 计算 ---
    outputs = self(**model_inputs)  # 一次 forward pass
    
    # --- Python 开销区 ---
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    # stopping criteria 检查、eos 判断、beam search 逻辑（如果有）...
    
    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    past = outputs.past_key_values
```

**C++ 版（目标）**：
```cpp
// 纯 C++ decode loop，零 Python 开销
for (int step = 0; step < max_new_tokens; step++) {
    // 直接构造 input tensor，不经过 prepare_inputs_for_generation
    auto out = model.forward({input_ids, attention_mask, position_ids, kv_cache});
    
    auto logits = out.toTuple()->elements()[0].toTensor();
    auto next_token = logits.index({0, -1}).argmax();
    
    if (next_token.item<int64_t>() == eos_token_id) break;
    
    // 更新 input_ids, attention_mask, position_ids（直接操作 tensor）
    input_ids.index_put_({0, 0}, next_token);
    position_ids.add_(1);
    // kv_cache 由模型内部管理
}
```

区别在哪里？
- **没有 `prepare_inputs_for_generation()`**——这个函数在 HuggingFace 里有几百行 Python 逻辑
- **没有 stopping criteria 的 Python 循环**——直接比较 eos token
- **没有 Python 对象创建/销毁**——所有 tensor 在循环外预分配
- **没有 GIL**——纯 C++ 执行

### 4.3 CUDA Graph 叠加

在 C++ decode loop 跑通后，进一步用 CUDA Graph 优化：

```
VQA decode step 的特性：
  - batch_size = 1（固定）
  - seq_len = 1（每步只输入一个 token，固定）
  - KV Cache 每步增长 1 个 token（shape 变化可预测）
  
→ 完美适合 CUDA Graph capture
```

**做法**：
1. 预分配 max_length 的 KV Cache（StaticCache），避免 shape 变化
2. Warmup 几次 decode step，让 CUDA runtime 确定 kernel 序列
3. 用 `torch.cuda.CUDAGraph()` capture 一次 decode step
4. decode 循环里直接 `graph.replay()`，不再逐个 launch kernel

**挑战**：
- HuggingFace 默认用 `DynamicCache`（每步 concat 新 KV），shape 会变 → 需要换成 `StaticCache`
- MoE 的 `permute/unpermute` 算子能否被 CUDA Graph 录制 → 需要验证（如果内部有 CPU 同步操作就不行）
- 如果 CUDA Graph 在 MoE 层失败，可以只 capture attention + FFN 部分，MoE 单独跑

---

## 五、模型导出：torch.jit.trace 的可行性

要用 C++ 跑推理，首先要把模型从 Python 导出。最直接的方式是 `torch.jit.trace`。

### 5.1 Trace Decode Step

VQA 的 decode step（单 token forward）比 prefill 简单得多，shape 全部固定：

```python
# Trace 的示例
example_input_ids = torch.zeros(1, 1, dtype=torch.long, device='cuda')
example_attention_mask = torch.ones(1, 420, dtype=torch.long, device='cuda')
example_position_ids = torch.tensor([[420]], dtype=torch.long, device='cuda')
example_past = prefill_outputs.past_key_values  # 从 prefill 拿到的 KV Cache

traced_decode = torch.jit.trace(
    model.forward,
    (example_input_ids, example_attention_mask, example_position_ids, example_past),
    strict=False
)
traced_decode.save("wall_x_decode_sm87.pt")
```

### 5.2 已知挑战

| 挑战 | 原因 | 应对方案 |
|------|------|----------|
| 自定义 CUDA ops | 需要注册到 TorchScript 命名空间 | `torch.ops.wallx_csrc.permute` 方式注册 |
| MoE 的 if/else 控制流 | trace 只能捕获一条执行路径 | 用 `torch.jit.script` 替代 trace 处理分支 |
| DynamicCache | 不是 Tensor，trace 不了 | 改成 tuple 形式的 `past_key_values` |
| HuggingFace 装饰器 | `@add_start_docstrings` 等干扰 trace | 直接 trace 内部的 `model.model.forward()` |

### 5.3 更简单的替代方案

如果 trace 太痛苦，还有一个更简单的路线：

**不导出模型，在 C++ 里用 Python 子进程做 prefill + 预处理，C++ 只做 decode loop。**

```
Python 进程：
  1. 加载模型
  2. 处理图片 + tokenize
  3. 跑 prefill，拿到 KV Cache
  4. 把 KV Cache 和 embeddings 通过共享内存/文件传给 C++

C++ 进程：
  1. 加载 traced decode model
  2. 从共享内存读取 KV Cache
  3. 跑 decode loop
  4. 输出 token ids
```

这样 Python 只跑一次 prefill（开销大但只跑一次），C++ 负责反复跑的 decode loop（消除了 N 次 Python dispatch）。

---

## 六、C++ 项目结构设计

```
wall-x/
  cpp_inference/
    CMakeLists.txt           # 编译配置
    vqa_inference.cpp         # C++ 推理主程序
    kv_cache.h               # StaticCache 管理
    tokenizer_wrapper.h      # tokenizer 简单包装（可选）
    
  scripts/
    export_decode_model.py   # 模型导出脚本
    prepare_inputs.py        # 预处理 + prefill，输出 tensor 文件
    vqa_cuda_graph.py        # CUDA Graph 版 Python benchmark（对照组）
```

**CMakeLists.txt**（在 Orin 上编译）：
```cmake
cmake_minimum_required(VERSION 3.18)
project(wallx_vqa_cpp)

# PyTorch libtorch
set(Torch_DIR "/data/wy/wall-x/venv/lib/python3.10/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(vqa_inference vqa_inference.cpp)
target_link_libraries(vqa_inference ${TORCH_LIBRARIES})
set_property(TARGET vqa_inference PROPERTY CXX_STANDARD 17)
```

---

## 七、执行计划

整个优化分 6 步，按顺序执行：

| 步骤 | 内容 | 预计时间 | 产出 |
|------|------|----------|------|
| **Step 1** | nsys profiling 量化 Python 开销 | 0.5 天 | Python overhead 占比数据 |
| **Step 2** | CUDA Graph 版 Python 推理 | 1 天 | CUDA Graph benchmark 脚本 |
| **Step 3** | torch.jit.trace 导出 decode model | 1 天 | .pt 模型文件 |
| **Step 4** | C++ decode loop + libtorch | 1-2 天 | 可运行的 C++ 推理二进制 |
| **Step 5** | Python 预处理 + C++ 推理桥接 | 0.5 天 | 完整 pipeline |
| **Step 6** | 全配置 benchmark 对比 | 0.5 天 | 文章数据 |

### 7.1 Step 1 是决策门

Step 1 的 profiling 结果决定后续所有步骤的优先级：

- **Python overhead < 10%**：C++ 替换收益有限，跳过 Step 3-5，直接做 Step 2（CUDA Graph）
- **Python overhead 10-20%**：做 Step 2（CUDA Graph），评估是否需要 Step 3-5
- **Python overhead > 20%**：全做，C++ 替换 + CUDA Graph 叠加

所以第一步不是"动手写 C++"，而是"跑 profiling 看数据"。**不拍脑袋，拿数据说话。**

### 7.2 Benchmark 对比矩阵

最终要跑的配置对比：

| 配置 | 说明 |
|------|------|
| Python baseline (SDPA) | 当前生产配置，纯 Python |
| Python + CUDA Graph | Python 预处理 + CUDA Graph decode |
| C++ decode loop | Python prefill + C++ decode |
| C++ decode + CUDA Graph | Python prefill + C++ CUDA Graph decode |

每个配置跑 3 次 warmup + 10 次计时，记录 mean / std / p95 / peak GPU memory。

---

## 八、落地思考：为什么不直接上最"重"的方案

很多做推理优化的同学有个倾向：直接上最"重"的方案（全模型 TensorRT 转换、全链路 C++ 重写、自定义 kernel fuse），觉得"一步到位"效率最高。

**在边缘端，这个思路通常是错的。** 原因有三：

### 8.1 调试成本被严重低估

在 Orin 上调试 TensorRT plugin 的体验：
- 编译一次 TensorRT plugin 要 5-10 分钟（ARM CPU 慢）
- 出了 shape mismatch 只有一行报错，没有 stack trace
- 没有 Python REPL 可以交互式检查中间 tensor
- 如果涉及 MoE 路由，需要对比 Python 版和 C++ 版的 token 排列顺序——一个 off-by-one 就能导致输出完全错误

相比之下，libtorch 的调试体验和 PyTorch Python 版几乎一样：可以 print tensor、可以用 gdb、可以逐行对比。

### 8.2 模型还在迭代

wall-x 的 MoE 路由逻辑（TokenTypeRouter）、Flow Action 的 ODE 积分步数、Action Head 的 MLP 结构——这些都在持续迭代中。如果把整个模型锁死在 TensorRT engine 里，每次改模型都要：

1. 修改 Python 模型代码
2. 重新导出 ONNX
3. 检查所有 plugin 是否兼容
4. 重新构建 TensorRT engine
5. 验证数值精度

libtorch + TorchScript 就没有这个问题——Python 侧改完，重新 trace 一下就行。

### 8.3 渐进式优化能更快出成果

```
Week 1: profiling + CUDA Graph   → 已经有数据可以写文章了
Week 2: C++ decode loop          → 进一步优化数据
Week 3: 如果还不够，再考虑 torch_tensorrt 混合编译

vs

Week 1-3: 写 TensorRT plugin    → 还在调试 shape inference
Week 4: 终于跑通                → 发现性能提升可能也就 20%
```

渐进式路线每一步都有产出，可以随时停下来写文章、评估 ROI。一步到位路线在中间任何地方卡住，都没有产出。

### 8.4 真正的性能瓶颈可能不在你想的地方

从第一篇的 profiling 数据来看：
- **GEMM/GEMV 占 54-69% GPU 时间**
- Python dispatch 开销**可能**占 10-30%（取决于 ARM CPU 速度）
- 即使 C++ 消除了全部 Python 开销，最大收益也就 ~30%

**真正能带来 2-3 倍加速的是量化**（把 bf16 GEMM 变成 INT8 GEMM），但量化是第四篇的事。这一篇的目标是：**把 Python 能省的先省掉，建立 C++ 推理框架，为后续量化做好基础设施。**

---

## 九、方案选型总结

| 方案 | 优势 | 劣势 | wall-x 适用性 |
|------|------|------|---------------|
| **torch.compile** | 零改造，自动优化 | Orin aarch64 不支持 | 不可用 |
| **纯 TensorRT** | 最大性能，自动 fusion | 6 个 plugin，2-3 周开发 | 成本过高 |
| **TRT-LLM** | LLM 专用优化 | wall-x MoE 不兼容 | 不现实 |
| **libtorch + CUDA Graph** | 自定义 ops 即插即用，调试友好 | 无自动 fusion | **推荐** |
| **torch_tensorrt 混合** | 标准层 TRT + 自定义层回退 | 需要验证兼容性 | Phase 3 备选 |

最终选择：**libtorch C++ decode loop + CUDA Graph**，原因：
1. 6 个自定义 CUDA 算子无需改造
2. 调试成本低，每步都有可验证的产出
3. CUDA Graph 消除 kernel launch 开销，拿到大部分 TensorRT "fusion" 效果
4. 为后续量化（第四篇）建立 C++ 推理基础设施
5. 如果不够，torch_tensorrt 混合编译是干净的升级路径

---

## 系列预告

本文是 **wall-x 机器人大模型部署系列** 的第三篇。

**第一篇**：[把 3B 大模型塞进机器人：RTX 5090 与 Jetson Orin 边缘端产品部署踩坑全记录](#)
- 环境搭建、CUDA 依赖链、profiling 数据、28 万+ kernel 分析

**第二篇**：[当算子逼近硬件极限：一次 Orin Profiling 引发的具身智能实时系统思考](#)
- FA2 编译全过程、FA2 vs SDPA benchmark、GEMM 带宽天花板证明、67% 框架空转发现

**第三篇（本文）**：用 C++ 替换 Python 推理
- VQA 热路径分析、CUDA vs TensorRT 方案选型、libtorch + CUDA Graph 实施计划

**第四篇（预告）**：INT8/INT4 量化——砍掉 54-69% 的 GEMM 主体延迟
- GEMM 占绝对主体，量化能砍多少
- PyTorch 内量化 vs TensorRT 量化
- 量化对 MoE 路由和 Flow Action 精度的影响
- 边缘端 VLA 模型的量化-精度 trade-off

**第五篇（预告）**：端侧 AI OS —— 从推理优化到系统架构
- 前四篇的结论汇聚：具身智能的瓶颈不在 model 而在 runtime
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
- **优化方向**：用 Triton 写 fused kernel（residual_add + rmsnorm、gate × up fusion、RoPE + attn score），等 C++ 改造消除框架开销后，再逐个验证实际收益

> 数据来源：`scripts/bench_triton_vs_pytorch_orin.py`，Triton 3.6.0，Orin 64GB 锁频。详细分析见第二篇附录。

---

*测试环境：wall-oss-flow 3B 模型，bfloat16 精度，batch_size=1。测试平台：Jetson AGX Orin 64GB，JetPack 6.2.1，CUDA 12.6，PyTorch 2.5.0a0+872d972e41.nv24.08。profiling 工具：Nsight Systems。2026 年 4 月。*
