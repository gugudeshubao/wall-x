# 当算子逼近硬件极限：一次 Orin Profiling 引发的具身智能实时系统思考

> 上一篇我们在 Orin 上部署 wall-x 时留了一个悬念：Flash Attention 2 编译不过，只能用 SDPA 替代。profiling 数据显示 Orin 上 SDPA 的 Softmax 是独立 kernel（因 HuggingFace 传入 `attention_mask` 导致 SDPA 退化为 math backend，比 5090 慢 57 倍），理论上 FA2 是解决这个问题的最直接路径。这一篇，我把 FA2 在 Orin 上编译通了，做了完整的 benchmark 和深度 profiling，一路挖到了 GPU 利用率只有 32.7% 的真正瓶颈。

**TL;DR**
- Flash Attention 2.8.3 **在 Orin 上编译成功了**，需要两个补丁：setup.py 加 SM 8.7 gencode + triton rotary 用纯 PyTorch 替换
- 编译耗时 ~50 分钟，73 个 .cu 文件，需要 SM 8.7 原生 SASS（不能用 SM 8.0 的 PTX 前向兼容）
- **FA2 比 SDPA 快 27%**（6360ms vs 8681ms），内存持平（8.14 vs 8.19 GB），方差只有 SDPA 的 1/4
- 与 5090 的差距从 8.0x 缩小到 5.9x（tok/s: 7.4 → 10.1，提升 36.5%）
- 但 27% 仍然低于 FA2 在 A100/H100 上的 2-3x 加速——深挖发现 FA2 的 tile size 是源码注释里写明 *"For sm86 or sm89"* 调优的，Orin 的 16 SM + 4MB L2 完全不在设计范围
- **NCU 实测验证**：decode L2 read hit 仅 11.5%，prefill 65%，ViT 96%——decode 低 L2 hit 是单次遍历无复用的固有特征，prefill 的 L2 miss 可通过缩小 tile size 改善
- TRT-LLM 自带 **180 个 SM 8.7 专用预编译 cubin**，和开源 flash-attn 根本不是一个东西
- Orin 上 Flash Attention 有**三条路线**：① 开源 FA2（已在用）② TRT-LLM cubin（可提取但 attention 仅占 1.4%，不划算）③ FlashInfer（非官方支持 SM 8.7，设 `FLASHINFER_CUDA_ARCH_LIST="8.7"` 可编译）
- **意外发现：绕过 HuggingFace `attention_mask` 后，cuDNN SDPA 延迟 0.076ms ≈ TRT-LLM 的 0.075ms**——C++ 框架改造后零成本获得
- wall-x 的 Flow Action（ODE 积分 + 动态 embedding）在 TRT-LLM 和 llama.cpp 里都跑不了
- **GEMM Profiling 新发现**：52.9% 的 GEMM 时间集中在 64x64 小矩阵 decode GEMV（batch=1 带宽瓶颈）。cuBLAS + CUTLASS 已是最优 bf16 实现，kernel 时间 ≈ 纯内存读取时间（实测带宽 168 GB/s，利用率 ~72%）——**不是算力瓶颈，是带宽瓶颈，只有量化能解**
- **torch.compile 仅快 1.7%**（自定义 CUDA op 导致 graph break）；**torch._int_mm 不支持 M=1**；**bitsandbytes 不支持 SM 8.7**
- **重大发现：GPU 利用率只有 32.7%，67% 的时间是 Python/HuggingFace 框架开销**
- 每个 decode step：97.2ms 总耗时，仅 29.7ms 在 GPU 计算，67.5ms 是框架"空气"
- 最终结论：**FA2 有效，继续用；最大的优化空间不是 GEMM 计算，而是消除 67% 的框架开销**
- **更深层的观点**：GEMM 已触达带宽天花板、框架空转占 67%、算子优化空间见顶——这三个发现共同指向一个结论：**具身智能的瓶颈不在 model，而在 runtime。** VLA 是有损压缩的物理模拟器，提高刷新率（1.5 Hz → 8 Hz）比提高单次精度更有价值。C++ 推理改造不只是性能优化，而是把具身智能从"跑模型"推向"实时系统"的第一步

---

## 一、先回顾问题：为什么上一篇不得不用 SDPA

上篇部署文章里，我们在 Orin 上直接用了 SDPA 做推理。当时的理由很简单：Flash Attention 编译不过。

但从 profiling 数据来看，这不只是"编译不过就算了"这么轻松。Orin 上 SDPA 的 Softmax 是一个独立的 `cunn_SoftMaxForward` kernel，耗时 746ms，占 GPU 时间的 4.7%，比 5090 上融合后的 13ms 慢了 **57 倍**。后来我们挖出了根因：HuggingFace 的 `generate()` 始终传入 `attention_mask`，而 cuDNN 和 flash_sdp 都不支持非空 mask，导致 SDPA 退化为最朴素的 math backend，softmax 被拆成独立 kernel。

Flash Attention 2 的核心优势之一，就是把 Q×K^T → Softmax → ×V 这三步融合进一个 kernel 里，不需要把中间的 attention score 矩阵写回 HBM，**且在 CUDA C++ 内部处理 causal mask，不依赖外部 `attention_mask` 参数**。理论上，如果 FA2 能在 Orin 上跑起来，这个 57 倍的 Softmax 差距应该直接消失。

所以这一篇的目标很明确：**把 FA2 编译通，跑个 benchmark，看看 Softmax 融合到底能省多少。**

---

## 二、第一个问题：为什么 FA2 在 Orin 上编译不过？

Flash Attention 2 的源码编译在 A100/RTX 3090 等主流 GPU 上通常一句话的事：

```bash
pip install flash-attn --no-build-isolation
```

但在 Jetson AGX Orin 上，这条路走不通。原因有三个层次。

### 2.1 没有预编译 wheel

PyPI 上的 flash-attn wheel 全是 x86_64 的，没有 aarch64 版本。Jetson 上只能源码编译。

### 2.2 SM 8.7 不在默认编译列表里

这是最关键的一个坑。

FA2 v2.8.3 的 `setup.py` 里，默认的 CUDA 架构列表是：

```python
os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120")
```

注意：**没有 87**。

Orin 的 GPU 是 Ampere 架构，SM 版本是 8.7。你可能会想："80 和 87 都是 Ampere，编译 SM 8.0 的 kernel 应该能在 8.7 上跑吧？"

答案是：**不能。** 至少在 FA2 v2.8.3 的编译方式下不能。

原因在于 FA2 v2.8.3 的 `add_cuda_gencodes()` 函数只生成 **SASS 代码**（`code=sm_80`），不生成 PTX 前向兼容代码（`code=compute_80`）。SASS 是针对特定 SM 版本的机器码，SM 8.0 的 SASS **不能**在 SM 8.7 上执行。

这和最新版 flash-attention main 分支不同。main 分支加了 PTX 生成（`code=compute_80`），PTX 可以被 GPU 驱动 JIT 编译成任何兼容架构的 SASS。但 v2.8.3 没有这个逻辑。

所以你用 `FLASH_ATTN_CUDA_ARCHS=80` 编译出来的 .so，load 的时候会直接报：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### 2.3 triton 不支持 aarch64

FA2 的 Python 层依赖 `flash_attn.ops.triton.rotary`，里面用了 triton 的 JIT kernel 做 rotary position embedding。

而 triton 目前根本不支持 aarch64 / ARM 架构。直接 import 就会报错：

```
ModuleNotFoundError: No module named 'triton'
```

这不是一个可以 `pip install` 解决的问题——triton 的编译器后端就没有 ARM 支持。

---

## 三、两个补丁，编译通过

知道了原因，修复起来就有方向了。

### 补丁一：setup.py 加 SM 8.7 gencode

在 `add_cuda_gencodes()` 函数里，SM 8.0 的 gencode 后面加上 SM 8.7：

```python
if "80" in cuda_archs():
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
# Orin (Jetson AGX Orin) SM 8.7
if "87" in cuda_archs():
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_87,code=sm_87")
```

同时把默认架构列表改成 `"80;87;90;100;120"`。

这样编译时每个 .cu 文件会同时生成 SM 8.0 和 SM 8.7 两套 SASS。编译时间大约翻倍，但保证了在 Orin 上有原生机器码可以跑。

### 补丁二：triton rotary 用纯 PyTorch 替换

把 `flash_attn/ops/triton/rotary.py` 改成自动检测 triton 是否可用：

```python
_HAS_TRITON = False
try:
    import triton
    _HAS_TRITON = True
except ImportError:
    pass

# 有 triton 就用 triton kernel，没有就用纯 PyTorch 实现
apply_rotary = _apply_rotary_triton if _HAS_TRITON else _apply_rotary_torch
```

PyTorch 版本的 `apply_rotary` 功能完全等价，只是没有 triton kernel 的性能优化。但 rotary embedding 本身不是推理瓶颈，这个替换对整体延迟的影响可以忽略。

### 编译命令

两个补丁打完，在 Orin 上跑：

```bash
FLASH_ATTN_CUDA_ARCHS='87' \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=6 NVCC_THREADS=2 \
python setup.py build_ext --inplace
```

73 个 .cu 文件，每个文件生成两套 gencode。Orin 的 ARM CPU 跑 nvcc 不快，全程大约 **50 分钟**。

最终生成：`flash_attn_2_cuda.cpython-310-aarch64-linux-gnu.so`

```python
>>> import flash_attn
>>> flash_attn.__version__
'2.8.3'
>>> from flash_attn import flash_attn_func
>>> print("OK")
OK
```

**编译成功。**

---

## 四、GPU 功能验证：FA2 在 Orin 上真的能跑了

编译通过不等于能用。还得验证 GPU 上 kernel 能不能正确执行。

```python
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 256, 8, 128, dtype=torch.float16, device='cuda')
k = torch.randn(1, 256, 8, 128, dtype=torch.float16, device='cuda')
v = torch.randn(1, 256, 8, 128, dtype=torch.float16, device='cuda')

out = flash_attn_func(q, k, v)
torch.cuda.synchronize()
print(f"Output shape: {out.shape}")  # [1, 256, 8, 128]
```

Batch=1, seqlen=256, 8 heads, headdim=128, fp16。

结果：**FA2 latency 0.67ms，输出正确。** SM 8.7 的原生 SASS 确实可以执行。

到这里，上一篇的悬念算是正式解开了：**FA2 在 Orin 上是可以编译和运行的**，只是需要手动加 SM 8.7 gencode 和 triton fallback。

---

## 五、VQA Benchmark：期望 vs 现实

功能验证通过，接下来是正题——把 FA2 插回 wall-x 推理流程，和 SDPA 做正式对比。

### 测试方法

写了一个 benchmark 脚本 `bench_fa2_vs_sdpa.py`，做法是：
1. 加载 wall-x 模型（`wall-oss-flow` 3B），指定 `attn_implementation = "sdpa"` 或 `"flash_attention_2"`
2. 用同一张 640×480 测试图片 + VQA（Visual Question Answering，视觉问答：给模型一张图片和一个文字问题，模型生成文字回答） prompt，生成 64 个 token（`--max_new_tokens 64`，比第一篇的 128 更短，更贴近机器人场景下的短回答长度；这是脚本参数，不是模型限制）
3. 3 次 warmup + 10 次正式计时
4. **系统空闲**（load avg < 5）、`jetson_clocks` 锁定 GPU 频率到最大值 1300.5 MHz

关键细节：模型加载时通过直接设置 `model_config._attn_implementation` 来切换 attention 后端，确保除了 attention 实现之外，其他所有条件完全一致。

### 结果：FA2 比 SDPA 快 27%（VQA 文本生成）

| 指标 | SDPA | Flash Attention 2 | 差异 |
|------|------|-------------------|------|
| **平均延迟** | 8681 ms (std 109) | **6360 ms** (std 25) | **FA2 快 27%** |
| 中位延迟 | 8631 ms | **6354 ms** | -26.4% |
| Min / Max | 8617 / 8953 ms | **6335 / 6409 ms** | |
| **吞吐量** | 7.4 tok/s | **10.1 tok/s** | +36.5% |
| 峰值 GPU 内存 | 8.19 GB | **8.14 GB** | 持平 |
| **延迟标准差** | 109 ms | **25 ms** | FA2 4x 更稳定 |

FA2 在每一个维度上都优于 SDPA：延迟低 27%、吞吐量高 36.5%、内存持平、方差小 4 倍。

> **Orin 上怎么看 GPU 内存？** Jetson 是统一内存架构，`nvidia-smi` 不可用。推荐用 [`jtop`](https://github.com/rbonghi/jetson_stats)（`pip install jetson-stats`，然后终端运行 `jtop`），可以实时看到 GPU/CPU 占用率、内存分配、功耗、温度等信息。上表的"峰值 GPU 内存"通过 `torch.cuda.max_memory_allocated()` 采集，`jtop` 则适合在推理过程中做实时监控和截图。

### 跨平台 VQA 对比：FA2 让 Orin 追近了 5090

上一篇文章中，我们在 **5090 用 FA2、Orin 用 SDPA** 做了 VQA 对比。现在 Orin 也有了 FA2，可以做一个完整的三路对比：

| 平台 | Attention | VQA 延迟 | tok/s | 峰值显存 | vs 5090 差距 |
|------|-----------|----------|-------|---------|-------------|
| **RTX 5090** | FA2 2.8.3 | **1,747 ms**<sup>①</sup> | **59.8** | 8.14 GB | 1.0x（基准） |
| **Orin + SDPA** | cuDNN SDPA | 8,681 ms<sup>②</sup> | 7.4 | 8.19 GB | ~8.1x |
| **Orin + FA2** | FA2 2.8.3 | **6,360 ms**<sup>②</sup> | **10.1** | 8.14 GB | ~5.9x |

<small>① 128 tokens，8 图平均（第一篇数据）。② 64 tokens，单图（本篇数据）。两平台 token 数不同，延迟不可直接比较，"vs 5090 差距"按 tok/s 计算（tok/s 与生成长度无关，是更准确的吞吐度量）。峰值显存统一按单图 64 tokens 条件采集（`torch.cuda.max_memory_allocated()`），三个平台一致。</small>

> **为什么第一篇报 15.7 GB，本篇只有 8.1 GB？**
>
> 差异来自模型加载方式，不是模型本身变了。wall-x 3B 模型共 ~3.1B 参数：
>
> ```
> 模型权重内存 = 参数量 × 每参数字节数
>   fp32:  3.1B × 4 bytes = ~12.4 GB
>   bf16:  3.1B × 2 bytes = ~6.2 GB
> ```
>
> 第一篇 `test_vqa_bench.py` 的写法：
> ```python
> model = from_pretrained(model_path)       # ① 默认 fp32 加载到 CPU
> model.eval().to("cuda").bfloat16()        # ② .to("cuda") 先把 fp32 搬上 GPU → 峰值 ~12.4 GB
>                                            # ③ .bfloat16() 再原地转 bf16 → 降为 ~6.2 GB
>                                            # 但 max_memory_allocated 已记录 ② 的 12.4 GB 峰值
> ```
>
> 本篇 `bench_fa2_vs_sdpa.py` 的写法：
> ```python
> model.eval().to(device, dtype=torch.bfloat16)  # 一步完成：fp32→bf16 转换 + CPU→GPU 搬运
>                                                  # GPU 上从头到尾只有 bf16 → 峰值 ~6.2 GB
> ```
>
> | | 第一篇 | 本篇（第二篇） |
> |---|---|---|
> | GPU 上权重峰值 | ~12.4 GB（fp32 先上 GPU） | ~6.2 GB（bf16 直接上 GPU） |
> | + 推理激活值 | ~3.3 GB | ~1.9 GB |
> | **`max_memory_allocated`** | **~15.7 GB** | **~8.1 GB** |
>
> **教训**：PyTorch 中 `.to("cuda").bfloat16()` 和 `.to("cuda", dtype=torch.bfloat16)` 的内存峰值差 2×。部署时永远用后者。

关键数字：

1. **Orin + FA2 的吞吐量从 7.4 提升到 10.1 tok/s**，提升 36.5%——这在机器人 VQA 场景下直接从"勉强够用"变成"明显更快"
2. **与 5090 的差距从 8.0x 缩小到 5.9x**（按 tok/s 计算：59.8/10.1 = 5.9）。5090 有 8.7x 的带宽优势和 12x 的算力优势，FA2 编译这一步让 Orin 吃回了一部分差距
3. **延迟标准差从 109ms（SDPA）降到 25ms（FA2）**——FA2 的延迟稳定性好 4 倍，对控制回路更友好

更直观的理解：**Orin 用 SDPA 时每秒只能回答 7.4 个 token 的问题，FA2 之后涨到 10.1 个。** 对于 128 token 的典型 VQA 回答，这意味着从 17.3 秒降到 12.7 秒——省了将近 5 秒等待时间。

#### 补充：5090 为什么也是 FA2 而不是 FA3/FA4？

细心的读者可能注意到：5090 用的也是 FA2 v2.8.3，而不是更新的 FA3 或 FA4。这不是偷懒——**FA3/FA4 根本不支持 RTX 5090**。

Flash Attention 目前有三个版本，各自锁定了不同的硬件：

| 版本 | 代码路径 | 目标 SM | 硬件特性 | 5090 能用？ |
|------|---------|---------|---------|-----------|
| **FA2** v2.8.3 | `csrc/flash_attn/` | SM 8.0+ 通用，**含 SM 12.0** | 标准 CUDA（WMMA/cuBLAS） | **能，正在用** |
| FA3 | `hopper/` `_sm90.cu` | SM 9.0 | Hopper 专有 TMA + WGMMA | 不能 |
| FA4 beta | `hopper/` `_sm100.cu` | SM 10.0 | Blackwell 数据中心 MMA | 暂不能<sup>*</sup> |

<small>* FA4 的 SM 10.0（B100/B200 数据中心 Blackwell）和 5090 的 SM 12.0（消费级 Blackwell）属于同一架构家族，MMA 指令集兼容。但 FA4 仅有一个 `flash_fwd_hdim128_bf16_sm100.cu` 实例化文件，没有为 SM 12.0 做实例化和编译支持。</small>

FA2 v2.8.3 的 `setup.py` 默认 arch 列表已包含 SM 12.0（`FLASH_ATTN_CUDA_ARCHS = "80;90;100;120"`），在 CUDA >= 12.8 时会自动生成 `compute_120,code=sm_120` 的原生 SASS。5090 的 CUDA 13.1 完全满足，编译无需任何补丁。

但在运行时，5090（SM 12.0）走的是 FA2 v2 代码路径的**默认分支**（`is_sm8x = cc_major == 8 && cc_minor > 0` 对 SM 12.0 为 false），tile size 和 SM 8.0（A100）相同（128×64）。这意味着 **5090 没有用到 Blackwell 的新硬件能力**——TMEM（Tensor Memory）、TMA（Tensor Memory Accelerator）、TCGEN05（第五代 Tensor Core 指令）等特性全部没被利用。

换句话说，Orin 和 5090 跑的其实是**同一套 FA2 通用 kernel 代码**，只是编译时生成了各自 SM 的原生 SASS。5090 比 Orin 快 6x 纯靠硬件规格碾压（170 SM vs 16 SM，1.2 TB/s vs 204 GB/s），而不是 kernel 更优。**如果未来 FA4 加上 SM 12.0 支持，5090 的 attention 性能还有提升空间**——到时候 Orin 和 5090 的差距反而可能进一步拉大。

这也回答了"FA2 在 Orin 上有没有价值"这个问题：**绝对有。** 哪怕 FA2 的 tile size 是为 SM 8.6/8.9（A6000/RTX 4090 级别）调的、哪怕 v2 代码路径完全没有 L2 cache 感知——仅凭 Softmax 融合本身的算法优势，就实打实地把 Orin 的推理速度提升了 27%，把与 5090 的差距从 8x 收窄到 6x。

---

## 六、为什么 FA2 只快了 27%？——本应更快的三个理由

FA2 确实快了，但 27% 的提速远低于 FA2 在 A100/H100 上通常带来的 2-3x 加速。深挖源码发现，FA2 在 Orin 上有三层"减速带"，如果修复，还有优化空间。

### 6.1 序列太短，FA2 的 tiling 优势打了折扣

wall-x VQA 推理的输入序列长度大约 420 个 token（包含 ~350 个视觉 token + ~70 个文本 token）。

FA2 的核心优势是 **tiling**：把长序列切成小块，逐块在 SRAM 里完成 QK^T → softmax → ×V 的完整计算，避免把 N×N 的 attention score 矩阵写回全局内存。

但 420 个 token 的 attention score 矩阵只有 420×420 = 176,400 个元素（fp16 下 ~344KB），完全放得进 Orin GPU 的 L2 cache。SDPA 在这个规模下不需要 tiling 就能高效计算。而 FA2 的 tiling 机制在短序列下的收益被分块、同步和 reduce 的固定开销部分抵消。

**FA2 的设计甜区是 >2K token 的长序列。** 在 420 token 下仍然能快 27%，说明 Softmax 融合带来的收益在短序列下也是正的——只是没那么大。

### 6.2 FA2 的 tile size 是为 SM 8.6/8.9 调的——Orin 的 16 SMs + 4MB L2 完全不在设计范围

虽然我们让 nvcc 为 SM 8.7 生成了原生 SASS，但 FA2 v2.8.3 的 **tile size 选择**并不是为 Orin 这种边缘 GPU 设计的。

SM 8.7 走的是 v2 代码路径（`csrc/flash_attn/src/flash_fwd_launch_template.h`），通过 compute capability 分流：

```cpp
bool is_sm8x = cc_major == 8 && cc_minor > 0;
```

SM 8.7 的 `cc_minor=7 > 0`，所以 `is_sm8x=true`，拿到的 tile size 和 **SM 8.6（A6000/RTX 3090）/ SM 8.9（RTX 4090/L40）完全相同**。源码注释写明 *"For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square), and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM"*——tile size 是在这两个 SM 上 benchmark 得出的最优配置：

| head_dim | 模式 | SM 8.0 (A100) tile | **SM 8.7 (Orin)** = SM 8.6/8.9 tile |
|----------|------|-------------------|------|
| 128 | causal | 128×64 | **64×64**（方阵，适配 sm8x 的 SMEM 限制） |
| 128 | non-causal | 128×64 | **128×32**（48KB SMEM，可 2 CTA/SM） |
| 96 (ViT) | non-causal | 128×64 | 128×64（相同） |

tile size 本身并不"错"——它是 Tri Dao 在 SM 8.6（A6000/RTX 3090）和 SM 8.9（RTX 4090/L40）上 benchmark 出来的最快配置。但问题是 **Orin 和这些 GPU 的硬件规格差距太大**：

| 参数 | A6000 (SM 8.6) | RTX 3090 (SM 8.6) | RTX 4090 (SM 8.9) | **Orin (SM 8.7)** |
|------|------|------|------|------|
| SM 数量 | 84 | 82 | 128 | **16** |
| L2 Cache | 6 MB | 6 MB | 72 MB | **4 MB** |
| 内存带宽 | 768 GB/s | 936 GB/s | 1008 GB/s | **204 GB/s** |

128×32 的 tile 在 128 个 SM 上可以做到很高的 occupancy 和 L2 复用。但在 16 个 SM 上，并行度不够，L2 复用率大幅下降。

另一个关键点：v2 代码路径（SM 8.x 用的）**没有任何 L2 cache 感知逻辑**。hopper 路径（SM 9.0+）里有 L2 大小硬编码（50MB/32MB）来做 split 和 swizzle 优化，但 SM 8.7 完全不走那条路。v2 的 `num_splits_heuristic()` 纯做 occupancy 优化，不考虑 L2 容量。

**NCU 实测验证了这个差距。** 我们用 `ncu --kernel-name 'regex:flash_fwd'` 抓了 FA2 kernel 的硬件 counter（GPU 锁频 1300.5 MHz，系统空闲）：

| Kernel 类型 | Grid | Duration | **L2 read hit** | SM throughput | Occupancy |
|---|---|---|---|---|---|
| flash_fwd_kernel（prefill, head_dim=96）| (1, 30, 16) | ~130 us | **64.9%** | 28% | 16.5% |
| flash_fwd_kernel（ViT, head_dim=96）| (13, 1, 16) | ~629 us | **95.9%** | 64.2% | 16.9% |
| flash_fwd_splitkv_kernel（decode）| (1, 4, 2) | ~13 us | **11.5%** | 7.3% | 10% |

几个关键数字：

1. **Decode splitkv 的 L2 read hit 只有 11.5%** — 但这不是 tile size 的问题。Decode 的 split 已经到了上限：420 tokens / block_n=128 = 4 splits，每个 split 只访问 ~48KB KV 数据，远小于 4MB L2。**低 L2 hit 是 decode attention 单次遍历没有 temporal reuse 的固有特征** —— 每个 KV block 只被读一次就不再访问，L2 cache line 刚加载就被驱逐。
2. **Prefill L2 hit 65%** — 多 head 共享 KV 数据带来了一定的 temporal reuse，但 30 个 KV blocks 的 working set 超出 4MB L2 容量，35% miss 率来自 L2 容量不足。
3. **ViT L2 hit 96%** — 短序列数据能装进 4MB L2，此时 FA2 效率最高（SM 利用率 64%）。
4. **Decode occupancy 只有 10%** — Grid (1,4,2)=8 blocks 分配给 16 个 SM，一半 SM 完全空闲。

ViT 和 decode 的 L2 hit 差异（96% vs 11.5%）直观地展示了两种不同的瓶颈：**ViT 的高 L2 hit 来自 multi-head 的数据复用；decode 的低 L2 hit 来自单次遍历无复用。** 前者受 tile size 和 L2 容量影响，后者是 decode attention 的固有特征，无法通过调参解决。

总结一下：**FA2 在 Orin 上的主要问题不是"把 SM 8.7 当 SM 8.0"，而是 v2 代码路径的 tile size 是在 SM 8.6/8.9（82-128 SM 的 GPU）上 benchmark 调优的，Orin 的 16 SM + 4MB L2 + 204GB/s 带宽组合完全不在设计范围内。** NCU 数据表明，prefill 的 L2 miss 可以通过缩小 tile size 改善，但 decode 的 L2 miss 是固有特征。即便如此，Softmax 融合本身的算法优势还是让 FA2 快了 27%。

### 6.3 tensor 不连续——不影响内存，但有额外拷贝开销

FA2 模式下，日志里出现了 SDPA 模式没有的警告：

```
UserWarning: The input `q` of multimodal_rope op is discontiguous!
UserWarning: The input `k` of multimodal_rope op is discontiguous!
```

FA2 要求输入 tensor 是连续的（contiguous），但 wall-x 的 multimodal RoPE 操作会产生不连续的 Q/K tensor。这意味着 FA2 路径在 attention 前需要额外的 `.contiguous()` 调用，创建新的连续内存副本。

SDPA 对内存布局的要求更宽松，可以直接处理 stride 不连续的 tensor，省掉了这部分拷贝开销。

不过，从清洁环境的数据来看，内存峰值完全一样（8.14 vs 8.19 GB）。说明 `.contiguous()` 创建的是临时副本，用完即释放，不会推高峰值内存。但这部分拷贝的耗时仍然是 FA2 的一个额外开销，吃掉了一些 Softmax 融合带来的收益。

---

## 七、这说明什么？

### FA2 在 Orin 上有效，但没发挥全部潜力

FA2 在 A100/H100 上通常能带来 2-3x 的加速。在 Orin + 短序列这个组合下，只快了 27%。原因清楚：

- 短序列（420 token）tiling 收益打折
- tile size 按 SM 8.6/8.9（82-128 SM）调优，Orin 16 SM + 4MB L2 不在设计范围 — NCU 实测 decode L2 hit 仅 11.5%
- 统一内存下 `.contiguous()` 拷贝代价更高

但 27% 仍然是实打实的提速——每次推理省 2.3 秒（8681 → 6360ms），对机器人控制来说这很可观。

### FA2 优化空间分析

从 NCU 数据看，FA2 kernel 在 Orin 上的优化空间有限：

- **Decode（splitkv）**：L2 hit 11.5%，但这是 decode 单次遍历的固有特征，无法通过调参解决。Split 已到上限（4/4），每 split 仅 48KB，远小于 4MB L2。
- **Prefill**：L2 hit 65%，可以通过缩小 tile size（当前 Br=128, Bd=64）来降低每 tile 的 working set，提升 L2 hit。但需要修改 kernel 模板。
- **ViT**：L2 hit 96%，已经很好，不需要优化。

更重要的是，nsys 数据显示 **FA2 attention kernel 只占 GPU 总时间的 1.4%（30.4ms / 2101ms）**。GEMM 占 78.8%，elementwise/copy 占 21%。即使 FA2 L2 hit 率从 65% 提升到 96%（ViT 级别），对整体推理的收益也只有 ~0.5%。优化重心应该在 GEMM 量化和 C++ 框架上。

### GEMM 细分与优化实验

为了搞清楚 GEMM 该怎么优化，我们对 nsys 数据做了细粒度分析。1643ms 的 GEMM 时间（52280 个 kernel 中的 6264 次 GEMM 调用）拆解如下：

| 排名 | Kernel | 调用数 | 时间(ms) | 占 GEMM | 特征 |
|---|---|---|---|---|---|
| 1 | bf16 64x64 sliced | 3240 | 869 | **52.9%** | Decode 阶段 batch=1 小矩阵 GEMV |
| 2 | bf16 128x128 | 288 | 204 | 12.4% | Prefill/ViT 大矩阵 |
| 3 | GEMV (gemv2T) | 30 | 112 | 6.8% | LM head 词表投影，每次 3.7ms |
| 4 | cutlass 256x128 | 128 | 98 | 5.9% | 自定义 op（dual_asym_gemm）|
| 5 | bf16 128x64 | 2160 | 119 | 7.2% | Decode 小 GEMM |
| 6 | cutlass 128x256 | 2 | 68 | 4.1% | ViT patch embed，33.8ms/call |

### GEMM 为什么慢？—— 是带宽瓶颈，不是算力瓶颈

nsys kernel 名称揭示了一个关键事实：**wall-x 的 GEMM 已经在用 cuBLAS 和 CUTLASS——都是 NVIDIA 最快的矩阵乘实现。**

按库分类的矩阵运算 kernel 时间（含 GEMM 及相关辅助 kernel，如 MoE permute/unpermute、splitK reduce 等）：

| 库 | 调用数 | 时间(ms) | 占比 | 来源 |
|---|---|---|---|---|
| **cuBLAS** | 5972 | **1225.7** | **72.4%** | PyTorch `nn.Linear` → cuBLAS ampere_bf16_s16816gemm |
| **CUTLASS** | 4858 | 312.5 | 18.5% | wall-x 自定义 CUDA op（dual_asym_gemm, MoE permute 等） |
| 其他 | 4384 | 155.3 | 9.2% | gemv2T、cublasLt splitK reduce、gemmk1 |

cuBLAS 的 `ampere_bf16_s16816gemm` 是 Ampere 架构上 bf16 Tensor Core GEMM 的**最优实现**——使用 s16816 WMMA 指令、多 stage 双缓冲、ldg8 合并访存。你换不到更快的 bf16 GEMM 库了。

**那为什么还这么慢？因为 Orin 的 GEMM 瓶颈是内存带宽，不是 Tensor Core 算力。**

我们实测了 Orin GPU 的有效内存带宽：

```
Buffer  64MB:  178.9 GB/s
Buffer 256MB:  168.9 GB/s
Buffer   1GB:  168.1 GB/s
```

Orin LPDDR5 理论带宽 204 GB/s，GPU 实际可用约 **168 GB/s**（CPU/GPU 共享总线，GPU 拿到 ~82%）。wall-x 3B 模型 bf16 权重约 **6 GB**。每个 decode step（batch=1）需要把全部权重读一遍（M=1 的 GEMV 没有权重复用），理论最快：

```
6 GB ÷ 168 GB/s = 35.7 ms/step（带宽极限）
实际 GEMM 时间 ≈ 25.7 ms/step（MoE 稀疏路由只激活 2/4 专家，实际读权重 ~4.3 GB）
```

换句话说，**GEMM 已经跑在 Orin 带宽极限的 ~72% 利用率了**。cuBLAS 在这个 workload 上的效率已经很高——问题不在软件层面，而在硬件带宽。

### decode GEMV 的本质问题

decode 阶段的 `nn.Linear`（M=1, N=2048-5632, K=2048）本质是 **matrix-vector multiply（GEMV）**，不是 GEMM。cuBLAS 对 M=1 的处理方式是使用最小的 64×64 tile（`ampere_bf16_s16816gemm_bf16_64x64_sliced1x2`），每个 tile 只有第一行是有效计算，其余 63 行浪费。但这不是低效——**64x64 tile 是 cuBLAS 为了保证 memory coalescing 的最小单位**，真正的瓶颈是每个 tile 都要等内存把权重数据搬进来。

top-1 的 64x64 sliced GEMM：
- Grid (172,1,1)：172 个 block，每 block 处理 64 列权重 → N ≈ 172×64 ≈ 11008 → 这是 MLP 的 gate/up projection（hidden=2048 → ffn=5504×2=11008）
- 每次 0.27ms × 2160 次 decode 调用 = 564.5ms，**占 GEMM 总时间的 33%**

这就是带宽瓶颈的直观表现：一个 2048×11008 的 bf16 权重矩阵 = 43MB，读一次 43MB / 168 GB/s = 0.26ms，和实测 0.27ms 几乎完全吻合。**cuBLAS 已经把带宽榨干了，kernel 时间 ≈ 纯内存读取时间。**

### 真正能加速 GEMM 的方案：量化

既然瓶颈是带宽，优化路径就很清晰——**减少需要读的数据量**：

| 方案 | 原理 | 权重大小 | 理论加速 | Orin 可行性 |
|------|------|---------|---------|------------|
| **bf16（当前）** | cuBLAS s16816 GEMM | 6 GB | 1.0x（基准） | 已在用 |
| **INT8 权重量化** | cublasLt INT8 GEMM，带宽减半 | 3 GB | **~2x** | **可行：cublasLt 支持 M=1 + INT8** |
| **INT4 权重量化（AWQ/GPTQ）** | 4-bit 权重 + fp16 计算，带宽降 4x | 1.5 GB | **~3-4x** | 需要自定义 kernel（Marlin/GPTQ-Marlin） |
| **FP8（E4M3）** | Tensor Core FP8 指令 | 3 GB | ~2x | **SM 8.7 不支持 FP8 Tensor Core**（需 SM 8.9+） |

**关键洞察：`torch._int_mm` 不支持 M=1，但 `cublasLt` 原生支持。** 这意味着 INT8 量化必须在 C++ 层面通过 cublasLt API 实现，绕过 PyTorch 的限制。这也是为什么我们的优化路线是"先做 C++ 框架，再做量化"——量化的正确实现依赖 C++ 框架。

另一个重要方向是 **Marlin-style GEMV kernel**（来自 IST Austria / vLLM 社区）。Marlin 专门为 decode 的 M=1 GEMV 设计：把 INT4 反量化和 GEMV 交织在同一个 kernel 里，让 Tensor Core 在等内存的空隙做反量化计算，接近带宽理论极限。SGLang 和 vLLM 已经在生产环境用了这个。

基于这个分析，我们在 Orin 上逐一测试了三条 GEMM 优化路径：

**1. torch.compile（reduce-overhead 模式）**

```python
model = torch.compile(model, mode="reduce-overhead")
```

结果：6283ms vs baseline 6391ms，**仅快 1.7%**。

原因：wall-x 有 6 个自定义 CUDA op（`multimodal_rope`、`permute_topK`、`dual_asym_gemm` 等），每个都是 graph break 点。torch.compile 能融合的只有少量 elementwise ops，对 GEMM 本身毫无帮助。

**2. torch._int_mm（原生 INT8 矩阵乘）**

```python
# INT8 Tensor Core 吞吐是 bf16 的 2x
out_int32 = torch._int_mm(x_int8, w_int8.t())
```

结果：**直接失败**。`torch._int_mm` 要求 `M > 16`，但 decode 阶段 M=1。这个 API 是给 prefill/training 大矩阵用的，decode GEMV 根本不支持。

**3. bitsandbytes INT8（Linear8bitLt）**

```python
import bitsandbytes as bnb
bnb.nn.Linear8bitLt(in_features, out_features, has_fp16_weights=False)
```

结果：**崩溃**。`Error named symbol not found at line 399 in /src/csrc/ops.cu`。bitsandbytes 0.49.2 的预编译 CUDA binary 没有 SM 8.7 kernel，在 Orin 上直接挂。

**三条路全部撞墙。** 结论：**在 Orin 上，Python 层面的 GEMM 优化没有可用方案。** torch.compile 被自定义 op 打断，INT8 MM 不支持 decode 的 M=1，bitsandbytes 不支持 SM 8.7。

### 67% 的时间不是在计算——框架开销才是真正的大头

三条 GEMM 优化路都走不通后，我们退一步看全局数据：

```
generate(N) = 216.2 + 97.2 * N  （线性回归）

  Prefill + 框架初始化：  216.2 ms   (3.4%)
  Decode (64 * 97.2ms):  6217.8 ms  (96.6%)
  总计:                   6434.0 ms
```

> ⚠️ **216.2ms 的来源**：这不是直接测量值，而是 `profile_overhead.py` 在 N=[1, 4, 16, 64] 四个生成长度上运行 `model.generate()`（每个跑 3 次取平均）后，用 `numpy.polyfit` 拟合的线性回归截距。截距代表 N=0 时的外推值，即 ViT 编码 + Prefill + 框架初始化的总开销。216.2ms 看起来很小，但和 nsys 数据一致：nsys 显示全推理 GPU kernel 总时间 2101ms，其中 decode 约 1900ms，因此 ViT + Prefill 的 GPU kernel 时间约 201ms。ViT 和 Prefill 阶段是大批量矩阵运算（256+ tokens 过 36 层 Transformer），GPU 利用率接近 100%，所以 wall clock（216.2ms）≈ GPU kernel 时间（201ms）+ 很小的 CPU 开销。这和 decode 阶段形成鲜明对比——decode 每步 97.2ms wall clock 但 GPU 只算 29.7ms，利用率仅 30.6%。

每个 decode step 97.2ms，但 nsys 数据显示 64 步 decode 的 GPU kernel 总时间约 1900ms → **每步 GPU 只计算 29.7ms**：

```
  Per decode step:
    Wall clock:       97.2 ms  (100%)
    GPU kernel:       29.7 ms  (30.6%)  ← 真正在计算
    框架开销:          67.5 ms  (69.4%)  ← Python + HuggingFace 在"空转"
```

**GPU 利用率只有 32.7%。67.3% 的时间 GPU 在等 Python。**

这 67.5ms/step 的开销来自：
- HuggingFace `generate()` 循环的 Python dispatch（LogitsProcessor、StoppingCriteria、输入验证等）
- 52280 次 kernel launch 的调度延迟（测量值 20.7us/launch）
- 动态 KV cache 内存管理
- Python GIL 和垃圾回收

64 步 decode 总共浪费了 **~4320ms 在框架开销上**（67.5ms/step × 64 steps = 4320ms） —— 这比 GEMM 全部时间（1643ms）都多。换句话说，**优化 GEMM 计算本身最多省 1643ms；消除框架开销可以省 4320ms。**

### FA2 未来的优化空间

如果有人为 Orin 做以下调优，FA2 的提速可能从 27% 涨到 50% 甚至更多：

1. 在 `flash_fwd_launch_template.h` 里为 SM 8.7 单独加分支，用更小的 tile size（如 64×16）适配 16 SM + 4MB L2
2. 给 v2 的 `num_splits_heuristic()` 加入 L2 容量感知（目前完全不考虑 L2）
3. 减少 warp 数（16 个 SM 不需要 82-128 SM GPU 的高并行策略）

但这需要改 FA2 的 CUDA C++ 源码 + 重新编译 + 大量 benchmark 验证，门槛较高。

**另一条路线：用 Triton 在 Orin 上写自定义 kernel。** 我们实测验证了 Triton 3.6.0 在 Orin SM 8.7 上完全可用——不仅基础 kernel 能跑，flash-attn 自带的 Triton rotary kernel 也正常工作（0.28 ms/call）。之前 wall-x 的 Triton rotary 回退到 PyTorch 纯 fallback，不是因为 Triton 不支持 Orin，而是 venv 里没装 Triton。

Triton 的价值在于**大幅降低 Orin 自定义 kernel 的开发门槛**：

- **Triton 写 attention kernel 比 CUDA C++ 容易一个数量级**。用 Triton 可以快速原型化不同的 tile size、split 策略，找到适合 16 SM + 4MB L2 的最优配置
- **fused kernel 机会**：当前 wall-x 的 RoPE + attention 是分开的两个 kernel，中间有一次 `.contiguous()` 拷贝。用 Triton 可以把 rotary embedding 融进 attention kernel，省掉这次拷贝
- **decode 专用 kernel**：当前 decode 用的 splitkv_kernel occupancy 只有 10%（8 blocks / 16 SMs），用 Triton 可以写一个专门针对 batch=1、seq=420、16 SM 的 decode kernel
- Triton 的 auto-tuning 框架可以自动搜索 `BLOCK_M`、`BLOCK_N`、`num_warps` 等参数，不需要手动猜

当然，投入产出比仍然需要权衡——attention 只占推理总时间的一部分，更大的优化空间在 GEMM 和 Python 框架层。但 Triton 路线的存在，意味着 **Orin 上的 kernel 优化不一定要改 FA2 源码，可以绕开 CUDA C++ 直接用 Python 写高性能 kernel。** 我们计划在下一篇 C++ 推理框架改造中，消除 67% 框架开销后，回到算子优化层面进一步探索 Triton 自定义 kernel 的实际收益。

---

## 八、插一个对比：TRT-LLM 的 Flash Attention 和开源 flash-attn 根本不是同一个东西

在研究 FA2 为什么慢的过程中，我们还发现了一个重要的事实：**TensorRT-LLM 自带的 Flash Attention 和我们编译的开源 flash-attn 完全不是同一套 kernel。**

当时 Orin 上同时在编译 TensorRT-LLM，我翻了一下它的源码，发现了这个目录：

```
TensorRT-LLM/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/
```

里面有 **1104 个** cubin 文件，其中 **180 个是 SM 8.7 专用的**。

这些 cubin 不是在用户机器上现场编译的，而是 **NVIDIA 内部预编译好的 GPU 二进制机器码**，直接作为 C++ byte array 嵌入源码分发：

```cpp
unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm87_cu_cubin[] = {
    0x7f, 0x45, 0x4c, 0x46, ...  // SM 8.7 专用 ELF 二进制
};
```

文件命名规则也很说明问题：

```
fmha_v2_flash_attention_{dtype}_{blockM}_{blockN}_S_{layout}_{headdim}_sm87
```

每种 dtype × block size × layout × headdim 的组合，都有一个**为 SM 8.7 单独优化的 cubin**。

这和开源 flash-attn 的差异是根本性的：

| | 开源 flash-attn v2.8.3 | TRT-LLM FMHA |
|--|--|--|
| SM 8.7 处理 | 走 sm8x 通用路径（同 SM 8.6/8.9） | **180 个 SM 8.7 专用预编译 cubin** |
| L2 cache 感知 | v2 路径无 L2 感知，tile size 按 SM 8.6/8.9 调优 | NVIDIA 内部针对 SM 8.7 单独调优 |
| tile size 来源 | 开源代码，社区维护 | NVIDIA 内部工具链生成 |
| 编译方式 | 用户机器现场 nvcc 编译 | 预编译 ELF 嵌入 C++ 源码 |
| 覆盖 SM 版本 | 80, 86, 89, 90（87 走 sm8x 通用路径） | 80, 86, **87**, 89, 90 全部单独覆盖 |

### 实测性能对比

光说架构差异不够，我们实际做了一个 **isolated attention kernel micro-benchmark**，把三种后端放在完全相同的输入上对比（wall-x 实际维度：batch=1, 16 heads, 2 KV heads, head_dim=128, seq=420, bf16）：

**Prefill（Q=420, KV=420）：**

| 后端 | 延迟 | vs FA2 | vs SDPA |
|------|------|--------|---------|
| **TRT-LLM FMHA** (SM 8.7 cubin) | **0.075 ms** | **4.1x 更快** | **1.7x 更快** |
| SDPA (cuDNN) | 0.127 ms | 2.4x 更快 | 1.0x |
| 开源 FA2 v2.8.3 | 0.306 ms | 1.0x（最慢） | 2.4x 更慢 |

**Decode（Q=1, KV=420）：**

| 后端 | 延迟 | vs FA2 |
|------|------|--------|
| SDPA (cuDNN) | **0.090 ms** | **3.1x 更快** |
| 开源 FA2 v2.8.3 | 0.278 ms | 1.0x |

> TRT-LLM decode 未测（需要完整 TRT engine 才能跑 decode attention，standalone cubin 只支持 prefill 模式）。

### 一个反直觉的发现

**FA2 是三者里 attention kernel 最慢的，但全模型推理反而快 27%？**

原因在于 **softmax fusion**。nsys 数据对比显示：SDPA trace 里有 **64 次 `cunn_SoftMaxForward` 调用（共 746ms）**，FA2 trace 里 **一次都没有**。kernel 上下文进一步确认了这是 SDPA math backend 的典型模式——`cutlass GEMM (Q×K^T) → elementwise (mask add) → cunn_SoftMaxForward → cutlass GEMM (×V)`，即 Q×K^T、softmax、×V 作为三个独立 kernel 执行。

**为什么 SDPA 没有用 cuDNN 的融合 attention？** 我们验证过，cuDNN 9.3 确实有 fused attention 并且在 Orin 上可用（standalone benchmark 0.076ms）。但 cuDNN 和 flash_sdp 都**不支持非空 `attention_mask`**：

```
Flash Attention does not support non-null attn_mask
cuDNN Attention does not support non-null attn_mask
```

而 HuggingFace 的 `sdpa_attention_forward`（transformers 4.45.1）在 `generate()` 期间**始终传递 `attention_mask`**。这直接导致 cuDNN 和 flash_sdp 被禁用，SDPA 退化为 math backend——也就是最朴素的 `Q×K^T → softmax → ×V` 三步分离实现。具体来说，HuggingFace 的 GQA 处理逻辑（`use_gqa_in_sdpa`）源码注释写明：*"attention_mask is None (otherwise it will fall back to the math kernel)"*。

FA2 则完全不受这个限制。它在 CUDA C++ kernel 内部处理 causal mask，无论 HuggingFace 传不传 `attention_mask`，softmax 始终融合在 kernel 里。

所以 FA2 在 Orin 上 27% 的加速，**不是因为 attention kernel 本身更快**（isolated benchmark 实际上更慢），而是因为**绕开了 HuggingFace 的 `attention_mask` → math backend 退化路径，省掉了 746ms 的独立 softmax kernel。** 这是一个 kernel fusion + 框架交互的综合收益。

这解释了一个关键疑问：如果有人在 Orin 上用 TRT-LLM 跑推理并观察到 Flash Attention 有 2x 加速，那很可能是因为 **TRT-LLM 用的是 NVIDIA 内部专门为 SM 8.7 调优的 cubin，而不是开源 flash-attn 的 sm8x 通用 kernel。**

换句话说，"Flash Attention"这个名字在两个语境下指的不是同一个东西。开源 flash-attn 没有为 Orin 做过任何硬件级调优，TRT-LLM 的版本则是 NVIDIA 自己的 fused attention kernel，和 Dao 的开源实现共享算法思路，但 kernel 代码完全独立。

### Orin 上的三条 Flash Attention 路线

到这里，Orin 上的 Flash Attention 其实有**三条完全不同的路线**，值得放在一起对比：

| 路线 | Prefill 延迟 | SM 8.7 适配 | GQA 支持 | 接入方式 | 工程量 |
|------|-------------|------------|----------|---------|--------|
| **① 开源 flash-attn v2.8.3** | 0.306 ms | sm8x 通用路径（同 SM 8.6/8.9） | 原生 | PyTorch，已在用 | **零（已完成）** |
| **② TRT-LLM FMHA cubin** | **0.075 ms** | **180 个 SM 8.7 专用 cubin** | 有 `q_kv` 变体 | CUDA Driver API 加载 | 大 |
| **③ FlashInfer** | 待测 | 非官方（`FLASHINFER_CUDA_ARCH_LIST="8.7"` 可编译） | 原生 | Python API | 中等 |

**路线 ①：开源 flash-attn（当前方案）**

就是本文编译的 FA2 v2.8.3。优点是已经跑通了，和 HuggingFace 无缝集成。缺点是 tile size 按 SM 8.6/8.9 调优，Orin 不在设计范围，attention kernel 本身比 cuDNN SDPA 还慢（0.306ms vs 0.127ms）。27% 的全模型加速纯靠绕过 HuggingFace 的 `attention_mask` → math backend 退化路径。

**路线 ②：提取 TRT-LLM FMHA cubin**

TRT-LLM 0.12.0（Orin 上当前版本）没有 FlashInfer 依赖，FMHA 完全是内部 cubin。技术上可以提取——源码 Apache 2.0 开源，cubin 加载机制是标准 CUDA Driver API（`cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel`），runner 代码 `fmhaRunner.cpp` 里有完整的参数结构体和 launch 逻辑。wall-x 需要的 cubin 也确实存在：`fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm87`（bf16，GQA layout，head_dim=128）。

但**投入产出比很低**。我们前面的 profiling 已经表明，attention kernel 只占 GPU 总时间的 **1.4%**（30.4ms / 2101ms）。即使把 attention 从 0.306ms 优化到 0.075ms（快 4.1x），对 6360ms 的总延迟只省 **~20ms**。为了 20ms 去写 standalone cubin wrapper、处理 `Fused_multihead_attention_params_v2` 结构体的几十个字段，不划算。

**路线 ③：FlashInfer**

FlashInfer 是 Zihao Ye 等人开发的开源 attention 库，特点是原生支持 GQA、paged KV cache、多种 attention pattern，且有 Python API。高版本 TRT-LLM（0.15+）已经用 FlashInfer 替代了内部 cubin 做 attention。

FlashInfer 官方不支持 SM 8.7（官方列表：SM75, SM80, SM86, SM89, SM90, SM103, SM110, SM120, SM121）。但 GitHub issue [#2579](https://github.com/flashinfer-ai/flashinfer/issues/2579) 报告：**设置 `FLASHINFER_CUDA_ARCH_LIST="8.7"` 后，FlashInfer 在 Jetson Orin AGX 上可以正常编译运行，LLM 推理提速 25%。** 和 flash-attn 的情况几乎一样——SM 8.7 实际可用，只是没进官方列表。

FlashInfer 的优势在于它是**面向推理场景设计的**，比 flash-attn 更适合 decode（有专门的 decode kernel 和 batch decode API），且 tile size 和 split 策略有更灵活的 auto-tuning。如果后续做 C++ 推理框架改造，FlashInfer 是比 flash-attn 更好的 attention 底层选择。

**额外发现：cuDNN SDPA 本身就接近 TRT-LLM 水平**

我们在分析 softmax 问题时还发现了一个意外的结论：**如果能绕过 HuggingFace 的 `attention_mask`，直接调 `F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)`，cuDNN fused attention 就会生效——延迟 0.076ms，几乎等于 TRT-LLM 的 0.075ms。**

这意味着在 C++ 推理框架改造时，不需要提取任何 cubin，也不需要额外集成 FlashInfer——只要自己管理 causal mask 逻辑（在 C++ 里很容易），让 SDPA 不传 `attention_mask`，cuDNN 就直接给你接近 TRT-LLM 级别的 attention 性能。这是 **零额外成本** 的路线。

---

## 九、Wall-X 能不能用 TRT-LLM 或 llama.cpp 推理？

既然 TRT-LLM 有 SM 8.7 优化的 Flash Attention，一个自然的问题是：能不能直接把 wall-x 搬到 TRT-LLM 上推理？

我仔细分析了 wall-x 的模型结构。答案是：**取决于你要跑什么任务。**

### 先看模型结构

wall-x 不是一个标准的"文本进 → 文本出"的 LLM，它可以拆成 4 层：

| 层 | 内容 | 标准程度 |
|----|------|----------|
| 1. 视觉编码器 | Qwen2.5 ViT + window attention + spatial merge | 比较标准 |
| 2. 文本 Decoder | Qwen2.5 causal decoder + GQA/SDPA | **标准** |
| 3. MoE 改造 | TokenTypeRouter + SparseMoeBlock + 自定义 permute CUDA kernel | **非标准** |
| 4. 动作生成 | ActionProcessor + Flow Matching + ODE Euler 积分 + KV Cache 截断复用 | **完全非标准** |

### TRT-LLM：VQA 能做，Flow Action 做不了

**能做的部分：**

TRT-LLM 已经支持 Qwen2.5-VL 的基础架构（有现成 example），而且它有 SM 8.7 专用的 180 个 flash attention cubin。第 1、2 层理论上可以直接映射。

**做不了的部分：**

1. **wall-x 的 MoE 不是标准 MoE**。TRT-LLM 支持的 MoE 是 Mixtral 那种 top-K gating（router 网络输出 softmax 再选 top-K 专家）。但 wall-x 用的是 `TokenTypeRouter`，路由逻辑是 `expert_idx = token_type % num_experts`——按 token 类型硬分流，不是 softmax 选路。这需要写 TRT-LLM plugin。

2. **Flow Action 生成是根本性障碍。** wall-x 的动作预测走的是 ODE 积分：先用噪声初始化，经过完整前向拿到初始速度预测，然后截断 prefix KV Cache，在 Euler 循环里反复用截断的 KV Cache + 新的 action embedding 做增量前向。

   ```
   初始化噪声 → 前向得到 v_0 → 截断 KV Cache →
   Euler 循环 { 替换 action embedding → 只跑 postfix → 得到 v_t → 更新 noisy_action } → 反归一化
   ```

   这个循环里每一步都需要**运行时动态替换输入 embedding**——把 `<|action|>` 位置的 embedding 替换成 `ActionProcessor.step()` 的输出。TRT-LLM 是静态图引擎，不支持这种推理过程中动态修改中间张量的操作。

3. **Proprioception 嵌入**也是动态的。`<|propri|>` token 在运行时被替换成 `propri_proj(concat(agent_pos, dof_mask))` 投影的 embedding，这种 runtime scatter 同样需要自定义 plugin。

**结论：TRT-LLM 在理论上能跑 wall-x 的 VQA（纯文本生成），但跑不了 Flow Action（动作预测）。** 即使是 VQA，也需要开发 MoE 路由和 proprioception embedding 的 plugin，工程量不小。

### llama.cpp：不可行

llama.cpp 的限制更根本：

1. 模型必须转成 GGUF 格式，所有计算都在纯 C++ 里跑
2. wall-x 有 **6 个自定义 CUDA 算子**（permute, unpermute, dual_asym_gemm, multimodal_rope, rope_index, window_index），llama.cpp 没有 plugin 机制
3. llama.cpp 的多模态支持是最简单的"ViT + decoder"模式，不支持自定义 MoE
4. Flow Matching / ODE 积分在 llama.cpp 里完全没有对应概念
5. Qwen2.5-VL 的 3D RoPE（temporal/height/width 三维位置编码）支持也不完整

**结论：llama.cpp 对 wall-x 来说基本不可行。**

### 那真正可行的优化路线是什么？

| 路线 | 可行性 | 适用场景 | 工程量 |
|------|--------|----------|--------|
| **PyTorch + FA2（当前最优）** | 已验证 | 全部 | 零（已完成） |
| **TRT-LLM 只跑 VQA** | 中等 | 纯文本问答 | 需要 MoE + proprio plugin |
| **TRT-LLM 跑 Flow Action** | 很低 | 动作预测 | 需要重写整个 ODE 循环 |
| **llama.cpp** | 不可行 | — | — |
| **INT8/INT4 量化（PyTorch 内）** | 高 | 全部 | 中等 |
| **C++ 推理框架 + CUDA Graph** | 高 | 全部 | 较大 |

最务实的结论是：**不换推理引擎，在 PyTorch 内做量化和编译优化，投入产出比最高。** 因为 wall-x 的核心价值是 Flow Action，而这个能力在 TRT-LLM 和 llama.cpp 里都跑不了。

---

## 十、工程产出

虽然 FA2 在 Orin 上带来了 27% 的提速，但这次编译和 benchmark 过程留下的最大价值不只是性能数据，还有完整的工程资产：

1. **`3rdparty/flash-attention/`**：FA2 v2.8.3 完整源码，已打好两个 Orin 补丁
2. **`setup.py`**：SM 8.7 gencode 支持，默认 `FLASH_ATTN_CUDA_ARCHS="80;87;90;100;120"`
3. **`flash_attn/ops/triton/rotary.py`**：triton / PyTorch 自动切换，x86 用 triton，aarch64 用 PyTorch fallback
4. **`scripts/bench_fa2_vs_sdpa.py`**：标准化的 FA2 vs SDPA benchmark 脚本
5. **`ORIN_BUILD.md`**：完整的 Orin 编译和安装指南

---

## 十一、优化方向更新

结合本篇所有发现（包括 GEMM profiling 和框架开销分析），把优化方向重新评估：

| 方向 | 本篇结论 | 预估收益 | 后续计划 |
|------|----------|----------|----------|
| **消除 Python 框架开销** | **67% 时间 GPU 在等 Python，每步 67.5ms 浪费** | **~4320ms (67%)** | **最高优先级：C++ libtorch / CUDA Graph** |
| **Flash Attention（开源 FA2）** | 已编译通过，比 SDPA 快 27%。tile size 按 SM 8.6/8.9 调优 | 已获得 27% | 生产用 FA2 |
| **Flash Attention（cuDNN SDPA）** | **绕过 `attention_mask` 后 cuDNN 直接生效（0.076ms ≈ TRT-LLM 的 0.075ms）** | 与 TRT-LLM 持平 | **C++ 框架中直接用，零额外成本** |
| Flash Attention（TRT-LLM cubin） | 180 个 SM 8.7 专用 cubin，可提取但 attention 仅占 1.4% | ~20ms | 投入产出比低，不推荐 |
| Flash Attention（FlashInfer） | 非官方支持 SM 8.7，设置 `FLASHINFER_CUDA_ARCH_LIST="8.7"` 可编译 | 待测 | C++ 改造后可评估 |
| **INT8/INT4 量化** | torch._int_mm 不支持 M=1，bitsandbytes 不支持 SM 8.7 | ~800ms (12%) | **需要 C++ 层面 cublasLt INT8** |
| torch.compile | 自定义 CUDA op 导致 graph break，仅快 1.7% | ~100ms | 不推荐 |
| TRT-LLM / llama.cpp | VQA 部分理论可行，Flow Action 不可行 | — | 放弃 |

**核心结论逆转：之前以为 GEMM 计算（78.8% GPU 时间）是最大瓶颈，现在发现 Python 框架开销（67% 总时间）才是。** GEMM 的 1643ms 确实占了 GPU kernel 时间的 78.8%，但 GPU 只在 32.7% 的时间里工作——真正的大头是那 4320ms 的"GPU 空转等 Python"。

正确的优化顺序：
1. **Phase 1：C++ 框架**（消除 67% 框架开销） → 理论极限 ~2100ms（纯 GPU kernel 时间）
2. **Phase 2：INT8 量化**（在 C++ 中用 cublasLt INT8 GEMV） → 进一步砍 ~50% GEMM
3. **Phase 3：CUDA Graph**（消除 kernel launch 开销） → 额外省 ~260ms

---

## 十二、结语

这篇文章的故事比预期的曲折得多：

1. 上一篇说 FA2 编译不过 → 这一篇编译通了
2. Benchmark 实测：**FA2 比 SDPA 快 27%，内存持平，方差小 4 倍**
3. 与 5090 对比：Orin + FA2 把差距从 8.0x 缩小到 5.9x
4. 深挖原因：FA2 的 tile size 是源码注释写明 *"For sm86 or sm89"* 调优的（Orin 不在设计范围），但 Softmax 融合本身的算法优势足以弥补
5. 对比发现：TRT-LLM 有 180 个 SM 8.7 专用 flash attention cubin，和开源 flash-attn 不是一回事。加上 FlashInfer，Orin 上 Flash Attention 有三条路线——但最务实的发现是：**绕过 HuggingFace 的 `attention_mask` 后，cuDNN SDPA 就能达到 0.076ms（≈ TRT-LLM 的 0.075ms），不需要提取任何 cubin**
6. 评估引擎替换：wall-x 的 Flow Action 在 TRT-LLM 和 llama.cpp 里都跑不了
7. GEMM profiling 发现 52.9% GEMM 时间在 decode 小矩阵——但 cuBLAS 已是最优 bf16 实现，kernel 时间 ≈ 纯内存读取时间，**是带宽瓶颈不是算力瓶颈，只有量化能解**
8. **最终反转：GPU 利用率只有 32.7%，67% 的时间浪费在 Python 框架开销上——这才是真正的优化大头**

这不只是一个"FA2 编译和 benchmark"的故事，更是一层层剥洋葱发现真相的过程。先以为瓶颈是 Softmax 未融合（FA2 解决），再以为瓶颈是 GEMM 计算（78.8% GPU 时间），然后发现 GEMM 其实已经跑在 Orin 带宽极限（cuBLAS 利用率 72%），最后发现真正的大头是 **GPU 在等 Python 的 67% 空转时间**。

算子优化的天花板越来越低，系统优化的空间越来越大。这正是我在文末提出的核心观点：**具身智能的本质是实时系统。** VLA 不是 chatbot，它是一个有损压缩的物理模拟器——刷新率（采样频率）比单次精度更重要。优化路径已经从算子层面转移到了系统层面：C++ runtime、CUDA Graph、model set 协同调度。

---

## 系列预告

本文是 **wall-x 机器人大模型部署系列** 的第二篇。

**第一篇**：[把 3B 大模型塞进机器人：RTX 5090 与 Jetson Orin 边缘端产品部署踩坑全记录](#)
- 环境搭建、CUDA 依赖链、profiling 数据、28 万+ kernel 分析

**第二篇（本文）**：当算子逼近硬件极限
- FA2 编译全过程、benchmark 实测快 27%、深度 profiling 从 attention 挖到框架开销
- GEMM 已触达带宽天花板（cuBLAS 利用率 72%），67% 时间是框架"空转"
- 核心观点：具身智能的瓶颈不在 model，而在 runtime

**第三篇（预告）**：消除 67% 的框架空转——C++ 推理框架改造
- 发现：GPU 利用率只有 32.7%，67% 时间是 Python/HuggingFace 开销
- 每个 decode step：97.2ms 总耗时，仅 29.7ms GPU 计算
- libtorch / CUDA Graph / 自定义 decode loop 三条路线验证
- wall-x 的 Flow Action ODE 循环能不能用 C++ 重写
- **Triton 自定义算子探索**：消除框架开销后，回到算子层面，用 Triton 为 Orin 的 16 SM + 4MB L2 写专用 attention / decode kernel
- 目标：从 6.4s 压到 ~2.1s（纯 GPU kernel 时间理论极限）

**第四篇（预告）**：在 Orin 上给 wall-x 这个机器人 VLA 做 INT8：为什么理论 2× 加速一开始几乎没用
- 朴素 INT8 为什么一开始反而更慢：INT32 落地带宽 + kernel 路径选择的双重代价
- `vision_mlp` 的 padding 补齐、`MoE expert INT8` 的双路径运行时
- 从“量化几乎没用”到 Flow Action `~480ms`、VQA `~1.0s` 的覆盖率演化
- 量化的真正价值不是单个 kernel 跑多快，而是覆盖率打到哪里

**第五篇（预告）**：量化之后还剩什么：在 Orin 上给 wall-x 做 CUDA Graph、算子融合和 CUTLASS
- ODE / postfix fixed-shape 推理的 graph capture
- `residual + rmsnorm`、`quantize + layout transform`、`GEMM + SiLU` 这类高频短链融合
- 基于 CUTLASS 把 GEMM 前后的数据流继续往主算子里收
- 从手工 fusion 走到 compiler pass，最后自然逼近 runtime / AI OS

---

## 附录：CPU 负载对 GPU 推理的影响——Doorbell 调度延迟

> 这一节不在主线叙述中，但记录了一个在 Orin 统一内存架构下值得注意的现象：**CPU 高负载会显著拖慢 GPU 推理中由 CPU 负责的部分。**

### 背景

本文的 benchmark 数据是在系统空闲（load avg < 5）条件下采集的。但最初的第一轮 benchmark 是在 TRT-LLM 同时编译的情况下跑的——当时 `load avg ~18`，12 个 ARM 核心几乎被 nvcc 编译任务占满。

这次"失误"意外揭示了一个有价值的现象。

### 数据对比

| | 高负载（load avg ~18） | 空闲（load avg < 5） | 退化幅度 |
|--|--|--|--|
| SDPA | 9503 ms (std 487) | 8681 ms (std 109) | **慢 9%** |
| **FA2** | **16289 ms (std 1410)** | **6360 ms (std 25)** | **慢 156%** |
| FA2 峰值内存 | 15.97 GB | 8.14 GB | 多出 8GB 是 TRT-LLM 的 |
| FA2 标准差 | 1410 ms | 25 ms | **方差差 56 倍** |

SDPA 几乎不受影响（9%），而 FA2 慢了 **2.56 倍**，方差炸到 56 倍。

需要说明的是：SDPA 和 FA2 并非同一瞬间测的——TRT-LLM 编译负载不均匀（nvcc 并行编译 .cu 文件时 CPU 打满，cmake/链接阶段负载骤降），SDPA 测试时的实际 CPU 负载可能低于 FA2 测试时。**9% vs 156% 的巨大差异，很可能主要来自测试时点的负载差异，而非两种 attention 后端本身对 CPU 负载的敏感性不同。**

### 为什么 CPU 高负载会影响 GPU 推理？

GPU kernel 的执行不是纯 GPU 的事。每次 kernel launch 都需要 CPU 完成以下步骤：

1. **准备 kernel 参数**（CPU 端设置 grid/block/shared memory 配置）
2. **写入 doorbell 寄存器**（CPU 通过 PCIe/内存映射通知 GPU "有新任务"）
3. **GPU 调度器响应**（GPU 端从 command queue 取任务开始执行）

在传统数据中心 GPU（A100/H100）上，CPU 和 GPU 通过 **PCIe 独立通信**，CPU 负载不影响 GPU 端的带宽。但在 Orin 的**统一内存架构**下：

- CPU 和 GPU **共享同一条内存总线**（LPDDR5, 204 GB/s）
- CPU 高负载时，编译进程大量读写内存，和 GPU 的 kernel 数据传输**争抢总线带宽**
- 更关键的是：CPU 被编译任务占满后，**kernel launch 的 doorbell 写入被延迟**——CPU 来不及把下一个 kernel 的启动指令发给 GPU，GPU 只能空等

在 Orin 的统一内存架构下，任何 GPU 推理都对 CPU 调度延迟敏感——不论是 FA2 还是 SDPA。两者的模型结构完全一致（36 层 transformer，同样的 GEMM、LayerNorm、RoPE），总 kernel launch 数量级相当（FA2 因为融合了 softmax 反而更少）。这次 benchmark 中 FA2 退化更严重，主要原因很可能是测试时恰好赶上了编译负载的峰值。

### 实际影响

nsys 数据显示 wall-x 一次 VQA 推理有 **52280 次 kernel launch**，平均每次 launch 开销 20.7μs。当 CPU 负载从 load avg 5 升到 18 时，这个 launch 开销会显著膨胀——假设 launch 延迟从 20μs 涨到 ~200μs，52280 次 launch 就多出 **~9.4 秒**的纯调度等待。

这与实测数据吻合：FA2 从 6360ms 涨到 16289ms，多出的 ~10 秒主要来自 CPU 端 kernel launch doorbell 的排队延迟。

### 启示

1. **统一内存 ≠ 独立显存**。在 A100 上，CPU 编译不影响 GPU benchmark；在 Orin 上，CPU 密集型任务直接拖慢 GPU 推理
2. **看方差是最快的诊断手段**。std 从 25ms 炸到 1410ms 是明确的"数据有问题"信号
3. **边缘设备部署时要控制 CPU 负载**。如果 Orin 同时跑视觉预处理、网络通信等 CPU 密集任务，GPU 推理延迟可能退化到 2-3 倍。这不是 GPU 算力不够，而是 CPU 来不及"喂"GPU

---

## 附录：Triton vs PyTorch 算子性能摸底（Orin SM 8.7）

> 前文验证了 Triton 3.6.0 在 Orin 上可以正常编译和运行。但"能跑"和"跑得快"是两回事——Triton 的性能取决于其 MLIR pass 为 SM 8.7 生成的 PTX/SASS 质量。以下是几个主流算子的实测对比。

### 测试方法

选取 5 类算子，分别用标准 Triton kernel 和 PyTorch 原生实现对比。每个算子 warmup 10 次、正式计时 100 次取平均。系统空闲、jetson_clocks 锁频。

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

### 分析

**Triton 在 SM 8.7 上的 codegen 质量中等**：

1. **单算子打不过 cuBLAS/cuDNN**。GEMM 慢 40%，vector_add 慢 1.8x——这些算子 PyTorch 底层调用的是 NVIDIA 专门为每个 SM 调优过的闭源库，Triton 的通用 MLIR pass 在这些"已经被打磨到极致"的算子上没有优势
2. **小 tensor 不适合 Triton**。decode 阶段的 16×420 softmax，数据量太小，Triton kernel launch 开销（~0.04ms）就超过了计算本身
3. **Triton 的真正价值在 kernel fusion**。fused_add_rmsnorm 比 PyTorch 快 **3.9x**、大矩阵 softmax 快 **4.3x**。原因很简单：PyTorch 要分 3-4 个 kernel 完成（add → square → mean → normalize），每个 kernel 都要把中间结果写回全局内存再读回来；Triton 一个 kernel 全做完，中间结果留在寄存器里

### 对 wall-x 优化的启示

- **不要**用 Triton 替代 cuBLAS 做 GEMM/GEMV——这是做无用功
- **要做的**是用 Triton 写 fused kernel：把 LLM decode 中相邻的小算子合并（如 residual_add + rmsnorm、gate_proj × up_proj fusion、RoPE + attention score），每个 fusion 都能省掉一轮全局内存读写
- **decode 阶段 batch=1 的极小算子**需要谨慎评估，launch 开销可能吃掉 fusion 收益

> 脚本：`scripts/bench_triton_vs_pytorch_orin.py`，Triton 3.6.0，PyTorch 2.5.0a0+872d972e41.nv24.08，Orin 64GB 锁频。

---

## 写在最后：具身智能的本质是实时系统

这篇文章讲的是 Flash Attention 2 在 Orin 上的编译、benchmark 和深度 profiling。但走到这里，我越来越觉得这个系列真正在回答的，不是"怎么让模型跑快一点"的技术问题——而是一个更根本的问题：

**具身智能（Embodied AI）的本质是什么？**

### 一个 VLA 模型不是一个 chatbot

做 LLM 服务的人思考的是"token first"——first token latency、throughput、prefill/decode 分离、KV cache paging。这些指标对云端 chatbot 完全正确。但在机器人上，这些指标是错的。

机器人需要的不是"快速吐 token"——它需要在**固定的时间窗口内完成感知→决策→执行的完整闭环**。一个 VLA 模型（Vision-Language-Action），它的输入是图片和指令，输出是关节动作序列。这不是一个"语言生成"任务，而是一个**有损压缩的物理模拟器**：

- **ViT 编码器**把 640×480 的 RGB 图像压缩成 ~350 个视觉 token——这是对物理世界的有损感知
- **语言 Decoder** + **MoE 路由**把视觉 token 和文本指令融合成一个统一表征——这是对任务意图的有损理解
- **Flow Action Head** 用 ODE 积分从噪声中"生成"动作序列——这是对物理运动的有损预测

每一步都是有损的。模型精度的上限不是 100% —— 它本质上是在用一个 3B 参数的神经网络去**近似模拟**物理世界的动力学。它会犯错，它的轨迹预测会有偏差，它对遮挡物体的感知会有盲区。

### 如果模型本身是有损的，优化什么才有价值？

传统思路是：提高模型精度（更多参数、更好的训练数据、更精细的 loss）。这当然重要，但在边缘端部署时，还有另一条路：

**用系统层面的优化来弥补模型的缺陷。**

一个有损的物理模拟器，如果能以 **2x 的频率** 运行，那它的累积误差就会小很多——因为每步预测的时间跨度更短、修正更及时。这和控制理论里的 **采样定理** 一样：采样率越高，控制越精确。一个 100ms 的 VLA 模型，每秒做 10 次感知-决策，比一个 200ms 的模型精度更高的原因不是模型更好，而是**系统响应更快**。

所以这四篇文章的优化路径——

1. FA2 编译（本文）：省 27%，从 8.7s → 6.4s
2. C++ 框架改造（第三篇）：消除 67% 框架空转，理论极限 → ~2.1s
3. INT8 量化（第四篇）：砍 GEMM 带宽瓶颈，→ ~1.2s
4. Triton fused kernel：榨干最后的 3-4x fusion 收益

——不只是在"让推理更快"。它在做的事情是：**把一个有损物理模拟器的刷新率从 ~1.5 Hz 提到 ~8 Hz**。这不是优化，这是在改变系统的物理特性。

### Model Set Runtime

在第一篇文章里我提过一个观点：边缘端的模型部署和云端"token first"的服务化思路完全不同。Orin 上跑的本质是一个 **model set 的 runtime**——视觉编码器、语言模型、MoE 路由、Flow Action Head，多个模型在同一个进程内协同调度，而不是拆成微服务各自吐 token。

现在我越来越觉得这个 model set 的抽象需要进一步推广：

**真正需要优化的不是单个模型，而是整个 model set + OS 调度 + 硬件的联合系统。**

本文的 GEMM 分析验证了这一点——cuBLAS 已经跑在带宽极限的 72%，kernel 时间 ≈ 纯内存读取时间，算子层面的优化空间接近天花板了。但 67% 的框架空转、52280 次 kernel launch 的调度延迟、CPU-GPU 共享总线的带宽竞争——这些全是**系统层面**的问题，不是模型层面的问题。

换句话说，**具身智能的瓶颈不在 model，而在 runtime。**

### C++ 推理不只是"让它跑快一点"

回头看这个系列，C++ 推理框架改造（第三篇）不再只是一个性能优化项目。它实际上是在验证一个更宏大的假设：

> **当算子已经逼近硬件极限时，系统工程（OS 调度、内存管理、进程协同）才是具身智能真正的杠杆。**

具身智能不是"一个更好的模型"——它是一个**实时系统（Real-time System）**。它需要的不只是更快的 Tensor Core，更需要：

- 确定性的延迟上界（CUDA Graph 消除 launch jitter）
- 最小化的 CPU-GPU 同步开销（C++ 替代 Python GIL）
- 多模型的协同调度（model set runtime）
- 感知-决策-执行的流水线并行

这些都是 OS / 系统工程的范畴，不是深度学习的范畴。

本文证明了 FA2 有效（27% 加速），证明了 GEMM 已触达带宽天花板（cuBLAS 利用率 72%，只有量化能继续），证明了框架空转是最大的浪费（67%）。这三个结论共同指向同一个方向：**优化路径已经从模型/算子层面，转移到了系统层面。**

**算子已触顶，系统才是杠杆。** C++ 推理框架改造，就是这个方向的第一次实际验证。我们下一篇见。

---

*测试环境：wall-oss-flow 3B 模型，bfloat16 精度，batch_size=1。测试平台：Jetson AGX Orin 64GB，JetPack 6.2.1，CUDA 12.6，PyTorch 2.5.0a0+872d972e41.nv24.08。Flash Attention v2.8.3 源码编译，SM 8.7 gencode。Benchmark 脚本：bench_fa2_vs_sdpa.py / bench_gemm_opt.py / profile_overhead.py，640×480 测试图片，VQA prompt，max_new_tokens=64，warmup 2-3 次，正式计时 5-10 次。最终数据在系统空闲（load avg < 5）、GPU 锁频（jetson_clocks, 1300.5 MHz）条件下采集。GEMM profiling 数据来自 nsys trace（fa2_nsys.sqlite），NCU 数据来自 ncu --kernel-name regex 抓取。Triton benchmark 数据来自 bench_triton_vs_pytorch_orin.py，warmup 10 次，正式 100 次。2026 年 4 月实测数据。*
