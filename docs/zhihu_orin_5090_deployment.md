# 把 3B 大模型塞进机器人：RTX 5090 与 Jetson Orin 边缘端产品部署踩坑全记录

> 本文记录了将一个 3B 参数量的机器人视觉-语言-动作（VLA）大模型 wall-x 分别部署到 NVIDIA RTX 5090 和 Jetson AGX Orin 上的完整过程，包含大量真实踩坑经历和性能对比数据。如果你也在做端侧/边缘侧的大模型部署，希望这篇文章能帮你少走一些弯路。

**TL;DR**
- RTX 5090 动作预测 **14ms（70Hz）**，Orin **115ms（8.7Hz）**，端到端差距 **8 倍**
- Nsight Systems 抓了 **28 万+ kernel**，GPU 纯算时间差距其实是 **13.2 倍**，差值被 CPU/Python 开销吃掉了
- **GEMM/GEMV 占 54-69% GPU 时间**，是推理延迟的绝对主体；量化可以直接砍一半以上
- Orin 上 SDPA 的 Softmax 没有融合，独立 kernel 比 5090 **慢 57 倍**（746ms vs 13ms）——最大单点优化机会
- Flash Attention 在 Orin 上编译未通过，下一篇会攻克它

---

## 一、背景：为什么要同时部署两个平台？

wall-x 是一个基于 Qwen2.5-VL 架构改造的机器人 VLA 模型，参数量约 3B，支持视觉问答（VQA）和流式动作预测（Flow Action）。它的应用场景是机器人操控——机器人看到一张图，理解指令，输出一连串的关节动作。

在研发阶段，我们需要在两种截然不同的硬件上验证推理：

- **RTX 5090**（Blackwell 架构）：代表最新的桌面级算力，32GB VRAM，用于快速迭代和 benchmark
- **Jetson AGX Orin**（Ampere 架构）：代表真实的机器人端侧部署环境，64GB 统一内存，但 GPU 算力远弱于桌面卡

两个平台的软件栈差异较大，部署过程中需要注意不少细节。

---

## 二、硬件环境一览

| 项目 | RTX 5090 | Jetson AGX Orin |
|------|----------|-----------------|
| GPU 架构 | Blackwell (SM 12.0) | Ampere (SM 8.7) |
| CUDA 版本 | 13.0 | 12.6 |
| 显存 | 32 GB GDDR7 (独立) | 64 GB LPDDR5 (CPU/GPU 共享) |
| PyTorch | 2.11.0+cu130 | 2.5.0a0 (NVIDIA Jetson 定制版) |
| 系统 | Ubuntu + x86_64 | JetPack 6.2.1 + aarch64 |
| 驱动 | 570+ | 540.4.0 (L4T) |
| Flash Attention | 2.8.3 (源码编译) | 编译未通过，使用 SDPA 替代 |
| 模型精度 | bfloat16 | bfloat16 |
| GPU 监控工具 | nvidia-smi (完整) | tegrastats (有限) |
| Python | 3.10 | 3.10 |

光看这张表就知道，两个平台几乎没有任何依赖是完全通用的。

### 2.1 CUDA 软件栈依赖链

理解 NVIDIA GPU 的软件栈依赖关系，是部署前最重要的一步。整个链路是**自上而下锁定**的：

```
GPU 驱动 / BSP 包
    └── 决定 CUDA Toolkit 版本上限
            └── 决定 cuDNN / cuBLAS / CUTLASS 等库版本
                    └── 决定 PyTorch 版本（需要匹配 CUDA）
                            └── 决定 Flash Attention / transformers 等上层库
```

- **桌面 GPU（RTX 5090）**：驱动可以独立升级（如 570+），驱动版本决定了支持的最高 CUDA 版本（13.x）。升级驱动相对简单，所以 CUDA 版本选择空间较大。
- **Jetson 平台（Orin）**：驱动包含在 **BSP（Board Support Package）** 中，由 JetPack SDK 统一发布。JetPack 6.2.1 锁定了 L4T 驱动 540.4.0 和 CUDA 12.6，无法单独升级驱动或 CUDA。想用更高版本的 CUDA，必须等 NVIDIA 发布新的 JetPack。

### 2.2 SM 号与 PTX/SASS 编译

GPU 的 **SM（Streaming Multiprocessor）版本号**（如 SM 8.7、SM 12.0）在编译和运行时都扮演关键角色：

1. **编译阶段**：CUDA 代码通过 nvcc 编译为 **PTX**（平台无关的中间指令）和/或 **SASS**（特定 SM 版本的原生机器码）。`TORCH_CUDA_ARCH_LIST` 参数控制为哪些 SM 版本生成代码。

2. **运行时 JIT**：如果二进制中只包含 PTX 而没有匹配当前 GPU 的 SASS，驱动会在运行时将 PTX **JIT 编译**为 SASS。这个过程需要**驱动版本支持目标 SM 号**——如果驱动太老，不认识新的 SM 版本，JIT 会直接失败。

3. **实际影响**：
   - RTX 5090 的 SM 12.0 需要较新的驱动（570+）才能 JIT 编译 PTX → SASS
   - Orin 的 SM 8.7 是 Jetson 专属，标准 PyTorch wheel 的 `TORCH_CUDA_ARCH_LIST` 中不包含它，所以即使 PTX JIT 理论上可行，预编译 SASS 未覆盖 SM 8.7 的 wheel 仍然会失败
   - 这就是为什么 Orin 必须使用 NVIDIA Jetson 定制 wheel（显式包含 `sm_87` 和 `compute_87`）

简而言之：**驱动/BSP → CUDA → 计算库 → PyTorch → 上层框架**，这条链路上任何一环的版本限制，都会向上传递。而 SM 版本号决定了 GPU 能否正确执行编译产物，它是连接编译器和驱动的桥梁。

---

## 三、RTX 5090 部署踩坑

### 3.1 第一个坑：PyTorch 版本选择

RTX 5090 是 Blackwell 架构（SM 12.0），这是一个 2025 年才出现的新架构。

**结论**：必须使用 PyTorch >= 2.7，因为只有从 2.7 开始才支持 CUDA 13.x 和 SM 12.0 的 kernel。

我们最终使用的是 PyTorch 2.11.0+cu130。安装命令：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3.2 第二个坑：CUDA 版本不匹配导致编译失败

系统安装的 CUDA toolkit 是 13.1，而 PyTorch 自带的 CUDA runtime 是 12.8。当编译 flash-attn 等 CUDA 扩展时，PyTorch 的 `cpp_extension.py` 会严格校验两者版本，版本不一致直接报错：

```
RuntimeError: The detected CUDA version (13.1) mismatches the version that was used 
to compile PyTorch (12.8). Please make sure to use the same CUDA versions.
```

**解决方案**：Monkey-patch 绕过版本检查：

```python
from torch.utils import cpp_extension as ext
ext._check_cuda_version = lambda *a, **k: None
```

这个 patch 是安全的——版本小幅不一致（12.8 vs 13.1）在实际运行中不会有问题，PyTorch 的检查过于保守了。

### 3.3 第三个坑：Flash Attention 源码编译

flash-attn 没有预编译的 SM 12.0 wheel，必须从源码编译。但 RTX 5090 只有 32GB VRAM，编译时 nvcc 的并行度开太高会 OOM（是的，编译 CUDA 代码也会吃显存）。

```bash
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=2 pip install flash-attn==2.8.3 --no-build-isolation
```

关键参数：
- `TORCH_CUDA_ARCH_LIST="12.0"` — 只编译 SM 12.0 的 kernel，不编译其他架构
- `MAX_JOBS=2` — 限制并行编译任务数，避免显存 OOM
- `--no-build-isolation` — 使用当前环境的 PyTorch，不创建隔离环境

编译时间大约 20-30 分钟。

### 3.4 第四个坑：transformers 版本兼容性

wall-x 依赖 HuggingFace transformers，但不同版本之间有 breaking change：

- `transformers >= 4.55` 才有 `AttentionInterface`（wall-x 需要）
- `transformers >= 5.x` 移除了一些旧的 import 路径，比如 `is_flash_attn_greater_or_equal_2_10` 从 `transformers.modeling_flash_attention_utils` 移走了
- `rope_type` 默认值从 `"default"` 改成了其他值

每次报 `ImportError` 都得去翻 transformers 的 changelog，手动加 `try/except` 兼容。

### 3.5 第五个坑：模型下载

HuggingFace 在国内被墙，需要使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download wall-oss/wall-oss-flow --local-dir ./models/wall-oss-flow
```

模型约 8GB，下载速度取决于镜像站的心情。

### 3.6 RTX 5090 最终成果

```
Output shape: [1, 50, 153715] ✓
NaN: False ✓
Inf: False ✓
Peak VRAM: ~15.7 GB
单次推理延迟: 14.2 ms
```

---

## 四、Jetson AGX Orin 部署踩坑

如果说 5090 的部署主要是版本兼容问题，那 Orin 还需要额外处理平台差异。

### 4.1 第一个坑：标准 PyTorch wheel 不能用

你可能以为 `pip install torch` 加个 `--index-url cu126` 就行了？

**实际上不行。**

标准 PyTorch 的 cu126 wheel 确实有 aarch64 版本，也能装上，`torch.cuda.is_available()` 也返回 `True`。但只要一跑任何 CUDA kernel，立刻报错：

```
CUDA error: no kernel image is available for execution on the device
```

原因：标准 wheel 的 `TORCH_CUDA_ARCH_LIST` 是 `['sm_50', 'sm_80', 'sm_86', 'sm_89', 'sm_90']`，**没有 SM 8.7**。Jetson AGX Orin 的 GPU 是 SM 8.7，一个 Jetson 平台专属的 compute capability，标准 wheel 不管它。

**解决方案**：使用 NVIDIA 官方为 Jetson 定制编译的 PyTorch wheel：

```bash
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
```

这个 wheel 的 arch list 包含 `['sm_70', 'sm_72', 'sm_75', 'sm_80', 'sm_86', 'sm_87', 'compute_87']`，完美覆盖 Orin。

**教训**：Jetson 上永远不要用 PyPI 的标准 PyTorch，必须用 NVIDIA 的定制版本。

### 4.2 第二个坑：NumPy 版本冲突

Jetson 定制的 torch 2.5 是基于 NumPy 1.x 的 C ABI 编译的。但 pip 默认安装 NumPy 2.x：

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**解决方案**：
```bash
pip install 'numpy<2'  # 降级到 1.26.4
```

### 4.3 第三个坑：torchvision 必须源码编译

PyPI 上的 torchvision 0.20 是配合标准 torch 2.6 编译的，和 Jetson 的 torch 2.5 不兼容：

```
RuntimeError: operator torchvision::nms does not exist
```

**解决方案**：从源码编译 torchvision：
```bash
git clone --branch release/0.20 https://github.com/pytorch/vision
cd vision && python3 setup.py install
```

编译大约需要 15-20 分钟。

### 4.4 第四个坑：torch.distributed 不存在

这个坑比较隐蔽。Jetson 版本的 PyTorch 是精简版，**没有 `torch.distributed` 模块**：

```
ModuleNotFoundError: No module named 'torch._C._distributed_c10d'
```

wall-x 的代码在很多地方 import 了 FSDP 和 distributed 相关的模块，即使推理时根本用不到。三个文件需要打补丁：

**vla_mixin.py** — FSDP import：
```python
# 原始代码
from torch.distributed.fsdp import MixedPrecision as MP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 修改后
try:
    from torch.distributed.fsdp import MixedPrecision as MP
except (ImportError, ModuleNotFoundError):
    MP = None
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except (ImportError, ModuleNotFoundError):
    FSDP = None
```

**action_head.py** — torch.distributed.is_initialized()：
```python
# 原始代码
def print_rank_last(message):
    if torch.distributed.is_initialized():
        ...

# 修改后
def print_rank_last(message):
    try:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
                print(message, flush=True)
        else:
            print(message, flush=True)
    except AttributeError:
        print(message, flush=True)
```

**教训**：Jetson 上的 PyTorch 缺很多桌面版理所当然存在的模块。写推理代码时，所有训练相关的 import 都应该用 `try/except` 保护。

### 4.5 第五个坑：Flash Attention 编译不过

Flash Attention 2 在 ARM aarch64 + SM 8.7 的环境下编译问题较多。我们尝试了源码编译 `flash-attn`，但在 Jetson 的 torch 2.5 + CUDA 12.6 环境下未能成功通过编译。

**解决方案**：直接使用 PyTorch 原生的 SDPA（Scaled Dot Product Attention）作为替代。在模型的 `config.json` 中设置：

```json
{
  "_attn_implementation": "sdpa"
}
```

同时把 flash_attn 的 import 用 try/except 包裹：
```python
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
```

SDPA 的精度和 Flash Attention 完全一致，只是在长序列场景下性能稍差。对于 batch=1 短序列推理来说完全够用。

> **但值得注意的是**：从 5.7 节的 profiling 数据可以看到，Orin 上 SDPA 的 Softmax 没有融合进 attention kernel（独立 `cunn_SoftMaxForward` 耗时 746ms，比 5090 慢 57 倍）。这说明如果能让 Flash Attention 2 在 Orin 上跑起来，可能会有显著的性能提升。这也是我们后续要尝试的方向。

### 4.6 第六个坑：wallx_csrc CUDA 扩展编译

wall-x 有自定义的 CUDA kernel（wallx_csrc），编译时需要指定正确的 GPU 架构，并且设置 LD_LIBRARY_PATH：

```bash
TORCH_CUDA_ARCH_LIST='8.7' pip install .
export LD_LIBRARY_PATH=/path/to/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

不设置 LD_LIBRARY_PATH 的话，运行时会报找不到 `libc10.so`。

### 4.7 第七个坑：nvidia-smi 几乎不可用

习惯了在桌面 GPU 上用 `nvidia-smi` 看显存、GPU 利用率、功耗？在 Orin 上统统看不到：

```
| GPU  Name: Orin (nvgpu)           N/A |
| Fan  Temp   Perf   Pwr:  N/A / N/A   |
| Memory-Usage: Not Supported           |
| GPU-Util: N/A                         |
```

`nvidia-smi` 在 Orin 上虽然能运行，但几乎所有关键字段都是 **N/A** 或 **Not Supported**。原因是 Orin 是 SoC（片上系统），GPU 和 CPU 共享 LPDDR5 统一内存，没有独立 VRAM，nvidia-smi 的显存监控接口不支持。

Orin 上的替代工具是 **`tegrastats`**：

```
RAM 10936/62841MB
GR3D_FREQ 14%          ← GPU 频率利用率
gpu@56.9°C             ← GPU 温度
VDD_GPU_SOC 3679mW     ← GPU+SoC 功耗
```

但 `tegrastats` 也有局限：
- **没有 GPU 显存单独统计**（和 CPU 共享 RAM，无法区分）
- **`GR3D_FREQ` 只是 GPU 整体频率利用率**，不能区分 CUDA Core 和 Tensor Core 的使用情况
- **没有进程级 GPU 使用信息**

**解决方案**：用 PyTorch 的 CUDA 内存追踪 API 来监控显存占用：

```python
torch.cuda.max_memory_allocated()   # 峰值 GPU 显存分配
torch.cuda.mem_get_info()           # 当前 free/total GPU 内存
```

这些 API 在 Orin 上正常工作，因为它们追踪的是 PyTorch CUDA allocator 的内部分配，不依赖 nvidia-smi。我们 benchmark 脚本中的 `peak GPU: 15.74 GB` 就是用这个 API 获取的。

### 4.8 Orin 最终成果

```
Output shape: [1, 50, 153715] ✓
NaN: False ✓
Inf: False ✓
Peak GPU mem: ~15.7 GB (共享内存)
单次推理延迟: 115.1 ms
```

---

## 五、性能对比：RTX 5090 vs Jetson AGX Orin

在两个平台上使用完全相同的测试条件进行对比。

### 5.0 测试方法论

**测试工具**：自研 `test_vqa_bench.py` 脚本（[scripts/test_vqa_bench.py](scripts/test_vqa_bench.py)），基于 PyTorch + wall-x 推理接口。

**测试配置**：
- 模型：wall-oss-flow (3B 参数)
- 精度：**bfloat16**（两个平台一致，均使用 Tensor Core 加速 bf16 矩阵运算）
- Attention：RTX 5090 使用 Flash Attention 2.8.3，Orin 使用 SDPA
- 推理方式：`model.generate()`，max_new_tokens=128
- 预热：1 次
- 重复：每个测试 3 次取平均
- 计时：使用 `torch.cuda.synchronize()` + `time.perf_counter()` 精确计时
- 显存监控：`torch.cuda.max_memory_allocated()`（Orin 上 nvidia-smi 不可用，详见 4.7 节）

**测试数据**：8 张机器人桌面操控场景图片（640x480），每张图 3 个 VQA 问题，共 24 个测试用例：
- Q1: "Describe what you see in this image."
- Q2: "What objects are on the table?"
- Q3: "What action should the robot take?"

### 5.1 Fake Inference 延迟（纯前向推理，随机 tensor 输入）

| 指标 | RTX 5090 | Jetson AGX Orin | 倍数差 |
|------|----------|----------------|--------|
| **平均延迟** | **14.2 ms** | **115.1 ms** | 8.1x |
| 中位延迟 | 14.2 ms | 114.3 ms | 8.0x |
| 最小延迟 | 14.2 ms | 113.9 ms | 8.0x |
| 最大延迟 | 14.4 ms | 118.3 ms | 8.2x |

### 5.2 VQA 多图推理性能总览

| 指标 | RTX 5090 | Jetson AGX Orin | 倍数差 |
|------|----------|----------------|--------|
| 测试用例数 | 24 | 24 | — |
| **平均延迟** | **1,747 ms** | **13,898 ms** | **8.0x** |
| 中位延迟 | 2,069 ms | 16,279 ms | 7.9x |
| 最小延迟 | 326 ms | 2,744 ms | 8.4x |
| 最大延迟 | 2,076 ms | 16,811 ms | 8.1x |
| **平均 tok/s** | **59.8** | **7.5** | **8.0x** |
| 模型加载时间 | ~15 s | ~173 s | 11.5x |
| Peak GPU 显存 | 15.74 GB | 15.74 GB | — |

### 5.3 VQA 逐图对比（每张图 3 个问题的平均值）

| 测试图片 | 5090 延迟 (ms) | 5090 tok/s | Orin 延迟 (ms) | Orin tok/s | 加速比 |
|----------|---------------|------------|---------------|------------|--------|
| blocks_and_plates.png | 2,058 | 62.2 | 16,257 | 7.8 | 7.9x |
| dual_arm_robot.png | 1,738 | 61.0 | 13,904 | 7.6 | 8.0x |
| fruits_on_table.png | 1,501 | 58.6 | 12,671 | 7.2 | 8.4x |
| real_tabletop_1.jpg | 1,493 | 56.4 | 12,053 | 7.2 | 8.1x |
| real_tabletop_2.jpg | 2,049 | 61.6 | 15,064 | 7.7 | 7.3x |
| real_tabletop_3.jpg | 1,501 | 57.8 | 11,723 | 7.0 | 7.8x |
| robot_gripper.png | 1,747 | 60.5 | 14,977 | 7.8 | 8.6x |
| stacking_task.png | 1,906 | 61.3 | 14,868 | 7.8 | 7.8x |

### 5.4 显存占用

| 指标 | RTX 5090 | Jetson AGX Orin |
|------|----------|-----------------|
| Peak GPU 显存 | 15.74 GB | 15.74 GB |
| 总可用 | 32 GB (独立 GDDR7) | 61.4 GB (共享 LPDDR5) |
| 显存占比 | 49% | 25.6% |
| 显存监控方式 | nvidia-smi + PyTorch API | **仅 PyTorch API**（nvidia-smi 不可用） |

两个平台的峰值显存占用完全一致（15.74 GB），这说明模型本身的 bf16 参数和 KV Cache 内存开销是固定的，与硬件无关。

### 5.5 Attention 机制与 Tensor Core 使用

| 项目 | RTX 5090 | Jetson AGX Orin |
|------|----------|----------------|
| Attention 实现 | Flash Attention 2.8.3 | SDPA (PyTorch 原生) |
| Tensor Core 代数 | 第5代 (Blackwell) | 第3代 (Ampere) |
| 运算精度 | bf16 Tensor Core | bf16 Tensor Core |
| bf16 算力 | ~1,677 TFLOPS | ~138 TFLOPS |
| 显存带宽 | ~1,792 GB/s (GDDR7) | ~205 GB/s (LPDDR5) |

**关于 Tensor Core**：两个平台都在使用 Tensor Core。模型以 bf16 精度运行，PyTorch 底层的 cuBLAS 在做矩阵乘法时会自动调度 Tensor Core。Flash Attention 的 CUDA kernel 也是专门为 Tensor Core 设计的（使用 WMMA/MMA 指令）。但请注意，**Orin 上的 tegrastats 的 `GR3D_FREQ` 只能看到 GPU 整体利用率，无法区分 CUDA Core 和 Tensor Core 的使用比例**。

### 5.6 数据分析

**8 倍差距从哪来？**

1. **Tensor Core 算力差距**：RTX 5090 的 bf16 算力约 1,677 TFLOPS，Orin 约 138 TFLOPS，**算力差 ~12x**
2. **显存带宽差距**：RTX 5090 的 GDDR7 带宽约 1,792 GB/s，Orin 的 LPDDR5 约 205 GB/s，**带宽差 ~8.7x**
3. **LLM 推理是 memory-bound**：自回归生成的每个 token 都需要从内存读取全部模型参数，瓶颈在带宽而非算力。所以实际差距约 8x，更接近带宽比而非算力比
4. **Flash Attention vs SDPA**：Flash Attention 的内存访问模式更优，在 5090 上进一步减少 HBM 读写

**VQA 推理的 token 生成特征**：
- 大多数测试用例生成了 128 个 token（达到上限），此时 Orin 稳定在 ~7.8 tok/s，5090 稳定在 ~62 tok/s
- 短回答（如 `real_tabletop_3.jpg` 只生成 14 个 token）的 tok/s 反而更低（Orin 5.1, 5090 46.0），因为首 token 延迟（prefill）占比更大

**Orin 的 13.9 秒 VQA 延迟能接受吗？**

对于机器人控制来说，VQA 推理（128 token 文本生成）通常不在实时控制回路中使用。它更多用于场景理解和任务规划，13.9 秒是可以接受的。

真正需要低延迟的是 **动作预测（Action Prediction）**，即 Fake Inference 测试的场景：
- RTX 5090: 14.2 ms → **70 Hz** 控制频率
- Orin: 115.1 ms → **8.7 Hz** 控制频率

8.7 Hz 对于大多数非高速操控任务（抓取、搬运、桌面操作）是够用的。如果需要更高频率，可以考虑：
- **FP8 量化**（5090 的 Blackwell 原生支持，理论 ~2x 加速）
- **INT8 量化 + TensorRT**（Orin 上效果显著）
- `torch.compile()` kernel fusion

### 5.7 GPU 算子耗时分析（Nsight Systems profiling）

在两个平台上分别用 `nsys profile` 抓取了一次完整的 VQA 推理过程（单张 640×480 图片，bf16 精度，生成约 128 token），然后用 `nsys stats --report cuda_gpu_kern_sum` 按 kernel 类型汇总 GPU 时间。

**总体概览：**

| 指标 | RTX 5090 | Jetson AGX Orin | 倍数差 |
|------|----------|-----------------|--------|
| GPU kernel 总耗时 | 1,207 ms | 15,894 ms | 13.2x |
| kernel 实例总数 | 280,431 | 284,319 | — |
| 独立 kernel 类型数 | 119 | 124 | — |

两个平台跑的是同一个模型和代码，kernel 实例数几乎相同（28 万+），差异全在每个 kernel 的执行耗时上。

**按算子类别汇总 GPU 时间：**

| 算子类别 | 5090 耗时 | 5090 占比 | Orin 耗时 | Orin 占比 | 倍数差 |
|----------|-----------|-----------|-----------|-----------|--------|
| **GEMM/GEMV（线性投影）** | 654 ms | 54.2% | 10,991 ms | 69.2% | 16.8x |
| Elementwise（SiLU、add 等） | 195 ms | 16.1% | 1,404 ms | 8.8% | 7.2x |
| **Attention（fmha/flash）** | 148 ms | 12.3% | 435 ms | 2.7% | 2.9x |
| Copy / dtype 转换 | 89 ms | 7.4% | 1,361 ms | 8.6% | 15.3x |
| **Softmax** | 13 ms | 1.1% | **746 ms** | **4.7%** | 57.4x |
| RadixSort（MoE 路由排序） | 48 ms | 4.0% | 230 ms | 1.4% | 4.8x |
| CatArrayCopy | 33 ms | 2.7% | 435 ms | 2.7% | 13.2x |
| Reduce | 23 ms | 1.9% | 266 ms | 1.7% | 11.6x |
| MoE permute/unpermute | 4 ms | 0.3% | 26 ms | 0.2% | 6.5x |
| RoPE + Window index | < 0.1 ms | ~0% | < 0.2 ms | ~0% | — |

**两个平台 Top-5 最耗时 kernel：**

RTX 5090：

| # | Kernel | 占比 | 实例数 | 说明 |
|---|--------|------|--------|------|
| 1 | `gemvx::kernel<bf16,bf16>` | 27.9% | 18,144 | decoder 自回归 GEMV |
| 2 | `gemvx::kernel<bf16,float>` | 12.4% | 4,536 | decoder GEMV（fp32 累加） |
| 3 | `fmha_cutlassF_bf16_aligned` | 12.3% | 4,536 | SDPA memory-efficient attention |
| 4 | `gemvx::kernel<bf16,bf16>` 变体 | 5.9% | 9,198 | decoder GEMV |
| 5 | `RadixSortSingleTile` | 4.0% | 4,608 | MoE 路由排序 |

Jetson AGX Orin：

| # | Kernel | 占比 | 实例数 | 说明 |
|---|--------|------|--------|------|
| 1 | `ampere_bf16_s16816gemm_64x64` | 47.4% | 13,608 | decoder 矩阵乘（Tensor Core） |
| 2 | `gemv2T_kernel_val<bf16>` | 7.2% | 126 | prefill GEMV |
| 3 | `ampere_bf16_s16816gemm_128x64` | 6.9% | 9,072 | decoder 矩阵乘 |
| 4 | `cunn_SoftMaxForward` | 4.7% | 64 | 独立 Softmax（未融合） |
| 5 | `elementwise_kernel`（SiLU 等） | 4.2% | 9,280 | 激活函数 |

**关键发现：**

1. **GEMM/GEMV 是绝对主体。** 两个平台上线性投影都占了 54-69% 的 GPU 时间。28 层 decoder 每层 7 个投影矩阵（q/k/v/o + gate/up/down），每生成一个 token 都要全部跑一遍，这才是推理时间的核心来源。

2. **cuBLAS 在两个平台上选择了不同的 kernel。** 5090 上是 `gemvx`（矩阵-向量乘专用 kernel），Orin 上是 `s16816gemm`（Tensor Core GEMM tile）。同样是 batch=1 自回归推理，cuBLAS 在 SM 12.0 走了 GEMV 路径，在 SM 8.7 走的是 GEMM 路径。

3. **Softmax 在 Orin 上未融合，是一个独立瓶颈。** 5090 上 Softmax 仅 13ms（1.1%），被融合进了 CUTLASS memory-efficient attention kernel。Orin 上则是独立的 `cunn_SoftMaxForward`，耗时 746ms（4.7%），**差了 57 倍**。这是 SDPA 在不同 SM 版本上的 dispatch 差异。

4. **Attention 本身占比并不高。** 5090 上 12.3%（148ms），Orin 上 2.7%（435ms）。batch=1 时序列长度有限（~400 token），attention 还远没有成为瓶颈。

5. **wall-x 自定义 CUDA 算子（MoE permute + RoPE + window）加起来不到 0.5%。** 推理时间的大头完全在标准库（cuBLAS、SDPA）上。

**对 5.6 节 "8 倍差距" 的补充：**

- GPU kernel 总时间比是 **13.2x**（1,207ms vs 15,894ms），大于端到端延迟比 8.0x。差值来自 CPU 端的 Python 开销和 CUDA API 调用（5090 上 cudaLaunchKernel 平均 2.4μs，Orin 上 18.9μs）。
- GEMM/GEMV 的倍数差达到 **16.8x**，超过显存带宽比（8.7x），说明矩阵运算部分不完全是 memory-bound，也受到了 Tensor Core 算力差异的影响。
- Softmax 的 **57.4x** 差距最为显著，主要是融合 vs 非融合的实现差异，而非纯硬件算力差。这也是 Orin 上最有优化潜力的点之一。

**profiling 原始数据：**

| 文件 | 平台 | 大小 |
|------|------|------|
| `vqa_nsys_5090_20260424.nsys-rep` | RTX 5090 (SM 12.0) | 18 MB |
| `vqa_nsys_orin_20260424.nsys-rep` | Orin (SM 8.7) | 58 MB |
| `vqa_ncu_5090_20260424_load.ncu-rep` | RTX 5090 | 77 MB |
| `vqa_ncu_orin_20260424_load.ncu-rep` | Orin | 36 MB |

测试脚本：`scripts/profile_vqa.py`。ncu 文件当前抓取的是前 50 个 kernel（模型加载阶段为主），后续如需分析推理 kernel 的 Tensor Core 利用率和内存带宽效率，需要用 `--launch-skip` 跳过加载阶段 kernel。

---

## 六、踩坑总结对比

| 坑点 | RTX 5090 | Jetson AGX Orin |
|------|----------|-----------------|
| PyTorch 安装 | 标准 pip，指定 cu130 | 必须用 NVIDIA Jetson 定制 wheel |
| CUDA 版本限制 | 驱动新，支持 CUDA 13.x | 被 JetPack SDK 锁定在 CUDA 12.6 |
| Flash Attention | 源码编译，限制 MAX_JOBS | 编译未通过，用 SDPA 替代 |
| torch.distributed | 正常可用 | 不存在，需要 try/except 补丁 |
| torchvision | pip install 直装 | 必须源码编译 |
| NumPy | 无限制 | 必须 < 2.0 |
| 编译 CUDA 扩展 | ARCH=12.0 + monkey-patch | ARCH=8.7 + LD_LIBRARY_PATH |
| GPU 监控 | nvidia-smi 完整信息 | nvidia-smi 全是 N/A，用 tegrastats |
| 显存监控 | nvidia-smi + PyTorch API | 仅 PyTorch API (`torch.cuda.max_memory_allocated`) |
| Profiling | nsys 直接用，ncu 需要 sudo | nsys 直接用，ncu 需要 sudo + 完整路径 |
| GEMM kernel 路径 | cuBLAS `gemvx`（GEMV 专用） | cuBLAS `ampere_bf16_s16816gemm`（GEMM tile） |
| Softmax 融合 | 融合进 attention kernel（13ms, 1.1%） | 独立 `cunn_SoftMaxForward`（746ms, 4.7%） |
| 总部署时间 | ~2 小时 | ~4 小时 |

---

## 七、给后来者的建议

### 7.1 RTX 5090 / Blackwell 部署

1. **PyTorch >= 2.7 是硬性要求**，低于这个版本不支持 SM 12.0
2. 编译 CUDA 扩展遇到版本检查失败，**大胆 monkey-patch**，差 0.x 的版本不影响运行
3. Flash Attention 编译时 **一定限制 MAX_JOBS**，否则 OOM
4. 只编译你需要的 `TORCH_CUDA_ARCH_LIST`，不要编译全架构

### 7.2 Jetson AGX Orin 部署

1. **永远不要用 PyPI 的标准 PyTorch**，去 NVIDIA 的 Jetson 专用仓库下载
2. JetPack 版本决定了你能用的 CUDA 上限，**不要指望在 Orin 上用最新 CUDA**
3. Flash Attention 目前在 Orin 上编译不过，**直接用 SDPA**，短序列下性能差距可忽略。但 Softmax 未融合的问题值得关注（见 5.7 节）
4. 所有 `torch.distributed` 相关代码都要用 `try/except` 保护
5. torchvision 必须从源码编译，PyPI 的 wheel 和 Jetson torch 不兼容
6. **设置 `LD_LIBRARY_PATH`**，否则自定义 CUDA 扩展找不到 PyTorch 的动态库
7. **别指望 nvidia-smi**，用 `tegrastats` 看 GPU 利用率和功耗，用 `torch.cuda.max_memory_allocated()` 看 GPU 显存
8. **Softmax 未融合是 Orin 上最大的单点瓶颈**（占 GPU 时间 4.7%，绝对时间 746ms，比 5090 慢 57 倍）。如果后续 PyTorch 升级了 SDPA 在 SM 8.7 上的 dispatch 策略，这个差距有望大幅缩小

### 7.3 性能优化方向（基于 profiling 数据）

| 方向 | 预期收益 | 适用平台 | 说明 |
|------|----------|----------|------|
| INT8/INT4 量化 | GEMM 加速 2-4x | 两个平台 | GEMM/GEMV 占 54-69%，量化收益最大 |
| KV Cache 优化 | 减少 attention 内存访问 | 两个平台 | 当前 attention 占 2.7-12.3%，长序列场景收益更大 |
| torch.compile / CUDA Graph | 减少 kernel launch 开销 | 两个平台 | Orin 上 cudaLaunchKernel 平均 18.9μs（5090 上 2.4μs） |
| Softmax 融合 | Orin 上 Softmax 从 746ms → ~13ms | Orin | 最大单点优化潜力，需解决 Flash Attention 编译或等待 SDPA 更新 |
| FP8 推理 | GEMM 加速 ~2x | 仅 5090 | Blackwell 原生支持 FP8 Tensor Core |
| TensorRT | 端到端优化 | Orin 优先 | 图优化 + 算子融合 + 量化一体化 |

### 7.4 通用建议

1. **HuggingFace 镜像**：国内必须设置 `HF_ENDPOINT=https://hf-mirror.com`
2. **config.json 修改**：记得加 `pad_token_id` 和 `_attn_implementation`
3. **先跑 fake inference 再上真实数据**：用随机 tensor 验证整个流程，比用真实图片调试快得多
4. **SSH 稳定性**：远程调试时建议用 SSH key 而非密码，避免频繁断连

---

## 八、结语

在 2026 年的今天，把一个 3B VLA 模型同时部署到最新的桌面 GPU 和边缘端 Jetson 上，仍然需要处理不少兼容性问题。

RTX 5090 的 14ms 动作预测延迟让人兴奋——这意味着 70Hz 的控制频率，完全可以做精细操控。VQA 场景下 59.8 tok/s 的生成速度也非常流畅。而 Orin 虽然慢了 8 倍（动作预测 115ms，VQA 生成 7.5 tok/s），但对于大多数机器人任务来说已经够用，而且它只有巴掌大小，可以直接装在机器人身上。

**Nsight Systems profiling 让差距有了更具体的解释。** 两个平台跑的 28 万+ kernel 实例几乎完全对应，但 GPU kernel 总时间比达到 13.2x。其中 GEMM/GEMV（线性投影）独占 54-69% 的 GPU 时间，是推理延迟的绝对主体。Orin 上还存在 Softmax 未融合的问题（独立 kernel 耗时 746ms，比 5090 慢 57 倍），这是一个明确的可优化点。

**主要的痛点在于生态差异。** PyTorch 在 Jetson 上的体验和桌面端还有不小的差距。没有 torch.distributed，没有 Flash Attention，NumPy 版本受限，torchvision 要源码编译，nvidia-smi 看不到有用的信息，SDPA 的 Softmax 融合策略也和桌面端不同——这些都需要逐一适配。

两个平台都在使用 Tensor Core 做 bf16 矩阵运算，但 cuBLAS 在不同 SM 上选择了不同的 kernel 路径（5090 用 GEMV，Orin 用 GEMM tile）。未来如果引入 INT8 量化，GEMM/GEMV 占比 54-69% 意味着量化可以直接影响一半以上的 GPU 时间；加上 Softmax 融合和 TensorRT 优化，Orin 上还有不小的加速空间。

希望这篇文章能帮到正在做类似工作的你。如果你也有 Jetson 或新架构 GPU 的部署经验，欢迎在评论区交流。

---

## 系列预告

本文是 **wall-x 机器人大模型部署系列** 的第一篇。后续文章计划：

**第二篇：当算子逼近硬件极限——一次 Orin Profiling 引发的具身智能实时系统思考**
- FA2 在 Orin 上编译通了，快 27%，但深度 profiling 发现：GEMM 已触达带宽天花板（cuBLAS 利用率 72%），67% 时间是 Python 框架空转
- 三条 Flash Attention 路线对比（开源 FA2 / TRT-LLM cubin / FlashInfer），cuDNN SDPA 绕过 attention_mask 后接近 TRT-LLM 性能
- 核心观点：具身智能的瓶颈不在 model，而在 runtime——VLA 是有损压缩的物理模拟器，刷新率比单次精度更重要

**第三篇：C++ 推理引擎——从 Python 到生产部署**
- 用 C++ / libtorch 替代 Python 推理，消除 Python GIL 和解释器开销
- TensorRT 端到端优化
- 面向机器人实时控制的推理 pipeline 设计
- 边缘端的模型部署和云端 "token first" 的服务化思路完全不同——Orin 上跑的本质是一个 **model set 的 runtime**，多个模型（视觉编码器、语言模型、动作头）需要在同一个进程内协同调度，而不是拆成微服务各自吐 token

如果你对这些内容感兴趣，欢迎**关注**，我会持续更新。

---

*测试环境：wall-oss-flow 3B 模型，bfloat16 精度，batch_size=1。测试工具：自研 test_vqa_bench.py（VQA benchmark）和 profile_vqa.py（nsys/ncu profiling），基于 PyTorch + torch.cuda 计时。VQA benchmark 使用 8 张 640x480 机器人场景图片 × 3 个问题 = 24 个测试用例。Profiling 使用 Nsight Systems 2025.5.2 采集，nsys stats 汇总分析。显存通过 `torch.cuda.max_memory_allocated()` 监控（Orin 上 nvidia-smi 不可用）。2026 年 4 月实测数据。*
