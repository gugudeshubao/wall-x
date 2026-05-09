# FlashInfer on Orin: 实测记录

> 日期：2026-04-29  
> 目的：验证 `FlashInfer` 在 Jetson AGX Orin 64GB (`sm87`, `aarch64`) 上的实际可用性和性能上限，而不是只看官方文档口径。

---

## 1. 环境

- 机器：Jetson AGX Orin 64GB
- 架构：`aarch64`
- GPU：`sm87`
- Python：`3.10.12`
- PyTorch：`2.5.0a0+872d972e41.nv24.08`
- CUDA：`12.6`
- 测试环境：`/data/wy/wall-x/venv`

---

## 2. 安装结论

### 2.1 不能直接往当前 venv 里 `pip install flashinfer-python`

直接执行：

```bash
pip install flashinfer-python
```

会尝试拉新的：

- `torch 2.11`
- `cuda 13.x` 相关依赖
- 一整套新的 NVIDIA Python 包

这和当前 `wall-x` 的：

- `torch 2.5`
- `CUDA 12.6`

主环境不兼容，不适合直接装进现有 venv。

### 2.2 可行方案：隔离安装

这次实测采用的是：

```bash
rm -rf /tmp/flashinfer_orin_nodeps && mkdir -p /tmp/flashinfer_orin_nodeps
source /data/wy/wall-x/venv/bin/activate

python -m pip install --no-deps --target /tmp/flashinfer_orin_nodeps flashinfer-python

python -m pip install --target /tmp/flashinfer_orin_nodeps \
  apache-tvm-ffi click cuda-tile einops ninja nvidia-cudnn-frontend \
  nvidia-ml-py packaging requests tabulate tqdm

# 关键：不要让临时目录里的 numpy 2.x 覆盖当前环境的 numpy 1.26
rm -rf /tmp/flashinfer_orin_nodeps/numpy \
       /tmp/flashinfer_orin_nodeps/numpy.libs \
       /tmp/flashinfer_orin_nodeps/numpy-*.dist-info

PYTHONPATH=/tmp/flashinfer_orin_nodeps:$PYTHONPATH python -c "import flashinfer; print(flashinfer.__version__)"
```

结论：

- `flashinfer-python 0.6.9` 可导入
- 不需要替换当前 PyTorch 主包
- 但这仍然只是**隔离试用**，不代表适合直接并入正式依赖栈

---

## 3. 平台识别结果

`flashinfer show-config` 在 Orin 上能正常运行，并识别到：

- `FlashInfer version: 0.6.9`
- `Torch version: 2.5.0a0+872d972e41.nv24.08`
- `CUDA runtime available: Yes`
- `FLASHINFER_CUDA_ARCH_LIST={(8, 7)}`
- `FLASHINFER_CUDA_VERSION=12.6`
- `CUDA_HOME=/usr/local/cuda`
- `NVCC found: Yes`

同时它会开始准备 JIT / cubin artifact。

这说明：

> **FlashInfer 在 Orin 上不是“第一步就不支持”，而是至少能识别平台并进入正常初始化流程。**

---

## 4. 基础功能测试

### 4.1 `activation.silu_and_mul`

测试：

- 输入：`[4, 16]` bf16
- 输出：`[4, 8]`
- 参考：`torch.nn.functional.silu(x[..., :8]) * x[..., 8:]`

结果：

- 可运行
- `max_err = 0.00209`

这说明最基础的 elementwise fused op 在 Orin 上是可用的。

### 4.2 `prefill.single_prefill_with_kv_cache`

先做 shape 探测，以下 shape 都可运行：

- `(420, 420, 16, 128)`
- `(456, 456, 16, 128)`
- `(1, 420, 16, 128)`
- `(1, 456, 16, 128)`
- `(64, 64, 8, 64)`
- `(128, 128, 8, 64)`
- `(128, 128, 16, 64)`
- `(128, 128, 16, 128)`

但也观察到：

- 很小的某些 shape（例如 `(8, 8, 4, 32)`）会报 `Invalid configuration`

所以更准确的结论不是“全支持”，而是：

> **Orin 上可以跑，但支持面是“部分 kernel / 部分 shape 可用”。**

---

## 5. 性能实测

### 5.1 Prefill attention

测试 shape：

- `Q = 420`
- `KV = 420`
- `num_heads = 16`
- `head_dim = 128`
- dtype = `bf16`
- causal = `True`

对比对象：

- `flashinfer.prefill.single_prefill_with_kv_cache`
- `torch.nn.functional.scaled_dot_product_attention`

测试方式：

- warmup 20 次
- 统计 100 次平均时间
- GPU 前后 `torch.cuda.synchronize()`

结果：

| 测试 | 形状 | FlashInfer | PyTorch SDPA | 加速比 | 误差 |
|------|------|-----------|--------------|--------|------|
| prefill attention | `420 x 420 x 16 x 128` | **0.082 ms** | **0.500 ms** | **6.12x** | `max_err=0.015625` |

进一步按 `wall-x VQA` 更贴近的 **GQA** 形状测试：

- `Q = 456`
- `KV = 456`
- `q_heads = 16`
- `kv_heads = 2`
- `head_dim = 128`

结果：

| 测试 | 形状 | FlashInfer | PyTorch SDPA | 加速比 | 误差 |
|------|------|-----------|--------------|--------|------|
| prefill attention (GQA) | `456 x 456 x 16q / 2kv x 128` | **0.0634 ms** | **0.5013 ms** | **7.90x** | `max_err=0.015625` |

这说明：

> **至少在 wall-x 更关心的 prefill 级 attention shape 上，FlashInfer 在 Orin 上不仅能跑，而且性能明显优于 PyTorch SDPA。**

### 5.2 Decode API

`flashinfer.decode.single_decode_with_kv_cache` 的接口能正常导入；  
测试时确认它的输入 layout 约束是：

- `q`: 2D tensor，形状 `[num_qo_heads, head_dim]`
- `k/v`: 3D tensor，形状 `[kv_len, num_kv_heads, head_dim]`

实测 shape：

- `Q = 1`
- `KV = 420`
- `num_heads = 16`
- `head_dim = 128`
- dtype = `bf16`

对比对象：

- `flashinfer.decode.single_decode_with_kv_cache`
- `torch.nn.functional.scaled_dot_product_attention`

注意：

- 这里的 PyTorch 参考实现 **不能** 用 `is_causal=True`
- 因为 decode 场景下单个 query token 对整段历史 KV 是“全可见”的
- 对应参考是 `q.unsqueeze(1)` vs `[H, KV, D]` 的 **non-causal SDPA**

结果：

| 测试 | 形状 | FlashInfer | PyTorch SDPA | 加速比 | 误差 |
|------|------|-----------|--------------|--------|------|
| decode attention | `1 x 420 x 16 x 128` | **0.0563 ms** | **0.2046 ms** | **3.63x** | `max_err=0.001953` |

再按 `wall-x VQA` 更贴近的 **GQA decode** 形状测试：

- `Q = 1`
- `KV = 456`
- `q_heads = 16`
- `kv_heads = 2`
- `head_dim = 128`

结果：

| 测试 | 形状 | FlashInfer | PyTorch SDPA | 加速比 | 误差 |
|------|------|-----------|--------------|--------|------|
| decode attention (GQA) | `1 x 456 x 16q / 2kv x 128` | **0.0525 ms** | **0.2084 ms** | **3.97x** | `max_err=0.001465` |

这说明：

> **Orin 上 FlashInfer 不只是 prefill 有收益，单 token decode attention 也能稳定跑通，并且快于 PyTorch SDPA。**

### 5.3 更贴近 Flow Action postfix 的 non-causal GQA

为了判断它对 `wall-x` 的 `Flow Action postfix` 路径有没有意义，还额外测了一组更接近 ODE/postfix 的 attention 形状：

- `Q = 32`
- `KV = 488`
- `q_heads = 16`
- `kv_heads = 2`
- `head_dim = 128`
- `causal = False`

这组形状对应的是：

- query 端是一段较短的 postfix token block
- key/value 端是 prefix cache + postfix 的整段上下文
- query 内部允许双向可见

结果：

| 测试 | 形状 | FlashInfer | PyTorch SDPA | 加速比 | 误差 |
|------|------|-----------|--------------|--------|------|
| postfix attention (GQA, non-causal) | `32 x 488 x 16q / 2kv x 128` | **0.0788 ms** | **0.2358 ms** | **2.99x** | `max_err=0.003906` |

这说明：

> **FlashInfer 对 wall-x 的 Flow Action postfix attention 也不是完全无效，只是相比 VQA prefill / decode，收益幅度更温和。**

### 5.4 Python 端到端 VQA spike

为了判断这些 microbench 收益能不能真正转成 `wall-x` 的端到端收益，还做了一个不改模型源码的 Python spike：

- 脚本：`scripts/bench_flashinfer_vqa_spike.py`
- 做法：只 monkey patch `Qwen2_5_VLSdpaAttention.forward()`
- 条件：
  - batch = 1
  - attention_mask 全 1
  - prefill (`q_len > 1`) -> FlashInfer prefill
  - decode (`q_len == 1`) -> FlashInfer decode
  - 其余情况回退原始 SDPA

测试配置：

- 图像：`test_images/fruits_on_table.png`
- `max_new_tokens = 20`
- baseline：Python `sdpa`
- 对照：`flashinfer-patched-sdpa`

结果：

| 模式 | 平均延迟 | 吞吐 | 峰值显存 | token 是否一致 |
|------|---------|------|---------|----------------|
| Python SDPA | **3295.0 ms** | **6.07 tok/s** | **8.19 GB** | — |
| Python + FlashInfer spike | **3017.5 ms** | **6.63 tok/s** | **8.19 GB** | **一致** |

结论：

- 端到端加速约 **1.09x ~ 1.10x**
- 生成 token 与 `sdpa` baseline 一致
- 峰值显存没有明显恶化（单独运行时约 `8.19 GB`）

这说明：

> **FlashInfer 在 Orin 上的 attention kernel 确实快，但端到端 VQA 不是纯 attention 受限，所以最终收益显著小于 microbench 的 3x~8x。**

也就是说，它是一个真实可用的优化方向，但不是“换上就 5x”的银弹。

---

## 6. 对 wall-x 的实际意义

### 6.1 能做的事

从这次实测看，FlashInfer 在 Orin 上最值得关注的是：

- prefill attention
- decode attention
- sampling
- 一部分 fused activation / norm

也就是：

> **它更像一个“attention / serving kernel 工具箱”，而不是一个完整替代你现有 C++ runtime 的方案。**

如果只从当前 `wall-x` 的 attention 形状出发，优先级大致是：

1. **VQA prefill attention**
   - `456 x 456 x 16q / 2kv x 128`
   - 实测 `7.90x`
   - 最值得优先替换
2. **VQA decode attention**
   - `1 x 456 x 16q / 2kv x 128`
   - 实测 `3.97x`
   - 次优先级，但同样有价值
3. **Flow Action postfix attention**
   - `32 x 488 x 16q / 2kv x 128`
   - 实测 `2.99x`
   - 可试，但收益相对前两项更低

### 6.2 当前限制

对 `wall-x` 这条线来说，现实限制也很明确：

- 官方主形态是 Python package + JIT / cubin 流程
- 直接装到现有 venv 会碰依赖栈
- Orin 上不是所有 shape 都能自动选到合法 kernel 配置
- 你现在主线是 `cpp_infer`，而 FlashInfer 更偏 Python runtime 集成

所以它当前更像：

- **可验证的性能 spike 方向**
- 不是马上可替换现有 `cpp_infer` attention 路径的现成生产方案

---

## 7. 当前判断

一句话总结：

> **FlashInfer 在 Orin 上的支持力度，已经足够支撑“导入成功 + 部分 kernel 可跑 + prefill attention 有明显加速”的结论；但它还没到“可以无脑并入 wall-x 主线依赖栈”的程度。**

如果后续要继续推进，建议顺序是：

1. 再把 `wall-x` 当前 `VQA decode` / `Flow Action postfix` 的真实 shape 映射到 FlashInfer API  
2. 看 `prefill` / `decode` 的调用封装成本是否低于继续手工维护现有 attention 路径  
3. 最后再决定是走 Python spike、还是研究 C++ runtime 层的接入方式

当前已经补到第 2 步，并得到一个重要中间结论：

> **值得继续，但更像“可拿 10% 级端到端收益”的工程优化，不像“attention kernel 换掉就几倍加速”的路线。**
