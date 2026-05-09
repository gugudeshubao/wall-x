# FlashInfer 接入 wall-x 的路径判断

> 这份笔记不讨论“FlashInfer 在 Orin 上能不能跑”，那部分见 [flashinfer_orin_eval.md](/Users/sam/project/github/wall-x/docs/flashinfer_orin_eval.md)。  
> 这里只回答一个更具体的问题：**如果要把 FlashInfer 接进 `wall-x` 当前主线，最适合切哪一段？**

---

## 1. 当前 C++ attention 路径长什么样

`cpp_infer` 当前 attention 主路径在：

- [attention.cpp](/Users/sam/project/github/wall-x/cpp_infer/src/attention.cpp)
- [kv_cache.h](/Users/sam/project/github/wall-x/cpp_infer/src/kv_cache.h)

现状：

1. `q_proj / k_proj / v_proj`
2. reshape 到：
   - `q`: `[batch, num_heads, seq, head_dim]`
   - `k/v`: `[batch, num_kv_heads, seq, head_dim]`
3. 走自定义多模态 RoPE
4. 写入 `KVCache`
5. 取出完整 `cached_k / cached_v`
6. **手动把 KV heads 展开成 query heads**（GQA expand）
7. 最后调用 `torch::scaled_dot_product_attention`

也就是说，当前实现本质上是：

> **PyTorch SDPA + 手工 GQA head expand + 自己维护 KV cache。**

---

## 2. FlashInfer API 要求什么形状

这次在 Orin 上实际验证下来的两个关键 API：

### 2.1 Prefill

`flashinfer.prefill.single_prefill_with_kv_cache`

输入形状：

- `q`: `[qo_len, num_qo_heads, head_dim]`
- `k`: `[kv_len, num_kv_heads, head_dim]`
- `v`: `[kv_len, num_kv_heads, head_dim]`

特点：

- **天然支持 GQA**，不需要先手工 expand `kv_heads -> q_heads`
- 支持 `causal=True/False`

### 2.2 Decode

`flashinfer.decode.single_decode_with_kv_cache`

输入形状：

- `q`: `[num_qo_heads, head_dim]`
- `k`: `[kv_len, num_kv_heads, head_dim]`
- `v`: `[kv_len, num_kv_heads, head_dim]`

特点：

- 单 query token 的 decode 专用接口
- 也天然支持 GQA

---

## 3. 和 wall-x 现有路径的 shape 差异

差异其实不在数学，而在 layout：

### 当前 C++ 路径

- `q`: `[B, Hq, S, D]`
- `k/v`: `[B, Hkv, T, D]`

### FlashInfer 期望

- prefill:
  - `q`: `[S, Hq, D]`
  - `k/v`: `[T, Hkv, D]`
- decode:
  - `q`: `[Hq, D]`
  - `k/v`: `[T, Hkv, D]`

因此接入动作大致是：

1. 把 batch=1 这一层明确下来
2. 把当前 `q.transpose/contiguous` 结果再转成 `seq-major`
3. 从 `KVCache` 里取 `k/v` 后，改成 `[seq, kv_heads, dim]`
4. **去掉手工 expand KV heads**
5. 调 FlashInfer prefill/decode

所以对接难点不是算法，而是：

> **当前 `cpp_infer` 是纯 C++ libtorch runtime，而 FlashInfer 当前主形态是 Python package + JIT/cubin 管理。**

---

## 4. 三段 attention 哪些最值得切

结合 Orin 实测，优先级如下。

### 4.1 第一优先级：VQA prefill

对应形状：

- `Q = 456`
- `KV = 456`
- `16q / 2kv / 128`

实测：

- FlashInfer: `0.0634 ms`
- PyTorch SDPA: `0.5013 ms`
- `7.90x`

为什么优先：

- 形状最规则
- 完全匹配 `single_prefill_with_kv_cache`
- 还能顺便去掉当前 C++ 里的 `kv head expand`

### 4.2 第二优先级：VQA decode

对应形状：

- `Q = 1`
- `KV = 456`
- `16q / 2kv / 128`

实测：

- FlashInfer: `0.0525 ms`
- PyTorch SDPA: `0.2084 ms`
- `3.97x`

为什么第二：

- 也很规则
- 直接命中 `single_decode_with_kv_cache`
- 单步收益没 prefill 那么大，但在 20 token 累积里仍然值钱

### 4.3 第三优先级：Flow Action postfix

对应近似形状：

- `Q = 32`
- `KV = 488`
- `16q / 2kv / 128`
- `causal = False`

实测：

- FlashInfer: `0.0788 ms`
- PyTorch SDPA: `0.2358 ms`
- `2.99x`

为什么第三：

- 也能接
- 但收益没有前两段高
- 而且这条线还和 ODE/postfix cache 管理绑得更紧

---

## 5. 哪些地方暂时不适合直接切

### 5.1 纯 C++ 运行时直接接 FlashInfer

这是当前最大的现实障碍。

原因不是 kernel 本身，而是工程形态：

- `cpp_infer`：纯 C++ / libtorch
- FlashInfer：Python package + JIT/cubin 管理 + `tvm_ffi`

所以短期里最现实的两条路是：

1. **先做 Python spike**
   - 在现有 Python benchmark 里把 SDPA attention 局部换成 FlashInfer
   - 先确认端到端收益
2. **再研究 C++ 接入**
   - 看能不能直接复用其 cubin / runtime
   - 或者只借鉴 shape/layout 和 kernel 选型思路

### 5.2 带复杂 custom mask 的路径

`wall-x` 训练/原始 Python 路径里有更复杂的 mask 逻辑，尤其是 `type1` token 的双向可见区域。  
当前 `cpp_infer` 之所以相对简单，是因为它已经把一些路径收敛成：

- VQA: 不传 `attention_mask`
- postfix: `is_causal=false`

所以现在能直接切的是这些**已经被 runtime 规整过的推理形状**，而不是所有原始训练时的 mask 语义。

---

## 6. 当前判断

在 Orin 上已经做过一个 Python 端到端 spike（见 [bench_flashinfer_vqa_spike.py](/Users/sam/project/github/wall-x/scripts/bench_flashinfer_vqa_spike.py)）：

- baseline: Python `sdpa`
- spike: 仅替换文本 decoder 的 prefill/decode attention 为 FlashInfer
- 图像：`fruits_on_table.png`
- `max_new_tokens = 20`

结果：

- `sdpa`: `3295.0 ms`
- `flashinfer spike`: `3017.5 ms`
- 端到端收益约 **1.10x**
- token 与 baseline 一致

这说明：

> **FlashInfer 对 wall-x 是“值得继续做”的方向，但当前更像 10% 级端到端优化，而不是 attention microbench 那种 3x~8x 会直接等比例转化到整机延迟的路线。**

如果只谈“下一步最务实的推进顺序”，我会建议：

1. **先在 Python 里做 VQA prefill/decode 的 FlashInfer spike**
   - 目标：确认端到端不是只有 microbench 快
2. **如果收益成立，再决定是否值得为 `cpp_infer` 做更深的 C++ 层接入**
3. **Flow Action postfix 可以跟进，但优先级低于 VQA prefill/decode**

一句话总结：

> **对 `wall-x` 来说，FlashInfer 最值得接的不是“所有 attention”，而是 `VQA prefill` 和 `VQA decode` 这两段最规则、最贴 API、收益也最大的 GQA attention。**
