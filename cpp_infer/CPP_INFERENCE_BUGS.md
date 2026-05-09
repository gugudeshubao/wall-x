# C++ 推理引擎 Bug 记录

> 在验证 C++ 推理引擎（wallx_infer）VQA 推理正确性时发现的三个 Bug。
> 这三个 Bug 均位于 RoPE（旋转位置编码）相关的代码路径中，导致所有生成 token 为 0（logits 全部 NaN）。

## 表现

使用 8 张真实测试图片运行 C++ VQA 推理，所有生成 token 均为 0。

**排查定位路径**：model → transformer → layer 0 → attention → RoPE kernel，逐层添加 NaN 检查，发现 NaN 首次出现在 RoPE kernel 的输出中。

---

## Bug 1：RoPE kernel cos/sin 索引越界

**文件**：`csrc/rope.cu`

**根因**：CUDA kernel 假设 cos/sin 张量的 layout 是 `[3, batch, seq_len, head_dim]`（每个 section 一个独立的 slice），使用 `section_idx` 作为偏移来索引。但实际上 `compute_rotary_emb()` 生成的 cos/sin layout 是 `[batch, seq_len, head_dim]`（三个 section 的值已经在 head_dim 维度上拼接在一起）。

当 section_idx = 1 或 2 时，kernel 会读取到 tensor 边界之外的内存，得到垃圾值或 NaN。

**修复前**：
```cuda
// kernel 中按 section 偏移索引 cos/sin
int cos_sin_idx = section_idx * batch_size * seq_len * head_dim +
                  batch_idx * seq_len * head_dim +
                  seq_idx * head_dim + cos_sin_d;
```

**修复后**：
```cuda
// 直接用 flat [batch, seq, head_dim] 索引
int cos_sin_idx = batch_idx * seq_len * head_dim +
                  seq_idx * head_dim + dim_idx;
```

> 同样的修复也应用于 backward kernel 中的 `paired_cos_sin_idx` 索引。

---

## Bug 2：cos/sin dtype 不匹配（float32 vs bfloat16）

**文件**：`cpp_infer/src/model.cpp` — `compute_rotary_emb()`

**根因**：`compute_rotary_emb()` 中通过 `einsum` 计算 cos/sin，整个过程在 float32 下完成，返回时没有做类型转换。但 CUDA RoPE kernel 根据 Q 的 dtype（bfloat16）来 dispatch 模板类型 T，因此会把 float32 的字节按 bfloat16 解读。

一个 float32 值占 4 字节，被重新解释为两个 bfloat16 值，结果是完全错误的数值（包含大量 NaN）。

**这是导致 NaN 的最直接原因。**

**修复**：在 `compute_rotary_emb()` 返回前，将 cos/sin 显式转换为 bfloat16。

```cpp
// Double for the full head_dim (cos/sin for both halves of rotate_half)
cos = torch::cat({cos, cos}, -1);  // [batch, seq, head_dim]
sin = torch::cat({sin, sin}, -1);

// 转换为 bfloat16，匹配 Q/K 的 dtype（CUDA kernel 按 Q 的 dtype dispatch）
cos = cos.to(torch::kBFloat16);
sin = sin.to(torch::kBFloat16);

return {cos, sin};
```

---

## Bug 3：Q/K/V transpose 后非连续内存

**文件**：`cpp_infer/src/attention.cpp` — `Attention::forward()`

**根因**：Q/K/V 经过 `.view().transpose(1, 2)` 变换后，内存布局变为非连续（non-contiguous）。PyTorch 的 `.transpose()` 只改变 stride 元信息，不重排底层数据。但 CUDA RoPE kernel 使用裸指针算术索引（如 `batch_idx * total_heads * seq_len * head_dim + head_idx * seq_len * head_dim + ...`），假设数据是连续存储的。

非连续 tensor 传入 kernel 后，kernel 通过计算出的 index 读取到的是错误位置的数据（例如 head 1 的数据会从 head 0 的某个偏移处读取），虽然不一定产生 NaN，但会得到完全错误的计算结果。

**修复**：在 transpose 之后添加 `.contiguous()` 调用，确保内存布局与 kernel 假设一致。

```cpp
// Reshape: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
// 必须调用 .contiguous()，因为 CUDA RoPE kernel 使用裸指针索引
q = q.view({batch, seq_len, num_heads_, head_dim_}).transpose(1, 2).contiguous();
k = k.view({batch, seq_len, num_kv_heads_, head_dim_}).transpose(1, 2).contiguous();
v = v.view({batch, seq_len, num_kv_heads_, head_dim_}).transpose(1, 2).contiguous();
```

---

## 修复后验证结果

| 测试项 | 结果 |
|--------|------|
| 8 张测试图 C++ 输出 | 全部生成有意义的英文描述（修复前全为 0） |
| 首个生成 token 匹配率 | 6/8 与 Python baseline 一致 |
| logit 级别对比 | C++ 选择的 token 在 Python logit 排名中位于 top-3，logit 差异仅 0.25 |
| 结论 | C++ 引擎计算正确，token 级别差异为 bfloat16 正常数值噪声 |

## 经验总结

1. **自定义 CUDA kernel 必须验证输入 tensor 的 dtype 和 layout**：kernel dispatch 基于输入 dtype，如果 cos/sin 和 Q/K 的 dtype 不一致，会导致字节重解释错误。
2. **`.transpose()` 后传入 CUDA kernel 前必须 `.contiguous()`**：PyTorch 的 transpose 只改 stride，不重排数据。任何使用裸指针索引的 CUDA kernel 都要求连续内存。
3. **NaN 排查应该逐层二分定位**：在 36 层 transformer 中从外到内添加 NaN 检测（model → layer → sublayer → kernel），快速锁定根因所在的具体 kernel。
