# TensorRT Spike 最小验证方案

面向当前的 `C++ libtorch runtime + 选择性 W8A8/INT8 量化` 主线，TensorRT 不应该一上来就做全模型替换。这个 spike 的目标不是“把 wall-x 搬到 TensorRT”，而是用最小成本回答一个更窄的问题：

> **TensorRT 能否在不碰 Flow Action、不开 plugin 大坑的前提下，对当前 C++ 基线带来足够大的增益，值得继续投入？**

---

## 1. 当前前提

你现在已经有几件关键资产：

- **C++ runtime 已经消除了 Python 框架空转**
- **attention 已经走到接近 TRT-LLM 的 cuDNN SDPA 路径**
- **W8A8 路线已经在 C++ 里打通了核心骨架**：`CUTLASS/cublasLt + fused quant/dequant`
- **Flow Action 主路径本身不适合 TensorRT 全图静态化**

所以这个 spike 不能再问“TensorRT 理论上好不好”，而要问：

> **在 wall-x 剩余还能被 TensorRT吃到的标准子图里，它到底还能多给你多少？**

---

## 2. 这次 spike 明确不做什么

下面这些一律不进本轮：

- 不做 **全模型 TensorRT engine**
- 不做 **TRT-LLM 跑 Flow Action**
- 不碰 **ODE 积分 / prefix KV truncate / action embedding 动态替换**
- 不写 **任何 plugin**
- 不先做 **ONNX 全图导出**
- 不把 spike 扩成“顺手做生产版 TensorRT runtime”

只要一旦需要写 plugin，或者开始碰 Flow Action 动态控制流，这个 spike 就已经失败了，因为它不再是“最小验证”。

---

## 3. 最小 spike 选型

### 3.1 目标子图

只测 **VQA decode 路径里的单层 attention block**，不测整模型。

建议测这段：

```text
RMSNorm
  -> q_proj / k_proj / v_proj
  -> GQA SDPA
  -> o_proj
  -> residual add
```

### 3.2 为什么选它

这是当前 wall-x 里最适合做 TensorRT 最小验证的块，因为它同时满足：

1. **是标准子图**
   不涉及 MoE expert 自定义 GEMM，不涉及 action head，不涉及 runtime scatter。

2. **它就在真实热路径上**
   VQA decode 本来就是 TensorRT 最可能发挥的场景。

3. **它覆盖了 TensorRT 剩余可能的主要价值**
   包括：
   - QKV/O 的 INT8 GEMM
   - attention 内部 kernel 选择 / 融合
   - 可能的 subgraph fusion

4. **能直接对比你当前的 C++ 基线**
   当前 C++ 里已经有：
   - `Attention::forward()`
   - `LinearOp` 的 bf16 / INT8 两条路径

换句话说，这个 spike 是在问：

> **同样一层 attention block，TensorRT 能不能比你当前的 C++ 实现更快到值得继续做？**

---

## 4. 为什么不选别的

### 4.1 为什么不直接测 Flow Action

因为 Flow Action 的核心价值点正好是 TensorRT 最不擅长的部分：

- ODE 循环
- prefix/postfix KV cache 截断复用
- 每步动态替换 `<|action|>` embedding
- 动态 proprioception embedding scatter

一旦把这些带进来，spike 就会立刻变成 plugin / runtime 重写项目。

### 4.2 为什么不先做全 VQA engine

因为全 VQA engine 会过早引入这些工程噪声：

- 多层拼装
- 权重加载 / engine 序列化
- shape/profile 管理
- 多模态输入打包
- 可能碰到 MoE 路由边界

这会让你很难判断：最后的结果是 TensorRT 本身不值，还是你在工程接线上浪费了时间。

### 4.3 为什么不先做 decoder MLP block

因为 wall-x 的 decoder MLP 路线已经和普通 dense MLP 不完全等价，容易把 spike 引到 MoE / expert 路由上去，scope 会变脏。attention block 更干净。

---

## 5. 实验对象和输入尺寸

优先按当前 3B 配置的真实尺寸做，不要自创 toy shape。

可直接参考：

- [`qwen25_config.json`](/Users/sam/project/github/wall-x/workspace/lerobot_example/qwen25_config.json)

关键维度：

- `hidden_size = 2048`
- `num_attention_heads = 16`
- `num_key_value_heads = 2`
- `head_dim = 128`

建议测两组 shape：

### 5.1 Decode shape

最重要，必须测：

```text
batch = 1
q_len = 1
kv_len ≈ 420
hidden = 2048
heads = 16
kv_heads = 2
head_dim = 128
```

这是最贴近 VQA 真正热点的场景。

### 5.2 Prefill shape

次重要，可选：

```text
batch = 1
q_len ≈ 420
kv_len ≈ 420
hidden = 2048
```

如果时间有限，**decode 优先于 prefill**。

---

## 6. 比较基线

至少要有这四组：

| 组别 | 目的 |
|------|------|
| 当前 C++ bf16 attention block | 基础性能基线 |
| 当前 C++ W8A8 attention block | 你现在真正要打的主线基线 |
| TensorRT FP16/BF16 attention block | 验证 TRT 基本收益 |
| TensorRT INT8 attention block | 验证 TRT 对量化路径是否真有额外价值 |

如果 `TensorRT INT8` 很难在半天内跑起来，不要硬拖。先拿 `TensorRT FP16/BF16` 出结果，再决定要不要继续。

---

## 7. 最小实现方式

### 7.1 推荐实现方式

**优先用 Python 做 spike，不要一开始就接进 C++ runtime。**

原因很简单：

- 这是验证，不是交付
- Python 更适合快速拼 module / export / benchmark
- 先回答“值不值得”，再考虑 C++ 集成

推荐顺序：

1. 用一个最小 `nn.Module` 复刻单层 attention block
2. 加载一层真实的 `q_proj/k_proj/v_proj/o_proj` 权重
3. 输入真实 shape 的 tensor
4. 用 `TensorRT Python API` 或 `torch_tensorrt` 编译
5. benchmark

### 7.2 代码范围建议

建议新建一个非常独立的 spike 目录，不污染主线：

```text
workspace/trt_spike/
  README.md
  attention_block.py
  export_or_build.py
  bench.py
  results.md
```

如果后续证明确实值得做，再考虑把结果搬回 `scripts/` 或 `cpp_infer/`。

---

## 8. 数据来源

输入可以分两档：

### 档 1：最快

直接用随机 tensor，但 shape 必须和真实场景一致。

优点：

- 当天就能做
- 能先看纯 kernel / graph 层面收益

缺点：

- 只能说明性能，不说明数值稳定性

### 档 2：更可信

从当前 Python 或 C++ 路径里 dump 一份真实 attention block 输入：

- decode 时的 `x`
- 对应 `cached_k / cached_v`

优点：

- 既能测性能，也能测数值误差

建议：

> **先用随机 tensor 打通，再补一份真实 activation 做复验。**

---

## 9. 需要记录的指标

### 9.1 性能

每组至少记录：

- warmup 后 `100 ~ 1000` 次平均延迟
- `p50`
- `p95`
- 峰值显存

### 9.2 数值误差

至少记录：

- `max_abs_diff`
- `mean_abs_diff`
- `cosine_similarity`

如果拿到了真实 activation，还建议记录：

- attention 输出误差
- `o_proj` 后最终 block 输出误差

---

## 10. 成功标准

这个 spike 不是看“能不能跑”，而是看“值不值得继续”。

建议 go/no-go 标准如下：

### 10.1 继续投入的条件

只有同时满足下面两条，才值得继续：

1. **不写 plugin**
2. 相对当前 C++ 基线，达到下面任一收益：
   - decode attention block **> 15%** 加速
   - 或 extrapolate 到整步 decode，**> 5 ms / token** 的潜在收益

第二条很重要。

当前 C++ VQA decode 大约是几十毫秒一级别每步，如果这个 spike 只能省 `1~2 ms/token`，那它很可能不值得你再花一周去做 TensorRT 集成。

### 10.2 立即停止的条件

只要出现下面任一条，就停止 TensorRT 分支：

- 需要写 plugin
- 需要碰 Flow Action 动态控制流
- `INT8` 路线在半天内无法稳定 build
- 相对当前 C++ 基线，收益不足：
  - `decode < 10~15%`
  - 或 projected gain `< 5 ms/token`

---

## 11. 推荐时间盒

这个 spike 必须强行 timebox。

建议：

- **0.5 天**：搭最小 attention block + FP16/BF16 TensorRT benchmark
- **0.5 天**：补 INT8 版本
- **0.5 天**：误差验证 + 写结果

总计：

> **最多 1.5 天。**

超过这个时间还没有 clear result，就说明这条路的验证成本已经开始偏高了。

---

## 12. 结果解释方式

最后不要只给“TRT 更快 / 不更快”一句话，而要写成下面这种判断：

### 情况 A：收益明显

如果结果类似：

- 不写 plugin
- decode attention block 快 `20%+`
- projected gain `> 5 ms/token`

那么下一步可以升级为：

> **VQA-only 4-layer stack spike**

仍然不碰 Flow Action，不碰全模型。

### 情况 B：收益一般

如果结果类似：

- 只能快 `5~10%`
- 或只在 prefill 有收益，decode 没收益

那基本可以停，因为 wall-x 真正敏感的是 decode / 控制热路径，不是纸面上的 TensorRT 名义优势。

### 情况 C：需要 plugin / graph surgery

那就直接停。

这说明 TensorRT 对 wall-x 的真实进入门槛已经高于它的最小验证价值。

---

## 13. 这次 spike 的最终目标

这个 spike 只回答下面这个问题：

> **在 wall-x 当前 C++ + W8A8 主线下，TensorRT 对“标准 attention 子图”是否还能提供足够大的额外收益？**

它**不**回答下面这些问题：

- TensorRT 能不能替代整个 wall-x runtime
- TensorRT 能不能跑 Flow Action
- TensorRT 值不值得做成生产版本

这些问题只有在本次最小 spike 明确“有明显收益”之后，才值得继续问。

---

## 14. 建议的结论模板

spike 做完后，建议直接输出三行结论：

1. **结果**：TensorRT 相对当前 C++ 基线快 / 不快多少
2. **代价**：是否需要 plugin、是否需要额外 graph surgery
3. **决定**：继续做 VQA-only 扩展，还是停止 TensorRT 分支

如果只能得到“可能有点快，但工程还要再看看”，那就默认按 **不继续** 处理。

