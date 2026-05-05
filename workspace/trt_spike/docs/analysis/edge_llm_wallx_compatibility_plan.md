# `TensorRT-Edge-LLM + plugin` 路线下的 `wall-x` 兼容计划

这份文档只回答一个问题：

> **如果下一步我们不是继续手工 TRT spike，而是尽量走 `TensorRT-Edge-LLM + plugin`，`wall-x` 到底哪些部分能直接复用，哪些部分必须自己补，哪些部分最后仍然要自己写 runtime。**

当前已经确认的前提：

- `TensorRT-Edge-LLM release-0.7.0` 源码已在 Orin 上编译通过
- `Qwen2.5-VL-3B-Instruct` 已经完成：
  - LLM export
  - visual export
  - `llm.engine`
  - `visual.engine`
  - 最小 VQA inference
- `Qwen3-VL-2B-Instruct` 也已经跑通到最小 VLM benchmark

所以这份计划不是“理论讨论”，而是基于已经跑通的 edge runtime 结果，判断 `wall-x` 怎么接最现实。

---

## 1. 结论先说

### 1.1 VQA

`TensorRT-Edge-LLM + plugin` 对 `wall-x VQA` **是有机会接住大部分主干的**。

原因很直接：

- 视觉 encoder 可以单独走 export / build
- `Qwen2.5-VL` 本身就在 `TensorRT-Edge-LLM` 的支持矩阵里
- VQA 主干仍然更像标准 VLM / decoder-only 路线

所以对 VQA 来说，最现实的目标不是“完全重写一套 runtime”，而是：

> **尽量把视觉 + 语言主干交给 `TensorRT-Edge-LLM`，plugin 补齐少量自定义算子。**

### 1.2 Flow Action

`Flow Action` 不能简单按 VQA 的方式理解。

它的问题不只是主干模型，而是：

- `ActionProcessor.step()`
- `prefix / postfix`
- `moe_token_types`
- `ODE / Euler`
- 动作状态回灌

所以对 Flow 来说，`TensorRT-Edge-LLM + plugin` 的现实定位更像：

> **能吃掉一部分主干 block，但最后仍然要你自己写 runtime。**

这意味着：

- Flow 不适合硬追“全塞进 Edge-LLM”
- 更合理的是：
  - `Edge-LLM` 吃 VQA / VLM 主干
  - Flow 只借它的一部分 block / plugin / engine 能力

---

## 2. 哪些部分能直接复用 `TensorRT-Edge-LLM`

### 2.1 视觉前端

对于 `wall-x VQA`，最容易复用的是视觉前端：

- image encoder
- visual ONNX export
- visual engine build

这部分已经被 `TensorRT-Edge-LLM` 官方路径证明可行。

### 2.2 标准 decoder-only 主干

如果主干足够接近标准 LLM / VLM decoder，那么这部分也可以复用：

- prefill
- decode
- KV cache
- sampling

这就是 `TensorRT-Edge-LLM` 最核心的强项。

### 2.3 官方已有的 plugin

当前已经能看到的官方 plugin 路线包括：

- `AttentionPlugin`
- `ViTAttentionPlugin`
- 以及官方 runtime 里对多模态 / speculative / action 相关能力的扩展接口

这意味着：

> **如果 `wall-x` 的子图能映射到现成 plugin，优先复用官方 plugin，而不是自己重复造。**

---

## 3. 哪些部分必须自己补

### 3.1 `wall-x` 特有的多模态 glue

`wall-x` 不是纯 VLM。它还包含：

- image / vision token scatter
- `position_ids` / rope index 组织
- `moe_token_types`
- action token 占位

这些有些可以靠 plugin，有些要靠你自己的前处理 / runtime glue。

### 3.2 Flow 的动作控制链

Flow 的核心控制链基本不可能只靠 `TensorRT-Edge-LLM` 自动解决：

- `ActionProcessor.step()`
- `action_proj_back`
- `ODE / Euler`
- prefix KV trim / reuse
- postfix-only repeated forward

这部分本质上还是 runtime 逻辑。

### 3.3 engine orchestration

就算主干都进了 `Edge-LLM`，你仍然要自己负责：

- 预处理
- session 生命周期
- KV cache 生命周期
- loop control
- 输出后处理

也就是说：

> **plugin 解决的是算子洞，不是系统层控制流。**

---

## 4. `wall-x` 应该怎么接这条路线

### 4.1 VQA 的最现实接法

VQA 最适合的路线是：

```text
图片 + prompt
-> 视觉前端 / tokenizer / glue
-> TensorRT-Edge-LLM 的标准 VLM runtime
-> 输出文本
```

这条路的关键是：

- 尽量少写自定义 runtime
- 能用官方 plugin 就用官方 plugin
- 让 `Edge-LLM` 吃掉最值钱的标准 VLM 主干

### 4.2 Flow 的最现实接法

Flow 更像：

```text
image / state / action state
-> 你自己的 runtime
-> Edge-LLM 只吃其中一部分可复用 block
-> 剩下的 ODE / action loop 仍在 runtime 里
```

这条路的判断是：

- **不要试图把 Flow 强行变成标准 LLM**
- **而是把能复用的 block 给 Edge-LLM，剩下的 runtime 自己保留**

---

## 5. 对 `wall-x` 的工程建议

### 建议 A：先把 VQA 作为 Edge-LLM 的主攻入口

因为 VQA 和官方 `Qwen2.5-VL` 的重合度最高：

- 更像标准 VLM
- 更容易复用官方 export / build / inference 产物

### 建议 B：Flow 只做局部接入验证

Flow 不建议一开始就追求：

- 完整接管
- 单 engine
- 全链路 Edge-LLM 化

更现实的是：

- 先验证哪些 block 可复用
- 哪些 plugin 可直接用
- 哪些 runtime 逻辑必须保留

### 建议 C：不要再把“能不能跑通”当终点

`Edge-LLM` 已经证明能跑通官方 VLM。
下一步真正值得问的是：

- 它能帮你省多少 runtime 工程量
- 它和你现在手工 TRT / cpp_infer 的差距在哪里
- 对 `wall-x Flow`，最后是不是只会变成“能复用一部分 block，但 runtime 还是你自己写”

---

## 6. 最短结论

压成一句话：

> **`TensorRT-Edge-LLM + plugin` 很适合接 `wall-x VQA` 这种更像标准 VLM 的部分；但对 `wall-x Flow Action` 这种强控制流 VLA，它最多只能吃掉一部分 block，最后仍然要你自己保留 runtime。**

