# `wall-x` 端侧 Runtime 设计草图

这份文档不再讨论单个 block 能不能进 `TensorRT / TRT-LLM`，而是基于当前已经跑通的结果，回答一个更系统的问题：

> **如果把 `wall-x` 在端侧真正做成可用系统，它的 runtime 应该长什么样？**

当前已经有两条很明确的实验结论：

- **VQA**：已经从 `5 engine` 收到了 **`2 engine` 原型**
  - `prefill.engine`
  - `decode_dynamic.engine`
- **Flow Action**：已经验证了更合理的 **`2 engine` 形态**
  - `flow_prefetch_36.engine`
  - `flow_postfix_step_36.engine`

所以这份设计草图的目标不是再发明新路径，而是把这些结果收成一个更像真正 runtime 的结构。

---

## 1. 设计目标

这个 runtime 不追求“一切都塞进一个 engine”，而追求：

1. **engine 数量尽量少，但不为了 1 个 engine 硬牺牲合理性**
2. **让 `TensorRT` 专注做计算后端**
3. **让 task-specific 控制流留在 runtime**
4. **VQA 和 Flow 尽量共享基础设施**
5. **后续继续吸收 plugin / graph / KV 管理能力时，不推翻当前结构**

压成一句话：

> **不是“一个大 engine 包打天下”，而是“少量 engine + 一层清晰可控的 runtime”。**

---

## 2. 三层结构

整个系统更适合拆成三层。

### 2.1 上层任务层

负责和机器人应用直接打交道：

- 图像采集 / 相机同步
- 文本 prompt 组织
- 机器人 state / proprioception 组织
- 输出动作或文本结果

这层不关心具体 engine 怎么跑，只关心：

- 我要做的是 `VQA`
- 还是 `Flow Action`

### 2.2 中间 runtime 层

这是最关键的一层，也是当前真正应该建设的东西。

它负责：

- engine registry
- session 生命周期
- buffer 复用
- KV cache 生命周期
- prefill / decode / postfix orchestration
- token loop / Euler loop
- sampling / stopping 条件
- plugin 所需的 host-side 预计算

这层才是真正意义上的：

> **VLA runtime**

### 2.3 底层执行层

负责具体算：

- `TensorRT`
- `TRT-LLM`
- plugin
- 以及必要时的少量 host-side fallback

这层不负责“任务逻辑”，只负责执行 engine。

---

## 3. VQA 路径

当前最合理的 VQA 路径已经比较清楚：

```text
输入组织
-> 视觉前处理 / image embed scatter / rope index
-> prefill.engine
-> decode_dynamic.engine 循环
-> sampling / token stop
-> 文本输出
```

### 3.1 engine 形态

VQA 当前推荐形态是：

- `prefill.engine`
- `decode_dynamic.engine`

而不是：

- `decode_past420.engine`
- `decode_past421.engine`
- `decode_past422.engine`
- `decode_past423.engine`

也就是说，VQA 的目标不是“一个 engine”，而是先稳定在：

> **2 个 engine + runtime 管理 KV 和 decode loop**

### 3.2 runtime 负责什么

VQA runtime 负责：

- 管理输入 embedding 组织
- 持有 `prefill` 和 `decode_dynamic` session
- 管理当前 `past_len`
- 管理 token loop
- 负责 sampling / EOS

### 3.3 当前状态

当前已经验证：

- `2 engine` 版 VQA 能跑完整 `20 token`
- 文本结果与 reference 完全一致
- 但端到端还略慢于 `cpp_infer`

所以 VQA 这条线已经从“静态 spike”进到了：

> **runtime 原型阶段**

但还没进入“更强于 `cpp_infer` 的生产候选阶段”。

---

## 4. Flow Action 路径

Flow 的结构和 VQA 有本质区别，它本来就更像两阶段系统：

```text
输入组织
-> action step(t0) / 动作初始注入
-> flow_prefetch.engine
-> prefix KV trim
-> flow_postfix.engine 循环
-> Euler / ODE 更新
-> 动作输出
```

### 4.1 engine 形态

Flow 当前最合理的形态是：

- `flow_prefetch.engine`
- `flow_postfix.engine`

这里的 `flow_postfix.engine` 可以是：

- 基础版 postfix
- 或 fused postfix

但系统形态本身仍然是两阶段。

### 4.2 为什么不追求 1 个 engine

因为 `Flow` 天生有两种不同性质的计算：

- `prefetch`
  - 吃完整输入序列
  - 产出 prefix KV
- `postfix`
  - 固定长度动作段
  - 进入 ODE / Euler 循环

它们的输入集合、输出集合、执行频率都不同。  
硬追 1 个 engine，通常只会把：

- engine 做得更大
- runtime 更难调
- session 更重

而不会自动让端到端更快。

所以对 Flow 来说，更合理的目标是：

> **2 个 engine 就够了，把 runtime 做轻，而不是硬追 1 个 engine。**

### 4.3 runtime 负责什么

Flow runtime 负责：

- prefix / postfix 切分
- prefix KV 生命周期
- 每步 timestep 更新
- Euler / ODE 循环
- 动作状态回灌
- 最终动作轨迹组织

### 4.4 当前状态

当前已经验证：

- 两阶段 Flow TRT 已经完整跑通
- 端到端已经明显快于 Python baseline
- 也已经快于当前 `cpp_infer`

所以 Flow 已经进入：

> **值得继续建设 runtime 的阶段**

而不是“还在验证能不能做”的阶段。

---

## 5. 共享基础设施

VQA 和 Flow 看起来很不同，但 runtime 里其实有不少共用能力：

### 5.1 engine registry

统一管理：

- engine 文件路径
- engine 版本
- precision
- plugin 依赖
- profile 配置

### 5.2 session 池

统一管理：

- 常驻 session
- 按需创建 / 销毁
- 内存压力控制

### 5.3 buffer / tensor 池

统一管理：

- 输入输出 tensor 复用
- KV cache buffer
- host/device staging buffer

### 5.4 预处理与 post-processing

统一管理：

- token embedding 组织
- image embedding scatter
- position_ids / rope index
- 输出文本 / 动作解码

---

## 6. plugin 的位置

plugin 不应该被当成“把 runtime 省掉”的工具，而应该被放在正确位置：

### 6.1 plugin 该补什么

适合 plugin 的是：

- `multimodal_rope`
- `rot_pos_emb`
- `get_window_index`
- 必要的 scatter / gather / replace
- 必要的 vision 特殊算子

也就是：

> **图里的算子缺口**

### 6.2 plugin 不该试图补什么

不适合靠 plugin 硬补的是：

- token loop
- ODE / Euler
- prefix / postfix orchestration
- sampling / stopping
- KV cache 生命周期

也就是：

> **系统里的 runtime 缺口**

---

## 7. 当前最现实的落地顺序

如果从现在继续推进，我会建议按下面顺序做，而不是四处散开。

### 7.1 先把 VQA 的 `2 engine` 形态稳定住

目标：

- 固化 `prefill + decode_dynamic`
- 把 `run-only` 复用做稳定
- 再看还能不能继续压到更接近 `cpp_infer`

### 7.2 再把 Flow 的 `2 engine` 形态稳定住

目标：

- 固化 `prefetch + postfix`
- 不再继续追“一个 engine”
- 重点放在 session / I/O / runtime 组织

### 7.3 最后才考虑统一 runtime 框架

当 VQA 和 Flow 各自都稳定以后，再抽象出：

- 通用 engine manager
- 通用 session manager
- 通用 KV / buffer manager
- 通用 task runtime 接口

也就是说：

> **先让两条路各自稳定，再做统一 runtime，而不是一开始就大一统。**

---

## 8. 最终判断

压成一句话：

> **`wall-x` 端侧最终更像“少量 TRT engine + 一层 VLA runtime”，而不是“一个万能 engine 文件”。**

再具体一点：

- **VQA**
  - 目标形态：`2 engine`
  - 未来也许有机会进一步收成 `1 engine`
- **Flow**
  - 目标形态：`2 engine`
  - 不值得为“1 engine”硬付复杂度

所以真正该建设的，不只是更多 engine，而是：

> **一层清晰、轻量、可控的端侧 VLA runtime。**
