# `TensorRT-Edge-LLM` vs `wall-x TRT` vs `cpp_infer`

这份文档只做一件事：

> **把三条路线放在同一张图里，看它们各自到底在 `wall-x` 里扮演什么角色。**

---

## 1. 三条路线的定位

### 1.1 `cpp_infer`

这是当前 `wall-x` 自己已经打出来的主线：

- Python 框架空转已经去掉
- 关键热路径已经在 C++ / CUDA / libtorch 上跑起来
- 当前已经有很强的实际端到端结果：
  - Flow Action：`290.7 ms`
  - VQA（20 tokens）：`851.2 ms`

### 1.2 当前手工 `TensorRT / TRT-LLM` 路线

这是 `workspace/trt_spike/` 里做出来的路线：

- VQA-specialized：已经从 `5 engine` 收成 `2 engine`
- Flow Action：已经能跑 `prefetch + postfix` 两阶段，且已经明显快于 `cpp_infer`

它的意义是：

> **证明 `wall-x` 的主干计算块可以被拆成 TensorRT / TRT-LLM 可执行的 engine。**

### 1.3 `TensorRT-Edge-LLM`

这是 NVIDIA 官方更偏 edge / embedded 的那条线：

- 官方仓库：`NVIDIA/TensorRT-Edge-LLM`
- 当前在你的 Orin 上：
  - 已源码编译通过
  - 已完成 `Qwen2.5-VL-3B-Instruct` 的 LLM export
  - 已完成 visual export
  - 已生成 `visual.engine`
  - 已生成 `llm.engine`
  - 最小 VQA 推理已跑通

这条线的意义是：

> **证明官方 edge-side runtime 真的能把标准 VLM / VQA 路线完整打通。**

### 1.4 `TensorRT-Edge-LLM` 是不是只有一个 engine 文件

不是。至少对我们已经跑通的 `Qwen2.5-VL-3B` 来说，**不是一个 engine 文件，而是一个 engine 目录**。

当前已经实际生成出来的是：

- `llm.engine`
- `visual/visual.engine`

也就是说，它更像：

```text
engineDir/
  llm.engine
  visual/
    visual.engine
```

而不是：

```text
one_single_engine.engine
```

这件事很重要，因为它说明：

- VLM / VLA 在 `Edge-LLM` 这条线上，本来就更偏“多组件 runtime”
- 视觉和语言主干是天然分开的
- 后面判断这条线值不值，不该只盯“是不是一个 engine 文件”，而应该看：
  - 拆成几块
  - 每块多大
  - runtime orchestration 成本有多高

如果是纯 LLM，它可能更接近一个主 engine；  
但只要是 VLM / multimodal，通常就很难真的只剩一个文件。

### 1.5 它能不能直接跑 `wall-x` 这样的 VLA

要把这个问题说准确，必须拆成两层：

1. **它能不能跑 `wall-x` 里“像标准 VLM / VQA 的那部分”**
2. **它能不能直接接住 `Flow Action` 这种强控制流 VLA**

当前更准确的判断是：

> **`TensorRT-Edge-LLM` 可以接住 `wall-x` 里“像标准 VLM / VQA 的部分”，但不能直接把当前 `Flow Action` 当成开箱即用的完整 VLA 终态。**

更具体地说：

- **能接住的**
  - 视觉编码
  - 文本解码
  - 比较标准的 VLM / VQA 主干

- **不能直接开箱的**
  - `ActionProcessor.step()`
  - `prefix / postfix`
  - `moe_token_types`
  - `ODE / Euler`
  - 动态动作状态回灌

所以如果把 VLA 分成两类：

- **VLM 主干 + 薄 action head**
  - 更接近 `TensorRT-Edge-LLM` 的舒适区
- **带控制环的具身 runtime**
  - 最终仍然会逼你自己保留一层 runtime

那么 `wall-x` 尤其 `Flow Action`，明显更靠后者。

---

## 2. 当前已确认的最小结果

### 2.1 `cpp_infer`

- Flow Action：`290.7 ms`
- VQA（20 tokens）：`851.2 ms`

### 2.2 手工 `TensorRT / TRT-LLM`

- VQA（20 tokens，动态 decode，2 engine）：`943.018 ms`
- Flow Action（两阶段 TRT）：`176.837 ms`

### 2.3 `TensorRT-Edge-LLM`

最小 VQA inference 已跑通，输出文件：

- `/data/wy/wall-x/workspace/edge_llm_exp/output_vlm_qwen25vl3b.json`

对应最小 wall-clock benchmark：

- `run_1_ms = 7704.467`
- `run_2_ms = 7266.907`
- `run_3_ms = 7626.541`
- `mean_ms = 7532.639`

这个 benchmark 不是拿来和 `wall-x` 做同口径性能对比的（因为模型路径不同、任务组织也不同），但它证明了：

> **`TensorRT-Edge-LLM` 已经能在 Orin 上把一条官方 VLM 路线完整跑起来。**

### 2.4 对 `wall-x` 的直接启发

这组最小 benchmark 的意义不是“它一定快”，而是：

- 它证明 `TensorRT-Edge-LLM` 的官方 VLM 路线在 Orin 上**功能可行**
- 但同时也说明：**默认形态下的端到端时延并不低**

这对 `wall-x` 的现实结论是：

> **如果你的首要目标是“尽量压低端到端时延”，当前手工 `cpp_infer` / 自建 TRT runtime 仍然更有吸引力；如果你的首要目标是“尽量贴近 NVIDIA 官方 edge runtime 形态并保持可维护性”，`TensorRT-Edge-LLM` 值得继续研究。**

补充一条更贴近 `wall-x` 使用方式的事实：

我们另外做了一层独立 bridge：

- `workspace/edge_llm_wallx_bridge/bench_vqa_edge_llm.py`

它不是直接复用之前那条实验脚本，而是单独以：

- engine 目录
- 图片路径
- prompt

去调用 `llm_inference`，更接近后面真实接入 `wall-x` 时的调用方式。

在 `Qwen2.5-VL-3B` 上，这条 bridge benchmark 的结果是：

- `run_1_ms = 8319.879`
- `run_2_ms = 8171.447`
- `run_3_ms = 8091.836`
- `mean_ms = 8194.387`

这说明：

> **一旦按更接近 `wall-x` 接入层的方式去跑，`TensorRT-Edge-LLM` 在这条最小 VQA 路线上的真实端到端时延更接近 `8.2s`。**

我们又把问题换成了 `wall-x` 里更常见的原始问法：

- 图像：`fruits_on_table.png`
- prompt：`Describe what you see in this image.`

同口径的 wall-x baseline 参考答案是：

- `The image features a minimalist composition with a brown background. In the center, there are three distinct shapes`

在这个完全同图同 prompt 的对照下，`TensorRT-Edge-LLM` 的输出和 wall-x baseline 并不接近：

- `Qwen2.5-VL-3B-Instruct`
  - `run_1_ms = 7703.721`
  - 输出：`The image depicts four colorful objects placed on a flat surface...`
  - 文本相似度：`0.3083`
- `Qwen3-VL-2B-Instruct`
  - `run_1_ms = 5457.292`
  - 输出：`This image is a simple, stylized illustration of a face...`
  - 文本相似度：`0.3169`

这给出的更直接判断是：

> **`TensorRT-Edge-LLM` 可以把 `wall-x` 风格的 VQA 输入跑起来，但按当前官方默认路径，它并没有自然复现 wall-x 在同图同 prompt 下的 baseline 语义输出。**

为了避免只盯着一个 case，我们又把 `wall-x` 现有 VQA 结果集里的前 6 个 case 批量喂给了 Edge-LLM：

- `blocks_and_plates`
  - `Describe what you see in this image.`
  - `What objects are on the table?`
- `dual_arm_robot`
  - `Describe what you see in this image.`
  - `What objects are on the table?`
- `fruits_on_table`
  - `Describe what you see in this image.`
  - `What objects are on the table?`

在这 6 个代表性 case 上，批量结果是：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`7542.18 ms`
  - 平均文本相似度：`0.3611`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`5436.60 ms`
  - 平均文本相似度：`0.3944`

这组批量结果把结论压得更稳了：

> **`TensorRT-Edge-LLM` 的官方 VLM 路线在 wall-x 风格的 VQA case 上是可跑的，但当前默认路径下，它和 wall-x baseline 之间仍然有明显的语义分布差异；更小的模型更快，但并没有自动更贴近 wall-x 的答案风格。**

最后，我们还把它推进到了 websocket serving 层：

- `launch_vqa_serving.py --backend edge`
- 客户端发送 numpy 图像 + prompt 后，能收到：
  - `vqa` response
  - `server_timing.infer_ms`

这说明：

> **`TensorRT-Edge-LLM` 已经不只是 CLI / wrapper 级别可跑，而是已经能挂到 wall-x 的 websocket serving 层，作为真实 backend 返回结果。**

我们还专门做了 `propri/action` 的字面串 probe：

- `What objects are on the table?`
  - 加 `Proprioception: <|propri|>` / `<|action|>` 后，Edge-LLM 仍然保持对象枚举风格
- `Predict the next action in robot action.`
  - 加 `Proprioception: <|propri|> <|action|>` 后，Edge-LLM 才开始输出动作描述

这说明：

> **`propri/action` 作为文本 prompt 的字面内容不会报错，但并不会在 Edge-LLM 里自动变成 wall-x 的专用动作 token 体系。**

随后我们又把这条批量路线扩成了完整 `16` 个 wall-x VQA case。最终均值是：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`7488.74 ms`
  - 平均相似度：`0.3636`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`5452.96 ms`
  - 平均相似度：`0.4041`

这让结论更稳定：

> **全量 16-case 下，`TensorRT-Edge-LLM` 仍然证明了官方 VLM 路线可跑，但它和 wall-x baseline 的语义分布差异并没有因为 case 变多而消失。**

最后，我们又把输入组织收敛到 `wallx_vqa` preset：

- 去掉 system prompt
- 将 `max_generate_length` 对齐到 wall-x reference token 数

在这个 best preset 下，完整 `16-case` 的结果变成：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`6781.01 ms`
  - 平均相似度：`0.4320`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`5148.08 ms`
  - 平均相似度：`0.4726`

按问题类型拆开后：

- `Describe what you see in this image.`
  - `Qwen2.5-VL-3B`: `6723.48 ms / 0.4652`
  - `Qwen3-VL-2B`: `5250.01 ms / 0.4410`
- `What objects are on the table?`
  - `Qwen2.5-VL-3B`: `6838.53 ms / 0.3988`
  - `Qwen3-VL-2B`: `5046.15 ms / 0.5041`

这说明：

> **`TensorRT-Edge-LLM` 对 wall-x 的 VQA 不是“默认就能完全接上”，但只要把输入组织调成 wallx_vqa preset，它的语义输出会更接近 wall-x baseline，尤其在“对象枚举”类问题上。**

从问题类型角度看，这个 best preset 还给了一个更细的判断：

- `Describe what you see in this image.`
  - `Qwen2.5-VL-3B` 相似度略高：`0.4652 > 0.4410`
  - 但时延明显更大：`6723 ms > 5250 ms`
- `What objects are on the table?`
  - `Qwen3-VL-2B` 相似度更高：`0.5041 > 0.3988`
  - 同时也更快：`5046 ms < 6839 ms`

所以如果后面要先挑一个官方 VLM engine 去承接 wall-x VQA：

> **更现实的第一选择是 `Qwen3-VL-2B`，尤其当问题更偏“对象枚举”时。**

我们还把这件事推进到了主工程入口级别：

- `scripts/vqa_backend_switch.py`
- `scripts/vqa_backend_compare.py`

在 Orin 上，同图同题的单 case 对照已经能直接从主工程入口跑：

- 图像：`fruits_on_table.png`
- 问题：`Describe what you see in this image.`
- `Edge-LLM / Qwen3-VL-2B`
  - `latency_ms = 4970.847`
  - 输出：`This image is a simple, minimalist graphic composed of several geometric shapes on a two-tone background. The`
- `wall-x`
  - 输出：`The image features a minimalist composition with a brown background. In the center, there are three distinct shapes`
- 文本相似度：`0.4732`

这意味着：

> **现在已经不是“能不能跑 Edge-LLM”的问题，而是主工程里已经有了一个可以切 wall-x / Edge 两个 backend 的统一入口。**

我们还把这个统一入口扩成了 6-case 的 wall-x vs Edge 双后端对照：

- 图片：
  - `blocks_and_plates`
  - `dual_arm_robot`
  - `fruits_on_table`
- 问题：
  - `Describe what you see in this image.`
  - `What objects are on the table?`

结果：

- `mean_edge_latency_ms = 5508.13`
- `mean_similarity = 0.5216`

分布上也很稳定：

- “What objects are on the table?” 通常更接近 wall-x baseline
- “Describe what you see in this image.” 仍然保留明显风格差异

这说明：

> **从主工程入口层看，Edge-LLM 已经够资格做 wall-x VQA backend 的替换候选；但它现在更像“对象枚举类问题优先可替换”，不是所有 VQA 问题都已经无缝等价。**

进一步地，我们把两套 Edge backend 做成了一个规则路由器：

- `Describe ...` -> `Qwen2.5-VL-3B`
- `What objects ...` -> `Qwen3-VL-2B`

这个 `edge_router` 在完整 `16-case` 上的结果是：

- `mean_latency_ms = 5707.43`
- `mean_similarity = 0.4846`

也就是说：

> **如果只比较 Edge 家族内部，按问题类型做路由，已经能比单一 `Qwen2.5-VL-3B` 或单一 `Qwen3-VL-2B` 更接近 wall-x baseline。**

而且这条路由已经不只是离线 benchmark：

- `launch_vqa_serving.py --backend edge_router`
- `VQAClient`

在 Orin 上已经验证了 websocket 服务层会把对象问题路由到 `qwen3`，把描述问题路由到 `qwen25`。

我们又进一步做了一个输入组织消融：

- 去掉 system prompt
- 将 `max_generate_length` 对齐到 wall-x reference token 数

在 6-case 代表性 batch 上：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`9536.22 ms`
  - 平均相似度：`0.4285`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`6604.15 ms`
  - 平均相似度：`0.5216`

这说明：

> **输入组织不是小问题。system prompt 和生成长度对 Edge-LLM 的 wall-x 对齐程度有明显影响；但即便做了这层对齐，它仍然没有自然收敛到 wall-x baseline 的答案分布。**

我们现在已经把“更接近 wall-x 的 VQA 适配方式”收成了一个固定 preset：

- 去掉 system prompt
- 将 `max_generate_length` 对齐到 wall-x reference token 数

也就是说，后续如果要继续测试 Edge-LLM 承接 wall-x VQA，应该优先跑：

- `--compat-mode wallx_vqa`

而不是继续用默认官方问法。

我们还把模板和 token 体系做了静态对照，结论也很明确：

- `wall-x`
  - `processor_class = Qwen2_5_VLProcessor`
  - 模板里有 `You are a helpful assistant.` fallback
  - 模板里有 `tools` 分支
  - 有 `propri` / `action` / `2048` 个 `action_token_*`
- `TensorRT-Edge-LLM / Qwen2.5-VL-3B`
  - 默认 system prompt 也是 `You are a helpful assistant.`
  - 没有 `propri` / `action`
- `TensorRT-Edge-LLM / Qwen3-VL-2B`
  - 默认 system prompt 为空
  - 没有 `propri` / `action`

这意味着：

> **VQA 的主要问题还可以落在“输入组织 + 输出风格”上继续调；但 Flow 不是同一类问题，因为 token 体系本身就已经不一致了。**

### 2.5 第二个模型：`Qwen3-VL-2B-Instruct`

为了避免只看单一模型，我们又跑了第二个官方 VLM 样例：

- `Qwen/Qwen3-VL-2B-Instruct`

同样走的是：

- `tensorrt-edgellm-export-llm`
- `tensorrt-edgellm-export-visual`
- `llm_build`
- `visual_build`
- `llm_inference`

当前已经拿到完整结果：

- 输出文件：
  - `/data/wy/wall-x/workspace/edge_llm_exp/output_vlm_qwen3-vl-2b.json`
- 最小 VQA benchmark：
  - `run_1_ms = 5954.077`
  - `run_2_ms = 5650.564`
  - `run_3_ms = 5333.013`
  - `mean_ms = 5645.885`

这组结果比 `Qwen2.5-VL-3B` 更快，说明：

> **在 `TensorRT-Edge-LLM` 的官方 VLM 路线上，模型变小之后，Orin 上的端到端时延确实会继续下降。**

但它仍然和你现在 `wall-x` 里手工做出来的 TRT / `cpp_infer` 结果不在同一量级上讨论：

- 它证明的是 **官方 edge VLM 路线可行**
- 不是说它在你这个 `wall-x` 场景里已经是最优解

所以第二个模型给出的更具体启发是：

> **`TensorRT-Edge-LLM` 对 VLM 的支持是有实用价值的，但它的默认性能和你的手工 runtime 路线之间，仍然隔着一层“是不是要完全沿官方 edge runtime 走”的选择题。**

---

## 3. 三者的关系

### 3.1 `cpp_infer` 是当前最强的可控基线

优点：

- 完全贴合当前 `wall-x`
- runtime 结构可控
- 对 Flow Action 很强

缺点：

- 还不是官方 edge runtime
- 对标准 VLM / VLA 的复用程度有限

### 3.2 手工 `TensorRT / TRT-LLM` 是“中间态”

优点：

- 已经验证很多 block 能进 TRT
- Flow Action 已经明显赢过 `cpp_infer`
- VQA 已经证明能从多 engine 收成动态 decode

缺点：

- 当前仍然是手工 runtime
- engine 组织还没收敛到最终形态

### 3.3 `TensorRT-Edge-LLM` 是官方 edge 路线

优点：

- 官方支持 `Qwen2.5-VL`
- 在 Orin 上源码编译通过
- 已经能把 `Qwen2.5-VL-3B` 的 LLM / visual / inference 跑通
- `Qwen3-VL-2B` 也能跑通，而且更小模型确实更快
- 但在 `fruits_on_table + Describe what you see in this image.` 这个 wall-x 常用 case 上，输出语义和 wall-x baseline 仍有明显差异

缺点：

- 对 `wall-x Flow Action` 这种重控制流任务，不能直接假设开箱即用
- 更偏官方 VLM / VLM runtime 路线，不是 `wall-x` 的原生 runtime

---

## 4. 对 `wall-x` 的现实判断

### 4.1 VQA

现在更像是：

- `cpp_infer` 依然是强基线
- 手工 TRT 还没全面赢
- `TensorRT-Edge-LLM` 证明官方 VLM 路线可行

所以 VQA 这条线更适合继续做：

- runtime 收敛
- 动态 decode
- plugin / engine 数量收缩

### 4.2 Flow Action

现在结论更直接：

- `cpp_infer` 已经很强
- 手工 TRT 已经明显赢
- 这条线不该再回到“能不能做”的问题

更值得继续的是：

- 是否要进一步做成更轻的 runtime
- 是否要把 plugin / engine 形态继续收缩

### 4.3 `TensorRT-Edge-LLM`

它对 `wall-x` 的意义更像：

- **官方 edge runtime 可行性证明**
- **VLM 主干支持的参考系**

而不是：

- 直接替代当前 `cpp_infer`
- 直接替代当前手工 TRT spike

---

## 5. 最短结论

压成一句话：

> **`cpp_infer` 是当前 `wall-x` 最强的可控基线；手工 `TensorRT / TRT-LLM` 已经证明在 Flow 上有明确优势、在 VQA 上开始进入 runtime 收敛阶段；`TensorRT-Edge-LLM` 则证明官方 edge-side VLM 路线在 Orin 上是能完整跑通的。**

---

## 6. 最终选型建议

如果你的目标是现在就做决策，这三条路线可以直接这么选：

### 6.1 继续拥抱 `cpp_infer` 的情况

适合你如果优先看重：

- 端到端时延
- 对 `wall-x` 任务结构的完全贴合
- 快速调试和修改 runtime

当前最硬的事实是：

- `cpp_infer` 的 Flow Action 还在明显领先于 VQA 这边的官方 edge runtime 路线
- `cpp_infer` 的整体 runtime 形态仍然是你现在最可控的基线

所以如果你现在最关心的是：

> **“在 `wall-x` 这条线里，谁能更快、更稳地达到最好性能？”**

答案仍然更偏向：

> **继续主投 `cpp_infer` / 自建 runtime**

### 6.2 继续研究 `TensorRT-Edge-LLM` 的情况

适合你如果更看重：

- 官方 edge 路线
- 更强的模型兼容性
- 更接近 NVIDIA 未来支持方向
- 更低的长期维护成本

它已经证明了两件很重要的事：

- `Qwen2.5-VL-3B` 能在 Orin 上完整跑通
- `Qwen3-VL-2B` 也能跑通，而且更小模型确实更快

所以如果你的问题是：

> **“这条官方路线值不值得继续投资？”**

答案是：

> **值得继续研究，但它当前更像“官方 edge runtime 参考路线”，不是 `wall-x` 的最优性能答案。**

### 6.3 继续保留手工 TRT / TRT-LLM spike 的情况

这条路线的价值已经很清楚了：

- Flow 已经明显赢了 `cpp_infer`
- VQA 已经从 `5 engine` 收到了 `2 engine`
- `decode_dynamic.engine` 说明 runtime 方向是可行的

所以它更像：

> **`cpp_infer` 和 `TensorRT-Edge-LLM` 之间的一层高价值中间态**

适合继续作为：

- block / 子图验证平台
- runtime 演化平台
- 未来迁移到更统一 edge runtime 的过渡带

### 6.4 最短推荐

如果只给一个结论：

> **短期性能主线继续押 `cpp_infer`；中期如果你想吃 NVIDIA 官方 edge 路线的生态和可维护性，就继续研究 `TensorRT-Edge-LLM`；而当前手工 TRT / TRT-LLM spike 应该保留为过渡和验证层，而不是最终终态。**

---

## 7. 直接回答：`TensorRT-Edge-LLM` 的性能如何？它能不能跑 `wall-x`？

### 7.1 性能

如果只看我们这轮已经实际跑出来的最小 VQA benchmark：

- `Qwen2.5-VL-3B-Instruct`
  - 平均 wall-clock：`7532.639 ms`
- `Qwen3-VL-2B-Instruct`
  - 平均 wall-clock：`5645.885 ms`

这说明：

> **`TensorRT-Edge-LLM` 的默认官方 VLM 路线，在我们当前的 Orin 上是秒级，不是亚秒级。**

所以如果你的目标是：

- **极致低时延**

那它当前的默认形态**不是**你现在 `wall-x` 里最优的性能答案。

### 7.2 它能不能跑 `wall-x`

答案要拆开看。

#### 能跑的部分

`wall-x` 里**更像标准 VLM / VQA 的那部分**，`TensorRT-Edge-LLM` 是有希望接住的，甚至已经在我们的测试里证明了：

- `Qwen2.5-VL` 官方 VLM 路线能跑
- `Qwen3-VL` 官方 VLM 路线也能跑

所以如果你的 `wall-x` 目标只是：

- 视觉理解
- 文本回答
- 比较标准的 VLM 推理链

那它是**能跑一部分 wall-x 思路的**。

#### 不能直接开箱的部分

`wall-x` 的真正难点，尤其是 `Flow Action`，不是标准 VLM 推理，而是：

- `ActionProcessor.step()`
- `prefix / postfix`
- `moe_token_types`
- `ODE / Euler`
- 动态动作状态回灌

这些东西**不是 `TensorRT-Edge-LLM` 当前官方 VLM 流程里天然就有的抽象**。

所以更准确地说：

> **`TensorRT-Edge-LLM` 可以跑 `wall-x` 的“VLM / VQA 主干子集”，但不能直接把当前 `wall-x Flow Action` 当成开箱即用的目标。**

### 7.3 最终判断

压成一句最短的话：

> **`TensorRT-Edge-LLM` 能跑 `wall-x` 里“像标准 VLM/VQA 的部分”，但对于 `Flow Action` 这种强控制流路线，仍然不能直接当成完整答案；而且它当前默认形态的性能是秒级，不是你现在最需要的低时延解。**

### 7.4 wall-x Flow Action bridge on Orin

我们又单独把 `wall-x` 的 `Flow Action` 做成了一个 `TensorRT-Edge-LLM action_build` bridge，而不是继续只拿 prompt probe 猜边界。

这次在 Orin 上实际完成了：

- 导出 wall-x flow-action step ONNX
- 用 `TensorRT-Edge-LLM` 的 `action_build` 生成 `action.engine`
- 用自定义 runner 跑单步 denoise
- 用 host 侧 Euler loop 跑完整 flow

硬结果是：

- `action_step_ms = 31.435`
- `denoised_vs_ref cosine = 0.99999797`
- `flow_final_cosine = 0.98523772`
- `flow_step_ms_mean = 31.347`

做了 `--warmup 2 --iters 5` 的热 benchmark 以后，Orin 上更稳定的一组数字是：

- `action_step_ms_mean = 31.329`
- `action_step_ms_std = 0.018`
- `flow_step_ms_mean = 31.335`
- `flow_step_ms_std = 0.032`
- `flow_total_ms_mean = 157.661`
- `flow_total_ms_std = 0.138`

这说明：

> **`TensorRT-Edge-LLM` 已经不只是能承接 wall-x 的 VQA/VLM 子集；通过自定义 bridge，它也已经能在 Orin 上实际承接 wall-x 的 Flow Action 单步 engine + host Euler loop。**

但这同样说明：

> **这条线当前仍然是“custom bridge”，不是 stock `llm_inference` 官方多模态入口直接开箱。**

### 7.5 Flow Action 三方数据对比

把目前 `wall-x` 的 Flow Action 路线放在一起看，最清楚的对比是：

| 路线 | Flow Action latency | 相对 `cpp_infer` |
|---|---:|---:|
| `cpp_infer` | `290.7 ms` | `1.00x` |
| 手工 TRT / TRT-LLM | `176.837 ms` | `1.65x faster` |
| `TensorRT-Edge-LLM` custom bridge | `157.661 ms` | `1.85x faster` |

这里要强调两点：

- `TensorRT-Edge-LLM` 这条数值来自 **custom bridge**
- 它不是 stock `llm_inference` 直接吃 `wall-x` Flow 的官方默认路径

所以更准确的结论是：

> **在 `wall-x Flow Action` 上，`cpp_infer` 是当前最强的可控基线；手工 TRT / TRT-LLM 已经明显赢了它；而 `TensorRT-Edge-LLM` 通过 custom bridge 还能再往前压一点，但这条线不是开箱即用的 stock runtime。**

补充一点边界：

- 这次没有新写 TensorRT/CUDA plugin
- 我们写的是：
  - wall-x -> Edge-LLM 的 ONNX export bridge
  - `action_build` 调用脚本
  - action runner
- 真正复用的是 Edge-LLM 现成的 builder/runtime/plugin 库

另外，这条 bridge 最关键的导出兼容点是：

- ONNX opset 23 会把 RMSNorm 融成 native `RMSNormalization`
- 当前 Edge-LLM action build 路径在 Orin 上不接受这个 op
- 改成 opset 22 后，导出保持为标准 primitive graph，build 成功

再补一个很重要的精度/性能边界：

- 当前这条 `Edge-LLM custom bridge` **没有额外量化**
- 现在跑出来的 `157.661 ms` 不是 INT8 / W8A8 结果
- 而且从当前 `TensorRT-Edge-LLM` 源码看，`export_action` 这条 action-expert 路径目前仍然是 **FP16-only**

所以：

> **Edge-LLM 在 VQA/VLM 上有成熟量化路径，但在我们当前复用的 action-expert 路线上，还不能直接把“量化收益”当成现成选项。**

### 7.5 一眼看完的 Flow 总表

| 路线 | Flow Action latency | 备注 |
|---|---:|---|
| `cpp_infer` | `290.7 ms` | baseline |
| hand TRT / TRT-LLM | `176.837 ms` | stock TRT route |
| Edge-LLM custom bridge | `157.661 ms` | fastest, but custom |

### 7.7 Edge-LLM VQA 量化现状

我们又补做了一轮更干净的 `Qwen2.5-VL-3B-Instruct` VQA benchmark，使用统一脚本单独落目录：

- `Qwen/Qwen2.5-VL-3B-Instruct`
- prompt: `Please describe the image.`
- 3 次 wall-clock：
  - `8142.694 ms`
  - `7816.680 ms`
  - `7034.819 ms`
- 平均：
  - `7664.731 ms`

然后继续尝试：

- `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`

当前结论不是“量化已经更快”，而是：

1. **CPU export 路线失败**
   - 失败点：`aten::_convert_weight_to_int4pack_for_cpu` 不能导出到 ONNX
2. **改成 CUDA export 后，又被环境卡住**
   - 当前 `venv_edge` 里的 `torch` 是 **CPU-only**
   - 结果是 AWQ VLM export 不能直接用 `--device cuda`

我们随后又切到了 `TensorRT-Edge-LLM` 官方更推荐的 `experimental/llm_loader` 路线，并做了两件事：

- 用本地 cached snapshot 路径直接走 `llm_loader.export_all_cli`
- 用 hybrid wrapper 保住 Orin 上的 CUDA torch，同时借用 `venv_edge` 的新依赖

这一步已经比 legacy 路线走得更远，而且现在的新边界已经不同了：

- `llm_loader` 路线已经不再只停在 repack
- 我们已经继续推进到：
  - `model.onnx` 成功导出
  - 通过 `workspace/patch_awq_onnx_dynamic_shapes.py` 把静态 `1x1` 输入改回动态轴
  - `llm_build` 成功生成 `llm.engine`
  - runtime 成功加载 AWQ `llm.engine`

但当前最终硬边界是：

- 这版 AWQ LLM engine 仍然只支持 **decode-only** 的极短输入
- 为了让 builder 通过，当前只能用：
  - `--maxInputLen=2`
- 结果运行时配置里：
  - `maxSupportedInputLength = 2`
- 一旦跑完整 VQA，请求 prefill 长度约 `417`，就会直接失败
- 我们随后又尝试把 builder 直接拉到接近真实 VQA prefill：
  - `--maxInputLen=417`
- 这次 builder 不是简单的长度检查失败，而是在 profile 0 直接报：
  - `IShuffleLayer /_model/model/layers.0/self_attn/Reshape: reshaping failed for tensor: /_model/model/layers.0/self_attn/AttentionPlugin_output_0 reshape would change volume 425984 to 2048`
- 这说明当前 AWQ 导出的 LLM 图在结构上仍然是 decode-only，而不是单纯 profile 参数过小
- 我们随后继续把 ONNX 的 `self_attn` reshape 常量改成了 `[0, 0, 2048]`
- 重新 build 后，`llm.engine` 已经成功生成，runtime 也能成功加载
- 但纯文本 prompt 和完整 VQA prompt 的 `output_text` 仍然是空字符串
- 最小纯文本对照里：
  - FP16 路线会生成 `8` 个 token，输出 `2+2 is 4.`
  - FP16 的 token ids 是：`[17, 10, 17, 374, 220, 19, 13, 151645]`
  - AWQ 路线只记录到 `generated_tokens = 1`
  - AWQ 的 token ids 只有：`[151645]`
  - profile 中没有正常的 `llm_generation` stage
- `special_tokens_map.json` 里 `eos_token = <|im_end|>`
- 我们还加了 runtime 级 `DEBUG_TOP1`：
  - FP16 在 prefill / decode 中正常生成正文 token，最后才落到 `151645`
  - AWQ 在 prefill 阶段首 token 就直接是 `151645`
  - 具体日志里：
    - FP16 prefill 首 token = `17`，score = `29.203125`
    - AWQ prefill 首 token = `151645`，score = `16.937500`
- 我们又加了 runtime 级 `DEBUG_TOPK=5`：
  - FP16 prefill top-5：
    - `(17,29.2031), (785,27.4062), (11613,24.875), (19,24.6719), (1249,24.1094)`
  - AWQ prefill top-5：
    - `(151645,16.9375), (151644,16.0469), (151657,13.0469), (17,12.5625), (785,12.4922)`
- 我们还做了 checkpoint 侧的第一层 `q/k/v` 对照：
  - 原始 dequant 权重和 FP16 权重的 cosine 都接近 `0.99`
  - 同一条 prompt 的 embedding 经过第一层输入投影后：
    - `q_proj cosine = 0.999968`
    - `k_proj cosine = 1.0`
    - `v_proj cosine = 0.999745`
- 这说明当前 AWQ 路线已经越过结构性阻塞，但首 token 直接塌成 `eos`，进入了“能跑但生成链不对”的阶段

也就是说，当前更准确的状态是：

> **AWQ VLM 这条线，legacy export 的环境/算子阻塞已经越过去了；推荐的 `llm_loader` 路线也已经能导出 ONNX、build engine、load runtime，并且补过 reshape 常量后已经能承接 prefill；但 runtime 级 `DEBUG_TOP1/DEBUG_TOPK` 证明 AWQ 在 prefill 第一拍就把 EOS 顶成 top1，而 checkpoint 侧第一层 `q/k/v` 对照又证明问题不在第一层输入投影，说明当前离“可用的量化 VQA backend”还差更深层的生成正确性。**

所以更准确的说法是：

> **Edge-LLM 的 VQA 量化路线值得继续，但在当前 Orin 上，我们还没有拿到 AWQ VLM 的最终 wall-clock；当前新的主阻塞已经不是 repack，而是“如何让 AWQ 导出生成可承接 prefill 的 LLM 图”。**

### 7.8 官方量化主路：`Qwen3-VL-2B + int8_sq`

这条路线已经在 Orin 上完整跑通：

- 量化 checkpoint 成功导出
- `llm_loader` LLM / visual ONNX 成功导出
- `llm_build` / `visual_build` 成功
- `llm_inference` 成功
- 单次 VQA wall-clock: **6245.258 ms**

这说明：

> **官方量化主路在 Orin 上是可用的，但它仍然更像官方 edge runtime 路线，而不是直接把 wall-x 的 VQA 压成最优时延。**

### 7.6 最终建议

- **VQA**：优先保留 `cpp_infer` 作为最稳基线，Edge-LLM 作为可选 backend / router 候选。
- **Flow Action**：继续保留手工 TRT / TRT-LLM 和 Edge-LLM custom bridge 两条线；当前性能上 Edge-LLM custom bridge 最快，但工程形态仍是 custom bridge。
- **不要把 stock `llm_inference` 和 custom bridge 混成一条线。** 前者是官方路径，后者是我们为 wall-x 补的适配层。
