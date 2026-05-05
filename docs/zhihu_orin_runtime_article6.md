# TensorRT-Edge-LLM、trt-llm、cpp_infer 和我们自研 runtime，其实走的是同一条路

> 前五篇写到这里，wall-x 在 Orin 上的主线其实已经跑得很清楚了。我们最早做的不是“先选框架”，而是先把问题拆开：模型怎么导、runtime 怎么跑、量化怎么落、算子怎么融合、最后到底是哪一段在拖后腿。写到第五篇的时候，Flow Action 已经能在 `cpp_infer`、手工 TRT / TRT-LLM、以及 TensorRT-Edge-LLM custom bridge 这三条路上分别跑出结果；VQA 也已经能在 `cpp_infer`、手工 TRT / TRT-LLM、官方 Edge-LLM VLM backend 上做对照。到这一步，一个很关键的判断就越来越清楚：**TensorRT-Edge-LLM 不是另一套世界观，它和我们自己的 `cpp_infer + TRT runtime`，其实是同一类工程，只是 NVIDIA 把很多上层东西先替你写好了。**

**TL;DR**
- `cpp_infer`、手工 TRT / TRT-LLM、TensorRT-Edge-LLM，本质上都在做同一件事：**把模型变成 engine，然后把 runtime 组织好**
- 真正决定最终效果的，不是“是不是官方框架”，而是：
  - 你有没有把大头算子压下去
  - 你有没有把短链融合掉
  - 你有没有把 CPU dispatch / KV cache / decode loop / serving 组织好
- 这也是为什么我们最后会发现：
  - **VQA 上，`cpp_infer` 依然是最稳基线**
  - **Flow Action 上，手工 TRT / TRT-LLM 和 Edge-LLM custom bridge 已经明显压过 `cpp_infer`**
- 但这并不意味着“要不要拥抱 Edge-LLM”这个问题已经结束了；更准确的结论是：
  - **Edge-LLM 值得用**
  - **但它的 stock path 只适合一部分 VLM / VQA**
  - **Flow 这类强控制流 VLA，仍然会逼你自己保留 runtime**

> **本文环境（Orin 实机）**：JetPack `6.2.1`，Python `3.10`，CUDA Toolkit `12.6.11`，cuDNN `9.3.0.75`，cuBLAS `12.6.1.4`，TensorRT `10.3.0.30`，TensorRT-LLM `0.12.0`，TensorRT-Edge-LLM `release-0.7.0`（源码编译），Triton `3.6.0`，PyTorch `2.5.0a0+872d972e41.nv24.08`，flash-attn `v2.8.3`，`CUTLASS v4.1.0`，profiling 工具为 `nsys 2024.7.1` 与 `ncu`。

---

## 一、先把结论说死：这三条路线不是三种宇宙

如果只看名字，很容易把这三条线看成完全不同的路线：

- `cpp_infer` 看起来像“自研”
- `TRT / TRT-LLM` 看起来像“性能优化”
- `TensorRT-Edge-LLM` 看起来像“官方端侧方案”

但我们这轮在 Orin 上反复跑下来，结论其实很明确：

> **它们不是三种宇宙，而是三种实现方式。**

真正的共通点只有一个：

1. **导出 / 构图**
   - 把模型从原始 checkpoint 变成可 build 的图
2. **engine build**
   - 用 TensorRT 或对应 builder 把 ONNX / native graph 编成 engine
3. **runtime orchestration**
   - 管 KV cache
   - 管 rope / positional cache
   - 管 prefill / decode
   - 管 sampling
   - 管 serving / batch / stream
   - 对 Flow 还要管 ODE / Euler / postfix loop

这也是为什么我们最后会发现：

- `cpp_infer` 不是“过时方案”，它本质上就是一套更轻、更贴 wall-x 的 runtime
- `TRT / TRT-LLM` 不是“另一条线”，它只是把 engine 执行和一部分 runtime 抽象提前帮你搭好了
- `TensorRT-Edge-LLM` 更不是“另一个宇宙”，它只是 NVIDIA 版的端侧 runtime + builder + plugin 集合

所以真正要问的，不是：

> “要不要完全切到 TensorRT-Edge-LLM？”

而是：

> “哪些部分可以直接借它的 stock path，哪些部分必须保留我们自己的 runtime？”

---

## 二、为什么我们最后会越来越像 Edge-LLM，而不是相反

这不是因为我们“跟风”，而是因为我们在 Orin 上跑到最后，问题收敛成了同一类东西。

### 2.1 VQA 的问题不是“能不能跑”，而是 runtime 组织

VQA 这条线上，我们已经拿到了很清晰的数字：

- `cpp_infer`: **851.2 ms**
- 手工 TRT / TRT-LLM dynamic decode: **943.018 ms**
- TensorRT-Edge-LLM 官方 VLM 路线: **5.1s - 6.8s**
- 一条更干净的 unquantized `Qwen2.5-VL-3B-Instruct` 基线：**7664.731 ms**

这组数值说明了一件事：

> **Edge-LLM 在 VQA 上不是不能跑，而是它的默认 stock 路线还没把“wall-x 这种输入组织”吃到最佳。**

它能做的事情是：

- VLM / VQA 主干
- 统一的 prompt / chat template / multimodal input contract
- 既有 builder / runtime / serving 路径

但它还没自动解决这些事情：

- 大 engine 数量
- decode / prefill 切分方式
- 和 wall-x 自定义输入布局的完全对齐

所以在 VQA 上，最终比较合理的姿势反而是：

> **保留 `cpp_infer` 做默认基线，Edge-LLM 做 backend / router 候选。**

这不是保守，而是因为：

- `cpp_infer` 现在已经够稳
- Edge-LLM 的 stock path 还没在 wall-x 的 VQA case 上形成绝对性能优势

这里还有一个很容易让人误解的问题：

> **为什么官方 TensorRT-Edge-LLM 的 VQA 路线，看起来有时还会比历史上的 PyTorch baseline 慢？**

这个现象表面看起来反直觉，但其实并不矛盾。

先把历史数据摆清楚：

- Python + SDPA 的 VQA baseline：**8681 ms**
- Python + FA2 的 VQA baseline：**6360 ms**
- C++ libtorch VQA：**3469 ms**
- 当前这轮官方 Edge-LLM VLM 路线：大约 **5.1s - 6.8s**

这里真正需要分清的，不是“是不是 Python”，而是：

1. **历史 PyTorch baseline 本来就不是纯烂实现**
   它已经在用：
   - SDPA / FA2 这类 fused attention
   - 比较直接的 tensor flow
   - 相对轻的外层 runtime 组织

2. **Edge-LLM stock VLM path 更通用，也更重**
   它默认承接的是一套更标准化的端侧 runtime 契约：
   - multimodal 输入组织
   - tokenizer / chat template
   - visual engine + llm engine 的协同
   - 更完整的 runtime / serving 路径

3. **VQA 在 Orin 上本来就更吃 runtime orchestration**
   对 wall-x 这种场景来说，真正拖时间的早就不只是单个 GEMM，而是：
   - prefill / decode 组织
   - engine 数量
   - profile 切换
   - host 侧同步
   - multimodal glue

所以这个问题真正的答案不是：

> “为什么 TensorRT 比 PyTorch 慢？”

而是：

> **为什么一条更通用、更产品化的 edge runtime，在 wall-x 这种高度特化的 VQA 场景里，还没比已经贴任务打磨过的 baseline 更快？**

答案也很直接：

> **因为 VQA 在 Orin 上，决定上限的已经不是“是不是 TensorRT”，而是 runtime 有没有足够贴 wall-x 的输入组织、prefill / decode 切分和 serving 方式。**

这也是为什么我们最后没有把结论写成“全面切到 Edge-LLM”，而是写成：

- `cpp_infer` 保留为默认基线
- Edge-LLM 作为 backend / router 候选保留

换句话说，**不是 PyTorch 战胜了 TensorRT，而是“更贴 wall-x 的 runtime 组织”暂时赢了“更通用的官方 edge runtime”。**

### 2.2 Flow Action 的问题不是“模型不对”，而是控制链更像 runtime 而不是纯 LLM

Flow 的结论更硬：

- `cpp_infer`: **290.7 ms**
- 手工 TRT / TRT-LLM: **176.837 ms**
- Edge-LLM custom bridge: **157.661 ms**

也就是说：

> **Flow Action 上，性能最强的是 Edge-LLM custom bridge。**

但这条线要注意一个边界：

- 它不是 stock `llm_inference`
- 它不是官方 VLM 直连路径
- 它是我们自己写的 ONNX export bridge + `action_build` wrapper + custom runner + host Euler loop

所以更准确的判断应该是：

> **Flow Action 已经证明 TRT / Edge-LLM 这条 runtime 路线是有价值的；但 stock runtime 和 custom bridge 不能混成一条线。**

这点对后续最重要，因为它决定了我们不是“要不要拥抱官方路线”，而是：

- **VQA**：官方 backend + 自研 runtime 双线并行
- **Flow**：继续保留自研 runtime 和 custom bridge

#### 2.2.1 还有一层不会汇合的：API 形态本身

到这里很容易得到一个温和的结论 ——“反正大家都在做 engine + runtime，那就慢慢往官方靠”。但 Flow Action 这条线其实顺手暴露了一个**不会被时间抹平**的差异：

> **TRT-LLM / Edge-LLM 的 stock runtime 本质上是 server-style 的；wall-x 这条 Flow 控制链需要的是 in-process、同步、单请求的 runtime。**

- TRT-LLM `Executor::enqueueRequest(reqId, ...) → awaitResponses(reqId)`：长寿命进程、异步、N 个请求复用一张卡。它的优化目标是 token/s 和 GPU 利用率
- Edge-LLM stock VLM path：本质继承同一形态（reqId / iteration / response），只是多模态前处理替你拼好
- wall-x Flow Action：单请求、batch=1 是常态、调用方是 ROS / 控制 loop、强延迟敏感（动作链一拍 ≈ 几十毫秒）

这两种 API 形态没有“后者会逐渐变成前者”的演化路径——它们对应的是**两类客户**：

- 一类客户的 KPI 是“单卡每秒服务多少 prompt”
- 另一类客户的 KPI 是“一只机械手能不能在 50 ms 内闭环”

第二类客户从来不会需要 `enqueueRequest`，强行套上去只会多一层额外的队列、wakeup 和 host 同步开销。所以 Flow 这条线"继续保留自研 runtime"不是阶段性现象，而是**结构性结论**。

---

## 三、为什么说 Edge-LLM 的思路和 cpp_infer 是一致的

我们后来读 Edge-LLM 源码的时候，一个很自然的感受就是：

> **它其实和我们的 cpp_infer + TRT runtime，思路是一样的。**

这不是口头上的相似，而是工程上的一致。

### 3.1 两边都在做导出 / 构图

我们的 `cpp_infer` 主线一直是：

- 先把模型导成可以 build 的图
- 再把图喂给 engine builder
- 再由 runtime 去组织 KV / decode / sampling / serving

Edge-LLM 现在官方更推荐的也是这套思路：

- `llm_loader` 负责 checkpoint / weight / ONNX export
- `llm_build` 负责 engine build
- `llm_inference` / server 负责 runtime

这意味着：

> **它不是替代 runtime 思想，而是把 runtime 思想产品化了。**

### 3.2 两边都在做 runtime orchestration

Edge-LLM 的 runtime 里你能看到一模一样的东西：

- `inputs_embeds`
- `past_key_values`
- `rope_rotary_cos_sin`
- `context_lengths`
- `kvcache_start_index`
- `last_token_ids`
- `outputIds`
- `outputTexts`
- `sampling workspace`
- `streaming / batch / serving`

这和我们自己的 runtime 关注点其实完全一致。

所以当我们后来在 Edge-LLM 里看到：

- `llm_loader`
- `llm_build`
- `llm_inference`
- `action_build`
- `visual_build`

我们能很快理解它为什么能工作：

> **因为它不是“魔法”，而是把我们本来就在做的事情，拆成了更标准化的模块。**

### 3.3 这也是为什么 custom bridge 能成立

Flow Action 之所以能通过 custom bridge 跑到 Edge-LLM 上，不是因为我们“曲线救国成功了”，而是因为：

- 它的底层 engine / plugin / runtime 路径，和我们自己的 runtime 思路是兼容的
- 我们只是把 wall-x 的 flow-action step 翻译成了 Edge-LLM 的 `action_build` 能接受的形态
- 外层 Euler loop 仍然在 host 上跑

这件事非常关键，因为它说明：

> **Edge-LLM 不是把 wall-x 的 runtime 思想替你消灭了，而是把它吸收进了一个更标准的外壳里。**

---

## 四、为什么 AWQ 这条线最后反而最说明问题

这轮最有价值的不是 benchmark 本身，而是 AWQ 路线把很多隐性的边界都暴露出来了。

### 4.1 它先后跨过了很多结构性阻塞

我们在 Orin 上把 `Qwen2.5-VL-3B-Instruct-AWQ` 这条线一步步往前推，经历了：

- legacy export 的环境问题
- `gptqmodel` / CPU-only torch 问题
- `llm_loader` repack 问题
- ONNX 动态轴问题
- `llm_build` profile 问题
- `llm.engine` build / runtime load

最终确实能把 engine 建出来，也能让 runtime 跑起来。

### 4.2 但最后却卡在“首 token 就是 eos”

最硬的 runtime 证据是：

- `FP16`：
  - `output_text = 2+2 is 4.`
  - top-5 是正常正文 token
- `AWQ`：
  - `output_text = ""`
  - `output_ids = [151645]`
  - `151645 = <|im_end|>`
  - `DEBUG_TOP1` 第一拍就是 `<|im_end|>`
  - `DEBUG_TOPK=5` 里 `<|im_end|>` 仍然排第一
- 更重要的是，这不是 greedy 的问题：
  - 我们把 `temperature=1.0, top_k=50` 打开后重测
  - AWQ 仍然在 prefill 第一拍直接给出 `<|im_end|>`
  - `output_text` 依然为空

这时候问题已经不再像“采样策略不对”，而更像是 **prefill 后的 logits 分布本身就塌了**。

我们还继续查了：

- 第一层 `q/k/v` 输入投影
  - 和 FP16 基本对齐
  - `q_proj cosine = 0.999968`
  - `k_proj cosine = 1.0`
  - `v_proj cosine = 0.999745`
- 第一层 `LayerNorm + MLP`
  - `input_layernorm` 和 `post_attention_layernorm` 还能看
  - 但 `mlp_out` 在 AWQ 路线里直接塌成了全 0

这也解释了为什么 `AWQ` 的 `DEBUG_TOP1` 会直接顶到 `eos`：

- 不是第一层输入投影已经坏掉
- 也不是采样把本来正常的分布选歪了
- 而是更深一层的数值链已经开始把输出压扁

这说明 AWQ 的问题已经不是“没有量化收益”，而是：

> **量化后生成链的数值语义已经坏了。**

也就是说，AWQ 最后给我们的不是“更快的答案”，而是一个非常重要的边界事实：

> **量化不是终点，量化之后还能不能把 logits 语义保住，才是真问题。**

这也正好说明为什么我们在第五篇以后越来越往 runtime 和融合走：

- 量化吃掉了大头
- 接下来最值钱的是短链、dispatch、graph、cache、serving
- 而不是继续盲目追一个“看起来更激进”的 checkpoint

这段话再落到主工程入口上，意思也已经很明确：

- `wall_x.serving.VQAPolicy(backend=edge)` 已经可以直接切到 Edge-LLM
- `wall_x.serving.WallXPolicy` 也已经能在 `backend=edge` 下把观测里的图像交给 Edge-LLM
- 所以 **VQA 的 Edge-LLM 侧已经足够进入主工程入口**
- 但 `Flow` 还没有同样成熟的服务层收口，所以下一步更自然的投入点仍然是把 `edge_llm_wallx_flow` 这条 custom bridge 接入 `wall_x.serving`，而不是继续深挖 VQA 量化

### 4.3 官方量化主路也试了，结论更直接

我们后来又补试了两条官方量化主路：

- `fp8`
  - 量化和 `llm_loader` 导出都成功
  - 但 `llm_build` 在 Orin 上直接报：
    - `Networks with FP8 Q/DQ layers require hardware with FP8 support.`
  - 这不是脚本问题，是 **硬件边界**

- `int8_sq`
  - `Qwen3-VL-2B + int8_sq` 已经完整跑通
  - 单次 VQA wall-clock：**6245.258 ms**

所以 VQA 量化这条线最后收敛成了三个很清楚的判断：

- **AWQ**：结构上能跑，但生成链坏了
- **FP8**：导出能通，但 Orin 硬件不支持 build
- **INT8-SQ**：能跑通，但仍然没有把 wall-x 的 VQA 压到最优时延

---

## 五、所以最后该怎么选

如果把这轮所有事实压成最终建议，大概就是这三条：

### 5.1 VQA

- 默认基线：`cpp_infer`
- 可选候选：`TensorRT-Edge-LLM` backend / router
- 不建议为了“官方感”直接把 VQA 默认入口切离 `cpp_infer`

原因很简单：

- 当前 `cpp_infer` 还是最稳的
- Edge-LLM 官方 VLM 路线在 wall-x 风格 case 上还没变成绝对性能答案

这里还有一个很容易让人困惑的问题：

> **既然都是 TensorRT / edge runtime，为什么 VQA 这条官方 Edge-LLM 路线有时还会比历史上的 PyTorch baseline 慢？**

这个现象表面看起来反直觉，但其实并不矛盾。

先把历史数据摆出来：

- Python + SDPA 的 VQA baseline：**8681 ms**
- Python + FA2 的 VQA baseline：**6360 ms**
- C++ libtorch VQA：**3469 ms**
- 当前这轮官方 Edge-LLM VLM 路线：大约 **5.1s - 6.8s**

这里真正需要分清的，不是“是不是 Python”，而是：

1. **PyTorch baseline 本来就不是纯烂实现**
   历史那条 `Python + FA2` baseline 已经在用：
   - FA2 / SDPA 这类 fused attention
   - 相对直接的 tensor flow
   - 没有太重的通用 engine/runtime 外壳

2. **Edge-LLM stock VLM path 更通用，也更重**
   它要处理的是一套更标准化的端侧 runtime 契约：
   - multimodal 输入组织
   - tokenizer / chat template
   - visual engine + llm engine 的协同
   - 更完整的 runtime / serving 路径

3. **VQA 在 Orin 上本来就更吃 runtime orchestration**
   对 wall-x 这种场景来说，VQA 的大头早就不只是单个 GEMM，而是：
   - prefill / decode 组织
   - engine 数量
   - profile 切换
   - host 侧同步
   - multimodal glue

所以这个问题的更准确表述不是：

> “为什么 TensorRT 比 PyTorch 慢？”

而是：

> **“为什么一个更通用、更产品化的 edge runtime，在 wall-x 这种特化 VQA 场景里，还没比已经高度贴任务的 PyTorch/C++ baseline 更占优？”**

答案也很直接：

> **因为在 VQA 上，决定上限的已经不是底层算子是不是 TensorRT，而是 runtime 有没有足够贴 wall-x 这条特定路径。**

这正是为什么我们在这轮实验之后，没有把结论写成“全面切到 Edge-LLM”，而是写成：

- `cpp_infer` 继续做默认基线
- Edge-LLM 作为 backend / router 候选保留

换句话说，**PyTorch 并不是“解释器赢了 TensorRT”**；更准确地说，是历史 baseline 里那条更贴 wall-x 的 runtime 组织，暂时赢了当前这条更通用的官方 edge runtime。

### 5.2 Flow Action

- 继续保留手工 TRT / TRT-LLM
- 继续保留 Edge-LLM custom bridge
- 当前性能最强的是 Edge-LLM custom bridge

但它仍然是：

- custom bridge
- 不是 stock runtime

### 5.3 量化

- 量化已经证明自己能把大头压下去
- 但 AWQ 这条线已经给出一个很硬的反例：
  - **不是所有量化 checkpoint 都会自然变成可用 backend**
  - 首 token logits 语义守不住，后面就没有意义

所以量化之后更重要的不是“再追一个更狠的量化格式”，而是：

> **让 runtime、graph、fusion、cache、sampling 这些东西真正接住量化后的模型。**

### 5.4 验证：唯一不该汇合的一层

前面三条建议都是“向官方靠”——VQA 把 cpp_infer 留作基线、Flow 借 Edge-LLM custom bridge、量化复用 ModelOpt 那一套。但有一条线不能这样想：

> **VLA 的验证轨，不能复用 LLM 圈的验证习惯。**

LLM 圈（包括 TRT-LLM 自带的精度回归、Edge-LLM 的 acc test）默认的验证语言是 `perplexity` / `lm-eval`。这套语言对一个事实是有效的：**生成的下一个 token 在分布上没飘**。但它对 VLA / Flow Action 的常见失效模式**完全无感**：

- 动作链整体偏移几毫米 —— perplexity 看不到
- 抓取力度被量化压扁 —— perplexity 看不到
- AWQ 那种"首 token 直接 eos"虽然 perplexity 看得到，但等 perplexity 跌穿阈值的时候，机械手早就把杯子捏碎了

第四章 AWQ 的故事其实是个很好的注脚：我们最后定位到 `mlp_out` 塌成全 0，靠的不是任何 perplexity 工具，而是**逐层 cosine + 第一拍 logits TOPK** 的双轨对比。这种验证粒度，stock TRT-LLM/Edge-LLM 的 acc test 都不会自带，必须自己写。

所以再往后做，验证这一层会主动**和 TRT-LLM/Edge-LLM 分叉**：

- **数值轨**：自己持有 `cosine + 相对 L2 + NaN/Inf` 三件套，按层、按 chunk、按一次完整 Flow 推理多粒度比对
- **任务轨**：自己持有 ReplaySuite，把"同一段 ROS 输入回放在 FP16 / 量化 / 不同 backend 三条路上"做端到端动作差异比对

这条线已经在隔壁 `embodied_ai_os_runtime/quantization/validators/` 落成代码（M5.5 已交付，5090 e2e 跑通），不是设想。

换句话说：

> **量化层殊途同归是好事，runtime 层各走各路是必然，验证层各走各路是底线。**

---

## 六、收尾

我们这几篇在 Orin 上绕了一大圈，最后其实把一个很朴素的结论讲清楚了：

> **TensorRT-Edge-LLM 不是另一种思路，它只是把我们已经在做的 engine + runtime 思路，做成了官方、模块化、可维护的版本。**

所以后续最合理的路线也很清楚：

- **VQA** 继续保 `cpp_infer`，Edge-LLM 做候选 backend / 路由
- **Flow** 继续保手工 TRT / TRT-LLM 和 Edge-LLM custom bridge
- **量化** 继续做，但不要把“能 build”误认为“能用”

如果说前五篇讲的是：

- 怎么把 wall-x 跑起来
- 怎么把大头压下去
- 怎么把 runtime 里的短链一段段抠掉

那这一篇收尾想表达的就是：

> **到了最后，决定你是不是能把系统做成的，不是“官方还是自研”，而是你有没有把 engine、runtime、量化、以及控制流真正接成一条能稳定闭环的链。**
