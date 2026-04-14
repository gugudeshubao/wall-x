# 08. VLA Runtime 的 benchmark 应该怎么设计：从 cold start 到 camera-to-action latency

只要开始认真做 `VLA Runtime`，很快就会发现一个现实：

> 没有 benchmark，很多优化讨论其实都只是猜测。

这句话在普通 `LLM` 场景里已经成立，在 `VLA` 场景里只会更严重。

因为 `VLA` 的性能问题，远远不只是“每秒生成多少 token”这么简单。  
它真正关心的是一整条决策链：

- 输入进来以后多久能处理完
- 图像、文本、状态的组织成本是多少
- 主干前向到底慢在哪
- 动作多久能真正输出
- 控制周期稳不稳定

如果没有一套专门面向具身场景的 benchmark，你就很容易陷入一种错觉：

> 看起来模型在 GPU 上运行了，但实际上你根本不知道系统哪里慢、哪里值得优化、哪里只是心理安慰。

这篇文章就只讲这个问题：

> `VLA Runtime` 的 benchmark 到底应该怎么设计？

## 一、为什么不能直接照搬 LLM 的 benchmark 习惯

很多 `LLM` benchmark 的核心指标都很熟悉：

- tokens/s
- TTFT
- TPOT
- 吞吐
- batch 扩展性

这些指标当然有价值，但对 `VLA` 来说不够。

因为 `VLA` 面对的不是纯文本输出链，而是：

- 多模态输入
- batch=1 更常见
- 端侧与云侧差异大
- 动作输出和控制链路是最终目标

所以对 `VLA` 来说，如果 benchmark 只告诉你：

> 模型每秒能生成多少 token

那其实离真实系统还有很远。

具身场景更关心的是：

- camera-to-action latency
- 控制周期 jitter
- 预处理时间是否压过主干前向
- batch=1 steady-state 是否稳定
- 固定 shape 与动态 shape 的差别

也就是说，`VLA` 的 benchmark 不该只是一个模型指标面板，而更像一个系统剖面图。

## 二、先分清楚：你到底在 benchmark 什么

我觉得 `VLA Runtime` 的 benchmark 至少应该分成三层。

### 第一层：模型内部算子时间

这一层回答的是：

- 哪些 kernel 最慢
- `permute/unpermute` 占多少
- attention 前后的 projection 占多少
- `MLP MoE` 占多少
- 图片预处理和主干前向谁更重

这层更多是给算子优化和 fused path 设计用的。

### 第二层：单请求推理时间

这一层回答的是：

- cold start 多久
- warm start 多久
- prefill 多久
- decode 多久
- 端到端一次推理多久

这层更多是给 runtime 设计和服务部署用的。

### 第三层：控制链路时间

这一层回答的是：

- camera-to-action latency 多久
- 在连续控制下 jitter 多大
- steady-state latency 是否稳定
- 长时间运行会不会明显漂移

这层才是真正和机器人系统可用性直接相关的。

如果少了第三层，你可能只是做了一个“模型 benchmark”，而不是“具身系统 benchmark”。

## 三、VLA benchmark 的最小指标集合

如果让我先给一个最小可行集合，我会建议至少测下面这些。

### 1. Cold Start Latency

这是第一次真正启动推理链路的时间。

它通常包括：

- 模型加载
- 权重搬运
- runtime 初始化
- 可能的后端预热

这个指标决定了：

- demo 的可用感受
- 在线服务冷请求体验
- 端侧系统重启后的恢复成本

### 2. Warm Start Latency

也就是模型和运行时已经就绪之后，第一次正式推理请求的时间。

这个指标很重要，因为很多系统在 cold start 之后仍然会经历：

- kernel 初次编译
- memory pool 预热
- graph capture 前的准备

如果不单独测它，很容易把不同阶段的成本混在一起。

### 3. Prefill Latency

对多模态 `VLA` 来说，prefill 不只是“文本前缀处理”，而更像：

- 图像 token 进入模型
- 文本模板进入模型
- 状态输入进入模型
- 首轮主干计算完成

这通常决定：

> 系统从接到观测到进入有效推理状态的第一段延迟。

### 4. Decode / Predict Latency

如果输出是语言 token，就测 decode。  
如果输出是动作块，就测 predict/action decode。

这一段的意义在于：

- 它直接决定连续控制中的迭代开销
- 它通常更能反映 batch=1 场景的真实运行特征

### 5. End-to-End Camera-to-Action Latency

这是我觉得最不能少的指标。

它不再只看模型内部，而是从：

- 观测输入
- 前处理
- 模型推理
- 输出整理
- 动作可消费结果生成

整条链路去看时间。

这个指标之所以重要，是因为机器人系统最后要的是动作，而不是中间 token。

### 6. Steady-State Latency

很多系统第一次跑得不慢，但连续跑几十次、几百次后会出现：

- latency 漂移
- 内存碎片
- 额外同步
- 某些路径偶发抖动

所以 batch=1 的 steady-state latency 一定要单独看。

### 7. Jitter

平均值并不能代表控制系统真实体验。

如果某条控制链平均是 `20ms`，但时不时跳到 `60ms`，那它在机器人上往往就是不可接受的。

所以 `P50 / P90 / P99` 这类分位数统计，在 `VLA` benchmark 里非常必要。

### 8. Memory Footprint

显存不只是一个“能不能放下模型”的问题。

它还会影响：

- 是否能做更激进的缓存
- 能否容纳图像和状态中间结果
- 是否能上更大的 batch 或更大的动作 horizon
- 是否适合端侧部署

## 四、为什么 tokens/s 在 VLA 里经常会误导你

这一点我想单独说。

很多时候，一看到模型输出能力，大家就会下意识地问：

> tokens/s 多少？

这个问题不能说错，但在 `VLA` 里经常不够关键，甚至容易误导。

原因是：

### 1. 输出不一定是自然语言 token

很多 `VLA` 模型最后关心的是动作块，不是连续长文本。

### 2. prefill 往往更重

多模态输入、图像编码、状态拼接，常常让 prefill 比想象中更关键。

### 3. batch=1 更常见

这时 kernel launch、layout 变换和中间回写的固定成本占比会更高，而这些东西不一定会被 tokens/s 很好地表达出来。

### 4. 系统目标不是“多产 token”，而是“更快地产出可执行动作”

所以 `tokens/s` 可以测，但不应该作为 `VLA Runtime` 的主指标。

## 五、benchmark 设计里最容易被忽视的几个点

我觉得有四个点特别容易被漏掉。

### 1. 图像预处理必须单独拆开

很多多模态模型一测端到端，就把图像 resize、格式转换、张量组织全部和模型前向混在一起。

这样会导致你很难判断：

- 是模型慢
- 还是前处理慢
- 还是两者都慢

所以图像预处理一定要单独计时。

### 2. 动态 shape 和固定 shape 要分开测

如果你后面考虑：

- 低精度预打包
- CUDA Graph
- 局部 `megakernel`

那固定 shape 和动态 shape 的表现差别会非常大。

这两类场景不拆开测，后面做优化时很容易判断失真。

### 3. 端侧和云侧不能共用一套结论

同一个模型，在端侧和云侧可能会得到完全不同的热点排序。

云侧可能更受益于：

- 通用 fused op
- 更强的低精度线性层

端侧则可能更受限于：

- kernel launch
- memory movement
- batch=1 下的固定开销

所以 benchmark 结果必须和部署目标绑定。

### 4. 平均值不够，必须看分布

尤其在机器人场景里，分位数往往比均值更重要。

你至少应该长期记录：

- 平均值
- 中位数
- P90
- P99

这决定你看到的是“系统平时多快”，还是“系统最坏时会不会拖垮控制链路”。

## 六、如果 benchmark 是为了指导 fused path，该怎么切层

我更建议 benchmark 从一开始就按未来 runtime 的执行层去切，而不是只按 Python 函数名切。

更实用的切法通常是：

### Stage 1: Input Adapter

- 图像读取
- 图像 resize / normalize
- 文本模板组织
- 状态张量组织

### Stage 2: Token & Layout Preparation

- token 拼接
- token type 标记
- routing 相关预处理
- shape / layout 组织

### Stage 3: Core Forward

- attention 相关
- `MLP MoE`
- projection
- `permute / unpermute`

### Stage 4: Output Adapter

- 语言输出解码
- 动作输出整理
- 控制接口转换

这样切的好处是，benchmark 结果天然能反哺 runtime 设计。

## 七、一个真正能指导优化的 benchmark 输出长什么样

我理想里的 benchmark 报告，不该只是一行：

> latency = xx ms

而应该至少像这样：

- Deployment Target: `cloud` / `edge`
- Input Profile: 图像尺寸、prompt 长度、状态维度、action horizon
- Cold Start
- Warm Start
- Prefill
- Decode / Predict
- End-to-End Camera-to-Action
- Steady-State P50/P90/P99
- Memory Footprint
- Stage Breakdown

如果是做算子优化，还应该再有一层：

- Top kernels by time
- kernel launch count
- major tensor allocation sites
- shape stability summary

只有这种输出，才能真正支撑你去回答：

- 第一条 fused 路径该选哪里
- 低精度该先放哪里
- `megakernel` 有没有必要

## 八、VLA Runtime 为什么必须把 benchmark 当成第一组件

这一点我想再强调一次。

我越来越觉得，`VLA Runtime` 的第一组件不是：

- kernel backend
- 低精度库
- 执行调度器

而是 benchmark。

因为没有 benchmark，你就没有共同语言。

没有共同语言，后面所有讨论都只能变成：

- “我感觉这里慢”
- “我觉得 attention 应该先做”
- “我猜量化可能收益大”

但真正的 runtime 不该建立在感觉上，而应该建立在度量上。

## 九、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的 benchmark 设计，不该照搬 `LLM` 的 tokens/s 逻辑，而应该围绕 cold start、prefill、predict、camera-to-action latency、steady-state 和 jitter，去真实刻画一条具身推理链。

只有当这套 benchmark 建起来之后，后面的很多问题才有可能真正被答对：

- 第一条 fused path 先做哪里
- 低精度到底值不值得先做
- 局部 `megakernel` 是否真的有意义

也就是说，在 `VLA Runtime` 这件事上：

> benchmark 不是附属品，而是整个系统的起点。
