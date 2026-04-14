# 09. Wall-X 的第一条 fused 路径：为什么我会先做 MLP MoE

前面几篇文章里，我已经把 `VLA Runtime` 这条线的基本判断说清楚了：

- `VLA` 需要一套不同于 `LLM` 的推理优化栈
- 不能一上来就空谈通用平台
- 要先做 benchmark、再做 fused path、再接低精度、最后再评估局部 `megakernel`

那接下来真正进入实现阶段时，第一个绕不过去的问题就是：

> 如果只能先打一条 fused 路径，应该先选哪一条？

很多人第一反应会是：

> 当然先做 attention。

这个判断并不荒谬，因为 attention 确实是最显眼、也最容易被首先想到的热点。  
但如果把问题放回 `Wall-X` 这类具身多模态项目，我现在越来越倾向于另一种答案：

> 如果模型里存在明显的 expert 结构，那么第一条 fused 路径，很多时候更适合优先落在 `MLP MoE`。

这篇文章就只讲这件事。

## 一、为什么“第一条 fused 路径”这个选择很重要

很多推理优化最后做不下去，不是因为没有思路，而是因为第一刀砍错了地方。

第一条 fused 路径承担的任务，不只是“拿一点加速”，它实际上会决定后面很多事情：

- 你的 runtime 接口怎么设计
- 权重预打包会不会做
- layout 会不会过早定死
- 低精度后面好不好接
- 团队能不能快速拿到 first win

所以这一步不是“先挑一个看起来最酷的方向”，而是：

> 先挑那条最容易形成稳定正收益、也最容易成为后续基线的路径。

对 `Wall-X` 这种项目来说，这一点尤其重要。  
因为它不是一个纯 `LLM`，也不是一个只有几层线性变换的轻模型，而是同时有：

- 多模态输入
- token 类型路由
- attention expert 路径
- `MLP MoE`
- 动作相关输出

在这种结构下，“先做哪条 fused path”，本身就是 runtime 设计的一部分。

## 二、为什么很多人会本能地先选 attention

先替这个直觉说句公道话。

大家优先想到 attention，通常是因为：

- 它最显眼
- 论文和优化文章最多
- 各种内核后端讨论最集中
- 一提加速，社区默认就先谈它

如果是通用 `LLM serving`，这个顺序有时是合理的。  
因为 attention 确实经常是整个系统里最关键的一环。

但 `VLA` 和 `Wall-X` 这类项目有个不同点：

> attention 虽然重要，但不一定是第一条最适合自己下手的 fused 路径。

这里最大的误区就在于：

> “最重要的模块”不等于“最适合作为第一条 fused 路径的模块”。

这是两件不同的事。

## 三、为什么 MLP MoE 往往更适合作为 first win

我越来越倾向于先做 `MLP MoE`，主要有五个原因。

### 1. 它的结构更规则

典型的 `MLP MoE` 路径，逻辑通常是：

- token routing
- token permutation
- `gate_proj`
- `up_proj`
- 激活
- elementwise combine
- `down_proj`
- weighted accumulation / scatter

这条链虽然不短，但边界很清楚。

相较之下，attention 路径里往往混着更多额外因素：

- mask
- rope
- KV cache
- 不同 token 类型的布局切换
- QKV 的不同 head 组织

也就是说，`MLP MoE` 更像是一条适合“先把边界打通”的路径。

### 2. 更容易做成高精度 fused 基线

如果第一版的目标是先做一个 `BF16/FP16` fused 版本，那么 `MLP MoE` 的实现压力通常更小。

你需要解决的核心问题更集中在：

- expert 分桶
- weight layout
- `gate/up/down` 的执行顺序
- `SwiGLU` 或类似激活的融合
- 回写和累加

这些问题虽然不简单，但它们比 attention 的整体语义更集中，也更容易通过逐步替换验证正确性。

### 3. 更适合和低精度结合

这一点非常关键。

如果你后面一定会做低精度，那么 `MLP MoE` 本身就是很自然的切入点。

因为低精度最容易产生价值的地方，往往是：

- 大量线性层
- 权重带宽占比高的路径
- 易于预打包的结构
- 易于把 dequant 和 epilogue 融进 kernel 的路径

而 `MLP MoE` 恰好高度符合这些特点。

相比之下，attention 虽然也能做低精度，但它的系统依赖更多，调试链路也更长。

所以如果把“第一条 fused path”看成“后续低精度的前哨站”，`MLP MoE` 往往更顺手。

### 4. 更容易形成清晰的 benchmark 反馈

一条好的 first win 路径，除了要能优化，还要能被验证。

`MLP MoE` 的另一个优势在于：

> 它的优化前后对比通常更容易被 benchmark 清楚地反映出来。

你更容易看出：

- routing 开销占多少
- `gate/up/down` 各占多少
- fused activation 省了多少 launch
- 预打包和 workspace 复用有没有带来真实收益

这对 runtime 的早期迭代很重要。

因为你不仅需要“加速了”，还需要知道：

> 到底是哪里加速了，为什么加速了，后面还能继续怎么推进。

### 5. 更容易积累成通用经验

这点是我越来越重视的。

如果你花大量精力做第一条 fused 路径，我希望最后沉淀下来的不只是一个 patch，而是一种可复用的方法。

`MLP MoE` 在这方面更有优势，因为很多 `VLA` 项目即使名字不叫 `MoE`，也已经有类似工程结构：

- token 类型分支
- expert-aware MLP
- 动作相关分支
- 需要对不同输入类型做不同投影

这意味着你在这里积累下来的很多后端经验，不只会服务一个模型。

## 四、和 attention 相比，MLP MoE 真正省掉的是什么

如果只从算子数学上看，很多人会把 attention 和 `MLP MoE` 都理解成“大矩阵乘法 + 一些周边逻辑”。

但从推理系统角度看，`MLP MoE` 真正适合先做，不是因为它“数学上更高级”，而是因为它更容易减少那些最烦的固定开销：

- Python expert loop
- token 分桶后的重复切片
- 中间张量创建
- 多次 kernel launch
- 激活与逐元素乘拆成多个小步骤
- expert 结果回写时的零散 scatter

如果第一版 fused path 做得好，通常至少能把下面这些东西合起来：

- routing 后的 expert 分块执行
- `gate_proj + up_proj`
- 激活与 combine
- `down_proj`
- 输出累加或回写

这类收益在 batch=1、固定或半固定 shape、低时延场景下会特别明显。

## 五、但这不代表 attention 不值得做

这里要防止走到另一个极端。

我说“先做 `MLP MoE`”，并不是说 attention 不重要，也不是说 attention 不该优化。

更准确地说，我的判断是：

> 对很多 `VLA` 项目来说，attention 很可能是后续必须优化的主战场之一，但第一条 fused 路径未必最适合先从它开始。

什么时候 attention 更适合作为第一刀？

通常是在下面这些情况下：

- benchmark 已经明确显示 attention 前后的 expert projection 是主瓶颈
- 当前模型里 `MLP MoE` 并不重，或者根本不开
- QKV 路径的重复 `permute/unpermute` 特别重
- 你已经有比较稳定的 attention 后端基线可对照

也就是说，选择仍然应该让数据说话。

## 六、如果真的先做 MLP MoE，第一版应该怎么做

如果把这件事落回实现，我会更建议按下面的顺序推进。

### 第一步：先做 `BF16/FP16` fused 基线

不要一上来就把低精度和全部复杂逻辑一起塞进去。

第一版目标应该是：

- expert-aware execution 路径打通
- 把 `gate/up/down` 的边界理顺
- 把 activation fuse 进来
- 把回写和累加逻辑做稳

这一步最重要的不是极限性能，而是：

> 先把 fused path 的接口、layout 和正确性基线建立起来。

### 第二步：把权重预打包机制定下来

如果后面还想做低精度，这一步最好尽早考虑。

你至少要先回答：

- expert 权重怎样组织
- 不同 expert 的布局是否统一
- `gate/up/down` 是分开存，还是部分打包
- 后续 scale 放在哪里

这些问题如果不提前想，后面很容易返工。

### 第三步：把 benchmark 和 profiler 跟上

这一条 fused path 一旦接进去，应该立刻能回答：

- 端到端 latency 降了多少
- 单条路径里哪个环节变化最大
- kernel launch 数减少了多少
- 中间 buffer 分配有没有明显下降

第一轮 benchmark 的目标不是“证明天下无敌”，而是：

> 证明这条路径确实值得继续往低精度和更深融合推进。

### 第四步：再把低精度接进来

这时候再考虑：

- 哪些线性层先降精度
- dequant 放在 kernel 里还是外面
- accumulator 保留什么精度
- epilogue 是否还能继续融合

这样你是在一条已经成立的 fused path 上加低精度，而不是在一团还没理顺的逻辑上堆复杂度。

## 七、这条路径对 VLA Runtime 意味着什么

如果第一条 fused 路径先落在 `MLP MoE`，它对整个 runtime 的意义，绝不只是“这一段快了一点”。

它实际上会验证很多更大的问题：

- runtime 能不能承接 expert-aware path
- benchmark 能不能真正驱动实现选择
- 后端接口是否足够承接低精度
- fused path 能不能作为后续局部 `megakernel` 的基础

也就是说，第一条 fused 路径选得好，后面很多事情都会变容易。

## 八、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> 在 `Wall-X` 这类带 expert 结构的 `VLA` 项目里，第一条 fused 路径很多时候更值得先落在 `MLP MoE`，因为它更规则、更容易验证、更容易接低精度，也更容易形成真正的 first win。

这不是一个绝对规则。

如果 benchmark 告诉你 attention 路径更重，那当然应该调整选择。  
但如果你现在还处在“要先打一条 fused path、又想让这条路真正成为 runtime 基线”的阶段，那么：

> `MLP MoE` 往往会是一个比 attention 更稳、更务实的起点。
