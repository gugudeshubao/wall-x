# 25. VLA Runtime 什么时候该引入 graph capture 与局部 megakernel

一路把 `VLA Runtime` 写到这里，很多更偏系统层的问题其实已经逐渐收束到一个很现实的判断上：

> 什么时候该继续引入更激进的执行形态？

具体一点，就是：

> 到底在什么时候，`graph capture` 和局部 `megakernel` 才真的值得被引入？

这个问题很容易被讲成一句非常简化的话：

- “端侧当然要上 graph”
- “batch=1 就该做 megakernel”
- “越激进越好”

这些话都抓住了一部分直觉，但如果拿来直接指导系统实现，往往还是太粗。

因为对 `VLA Runtime` 来说，这两样东西都不是“越早越好”的默认选项，而更像是：

> 当系统已经长到某个阶段时，才值得真正引入的下一层执行形态。

这篇文章就只讲这个问题。

## 一、先说结论：graph capture 和局部 megakernel 都不该作为第一批能力引入

我先把这个判断说死一点：

> 如果 runtime 还没把 benchmark、planner、fused path、precision policy 和 fallback 这些基础层立住，那么过早引入 `graph capture` 或局部 `megakernel`，大概率会让系统更难解释，而不是更成熟。

这不是说它们不重要，而是说它们更像：

- 在基础路径已经明确之后的下一层放大器

而不是：

- 一开始就拿来替代系统思考的捷径

所以这篇文章最重要的出发点就是：

> 不要先问“能不能做”，而要先问“什么时候做才值”。

## 二、为什么这两件事在 VLA 里很诱人

如果站在 `VLA` 场景里看，这两样技术之所以总会很快进入讨论，很正常。

因为它们确实和很多 `VLA` 现实场景天然贴近。

### 1. graph capture 的诱因

`VLA` 很多路径有这些特点：

- batch 很小
- shape 越来越固定
- 端侧单请求执行频繁
- launch overhead 容易显眼

所以一旦 benchmark 开始显示 launch 成本不轻，大家自然就会想到：

> 那是不是该用 graph capture 把固定路径收起来了？

### 2. 局部 megakernel 的诱因

同样，很多 `VLA` 路径又有这些特点：

- 中间阶段很多
- 每个阶段都不算特别大
- 但拼起来固定成本不小
- memory movement 和 layout 切换很烦

于是大家自然会继续往前想：

> 那是不是该把其中几段合成一个更大的执行单元？

也就是：

> 局部 `megakernel`

所以这两样东西之所以总会被提出来，不是因为大家喜欢激进，而是因为它们确实对 `VLA` 很有吸引力。

问题只在于：

> 它们并不是越早做越好。

## 三、为什么它们不适合作为第一批能力

这个问题必须讲清楚。

不然系统很容易在“想象中的最优路径”里迷路。

### 1. graph capture 强依赖路径稳定

它最怕的不是“图不好用”，而是：

> 你的执行链自己还没稳定。

如果：

- shape 还常变
- planner 规则还常改
- fallback 还不清楚
- fused path 还在频繁调整

那这时引入 graph，通常只会让你更难 debug。

### 2. 局部 megakernel 强依赖边界清楚

它最怕的不是“写 kernel 难”，而是：

> 你还没把哪几段该合、为什么该合、合了之后语义怎么保持清楚。

如果这一点不清楚，局部 `megakernel` 很容易从“高级优化”退化成“难以维护的黑盒”。

### 3. 两者都需要 benchmark 已经足够成熟

因为它们本质上都是“更后期的路径收紧手段”。

只有当前面已经有一条相对稳定、可解释、可 benchmark 的基线时，它们的价值才真正显现。

## 四、graph capture 最应该在什么条件下引入

如果单独说 `graph capture`，我更认可的前提条件通常有下面几条。

## 五、第一条：固定或半固定 shape 已经成为主要 workload

这是最核心的条件。

如果你的 workload 还大量是动态 shape，那 graph 当然也不是完全不能做，但它通常不会是第一优先。

相反，如果 benchmark 已经告诉你：

- 大部分请求 shape 很稳定
- edge 路径尤其稳定
- batch 基本恒为 1

那 graph 才真正开始变得有吸引力。

## 六、第二条：launch overhead 已经在 benchmark 里显著暴露

这一点也非常关键。

如果现在的系统瓶颈还主要在：

- compute 本身
- memory movement
- output path

那过早上 graph，收益通常有限。

只有当 benchmark 已经比较明确地显示：

> 路径里有一批重复、固定但偏碎的执行阶段，launch overhead 已经开始成为系统成本的一部分

这时候 graph 才真正值得。

## 七、第三条：planner 已经能稳定区分“graph-friendly workload”

也就是说，planner 至少要知道：

- 哪些 workload 更稳定
- 哪些 mode 下值得优先命中 graph
- graph miss 之后退到哪里

如果 planner 还完全没有这层语言，graph 引入之后通常只会被散落 if/else 消耗掉。

## 八、局部 megakernel 最应该在什么条件下引入

如果单独说局部 `megakernel`，我更认可的前提条件则比 graph 更苛刻一点。

## 九、第一条：路径边界已经稳定

这是第一位的。

如果你还没想清楚：

- `routing/layout`
- `compute`
- `scatter/output`

这些阶段的边界是否真的稳定，那局部 `megakernel` 基本不该太早做。

因为它本质上是在把若干阶段重新焊接成更大的执行单元。

边界没稳，焊得越早，返工越大。

## 十、第二条：被合并的几段之间已经有明确的数据复用价值

这点也不能靠直觉。

局部 `megakernel` 真正应该做的地方，不是“看起来挺近的两段”，而是：

> 已经被 benchmark 和 profiler 同时证明，彼此之间存在明显的数据复用、memory movement 冗余或 launch 冗余的几段。

比如：

- `gate/up` 到 `activation/combine`
- `compute` 到一部分固定形态的 output writeback

而不是纯粹因为“语义上顺手”就硬合。

## 十一、第三条：fallback 已经足够清楚

这一点尤其重要。

因为局部 `megakernel` 往往比普通 fused path 更难 debug，也更难覆盖所有边界条件。

如果你还不能清楚回答：

- megakernel 不命中时退到哪里
- 是退回局部 fused 还是整个 baseline
- benchmark 怎么记录“主路径”和“退回路径”

那过早引入它，系统可解释性会掉得很快。

## 十二、那 edge 和 cloud 下，引入时机会一样吗

通常不会一样。

### cloud 更可能先引入 graph

因为 cloud 环境里：

- baseline 和 fused path 往往先更容易稳定
- graph 更像是在已有通用路径上继续压 launch 成本

### edge 更可能更早认真评估局部 megakernel

因为 edge 场景里：

- batch 更小
- jitter 更敏感
- 固定 shape 更多
- 局部数据流重复更明显

但即使如此，我也仍然不建议在 edge 一开始就猛上 megakernel。

更合理的顺序通常还是：

- 先 benchmark
- 先稳定 path
- 再 graph
- 再局部 `megakernel`

只不过 edge 这条链往往会比 cloud 更早走到最后两步。

## 十三、一个更现实的引入顺序

如果让我给 `VLA Runtime` 写一个更现实的时间顺序，我会更推荐下面这样。

### Stage 1

先有：

- baseline path
- `BF16 fused`
- benchmark
- planner
- fallback

### Stage 2

再有：

- low-precision fused
- packed weight
- capability matrix

### Stage 3

这时才开始认真评估：

- 哪些 workload 适合 graph
- 哪些局部 stage 值得继续合成更大的执行单元

### Stage 4

如果 benchmark 和 planner 都已经足够成熟，再继续引入：

- graph capture path
- 局部 `megakernel path`

也就是说，这两样东西更像：

> runtime 中后期的“路径收紧器”

而不是早期的“路径定义器”。

## 十四、benchmark 应该怎样证明“现在该上 graph / megakernel 了”

这一点我也想说得更具体一点。

如果 benchmark 要真正指导这件事，它至少应该开始回答下面这些问题：

### 对 graph

- 当前主要 workload 的 shape 稳定度是多少
- graph-friendly 请求比例是多少
- launch overhead 在整体里占比是否已经值得处理

### 对局部 megakernel

- 当前最值得合并的几段是哪几段
- 它们之间的数据 movement 是否真的冗余
- 这些阶段的边界是否已经稳定
- fallback 成本是否可接受

只有 benchmark 能回答这些问题时，graph 和局部 `megakernel` 的引入才算是“被系统推着走”，而不是“被直觉推着走”。

## 十五、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 里的 `graph capture` 和局部 `megakernel`，都不该作为第一批能力引入，而应该在 benchmark、planner、fused path、precision policy 和 fallback 已经把系统底座立住之后，再作为更后期的路径收紧手段被有条件地引入。

因为对这种系统来说，真正重要的不是：

> 你能不能尽快上最激进的技术。

而是：

> 你是不是知道它为什么该在现在引入、该引入到哪一段、以及引入之后如果不合适要怎样退回去。
