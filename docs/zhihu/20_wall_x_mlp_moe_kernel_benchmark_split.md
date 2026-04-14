# 20. Wall-X 的 MLP MoE kernel benchmark 应该怎么拆

如果一路把 `Wall-X` 的 `MLP MoE` 路径讲到这里，一个非常现实的问题就会越来越突出：

> 你怎么证明自己写的 kernel 真的有价值？

这句话看起来简单，但真正做起来并不轻松。

因为很多时候，大家以为自己在 benchmark kernel，实际上测到的却是一整串混杂的成本：

- Python 调用成本
- 输入重排成本
- 权重准备成本
- kernel 本体成本
- scatter / output 成本
- 甚至还混着上游和下游的系统开销

结果就是：

> 看起来有一组数字，但你根本说不清楚到底是哪里快了，哪里没快，哪里只是测法不干净。

这篇文章就只讲这个问题：

> `Wall-X` 的 `MLP MoE kernel benchmark` 到底应该怎么拆，才能真正服务 kernel 设计和 runtime 迭代？

## 一、先说结论：benchmark 不能只问“快了多少”，还必须问“哪一段快了多少”

很多 kernel benchmark 最容易犯的一个问题，就是最后只给出一个总时间：

> latency = xx ms

这个数字当然不是完全没用，但如果只停在这里，对 `MLP MoE` 这种路径来说远远不够。

因为这条路径天然包含多个阶段：

- routing / layout prepare
- expert compute
- scatter / weighted accumulation
- output 回接

如果 benchmark 不把这些阶段拆开，你最后就会遇到几个很典型的问题：

### 1. 你不知道优化真正打在了哪里

也许 kernel 本体快了，但总时间没变，因为别的阶段变成了瓶颈。

### 2. 你不知道下一步该优化哪一段

也许现在真正最重的已经不是 compute，而是 scatter 或 weight pack。

### 3. 你不知道这个 kernel 对 runtime 的真实价值

因为 runtime 关心的不是某个孤立 kernel 漂不漂亮，而是：

> 它放进完整执行链之后，到底能带来什么。

所以我的判断是：

> `MLP MoE kernel benchmark` 第一原则不是“先出一个大数字”，而是“先把路径拆干净”。

## 二、为什么 MLP MoE 的 benchmark 比普通线性层更难拆

如果你只 benchmark 一个普通 matmul，问题相对简单很多。

但 `MLP MoE` 的复杂度在于，它不是单一算子，而是一条小执行链。

它至少会同时涉及：

- token 按 expert 的分组
- expert 内部的 `gate/up/down`
- 激活与 combine
- scatter / 加权回写

这意味着如果你只是测“整个 `MLP MoE` 函数”，你最后得到的几乎必然是一个混合值。

而这种混合值最危险的地方在于：

> 它经常看起来很完整，实际上却最难指导下一步优化。

所以对 `MLP MoE` 来说，benchmark 一开始就应该按路径来拆，而不是按“有没有一个 API 调用”来拆。

## 三、我更推荐的最小拆分方法

如果让我给 `Wall-X` 的 `MLP MoE kernel benchmark` 一个最小可行拆法，我会建议至少拆成四层。

## 四、第一层：Layout / Routing Cost

这一层回答的是：

- token 分组到底花了多少
- `expert_offsets / row_id_map` 的生成到底花了多少
- 输入重排到底花了多少

这一层必须单独看，因为很多人做 fused compute 时，会不自觉地把这些成本一起混进去。

结果一旦看到“总时间没有下降很多”，就误以为 kernel 价值不大。

但真实情况可能只是：

> 你把 compute 打快了，layout 准备反而成了新的主瓶颈。

### 这层最适合回答的问题

- `expert-aware` 路径到底贵不贵
- 是否值得缓存某些 metadata
- 是否值得后面进一步融合 routing 与 layout

## 五、第二层：Core Compute Cost

这一层才是 `MLP MoE fused kernel` 最核心的 benchmark。

它关注的是：

- `gate_proj`
- `up_proj`
- activation + combine
- `down_proj`

这层的核心目标不是测“整条路径多快”，而是回答：

> 你真正写的 compute kernel，到底比基线强多少？

这一层如果拆清楚了，后面很多设计决策都会容易得多，比如：

- 是否值得继续做 low-precision
- `gate/up` 是否值得进一步合并
- epilogue 是否已经是热点

## 六、第三层：Scatter / Output Cost

这一层经常被低估。

很多人一想到 kernel，就注意力都在 compute 上。  
但对 `MLP MoE` 路径来说，scatter 和加权回写常常也是很实在的成本。

如果这一层不单独测，你很容易出现一种错觉：

> compute kernel 已经很强了，为什么端到端收益还是一般？

这时候真正的答案，往往就在 scatter 这一层。

### 这层最适合回答的问题

- 是否值得进一步把 scatter 融进 compute
- 当前 output layout 是否合理
- 是否需要为后续 runtime 保留 expert-ordered 中间结果

## 七、第四层：Path-Level End-to-End Cost

前面三层回答的是局部。

但你最终还需要一层全局视角：

> 这条 `MLP MoE` path 放进系统里以后，整体到底值不值？

所以除了局部 benchmark，我很建议再单独保留一层 path-level end-to-end 测试：

- 从 expert-aware path 入口开始
- 到输出重新接回主干为止

这层的价值在于：

- 看整体收益
- 看局部优化是否真的外溢到系统层
- 看是否出现“局部很快、整体一般”的情况

## 八、benchmark 不应该只按阶段拆，还应该按 workload 拆

这点我觉得特别重要。

同一个 kernel，在不同 workload 下的表现很可能完全不一样。

如果只看单一 workload，很容易得到误导性结论。

对 `MLP MoE` 路径来说，我至少会建议按下面这些维度拆 workload：

### 1. token 数量

也就是：

- 当前到底有多少有效 token 进入 expert path

因为这直接影响：

- 小 batch / 小 token 下的固定开销占比
- 大 token 下 compute 和带宽的相对关系

### 2. expert 分布

也就是：

- token 是不是均匀分到多个 expert
- 还是极度偏向少数 expert

这会直接影响：

- segment 长度
- layout 效率
- grouped execution 的实际收益

### 3. shape 稳定性

例如：

- 固定 shape
- 半固定 shape
- 动态 shape

这会决定：

- pack 能不能复用
- graph 是否有意义
- 某些 fused path 是否稳定命中

### 4. precision mode

至少应该区分：

- 高精度 baseline
- 高精度 fused
- 低精度 fused

否则你后面很难真正判断 low-precision 的价值。

## 九、一个更现实的 benchmark 输出应该长什么样

我理想里的 `MLP MoE kernel benchmark` 不该只是一张“版本对比表”，而应该至少有四类输出。

### 1. Workload 描述

例如：

- token 数
- expert 数
- expert 分布
- hidden / intermediate 维度
- precision mode
- shape 类型

### 2. Stage Breakdown

例如：

- layout prepare
- compute
- scatter
- total path

### 3. Backend 信息

例如：

- baseline / fused / low-precision fused
- 当前 pack format
- 当前是否命中预打包路径

### 4. 稳定性信息

例如：

- 平均值
- P50 / P90 / P99
- fallback 次数

只有这些都在，你的 benchmark 才真的能指导 runtime。

## 十、为什么 benchmark 必须让 planner 看得懂

如果 benchmark 做得很好，但 planner 根本消费不了，那它对 runtime 的价值还是有限。

我更希望 benchmark 最终能回答 planner 关心的问题，比如：

- 在这种 token 数和 expert 分布下，fused path 值不值
- 在这种 precision mode 下，收益有多稳定
- 当前 scatter 是否已经成为新瓶颈
- 当前是否应该继续保留 expert-ordered 中间结果

也就是说：

> benchmark 的输出应该能自然转成 planner 的规则输入。

如果做不到这一点，benchmark 就很容易退化成“独立的技术报告”，而不是 runtime 的反馈系统。

## 十一、第一版最应该避免的 benchmark 误区

为了把边界说清楚，我也把几个最常见的误区写出来。

### 1. 只测总时间，不拆阶段

这会让你很难真正知道问题出在哪。

### 2. 只测单一 workload

这会让你对 kernel 的真实适用范围判断失真。

### 3. 不记录 pack format 和 precision mode

最后你根本不知道某组数字对应哪条实现路径。

### 4. 不记录 fallback

你以为自己在测 fused path，实际上也许已经退回 baseline 了。

## 十二、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE kernel benchmark` 不该只测一个总 latency，而应该至少按 `layout/routing`、`core compute`、`scatter/output` 和 `path-level total` 四层拆开，并同时对 workload、precision 和 pack format 做显式描述。

因为真正值钱的 benchmark，不是告诉你“某个 kernel 看起来快”，而是：

> 让你知道它为什么快、在哪些场景快、快完之后系统下一步该优化哪里。
