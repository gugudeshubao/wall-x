# 29. VLA Runtime 的执行报告应该怎样设计

前面几篇文章里，我已经反复提到过 `ExecutionReport` 这个对象。

它总是和这些东西一起出现：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `benchmark`
- `fallback`
- `capability matrix`

但到现在为止，我都还只是把它当成一个系统里“应该存在的对象”在讲。  
如果继续往下做，迟早要把这个问题彻底正面回答：

> `VLA Runtime` 的执行报告，到底应该怎样设计？

这个问题比看起来重要得多。

因为如果没有一份足够稳定、足够结构化的执行报告，后面很多关键能力都会失去抓手：

- benchmark 只能看到总时间
- planner 很难被真正修正
- fallback 变得不可解释
- capability matrix 很难和真实执行情况对齐

所以这篇文章就只讲这个问题。

## 一、先说结论：ExecutionReport 不是日志汇总，而是 runtime 对“这次执行到底发生了什么”的正式描述

很多系统一谈报告，第一反应都是：

> 记点日志、打点时间、最后输出几行信息。

这当然有用，但对 `VLA Runtime` 来说不够。

如果把我对这个对象的判断压缩成一句话，我会说：

> `ExecutionReport` 不该只是日志汇总，而应该是 runtime 对“这次执行到底发生了什么”给出的正式、结构化、可被其他层消费的描述。

这里有几个关键词：

- 正式
- 结构化
- 可被其他层消费

为什么要说得这么重？

因为如果执行报告只是给人看，而不是给系统也看，那么它的价值会迅速缩水。

真正成熟的 `ExecutionReport`，应该至少同时服务：

- benchmark
- planner 修正
- fallback 解释
- path/capability 命中分析

这时它才是 runtime 的核心对象，而不是附属输出。

## 二、为什么 VLA Runtime 特别需要 ExecutionReport

如果只是简单推理脚本，很多时候一个总 latency 加几行 print 就够了。  
但 `VLA Runtime` 天生更复杂，至少有这些层会同时对“执行发生了什么”感兴趣：

- planner
- kernel backend
- benchmark
- output adapter
- fallback 机制

同时它又有很多比普通系统更复杂的维度：

- edge / cloud
- precision policy
- fused path / baseline
- graph / local megakernel
- expert-heavy workload

如果没有一份统一的执行报告，最后很容易出现这样一种状况：

> 每个层都记录了一点信息，但整个系统没有一个对象能把“这次到底怎么跑的”说清楚。

所以 `ExecutionReport` 的意义，本质上是：

> 把分散在各层的执行事实，收敛成 runtime 的统一现实。

## 三、执行报告最少应该回答哪些问题

如果现在就要给 `VLA Runtime` 定义第一版执行报告，我觉得它至少要能回答下面五类问题。

## 四、第一类：这次请求是什么

也就是：

- 当前 workload 属于什么类型
- 当前输入 shape 是什么级别
- 当前请求属于 edge 还是 cloud mode

这部分其实和 `WorkloadDescriptor` 强相关，但在执行报告里仍然需要保留一份“执行时刻看到的请求画像”。

为什么？

因为后面很多分析都需要知道：

> 这次结果是在哪种 workload 下发生的。

## 五、第二类：planner 最初决定了什么

执行报告不该只写“最后发生了什么”，还应该记录：

> planner 原本打算让系统怎么跑。

例如：

- 目标 deployment mode
- 目标 precision policy
- 首选 path
- fallback chain

这部分的价值在于，它让你后面能清楚比较：

- 计划是什么
- 实际发生了什么

没有这层，你就很难区分：

- 是 planner 决策本身有问题
- 还是 backend 执行时发生了偏离

## 六、第三类：实际命中了哪条 path

这是我觉得最关键的一层之一。

执行报告必须明确告诉你：

- 实际命中的 path 是哪条
- 实际命中的 precision family 是什么
- 是否命中了 graph path
- 是否命中了 local megakernel path

这部分为什么重要？

因为 runtime 后面所有的 benchmark、fallback 和 capability 分析，都不能只看“计划想走哪里”，而必须知道“最后真走了哪里”。

这也正是 `ExecutionReport` 区别于普通调试日志的地方。

## 七、第四类：是否发生 fallback，以及怎样发生的

前面已经专门写过一篇文章讲 fallback，这里它必须进入执行报告的正式结构。

也就是说，执行报告应该能明确表达：

- 是否发生 fallback
- 是 path fallback、precision fallback，还是 mode fallback
- 从哪条 path 退到了哪条 path
- 是在什么条件下退的

这一步之所以重要，是因为如果你后面看到 benchmark 某组数字异常，第一件要问的往往就是：

> 这次是不是根本没跑到我以为的那条 path？

而这个问题只有执行报告能给出稳定答案。

## 八、第五类：各 stage 的实际开销

如果执行报告里没有 stage breakdown，它对 benchmark 的价值会大幅下降。

我更推荐的第一版至少记录：

- input adapter
- routing/layout
- core compute
- scatter/output
- output adapter

如果系统已经更复杂，还可以逐渐再细分。

这部分的核心价值在于：

> 让 benchmark 不只是“这次一共花了多久”，而是“这次每段到底花了多少”。

只有这样，执行报告才真正能反哺 planner 和后续优化。

## 九、一个更现实的 ExecutionReport 结构草图

如果把它写得更像 runtime 对象，而不是概念描述，我更推荐它至少分成下面几层。

## 十、第一层：Request Context

也就是：

- workload id / type
- shape class
- deployment mode

这层回答的是：

> 这次执行面对的是一个什么样的请求。

## 十一、第二层：Planned Execution

也就是：

- selected family/scope
- target precision policy
- expected path
- configured fallback chain

这层回答的是：

> planner 原本希望系统怎么跑。

## 十二、第三层：Actual Execution

也就是：

- actual path hit
- actual precision used
- actual pack/layout hit
- graph / megakernel 是否命中

这层回答的是：

> 系统最后到底怎么跑了。

## 十三、第四层：Runtime Events

也就是：

- fallback events
- path switch events
- capability miss

这层回答的是：

> 执行中发生了哪些重要偏移。

## 十四、第五层：Performance Summary

也就是：

- total latency
- stage breakdown
- memory / jitter / optional counters

这层回答的是：

> 这次执行的结果怎样。

如果把这五层立住，执行报告就已经不只是“打点结果”，而是一份系统级对象了。

## 十五、为什么 ExecutionReport 不该做成“越详细越好”的大杂烩

这里我也想说一个很常见的误区。

一旦开始设计报告，很多人会自然觉得：

> 那就把所有东西都记录进去。

这听起来安全，但实际很容易把执行报告做成一个：

- 太重
- 太难解释
- 太难稳定

的大对象。

我更推荐的原则是：

> 第一版执行报告优先记录那些会同时被 planner、benchmark、fallback 和 capability 分析消费的信息。

也就是说，不是所有细节都要先放进去，而是先放那些真正跨层有价值的信息。

## 十六、ExecutionReport 和 benchmark 的关系到底是什么

我更倾向于把二者关系定义成：

- `ExecutionReport` 是单次执行的结构化事实
- benchmark 是对一组 `ExecutionReport` 的聚合和解释

这个区分非常重要。

如果没有它，系统很容易把：

- 原始执行事实
- 后处理统计
- 路径解释

全部混成一层。

更好的做法是：

> 先把单次执行报告稳定下来，再让 benchmark 围绕它做聚合。

这样后面很多事情都会更清楚：

- 哪些问题是单次行为异常
- 哪些问题是 workload 级趋势
- 哪些问题是 planner 规则该被修正

## 十七、为什么 ExecutionReport 也必须和 taxonomy、capability matrix 对齐

这一点在后面会变得越来越重要。

如果 `ExecutionReport` 里的 path 和 capability 语言，和系统其他层都不是同一套，那它最后的作用会非常有限。

更理想的状态应该是：

- planner 用 taxonomy 选 path
- capability matrix 说 path 能做什么
- backend 真正执行 path
- `ExecutionReport` 用同一套 taxonomy 和记下结果

这样你后面才能说出真正有系统意义的话，比如：

> 在这类 workload 下，planner 计划命中 `fused/local_mlp_moe`，但 capability miss 导致实际退回 `baseline/global`，于是 stage breakdown 里 scatter 的占比重新升高。

这时候，报告才真正变成 runtime 的语言，而不是人类肉眼看一眼的日志。

## 十八、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的 `ExecutionReport` 不该只是日志汇总，而应该被设计成一份结构化的“执行事实对象”，至少同时描述请求上下文、planner 原计划、实际命中的 path、fallback 事件以及各 stage 的性能摘要，让 benchmark、planner 修正、capability 分析和 fallback 解释都能基于同一份现实展开。

因为对这种系统来说，真正值钱的从来不是：

> 打了多少日志。

而是：

> 系统能不能把“这次执行到底发生了什么”这件事，用一种所有层都能共同理解的方式说清楚。
