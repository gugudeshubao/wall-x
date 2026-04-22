# 34. Wall-X 的 MLP MoE path 在 cloud workload 下该先优化哪一段

前一篇文章里，我已经把一个判断讲清楚了：

> 在 `edge workload` 下，`Wall-X` 的 `MLP MoE path` 往往会更早地从“最重 compute”转向“最拖系统的执行链”，也就是更重视 path 命中率、fallback、scatter/output 的 tail cost 和 layout/routing 的固定成本。

但如果只写到这里，整件事还不完整。

因为真正做 runtime 时，你几乎一定会同时面对两类完全不同的 workload 画像：

- `edge`
- `cloud`

而且这两种画像，并不会只是“同一套优先级的不同设备版本”。  
它们往往会把优化顺序真正拉开。

所以这篇文章想做的事情很直接：

> 把 `cloud workload` 这一侧的优先级也说清楚。

更准确地说，就是回答：

> 在 `cloud workload` 下，`Wall-X` 的 `MLP MoE path` 到底更该先优化哪一段？

## 一、先说结论：在 cloud workload 下，优先级通常会更晚离开 compute 主轴

如果把我的判断压缩成一句话，我会说：

> 对 `Wall-X` 的 `MLP MoE path` 来说，在 `cloud workload` 下，优化优先级通常会更长时间地围绕 compute 主轴展开，也就是更优先关注 `gate/up/down` 的主计算链、low-precision 覆盖率、pack 命中率和 path 覆盖面，而不是像 edge 那样更快转向 tail cost 和 jitter。

这并不意味着：

- `scatter/output` 不重要
- `layout/routing` 不重要
- fallback 不重要

而是说，在 cloud 场景里，这些问题通常抬头得更晚，或者说：

> 它们更经常是在 compute 已经明显收敛之后，才会进入下一轮主要矛盾。

这就是 cloud 和 edge 的核心差别之一。

## 二、为什么 cloud 会把优先级重新拉回 compute 主轴

这件事如果只从设备角度看，很容易被误解成：

> 因为云端 GPU 更强，所以当然先优化大算子。

这个说法只说对了一半。

更准确的原因其实来自 workload 本身：

- token 数往往更大
- expert 覆盖更广
- path 兼容性要求更高
- 更多时候需要兼顾吞吐和覆盖面

也就是说，在 cloud 场景里，你更容易遇到这样的现实：

> 单次请求虽然可能也不算特别大，但整体 workload 分布更宽、命中路径更多样，compute 和 pack 的收益更容易在总体层面放大。

这会导致优先级自然往这些方向偏：

- 主计算链的总成本
- low-precision 带来的总体吞吐收益
- packed weight 能否稳定命中
- fused path 在更广 workload 上的适用范围

所以 cloud 下的优先级，不是简单地“云上卡大就先看算力”，而是：

> 在更宽 workload 分布下，compute 主轴更容易继续保持为首要矛盾。

## 三、先把 MLP MoE path 的几个候选段重新摆一遍

为了把讨论说实，我还是沿用之前那条路径分段：

- `layout/routing`
- `gate/up + activation/combine`
- `down`
- `scatter/output`
- path hit / fallback

问题是：

> 放到 cloud 场景里，这几段的优先级会如何重排？

我的判断会更接近下面这种顺序：

### cloud 更早优先看的

- `gate/up/down` compute
- low-precision 覆盖率
- pack 命中率
- path 适用范围

### cloud 通常更晚抬头的

- `scatter/output` tail cost
- jitter 导向的路径治理
- 某些非常 edge-specific 的固定成本

这并不是绝对规律，但它是一个非常实用的默认起点。

## 四、为什么 `gate/up/down` 在 cloud 下通常还是第一优先

这一点其实很自然。

在 cloud workload 下，`MLP MoE path` 最容易保持主导地位的，往往还是：

- `gate_proj`
- `up_proj`
- `down_proj`

原因主要有三个。

### 1. 它们的总 compute 规模更容易被放大

在更宽的 workload 覆盖下，这些大块线性层的总体成本更容易继续成为路径主轴。

### 2. 它们更容易直接受益于 low-precision

这意味着一旦你能让：

- `gate/up`
- `down`

这些部分稳定命中某种低精度 fused path，总体收益通常会比较直观。

### 3. 它们的优化更容易被吞吐和均值指标放大

这和 edge 很不同。

在 edge 下，你常常更在意：

- 最坏时怎样
- 抖不抖
- 单次尾延迟会不会失控

而在 cloud 下，你更容易看到：

> 主计算链的小幅持续改进，会在总体 workload 里被放大成更稳定的收益。

所以如果你还在 cloud workload 的前几轮优化里，我通常仍然会优先盯 compute。

## 五、为什么 low-precision 在 cloud 下会更早变成“路径问题”而不是“实验问题”

这一点和 edge 也不太一样。

在 edge 下，low-precision 往往更容易先被问成：

> 会不会影响稳定性？会不会把 jitter 放大？

而在 cloud 下，更常见的问题会变成：

> 这条 low-precision path 的覆盖面够不够？命中率高不高？pack 和 capability 能不能稳定支撑它在更多 workload 上跑起来？

也就是说，cloud 下 low-precision 更早地从“局部算子实验”变成：

> 一条 runtime 正式 path 的覆盖能力问题。

这意味着优先级会自然落到：

- 哪些 `MLP MoE` workload 已经能稳定走 low-precision
- 哪些还需要保守回退
- pack format 和 capability matrix 是否足够支持这条 path 扩大使用范围

所以 cloud 下，low-precision 的讨论会更早和：

- path taxonomy
- capability matrix
- plan/report 对齐

这些 runtime 级对象绑在一起。

## 六、为什么 pack 命中率在 cloud 下通常比 scatter tail 更早值得看

这点非常关键。

在 edge 下，你很容易更早关心：

- `scatter/output` 是否在拖路径尾巴
- fallback 是否放大了 jitter

但在 cloud 下，如果你的目标是让某条 `MLP MoE` path 真的开始承担更多 workload，那么一个更早冒出来的问题往往是：

> packed weight 和相关 low-precision 资源，到底能不能稳定命中？

因为只要 pack 命中率不高，后面的很多漂亮路径都会迅速退化成：

- 理论上存在
- 实际上经常退回

这对 cloud 场景尤其不划算，因为它会让：

- 你以为自己在扩路径覆盖
- 结果系统却一直在吃回退成本

所以 cloud 下我会更早把下面这些纳入重点观察：

- pack 准备是否稳定
- capability 是否准确表达命中前提
- planner 是否能合理地区分“这次该不该尝试某种 low-precision fused path”

## 七、那 scatter/output 在 cloud 下是不是就不重要

不是。

只是通常它不会像在 edge 那样更早成为第一优先。

更准确的说法应该是：

> 在 cloud 下，`scatter/output` 往往更像是“compute 主轴收敛之后的下一轮问题”，而不是最早就该抢在前面的主要问题。

尤其在下面这些情况下，它仍然会抬头：

- compute 已经明显压下去
- pack / low-precision path 已经比较稳定
- benchmark 开始显示 tail cost 占比抬头
- output writeback 成为新的 memory movement 主因

也就是说，cloud 并不是永远不该看 `scatter/output`，而只是：

> 它更常发生在后一轮，而不是前一轮。

## 八、为什么 path 覆盖面在 cloud 下比在 edge 下更早重要

这点我觉得特别值得单独说。

在 edge 下，一条 path 有时只要服务好一类稳定 workload，就已经很有价值。  
但在 cloud 下，如果你想让 runtime 真正在更多请求里稳定收益，路径覆盖面通常会更早变成问题。

这意味着你会更早问这些问题：

- 这条 `MLP MoE` path 适合哪些 workload
- 哪些 shape class 下还能稳定命中
- 哪些 expert 分布会让它收益下降
- 哪些条件下 planner 应该主动回退

也就是说，cloud 下“先优化哪一段”，不只是算子问题，还更早会变成：

> 这条 path 到底能覆盖多宽 workload 的问题。

## 九、一个更现实的 cloud 优先级顺序

如果我现在要给 `Wall-X` 的 `MLP MoE path` 在 cloud workload 下排一个更现实的优先级，我会更倾向于下面这个顺序。

## 十、第一优先：先压主 compute 链

也就是：

- `gate/up/down`
- `activation + combine`

这一条通常仍然是 cloud 下最值钱的第一主轴。

## 十一、第二优先：让 low-precision path 的覆盖面和 pack 命中率站稳

这一步会比在 edge 下更早抬头。

因为 cloud 更容易受益于：

- 覆盖更广
- 命中更稳
- 退回更少

这些东西。

## 十二、第三优先：再看 path 覆盖与 capability 边界

这一步更多是 runtime 级问题：

- 哪些 workload 真该被这条 path 接住
- 哪些 workload 不该勉强命中
- capability matrix 是否表达清楚

## 十三、第四优先：最后再让 scatter/output 抬头

当上面几层已经逐渐收敛后，`scatter/output` 才更像 cloud 下的下一轮主要矛盾。

这时再去看：

- output writeback
- weighted accumulation
- tail cost

通常会更值。

## 十四、为什么这套顺序不能机械套回 edge

这点我想再强调一次。

如果你把 cloud 这套优先级机械搬到 edge，很容易出现一个问题：

> 继续盯 compute、pack、覆盖面，但系统真正难受的是 tail、命中率和 jitter。

这就是为什么我一直在反复强调：

> 同一条 path，不同 deployment mode 下，本来就应该有不同优先级。

这不是“实现分叉”，而是 runtime 真正开始理解自身场景的表现。

## 十五、benchmark 在 cloud 下应该怎么改口径

如果你真的要按这套优先级来推进，那 cloud 场景下的 benchmark 也应该更强调：

- compute chain 占比
- low-precision path 覆盖率
- packed weight 命中率
- path 命中率和 fallback 分布
- 在更宽 workload 下的平均收益

相比之下，虽然 `P99` 和 jitter 仍然重要，但它们通常不会像在 edge 下一样那么早成为第一层指标。

## 十六、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> 在 cloud workload 下，`Wall-X` 的 `MLP MoE path` 通常会更长时间地围绕 compute 主轴展开优化，也就是更优先关注 `gate/up/down` 的主计算链、low-precision 覆盖率、pack 命中率和 path 覆盖面；而 `scatter/output` 的 tail cost 和更强的路径治理，通常会在这些主轴相对收敛之后，才更适合作为下一轮重点。

因为在 cloud 场景里，真正更早决定路径价值的往往不是：

> 某个尾部阶段是不是已经看起来不太顺眼。

而是：

> 这条 path 能不能先在更宽 workload 上稳定、持续地把主计算链的收益兑现出来。
