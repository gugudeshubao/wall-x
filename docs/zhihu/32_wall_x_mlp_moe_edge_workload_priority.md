# 32. Wall-X 的 MLP MoE path 在 edge workload 下该先优化哪一段

一路把 `Wall-X` 的 `MLP MoE path` 写到这里，很多问题其实已经越来越不再是：

> 这个算子本身能不能优化？

而是：

> 在不同 workload 下，到底哪一段更值得先优化？

这个问题放到 `edge workload` 时，会变得尤其现实。

因为很多在云侧看起来还算“平滑”的开销，到了 edge 场景里，很可能会一下子变成主成本：

- batch 更小
- shape 更固定
- launch overhead 更显眼
- jitter 更敏感
- fallback 的代价也更大

所以如果把问题说得更直白一点：

> 同一条 `MLP MoE path`，在 edge workload 下，不一定该按云侧那套优先级继续优化。

这篇文章就只讲这个问题。

## 一、先说结论：在 edge workload 下，优先级通常会从“算力优先”转向“执行链优先”

如果把我的判断压缩成一句话，我会说：

> `Wall-X` 的 `MLP MoE path` 在 edge workload 下，优化优先级通常会更快地从“哪个 compute kernel 最重”转向“哪一段执行链最拖系统”，也就是说，会更强调 layout、tail cost、命中率和 jitter，而不仅仅是裸算力。

这不是说 compute 不重要了。  
而是说在 edge 场景里，很多原本在云侧不那么显眼的问题，会提前抬头：

- `layout/routing` 的固定成本
- `scatter/output` 的 memory movement
- 路径命中和 fallback 对尾延迟的影响
- launch overhead 对 batch=1 的影响

所以如果还继续机械地沿用：

> 先盯最重 matmul、再盯别的

这种顺序，就很容易错过 edge 场景真正更值得优先处理的问题。

## 二、为什么 edge workload 会改写优先级

这一点其实很好理解，但必须明说。

在很多云侧场景里，系统更容易被这些东西主导：

- 大一点的 token 规模
- 更多并发
- 更高吞吐目标

这时很多优化自然会优先指向：

- 最重 compute
- 低精度吞吐
- 更通用的 fused path

但 edge workload 往往不同。

它更常见的画像是：

- batch 很小，甚至长期为 `1`
- 请求频率高，但每次 payload 不大
- 更看单次 latency 和 jitter
- 输入 shape 更稳定
- output 更直接地服务控制链路

这会带来一个重要变化：

> 很多在云侧只是“还行”的固定成本，在 edge 下会变得特别不值。

也就是说，在 edge 场景里，优化优先级很容易往这些方向偏：

- 更少的阶段切换
- 更少的中间回写
- 更稳定的 path 命中
- 更清楚的 fallback

所以如果只从“哪个算子最重”出发，很可能会低估 edge 的真实瓶颈。

## 三、先看 MLP MoE path 的几个典型候选段

如果把这条 path 粗略拆开，前面已经讨论过的关键段大致是：

- `layout/routing`
- `gate/up + activation/combine`
- `down`
- `scatter/output`
- `path-level fallback / path switch`

问题是：

> 在 edge workload 下，它们的优先级会怎么变化？

我自己的判断是：

### 云侧更容易优先看的

- `gate/up/down` compute
- low-precision throughput

### edge 更容易提前抬头的

- `layout/routing`
- `scatter/output`
- path hit / fallback

这就是这篇文章真正想展开的部分。

## 四、为什么 layout/routing 在 edge 下更容易变重要

这部分很多时候会被低估。

在云侧，如果 token 数更大、compute 更重，`layout/routing` 可能只是路径里的一个前导成本。  
但在 edge 下，由于 batch 更小、每次请求更轻，它的相对占比很容易上升。

而且它还有两个更 edge-sensitive 的特点：

### 1. 它是固定成本

也就是说，不管后面 compute 多不多，这段都要发生。

一旦 batch 变小、path 变短，它的相对占比就会明显抬高。

### 2. 它会影响后续路径的稳定性

例如：

- expert 分布是否更碎
- output 是否更难写回
- 后续 path 命中是否更稳定

这些问题在 edge 下会更直接地体现在整体 jitter 上。

所以如果 edge workload 下 benchmark 已经显示：

- compute 收益开始边际递减
- `layout/routing` 占比抬头

那这时候继续只盯 compute，通常就不是最优策略了。

## 五、为什么 scatter/output 在 edge 下会更早成为显性问题

这一点和前面讨论过的 `scatter/output` 融合时机强相关，但在 edge 下要更激进地重视它。

原因很简单：

### 1. memory movement 的相对代价更高

尤其当 compute 本身已经被压下去之后，后半段的输出回写会更显眼。

### 2. output 更接近控制链路

也就是说，在 edge 场景里，`scatter/output` 不只是“尾部开销”，它更接近 camera-to-action 这条链路的真正终点。

### 3. jitter 更容易被放大

一旦这段不稳定，往往很快就会体现在：

- P99 抖动
- 控制链路不稳

所以如果在 edge 场景里，benchmark 已经表明：

- `scatter/output` 占比不低
- 或者 fallback 之后总是这段变重

那么它通常应该比在云侧更早进入重点观察对象。

## 六、为什么 path 命中率在 edge 下比在 cloud 下更值得优先看

这点我认为特别重要。

在很多云侧系统里，即使某条 path 偶尔 miss，整体吞吐也许还能通过别的方式摊平。  
但在 edge 下，path miss 的代价往往会更直接：

- 单次 latency 立刻拉高
- jitter 明显上升
- fallback 成本更难藏住

所以 edge workload 下，我会更建议把下面这些东西提早纳入核心指标：

- fused path 命中率
- low-precision path 命中率
- fallback 频率
- fallback 之后的尾延迟变化

也就是说：

> 在 edge 场景里，路径治理的重要性会更早超过单个 kernel 的漂亮数字。

## 七、一个更现实的 edge 优先级顺序

如果把它说得更实一点，我现在更推荐这样的顺序，而不是机械沿用云侧顺序。

## 八、第一优先：先看 path hit / fallback 稳定性

理由很简单。

因为如果路径本身在 edge 下就不稳定：

- 再快的局部 kernel
- 再漂亮的 low-precision 数字

都很难真正转化成稳定收益。

所以 edge workload 下，我会更早把下面这些放到第一层：

- 当前 path 是否稳定命中
- fallback 是否频繁
- fallback 后代价是否过大

## 九、第二优先：再看 scatter/output 和 tail cost

因为一旦路径已经基本稳定命中，下一步最容易暴露出来的就是尾部问题。

尤其是在：

- batch=1
- 低 jitter 目标
- 输出直接服务控制链路

这些条件下，tail cost 很可能比继续追某个局部 compute kernel 的极限更值钱。

## 十、第三优先：再看 layout/routing 的固定成本

这一步通常会和前两步一起抬头。

如果 edge workload 显示：

- 路径已经基本稳定
- 但整体 fixed cost 仍然不低

那 `layout/routing` 就会比在云侧更早进入下一轮重点优化区。

## 十一、第四优先：最后再继续压 compute 的极限

这不是说 compute 不重要，而是说：

> 在 edge workload 下，它未必总是第一优先。

如果你前面三层问题都还没处理好，过早继续压某个 compute kernel 的极限，往往只会让局部数字更漂亮，而系统体感变化不大。

## 十二、为什么这个顺序和 cloud 不一定一样

这点我想单独再强调一下。

在 cloud 下，更合理的优先级可能还是：

- compute
- low-precision throughput
- 再看 path tail

因为 cloud 更容易受这些因素驱动：

- 吞吐
- workload 覆盖面
- 通用 backend 收益

而 edge 更容易受这些因素驱动：

- 单次 latency
- 命中率
- jitter
- 输出阶段尾巴

也就是说，同一套 runtime，不同 deployment mode 下优化优先级本来就不应该完全一样。

## 十三、benchmark 在 edge 场景里应该怎么改口径

如果你真的要按 edge 优先级来做，那 benchmark 的口径也得跟着调整。

我会更建议边缘场景至少更强调这些指标：

- fused path 命中率
- fallback 频率
- `scatter/output` 占比
- P50 / P90 / P99
- camera-to-action latency

这比继续只盯：

- kernel 平均 latency
- 单个 compute 的最好成绩

要更贴近 edge 场景真实需要。

## 十四、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE path` 在 edge workload 下，优化优先级通常会更早地从“最重 compute”转向“最拖系统的执行链”，也就是更重视 path 命中率、fallback 频率、scatter/output 的 tail cost 和 layout/routing 的固定成本，而不是机械沿用云侧以算力为先的顺序。

因为在 edge 场景里，真正决定系统可用性的往往不是：

> 某个 kernel 能不能再快一点。

而是：

> 这条路径能不能稳定命中、稳定收尾、稳定把结果交给后续控制链路。
