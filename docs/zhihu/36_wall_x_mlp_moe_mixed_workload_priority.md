# 36. Wall-X 的 MLP MoE path 在 mixed workload 下该怎样做优先级折中

前面两篇文章里，我分别把 `edge workload` 和 `cloud workload` 下的优先级说清楚了：

- 在 `edge` 下，更容易提前重视 path 命中率、fallback、`scatter/output` 的 tail cost 和 `layout/routing` 的固定成本  
- 在 `cloud` 下，更容易更长时间围绕 compute 主轴展开，也就是优先关注 `gate/up/down`、low-precision 覆盖率、pack 命中率和 path 覆盖面

但只要真正进到系统里，很快就会发现一个现实：

> 真正的 runtime，往往不是纯 edge workload，也不是纯 cloud workload，而是 mixed workload。

也就是说，你经常面对的不是一个单一世界，而是这些场景混在一起：

- 一部分请求很像 edge：batch 小、shape 稳、对 jitter 敏感  
- 一部分请求很像 cloud：覆盖面更宽、shape 更散、对 path 覆盖和平均收益更敏感  
- 同一个系统里，甚至同一条 `MLP MoE path`，可能同时面对这两类压力

这时问题就会自然变成：

> 如果 workload 是混合的，那优先级到底该怎么折中？

这篇文章就只讲这个问题。

## 一、先说结论：mixed workload 下不该再问“哪一类优先级绝对正确”，而该问“哪一种折中能让 planner 少犯系统级错误”

如果把我的判断压成一句话，我会说：

> 在 mixed workload 下，`Wall-X` 的 `MLP MoE path` 不该继续用“edge 优先级”或“cloud 优先级”二选一的思路来推进，而应该优先寻找一种能让 planner 在不同 workload 间少犯系统级错误的折中顺序。

这里最重要的词其实不是“折中”，而是：

> 少犯系统级错误

为什么？

因为 mixed workload 真正危险的地方不在于：

- 某一段没有被优化到极致

而在于：

- 你把系统整体优化方向带偏了  
- planner 过于偏向某一类 workload  
- benchmark 给出的结论被另一类 workload 稀释  
- 最终每一类场景都只拿到半吊子收益

所以 mixed workload 下最先要避免的，不是“局部还不够完美”，而是“全局判断已经失真”。

## 二、为什么 mixed workload 比看起来更麻烦

如果只是分别看 `edge` 和 `cloud`，很多优先级判断其实都还算清楚。  
麻烦恰恰在于，当它们混在一起之后，几类矛盾会一起出现：

### 1. 平均值会变得很危险

例如：

- 某条 path 在 `cloud` 请求上平均收益不错  
- 但在 `edge` 请求上会偶发很差的 tail latency

如果你只看总体平均值，很容易误以为它是“正确的下一步”。

### 2. 主瓶颈会变得不单一

在 mixed workload 下，你很可能同时看到：

- 一类请求主要受 compute 拖累  
- 另一类请求主要受 tail cost 和 fallback 拖累

这时“先优化哪一段”不再有一个非常纯净的答案。

### 3. planner 的错误代价会上升

在单一场景里，planner 偶尔选错路径，可能只是让该场景稍微慢一点。  
但在 mixed workload 下，planner 的偏置会被放大成系统级问题：

- 对 edge 太激进  
- 对 cloud 太保守  
- 或反过来

所以 mixed workload 下，很多问题会更早地从“算子问题”变成“路径治理问题”。

## 三、先分清楚：mixed workload 下最容易被搞混的三类目标

我觉得这件事必须先拆开，否则“折中”很容易变成一句空话。

在 mixed workload 下，最容易被混起来的其实是三类不同目标：

### 1. 局部 compute 最优

也就是：

- 某个 kernel
- 某个 fused stage

本身能不能跑得更快。

### 2. 路径级收益最优

也就是：

- 这条 `MLP MoE path` 在一类 workload 上是否更值得被命中。

### 3. 系统级风险最小

也就是：

- planner 是否容易选错  
- fallback 是否容易放大尾延迟  
- benchmark 是否容易被平均值误导

如果不先把这三层拆开，mixed workload 下的所有讨论都很容易失真。

因为很多时候真正该优化的，并不是：

> 哪个局部段最快

而是：

> 哪个方向最能让系统在 mixed workload 下少犯结构性错误。

## 四、为什么 mixed workload 下通常不能再简单沿用 cloud 的优先级

如果只看 cloud 的直觉，很容易得出这样的排序：

- 先压 compute 主轴  
- 再扩 low-precision 覆盖  
- 再提高 pack 命中率  
- 再看 tail cost

这套顺序在 cloud-only 场景里是很有道理的。  
但放到 mixed workload 下，它有一个很大的风险：

> 它会低估那些虽然均值不大，但对 edge-like 请求伤害很强的尾部问题。

例如：

- `scatter/output` 的偶发抬高  
- fallback 带来的尾延迟放大  
- 某类 shape 命不中目标 path

这些问题在总体平均值里可能不突出，但对 mixed workload 下的体验影响很大。

所以 mixed workload 下，如果完全沿用 cloud 优先级，系统很容易“平均意义更好，但实际体验更差”。

## 五、为什么 mixed workload 下也不能完全沿用 edge 的优先级

反过来，如果只看 edge 的直觉，也很容易走向另一头：

- 过早把所有注意力放到 tail cost  
- 过早为 batch=1 和固定 shape 特化  
- 过度保守地降低某些 path 的使用范围

这样的问题在 mixed workload 下也会很明显：

> 系统为了保护 edge-like 请求，过早放弃了在更宽 workload 上可以稳定兑现的 compute 和 low-precision 收益。

于是最后就会出现：

- 系统的 worst case 控住了一点  
- 但整体覆盖面和平均收益又被压住了

所以 mixed workload 的核心难点，从来不是“选 edge 还是选 cloud”，而是：

> 哪些问题该优先保证不出错，哪些问题可以后面再继续深挖。

## 六、一个更现实的折中顺序

如果让我给 `Wall-X` 的 `MLP MoE path` 在 mixed workload 下排一个更现实的优先级，我会更倾向于下面这个顺序。

## 七、第一优先：先把 path hit / fallback 这一层稳住

这是我最想强调的一点。

在 mixed workload 下，最容易把系统带偏的其实不是某个 compute 没压到极致，而是：

- path 命中率不稳  
- fallback 行为不清楚  
- 一类 workload 持续在吃另一类 workload 的路径代价

所以我更推荐第一优先先放在：

- 哪些请求命中哪条 path  
- 哪些请求为什么 fallback  
- fallback 是否具有明显 workload 偏置

这一步之所以放第一，不是因为它最“硬核”，而是因为：

> mixed workload 下最先应该控制的是系统偏置，而不是局部算力极限。

## 八、第二优先：再看 path-level stage breakdown，而不是立刻只看单个 kernel

一旦 hit / fallback 基本清楚，下一步我更推荐去看 path-level breakdown：

- `layout/routing`
- `core compute`
- `scatter/output`
- `path total`

为什么这一步比继续盯某个 kernel 更适合放第二？

因为 mixed workload 下，真正会变化的往往不是某个单点，而是：

> 不同 workload 会把同一条 path 的重心推向不同 stage。

只有先从 path 视角看清楚，后面你才知道：

- 哪些问题是 cloud-like 的  
- 哪些问题是 edge-like 的  
- 哪些问题是所有 workload 共享的

## 九、第三优先：再压 compute 主轴

这一步在 mixed workload 下仍然重要，但我不建议一上来就把它放在绝对第一。

更现实的节奏是：

- 先保证 path 命中和回退不会把系统带偏  
- 再确认 path 级 stage 结构  
- 然后继续压 compute 主轴

到这一步时，你会更知道：

- `gate/up/down` 的收益究竟服务了哪一类 workload  
- low-precision 的路径收益究竟能覆盖哪些 workload

这会让 compute 优化更像“精准打击”，而不是“平均意义上的好看”。

## 十、第四优先：再针对 mixed workload 调整 tail cost

这一点也很重要。

在 mixed workload 下，`scatter/output` 不一定像 edge-only 场景那样早早升到最前，但它通常也不会像 pure cloud 那样可以一直往后排。

更现实的状态往往是：

> 一旦 compute 主轴已经较清楚，`scatter/output` 会很快因为 mixed workload 中 edge-like 请求的存在而重新抬头。

所以它更像 mixed workload 下的“第二轮主矛盾”，而不是最早的第一主矛盾。

## 十一、为什么 mixed workload 特别需要分层 benchmark，而不是只看平均值

这点必须单独说。

如果你在 mixed workload 下还只看：

- overall average latency

那系统几乎一定会被误导。

我更推荐至少同时看：

- edge-like workload 的 path 命中率和 tail  
- cloud-like workload 的 compute 占比和覆盖面  
- mixed set 里的整体 path-level breakdown

也就是说：

> mixed workload 的 benchmark，不该是把所有请求混成一个均值，而应该是先分类、再聚合。

否则你后面做的很多“折中优化”，其实只是把不同问题平均了，而不是解决了。

## 十二、planner 在 mixed workload 里真正该学会什么

如果把 mixed workload 说到底，它最终会把问题推给 planner。

planner 最终真正要学会的不是：

- 统一给所有请求选一条最优 path

而是：

> 在不同请求类型之间，避免持续犯同一种系统性偏置错误。

例如：

- 不要长期偏向 cloud-like path，结果 edge-like 请求持续被尾延迟伤害  
- 也不要过度偏向 edge-safe path，结果 cloud-like 请求一直吃不到应有的 compute 收益

所以 mixed workload 下，planner 真正最先要学会的是：

> workload-aware 的折中，而不是某种单一标准下的最优。

## 十三、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> 在 mixed workload 下，`Wall-X` 的 `MLP MoE path` 不该继续简单沿用 edge 或 cloud 的单侧优先级，而应该先稳住 path hit / fallback，先用 path-level benchmark 看清不同 workload 把哪一段推成主要矛盾，再去继续压 compute 主轴，并在第二轮再处理更容易伤害 edge-like 请求的 tail cost。

因为 mixed workload 下真正最先该避免的，不是：

> 某个局部还不够极致。

而是：

> 系统已经因为 workload 混合而开始沿着错误的全局优先级一路优化下去。
