# 22. Wall-X 的 MLP MoE 从 BF16 fused 到 low-precision fused 的演化路线

前面几篇文章里，我已经把 `Wall-X` 的 `MLP MoE` 路径从几个层面拆开了：

- 为什么第一条 fused path 很适合先落在 `MLP MoE`
- `MLP MoE fused kernel` 的接口应该怎么拆
- 数据流该怎么组织
- low-precision 第一批该先落在哪一层
- packed weight 应该怎样作为 runtime/backend 合同来设计

如果继续往实现推进，下一步最自然的问题其实已经很明确了：

> 既然第一版建议先把 `BF16 fused` 路径做稳，那么它到底应该怎样一步步演化成 `low-precision fused` 路径？

这件事如果只从口号上看，很容易被讲成一句：

> 先做高精度，再做低精度。

但真正落到工程里，这句话远远不够。

因为你真正需要回答的是：

- 先把哪些边界做稳
- 先在什么地方引入 low-precision
- 哪些步骤必须继续保持更高精度
- 哪一阶段开始引入 packed weight
- 哪一阶段 planner 才应该真正把 low-precision path 当成常规候选

这篇文章就只讲这个演化路线。

## 一、先说结论：不要把 low-precision 理解成“BF16 fused 之后顺手降一下 dtype”

这是我最想先说清楚的一点。

很多时候，大家很容易把这条路线理解成：

1. 先有一个 `BF16 fused kernel`
2. 然后把里面某些 tensor 的 dtype 换掉
3. 就变成 `low-precision fused kernel`

这种理解会严重低估问题。

因为一旦从 `BF16 fused` 真正走向 `low-precision fused`，变化的不只是 dtype，还包括：

- weight format
- scale 组织方式
- backend capability
- precision policy
- benchmark 口径
- fallback 层级

也就是说，这不是“在已有 kernel 上打一层薄薄的 dtype patch”，而是：

> 以 `BF16 fused` 为稳定语义基线，逐步引入一整套新的后端能力。

所以我更愿意把它理解成一个多阶段演化过程，而不是一次性切换。

## 二、为什么 BF16 fused 必须先稳定

这件事在前面几篇里其实已经反复提到过，但这里必须再强调一遍。

如果 `BF16 fused` 本身还不稳定，就过早推进 `low-precision fused`，通常会有三个直接问题：

### 1. 你会同时失去结构边界和数值边界

也就是说：

- 你既不知道 fused path 的系统边界是不是对的
- 也不知道数值问题到底来自哪里

### 2. benchmark 会失去解释力

如果第一版 high precision path 还没站稳，你后面看到的 low-precision 数字就很难判断：

- 是路径本身更好
- 还是某些高精度路径本来就还没做对

### 3. planner 和 fallback 会被过早复杂化

一旦高精度 fused 还不稳，planner 里就会被迫同时面对：

- 路径成熟度问题
- 低精度兼容性问题
- backend 能力不稳定问题

这会让系统非常难收敛。

所以我会坚持一个判断：

> `BF16 fused` 不是“过渡版本”，它是整条 `low-precision fused` 路线的稳定语义地基。

## 三、第一阶段：先把 BF16 fused 当成语义模板

如果把演化路线拆开看，我最推荐的第一阶段不是“追求极限速度”，而是先把 `BF16 fused` 做成一个结构稳定的模板。

这一步的重点是：

- 输入输出边界清楚
- `gate/up/down` 的顺序清楚
- expert-ordered layout 是否保留清楚
- scatter 的职责清楚
- benchmark 的分层清楚

换句话说，这一阶段真正要得到的，不只是一个能跑的 kernel，而是：

> 一条 runtime 可以理解、planner 可以选择、benchmark 可以拆分的 fused path。

这一步越稳，后面低精度接起来越容易。

## 四、第二阶段：先把 weight pack 和 precision policy 引入，但不急着全开

我觉得很多人最容易跳过的，就是这个中间阶段。

一看到要做 low-precision，就会自然想：

> 那下一步就直接把计算也降精度。

但实际上，我更建议先补一层：

> 先让 runtime 明确“自己以后要怎样承接低精度”，而不是立刻把所有计算都切过去。

这一步通常包括：

- 引入 `packed weight` 的正式描述
- 引入 low-precision 相关 metadata
- 让 `precision policy` 开始能表达更细的意图
- 让 planner 开始知道某些 path 未来可支持 low-precision

这一步的价值在于：

> 它先把后端语言准备好，再去改后端行为。

这会让系统演化得更稳。

## 五、第三阶段：先让 gate/up 走 low-precision fused

这一步是我最推荐的第一批真正落地 low-precision 的地方。

原因前面已经讲过很多次，但这里再收束一下：

- `gate/up` 都接同一个输入
- 它们在语义上天然是一组
- 它们在 `activation + combine` 之前被一起消费
- 它们通常是最重、也最像“可统一处理”的那组线性层

所以我更推荐把这一步理解成：

> 不要先问“整条 MLP MoE 能不能 low-precision”，而先问“`gate/up` 这组双投影能不能先 low-precision fused”。

这件事一旦成了，很多东西都会变得更清楚：

- packed weight 格式是否合理
- scale 放置方式是否合理
- `activation + combine` 前后的数值行为是否可接受

这会给后面继续下沉提供非常有价值的反馈。

## 六、第四阶段：再让 down 接上 low-precision fused

当 `gate/up` 这组已经比较稳了，下一步通常才是：

> 把 `down` 也纳入 low-precision fused 路线。

这一阶段的意义在于，`MLP MoE` 这条 compute 路径的三块主要线性层终于开始形成一个更完整的 low-precision backend。

但这一步之所以不该抢在前面，是因为：

- `down` 会更靠近输出语义
- 一旦出问题，更容易放大到后续路径
- 它和前面阶段相比，更像“把 low-precision 从局部能力扩展到完整 compute path”

所以更好的顺序通常是：

- 先在 `gate/up` 里验证
- 再在 `down` 里补完整条链

## 七、第五阶段：再考虑更深的融合和更激进的 low-precision

只有当前面这些都比较稳的时候，我才会建议再往更深的方向走。

例如：

- 更激进的 block-scale
- 更低的 bit width
- 更强的 fused epilogue
- `scatter` 是否继续下沉
- 局部 `megakernel` 是否值得做

这一步如果太早做，很容易让你在系统还没把“基本秩序”立住之前，就提前进入极度复杂的调试状态。

所以我更倾向于把它定义成：

> `BF16 fused -> 基础 low-precision fused -> 完整 low-precision fused` 之后的下一阶段，而不是一开始就要追求的东西。

## 八、planner 在这个演化路线里什么时候真正介入

这一点也很重要。

很多人会觉得 low-precision 主要是 backend 的事，但其实 planner 介入的时机很关键。

我更推荐的节奏是：

### 在第一阶段

planner 只需要知道：

- 有 baseline path
- 有 `BF16 fused` path

### 在第二阶段

planner 开始知道：

- 某条 path 有 low-precision capability
- 但可能还不应该默认启用

### 在第三、四阶段

planner 才真正开始把 low-precision fused path 当成正式候选路径：

- `low_precision_gate_up_fused`
- `low_precision_full_mlp_moe_fused`

这样它不会过早把系统带入过度复杂的策略空间。

## 九、benchmark 在这个演化路线里应该怎么看

如果只说实现顺序，还是不够。

这条演化路线真正要稳住，benchmark 的观察重点也要跟着变化。

### 阶段一关注

- `BF16 fused` 相对 baseline 的收益
- 路径边界是否清楚
- `layout/compute/scatter` 各阶段比例

### 阶段二关注

- packed weight 和 metadata 是否稳定命中
- precision policy 是否能被正确记录

### 阶段三关注

- `gate/up` low-precision fused 的局部收益
- 对 path-level latency 的贡献
- 是否出现数值或 fallback 异常

### 阶段四关注

- 完整 `MLP MoE` low-precision fused 的整体收益
- 是否真的比 `BF16 fused` 更值得常态化使用

也就是说：

> 不是同一套 benchmark 指标机械地一路往下跑，而是每个阶段看重点都会有变化。

## 十、为什么这条路线比“一步到位 low-precision fused”更值

如果只从短期代码量看，分阶段推进似乎更慢。  
但从 runtime 视角看，它往往更值，原因就在于：

### 1. 每一步都更可解释

你知道自己在验证什么。

### 2. 每一步都更容易 benchmark

你知道该看哪组指标。

### 3. planner 和 fallback 都能同步长起来

不会在某天突然被一条难以解释的新路径打爆。

### 4. packed weight 和 precision policy 不会成为附属品

而会自然变成 runtime 的正式能力。

## 十一、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE` 从 `BF16 fused` 演化到 `low-precision fused`，不应该被理解成一次性换 dtype，而应该被理解成一条分阶段演化路线：先稳定高精度 fused 语义边界，再引入 packed weight 和 precision policy，再先让 `gate/up` 落地 low-precision fused，最后补完整条 compute path，并在更后面再考虑更激进的融合和 bit-width。

对 `VLA Runtime` 来说，这种演化路线最值钱的地方不是“它够慢”，而是：

> 它让 low-precision 从一堆 scattered tricks，变成一条真正能被 planner、backend、benchmark 和 fallback 共同理解的系统路径。
