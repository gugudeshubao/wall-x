# 17. VLA Runtime 在 edge 和 cloud 下的最小差异化策略

前面讲 `Planner` 的时候，我已经把一个判断提出来了：

> `cloud` 和 `edge` 在 `VLA Runtime` 里，不应该只是设备位置标签，而应该是执行策略标签。

但如果只说到这里，还是很容易显得抽象。

因为真正做系统时，大家马上会问：

> 那到底该怎么分？

也就是说：

> `VLA Runtime` 在 `edge` 和 `cloud` 下，最小差异化策略到底应该长什么样？

这篇文章就只讲这个问题。

我会尽量把它收得很实，不去空谈“未来统一框架”，而是先回答：

> 第一版 runtime 到底该在哪些地方显式区分 `edge` 和 `cloud`？

## 一、先说结论：不要一开始追求“一套路径跑遍 edge 和 cloud”

很多系统在早期都会天然有一个愿望：

> 最好同一套运行路径，在端侧和云侧都能直接复用。

这个愿望在工具层面可以理解，但在 runtime 设计层面往往太理想化。

原因很简单。

`edge` 和 `cloud` 在 `VLA` 场景里，优化目标通常就不一样：

### cloud 更常见的目标

- 更高吞吐
- 更灵活的 workload 兼容性
- 对动态 shape 更宽容
- 更容易接受较复杂的通用后端

### edge 更常见的目标

- 更低单次延迟
- 更低 jitter
- batch=1 的稳定性
- 更愿意牺牲通用性换固定 shape 优化

所以如果你硬要让它们从第一天起共享完全同一条路径，最后常见的结果通常是：

- cloud 没吃到它本该吃到的通用优化
- edge 也没拿到它真正需要的低时延优化

所以我更认可的策略是：

> 第一版就显式承认 `edge` 和 `cloud` 是两种不同的 execution mode。

## 二、最小差异化策略应该先落在哪几层

如果按前面定义的最小执行图来看：

- `Input Adapter`
- `Planner`
- `Kernel Backend`
- `Output Adapter`

我不建议四层都一开始做完全不同实现。  
第一版更现实的做法，是在下面三层先拉开差异：

### 1. Planner

这是最应该先分的地方。

因为：

- 是否优先 latency
- 是否允许更激进 fused path
- 是否优先固定 shape 特化
- 是否允许更激进的 low-precision policy

这些，本来就应该由 planner 来决定。

### 2. Kernel Backend

这也是必须分的地方。

因为哪怕同一组算子语义，`edge` 和 `cloud` 最优后端也未必一样：

- cloud 更可能优先通用 fused op
- edge 更可能优先固定 shape、graph、局部 `megakernel`

### 3. Benchmark

虽然 benchmark 不是执行层，但它必须跟着分。

因为：

- `cloud` 更该看吞吐和路径兼容性
- `edge` 更该看 camera-to-action latency 和 jitter

如果 benchmark 不分，planner 后面也分不对。

## 三、第一版最不需要先分开的是什么

为了避免系统一上来就分叉过度，我也想把“不急着分什么”说清楚。

## 四、第一类：高层输入语义

图像、文本、状态这些输入的“语义结构”通常不需要在 `edge/cloud` 上完全分两套。

也就是说：

- `Input Adapter` 的接口形式可以尽量统一
- 输入描述对象可以尽量统一

真正差异化的，通常不是“输入是什么”，而是：

> 这些输入在后面怎样被规划和执行。

## 五、第二类：系统目标对象

例如：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`

这些 runtime 核心对象，我更建议尽量统一，而不是为 `edge` 和 `cloud` 各自发明一套不同语义。

因为一旦这层也分裂，后面整套系统就很难再形成统一语言。

## 六、那最小差异化策略具体该怎么写

如果让我给第一版 runtime 写一个很朴素的 `edge/cloud` 差异化策略，我会建议先从下面三条开始。

## 七、策略一：Planner 先分“目标”而不是先分“设备”

这是我最看重的一条。

也就是说，planner 里先显式表达：

- `deployment_mode = edge`
- `deployment_mode = cloud`

而不是到处根据硬件条件隐式猜。

因为这一步表达的不是“你在哪里跑”，而是：

> 这次请求到底优先优化什么。

例如：

### `cloud` policy 倾向

- 优先兼容更多 workload
- 优先复用更通用的 fused path
- 更接受动态 shape
- 可把低精度作为平衡型选择

### `edge` policy 倾向

- 优先 batch=1 低延迟
- 优先固定 shape path
- 优先 jitter 更稳的策略
- 更积极评估 graph 和局部 `megakernel`

这一步一旦明确，后面很多模块都更容易协作。

## 八、策略二：Kernel Backend 先分“候选路径”，再分“极致优化”

第一版不要追求：

- cloud 一套超复杂 backend
- edge 一套完全不同 backend

我更推荐的是：

### 先共享语义层

也就是：

- baseline path
- fused path
- low-precision variants

这套“路径名字”先统一。

### 再让 `edge/cloud` 选择不同候选

例如：

- `cloud` 默认先在 baseline 和通用 fused path 之间选
- `edge` 默认更积极尝试 fixed-shape path、graph path 或局部特化 path

这样做的好处是：

- 系统语言统一
- 后端实现仍然能差异化
- benchmark 更容易横向比较

## 九、策略三：Benchmark 先分主指标

这是最容易被忽视，但其实非常关键的一步。

如果 benchmark 仍然只给一套统一指标，你最后很容易又把 `edge/cloud` 做成表面区分。

我更推荐的最小差异化是：

### cloud 重点看

- 吞吐
- 动态 shape 下的稳定表现
- baseline/fused/low-precision 各路径兼容性

### edge 重点看

- cold start
- warm start
- batch=1 steady-state latency
- camera-to-action latency
- jitter

也就是说，`edge/cloud` 差异化不只是 planner 的配置差异，还要体现在 benchmark 评价目标差异上。

## 十、为什么第一版不要把 edge 特化做得太满

这点我想特别提醒。

很多人一想到端侧，就会自然想：

> 那是不是第一版就应该直接冲最激进的 edge 特化？

比如：

- 完全固定 shape
- 全路径 graph
- 局部 `megakernel`
- 更激进 low-precision

这些方向当然值得做，但不一定适合作为第一版最小差异化策略。

原因是：

### 1. 你可能还没建立稳定 baseline

如果 baseline 还不够清楚，过早做满 edge 特化，后面很难判断收益到底来自哪里。

### 2. planner 和 benchmark 还没形成闭环

没有闭环时，激进特化很容易让系统变得不可解释。

### 3. 你的系统语言还没统一

如果在路径名字、policy 语义、benchmark 口径都没统一时就先冲最深 edge 特化，后面会很难收回来。

所以我更推荐：

> 第一版先把 `edge/cloud` 做成“最小差异化”，而不是“最强分叉化”。

## 十一、一个更现实的第一版 edge/cloud 分流草图

如果把它压缩成最小执行规则，我会更推荐像这样：

### Step 1

`Input Adapter` 统一输入语义和 metadata。

### Step 2

`Planner` 先根据：

- deployment mode
- shape stability
- batch size
- current precision policy

产出：

- `cloud plan`
或
- `edge plan`

### Step 3

`Kernel Backend` 在统一路径名下选择不同实现偏好。

例如：

- `baseline`
- `fused`
- `low_precision_fused`
- `fixed_shape_specialized`

### Step 4

`Benchmark` 用不同主指标验证：

- `cloud` 是否更有效率
- `edge` 是否更低延迟、更低 jitter

这就是我觉得最像“最小可行差异化策略”的形式。

## 十二、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 在 `edge` 和 `cloud` 下，第一版最应该做的不是追求两套彻底不同的系统，而是先在 `Planner`、`Kernel Backend` 和 `Benchmark` 三层上做最小但明确的差异化，让两种 deployment mode 从第一天起就有不同的优化目标和不同的执行偏好。

因为真正重要的不是“系统有没有分流”，而是：

> 它是不是从第一版开始，就知道自己为什么要分流，以及分流之后到底在优化什么。
