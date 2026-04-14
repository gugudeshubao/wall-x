# 16. Wall-X 的 MLP MoE low-precision path 应该先落在哪一层

前面几篇文章里，我已经把几件事分别说清楚了：

- 第一条 fused path 很多时候更适合先落在 `MLP MoE`
- `MLP MoE fused kernel` 的接口和数据流应该怎么拆
- `precision policy` 不该是几个散落的 dtype 开关

如果继续往实现层推进，下一个真正会挡在你面前的问题就是：

> 既然低精度最终要做，那 `MLP MoE` 这条路径里，第一批 low-precision 应该先落在哪一层？

这个问题如果回答得太粗，很容易走成两种都不太理想的路线：

### 路线 A

“反正都要低精度，干脆从第一天起全链路一起量。”

### 路线 B

“低精度太复杂，先完全不碰，等所有高精度路径都做完再说。”

我现在更认同的答案，是介于两者之间的：

> 低精度要尽早纳入设计，但第一批真正落地的地方，应该优先选那些既最自然、又最容易验证收益、还最不容易把系统语义搞乱的层。

而对 `Wall-X` 这种 `MLP MoE` 路径来说，我更推荐的顺序通常不是“全都一起降”，而是分层推进。

这篇文章就只讲这个顺序。

## 一、先说结论：第一批 low-precision 最适合优先落在权重最重、语义最稳的线性层

如果把问题压缩成一句话，我现在的判断是：

> 在 `MLP MoE` 这条路径里，第一批 low-precision 最适合先落在 `gate/up/down` 这类线性层权重及其 matmul 上，而不是先碰 routing、激活、scatter 或更靠系统边界的部分。

原因并不复杂。

因为 low-precision 真正最容易带来直接收益的地方，通常同时满足下面几个条件：

- 权重带宽重
- 结构规则
- 可预打包
- 易于和 fused compute 结合
- 数值角色相对清楚

而 `gate_proj`、`up_proj`、`down_proj` 正好高度符合这些条件。

相反，像这些部分则不太适合作为第一批落点：

- routing metadata
- `expert_offsets / row_id_map`
- 激活控制逻辑本身
- scatter / output adapter

这些地方不是完全不能碰，而是更不适合作为第一批 low-precision 的主落点。

## 二、为什么不能一上来就全链路一起降精度

很多人对 low-precision 的第一反应是：

> 如果带宽和显存是问题，那当然越多地方降精度越好。

这在长期可能没错，但在第一版实现里通常风险太大。

原因主要有四个。

### 1. 调试变量会同时爆炸

如果你同时改：

- 权重精度
- 激活精度
- epilogue
- scatter 路径
- 输出路径

那一旦系统行为异常，你就很难知道是哪一层出了问题。

### 2. benchmark 的解释力会下降

你看到延迟变化时，很难分辨：

- 是线性层降精度带来的收益
- 还是某个 scatter 路径改动造成的副作用

### 3. planner 和 fallback 会变复杂

如果第一版 low-precision 语义太宽，planner 很快就会面对一堆很难解释的条件分支。

### 4. 动作相关路径更容易放大数值问题

在 `VLA` 里，数值误差未必只体现为“模型输出差一点”，它可能直接体现在：

- 动作抖动
- 轨迹偏移
- 稳定性下降

所以第一批 low-precision 最好先放在那些“更像纯算力/带宽问题”的地方，而不是先碰更靠控制语义边界的部分。

## 三、为什么 `gate/up/down` 是最自然的第一批落点

如果把 `MLP MoE` 路径拆开来看，最自然的低精度首批落点通常就是：

- `gate_proj`
- `up_proj`
- `down_proj`

原因我会归成五条。

### 1. 它们最重

这条最直接。

在 `MLP MoE` 路径里，最典型的重计算和重带宽负担，本来就集中在这些线性层。

所以如果低精度想立刻带来收益，这里几乎一定是最值得优先试的地方。

### 2. 它们最规则

规则结构意味着：

- 更容易预打包
- 更容易定义 weight layout
- 更容易和某种统一 tile 策略结合

这对第一版 low-precision 特别重要。

### 3. 它们最适合和 fused compute 融在一起

如果你已经有：

- `expert-ordered input`
- `gate/up/down` 的固定语义顺序
- 中间 `activation + combine`

那么低精度最自然的插入点，本来就应该在这条 compute 链内部。

### 4. 它们最容易做成 backend 能力

也就是说，你更容易把 low-precision 表达成：

- packed weights
- scale metadata
- low-precision matmul kernel
- epilogue 选择

这些都很适合成为 backend 层能力，而不是散落在 Python 调用点里的特殊逻辑。

### 5. 它们最容易先拿到“可验证的 first win”

第一批 low-precision 最重要的不只是快，而是：

> 快得足够明显，而且问题足够容易被解释。

`gate/up/down` 正好满足这一点。

## 四、第一批不建议优先降精度的部分有哪些

为了把边界说清楚，我也把“先别碰哪里”写明确一点。

## 五、第一类：routing metadata

例如：

- `expert_ids`
- `expert_offsets`
- `row_id_map`
- 各类索引相关 tensor

这些东西在存储和带宽上不是完全没有成本，但它们不是 `MLP MoE` low-precision 第一批的最优落点。

原因是：

- 它们的收益通常不如线性层直接
- 它们更容易影响系统语义正确性
- 很多时候它们更适合优先做 layout/缓存优化，而不是先做低比特

## 六、第二类：激活和控制逻辑本身

例如：

- 激活函数本身的执行
- 某些中间 gating 逻辑

这些部分当然也会受精度影响，但第一批通常更适合先让它们待在相对稳的精度里。

原因很简单：

> 第一版 low-precision 的目标是先抓最大收益，而不是让每一个局部都变成实验区。

## 七、第三类：scatter / output adapter

这一类路径更靠系统边界。

如果你太早在这里引入 low-precision，很容易把“kernel 优化问题”和“系统输出语义问题”缠在一起。

对第一版实现来说，这通常不划算。

## 八、一个更现实的落地顺序

如果让我给 `Wall-X` 这条 `MLP MoE` path 设计第一批 low-precision 落地顺序，我会更推荐像下面这样。

### Stage 1：先做高精度 fused 基线

先把：

- `expert-ordered input`
- `gate/up/down`
- `activation + combine`
- `expert-ordered output`

这条链跑稳。

这一步的意义不是保守，而是为了：

> 先把 low-precision 以后要落的结构边界做对。

### Stage 2：先给 `gate/up` 接 low-precision

这是我最推荐的第一刀。

因为：

- `gate/up` 常常天然成对出现
- 中间还会进入同一个 `activation + combine`
- 它们是最像“同一组投影”的部分

所以这一步的收益和设计清晰度通常都比较好。

### Stage 3：再给 `down` 接 low-precision

这一步往往会比前一步更接近完整路径收益，因为它补全了：

- 输入投影
- 中间 combine
- 输出投影

到这里，`MLP MoE` compute 路径的大头 low-precision 通常已经成型。

### Stage 4：再评估是否要进一步下沉到更广的范围

例如：

- 更激进的 block-scale
- 更低的位宽
- 更深的 epilogue 融合

这一步不应该一上来就做，而应该在前几步收益已经明确之后再推进。

## 九、这件事和 precision policy 的关系是什么

这里其实能看出一件重要的事：

> “low-precision 先落在哪一层” 本身就应该是 `precision policy` 的一部分。

换句话说，runtime 不应该只知道：

- 当前请求想跑低精度

它还应该知道：

- 当前哪些模块允许先降精度
- 当前哪些模块仍然保持高精度
- 某条 fused path 当前到底支持到什么程度

也就是说，真正成熟的 `precision policy` 不是一句：

> 这次跑 `int8`。

而更像是：

> 这次在 `MLP MoE` 上优先让 `gate/up/down` 使用低精度后端，其余边界相关部分保持高精度安全路径。

这才是一种真正可规划、可回退、可 benchmark 的 runtime 规则。

## 十、这件事和 planner 又是什么关系

一旦“low-precision 先落在哪一层”这个问题被讲清楚，planner 的工作也会变得更明确。

planner 不必只做一个粗糙决策：

- 开低精度
- 关低精度

它可以做更细一点、也更合理的决策：

- 当前请求允许 `MLP MoE` low-precision
- 当前请求仍然要求 output path 保持安全精度
- 当前 fused path 只启用 `gate/up/down` 低精度版本

也就是说，这种“分层落地”的 low-precision 策略，本身就是 planner 真正能执行的东西。

## 十一、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE low-precision path` 第一批最适合优先落在 `gate_proj / up_proj / down_proj` 这类线性层和它们对应的 matmul 上，而不是先把 routing、激活控制逻辑和 scatter/output 一起拖进低精度实验区。

因为对第一版实现来说，最重要的不是“尽可能多地降精度”，而是：

> 先在最自然、最重、最容易验证收益的那一层，把 low-precision 做成一条真正能继续往下长的 path。
