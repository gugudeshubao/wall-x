# 18. Wall-X 的 MLP MoE packed weight 应该怎么组织

如果一路顺着前面的文章往下走，到这里基本会遇到一个越来越具体、也越来越绕不过去的问题：

> 既然 `MLP MoE` 这条路径要做 fused kernel、要做 low-precision、还要让 runtime 能规划，那权重到底应该怎么组织？

很多时候，大家对这个问题的第一反应会比较朴素：

> 不就是把原来的权重存一下、读一下吗？

但一旦真正进到算子和 runtime 设计里，你会发现事情没这么简单。

因为对 `Wall-X` 这种 `MLP MoE` 路径来说，权重组织方式会同时影响：

- kernel 输入接口
- memory access pattern
- tile 设计
- low-precision scale 的放置方式
- 是否容易做 expert-aware 执行
- planner 能不能稳定选择 backend

也就是说，`packed weight` 从来不只是“把 tensor 换个顺序存”这么简单。  
它本质上是在定义：

> runtime 和 kernel 到底怎样理解这条 expert 路径。

这篇文章就只讲这个问题。

## 一、先说结论：packed weight 不应该只是优化细节，而应该是 backend 合同的一部分

我现在越来越倾向于把 `packed weight` 看成 runtime 里的一个正式对象，而不是某个 kernel 内部的临时技巧。

换句话说，我更认可这种理解：

> `packed weight` 不是“某个优化版本顺手做的变换”，而是一种由 runtime 生成、由 backend 消费、并且和 precision policy、fused path 强相关的正式布局合同。

这一定义听起来有点重，但它能解决很多后面会越来越麻烦的问题。

如果你不这么做，常见后果通常是：

- 每个 kernel 都偷偷组织自己的权重布局
- 高精度和低精度路径的 weight format 完全脱节
- planner 不知道当前 backend 实际要求什么
- benchmark 结果很难解释到底对应哪种 weight 组织方式

所以第一步不是去想“具体怎么 pack”，而是先承认：

> `packed weight` 这件事应该被 runtime 正式建模。

## 二、为什么 MLP MoE 的 weight 组织特别值得单独设计

如果只是普通 dense MLP，很多时候权重布局问题还能相对简单一点。

但 `MLP MoE` 有几个额外复杂度：

- 有多个 expert
- 每个 expert 至少有 `gate/up/down`
- 后面还可能有低精度和 scale
- 路径可能要走 expert-aware fused compute

这意味着它天然比普通单一线性层更需要一种“专门为执行设计”的 weight 组织方式。

这里最容易掉进的误区有两个。

### 误区一：继续沿用训练态权重布局

训练态的权重布局，重点通常是：

- 易于保存
- 易于加载
- 易于被框架直接使用

但推理时的 fused path 更关心的往往是：

- 连续访问
- expert 分段
- tile 对齐
- 低精度 pack 后的布局一致性

这两个目标天然不完全一致。

### 误区二：每个 backend 自己定义一套私有布局

这短期看起来省事，但后面很容易把 runtime 撕裂。

因为你最终会得到：

- backend A 要一种 pack
- backend B 要另一种 pack
- 高精度一套
- 低精度一套
- planner 和 benchmark 无法统一解释

所以更现实的路线应该是：

> runtime 先定义出“对哪些维度必须显式化”，再允许 backend 在这个约束内做自己的 pack。

## 三、一个现实的 weight 组织问题到底包含哪些维度

如果把 `MLP MoE` 这条路径里的 packed weight 问题拆开来看，我觉得至少要同时回答下面四个维度。

### 1. 按什么分组

也就是：

- 按 expert 分
- 按 `gate/up/down` 分
- 还是进一步按某种 tile/block 分

这是第一层问题，因为它决定 runtime 里最基本的权重对象长什么样。

### 2. 按什么顺序排

也就是：

- 先 expert 再 projection
- 先 projection 再 expert
- 还是更底层的 tile-major 组织

这一步直接影响：

- kernel 读取顺序
- cache 局部性
- 是否容易做成 `gate/up` 成对读取

### 3. scale 和 quant metadata 放哪

一旦要支持 low-precision，这件事就必须明确。

例如：

- scale 是独立 tensor
- 还是和 packed weight 某种方式绑定
- 是 per-expert、per-group 还是 per-block

### 4. runtime 到底把哪一层当成“稳定接口”

这一步最容易被忽视。

因为 packed weight 最终不是只给 kernel 看，还会影响：

- planner 是否能知道当前 path 是否可用
- benchmark 是否能区分不同 pack format
- fallback 是否能平滑切回别的 path

## 四、我更推荐的第一版组织原则

如果现在真的要给 `Wall-X` 的 `MLP MoE` 设计第一版 packed weight，我更推荐几个比较务实的原则。

## 五、原则一：先按 expert 显式分组

这是我觉得最应该先立住的一点。

也就是说，runtime 层至少应该清楚表达：

- 当前有几个 expert
- 每个 expert 各自对应哪些 `gate/up/down` 权重对象

这件事看起来基础，但非常关键。

因为一旦 expert 边界不显式，后面很多东西都会变混乱：

- fused compute 的输入不好定义
- low-precision 的 scale 边界不好定义
- planner 也很难理解当前 path 的约束

所以第一版哪怕还不追求最强融合，也应该先保证：

> “按 expert 分组” 是 runtime 语义的一部分，而不是 kernel 私下知道的事情。

## 六、原则二：`gate/up` 应该优先被视为一组关系更近的权重

这条原则非常重要。

在 `MLP MoE` 路径里，`gate_proj` 和 `up_proj` 通常是天然更接近的一组：

- 它们都接同一个输入
- 它们都会在 `activation + combine` 之前被一起使用
- 它们在 fused compute 里经常更适合成对考虑

所以如果 runtime 要定义 packed weight 的第一版组织方式，我更建议：

> 不要把 `gate/up/down` 三者完全同等对待，而应该优先承认 `gate/up` 是一组更接近的对象。

这并不一定意味着第一版就要物理合并成一个 tensor。  
但至少在语义上，planner 和 backend 应该知道：

- `gate/up` 通常会一起被消耗
- `down` 更像后半段的单独阶段

这会让后面的 pack 设计顺很多。

## 七、原则三：高精度 pack 和低精度 pack 应该尽量共用上层语义

这一点是我现在越来越看重的。

第一版设计时，很容易出现一种诱惑：

> 高精度先做一套最简单的 pack，低精度以后再完全另起一套。

这短期看可能快，但长期很容易制造 runtime 分裂。

更好的做法通常是：

- 上层语义统一
- 下层 backend layout 可以不同

也就是说，runtime 层可以统一表达：

- `expert_id`
- `projection_group = gate_up | down`
- `precision_family = high | low`
- `pack_format`

而具体：

- 高精度怎么排
- 低精度怎么 pack
- scale 怎样贴着 weight 走

则交给 backend 层去实现。

这会让 planner 和 benchmark 的语言稳定很多。

## 八、原则四：pack format 必须可被 benchmark 感知

这是我觉得很多实现会漏掉的点。

很多时候 packed weight 做完之后，就像 backend 私有细节一样被藏起来了。  
但这样会让后面很多事情变得不可解释。

因为你最终会想知道：

- 某个 fused path 为什么快
- 某个 low-precision path 为什么没有预期收益
- 某次 fallback 到底是因为 kernel 不支持，还是因为 weight pack 没准备好

所以我更推荐 runtime 明确记录：

- 当前 path 使用的 pack format
- 当前 precision family
- 当前 backend 是否成功命中预打包路径

只有这样，benchmark 才能真正为系统设计服务。

## 九、一个更现实的第一版组织草图

如果不急着写成代码，而是先给第一版 `packed weight` 一个更现实的草图，我会更倾向于下面这种分层表达。

### 层 1：runtime 语义对象

例如你可以先把它理解成：

- `expert_weights[expert_id].gate`
- `expert_weights[expert_id].up`
- `expert_weights[expert_id].down`

如果要再进一步，也可以变成：

- `expert_weights[expert_id].gate_up_group`
- `expert_weights[expert_id].down_group`

这层重点不是物理 layout，而是让 runtime 的表达先稳定下来。

### 层 2：backend pack descriptor

也就是：

- 当前是 high precision 还是 low precision
- 当前 pack 是哪种 format
- 当前 scale/metadata 怎样附着

这层就是 runtime 和 backend 之间真正的合同。

### 层 3：physical packed buffer

这才是最终给 kernel 消费的具体内存对象。

这里可以完全按 backend 自己的需要去实现：

- tile-major
- block-major
- fused gate/up
- 单独 down
- 带 scale 的 block layout

但前提是：前两层语义已经清楚。

## 十、为什么这件事和 planner 强相关

packed weight 看起来像是 backend 细节，但实际上和 planner 关系很大。

因为 planner 要决定的不只是：

- 走哪条 path

还包括：

- 当前请求是否满足某种 pack format 的使用条件
- 当前低精度 path 是否已经有对应 packed weight
- 是否需要 fallback 到别的 backend

所以 planner 不能只知道“有 low-precision path”，它还应该知道：

> 当前这个 path 对应的 packed weight 是否已经就绪、是否匹配当前 shape 和 precision policy。

这也是为什么我一直觉得：

> packed weight 不该只是一个 tensor，而应该是 runtime 可理解的资源对象。

## 十一、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE packed weight` 第一版不该被当成某个 kernel 私下使用的存储技巧，而应该被设计成 runtime 和 backend 之间的一种正式合同：先显式按 expert 分组，再把 `gate/up` 视为更接近的权重组，在统一语义下分别承接高精度和低精度的不同 pack format。

对 `VLA Runtime` 来说，packed weight 真正重要的地方不只是“它能不能更快读”，而是：

> 它是不是能让 planner、backend、benchmark 和 fallback 全部说同一种语言。
