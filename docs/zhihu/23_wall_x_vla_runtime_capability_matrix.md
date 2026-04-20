# 23. VLA Runtime 的 capability matrix 应该怎样表达

一路把 `VLA Runtime` 写到这里，很多关键对象其实已经慢慢浮出来了：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`
- `Pack/LayoutDescriptor`
- `precision policy`
- `fallback chain`

再往前走一步，很快就会碰到一个更系统化的问题：

> planner 到底依据什么来判断当前 path 能不能走？

很多时候，大家会把这个问题直接写成大量零散条件：

- 这个 backend 支持 `bf16`
- 那个 kernel 支持固定 shape
- 另一个 path 只支持某种 pack
- edge 模式下又要额外判断一堆前提

短期当然能跑，但一旦系统继续长大，就会遇到一个非常典型的问题：

> 所有能力约束都散落在代码里，planner 越来越像一大坨写死的特判逻辑。

这也是为什么我越来越觉得，`VLA Runtime` 到这个阶段之后，必须补一层新的抽象：

> capability matrix

也就是：

> 系统到底应该怎样显式表达“某条 path 具备什么能力、需要什么前提、适合什么 workload、在什么条件下应该 fallback”？

这篇文章就只讲这个问题。

## 一、先说结论：capability matrix 不是文档表格，而是 planner 的事实来源

很多人一看到 “capability matrix” 这个词，第一反应容易是：

> 哦，就是写张表记录一下哪些 backend 支持什么。

这种理解不完全错，但在 runtime 里远远不够。

我更认可的定义是：

> capability matrix 不应该只是文档里的说明表，而应该是 planner 在做执行决策时真正依赖的一份事实来源。

也就是说，它不是“看着参考一下”，而应该是：

- planner 生成 plan 的输入之一
- benchmark 解释路径差异的依据之一
- fallback 判断条件的依据之一

一旦你这样理解它，很多事情就会自然变清楚：

- 为什么这条 path 今天不能走
- 为什么这个 workload 没命中 low-precision fused
- 为什么某个 edge path 必须退回 baseline

## 二、为什么 VLA 比很多系统更需要 capability matrix

普通系统当然也需要能力描述。  
但 `VLA` 特别需要，是因为它的维度更多，而且很多维度并不是简单正交的。

例如一条 path 是否可用，可能同时依赖：

- deployment mode
- precision family
- shape 稳定性
- 是否有 packed weight
- 是否是 expert-heavy workload
- 当前 backend 是否支持

如果这些条件不被集中表达，系统最后很容易退化成：

> 每个模块都知道一点条件，但没有任何一层知道全貌。

这会直接导致：

- planner 很难解释
- benchmark 很难归因
- fallback 很难统一

所以 capability matrix 的价值就在于：

> 把“系统到底能做什么、在什么条件下能做”这件事从零散特判变成显式对象。

## 三、一个最小 capability matrix 至少要表达哪些维度

如果现在要给 `VLA Runtime` 做第一版 capability matrix，我觉得至少要覆盖下面五类信息。

## 四、第一类：Path Identity

首先它必须知道自己在描述谁。

也就是：

- baseline path
- `BF16 fused`
- `low_precision_gate_up_fused`
- `full_low_precision_mlp_moe_fused`
- `fixed_shape_edge_specialized`

这件事看起来基础，但非常关键。

因为如果 path 本身的身份都不稳定，后面所有 capability 描述都会漂。

## 五、第二类：Precision Capability

这层回答的是：

- 支持哪些 precision family
- 是否支持 mixed precision
- 是否要求某种 precision policy 才能启用

这一步会让 planner 不用再到处散写：

- “这里支持 `bf16`”
- “那里支持 `int8`”

而是能够统一判断：

> 当前这条 path 在当前 precision intent 下是否具备执行资格。

## 六、第三类：Shape / Workload Capability

这层回答的是：

- 是否支持动态 shape
- 是否更适合固定 shape
- 对 token 数、expert 分布、batch size 有没有明显约束

这一步对 `VLA` 特别重要。

因为很多路径不是“理论上都能跑”，而是：

- 这类 workload 下才真正值得跑
- 那类 workload 下虽然能跑，但不值得

所以 capability matrix 最好不仅表达“能不能”，还表达“适不适合”。

## 七、第四类：Pack / Layout Capability

这一层是我觉得很多系统会漏掉，但在 `VLA Runtime` 里非常关键的一层。

因为随着：

- fused path
- low-precision
- expert-aware execution

这些能力引入后，一条 path 是否可用，往往强依赖：

- 当前 packed weight 是否存在
- 当前 layout descriptor 是否匹配
- 当前 path 需要的 pack format 有没有准备好

如果 capability matrix 不表达这件事，planner 最后还是得在别处写一堆隐式条件。

## 八、第五类：Fallback Capability

这一层回答的是：

- 当前 path 失败后允许退到哪里
- 是 path fallback 还是 precision fallback
- 哪些前提不满足时应直接退回

这一步为什么重要？

因为 capability matrix 不只是告诉系统“能做什么”，还应该告诉系统：

> 如果不能做，应该怎么合理地退。

也就是说，一个完整的能力描述，不该只描述正向通路，还应该描述回退边界。

## 九、为什么 capability matrix 不该做成“巨大静态配置表”

讲到这里，很容易有人会想：

> 那是不是就做一张很大的 YAML / JSON 表，把所有 backend、所有 path、所有 shape 条件都列进去？

我不太建议第一版这样做。

因为这会很快遇到两个问题：

### 1. 它会变成死表

配置写得很多，但 planner 和 benchmark 不一定真的消费得好。

### 2. 它会失去 runtime 语义

也就是说，表很大，但并没有真正和：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`

这些核心对象形成统一语言。

所以我更建议第一版 capability matrix 先长成：

> 一组能被 planner 查询、能被 benchmark 解释、能被 backend 提供的运行时能力描述对象。

也就是说，它首先应该是 runtime 对象，其次才可能导出成文档表格。

## 十、一个更现实的第一版 capability matrix 草图

如果现在就要给 `VLA Runtime` 定义一个最小 capability matrix，我更推荐它至少能回答下面这些问题：

### 对一条 path：

- 它叫什么
- 它支持哪些 precision family
- 它对 shape/stability 有什么要求
- 它需要哪些 pack/layout 资源
- 它更适合 edge 还是 cloud
- 它失败时允许回退到哪些 path

如果把它写成更接近对象的样子，它大概会像：

```text
PathCapability {
  path_id
  precision_families
  workload_constraints
  pack_requirements
  deployment_preferences
  fallback_targets
}
```

这已经足够支撑第一版 planner 做很多更干净的决策了。

## 十一、为什么 capability matrix 和 benchmark 必须绑在一起

如果 capability matrix 只是 planner 的内部条件表，那它的价值仍然不够完整。

我更希望 benchmark 也能显式引用这层语言。

例如 benchmark 结果最好能告诉你：

- 当前测试命中了哪条 capability
- 某个 workload 为什么没命中目标 path
- 某个 fallback 是 capability 不满足，还是性能策略主动回退

这样，你看到 benchmark 时不再只是看到：

> 这次跑了 xx ms

而是能看到：

> 这次因为 capability X 命中了 path Y，在 precision Z 和 pack format W 下跑出了这个结果。

这才是 runtime 层真正有意义的“可解释 benchmark”。

## 十二、为什么 capability matrix 又和多模型抽象强相关

前一篇文章里，我提到从单模型走向多模型时，最应该先稳定的是：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`
- `PathCapability`
- `Pack/LayoutDescriptor`

现在再回头看，就会发现 capability matrix 正好是把这些对象粘起来的那层胶水。

它的意义在于：

- 不同模型可以有不同 path
- 不同模型可以有不同 pack
- 但 planner 仍然用统一语言去理解“这些 path 能做什么”

也就是说：

> capability matrix 其实是在帮 runtime 把“单模型特判逻辑”提升成“多模型可理解的系统语言”。

## 十三、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的 capability matrix 不该只是文档里的说明表，而应该是一组能被 planner 查询、被 backend 提供、被 benchmark 解释、被 fallback 使用的正式能力描述对象，用来集中表达每条 path 在 precision、shape、pack/layout、deployment mode 和回退层级上的真实能力边界。

因为对这种系统来说，真正重要的不只是：

> 它“理论上支持很多路径”。

而是：

> 它能不能把这些路径的能力、约束和回退条件，用同一种系统语言清楚地表达出来。
