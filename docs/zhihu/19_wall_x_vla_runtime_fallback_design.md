# 19. VLA Runtime 的 fallback 机制应该怎样设计

只要开始认真做 `VLA Runtime`，很快就会发现一个事实：

> 你不可能一开始就让所有路径都永远稳定命中。

原因很现实。

随着系统继续往下长，你会越来越多地引入这些能力：

- fused path
- low-precision path
- fixed-shape path
- edge/cloud 差异化 path
- 后续可能还有 graph 或局部 `megakernel`

这些能力当然值得做，但它们也天然意味着一件事：

> runtime 必须学会“在某条路径不适合或不可用时，怎样体面地退回去”。

这就是为什么我越来越觉得，`fallback` 不是附属逻辑，而是 `VLA Runtime` 的核心能力之一。

这篇文章就只讲这件事：

> `VLA Runtime` 的 fallback 机制应该怎样设计？

## 一、先说结论：fallback 不是异常处理，而是执行计划的一部分

很多系统早期都会把 fallback 写成一种比较临时的东西：

- 这个 backend 报错了，try 一下另一个
- 某个条件不满足了，临时改个分支
- 某个 low-precision 路径看起来不稳，就在某处 if/else 硬切回来

这在验证阶段能跑，但很快就会暴露问题。

因为这种 fallback 没有被正式建模，于是你会遇到：

- planner 不知道自己到底规划了什么
- benchmark 不知道实际跑的是哪条路径
- backend 不知道自己是主路径还是兜底路径
- 系统行为越来越不可解释

所以我现在的判断很明确：

> `fallback` 不应该被当成异常处理，而应该被当成执行计划的一部分。

也就是说，在 runtime 里，它不该是“出了事再想办法”，而应该是：

> planner 在生成执行计划时，就显式写进去的回退层级。

## 二、为什么 VLA 比很多普通推理系统更依赖清晰 fallback

如果只是普通 `LLM` 生成服务，fallback 当然也重要。  
但 `VLA` 对它的要求通常更高，原因主要有三个。

### 1. 路径更多

`VLA` 往往同时存在：

- baseline path
- fused path
- expert-heavy path
- low-precision path
- edge/cloud 差异化 path

路径越多，fallback 如果不清楚，系统越容易变得不可控。

### 2. 输出语义更敏感

在纯文本场景里，某次 fallback 也许主要影响吞吐或 TTFT。  
但在 `VLA` 里，它可能影响：

- camera-to-action latency
- jitter
- 控制稳定性

所以 fallback 不只是为了“不要挂掉”，还要尽量保证系统语义可接受。

### 3. benchmark 必须解释得清楚

如果 benchmark 最后测到的是一条“表面上启用了 fused path，实际上大量退回 baseline”的执行链，而系统自己又说不清楚，那很多优化判断都会失真。

所以 `VLA Runtime` 里的 fallback 必须比很多普通脚本式系统更清晰。

## 三、fallback 到底在退什么

如果把这个问题说清楚，设计就会简单很多。

我觉得 fallback 至少要分成三类，而不是混成一个概念。

### 1. Path fallback

也就是：

- 从 fused path 退回 baseline path
- 从特化 path 退回通用 path

这是最常见的回退。

### 2. Precision fallback

也就是：

- 从 low-precision path 退回高精度 fused path
- 再从高精度 fused path 退回 baseline

这类 fallback 对 `VLA` 特别重要，因为低精度不一定总该被强行命中。

### 3. Mode fallback

也就是：

- 从 edge-oriented policy 退回更保守的通用 policy
- 从固定 shape path 退回动态 shape 兼容路径

这类 fallback 更接近 runtime 级别的策略切换。

一旦把这三类区分开，很多逻辑都会自然清晰很多。

## 四、一个现实的 fallback 层级长什么样

如果只讲抽象原则，还是容易空。

所以我更喜欢把第一版 fallback 想象成一个很明确的层级。

例如：

### 第一层

首选路径：

- 当前 planner 认为在这次请求下“最合适”的路径

### 第二层

同语义但更保守的备选路径：

- 例如从 low-precision fused path 退回 high-precision fused path

### 第三层

通用 baseline：

- 不一定最快，但最容易保证可执行和可解释

如果再往后，你甚至可以有：

### 第四层

调试/安全模式：

- 明确以稳定和可解释为先，而不是性能为先

这套层级很重要，因为它让 runtime 始终知道：

> 自己不是在“随便换一条路”，而是在按预先定义的层级回退。

## 五、Planner 应该怎样表达 fallback

我不建议把 fallback 只写成 backend 内部的临时行为。  
更合理的做法通常是：

> Planner 在生成 `ExecutionPlan` 时，就把 fallback chain 写进去。

也就是说，一个执行计划里不该只有：

- `primary_path`

还应该有：

- `fallback_path_1`
- `fallback_path_2`
- 对应的 precision/mode 变化

例如你可以抽象成：

```text
primary: low_precision_fused
fallback_1: high_precision_fused
fallback_2: baseline
```

这样后面的 `Kernel Backend` 就不是在“自己想办法”，而是在执行 planner 已经定义好的层级。

## 六、什么情况下应该触发 fallback

这件事也不能模糊。

我更建议第一版先明确三类触发条件。

## 七、第一类：硬条件不满足

例如：

- shape 不匹配
- 当前 backend 不支持
- 当前 packed weight 不存在

这类 fallback 最直接，也最应该自动发生。

## 八、第二类：运行时能力不满足

例如：

- 某条 fused path 当前初始化失败
- graph path 当前未命中
- 某种低精度后端资源未就绪

这类情况也适合自动 fallback，但必须被记录进执行报告。

## 九、第三类：策略上不应继续激进

例如：

- 当前是 debug/safe 模式
- 当前 edge policy 更在意 jitter
- 当前请求的输出语义更敏感

这类 fallback 更像 planner 主动做的保守选择，而不是 runtime 事后补救。

## 十、为什么 fallback 必须对 benchmark 可见

这是我觉得最不能退让的一条。

如果 fallback 对 benchmark 不可见，那么系统会很快进入一种危险状态：

> 你以为自己在测某条先进路径，实际上运行时大部分请求都悄悄退回了 baseline。

结果就是：

- benchmark 失真
- planner 学到错误规则
- 路径收益被夸大或被误读

所以 runtime 至少应该在执行报告里明确记录：

- 是否发生 fallback
- 从哪条路径退到了哪条路径
- 是 path fallback、precision fallback 还是 mode fallback

只有这样，benchmark 才能真正服务系统迭代。

## 十一、fallback 不该一开始就做得过度复杂

虽然我一直强调 fallback 很重要，但第一版也没必要把它做成一个巨大的状态机。

更现实的做法通常是：

### 1. 先定义清楚层级

例如：

- `low_precision_fused -> high_precision_fused -> baseline`

### 2. 先把触发条件分清楚

例如：

- 硬条件不满足
- backend 当前不可用
- planner 主动保守回退

### 3. 先保证报告完整

即使规则还不够聪明，也先让系统能说清楚：

> 它为什么退、退到了哪里、代价是什么。

这往往比一上来把 fallback 写得非常花哨更有价值。

## 十二、一个更现实的第一版 fallback 语义

如果把第一版 runtime 的 fallback 压缩成最小语义，我更推荐像下面这样：

### 对 planner

planner 负责：

- 定义主路径
- 定义回退层级
- 定义回退触发规则类别

### 对 backend

backend 负责：

- 按 plan 尝试执行
- 在需要时按层级回退
- 生成可解释的执行报告

### 对 benchmark

benchmark 负责：

- 把 fallback 作为正式观察对象
- 区分“命中主路径”和“实际回退执行”的表现差异

这样，fallback 才会真正成为 runtime 的一部分，而不是一种掩盖问题的副作用。

## 十三、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的 fallback 机制，不该被写成零散的异常处理，而应该被设计成由 planner 显式定义、backend 按层级执行、benchmark 可完整观察的执行计划组成部分。

因为对这种系统来说，真正重要的从来不只是：

> 某条先进路径能不能跑起来。

而是：

> 当它跑不起来、或者不该继续跑的时候，系统能不能以一种可解释、可度量、可继续优化的方式退回去。
