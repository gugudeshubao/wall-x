# 11. VLA Runtime 的 Planner 应该怎样决定 cloud/edge、precision 和 fused path

前面几篇文章里，我已经把 `VLA Runtime` 的几层骨架慢慢搭出来了：

- 为什么 `VLA` 需要不同于 `LLM` 的推理优化栈
- 为什么 runtime 的 `MVP` 应该先从 benchmark 和 fused path 开始
- 为什么第一条 fused 路径很多时候更适合先落在 `MLP MoE`
- 为什么执行图里至少要有 `Input Adapter`、`Planner`、`Kernel Backend`、`Output Adapter`

但只要真的把这几层往系统里放，马上就会遇到一个关键问题：

> 谁来决定当前请求到底该走哪条执行路径？

这件事如果不单独拎出来，最后就会变成一堆散落在调用代码里的 if/else：

- 这里判断是不是端侧
- 那里判断是不是固定 shape
- 另一处再判断要不要开低精度
- 某个 Python 函数里再临时决定要不要走 fused op

短期看这也能跑。  
但长期看，它会直接把 runtime 做散。

所以这篇文章只讲一个模块：

> `Planner`

更具体一点，就是：

> `VLA Runtime` 的 planner 应该怎样决定 `cloud/edge`、`precision` 和 `fused path`？

## 一、先说结论：Planner 不是“调度附属逻辑”，而是 runtime 的决策中心

很多系统在早期会下意识地把 planner 理解成一种可有可无的“辅助层”。

比如觉得：

- 不就是根据设备选个 backend 吗
- 不就是在几条路径里做个选择吗
- 直接写在 Python 调用代码里也行

但对 `VLA Runtime` 来说，我越来越觉得这层不能省。

原因是：

> `VLA` 的执行路径选择，比普通模型推理更依赖上下文条件。

这些条件至少包括：

- 当前目标是 `cloud` 还是 `edge`
- 输入 shape 是否稳定
- 当前 batch 是否恒为 `1`
- 当前模型是不是走 expert-heavy 路径
- 当前请求更在意 latency 还是吞吐
- 当前后端支持哪些 precision
- 当前 fused path 是否满足启用条件

如果这些判断散在各处，系统很快就会出现两个问题：

### 1. 你不知道决策是怎么做出来的

最后你只能看到：

> 这次为什么走了这个 backend？

但你说不清：

- 是因为设备类型
- 是因为 shape
- 是因为 precision fallback
- 还是因为某个 fused kernel 当前不支持

### 2. benchmark 也会失去解释力

如果运行路径是“隐式决定”的，那你很难把 benchmark 结果和具体执行策略对应起来。

于是你看到一组 latency 数据，却不知道它到底对应哪条路径。

所以我的判断是：

> planner 不是附属逻辑，而是 runtime 的决策中心。

## 二、Planner 到底在决定什么

对 `VLA Runtime` 来说，planner 至少要显式决定三类问题。

### 1. 部署模式

也就是：

- 这次请求按 `cloud` 还是 `edge` 思路来跑

这不是简单地问“设备在哪”，而是在决定：

- 优先吞吐还是优先单次延迟
- 优先通用性还是优先固定 shape 特化
- 是否值得启用更激进的 fused path
- 是否值得启用 graph 或局部 `megakernel`

### 2. 精度模式

也就是：

- 跑 `BF16`
- 跑 `FP16`
- 跑某种低精度变体
- 或者回退到高精度安全路径

planner 的作用不只是“选一个 dtype”，而是：

> 选一组和当前后端能力、输入条件、稳定性目标匹配的 precision policy。

### 3. 执行路径

也就是：

- 普通 baseline path
- fused path
- expert-heavy path
- 固定 shape path
- 某个局部特化路径

对 `VLA` 来说，这个维度尤其重要，因为很多路径不是“更快版本”和“更慢版本”的简单关系，而是适用条件完全不同。

## 三、为什么 `cloud/edge` 不能只靠“设备名”判断

这一点非常容易被想当然。

很多人会觉得：

> planner 判断 `cloud/edge`，不就是看当前 GPU 在服务器还是在端侧吗？

但真实情况通常更复杂。

因为 `cloud/edge` 在 runtime 里不该只是一个“设备位置标签”，而应该是一个执行策略标签。

举个例子：

同样是一张 GPU：

- 在离线批量评测时，你可能更关心吞吐
- 在交互式 demo 时，你可能更关心 TTFT
- 在机器人闭环控制时，你更关心 camera-to-action latency 和 jitter

也就是说，`cloud/edge` 更像是在表达：

> 这次请求的优化目标是什么。

所以我更推荐 planner 里显式维护一个类似这样的概念：

- `deployment_mode = cloud | edge`

而不是试图从环境里隐式猜一切。

## 四、Planner 决定 precision 时，真正该看什么

对很多系统来说，precision 最后会变成几个零散开关：

- `--bf16`
- `--fp16`
- `--int8`
- `--fp8`

这在工具层面可以接受，但在 runtime 层面不够。

我更建议 planner 把 precision 看成一种 policy，而不是一个裸数据类型。

例如，它至少应该考虑：

### 1. 当前 backend 是否支持

并不是每条 fused path 都立刻支持所有 precision。

所以 planner 必须知道：

- 哪条 path 支持 `BF16`
- 哪条 path 支持低精度
- 哪条 path 目前只能回退

### 2. 当前请求是否适合

比如：

- 固定 shape 更适合某些低精度预打包路径
- 动态 shape 可能更适合保守回退
- 端侧更可能愿意牺牲一点通用性换速度

### 3. 当前输出风险是否可接受

这点在 `VLA` 里比在纯文本生成里更重要。

因为低精度误差不一定只体现在“文字好不好看”，而可能体现在：

- 动作漂移
- 输出抖动
- 控制稳定性变差

所以 precision policy 最终应该是 planner 明确下发的，而不是各处自己猜测。

## 五、Planner 怎么决定要不要走 fused path

这可能是 planner 最核心的职责之一。

我不赞成把 fused path 做成一种“默认总是开启”的东西。

更现实的做法是，planner 显式判断它是否满足启用条件。

这些条件可能包括：

### 1. shape 条件

- 是否满足固定 shape
- 是否满足某个 tile 友好的尺寸
- 是否满足当前 fused kernel 的输入约束

### 2. 路由条件

- 当前 expert 路径是否真的是热点
- 当前请求是否走到了那条 expert-heavy path

### 3. precision 条件

- 当前 precision policy 是否和 fused path 兼容

### 4. 稳定性条件

- 当前是否在 debug 模式
- 当前是否需要强确定性
- 当前是否需要回退到 baseline path 做比对

也就是说，planner 不该只是“选最快的”，而应该是：

> 在当前约束下，选最合适的。

## 六、一个更现实的 Planner 输出长什么样

如果只说抽象职责，planner 很容易讲空。

所以我更愿意把 planner 的输出想象成一份明确的执行计划。

例如，这份计划里至少应该有：

- `deployment_mode`
- `precision_policy`
- `execution_path`
- `input_shape_class`
- `layout_policy`
- `routing_mode`
- `fallback_policy`

这样后面的 `Kernel Backend` 才不是在猜，而是在执行一个清楚的 plan。

这也是为什么我觉得 planner 最值钱的地方不是“它会做很多复杂判断”，而是：

> 它把系统层面的隐式决策，变成了显式的数据结构。

## 七、为什么 Planner 必须和 benchmark 绑定

如果 planner 不和 benchmark 绑定，它最后很容易退化成一套拍脑袋规则。

真正有意义的 planner，应该能回答：

- 这次为什么走 fused path
- 为什么没有走低精度 path
- 为什么从 edge policy 回退到了 baseline
- 为什么这类 shape 被归类成 fixed-shape friendly

这些解释能力，实际上都需要 benchmark 和 profiling 的反馈。

所以更好的做法通常是：

- benchmark 给出路径表现
- planner 吸收这些表现形成规则
- runtime 再根据规则下发 plan

这就是为什么 planner 不是单纯写几个判断，而是 runtime 里真正承上启下的一层。

## 八、一个最小可行的 Planner 先做到什么程度就够了

第一版 planner 不需要很复杂。

我觉得最小可行版本只要做到下面几件事就已经足够有价值：

### 1. 显式区分 `cloud` 和 `edge`

即便内部规则还很简单，也要先把执行目标显式化。

### 2. 显式区分 baseline path 和 fused path

不要把 fused path 写成某个调用点里的隐式捷径。

### 3. 显式下发 precision policy

哪怕第一版只有：

- `bf16_safe`
- `low_precision_preferred`

这种粗粒度 policy，也比到处散落开关强很多。

### 4. 保留 fallback 机制

planner 一定要允许：

- fused path 回退
- 低精度回退
- edge policy 回退到安全路径

因为 runtime 早期阶段最怕的不是“不够快”，而是“出了问题没法解释也没法退”。

## 九、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的 planner，不该只是一个写满 if/else 的辅助层，而应该是一个显式决定 `cloud/edge`、`precision policy` 和 `fused path` 的执行计划生成器。

只有这层存在，后面的很多事情才真正有可能成立：

- benchmark 结果可以被解释
- fused path 可以被有条件启用
- 低精度不再只是散落开关
- `Kernel Backend` 不再被迫自己猜执行策略

也就是说，planner 这一层看起来不像 kernel 那么“硬核”，但它实际上是 runtime 能不能长成系统的关键节点。
