# 15. VLA Runtime 的 planner 与 benchmark 如何闭环

前面几篇文章里，我把 `planner` 和 `benchmark` 分开讲了：

- `planner` 决定 `cloud/edge`、`precision` 和 `fused path`
- `benchmark` 决定系统到底慢在哪、值不值得优化哪里

但如果继续往下做 runtime，很快就会发现：

> 真正关键的不是它们分别存在，而是它们能不能形成闭环。

因为如果这两者是断开的，系统最后通常会走向两个极端之一：

### 极端 A

benchmark 很全，但 planner 还是靠拍脑袋规则做决策。

### 极端 B

planner 很复杂，但根本没有可靠 benchmark 来约束它。

这两种情况最后都会让 runtime 失去方向。

所以这篇文章只讲一件事：

> `VLA Runtime` 的 planner 和 benchmark，应该怎样形成真正的闭环？

## 一、先说结论：benchmark 不是“测完就结束”，planner 也不是“写完规则就结束”

如果把系统说得更直白一点，我现在越来越相信下面这个判断：

> benchmark 决定 planner 应该学什么，planner 决定 benchmark 应该继续验证什么。

也就是说，它们不是前后关系，而是循环关系。

如果这个循环不存在，后果通常很直接：

- benchmark 只剩汇报意义
- planner 只剩配置意义
- runtime 没有真正的自我校正能力

所以比起问“benchmark 怎么做”或者“planner 怎么设计”，更值得问的是：

> 这两者怎么形成反馈回路？

## 二、为什么闭环在 VLA 里特别重要

如果是普通 `LLM` 服务，很多执行决策相对已经有成熟套路：

- 哪种 attention backend
- 哪种 cache 策略
- 哪种 batching 方式

虽然也要 benchmark，但整体经验相对成熟。

而 `VLA` 不一样。

它面对的是一套更不稳定、更依赖场景的系统：

- 输入模态更多
- batch=1 更常见
- 控制链路对 jitter 更敏感
- edge/cloud 差异更大
- fused path 和低精度收益更依赖实际 workload

在这种前提下，planner 如果没有 benchmark 反馈，很容易变成：

> 看起来很聪明，但其实只是在写死几条未经验证的规则。

反过来，benchmark 如果不反哺 planner，也很容易变成：

> 数据很多，但系统行为并没有真的因此变好。

所以闭环在 `VLA` 里比在很多成熟场景里都更重要。

## 三、闭环的最小形态应该是什么

我不建议一上来就把这个闭环理解成某种自动学习系统。  
第一版其实完全可以很朴素。

我更推荐的最小闭环是：

1. benchmark 记录关键 workload 的真实表现  
2. planner 根据这些表现选择执行策略  
3. runtime 执行后继续记录结果  
4. benchmark 再验证 planner 的选择是否真的有效

这听起来简单，但它已经比“完全手工经验决策”强很多。

## 四、benchmark 应该给 planner 提供什么样的信息

如果 benchmark 只是输出一行：

> latency = xx ms

那 planner 几乎用不上。

真正能服务 planner 的 benchmark，输出应该更结构化。

至少应该包含：

### 1. workload profile

例如：

- 图像尺寸
- prompt 长度
- 状态维度
- action horizon
- batch size
- 是否固定 shape

### 2. stage breakdown

例如：

- input adapter
- routing/layout
- core forward
- output adapter

### 3. path-specific results

例如：

- baseline path 的表现
- fused path 的表现
- 高精度和低精度的表现
- edge policy 和 cloud policy 的表现

### 4. stability metrics

例如：

- P50 / P90 / P99
- jitter
- fallback 次数
- 长时间 steady-state 漂移

只有这种结构化输出，planner 才有可能真正用起来。

## 五、planner 真正应该从 benchmark 学到什么

planner 不是来“背 benchmark 结果”的，而是要从 benchmark 中提炼出可执行规则。

我觉得第一版 planner 至少该学到三类东西。

### 1. 哪些 workload 更适合哪条 path

比如：

- 固定 shape + batch=1 + expert-heavy 更适合 fused path
- 动态 shape + 调试场景更适合 baseline

### 2. 哪些 precision policy 在哪些场景下更稳

比如：

- 某种 mixed precision 在 cloud 下收益明显
- 某种 aggressive policy 在 edge 控制场景 jitter 太大

### 3. 哪些 fallback 是常见且必要的

比如：

- fused path 不满足 shape 条件时退回 baseline
- 低精度 path 在某些动作路径上退回高精度

也就是说，planner 不是“死记结果”，而是把 benchmark 提炼成 runtime 规则。

## 六、这个闭环具体怎么跑

如果把它写成一条最小执行流程，我更推荐像下面这样理解：

### 第一步：先定义基准 workload 集

不要等系统很大了再临时找样例。  
应该先明确：

- edge workload
- cloud workload
- 固定 shape workload
- 动态 shape workload
- 语言为主路径
- 动作为主路径

这样后面的 benchmark 和 planner 才有共同语境。

### 第二步：benchmark 先跑 baseline

这一轮的目标不是赢，而是看清：

- 系统最原始的行为
- 哪些 stage 最重
- 哪些 path 值得成为候选优化目标

### 第三步：planner 产出第一版规则

第一版完全可以很朴素，例如：

- 若 `edge && batch=1 && fixed_shape`，优先某种 fused path
- 若 `safe_precision && debug_mode`，优先 baseline
- 若 `expert_heavy && mlp_moe_hot`，优先某个 expert-aware path

### 第四步：benchmark 再验证 planner 规则

这一轮就不只是测“快不快”，而是测：

- planner 的选择是否真的比 baseline 更优
- 哪些 workload 被误判
- 哪些路径虽然均值更快，但 jitter 更差

### 第五步：用结果修 planner

这一步才是闭环真正成立的地方。

如果验证发现：

- 某些规则不成立
- 某些 fallback 频率太高
- 某些低精度 policy 虽快但不稳

那么 planner 就要被修，而不是让 benchmark 报告留在文档里。

## 七、为什么这个闭环不该一开始就自动化过度

讲闭环很容易让人想到：

> 那是不是要做自动学习、自动搜索、自动调参？

长期看也许值得，但第一版我不建议走这么远。

因为对早期 runtime 来说，最大的瓶颈通常不是“不会自动学”，而是：

- 还没有足够干净的 benchmark
- 还没有足够明确的 path 定义
- 还没有足够稳定的 fallback 语义

所以第一版更现实的闭环通常是：

- 人工定义 workload 集
- benchmark 产出结构化结果
- planner 用显式规则消费结果
- 人工或半自动修正规则

这已经足够让系统从“拍脑袋”进化到“有依据地迭代”。

## 八、planner 与 benchmark 闭环最容易失败在哪里

我觉得最常见的失败点有三个。

### 1. benchmark 输出太粗

结果就是 planner 根本学不到东西。

### 2. planner 决策不可解释

结果就是你不知道它为什么这样选，也就没法用 benchmark 修它。

### 3. fallback 不被记录

结果就是 benchmark 看起来像在测某条 path，实际上运行中可能已经大量退回别的 path。

这三件事如果处理不好，闭环基本建不起来。

## 九、一个更现实的最小闭环接口

如果把这件事再具体一点，我会希望 runtime 至少有下面三类可观测对象：

### 1. `WorkloadDescriptor`

表达当前请求是什么样的 workload。

### 2. `ExecutionPlan`

表达 planner 最后选择了什么路径、什么 precision、什么 fallback policy。

### 3. `ExecutionReport`

表达这次真实跑出来的结果：

- latency
- stage breakdown
- fallback 是否发生
- memory/jitter 情况

这样 benchmark 和 planner 的闭环就不再只是口头概念，而会变成三个明确的 runtime 对象。

## 十、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 里的 benchmark 和 planner，不该是彼此独立的两个模块，而应该通过“workload 描述 -> 执行计划 -> 执行报告 -> 规则修正”形成最小闭环。

只有这个闭环存在，后面的很多事情才会真正开始变得有秩序：

- fused path 不是乱开
- 低精度不是乱试
- edge/cloud 不是拍脑袋分流
- benchmark 也不再只是汇报材料

对 `VLA Runtime` 来说，这种闭环能力本身，就是系统走向成熟的标志之一。
