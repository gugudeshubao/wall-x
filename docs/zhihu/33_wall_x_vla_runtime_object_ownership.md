# 33. VLA Runtime 的 benchmark、report、capability 到底该由谁持有

前面几篇文章里，我其实已经把几个 runtime 核心对象逐步讲出来了：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`
- `PathCapability`
- benchmark 这层聚合逻辑

同时也一直在强调一个判断：

> 这些对象必须说同一种语言，否则系统会越来越散。

但只要真的开始往实现落，马上又会出现另一个非常现实的问题：

> 这些对象到底该由谁持有？

这个问题之所以关键，是因为很多系统不是抽象没有，而是：

> 抽象都在，但归属关系是乱的。

比如你很容易看到这样的情况：

- benchmark 在一套独立工具链里持有自己的 workload 语义
- backend 在内部维护一套 capability 描述
- planner 在别处偷偷派生一版 path 选择规则
- report 又在 logging 层单独长一套结构

最后系统看起来什么都有，但实际上：

> 没有任何一层真正对这些对象负全责。

这篇文章就只讲这个问题：

> `VLA Runtime` 里的 benchmark、report、capability，这三个对象到底该由谁持有？

## 一、先说结论：不要按“谁最方便写”来决定对象归属，而要按“谁对它的语义最负责”来决定

这是我最想先说清楚的一点。

很多系统早期分对象归属时，往往会按一个很顺手但很危险的逻辑：

> 这个对象谁写起来最方便，就先放谁那。

例如：

- capability 放 backend 最方便
- report 放 logging 层最方便
- benchmark 放独立脚本最方便

短期看这完全可以跑。  
但长期看，它往往会让对象语义越来越漂。

所以我现在更认可的原则是：

> 这些对象的归属，不应该按“谁最方便写”，而应该按“谁对它的语义最负责”。

这句话其实就是在回答：

- 谁最知道这个对象应该长什么样
- 谁应该对它的一致性负责
- 谁最不应该让它在不同地方长出不同版本

只要抓住这一点，很多边界就会自然清楚很多。

## 二、为什么对象归属在 VLA Runtime 里特别重要

如果是简单推理脚本，这种归属问题很多时候不会立刻变成痛点。  
但 `VLA Runtime` 一旦开始有：

- planner
- backend
- benchmark
- fallback
- edge/cloud 分流

这些层之后，对象归属就不再只是代码整洁问题，而会直接影响：

- 语义一致性
- benchmark 解释力
- planner 可维护性
- 多模型扩展能力

也就是说，这已经是系统设计问题了。

## 三、先把问题分开：这三个对象并不应该有同一个 owner

这一点很重要。

如果只说“这些对象都很重要”，很容易让人下意识觉得：

> 那是不是应该都放在 runtime 顶层某个统一 manager 里？

我不太建议这么理解。

因为这三者虽然关联很强，但职责其实不同：

- `benchmark` 更像聚合与解释层
- `ExecutionReport` 更像执行事实层
- `PathCapability` 更像能力边界层

所以真正合理的做法通常不是：

> 找一个超级 owner 把它们都塞进去

而是：

> 分别找对各自语义最负责的 owner，同时保证它们通过统一语言对齐。

## 四、谁该持有 PathCapability

这个我觉得相对最清楚。

如果按“谁对语义最负责”的原则，我更倾向于：

> `PathCapability` 应该由 runtime 的 backend registry / path registry 这一层持有，而不是由 planner 或 benchmark 自己复制一份。

为什么？

因为 capability 说的本质是：

- 某条 path 支持什么
- 需要什么
- 不支持什么
- 适合什么 deployment mode / precision / shape class

这些信息最接近：

> backend 对自己能力边界的正式声明。

所以如果把它放到 planner 里，问题会很快出现：

- planner 开始变成“既做决策，又维护能力事实”

这会把职责搅乱。

如果把它放到 benchmark 里，也会有类似问题：

- benchmark 开始在“解释结果”的同时偷偷定义能力边界

所以 capability 最合适的 owner，通常还是：

> path/backend 这一层的注册与能力声明系统。

planner 应该消费它，而不是持有它。

## 五、谁该持有 ExecutionReport

这个问题比 capability 稍微更容易混乱一点。

因为大家很容易下意识觉得：

> 既然 report 最后是拿来看的，那应该由 logging/benchmark 之类的层来持有。

我现在不太认可这种归属。

我更倾向于：

> `ExecutionReport` 应该由 runtime 的 execution layer 自己生成并持有其原始事实版本，而 benchmark 和 logging 只消费它。

原因很简单。

因为 `ExecutionReport` 说的不是“分析结果”，而是：

> 这次执行到底发生了什么。

例如：

- planner 原计划是什么
- 实际命中了哪条 path
- 是否发生 fallback
- 各 stage 实际花了多少时间

这本质上是执行事实，而不是后处理观点。

所以如果把它的原始版本交给 benchmark 层来定义，系统很快就会走向一种风险：

> report 的结构开始被“方便统计”主导，而不是被“真实表达执行事实”主导。

这会让 report 的 runtime 价值迅速下降。

所以我更推荐：

- runtime execution layer 负责生成原始 `ExecutionReport`
- benchmark / logging / UI 层负责消费、聚合、展示

## 六、谁该持有 benchmark

这个问题最容易被讲得模糊。

我会更明确一点：

> benchmark 不应该被某一个 path/backend 模块私有持有，而应该被 runtime 的评估层或 profiling/analysis 层持有。

但这里要注意区分两件事：

### 1. benchmark 依赖的原始事实

这些不该由 benchmark 自己发明，而应该来自：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`

### 2. benchmark 的聚合逻辑

这部分才属于 benchmark 自己。

例如：

- P50/P90/P99
- 按 workload 分类聚合
- 对 path 命中率做统计
- 对 fallback 做汇总

也就是说，我更建议 benchmark 被理解成：

> 一层消费 runtime 核心对象、并对它们做结构化聚合和解释的评估系统。

而不是一层自己定义事实的系统。

## 七、这三个对象的归属关系应该怎么配合

如果把前面的判断合起来，我更推荐这样理解它们的所有权：

### PathCapability

由 backend/path registry 持有并声明。

### ExecutionReport

由 runtime execution layer 生成并持有原始事实版本。

### benchmark

由 profiling / analysis 层消费前两者以及 plan/workload，做聚合和解释。

如果用更压缩的话来说，就是：

- capability 属于“能力边界”
- report 属于“执行事实”
- benchmark 属于“评估解释”

这三者一旦这样分清，系统会清爽很多。

## 八、为什么 planner 不应该偷偷持有这三者的“私有版本”

这一点我特别想单独说。

很多系统最后会变得混乱，不是因为没有抽象，而是因为 planner 常常会慢慢开始自己维护一套私有理解：

- 自己记一份 capability
- 自己拼一份 report-like 状态
- 自己做一份 benchmark-like 命中计数

这短期很方便，但长期代价很高。

因为 planner 的职责本来应该是：

> 基于统一对象做决策。

而不是：

> 逐渐长成“另一个私有 runtime 世界”。

所以我会更坚持一个原则：

> planner 应该是这些对象的消费者和转换者，而不该变成它们的第二持有者。

## 九、一个更现实的对象流转方式

如果把这件事写得更像系统里的真实流转，我更推荐下面这种关系。

### Step 1

backend/path registry 暴露 `PathCapability`

### Step 2

planner 读取 capability 和 workload，生成 `ExecutionPlan`

### Step 3

execution layer 执行 `ExecutionPlan`，产出 `ExecutionReport`

### Step 4

benchmark / profiling 层消费：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`
- `PathCapability`

做统计和解释

这样四层职责会非常清楚：

- capability 给 planner 约束空间
- plan 给 execution 定执行意图
- report 给 benchmark 提供事实
- benchmark 再反过来修 planner

## 十、为什么这件事和多模型扩展强相关

这也是我觉得对象归属必须早点想清楚的原因。

因为一旦 runtime 从单模型走向多模型，如果这些对象的 owner 还是混乱的，你后面几乎一定会遇到下面的问题：

- 新模型自带一套 capability 说法
- 旧模型的 report 格式和新模型不一致
- benchmark 需要为每个模型单独写特殊解析

最后系统虽然“也能扩”，但其实是在堆特判。

反过来，如果对象归属一开始就比较清楚，那么多模型扩展时更像是在：

> 往统一对象系统里增加新的 path 和新的 workload，而不是往系统里塞新的私有语言。

## 十一、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> 在 `VLA Runtime` 里，`PathCapability` 最适合由 backend/path registry 持有，`ExecutionReport` 最适合由 execution layer 生成并持有原始事实版本，而 benchmark 最适合作为一层消费这些统一对象并做聚合解释的评估系统；真正危险的不是对象少，而是这些对象各自被方便地放错地方，最后谁都对语义一致性不负责。

因为对这种系统来说，最关键的从来不是：

> 这些对象有没有。

而是：

> 它们到底由谁负责定义、由谁负责生产、又由谁负责消费。
