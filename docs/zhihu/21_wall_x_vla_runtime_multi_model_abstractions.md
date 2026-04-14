# 21. VLA Runtime 从单模型走向多模型时，哪些抽象必须先稳定

一路把 `VLA Runtime` 写到这里，下一步很自然就会遇到一个更大的问题：

> 如果这套 runtime 不再只服务 `Wall-X`，而是将来想承接更多 `VLA` 模型，那么哪些抽象必须先稳定？

这个问题如果问得太早，很容易流于空泛；  
但如果问得太晚，又很容易让系统被单模型特化代码锁死。

所以我更愿意在这个时间点来回答它：

> 不是为了现在就做“大一统平台”，而是为了避免当前所有设计都变成一次性 patch。

这篇文章就只讲这个问题。

## 一、先说结论：不是所有抽象都值得先稳定，应该先稳定那些会反复穿过 planner、backend、benchmark 的少数对象

很多系统一谈“多模型抽象”，第一反应都会是：

> 那是不是应该先把所有模块都抽象得非常漂亮？

我现在越来越不认同这种起手方式。

因为 runtime 早期阶段最怕的，就是：

- 抽象做得很多
- 但没有一层真正落到执行路径上

所以如果现在要问“哪些抽象必须先稳定”，我的判断会更聚焦：

> 应该优先稳定那些会同时穿过 `planner`、`kernel backend`、`benchmark` 和 `fallback` 的对象。

因为只有这类对象，才是真正的 runtime 级抽象。  
那些只在某个局部函数里好看一点、但不会穿透整个系统的抽象，优先级其实没那么高。

## 二、为什么单模型阶段最容易把系统做死

单模型阶段最大的诱惑，就是很多东西都可以“先写死”。

例如：

- 输入 shape 写死
- token 类型语义写死
- expert 路径写死
- 输出 adapter 写死
- 某些 backend 约束写死

这在验证阶段当然很高效。  
问题在于，如果这些写死的东西慢慢渗透到整个系统，后面当你真的想加第二个模型时，就会发现：

> 你不是在扩 runtime，而是在重写 runtime。

所以多模型抽象的目的，不是为了炫耀“平台化”，而是为了防止现在的实现把未来路堵死。

## 三、第一批必须先稳定的抽象是什么

如果现在就要给 `VLA Runtime` 选一批“必须先稳定”的对象，我会优先选下面这些。

## 四、第一类：WorkloadDescriptor

这是我觉得最应该先稳定的抽象之一。

它表达的不是具体模型内部张量，而是当前请求对 runtime 来说最重要的工作负载特征，例如：

- 图像尺寸
- prompt 长度
- 状态维度
- action horizon
- batch size
- shape 是否稳定
- 是否是 expert-heavy 路径

为什么它必须先稳定？

因为：

- planner 要用它做决策
- benchmark 要用它分类统计
- backend 要知道当前 workload 属于哪类条件

如果这一层没有统一语言，后面整个 runtime 就很难形成闭环。

## 五、第二类：ExecutionPlan

这也是我觉得必须先稳定的核心对象。

因为 runtime 最终最重要的一件事，其实就是：

> 在当前 workload 下，选择怎样的一条执行路径。

这条路径至少应该能表达：

- deployment mode
- precision policy
- selected path
- fallback chain
- layout / pack 相关约束

为什么它必须稳定？

因为：

- planner 生成它
- backend 消费它
- benchmark 解释它
- fallback 修正它

只要这一层不稳定，后面所有讨论都会散。

## 六、第三类：ExecutionReport

如果只稳定 plan，而不稳定 report，系统还是会缺一条腿。

因为 runtime 后面所有反馈环，最终都得落到：

- 这次到底走了什么 path
- 是否发生 fallback
- latency 怎样
- 哪个 stage 最重
- 当前 precision 和 pack 是否命中

这就需要一份结构清楚的 `ExecutionReport`。

它的价值在于：

> 它把“实际发生了什么”统一成可以被 benchmark、planner 和开发者同时理解的语言。

## 七、第四类：Path Capability 描述

这类抽象很容易被忽视，但其实很关键。

所谓 capability，大致就是表达：

- 某条 path 支持哪些 precision
- 支持哪些 shape
- 需要哪些 packed weight
- 适合什么 deployment mode
- 有哪些已知 fallback 条件

为什么这层必须先稳定？

因为如果没有它，planner 很容易退化成一堆硬编码条件，而 backend 也很难把自己的真实能力说清楚。

这会让系统越来越难扩。

## 八、第五类：Pack / Layout Descriptor

前面讲 `packed weight` 的时候，我已经强调过：

> 它不该只是 kernel 私下知道的存储技巧。

如果想从单模型走向多模型，这一点就更重要。

因为不同模型之间最容易差异化的地方之一，就是：

- 路由方式
- expert 布局
- weight pack
- output layout

这并不意味着你要现在就统一所有物理 layout。  
但至少应该先稳定“如何描述 layout / pack”这件事。

否则后面每来一个新模型，runtime 语义层都会被迫多长一堆特殊分支。

## 九、哪些抽象暂时不必过早稳定

为了避免系统一开始就抽象过度，我也想明确说说哪些东西没必要现在就追求“终局形式”。

## 十、第一类：具体 kernel API 细节

这一层当然重要，但它更适合在路径稳定后逐步收敛。

第一阶段不一定要急着把所有 kernel API 做成跨模型统一的完美接口。

更重要的是先稳定：

- 它们属于哪类 path
- 需要哪些 capability
- 输入输出在 runtime 语义层怎么表达

## 十一、第二类：所有模型的统一输入 schema

这是另一个很容易过度设计的地方。

不同 `VLA` 模型的输入确实可能差异很大。  
第一阶段与其强行统一所有字段，不如先统一更高层的 workload 描述和执行计划语言。

也就是说：

> 不一定要先统一“模型怎么长”，但应该先统一“runtime 怎么理解它们”。

## 十二、第三类：所有 backend 的完全统一实现

这通常也太早。

第一阶段更现实的目标应该是：

- backend 能被统一描述
- planner 能理解 backend capability
- benchmark 能解释 backend 路径差异

这已经足够支撑多模型扩展的第一步。

## 十三、一个更现实的多模型演化顺序

如果把这件事说得更实一点，我会建议单模型走向多模型时按下面这个顺序推进。

### Stage 1：先稳定 runtime 核心对象

也就是：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`
- `PathCapability`
- `Pack/LayoutDescriptor`

### Stage 2：再让第二个模型接入这些对象

这一步的目标不是追求“完全无痛接入”，而是观察：

- 哪些对象真的够稳定
- 哪些对象仍然带着过强的单模型假设

### Stage 3：只修改跨层对象，不先急着改所有局部 API

这样你会更快知道：

> runtime 到底缺的是语言层抽象，还是局部实现层抽象。

### Stage 4：等到第二个模型也能走完整闭环，再收敛具体接口

这时候再去谈更激进的平台化，通常会更扎实。

## 十四、为什么这些抽象必须同时被 benchmark 看见

这一点我想再强调一次。

多模型抽象如果只存在于代码层，而 benchmark 完全看不见，最后很容易沦为形式主义。

真正值钱的抽象应该能让 benchmark 说清楚：

- 哪类 workload 属于哪个模型
- 哪条 plan 被采用
- 哪个 backend capability 被命中
- 哪种 pack/layout 在生效
- 实际报告表现怎样

只有这样，你才不是在“写看起来很通用的代码”，而是在真的让系统逐步获得跨模型能力。

## 十五、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 从单模型走向多模型时，最应该先稳定的不是所有局部 API，而是那些会反复穿过 planner、backend、benchmark 和 fallback 的核心对象：`WorkloadDescriptor`、`ExecutionPlan`、`ExecutionReport`、`PathCapability` 和 `Pack/LayoutDescriptor`。

因为真正决定 runtime 能不能长成“多模型系统”的，从来不是它现在有多少代码，而是：

> 它有没有先把自己最重要的系统语言稳定下来。
