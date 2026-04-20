# 31. VLA Runtime 的 plan / report / capability 三个对象应该怎样对齐

一路把 `VLA Runtime` 写到这里，有三个对象其实已经反复出现很多次了：

- `ExecutionPlan`
- `ExecutionReport`
- `PathCapability`

单看每一个，它们的职责都已经不难理解：

- `ExecutionPlan` 负责说“准备怎么跑”
- `ExecutionReport` 负责说“实际怎么跑了”
- `PathCapability` 负责说“这条 path 能做什么、需要什么”

但真正关键的问题，其实不在于它们分别定义得漂不漂亮，而在于：

> 这三个对象之间，到底有没有真正对齐？

因为如果没有对齐，系统很容易走向一种很糟糕的状态：

- planner 说的 path 和 backend 理解的不是一回事
- capability 描述的条件，报告里又根本看不到
- benchmark 看到了一些结果，却没法和 plan/capability 对上

这篇文章就只讲这个问题：

> `plan / report / capability` 这三个对象，到底应该怎样对齐？

## 一、先说结论：这三者不该是三个平行对象，而应该是同一条执行语义在不同阶段的三种投影

如果把我对这个问题的判断压缩成一句话，我会说：

> `ExecutionPlan`、`ExecutionReport` 和 `PathCapability` 不该被理解成三个独立并列的对象，而应该被理解成同一条执行语义在“事前承诺”、“静态边界”和“事后事实”三个阶段的不同投影。

这句话里最重要的词是：

- 同一条执行语义

也就是说，它们三者真正值钱的地方，不是各自信息很多，而是：

> 它们说的是同一件事。

如果这一点做不到，那么系统虽然表面上对象很多，实际上语言还是散的。

## 二、为什么这三者很容易在系统里慢慢漂开

这件事其实非常普遍。

很多 runtime 一开始都会很自然地这样长：

### `ExecutionPlan`

一开始是 planner 输出的一组决策。

### `PathCapability`

后来是 backend 或配置层加上的一组能力描述。

### `ExecutionReport`

再后来是 benchmark / logging 层为了观测执行而加上的报告对象。

这样长出来的问题就在于：

> 这三个对象并不是从同一个统一语言设计出来的，而是从不同需求各自长出来的。

于是系统很快就会出现这些典型症状：

### 1. plan 说得太抽象

例如只说：

- “走 fused path”

但它没说清楚到底是哪一类 fused path。

### 2. capability 说得太静态

例如它知道：

- 某 path 支持 low-precision

但它没和 planner 当前的 precision intent、shape 条件和 deployment mode 真正对上。

### 3. report 说得太结果导向

例如它只能告诉你：

- 这次跑了某个 path

却没法解释：

- 这是否符合原计划
- 为什么没命中原计划
- capability 到底在哪一步没满足

所以“对象都在”不等于“对象已经对齐”。

## 三、如果要真正对齐，第一步应该统一什么

我现在越来越觉得，要对齐这三个对象，第一步不是去强行合并它们，而是：

> 先统一它们共同引用的最小执行语言。

这套最小执行语言至少应该包含：

- path identity
- execution family
- execution scope
- precision family
- deployment mode
- shape class
- pack/layout descriptor
- fallback level

只有当三者都说同一套这些基本词汇时，对齐才真正开始有意义。

否则你表面上是在谈同一条 path，实际上每层说的是不同概念。

## 四、plan / capability / report 各自最应该表达什么

如果先不谈它们之间的映射，只谈各自最该承担的责任，我会这样理解。

## 五、ExecutionPlan：表达“意图”

它最该回答的是：

- 当前想走哪条 path
- 在什么 deployment mode 下走
- 目标 precision policy 是什么
- fallback chain 是什么

这层的关键词是：

> 意图

它不应该伪装成“已经发生的事实”，也不应该替 capability 说边界。

## 六、PathCapability：表达“边界”

它最该回答的是：

- 这条 path 支持什么
- 需要什么
- 不支持什么
- 哪些条件下该 fallback

这层的关键词是：

> 边界

它不应该冒充 planner，也不应该代替 report 记录结果。

## 七、ExecutionReport：表达“事实”

它最该回答的是：

- 这次最终命中了什么
- 哪些 fallback 真的发生了
- 各个 stage 真实花了多少时间
- 哪个 pack/layout 实际生效了

这层的关键词是：

> 事实

它不应该反过来“重新定义”计划或能力，而应该是对实际执行的结构化描述。

## 八、真正的对齐点应该长在哪里

一旦把三者各自的职责说清楚，接下来真正关键的问题就是：

> 它们应该在哪些字段上显式对齐？

我觉得至少有五个对齐点必须存在。

## 九、第一个对齐点：Path Identity

这是最基础的一条。

`plan` 里说的 path、`capability` 里描述的 path、`report` 里记录的 path，必须能明确映射到同一身份。

如果这一步都不统一，后面所有分析都会漂。

## 十、第二个对齐点：Execution Attributes

也就是：

- precision family
- deployment mode
- shape class
- graph state
- megakernel state

这些属性如果在三者里使用不同表达方式，系统就很快会出现一种很难处理的错位：

> planner 以为自己选的是一条 path 的某种属性组合，而 report 记录的却是另一套叫法。

所以这些属性必须对齐成一套正式语言。

## 十一、第三个对齐点：Pack / Layout 约束

这点对 `VLA Runtime` 特别重要。

因为很多 path 是否能命中，本来就很依赖：

- packed weight 是否存在
- 当前 layout 是否匹配

所以：

- `capability` 应该描述需要什么 pack/layout
- `plan` 应该说明本次打算命中什么 pack/layout
- `report` 应该说明最终实际用了什么 pack/layout

这就是非常典型的三方对齐场景。

## 十二、第四个对齐点：Fallback 语义

如果 fallback 只存在于 plan，或者只存在于 report，系统都会很难解释。

更好的结构应该是：

- `capability` 说明这条 path 在哪些条件下允许回退
- `plan` 说明当前配置的 fallback chain
- `report` 记录这次到底发生了哪些 fallback

只有这样，你后面看到一条 benchmark 结果时，才能真正回答：

> 是计划本来就保守，还是能力不满足，还是实际运行中触发了回退。

## 十三、第五个对齐点：阶段级观测

这点通常最容易被忽视。

很多人会觉得：

- plan 和 capability 主要管路径
- report 主要管时间

但实际上，path 是否真的成立，常常要靠阶段级观测来解释。

例如：

- 某条 path 理论支持 low-precision
- plan 也想命中 low-precision fused
- report 里却显示 scatter 阶段异常抬高

这时如果阶段级 breakdown 语言在三者之间完全断裂，系统就很难真正“对齐解释”。

所以即使 plan 和 capability 不直接存所有 stage 时间，它们也至少要和 report 使用同一套 stage 命名语言。

## 十四、一个更现实的对齐方式

如果现在就要给这三者设计一版更现实的对齐方式，我更推荐下面这种思路。

## 十五、第一步：共享同一套 PathDescriptor

不要让三者各自发明自己理解的 path 表达。

最好先有一层共享的：

- `PathDescriptor`

至少统一：

- family
- scope
- attributes

这层一旦稳定，三者很多对齐问题都会轻很多。

## 十六、第二步：让 capability 生成“可选空间”

也就是说：

- `capability` 不直接替 planner 选路
- 它只描述哪些 path/attribute 组合在当前是可行的

这样 planner 的角色就会清楚很多。

## 十七、第三步：让 plan 从 capability 约束中选出一个意图

也就是：

- 在可行空间里选一个主路径
- 配一个 fallback chain

这样 `plan` 就是：

> 在能力边界内做出的执行意图。

## 十八、第四步：让 report 记录“意图与事实之间的差”

这是我觉得最重要的一步。

report 的价值，不只是记“最后发生了什么”，更是记录：

> 计划和现实之间，到底差了什么。

这一步如果做好，后面 benchmark 和 planner 修正都会变得非常自然。

## 十九、为什么这件事对 benchmark 特别重要

如果这三者没有对齐，benchmark 最后很容易只剩两种状态：

### 状态 A

它只能看到结果，解释不了原因。

### 状态 B

它记录了很多上下文，但这些上下文和 runtime 的真实对象对不上。

而真正高质量的 benchmark，应该能够自然说出这种话：

> 在 workload X 下，planner 原计划命中 path Y，capability 显示该路径在当前 pack/layout 下可行，但 runtime 实际因为 fallback 条件 Z 退回到 path W，于是最终报告里 core compute 时间下降了，但 total path 收益没有预期高。

这类解释能力，本质上就是三者对齐之后的产物。

## 二十、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `ExecutionPlan`、`ExecutionReport` 和 `PathCapability` 不该被当成三个各写各的对象，而应该围绕同一套 path identity、execution attributes、pack/layout 约束、fallback 语义和 stage 语言来对齐：capability 负责描述边界，plan 负责在边界内表达意图，report 负责记录意图与现实之间的差。

因为对 `VLA Runtime` 来说，真正决定系统是否成熟的，不只是对象有没有，而是：

> 这些对象之间，是否真的在说同一种运行时语言。
