# 27. VLA Runtime 的 path taxonomy 应该怎样定义

一路把 `VLA Runtime` 讨论到这里，很多关键对象其实已经逐渐出现了：

- `WorkloadDescriptor`
- `ExecutionPlan`
- `ExecutionReport`
- `PathCapability`
- `precision policy`
- `fallback chain`
- `capability matrix`

但只要真正开始把这些东西往系统里放，很快就会遇到一个非常现实的问题：

> 这些“路径”本身，到底该怎么命名、怎么分类、怎么表达？

这个问题看起来像是术语问题，但实际上它直接关系到系统是不是会越来越乱。

因为如果没有一套清楚的 path taxonomy，最后很容易变成：

- 这里叫 baseline
- 那里叫 normal
- 某个地方叫 fused
- 另一个地方叫 fast
- 低精度、edge、graph、fixed-shape、fallback 又混在一起

到最后，系统里到处都在说“某条 path”，但没人真的知道彼此说的是不是同一类东西。

所以这篇文章就只讲一个问题：

> `VLA Runtime` 的 path taxonomy，到底应该怎样定义？

## 一、先说结论：path taxonomy 不是命名美学，而是 runtime 的系统语言

我现在越来越觉得，`path taxonomy` 最重要的价值，不是为了代码看起来更漂亮，而是为了让 runtime 各层真正开始说同一种语言。

如果把这个判断压成一句话，我会说：

> `path taxonomy` 不是一套命名规范，而是 planner、backend、benchmark、fallback 和 capability matrix 之间共享的系统语言。

为什么要说得这么重？

因为如果路径命名和分类本身是混乱的，后面很多东西都会一起变乱：

- planner 很难表达“到底选了什么”
- benchmark 很难表达“到底测了什么”
- fallback 很难表达“到底退到了哪里”
- capability matrix 也很难表达“这条 path 到底属于哪类能力”

所以 taxonomy 这件事，看起来不像 kernel 那么“硬核”，但它其实非常核心。

## 二、为什么 VLA 比很多系统更容易把 path 说乱

如果是普通 `LLM` 服务，路径命名虽然也会复杂，但通常主轴相对集中：

- baseline
- cache-aware path
- 某个 backend path
- quantized path

而 `VLA` 会天然多出更多维度：

- 模态差异
- expert 路径
- `MLP MoE` 与 attention 路径
- precision family
- edge/cloud mode
- fixed-shape 与 dynamic-shape
- graph / local megakernel

这意味着如果没有分类原则，你最后很容易得到一种“命名越来越长，但语义越来越模糊”的系统。

例如一条路径可能会被同时描述成：

- low-precision
- fused
- edge
- fixed-shape
- graph
- expert-heavy

如果这些维度不分层，taxonomy 很快就会崩掉。

## 三、第一原则：不要把所有属性都塞进 path 名字里

这是我觉得最重要的一条。

很多系统一开始会自然倾向于：

> 路径是什么，就全写进名字里。

这样短期看似乎很清楚，但很快你就会得到一堆过长且难维护的名字，例如：

- `edge_low_precision_fixed_shape_fused_graph_path`

这种命名最大的问题不是长，而是：

> 它把“路径的身份”和“路径的属性”混成了一个东西。

我更推荐的做法是：

- path 有自己的身份
- 路径的属性由别的对象表达

这才是 taxonomy 能稳定下来的前提。

## 四、第二原则：先定义 path 的主轴，再定义附加维度

如果现在要给 `VLA Runtime` 的路径分类，我更推荐先按“主轴”去分，而不是先按所有属性去拼。

对我来说，最适合做主轴的通常是：

> 这条路径在执行结构上属于哪一类。

例如，你可以先区分：

- baseline path
- fused path
- specialized path

这三类本身就是不同层级的执行结构。

然后再在这些主轴之上，附加：

- precision
- deployment mode
- shape class
- graph status

这样路径语言就会稳定很多。

## 五、一个更现实的 path taxonomy 分层方式

如果把它写得更具体一点，我更推荐至少分成三层。

## 六、第一层：Execution Family

这层回答的是：

> 这条路径在执行结构上属于哪一类？

例如：

- `baseline`
- `fused`
- `specialized`

这层是最核心的“路径身份”。

为什么先要这层？

因为 planner 和 benchmark 最先需要分清楚的，就是当前系统到底在跑：

- 一个通用保守路径
- 一个已经收紧过的 fused 路径
- 还是一个更强约束下的特化路径

## 七、第二层：Execution Scope

这层回答的是：

> 这条路径作用在整个模型上，还是只作用在某一段局部路径上？

例如：

- `global`
- `local_mlp_moe`
- `local_attention_qkv`
- `local_output_stage`

这层非常重要，因为对 `VLA Runtime` 来说，很多优化本来就不是全局替换，而是局部路径替换。

如果没有这层，系统很容易把：

- “全局 baseline”
- “局部 fused `MLP MoE`”

混成看起来像同一级的东西。

## 八、第三层：Execution Attributes

这层回答的是：

> 这条路径现在带了哪些属性？

例如：

- precision family
- deployment mode
- shape class
- graph enabled / disabled
- local megakernel enabled / disabled

这层不该负责定义“路径身份”，而应该负责描述：

> 在当前上下文下，这条路径带了哪些附加执行条件。

这会让 taxonomy 保持稳定很多。

## 九、一个更清楚的表达例子

如果按这个分层去表达，一条路径就不必再写成一个又长又乱的字符串。

而可以被理解成：

### Family

`fused`

### Scope

`local_mlp_moe`

### Attributes

- `precision = low_precision`
- `deployment = edge`
- `shape = fixed`
- `graph = disabled`
- `megakernel = disabled`

这样一来，你系统里的不同层都更容易消费这条信息：

- planner 更容易选
- benchmark 更容易记录
- fallback 更容易表达
- capability matrix 更容易描述

## 十、为什么 taxonomy 必须和 capability matrix 配合

如果 taxonomy 只负责命名，而 capability matrix 只负责描述能力，两者又互不相通，那它们的价值都会下降。

我更推荐的关系是：

- taxonomy 定义“这条 path 是什么”
- capability matrix 定义“这条 path 能做什么、需要什么”

这两者一旦对齐，系统语言就会清楚很多。

例如 planner 不再只是说：

> 我选了一个 low-precision 路径

而是能更准确地说：

> 我选了 `fused / local_mlp_moe` 这一类 path，并在当前 workload 下给它附加了 `low_precision + edge + fixed_shape` 这些属性，而 capability matrix 告诉我它在这些条件下是可用的。

这才是 runtime 级的表达方式。

## 十一、为什么 taxonomy 也必须和 fallback 对齐

这点也很重要。

fallback 如果没有 taxonomy，最后很容易退化成一堆模糊句子：

- 从 advanced path 退回 safe path
- 从 fast path 退回 normal path

这类说法听起来好像也能懂，但一旦系统复杂一点就不够了。

更好的表达应该是：

- 从 `specialized / local_mlp_moe` 退回到 `fused / local_mlp_moe`
- 再从 `fused / local_mlp_moe` 退回到 `baseline / global`

这时 fallback 才真正和 runtime 其他层说的是同一种语言。

## 十二、第一版 taxonomy 最不该做什么

为了避免过度设计，我也想明确说说 taxonomy 第一版最不该做什么。

### 1. 不要试图把所有可能路径一次性命名完

第一版只要把主要 family、scope 和 attributes 稳住就够了。

### 2. 不要把所有 attribute 都硬写进 path id

这会让系统非常难演化。

### 3. 不要让 taxonomy 脱离 planner/benchmark/fallback 独立存在

如果 taxonomy 最后只是 README 里的命名约定，它的价值会非常有限。

## 十三、一个更现实的第一版 path taxonomy 草图

如果现在就要给 `VLA Runtime` 写一版最小 taxonomy，我会更推荐像下面这样理解：

### Family

- `baseline`
- `fused`
- `specialized`

### Scope

- `global`
- `local_mlp_moe`
- `local_attention`
- `local_output`

### Attributes

- `precision`
- `deployment`
- `shape_class`
- `graph_state`
- `megakernel_state`

这已经足够支撑一版相当清楚的 planner、benchmark 和 fallback 语言了。

## 十四、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的 path taxonomy 不该通过越来越长的路径名字来表达，而应该通过“Execution Family + Execution Scope + Execution Attributes”这三层来稳定系统语言，让 planner、benchmark、capability matrix 和 fallback 都能用同一种方式理解“当前到底是哪条路径”。 

因为对这种系统来说，真正值钱的不是：

> 路径名字起得多好听。

而是：

> 整个 runtime 能不能围绕路径这件事，说一种稳定、可扩展、可被所有层共同消费的语言。
