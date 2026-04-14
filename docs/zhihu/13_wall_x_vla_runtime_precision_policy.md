# 13. VLA Runtime 的 precision policy 应该怎样设计

前面几篇文章里，我已经反复强调过一个判断：

> 低精度不是一个最后再打开的开关，而应该被当成 `VLA Runtime` 的后端能力来设计。

但如果只讲到这里，还是容易让人觉得有点抽象。  
因为真正进到系统实现里，马上就会碰到一个更具体的问题：

> 既然低精度不是一个裸开关，那 `precision policy` 到底应该长什么样？

这篇文章就只讲这个问题。

更准确地说，我想回答的是：

> `VLA Runtime` 里的 precision，不应该只是 `bf16 / fp16 / int8` 这种散落参数，而应该怎样被组织成一套真正可执行、可回退、可和 planner 协作的 policy？

## 一、先说结论：precision policy 不是数据类型表，而是一组执行规则

很多系统谈 precision 时，最容易掉进一个误区：

> 把 precision 理解成“当前用什么 dtype”。

这当然不完全错，但对 runtime 来说远远不够。

因为一旦你真的开始做：

- fused path
- 低精度权重
- 不同 backend
- 端侧/云侧差异化路径

你就会发现，precision 决定的从来不只是一个数据类型，而是整组执行规则：

- 哪个模块能用低精度
- 哪个模块必须保持更高精度
- 某条 fused path 是否支持当前精度
- 某种低精度格式是否要求特定 weight pack
- 当前请求失败时应该怎样回退

所以我的判断是：

> precision policy 本质上是一份 runtime 执行规则，而不只是一个 dtype 选择。

## 二、为什么 VLA 比普通 LLM 更需要显式 precision policy

如果是普通 `LLM`，很多系统里 precision 最后都可以简化成比较直接的配置：

- 模型主体用某个 dtype
- 部分 kernel 走更低精度
- 失败时 fallback

但 `VLA` 的难点在于，它的模块构成通常更杂：

- 图像相关输入
- 文本相关输入
- 状态输入
- expert-aware path
- 动作输出
- 可能还有控制相关逻辑

这意味着不同部分对精度的敏感度可能完全不同。

例如：

- 某些线性层非常适合早期就走低精度
- 某些 routing 或 norm 路径最好先保守
- 某些动作相关输出路径对数值抖动更敏感

如果没有显式 policy，最后常见的结果就是：

- 低精度开关散落在不同文件
- fused path 自己偷偷决定
- 某些模块 silently fallback
- benchmark 结果很难解释

所以 `VLA` 比普通 `LLM` 更需要一层显式的 precision policy。

## 三、一个最小可行的 precision policy 至少要回答什么

如果让我给 `VLA Runtime` 定义最小可行的 precision policy，我会要求它至少回答下面五件事。

### 1. 当前请求的全局精度目标是什么

例如可以抽象成：

- `safe_high_precision`
- `mixed_precision_preferred`
- `low_precision_preferred`

这一步的意义是：

> 不先问“具体用什么位宽”，先问“这次请求整体偏保守还是偏激进”。

### 2. 各模块的精度边界是什么

例如：

- attention 路径允许什么精度
- `MLP MoE` 允许什么精度
- router / norm 保持什么精度
- output adapter 哪部分必须保持更稳

这一步会把“不是所有模块同时降精度”这件事显式化。

### 3. fused path 对 precision 的支持矩阵是什么

不是所有 fused kernel 都会同时支持：

- `BF16`
- `FP16`
- `INT8`
- `FP8`

所以 policy 里必须明确：

> 当前请求想走的那条 path，和当前精度目标是否兼容。

### 4. 当前 precision 需要什么辅助条件

例如：

- 是否要求预打包 weight
- 是否要求固定 shape
- 是否要求某种 scale layout
- 是否要求某个 backend 可用

如果这些条件不显式化，低精度就会很快从“能力”退化成“玄学开关”。

### 5. 回退策略是什么

这是我觉得最不能少的部分。

因为 runtime 早期阶段最怕的不是“低精度没开成”，而是：

- 开成了但结果不可解释
- 出问题了却没有清楚 fallback

所以 policy 里必须有清楚的回退语义，例如：

- 低精度 fused path 失败后，回退到高精度 fused path
- 高精度 fused path 再失败，回退到 baseline

## 四、我更推荐的 precision policy 组织方式

如果要把 precision policy 写得更像 runtime 里的真实对象，我更倾向于把它拆成三层。

## 五、第一层：Global Precision Intent

这层表达的是“系统想要什么”，而不是“系统最后一定做到了什么”。

例如：

- `safe`
- `balanced`
- `aggressive`

这三个词看起来很简单，但它们的价值在于：

> 先把运行目标说清楚，再让 planner 和 backend 去协商能实现到什么程度。

比如：

- `safe` 更偏高精度和稳定回退
- `balanced` 更偏默认 mixed precision
- `aggressive` 更偏优先启用低精度和 fused path

## 六、第二层：Module Precision Rules

这层表达的是：

> 对不同模块，系统允许的精度边界是什么。

例如可以像这样理解：

- `attention`: `bf16 | fp16`
- `mlp_moe`: `bf16 | fp16 | low_precision`
- `router`: `bf16_only`
- `norm`: `bf16_only`
- `output_head`: `bf16_preferred`

这层的价值在于，它能把“低精度应该先落在哪里”从经验判断变成显式规则。

## 七、第三层：Backend Capability & Fallback

这层回答的是最现实的问题：

- 当前 backend 真支持什么
- 当前 shape 满不满足
- 当前 fused path 能不能开
- 不满足时往哪回退

也就是说，前两层是“意图”和“规则”，这一层才是：

> 在当前上下文里，最终到底采用什么精度执行路径。

这也是为什么我一直觉得 precision policy 不能离开 planner 单独存在。

因为最后把意图落成执行的，一定是 planner 和 backend 的协作。

## 八、为什么我不建议第一版就把 policy 做得过细

这一点很重要。

很多系统一谈到 policy，就容易把它设计成一个巨大的配置矩阵：

- 每层一个配置
- 每个 kernel 一个配置
- 每个 shape 一组例外

这在长期可能需要，但第一版通常太重。

我更推荐的第一版，是从少数几个清楚的 policy 开始，比如：

### Policy A：`safe_high_precision`

- 默认高精度
- fused path 可开，但以稳定为先
- 低精度尽量少
- fallback 最积极

### Policy B：`balanced_mixed_precision`

- 优先 mixed precision
- 允许部分 expert-heavy path 走低精度
- 不满足条件时退回高精度 fused path

### Policy C：`aggressive_low_precision`

- 优先低精度 path
- 优先启用更激进的 fused path
- 不适合控制安全敏感场景，但适合压测和探索

这三类已经足够支撑很多早期实验。

## 九、precision policy 和 benchmark 为什么必须绑在一起

如果 policy 设计得很好看，但 benchmark 根本看不出不同 policy 的行为差异，那它最后还是会失去意义。

我觉得 runtime 至少应该能在 benchmark 输出里明确记录：

- 当前采用了哪种 precision policy
- 哪些模块实际走了低精度
- 哪些模块发生了 fallback
- 当前 fused path 是否按预期启用

这样你后面看到一组性能数据时，才不会只看到一个结果，而是能知道：

> 这个结果到底是在哪种 precision 规则下得到的。

否则 policy 很快就会沦为一个“配了但没人真的理解它”的配置项。

## 十、对 Wall-X 这类项目，我更推荐的第一版 policy 路线

如果把这件事落回 `Wall-X` 这种项目，我更推荐的第一版路线大概是：

### 第一阶段

- 全局以 `safe/balanced` 为主
- 先把 fused path 高精度基线跑稳

### 第二阶段

- 优先给 `MLP MoE` 或最重的线性路径接低精度
- router、norm、输出相关路径保持更稳的精度

### 第三阶段

- benchmark 证明收益后，再扩大低精度覆盖范围
- planner 开始显式区分 cloud/edge 下的 policy 选择

这条路线的重点在于：

> 先把 precision policy 做成 runtime 的秩序，而不是先把低精度做成 scattered hacks。

## 十一、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的 precision policy 不该只是一个 dtype 配置表，而应该是一套由“全局精度意图 + 模块精度规则 + backend 能力与回退”组成的执行规则系统。

只有这样，低精度才不会沦为几个散落开关，而能真正成为：

> runtime 可以规划、可以解释、可以 benchmark、也可以安全回退的一部分能力。
