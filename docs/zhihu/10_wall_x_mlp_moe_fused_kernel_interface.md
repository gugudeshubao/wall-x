# 10. Wall-X 的 MLP MoE fused kernel 应该怎么拆接口

前一篇文章里，我已经把一个判断讲清楚了：

> 如果 `Wall-X` 这类 `VLA` 项目里存在明显的 expert 结构，那么第一条 fused 路径，很多时候更适合先落在 `MLP MoE`。

但把方向讲清楚，只是第一步。  
真正进入实现之后，马上会遇到另一个更具体的问题：

> 如果真的要把 `MLP MoE` 做成 fused kernel，这个 kernel 的接口到底应该怎么拆？

这件事看起来像是“代码层细节”，但实际上它会直接决定：

- 你的 runtime 能不能承接这条 fused path
- 后面低精度好不好接
- layout 会不会过早被写死
- kernel 会不会一上来就做成不可维护的巨型黑盒

所以这篇文章不讲宏观方向，只讲一件事：

> `Wall-X` 的 `MLP MoE fused kernel`，接口应该怎样拆，才更适合做成第一条可演进的 fused path。

## 一、先说结论：不要一上来就做“一个超级 kernel”

如果只是从追求性能的直觉出发，很多人第一反应会是：

> 既然都要 fused 了，那干脆把 routing、permute、`gate/up/down`、激活、scatter 全部揉进一个核里。

这个方向不能说完全错，但对第一版实现来说，我不建议这么做。

原因很简单：

### 1. 调试边界会立刻消失

如果所有步骤一开始就塞进一个 kernel，你很快就会分不清：

- 是 routing 错了
- 是 layout 错了
- 是某个 projection 错了
- 还是激活、累加、回写的数值行为错了

### 2. 低精度以后更难接

一旦后面要上：

- `W8A16`
- `W8A8`
- `FP8`
- 更激进的 block-scale 方案

那个“大一统 kernel” 往往会比想象中更难演化。

### 3. Runtime 也很难接

如果接口一开始就只有一个巨大的“黑箱调用”，runtime 很难知道：

- 哪些输入是 routing 相关
- 哪些输入是 weight 相关
- 哪些是 layout metadata
- 哪些中间结果以后可以缓存或复用

所以我的判断是：

> 第一版的目标，不该是“最少函数数”，而该是“最清楚的执行边界”。

## 二、第一版 fused path 应该拆成哪几层

对 `Wall-X` 这种 `MLP MoE` 路径，我更推荐按三层拆。

### 第一层：Routing / Layout 层

这层负责回答：

- 每个 token 去哪个 expert
- 每个 expert 负责处理哪些 token
- token 顺序怎样重排
- 每个 expert 的 segment 边界在哪里

这层的输出不该只是一个“排序后的 tensor”，还应该显式提供 metadata，例如：

- `sorted_token_ids`
- `expert_ids`
- `expert_offsets`
- `row_id_map`
- 可能的 `weights / probs`

这一层的意义在于：

> 把 expert-aware 执行的结构信息显式化。

因为一旦这些信息不显式化，后面 fused kernel 很快就会和具体模型代码死绑在一起。

### 第二层：Expert Compute 层

这一层才是 `MLP MoE fused kernel` 的核心。

它负责：

- `gate_proj`
- `up_proj`
- 激活
- combine
- `down_proj`

如果第一版要 fused，我建议就把焦点先放在这一层。

关键不是“要不要所有东西全融”，而是：

> 专门把 expert 内部最规则、最像固定模板的那段算子链融起来。

### 第三层：Scatter / Output 层

这层负责把 expert 输出重新组织回后续模型或 runtime 所需要的 layout。

它要解决的是：

- 是否要 `unpermute`
- 是否要带权累加
- 输出 buffer 怎样写回
- 是否能直接写进下一阶段需要的布局

这层单独拆出来很重要，因为：

- 有些模型希望还保持 expert-ordered layout
- 有些模型必须回到原始 token 顺序
- 有些场景后面可能还想进一步做局部 `megakernel`

如果一开始就把回写逻辑写死在 compute kernel 里，后面很容易失去灵活性。

## 三、第一版接口为什么应该“偏瘦”

如果现在真的开始做第一版 `MLP MoE fused kernel`，我会更倾向于设计一个“偏瘦接口”。

也就是说，它只负责自己最该负责的那一段，不抢太多语义。

我会更建议接口层按下面这类思路来组织。

## 四、推荐的第一版接口结构

如果用概念形式表示，我会建议先拆成三个显式接口。

### 接口 A：`prepare_moe_layout(...)`

这一层不做真正的 `MLP` 计算，只做 routing 和 layout 准备。

输入可以是：

- `token_types` 或 `expert_ids`
- 可选的 routing weights
- 当前 batch / seq 的 shape 信息

输出应该尽量结构化，而不是只返回一个大 tensor。  
例如：

- `sorted_token_ids`
- `expert_offsets`
- `row_id_map`
- `expert_weights`
- `num_valid_tokens`

这层不应该被写死成某个模型私有逻辑，而应该尽量接近 runtime 可复用的 layout primitive。

### 接口 B：`moe_mlp_fused_forward(...)`

这是核心计算接口。

输入建议只包含三类东西：

1. 已经准备好的 expert-ordered input
2. 按 expert 组织好的权重
3. layout metadata

它的职责就是：

- 在 expert 维度上执行 `gate/up/down`
- 完成激活和 combine
- 输出 expert-ordered 结果

如果第一版先做高精度 fused 基线，这一层最好不要再额外承担太多别的语义。

### 接口 C：`scatter_moe_output(...)`

这一层负责把 expert 输出重新整理成后面需要的 layout。

输入：

- expert-ordered output
- `row_id_map`
- 可能的 routing weights
- 输出目标 buffer 或目标 shape

输出：

- 原顺序 hidden states
- 或继续保持某种 runtime 友好的中间布局

这层单独存在的价值是：

> 让 compute 和 output organization 解耦。

## 五、为什么不要把 routing 和 compute 完全绑死

这件事很容易被忽视。

很多人一做 fused kernel，就会天然希望：

> routing 和 compute 最好在一个地方全部解决。

但对 `VLA Runtime` 来说，这种耦合通常会太重。

因为 runtime 后面很可能需要：

- 复用 routing 结果
- 记录 shape 和 expert metadata
- 做 profiling
- 对端侧和云侧走不同 planner

如果 routing 和 compute 完全绑死，那么 runtime 层会非常被动。

更好的做法通常是：

- routing 结果显式化
- compute kernel 专注算子链
- scatter/output 再单独处理

这会让整个系统更像一个可演化的执行层，而不是一堆难拆的 patch。

## 六、这个接口设计和低精度有什么关系

关系非常大。

很多人会把“接口设计”和“低精度”看成两件不同的事，但实际上它们强相关。

因为一旦后面要支持低精度，你至少要新增这些东西：

- packed weight
- scales
- block size 或分组信息
- accumulator 策略
- 可能的 epilogue 选择

如果第一版接口没有为这些东西留位置，后面就会很痛苦。

所以第一版即使先只跑 `BF16`，接口层也应该预留：

- `weight_pack_format`
- `scale_ptr` 或 scale metadata
- `epilogue_mode`
- `output_layout`

不一定一开始就全部实现，但至少应该在语义上给这些能力留钩子。

这就是为什么我一直强调：

> 低精度不只是一个数据类型问题，而是 runtime 后端设计问题。

## 七、对 Wall-X 这种项目，第一版最应该避免什么

如果把话说得更直白一点，我觉得第一版最应该避免三件事。

### 1. 避免把所有步骤一次性塞进一个 mega-interface

这会让正确性验证、benchmark 和后续替换都变得困难。

### 2. 避免让接口只对当前这一个 Python 调用点成立

如果接口完全长成“为某个调用函数硬适配”的样子，那它很难真正成为 runtime 的一部分。

### 3. 避免先把低精度语义排除在外

你可以先不实现，但不要在接口语义上堵死这条路。

## 八、一个更现实的演化顺序

如果让我给一个更现实的推进顺序，我会建议：

### 第一步：先把 routing/layout metadata 显式化

哪怕还没写 fused kernel，这一步也值得先做。

### 第二步：先做 `BF16` 的 `moe_mlp_fused_forward`

重点是把：

- `gate/up/down`
- activation
- combine

先稳定融合起来。

### 第三步：再决定 scatter 要不要进一步融合

如果 benchmark 显示回写很重，再往下融合；如果不是热点，就先保留分层。

### 第四步：在这个基础上接低精度

这时候接口已经清楚了，接 `W8A16 / W8A8 / FP8` 的成本会小很多。

## 九、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE fused kernel` 第一版不该做成一个不可拆的超级核，而应该拆成 `routing/layout`、`expert compute`、`scatter/output` 三层，让它既能拿到 first win，又能自然承接低精度和后续 runtime 演化。

对 `VLA Runtime` 来说，最值钱的从来不只是“某个核快了”，而是：

> 你有没有把第一条 fused path 做成一条以后还能继续长的路。
