# 12. Wall-X 的 MLP MoE fused path 数据流草图

前一篇文章里，我把 `MLP MoE fused kernel` 的接口拆分讲清楚了：

- `routing/layout`
- `expert compute`
- `scatter/output`

但如果只停留在接口命名层，还是容易显得有点抽象。  
真正一进入实现阶段，大家马上会问的通常是：

> 这条 `MLP MoE fused path` 的数据到底应该怎么流？

这篇文章就不再讲“大方向”，而是更像一份实现前的草图说明：

> 从输入 hidden states 开始，到 expert 内部计算，再到最终输出回写，这条 fused path 的数据流应该怎样组织？

我不会把它写成某种唯一正确答案。  
但我会尽量给出一套对 `Wall-X` 这类 `VLA` 项目更现实、更容易演化的最小草图。

## 一、先说结论：第一版数据流的目标不是最短，而是最清楚

很多人一想到 fused path，就会自然希望：

> 数据流越短越好，最好中间没有任何显式阶段。

这个方向只有在系统边界已经非常清楚时才成立。  
对第一版 `MLP MoE fused path` 来说，我更认可的目标是：

> 不是让路径最短，而是让路径最清楚。

为什么？

因为第一版的真正任务不是冲极限性能，而是先把下面几件事做成立：

- routing metadata 是显式的
- expert 内部计算边界是清楚的
- scatter/output 的语义是独立的
- benchmark 可以准确拆分
- 后面可以自然接低精度

如果这几件事还没成立，你就过早追求“所有中间阶段都隐藏掉”，最后通常只会得到一个很难 debug 的黑盒。

## 二、这条数据流一开始到底从哪里开始

在 `Wall-X` 这种模型里，`MLP MoE` 通常不会直接接原始输入，而是接上游层已经产出的 hidden states。

所以第一版 fused path 的输入，可以先抽象成：

- `hidden_states`
- `token_types` 或等价的 expert assignment 信息
- 可选的 routing weight / prob
- 当前 batch / seq 的 shape metadata

注意这里一个容易犯的错：

> 不要把所有“决定 token 去哪”的语义都推给 kernel 自己去猜。

更合理的做法通常是：

- `Planner` 或 `routing/layout` 阶段先把结构信息整理出来
- `MLP MoE fused path` 只接收已经可执行的中间表示

这会让整个系统清楚很多。

## 三、推荐的数据流分成四段，而不是一口气做完

如果按实现阶段来画，我会建议第一版把数据流拆成下面四段。

## 四、Stage 1：输入展平与 token 分组

第一段要做的，不是计算，而是把输入整理成 expert-aware 执行能接受的形式。

这一段通常包括：

- `hidden_states` 从 `[B, S, H]` 展平成 `[T, H]`
- 根据 `token_types` 或 routing 结果决定 expert assignment
- 生成：
  - `sorted_token_ids`
  - `expert_offsets`
  - `row_id_map`
  - 可选的 `weights/probs`

这一段的输出，不应该只是一块“排序后的 hidden_states”，而应该是：

### 数据张量

- `hidden_states_perm`

### 结构 metadata

- `expert_offsets`
- `row_id_map`
- `routing_weights`
- `num_valid_tokens`

这样做的价值是：

> 后面的 compute kernel 可以只关心“按 expert 分段后的连续数据”，而不用重新理解整个输入语义。

## 五、Stage 2：expert 内部 fused compute

这一段才是 `MLP MoE fused path` 的核心。

如果按最典型的 `MoE MLP` 结构来看，它内部通常会经历：

1. `gate_proj`
2. `up_proj`
3. 激活
4. elementwise combine
5. `down_proj`

如果第一版要 fused，我最推荐的目标是：

> 把 expert 内部这一整段作为主融合单元。

从数据流角度，它接收的是：

- `hidden_states_perm`
- 按 expert 预打包的 `gate/up/down` 权重
- `expert_offsets`

产出的是：

- `expert_output_perm`

也就是说，**第一版的核心计算最好仍然保持 expert-ordered layout**。

为什么？

因为这样你可以先避免：

- 一边算，一边回原顺序
- 一边算，一边再做复杂 scatter
- 计算和输出布局切换强耦合

这对第一版调试和 benchmark 都更友好。

## 六、Stage 3：加权累加与 scatter

一旦 `expert_output_perm` 产出来，下一步就是把结果写回后续系统真正需要的布局。

这一段具体怎么做，要看模型语义，但通常会涉及：

- 按 `row_id_map` 回到原 token 顺序
- 如果一个 token 有多个 expert 结果，需要按 `routing_weights` 做加权
- 写回 `[T, H]` 或重新 reshape 成 `[B, S, H]`

这一段我建议先明确拆成一个独立 stage，而不是上来和 expert compute 完全绑死。

原因很简单：

### 1. 它的热点性质不一定和 compute 一样

有时这里是热点，有时不是。  
如果你提前全部绑死，会影响后续单独优化。

### 2. 它和 output layout 强相关

而 output layout 又很可能和：

- 后续层需求
- runtime path
- 是否继续保持 expert-ordered layout

这些条件强相关。

把它独立出来，会让后面演化空间大很多。

## 七、Stage 4：输出整理与回接主干

第一版 fused path 的最后一段，不是 GPU 算子本身，而是把结果重新接回主干执行图。

这一步需要清楚回答：

- 输出给后续层的格式是什么
- 是否已经恢复成标准 hidden states layout
- 是否要记录额外 benchmark 数据
- 是否需要把某些中间 metadata 继续保留下去

这一步虽然看起来不“硬核”，但对 runtime 很关键。

因为如果这一步语义含糊，你后面很容易出现：

- kernel 已经快了
- 但系统接不上
- 或者为了接上系统，又在 Python 层重新做了一堆 layout 变换

那前面的收益就会被吃掉。

## 八、第一版最推荐的数据流形态

如果把上面四段压缩成一张最小草图，我更推荐第一版长成这样：

```text
hidden_states [B, S, H]
  -> flatten
  -> routing/layout prepare
  -> hidden_states_perm [T, H] + metadata
  -> expert fused compute
  -> expert_output_perm [T, H]
  -> scatter / weighted accumulation
  -> hidden_states_out [B, S, H]
```

这个版本的优点是：

- 每一段职责清楚
- benchmark 容易拆
- 低精度容易插
- scatter 是否继续融合可以后置决策

这对第一版非常重要。

## 九、低精度应该插在这条数据流的哪里

如果后面要接低精度，我最推荐的切入位置不是 routing，也不是 scatter，而是：

> 先插在 Stage 2，也就是 expert fused compute 内部。

原因很直接。

低精度最自然、最有收益的地方通常是：

- 线性层权重
- dequant + matmul
- epilogue

而这些都集中在 Stage 2。

这意味着如果你的数据流先按四段拆清楚，后面接低精度时就会更自然：

- Stage 1 仍然主要是 metadata 和 layout
- Stage 2 负责真正承接低精度计算
- Stage 3 先保持更稳的高精度或较保守策略

这比“一开始整条路都想做成低精度”现实得多。

## 十、局部 megakernel 最可能长在哪一段

如果后面你真的要继续往下做局部 `megakernel`，我最看好的位置通常也是：

- Stage 2 单独长成更强的 compute mega-stage
- 或者 Stage 2 + Stage 3 局部合并

我不太建议一开始就把 Stage 1 到 Stage 4 全部揉在一起。

因为那会把：

- routing 语义
- 计算语义
- 输出布局语义

全部混成一个黑盒。

更现实的做法是：

> 先让数据流分段清楚，再观察哪两段之间的数据复用和固定 shape 足够强，值得进一步合并。

## 十一、这条草图真正想解决什么问题

说到底，这种“数据流草图”并不是为了画得好看，而是为了提前解决三个会反复出现的问题：

### 1. fused path 到底在替代哪一段

不是“把一切都优化了”，而是知道它精确替代哪段执行链。

### 2. 低精度和后续演化应该插在哪里

而不是最后才发现接口根本没有位置可放。

### 3. runtime 到底该怎样承接这条 path

只有数据流清楚，`Planner` 和 `Kernel Backend` 才能真正合作起来。

## 十二、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE fused path` 第一版最适合按“输入展平与分组 -> expert 内部 fused compute -> scatter/加权累加 -> 输出回接”四段来画数据流草图，这样既能支持 first win，又能自然承接低精度和局部 `megakernel` 的后续演化。

对 `VLA Runtime` 来说，最值钱的不是把第一版做得多极致，而是：

> 让第一版从一开始就是一条以后还能继续长的路径。
