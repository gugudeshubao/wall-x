# 24. Wall-X 的 MLP MoE scatter/output path 什么时候值得继续 fused

一路把 `Wall-X` 的 `MLP MoE` 路径写到这里，前半段其实已经越来越清楚了：

- 第一条 fused path 很适合先落在 `MLP MoE`
- `gate/up/down` 的接口、数据流和伪代码都可以先稳定下来
- low-precision 第一批也更适合先落在线性层和 matmul 上

但只要前半段一稳定，问题很快就会自然往后推：

> 既然 `expert compute` 这一段已经 fused 了，那 `scatter/output` 这一段什么时候值得继续 fused？

这不是一个“要不要更极致”的审美问题，而是一个很现实的系统问题。

因为如果你继续做：

- `MLP MoE fused compute`
- low-precision
- packed weight
- planner / benchmark / fallback

最后总会遇到下面这种情况：

> compute 已经很快了，但 path-level latency 还是没有按预期继续下降。

这时候，后半段 `scatter/output` 往往就会从“看起来只是收尾逻辑”变成下一步真正的瓶颈候选。

这篇文章就只讲这个问题：

> `MLP MoE` 的 `scatter/output path` 到底什么时候值得继续 fused？

## 一、先说结论：不是 compute fused 完就自然应该继续 fused scatter

很多人会天然有一个线性直觉：

> 前半段 fused 了，下一步当然就该把后半段也继续 fused。

这个直觉不完全错，但它非常容易把人带进一个“只看局部完美、不看系统收益”的误区。

我现在更认可的判断是：

> `scatter/output path` 是否值得继续 fused，不取决于它在逻辑上是不是“下一个还没 fused 的地方”，而取决于它是否已经成为 benchmark 里真正阻碍路径继续缩短的主成本之一。

也就是说，这一步不该按顺序感来做，而应该按收益感来做。

如果换成更直白的话：

- 不是因为它“还没 fused”就该 fused
- 而是因为它“已经贵到足以值得被单独处理”才该 fused

## 二、为什么 scatter/output 经常会在后半段突然变重要

这件事其实很常见。

在 `MLP MoE` 路径里，很多人最开始注意力都会放在：

- `gate_proj`
- `up_proj`
- 激活和 combine
- `down_proj`

这是合理的，因为这些地方通常最显眼、最重，也最像典型的 kernel 优化对象。

但一旦这些部分被做得更紧，系统的时间构成就会发生一个很自然的变化：

> 原来被 compute 掩盖掉的后半段固定成本，会变得越来越显眼。

尤其是在下面这些场景里，这种现象会更明显：

### 1. batch 很小

比如 `batch=1` 或非常小的 token 规模。

这时很多固定开销的相对占比会上升得很快。

### 2. expert 分布更碎

如果 token 分到多个 expert 之后，输出回写和 weighted accumulation 很零散，那么 scatter 本身就可能变得不轻。

### 3. output layout 和下游需求不一致

如果 compute 输出保持的是 expert-ordered layout，但后面马上又要回到原顺序、或者要进入另一种布局，那这里天然就会出现一次昂贵的“重新组织”。

所以 `scatter/output` 之所以会在后半段突然重要，不是因为它本身被高估了，而是因为：

> 一旦前半段被压下去，它就更容易暴露成新的瓶颈。

## 三、什么时候“不该”继续 fused scatter/output

为了避免过度优化，我更想先把“不该做”的情况说清楚。

## 四、第一种情况：它还不是主瓶颈

这是最直接的一条。

如果 benchmark 还清楚地显示：

- compute 仍然最重
- low-precision 还有很大空间
- layout/routing 还没有稳定

那过早去 fused `scatter/output`，大概率只会把系统复杂度抬高，却拿不到相称收益。

换句话说：

> 后半段是不是“看起来还没优化”，和它是不是“当前最值得优化”不是一回事。

## 五、第二种情况：输出语义还没稳定

如果你现在还在反复改这些东西：

- 是否保持 expert-ordered 中间结果
- 输出到底给哪一层
- 是否要加权累加
- output adapter 要不要再裁剪或映射

那这个阶段就不太适合太早把 `scatter/output` 硬固化进 fused kernel。

因为这会让你很快遇到一个老问题：

> 还没稳定的系统语义，被过早写进了低层实现。

这往往不是加速，而是提前锁死。

## 六、第三种情况：fallback 还没讲清楚

这一点在 `VLA Runtime` 里尤其重要。

如果你还不能清楚回答：

- fused scatter/output 失败了退到哪里
- 是退回高精度 scatter 还是整个 path 回退
- benchmark 如何区分“命中 fused output”和“退回普通 output”

那过早做这一步，系统就会很难解释。

所以我会坚持一个判断：

> 没有清楚 fallback 语义的后半段融合，通常做得越快，后面越容易失控。

## 七、那什么时候“值得”继续 fused scatter/output

如果要说积极条件，我现在更认可下面几条。

## 八、第一条：benchmark 已经证明 scatter/output 是主成本之一

这是最根本的条件。

它不一定要成为全路径第一瓶颈，但至少应该满足：

- compute 已经明显收敛
- `scatter/output` 在 path-level breakdown 里占比显著
- 再优化前半段已经不如继续优化后半段划算

也就是说，只有当 benchmark 已经在用数据把你往这里推时，这一步才真正值得做。

## 九、第二条：输出布局已经足够稳定

这一点非常关键。

如果你已经基本确定：

- `expert_output_perm` 的语义
- 何时恢复原顺序
- weighted accumulation 是不是固定需要
- 最终输出交给后续层的布局

那这时候继续 fused `scatter/output` 才更像是在“加速一个明确问题”，而不是在“猜未来的系统形态”。

## 十、第三条：后半段的 memory movement 已经成为明显负担

很多时候，继续 fused `scatter/output` 的真正动机，不是算术本身，而是：

> 它正在制造额外的中间 buffer、额外的 memory movement 和额外的 launch。

如果 benchmark 和 profiler 已经表明：

- 输出回写次数很多
- 中间结果经常来回搬
- weighted accumulation 单独成了一段昂贵的尾巴

那继续 fused 的理由就会很充分。

## 十一、第四条：planner 已经能把它作为独立 path 理解

这一点我觉得特别重要。

如果 planner 还只能理解：

- baseline
- compute fused
- low-precision fused

却还完全不知道：

- 当前 output path 是普通版还是 fused 版
- 当前 fallback 是退回哪一级

那你贸然往下做，很容易把 runtime 语义搞乱。

所以更好的节奏通常是：

> 先让 runtime 语言里有“后半段 fused path”的位置，再真的把它做出来。

## 十二、一个更现实的融合顺序

如果现在真的要开始往 `scatter/output` 这一段推进，我更推荐的顺序通常不是“一步并到底”，而是分层往前走。

## 十三、阶段一：先让 scatter 成为可测的独立 stage

哪怕还没 fused，这一步也很值。

你至少要先清楚看到：

- scatter 本体的时间
- weighted accumulation 的时间
- 输出 layout 转换的时间

如果这一步都看不清，就谈不上后面要不要继续融合。

## 十四、阶段二：先做 output layout writeback 的局部优化

很多时候，后半段最先值得动的，不一定是把所有东西都揉进一个 kernel，而是：

> 先把最重复、最固定、最显眼的 writeback 路径做得更紧。

例如：

- 减少不必要的中间 buffer
- 合并某些重复回写
- 让输出直接落到下一阶段更接近的布局

这一步往往就能带来一轮很现实的收益。

## 十五、阶段三：再考虑把 weighted accumulation 一起拉进来

这一步通常比前一步更激进，因为 weighted accumulation 更容易和数值语义、output 稳定性绑定得更紧。

所以我更建议：

- 先看 writeback 是否已经很重
- 再决定 weighted accumulation 是否值得一起 fused

而不是一开始就全做。

## 十六、阶段四：最后再评估是否形成“后半段局部 megakernel”

这一步通常不该太早。

只有当下面这些都比较清楚时，它才真正值得：

- 输出布局稳定
- fallback 清楚
- benchmark 证明后半段真的很贵
- edge/cloud 场景差异也开始明确

到这时，`scatter/output` 才可能从“值得小步优化的尾巴”，变成“值得成为下一块局部 mega-stage 的后半段执行单元”。

## 十七、这件事和 low-precision 的关系是什么

这一点我也想顺手说清楚。

有些人会自然觉得：

> 如果后半段也要继续 fused，那是不是也该尽快一起低精度？

我的判断会更保守一点：

> `scatter/output` 的融合时机，通常应该先由执行链和 memory movement 决定，而不应该一开始就让 low-precision 成为它的主要推动力。

原因很简单。

因为对后半段来说，更先出现的痛点通常是：

- 回写形态不合理
- memory movement 过多
- launch 过多

而不是它自己本身最适合作为 low-precision 首批落点。

所以这一步更适合先按执行链来优化，而不是一上来就按 bit-width 来优化。

## 十八、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE scatter/output path` 不该因为“前半段已经 fused”就被顺手继续下沉，而应该等到 benchmark 已经明确表明后半段的 memory movement、weighted accumulation 和 output layout writeback 真的成为下一阶段主成本，并且输出语义、planner 语言和 fallback 层级都足够稳定之后，才值得继续 fused。

因为对 `VLA Runtime` 来说，真正值钱的不是把每一段都尽快塞进 kernel，而是：

> 让每一次继续下沉，都是在为系统真正解决下一轮已经暴露出来的问题。
