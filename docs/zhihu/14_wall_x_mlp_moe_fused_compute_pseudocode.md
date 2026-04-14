# 14. Wall-X 的 MLP MoE fused compute kernel 最小伪代码

前面几篇文章里，我已经把 `MLP MoE fused path` 的接口和数据流拆开了：

- `routing/layout`
- `expert compute`
- `scatter/output`

如果继续往实现层走，下一步最自然的问题就是：

> 那么 `expert compute` 这一段，第一版到底应该长成什么样？

这篇文章不准备直接给出某种最终 CUDA 实现，因为那样反而会把问题说得过早收死。  
我更想做的是：

> 先给出一份足够接近实现、但仍然保留演化空间的“最小伪代码草图”。

也就是说，目标不是写成能编译的 kernel，而是把下面几件事提前讲清楚：

- kernel 到底接收什么输入
- expert 内部的计算顺序是什么
- 哪些东西应该先做成显式 buffer
- 哪些地方以后最适合插低精度
- 第一版不该抢哪些语义

## 一、先说结论：第一版 compute kernel 要先做“可替换的模板”，不是“终极最优核”

很多人在真正准备写 fused kernel 时，会有一个自然冲动：

> 既然都写到这一步了，那最好一上来就写成终极版。

这通常不是最优策略。

对 `Wall-X` 这种 `MLP MoE` 路径，我更认可的第一版目标是：

> 先写出一个结构边界极清楚、后面能承接低精度和更深融合的 compute 模板。

这意味着第一版的核心任务不是：

- 追求所有边角都 fused 掉
- 一次性做成 megakernel
- 一开始就把所有精度后端都接完

而是：

- 把输入/输出边界做对
- 把 `gate/up/down` 计算顺序做清楚
- 把激活和 combine 放在最自然的位置
- 给低精度和后续 epilogue 留出接口

也就是说，第一版更像是一块“正确的地基”。

## 二、第一版 compute kernel 不应该负责什么

这一点我想先说清楚。

如果这一段的职责边界不先画出来，伪代码最后很容易变成一坨“什么都做一点”的东西。

第一版 `MLP MoE fused compute kernel`，我不建议它直接负责：

### 1. 重新理解 routing 语义

它不应该自己从原始 token type 推断所有 expert 路由逻辑。

更合理的前提是：

- routing/layout 阶段已经给出 `expert_offsets`
- 输入已经按 expert 排好
- kernel 只需要按 expert 连续段去算

### 2. 最终系统级输出组织

它不应该一上来就自己决定：

- 是否恢复原 token 顺序
- 是否进入下一层某种特殊 layout
- 是否顺手做系统级 benchmark 汇总

这些更适合放在 scatter/output 或 runtime 层。

### 3. 复杂 fallback 逻辑

第一版 kernel 自己不该做复杂策略选择。  
“当前到底走哪条 path”应该是 `Planner` 的职责。

kernel 更适合做的事情是：

> 在既定路径下，把 expert 内部那条最规则的算子链做对。

## 三、这段 kernel 的输入应该长什么样

如果把第一版输入先压到最小集合，我更推荐像下面这样组织。

### 数据输入

- `x_perm`
  含义：已经按 expert 排序后的输入 hidden states  
  形状：`[T, H]` 或更接近 expert block 的视图

- `expert_offsets`
  含义：每个 expert 在 `x_perm` 中的起止边界  
  形状：`[E + 1]`

- `w_gate`
- `w_up`
- `w_down`

这些权重最好已经是 runtime/backend 能接受的布局，而不是临时在 kernel 里再做组织。

### 可选输入

- `routing_weights`
  如果后面要做带权输出，通常更适合由下一阶段使用，但也可以在某些 epilogue 设计里提前准备

- `scale / quant metadata`
  第一版可以不实现低精度，但接口最好为它留位置

### 输出

- `y_perm`
  含义：仍然保持 expert-ordered layout 的输出

这个点我会坚持一下：

> 第一版输出最好先保持 expert-ordered，不要急着在 compute 内部就恢复成系统最终布局。

原因很简单：

- 更容易 debug
- 更容易 benchmark
- 更容易以后把 scatter 单独优化

## 四、第一版 compute 的最小语义顺序

如果只看 expert 内部最小语义，我会把它压成下面五步：

1. 取出当前 expert 对应的 token 段  
2. 做 `gate_proj`  
3. 做 `up_proj`  
4. 做激活和逐元素 combine  
5. 做 `down_proj` 并写回 `y_perm`

这里有两个关键点。

### 1. `gate_proj` 和 `up_proj` 的相对关系

它们很适合被视为一个“成对的投影阶段”。

第一版即使还没做到极致融合，也应该在语义上把它们视为同一组操作，而不是两条互不相干的路径。

### 2. 激活与 combine 的位置

这一步通常最自然地放在：

- `gate_proj` / `up_proj` 之后
- `down_proj` 之前

如果后面要做更深融合，这里通常也是最自然的 epilogue 扩展点。

## 五、一个更接近实现的最小伪代码

下面这段伪代码，不是要你照抄成 CUDA，而是想把第一版 compute 的职责边界讲清楚。

```text
function moe_mlp_fused_compute(
    x_perm,
    expert_offsets,
    w_gate,
    w_up,
    w_down,
    optional_quant_meta
):
    allocate y_perm

    for expert_id in experts:
        start = expert_offsets[expert_id]
        end = expert_offsets[expert_id + 1]

        if start == end:
            continue

        x_e = x_perm[start:end]

        gate_e = matmul(x_e, w_gate[expert_id])
        up_e   = matmul(x_e, w_up[expert_id])

        act_e  = activation(gate_e)
        mid_e  = act_e * up_e

        out_e  = matmul(mid_e, w_down[expert_id])

        y_perm[start:end] = out_e

    return y_perm
```

这段看起来很朴素，但它其实已经明确了几个重要边界：

- kernel 的输入是 expert-ordered
- expert 是按 segment 处理
- `gate/up/down` 是一条清楚的连续链
- 输出先不抢最终布局语义

第一版就应该先把这件事立住。

## 六、如果想再往前一步，第一版最值得融合哪一段

在上面这份最小伪代码里，真正最值得优先融合的，不是所有步骤都一起，而通常是下面这两块：

### 1. `gate_proj + up_proj`

这两步天然是并列的。

如果权重布局和 tile 合适，第一版就很适合把它们当成一组“成对投影”来做。

### 2. `activation + combine`

这一步通常就是最典型的 fused epilogue 候选。

因为它：

- 非常局部
- 数据复用强
- 单独拆出来会增加小 kernel 或额外回写

也就是说，如果要从“最小语义伪代码”再往“更像 fused kernel”迈一步，我最推荐先融合的就是这两处。

## 七、低精度以后最自然插在哪里

如果后面要接低精度，我最推荐优先插在这三个位置：

### 1. 权重读取阶段

也就是 `w_gate / w_up / w_down` 的 backend 组织方式。

这里最适合做：

- packed weight
- scale metadata
- block/group layout

### 2. `matmul` 内部

这里才是真正低精度最容易带来收益的地方。

### 3. `activation + combine` 之后的 epilogue

如果后面要进一步做融合，这里很可能是最自然的扩展点。

而我不建议第一版优先碰的，是：

- routing metadata 本身的低精度
- scatter/output 这类更偏系统边界的部分

## 八、这段伪代码真正应该服务什么

我觉得这类“最小伪代码”最重要的价值，不是看起来多像论文算法，而是能提前服务下面几件事：

### 1. benchmark 拆分

你能知道：

- `gate/up` 占多少
- `activation/combine` 占多少
- `down` 占多少

### 2. kernel 分工

你能知道第一版到底是不是只该先做 compute，还是连 scatter 一起做。

### 3. precision policy 设计

你能知道低精度最自然该插在哪几层。

### 4. planner 与 backend 接口

你能知道 planner 下发给 backend 的最小语义单元应该是什么。

## 九、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE fused compute kernel` 第一版最适合先按“expert-ordered 输入 -> gate/up 双投影 -> activation/combine -> down 投影 -> expert-ordered 输出”这条最小语义链来写伪代码，而不是一上来就做成职责混杂的超级核。

对 `VLA Runtime` 来说，第一版最值钱的不是极限融合，而是：

> 把 compute 这件事先做成一个以后还能继续长的、语义清楚的后端模板。
