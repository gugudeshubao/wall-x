# Wall-X 的 MoE 到底加在了哪里：从 Decoder Layer 到 Token 路由

前两篇我已经把两件基础事情铺平了：

- 一是 Wall-X 不是什么“整套靠自定义 CUDA 算子重写”的模型
- 二是它的多模态主干，本质上还是“视觉先编码成 embedding，再回填进统一序列，最后交给 decoder”

到了这里，就该回答真正让 Wall-X 和基础 Qwen2.5-VL 拉开差距的问题了：

它的 MoE，到底加在了哪里？

很多人第一次看到 “MoE” 这三个字，脑子里会自动补成一种很熟悉的结构：  
无非是在 Transformer 的 FFN 位置换成专家 MLP，然后加一个 router。

Wall-X 不是这么简单。

更准确地说，它确实有专家 MLP，也确实有 router，但它的 MoE 改造不是“只改 MLP 一处”，而是把 decoder layer 这一层整体做成了可路由版本。你在这条链上会看到：

- expert-aware 的 norm
- expert-aware 的 attention
- expert-aware 的 MLP
- 为了这些 token type 路由额外调整的 mask 和 position 逻辑
- 以及在 `mot_opt=True` 时，为了跑得更快做的 token 重排

所以这一篇我想把一个判断写清楚：

> Wall-X 的 MoE，不是一个挂在 decoder 后面的附加模块，而是直接长在 decoder layer 里面。

## 先说结论：MoE 不在视觉分支，在 decoder 里

先把最容易混淆的一点说掉。

Wall-X 的视觉分支还是上一篇讲的那套东西：

- `patch_embed`
- `rot_pos_emb`
- `get_window_index`
- `vision blocks`
- `merger`

视觉入口没有变成专家结构。  
真正的 MoE 化，发生在 `self.model` 这一侧，也就是统一序列进入 decoder 之后。

这一点在 Wall-X 动作版模型的初始化里其实写得非常直白：

```python
self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(...)
self.model = Qwen2_5_VLMoEModel(...)
```

这句话的含义非常明确：

- 视觉主干还是原来那套多模态视觉编码器
- 真正被 Wall-X 改造成 MoE 版本的，是 decoder 主干

所以如果你后面要分析 MoE 带来的性能和行为变化，千万别把它想成“视觉专家网络”，也别把它想成“独立的 action 子网”。它插入的位置，就是 decoder 本体。

## 基础替换动作只有一句话，但影响很大

Wall-X 对基础 Qwen2.5-VL 的第一层替换，看上去其实非常朴素：

```python
self.model = Qwen2_5_VLMoEModel(...)
```

但这句替换的后果很大，因为 `Qwen2_5_VLMoEModel` 里面的每一层，都不再是原版的 `Qwen2_5_VLDecoderLayer`，而是：

```python
Qwen2_5_VLDecoderLayer_with_MoE(...)
```

也就是说，MoE 不是在 decoder 之外再加一层处理，而是把每个 decoder layer 的内部逻辑改掉了。

到这里你就可以先立一个判断：

- Wall-X 不是“少数几层加了专家”
- 它是“整个 decoder stack 的层定义都被 MoE 化了”

这也是为什么后面你会在 layer 内部看到一整套跟 token type、expert、gate、permute 相关的逻辑。

## 一个 MoE decoder layer 到底改了什么

如果把 `Qwen2_5_VLDecoderLayer_with_MoE.forward()` 压缩一下，它的执行顺序大致是：

```text
输入
-> 第一次 norm MoE
-> attention（可能是 joint attention）
-> gated residual
-> 第二次 norm MoE
-> MLP MoE
-> gated residual
-> 输出
```

这个顺序看起来和普通 decoder layer 很像，但真正的区别在于，里面的几个关键部位都不再是“所有 token 共用一套参数”的统一路径。

Wall-X 这里至少有 3 个开关：

- `norm_moe`
- `attention_moe`
- `mlp_moe`

这 3 个开关决定的是：

- norm 是否按 expert 分开
- attention 是否按 expert 分开生成 QKV/O
- MLP 是否走 `SparseMoeBlock`

所以别把 Wall-X 的 MoE 理解成“FFN 改成专家 MLP 就结束了”。它的想法更接近：

> 既然序列里本来就有不同语义类型的 token，那就不要强迫所有 token 在 layer 内走完全相同的处理路径。

## `moe_token_types` 才是这套路由的核心信号

那路由信号从哪来？

答案非常直接，就是 `moe_token_types`。

这一点其实脚本层就已经埋好了。上一篇讲过，当前推理入口里会这么生成它：

```python
moe_token_types = inputs.input_ids == action_token_id
```

也就是说，在当前最常见的推理场景下，`moe_token_types` 本质上是一个布尔张量：

- 普通上下文 token 是 0
- action token 是 1

再往下看 `TokenTypeRouter`，它的实现甚至简单到有点“反高潮”：

```python
experts_indices = token_types % self.num_experts
```

这其实很好理解。

Wall-X 当前最常见的 routing，并不是某种复杂的 learned router 在在线打分，而是“先根据 token 语义类型把路由标签标出来，再把这些标签映射到 expert index”。

这也带来一个很清楚的工程后果：

- 路由可解释性很强
- 你能明确知道哪些 token 被送去了哪个 expert
- 代价是它不像经典 top-k router 那样完全靠内容动态路由

从 Wall-X 这个任务设定看，这么做其实挺合理。因为这里 token 的语义分工本来就比较明确，比如普通上下文 token 和 action token，天然就可以走不同处理路径。

## `start_indices / end_indices` 不是玄学，就是 expert 段边界

理解 Wall-X 的 MoE，另一个绕不过去的点是：

- `start_indices`
- `end_indices`

第一次看到这两个名字，很多人会觉得抽象。其实它们的含义很简单：

> 如果把 token 先按 expert 分组重排，那么每个 expert 在这条重排后序列里会占一段连续区间，`start_indices / end_indices` 就是在标这段区间的左右边界。

在顶层模型里，如果这两个值没有显式传进来，代码会根据 `moe_token_types` 直接数每个 expert 有多少 token，然后做 cumulative sum：

```python
group_size[i] = (moe_token_types == i).sum()
start_indices = cumsum(group_size) - group_size
end_indices = cumsum(group_size)
```

这段逻辑其实已经把它的语义写得很直白了：

- 先统计每个 expert 分到多少 token
- 再把这些 token 在重排后序列里分成连续区间
- 最终得到每个 expert 的 `[start, end)` 段

所以 `start_indices / end_indices` 并不是某个额外的模型参数，而是 MoE 路由在实现层面的“段描述符”。

## `mot_opt=True` 时，Wall-X 会先把 token 排成 expert 顺序

到这里就能理解 `mot_opt` 是干什么的了。

如果不开 `mot_opt`，那么大体上还是在原序列布局里，用 mask 去挑出属于某个 expert 的 token，再分别处理。  
如果开了 `mot_opt`，则会先调用：

```python
hidden_states, row_id_map = ops.permute(hidden_states, moe_token_types.view(-1))
```

把 token 按 expert 分组重排。

重排之后的好处很直接：

- 同一个 expert 的 token 变成连续片段
- 后面不管是 norm、MLP 还是输出投影，都可以按连续内存块处理
- 更容易对接 `start_indices / end_indices`

最后在整层甚至整栈处理完之后，再用：

```python
hidden_states = ops.unpermute(hidden_states, row_id_map, probs)
```

把顺序还原回来。

所以 `mot_opt` 本质上不是模型语义变化，而是一种执行优化：

> 既然已经知道 token 属于哪个 expert，那就干脆先按 expert 顺序排好，再做专家计算。

这也是 Wall-X 自定义算子里 `permute / unpermute` 为什么会是核心组件的原因。它们不是边角料，而是 MoE 高效执行的重要支点。

## expert-aware 的 norm：不是所有 token 共用同一个 RMSNorm

先看 layer 里的第一处改造：`norm_moe`。

如果 `norm_moe=True`，decoder layer 不再只用一套共享的 `input_layernorm` 和 `post_attention_layernorm`，而是会为每个 expert 分别准备一组 norm：

- `input_layernorms`
- `post_attention_layernorms`

然后在 `_apply_norm_moe(...)` 里，根据 token type 把不同 token 送进不同的 norm。

这一步很值得注意，因为它说明 Wall-X 的 MoE 不只是让不同 token 走不同 MLP，它甚至允许不同 token 在进入 attention / MLP 之前就走不同的归一化路径。

从建模角度看，这相当于承认了一件事：

> 普通上下文 token 和动作 token，不一定适合用完全同一套归一化统计来预处理。

而如果 `use_adarms` 也打开了，flow expert 那一支还会额外带条件输入，进一步把 norm 做成 conditional 的。

这已经远远不是“给 FFN 换个专家 MLP”那么简单了。

## expert-aware 的 MLP：这里才是最像传统 MoE 的地方

再看 `mlp_moe`。

这一块是大家最熟悉的 MoE 形态：`SparseMoeBlock` 里挂着多个 expert，每个 expert 本质上都是自己的 `BlockSparseMLP`。

执行逻辑也很直接：

- 先根据 `token_types` / `start_indices` / `end_indices` 找到每个 expert 对应的 token 段
- 对每个 expert 只取自己负责的输入维度 `dim_input`
- 跑自己的 MLP
- 再把结果 scatter 回输出缓冲区

这条链和你平时理解的专家 FFN 已经很接近了。  
所以如果只从“哪一块最像传统 MoE”来讲，答案就是这一块。

但 Wall-X 的特点恰恰在于，它没有把 MoE 停留在这里。

## 更特别的是 `attention_moe`

Wall-X 比较有意思的一点，在于它允许 attention 本身也做 expert 化。

当 `attention_moe=True` 时，decoder layer 里的 `self.self_attn` 不再是普通的 Qwen attention，而会被替换成：

```python
JOINT_QWEN_ATTENTION_CLASSES[config._attn_implementation](...)
```

对应实现就是 `JointQwen2VLAttention`，以及它的 flash 版本 `JointQwen2VLFlashAttention`。

这里最值得强调的一句话是：

> attention kernel 本身没有换成一套全新的数学，真正变化的是 Q、K、V、O 这些投影的生成方式。

为什么这么说？

因为你顺着 `JointQwen2VLAttention` 往里看，会发现它内部是按 expert 分别维护：

- `q_proj_experts`
- `k_proj_experts`
- `v_proj_experts`
- `o_proj_experts`

也就是说，不同 expert 的 token，不再共用同一套 QKV/O 投影矩阵。

但这些 expert-specific QKV 一旦生成出来，后面的 attention 计算本身，依然可以落到：

- `torch.nn.functional.scaled_dot_product_attention`
- 或 `flash_attn_func`

所以这里一定要分清楚两件事：

- Wall-X 确实把 attention 做了 expert 化
- 但它并没有把 attention 数学本身另起炉灶

这和上一篇“为什么它不是 MLA”那条判断是一致的。  
它改的是 attention 的参数分配方式，不是换掉整个 attention 核心形式。

## `JointQwen2VLAttention` 具体变了什么

如果把 `JointQwen2VLAttention` 再往下压一层，它大致在做 3 件事：

1. 根据 token 所属 expert，分别生成 Q/K/V  
2. 对统一序列执行 attention  
3. 再根据 token 所属 expert，分别用对应的 `o_proj` 投回去  

这里最巧妙的点在于：

- token 在语义上可以按 expert 分开
- 但 attention 仍然可以在统一序列上下文里发生

这就是为什么它叫 joint attention。  
不是说每个 expert 各自闭门算一套 attention，而是说：

> 不同 expert 的 token 可以用不同投影方式生成表示，但注意力上下文本身仍然处在一条联合序列里。

这比“把序列切碎，每组 token 只看自己那组”更接近 Wall-X 真实的需求。因为动作 token 虽然有自己的专家处理路径，但它并不是完全脱离上下文独立存在的。

## mask 和 position 也跟着变了

MoE 一旦进入 decoder，就不可能只影响线性层和 MLP，它还会反过来影响“哪些 token 应该彼此可见”、“它们的位置该怎么编号”。

Wall-X 在这方面也做了对应处理。

### 第一，某些 token 不再严格服从纯因果掩码

在 `_update_causal_mask(...)` 里，如果存在 `moe_token_types`，代码会把 type-1 token 区域单独找出来，然后把这块区域从纯 causal 约束里放开。

换句话说，在某些配置下，属于 flow / action 这类 expert 的 token，可以在它们自己的局部区域里双向可见，而不是只能单向看历史。

这一步非常重要。因为它说明 Wall-X 的动作 token，不是被简单塞进一个标准 next-token prediction 框架里，而是为了动作建模需求，改过可见性规则。

### 第二，位置编号也要重对齐

在 `_update_position_ids(...)` 里，代码还会根据 `ar_predict_token_positions` 和 `flow_mask` 去重算部分位置编号，让 flow token 的位置和前面的 AR token 区段对齐。

这一步背后的直觉其实也很清楚：

- 如果序列里混了不同语义阶段的 token
- 而这些 token 又要走不同 attention / routing 路径
- 那原始的简单顺序位置未必还是最合适的

所以位置也被当成可调的一部分，而不是完全固定不动。

### 第三，joint attention 需要自己那套 mask 处理

如果用了 `attention_moe=True`，Wall-X 还会额外走 `_update_joint_attention_mask_2d(...)`，专门把 action token 区域、padding 区域、AR predict token 区域这些关系重新编码到 2D mask 里。

这进一步说明：  
Wall-X 的 MoE 不是只在算子层“插一个专家模块”，它会顺带牵动一整套序列语义规则。

## 为什么说它的复杂度主要在“路由”和“序列语义”

到这里其实可以回头看一眼，Wall-X 的 MoE 为什么会比常见印象里的 MoE 更绕一点。

不是因为它的专家 MLP 特别神秘，而是因为它把 MoE 放进了多模态动作生成这条统一序列里。

一旦这样做，你就不得不同时回答这些问题：

- 哪些 token 属于哪个 expert
- 它们是按原始顺序处理，还是先 permute
- 它们应该共享还是拆开 norm
- attention 是共享投影，还是 expert-specific QKV/O
- 某些动作 token 之间要不要双向可见
- 它们的位置编号要不要重对齐

这些问题加起来，才是 Wall-X MoE 的真实复杂度来源。

所以如果有人问“Wall-X 的 MoE 核心难点在哪”，我会说答案不是某一个类名，而是：

> 它把专家路由、统一序列建模和动作语义这三件事绑在了一起。

## 把这一篇压成一句话

如果把这一篇的结论只压成一句话，那就是：

> Wall-X 的 MoE 不是只改 FFN，而是直接长进了 decoder layer：token 先按 `moe_token_types` 被分成不同 expert 路由，随后 norm、attention、MLP、mask 和 position 都会围绕这套路由一起变化。

这句话一旦立住，后面再去看：

- `permute / unpermute`
- `JointQwen2VLAttention`
- `start_indices / end_indices`
- flow token 的双向可见性

这些实现细节就都不会显得零散了。

下一篇我会继续沿着统一序列这条线往下讲，专门看 Wall-X 是怎么生成动作的。也就是：AR 和 flow 两条路径分别怎么走，`ActionProcessor` 在做什么，为什么推理时会出现 prefix prefill、KV cache 截断和 Euler/ODE 积分。

## 附：文中对应的关键源码位置

- MoE decoder layer：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L82)
- MoE 主干 `Qwen2_5_VLMoEModel`：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L284)
- `moe_token_types`、`start_indices`、`end_indices` 在顶层 forward 中的使用：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L1290)
- `TokenTypeRouter`：[`vla_mixin.py`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L34)
- `SparseMoeBlock`：[`vla_mixin.py`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L85)
- `_apply_norm_moe(...)`：[`vla_mixin.py`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L179)
- `_gated_residual(...)`：[`vla_mixin.py`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L356)
- `_update_position_ids(...)`：[`vla_mixin.py`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L444)
- `_update_joint_attention_mask_2d(...)`：[`vla_mixin.py`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L475)
- `_update_causal_mask(...)` 里对 type-1 token 的双向可见性处理：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L584)
- `JointQwen2VLAttention`：[`joint_attention.py`](/Users/sam/project/github/wall-x/wall_x/model/joint_attention.py#L43)
- expert-specific `q_proj / k_proj / v_proj / o_proj`：[`joint_attention.py`](/Users/sam/project/github/wall-x/wall_x/model/joint_attention.py#L77)
- joint attention 的 SDPA 路径：[`joint_attention.py`](/Users/sam/project/github/wall-x/wall_x/model/joint_attention.py#L132)
- joint attention 的 flash 路径：[`joint_attention.py`](/Users/sam/project/github/wall-x/wall_x/model/joint_attention.py#L525)
