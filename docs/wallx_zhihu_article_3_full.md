# Wall-X 的视觉主干和文本主干：它不是两套模型，而是一条合流链

上一篇我顺着 `infer_robochallenge.py` 讲了一条更偏运行时的链：Wall-X 是怎么把 `state`、`views`、`instruction` 这些现场输入，组装成模型真正接收的复合 batch 的。

但当你把这条入口链看明白之后，接下来一定会遇到一个更关键的问题：

模型内部到底是怎么组织的？

尤其是第一次看多模态模型的人，很容易在这里形成一个直觉：  
是不是有一套视觉模型专门跑图像，再有一套语言模型专门跑文本，最后把它们的结果拼起来？

Wall-X 不是这么跑的。

更准确地说，它确实有视觉主干，也确实有文本主干，但它们不是“两套模型各算各的，最后做晚融合”。Wall-X 的真实结构更接近这样一条链：

```text
图像 / 视频
-> 视觉编码器
-> 视觉 embedding
-> 回填进文本 token 序列
-> 统一进入 decoder
-> 输出 hidden states / logits / action 相关结果
```

所以这一篇我只回答一个问题：  
Wall-X 的视觉主干、文本主干，以及它们之间的合流点，到底在哪里。

## 顶层结构其实已经把答案写出来了

如果你直接看基础多模态模型 `Qwen2_5_VLForConditionalGeneration` 的初始化，会看到非常关键的三句：

```python
self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(...)
self.model = Qwen2_5_VLModel(config)
self.lm_head = nn.Linear(...)
```

这三句几乎已经把整个骨架写在脸上了：

- `self.visual` 是视觉编码器
- `self.model` 是文本 decoder 主干
- `lm_head` 负责把 decoder hidden states 投到词表空间

而在 Wall-X 自己的动作版模型 `Qwen2_5_VLMoEForAction` 里，这个结构也没有变掉，只是中间那句被换成了：

```python
self.model = Qwen2_5_VLMoEModel(...)
```

这件事很重要，因为它说明：

- 视觉入口没有被另起一套
- 多模态的合流方式没有被改写
- Wall-X 相对基础 Qwen2.5-VL 的主要改动，不在“有没有视觉分支”，而在“decoder 被怎么增强了”

所以如果先不展开 MoE，Wall-X 的顶层骨架完全可以先按“视觉编码器 + 文本 decoder + 输出头”来理解。

## 视觉主干是一套独立的 Vision Transformer

先看 `self.visual`。

它对应的类是 `Qwen2_5_VisionTransformerPretrainedModel`。如果把这个类的初始化拆开，会看到它的结构其实很规整：

- `patch_embed`
- `rotary_pos_emb`
- `blocks`
- `merger`

这基本就是一套标准 Vision Transformer 的变体组织方式。

### 第一层：`patch_embed`

视觉输入最开始不是 token，而是图像或视频 patch。

`patch_embed` 的职责，就是把原始视觉张量映射到模型的隐藏维度上。到这一步为止，视觉数据才第一次变成“能参与 Transformer 计算的序列表示”。

这也是为什么我上一篇会强调 `image_grid_thw` 很重要。因为视觉输入并不是只对应一个符号，它后面会被拆成一长串视觉位置单元。`patch_embed` 就是这个过程开始的地方。

### 第二层：视觉位置编码不是随手加的

接下来，视觉主干会调用：

```python
rotary_pos_emb = ops.rot_pos_emb(...)
window_index, cu_window_seqlens = ops.get_window_index(...)
```

这两步在 Wall-X 里非常关键。

第一步 `rot_pos_emb` 是在给视觉 token 生成旋转位置编码。  
第二步 `get_window_index` 是在为后面的 window attention 做索引重排准备。

这说明视觉分支不是“patch 之后直接全局 attention 一把梭”，而是会先把位置和窗口布局都准备好，再进入后续 block。

从源码结构上你也能看出来，视觉这条链是 Wall-X 自定义算子最集中介入的位置之一。因为多模态视觉 token 的位置关系和文本 token 不一样，它天然更需要额外的索引与布局处理。

### 第三层：视觉 blocks 不是纯全局 attention

视觉主干中间是一串 `Qwen2_5_VLVisionBlock`。

每个 block 的内部结构不复杂，本质上还是：

- `norm`
- `attention`
- `residual`
- `norm`
- `mlp`
- `residual`

但它有一个很值得注意的细节：视觉分支不是所有层都统一走全局 attention。

在 `Qwen2_5_VisionTransformerPretrainedModel.forward()` 里，每一层会根据 `layer_num` 是否落在 `fullatt_block_indexes` 里，来决定当前这一层到底用：

- 全量序列的 `cu_seqlens`
- 还是 window 划分后的 `cu_window_seqlens`

换句话说，视觉主干里是“部分 full attention + 部分 window attention”的混合结构，而不是每一层都同构。

这也解释了为什么视觉侧会需要 `get_window_index(...)` 这类额外算子。因为 window 布局本身就是这条链的重要一部分，不是某个边角优化。

### 第四层：视觉 attention 后端和文本不一样

上一篇我已经提过，当前推理配置下：

- 文本主干走 `sdpa`
- 视觉主干走 `flash_attention_2`

把这个判断放到视觉内部再看，就更容易理解了。因为视觉分支的 attention 类本身就是按 backend 选的：

- `eager`
- `flash_attention_2`
- `sdpa`

而当前配置选择的是 `flash_attention_2`。所以视觉 block 在推理时，注意力内核会落到 `flash_attn_varlen_func(...)` 这一路。

这说明一件事：  
Wall-X 的视觉主干虽然是独立分支，但它并不是一套特殊得看不懂的结构。它还是 ViT 式样的 block 栈，只是针对视觉 token 的位置组织做了更多工程处理。

### 最后一层：`merger`

视觉 blocks 跑完之后，并不会直接把中间 token 原样交给语言模型，而是还要过一层 `merger`。

这层的作用可以粗略理解成：把视觉分支处理后的 patch 级表示，整理成更适合送入 LLM 序列的视觉 embedding。

最后视觉主干还会做一次 `reverse_indices = torch.argsort(window_index)`，把前面为了 window attention 重排过的 token 顺序还原回来。

这一步也很关键，因为只有把顺序还原回来，后面才能和文本序列里的 image/video token 一一对齐。

## 文本主干本质上还是 decoder 栈

再看 `self.model`。

基础模型里对应的是 `Qwen2_5_VLModel`。它的初始化也很直白：

- `embed_tokens`
- `layers = nn.ModuleList([...])`
- `norm`
- `rotary_emb`

也就是说，它本质上就是一套 decoder-only Transformer。

这条链并没有因为它是多模态模型就变成另一种结构。视觉输入进来以后，也不是切到一套“跨模态专用大模块”，而是最后统一进入这套 decoder 栈。

这里有两个特别关键的点。

### 第一，文本主干负责统一处理整个序列

一旦视觉特征被回填进 token 序列，后面的 decoder 看到的就不再是“纯文本 token”，而是一条混合序列：

- 一部分位置是普通文本 embedding
- 一部分位置是视觉 embedding
- 在 Wall-X 动作版里，还可能有 proprioception embedding、action embedding 等额外内容

但从 decoder 的视角看，这些东西一旦进入 `inputs_embeds`，后续就是同一种 hidden state 流。

所以真正把多模态信息整合起来的，不是某个单独的拼接函数，而是后面的整条 decoder 计算。

### 第二，位置编码也是统一在这里接上的

文本主干在 forward 里会先准备：

- `position_ids`
- `causal_mask`
- `position_embeddings = self.rotary_emb(...)`

其中最容易被忽略的是 `position_ids` 这件事。

在纯文本模型里，位置一般就是一条简单递增的 1D 序列。  
但在 Qwen2.5-VL 这条链里，如果序列里混了视觉 token，位置就没这么简单了。

对应逻辑在顶层模型的 `get_rope_index(...)` 里写得很清楚：

- 视觉部分使用 3D RoPE
- 文本部分使用 1D 语义上的位置递增

更准确地说，源码里会为每个序列位置准备 `(time, height, width)` 三个维度的位置编号。对于纯文本位置，这三个维度会退化成同样的递增序列；对于视觉位置，则会根据 `image_grid_thw` 或 `video_grid_thw` 真正展开成时空网格。

这就是多模态位置编码在这条链里的落点。

## 真正的合流点，不在模型初始化，而在 `forward()`

很多人看模型装配时，看到 `self.visual` 和 `self.model` 这两句，会自然以为“合流已经在这里完成了”。

其实不是。

真正的合流点发生在顶层 `forward()` 里，而且非常具体。

先看第一步：

```python
inputs_embeds = self.model.embed_tokens(input_ids)
```

这意味着一开始，整条序列先被当成普通 token 序列做 embedding。此时哪怕 prompt 里已经有 `<|image_pad|>`，它们也还只是普通 token id 对应出来的 embedding。

接下来，如果存在图像输入：

```python
image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
```

视觉分支就会真正跑起来，把图像编码成一串视觉 embedding。

然后最关键的一步出现了：

```python
mask = input_ids == self.config.image_token_id
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

这几行其实就是 Wall-X 多模态合流的核心。

它的语义非常直接：

- 先在文本序列里找到所有 image token 的位置
- 再把这些位置上的原始 token embedding，替换成真正的视觉 embedding

视频也是同样的逻辑，只不过匹配的是 `video_token_id`。

所以不要把多模态合流想象成“先跑完整个视觉模型，再和文本模型输出拼一层”。  
真实情况是：视觉特征在 embedding 层就被塞回到了文本序列里。

## 为什么我说它不是“两套模型各跑各的”

现在可以更精确地回答这个问题了。

Wall-X 确实有两个子模块：

- `self.visual`
- `self.model`

但它们的关系不是平级晚融合，而是前后串联：

1. 视觉分支先把图像转成视觉 embedding  
2. 这些视觉 embedding 替换掉序列里的视觉占位 token  
3. 替换后的整条混合序列统一送进 decoder  

也就是说，真正承担多模态语义整合工作的，是后面的 decoder。

这和很多人想象中的“双塔结构”很不一样。双塔更像是两边各自编码完，再做相似度或拼接；而 Wall-X 这里更像是“视觉先转写成语言模型能接受的 token 级表示，然后进入同一条上下文链”。

这也是为什么你在后面分析 cache、生成、MoE、action token 时，始终都要以“统一序列”作为视角，而不是把视觉和文本分成两个完全独立的推理系统。

## 合流之后，decoder 才是总主干

视觉 embedding 被回填以后，顶层模型接着做的事就很顺了：

```python
outputs = self.model(
    input_ids=None,
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    ...
)
hidden_states = outputs[0]
logits = self.lm_head(hidden_states)
```

这条链说明两件事。

第一，进入 decoder 之后，模型已经不再区分“这段 hidden state 原来是图像来的还是文本来的”，因为它们都已经统一成了序列上的 embedding。  
第二，从总体执行路径看，decoder 仍然是 Wall-X 的总主干。

所以如果后面你要评估推理成本，不能只看视觉分支算了几层 block，还要看它最终把序列拉长到什么程度，以及统一序列进入 decoder 之后带来的整体负担。

这也是为什么在多模态 LLM 里，视觉编码器本身不一定是全部瓶颈。很多时候真正重的，反而是视觉 token 回填之后，后面整条 decoder 链都要一起处理更长的上下文。

## Wall-X 动作版沿用了同样的合流方式

到这里顺手说一句 Wall-X 自己的动作版。

在 `Qwen2_5_VLMoEForAction` 里，顶层仍然是：

- `self.visual`
- `self.model`
- `self.lm_head`

区别只在于 `self.model` 不再是基础的 `Qwen2_5_VLModel`，而是 `Qwen2_5_VLMoEModel`。  
也就是说，视觉分支和多模态合流方式本身并没有被推翻，变化主要发生在 decoder 内部。

这也是为什么我把第 4 篇单独留给 MoE。因为你只有先把“视觉怎么进来、文本怎么接住、两者在哪里合流”看清楚，后面再讲 MoE 插在 decoder 的哪一层，才不会乱。

## 从性能视角再看这条合流链

如果这时把视角重新切回“推理优化”，你会发现这篇文章其实给了几个非常现实的判断。

第一，视觉分支的成本不只在 attention kernel，还在 patch、window index、位置编码和 merger 这整条链。  
第二，视觉 token 一旦回填进文本序列，后面的 decoder 会把它们和普通 token 一起处理，所以上下文长度变化本身就是成本来源。  
第三，Wall-X 的多模态复杂度更多体现在“怎么组织统一序列”，而不是“有没有一个神秘的跨模态总控模块”。

这几个判断都很重要。因为只有先看到这条总链，你后面分析 MoE、action token、KV cache 的时候，才知道这些东西是加在什么地方上的。

## 把这一篇压成一句话

如果把这一篇的核心结论只压成一句话，那就是：

> Wall-X 确实有视觉主干和文本主干，但它们不是两套模型各跑各的；视觉分支先把图像编码成 token 级 embedding，再回填进文本序列，最后统一由 decoder 主干完成后续建模。

这句话一旦立住，后面的很多事情都会顺下来：

- 为什么视觉 token 会影响 decoder 长度
- 为什么位置编码要做 3D/1D 混合
- 为什么动作 token、MoE token type 最终也都要放在统一序列里看

下一篇我就沿着这条统一序列继续往下走，专门讲 Wall-X 的 MoE 到底加在了哪里。也就是：它不是在视觉分支上加专家，也不是另起一套专家网络，而是把 decoder layer 本身做了 MoE 化改造。

## 附：文中对应的关键源码位置

- 基础多模态模型顶层装配：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L1774)
- Wall-X 动作版顶层装配：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L952)
- 视觉主干 `Qwen2_5_VisionTransformerPretrainedModel`：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L483)
- 视觉 `patch_embed`、`blocks`、`merger`：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L495)
- 视觉 `rot_pos_emb` 和 `get_window_index`：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L607)
- 视觉 block 与 attention backend：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L410)
- 文本主干 `Qwen2_5_VLModel`：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L1313)
- `embed_tokens` 和 decoder layers：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L1319)
- 多模态位置索引 `get_rope_index(...)`：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L1804)
- 图像 embedding 回填进文本序列：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L2103)
- 视频 embedding 回填进文本序列：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L2126)
- 统一进入 decoder 并经 `lm_head` 输出：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L2184)
