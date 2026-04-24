# Wall-X 的视觉主干和文本主干：它不是两套模型，而是一条合流链

> 很多人第一次看多模态模型时，总会下意识地把它理解成“两套模型各跑各的，最后把结果拼起来”。Wall-X 不是这么组织的。

**TL;DR**
- Wall-X 的顶层确实有 `self.visual` 和 `self.model` 两套子模块。
- 但它们不是平级晚融合，而是前后串联。
- 图像先变成视觉 embedding，再回填进文本 token 序列。
- 真正承担整体建模工作的，还是后面的 decoder 主干。

## 一、顶层结构已经把答案写出来了

如果你直接看顶层模型装配，最关键的就是这三句：

```python
self.visual = ...
self.model = ...
self.lm_head = ...
```

这三句其实已经把骨架讲得很清楚了：

- `self.visual`：视觉编码器
- `self.model`：文本 decoder 主干
- `lm_head`：词表输出头

在 Wall-X 的动作版里，这个大框架并没有变。变化主要在于：

- `self.model` 不再是基础 decoder
- 而是被替换成了带 MoE 和动作能力的版本

所以理解 Wall-X 的第一步，不是把视觉和文本看成完全独立的两套系统，而是先接受一个更准确的事实：

> 它是一条“视觉编码 -> 序列回填 -> 统一 decoder 建模”的链。

## 二、视觉主干是一套独立的 Vision Transformer

先看 `self.visual`。

它内部最核心的组件很规整：

- `patch_embed`
- `rotary_pos_emb`
- `blocks`
- `merger`

你可以把它理解成一套带多模态位置和窗口处理的 Vision Transformer。

### 1. `patch_embed`

图像或视频一开始当然不是 token。  
`patch_embed` 的作用，就是先把视觉输入切成 patch 并映射到隐藏空间。

从这一步开始，视觉信息才第一次变成“能参与 Transformer 计算的序列表示”。

### 2. 视觉位置编码不是随手加的

视觉主干里接下来会调用两步很关键的东西：

- `rot_pos_emb`
- `get_window_index`

第一步是在给视觉 token 准备旋转位置编码。  
第二步是在为窗口级 attention 做索引重排。

这说明视觉分支不是“patch 完直接全局 attention 一把梭”，而是先把位置和窗口布局都整理好，再进入 block。

### 3. 视觉 block 不是每层都纯全局 attention

视觉分支中间是一串 `VisionBlock`。  
每个 block 本质上还是：

- norm
- attention
- residual
- norm
- mlp
- residual

但有个很重要的细节：不是每一层都用同一种 attention 范围。

Wall-X 这里会混合：

- full attention
- window attention

所以 `get_window_index(...)` 这类算子并不是边角优化，而是视觉主干正常工作的一部分。

### 4. 视觉 attention 后端和文本不一样

当前常见推理配置里：

- 视觉走 `FlashAttention2`
- 文本走 `SDPA`

这件事放到视觉内部再看，会更容易理解。  
因为视觉侧本来就更适合走高效的 varlen attention 路线。

### 5. 最后还有一层 `merger`

视觉 blocks 跑完之后，并不会直接把 patch 级表示原样交给语言模型。  
它还会经过一层 `merger`，把视觉特征整理成更适合回填到 LLM 序列里的视觉 embedding。

同时，前面为了 window attention 做过的 token 重排，也会在这里被还原回原始顺序。

## 三、文本主干本质上还是 decoder 栈

再看 `self.model`。

基础版本里的结构其实非常标准：

- `embed_tokens`
- `layers`
- `norm`
- `rotary_emb`

这说明 Wall-X 的文本主干本质上还是一套 decoder-only Transformer。

也就是说，多模态并没有让它变成另一种完全不同的骨架。  
图像、状态、动作这些东西，最终还是会进入同一条 decoder 链。

## 四、真正的合流点不在初始化，而在 `forward()`

很多人看到 `self.visual` 和 `self.model`，会自然以为“合流已经在模型初始化里完成了”。  
其实不是。

真正的合流发生在 `forward()` 里，而且非常具体。

先是文本 token 做 embedding：

```python
inputs_embeds = self.model.embed_tokens(input_ids)
```

这时候哪怕 prompt 里已经有 `<|image_pad|>`，它们也还只是普通 token 对应出来的 embedding。

然后，如果有图像：

```python
image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
```

视觉分支开始真正工作，把图像编码成视觉 embedding。

接下来最关键的一步出现了：

```python
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

它的含义很直接：

- 先找到文本序列里 image token 的位置
- 再把这些位置上的普通 token embedding，替换成真正的视觉 embedding

视频也是完全一样的逻辑。

这就是 Wall-X 多模态合流的真正位置。

## 五、为什么我说它不是“两套模型各跑各的”

现在可以把这个问题说得更准确了。

Wall-X 的确有两个子模块：

- `self.visual`
- `self.model`

但它们的关系不是：

- 各自编码完
- 最后在高层做晚融合

而是：

1. 视觉先编码成 embedding
2. 这些 embedding 被回填进统一序列
3. 后面的 decoder 统一处理整条混合序列

所以真正负责“把视觉和文本一起建模”的，其实是后面的 decoder。

这也是为什么你后面看 cache、MoE、action token 时，始终都应该以“统一序列”作为视角，而不是把视觉和文本拆成两个完全独立的系统。

## 六、Wall-X 动作版沿用了同样的合流方式

到了动作版模型，这条合流方式并没有被推翻。

顶层仍然是：

- `self.visual`
- `self.model`
- `lm_head`

变化主要发生在：

- `self.model` 被替换成 `Qwen2_5_VLMoEModel`

也就是说，视觉入口和多模态合流方式本身是稳定的，变化主要发生在 decoder 内部。

这也正是为什么下一篇专门讲 MoE 会很自然。因为你只有先把“图像是怎么进来的、视觉特征是怎么回填的、序列是怎么统一的”看清楚，后面再讲 expert 路由才不会乱。

## 七、从性能视角再看一遍这条链

如果你把视角重新切回“推理优化”，这篇文章其实给了你三个非常现实的判断：

- 视觉分支的成本不只在 attention kernel，还在 patch、window index、位置编码和 merger
- 视觉 token 一旦回填进文本序列，后面的 decoder 成本也会整体抬高
- 多模态模型的复杂度很多时候不在“有没有双塔”，而在“统一序列怎么组织”

所以如果你后面要 profile Wall-X，不能只盯视觉编码器，也不能只盯 decoder，而要看两者交接之后整条序列的长度和结构。

## 八、这篇文章真正要立住的判断

如果把这篇压成一句话，那就是：

> Wall-X 确实有视觉主干和文本主干，但它们不是两套模型各跑各的；视觉分支先把图像编码成 token 级 embedding，再回填进统一序列，最后由 decoder 主干完成整体建模。

下一篇我就沿着这条统一序列往下讲：Wall-X 的 MoE，到底加在了哪里。

