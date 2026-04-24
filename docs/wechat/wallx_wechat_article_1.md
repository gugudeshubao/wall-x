# 别一上来就谈 CUDA：先看清 Wall-X 到底是什么模型

> 很多人第一次看 Wall-X 代码，第一反应都是先找自定义 CUDA 算子。这个动作不算错，但如果一开始就把注意力全压在 kernel 上，你后面大概率会越看越乱。

**TL;DR**
- Wall-X 不是一个“整套靠自定义算子重写”的模型。
- 除了 MoE、RoPE 和视觉 window/index 相关算子，主体仍然是 PyTorch 技术栈。
- 它不是 MLA 路线，文本主干还是标准 `QKV + GQA`。
- 当前常见推理配置下，文本走 `SDPA`，视觉走 `FlashAttention2`。
- 如果想优化推理，先看结构，再看 kernel。

## 一、为什么不能一上来就盯 CUDA

看一个模型项目时，最容易犯的错误，就是看到 `csrc/ops.cu`、`PYBIND11_MODULE`、若干自定义 kernel，然后脑子里自动补一句：

“这个项目的核心复杂度肯定都在 CUDA 里。”

Wall-X 不是这样。

它的确有自定义算子，但这些算子并没有覆盖整套模型主干。更准确地说，Wall-X 的自定义算子主要集中在几条性能敏感路径上，而不是把整套多模态建模框架都重写了一遍。

所以理解它的更好顺序不是：

1. 先看 CUDA
2. 再反推模型

而是：

1. 先看模型结构
2. 再确认哪些局部路径下沉成了 CUDA

这个顺序一旦反过来，很多问题你都会问偏。比如：

- 它为什么没有 MLA？
- 它的 attention 为什么看起来还是标准 QKV？
- 为什么视觉分支又在用 FlashAttention？

这些问题本身没错，但它们都建立在同一个前提上：你默认 Wall-X 应该是一套“从注意力到底层 kernel 都被完全重写”的模型。

这个前提不成立。

## 二、Wall-X 到底有哪些自定义算子

如果只看导出给 Python 的接口，Wall-X 的自定义算子其实非常集中，大致就是这几类：

```cpp
asym_dual_gmm
permute
unpermute
unpermute_bwd
rope
rope_bwd
rope_index
rot_pos_emb
get_window_index
```

它们基本可以分成三组：

- MoE 相关：`asym_dual_gmm`、`permute`、`unpermute`
- RoPE / 多模态位置编码相关：`rope`、`rope_index`、`rot_pos_emb`
- 视觉窗口索引相关：`get_window_index`

这说明什么？

说明 Wall-X 的自定义算子是“局部关键路径下沉”，不是“模型主体换了一套框架”。

也就是说，如果把这些算子先放到一边，剩下的大部分模型组织方式，你还是能按一套比较标准的 PyTorch / Transformers 风格去理解。

## 三、为什么说它不是重度自定义推理框架

判断一个项目是不是“重度自定义推理框架”，关键不是看它有没有 CUDA 扩展，而是看它有没有把模型最核心的计算主干整套换掉。

Wall-X 没有这么做。

它的主体路径里，仍然能看到这些非常熟悉的部件：

- 视觉编码器
- 文本 decoder
- embedding 替换
- decoder layers
- `lm_head`
- `generate()`

attention 核本身也没有换成什么全新的数学形式。文本分支依然可以落到 `scaled_dot_product_attention`，视觉分支依然可以落到 `flash_attn_varlen_func`。

所以更准确的描述应该是：

> Wall-X 是在 Qwen2.5-VL 这条多模态主干上，围绕 MoE、位置编码和动作生成做了定向增强。

这句话很关键。因为它直接影响你后面的优化思路。

如果你把它想成“全自定义推理系统”，你自然会先去盯 kernel。  
但如果你把它理解成“标准多模态主干 + 局部增强”，那你就会先去看调用链和结构分工。

后者更接近事实。

## 四、它为什么没有 MLA，只有标准 QKV + GQA

很多人一看到“大模型推理优化”，脑子里第一反应就是 MLA。

但 Wall-X 这条线不是 MLA。

它的文本 attention 结构仍然是很标准的：

```python
q_proj
k_proj
v_proj
```

然后再根据 `num_key_value_heads` 和 `repeat_kv` 做 GQA。

换句话说，它的文本主干更接近：

- `QKV + GQA`
- 而不是 MLA

这里一定要避免一个误解：

> 没有 MLA，不代表这个模型“低级”。

它只说明一件事：Wall-X 继承的是 Qwen2.5-VL 这一路的主干结构。  
你后面做优化，也应该围绕这个真实存在的结构去想办法，而不是围绕一个并不存在的 MLA 去脑补。

## 五、文本为什么走 SDPA，视觉为什么走 FlashAttention2

Wall-X 还有一个很容易被忽略的点：

它不是所有分支统一用同一种 attention 后端。

当前常见推理配置下：

- 文本主干走 `SDPA`
- 视觉主干走 `FlashAttention2`

这件事看起来只是实现细节，但实际上非常重要。

因为它进一步说明：

- Wall-X 的复杂度不是“全部换成一套神秘 attention”
- 而是“不同分支按自己的场景选后端”

文本这边更像标准 decoder 栈。  
视觉这边则是 Vision Transformer 里的高效 attention 路线。

所以你如果只用一个词去概括整个模型，比如“它就是 FlashAttention 模型”或者“它就是自定义 attention 模型”，都不准确。

## 六、如果你真要优化它，第一步该看什么

如果你的目标是优化 Wall-X 推理速度，我会建议你按这个顺序看：

1. 先搞清楚推理入口和输入张量怎么组织
2. 再搞清楚视觉分支和文本分支怎么合流
3. 再看 decoder 里的 MoE 到底插在哪里
4. 最后才去看自定义 CUDA 算子到底占了多少比重

这个顺序的好处是，你能先找到“结构性瓶颈”，而不是一开始就陷进某个局部 kernel 的细节里。

否则很容易出现一种典型情况：

你在做推理优化，结果一路都在追问：

- 为什么它没有 MLA？
- 为什么它只有 MHA/GQA 和 FlashAttention？

这恰恰说明你先看结构是对的。

## 七、这一篇真正想立住的判断

如果把这篇文章压成一句话，那就是：

> Wall-X 不是一个“除了 CUDA 什么都不用看”的模型。它的自定义算子主要服务于 MoE 和位置编码，主体仍然是 Qwen2.5-VL 这条多模态主干。想优化它，第一步不是改 kernel，而是先看清结构。

下一篇我会顺着这条思路往下走，直接从推理入口开始，看它的数据加载、prompt 构造和模型加载到底是怎么接起来的。

