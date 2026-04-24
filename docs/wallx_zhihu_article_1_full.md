# 别一上来就谈 CUDA：Wall-X 推理优化前，先搞清楚它到底是什么模型

很多人第一次看 Wall-X 这类项目时，第一反应都是先找自定义 CUDA 算子，然后顺手给它贴一个标签：这是个深度魔改的推理项目。

这个判断不准确。

Wall-X 除了 MoE 和 RoPE 相关算子，没有别的自定义算子，其他算子基本都还在 PyTorch 技术栈里。你如果要优化它的推理速度，先得搞清楚它的模型结构，然后才有优化空间。否则你很容易看着源码发问：为什么它没有 MLA，只有 MHA/GQA 和 FlashAttention？

这不是枝节问题，而是起点问题。因为你对模型结构的第一判断，会直接决定后面优化的方向：你是会去盲目翻 CUDA kernel，还是先去识别真正的热点路径；你是会误以为它改了注意力数学本身，还是能看出来它的大部分复杂度其实在 MoE 路由、视觉 token 组织和动作生成链上。

所以这篇文章不先讲训练，也不先讲 flow action，更不先讲某个 kernel 的实现细节。先只回答一个最基础的问题：Wall-X 到底是什么模型，它的推理主干到底长什么样。

## 先把边界看清楚

从工程角度看，一个模型项目里有没有自定义 CUDA 算子，和“这个模型的核心复杂度在哪里”，并不是一回事。

Wall-X 很适合用来说明这个问题。因为它表面上确实有一层很容易吸引注意力的东西：仓库里有 `csrc/ops.cu`，有 `PYBIND11_MODULE`，有若干导出给 Python 的底层算子。很多人到这里就会很自然地下一个判断：这个项目的主要难点肯定都在自定义 kernel 里。

但如果你顺着调用链继续往上看，就会发现事情不是这样。模型顶层仍然是一个很标准的多模态结构：有视觉编码器，有文本 decoder，有 embedding 替换，有 decoder stack，有 `lm_head`，有 `generate()`。只是其中部分性能敏感路径，被额外下沉成了自定义算子。

这意味着一个很实际的阅读策略变化。对于 Wall-X 这种项目，最有效的理解路径不是“先看 kernel，再反推模型”，而是“先把顶层前向结构看明白，再回头确认哪些局部路径值得下钻”。这样你才能区分：

- 哪些代码是在描述模型主干
- 哪些代码是在描述性能优化
- 哪些代码是在描述 Wall-X 自己新增的任务能力

这三者如果一开始混在一起看，基本一定会乱。

## Wall-X 到底有哪些自定义算子

如果只看 `csrc/ops.cu` 暴露出来的接口，Wall-X 的自定义算子其实非常集中。

```cpp
m.def("asym_dual_gmm", &AsymmetricDualExpertGemm, "Asymmetric Dual Expert Grouped GEMM.");
m.def("permute", &moe_permute_topK_op, "Token permutation kernel");
m.def("unpermute", &moe_recover_topK_op, "Token un-permutation kernel");
m.def("unpermute_bwd", &moe_recover_topK_bwd_op, "Token un-permutation backward kernel");
m.def("rope", &launch_multimodal_rope_forward, "Multimodal RoPE forward kernel");
m.def("rope_bwd", &launch_multimodal_rope_backward, "Multimodal RoPE backward kernel");
m.def("rope_index", &get_rope_index, "Get RoPE index kernel");
m.def("rot_pos_emb", &fused_rot_pos_emb_cuda, "Fused Rotary Position Embedding kernel");
m.def("get_window_index", &get_window_index_cuda, "Get window index kernel");
```

你会发现它基本被分成三组：

第一组是 MoE 相关，核心是 token 重排和 grouped GEMM。  
第二组是 RoPE 和多模态位置编码相关，核心是把视觉 token 和文本 token 的位置关系处理好。  
第三组是视觉窗口索引相关，核心是把图像或视频切成模型能接受的布局。

这个边界很重要。因为这说明 Wall-X 的自定义算子是“局部性能关键路径下沉”，不是整套模型重新实现。也就是说，模型主体并没有离开 PyTorch / Transformers 的主干范式。

## 它为什么不是重度自定义推理框架

判断一个项目是不是“重度自定义推理框架”，不能只看它有没有自定义 CUDA，而要看它有没有把最核心的计算路径整个换掉。

Wall-X 没有这么做。

你在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py` 里会看到，它的 attention 仍然是标准的 QKV 投影，然后再根据不同后端走不同实现：

- 文本分支可以落到 `torch.nn.functional.scaled_dot_product_attention`
- 视觉分支可以落到 `flash_attn_varlen_func`

换句话说，模型主体的组织方式还是非常“Transformer 原教旨”的：embedding、attention、MLP、norm、lm head，这些核心部件并没有被整套换掉。

所以更准确的描述不是“Wall-X 是一个全自定义算子模型”，而是：

> Wall-X 是一个以 Qwen2.5-VL 为主干、在 MoE 路由、视觉位置编码和动作预测链上做了定向增强的多模态模型。

这句话很关键。因为它直接决定了你后面怎么优化：

- 如果它是全自定义算子模型，你的第一反应应该是看 kernel 和内存访存
- 但它不是
- 所以你应该先看调用链、token 布局、视觉和文本怎么合流、MoE 怎么插进去

只有把这些讲清楚，后面的性能分析才不会跑偏。

## 为什么它没有 MLA，只有标准 QKV + GQA

很多人看多模态大模型源码时，最容易冒出来的一个问题就是：它为什么不是 MLA？

这个问题在 Wall-X 里其实很好回答，因为它的文本 attention 根本就不是 MLA 路线。

在 `wall_x/model/qwen2_5_based/configuration_qwen2_5_vl.py` 里，`num_key_value_heads` 的注释已经把逻辑说得很直白了：`num_key_value_heads=num_attention_heads` 是 MHA，`num_key_value_heads=1` 是 MQA，其余情况就是 GQA。

对应到实现上，`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py` 里还是标准的：

```python
self.q_proj = nn.Linear(...)
self.k_proj = nn.Linear(...)
self.v_proj = nn.Linear(...)
```

然后再通过 `repeat_kv(...)` 把 K/V 扩展到多头需要的形状。

这意味着什么？

意味着它的注意力结构仍然是 QKV + GQA，而不是 MLA。  
也意味着你后面如果要做优化，应该围绕它真实存在的结构来想办法，而不是围绕一个并不存在的 attention 变体去脑补改造方向。

说得更直白一点：  
没有 MLA，不代表这个模型“低级”；只说明它继承的是 Qwen2.5-VL 这一路的注意力结构。

## 文本为什么走 SDPA，视觉为什么走 FlashAttention2

Wall-X 不是“所有分支统一一个 attention 后端”的模型，它是分路的。

在推理配置里，文本主干和视觉主干被明确分开设置：

- 文本分支：`_attn_implementation = "sdpa"`
- 视觉分支：`vision_config._attn_implementation = "flash_attention_2"`

这在 `wall_x/infer/infer_config.py` 和 `scripts/infer_robochallenge.py` 里都能看到。

这件事再一次说明，Wall-X 的复杂度不是“所有东西都被魔改成一套新算子”，而是“不同分支按任务特性选择不同后端”。文本 decoder 走 SDPA，视觉 encoder 走 FlashAttention2，本质上还是在 PyTorch 体系里做工程级组合。

这也顺手回答了另一个常见误解：  
你不能因为看到某一层用了 FlashAttention，就推断整个模型的注意力数学都变了；同样，也不能因为它没有 MLA，就觉得它没有优化空间。它的优化空间，更多来自结构理解之后的路径识别，而不是只盯一个名字。

## 它的顶层结构其实很清楚

如果把顶层模型装配代码直接展开，你会看到两句非常朴素的话：

```python
self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(...)
self.model = Qwen2_5_VLModel(config)
```

这就已经把 Wall-X 的骨架讲明白了：

- `self.visual` 负责视觉编码
- `self.model` 负责文本 decoder
- `lm_head` 负责最终 vocab 输出

真正的多模态合流发生在 `forward()` 里：先把文本 token embed 出来，再把图像或视频特征按 token 位置 scatter 回去，最后统一送进 `self.model(...)`。

也就是说，Wall-X 不是“两套模型跑完再拼一下”，而是“视觉先变成 embedding，再回填进文本序列，然后整体进入 decoder”。

这个结构判断一旦建立，你后面看 MoE、看 action token、看 flow route，都会顺很多。因为你知道它不是一个完全分叉式的系统，而是一条以文本 decoder 为主轴的多模态链。

## 真正该先盯哪几条链路

如果你接下来真的想优化 Wall-X 的推理速度，我建议优先级是这样的：

1. 先搞清楚推理入口和输入张量怎么组织
2. 再搞清楚视觉分支和文本分支怎么合流
3. 再看 decoder 里 MoE 是插在哪一层
4. 最后才去看自定义 CUDA 算子到底占了多少比重

这个顺序看起来有点“反直觉”，但实际上最省时间。因为你只有先把模型结构看透，才能判断某个 kernel 到底是不是瓶颈，某个后端切换到底值不值得做，某条链路到底是在做任务增强还是在做性能优化。

否则你就很容易陷入一种典型困惑：明明在找“推理优化”，最后却一直在问“为什么它没有 MLA，只有 MHA/GQA 和 FlashAttention”。

这恰恰说明你先看结构是对的。

下一篇我会顺着调用顺序继续往下走，直接从推理入口开始，把 Wall-X 的数据加载、prompt 构造和模型加载链路串起来。这样你就能看到它到底是怎么把一个现场 observation 变成模型输入的。

## 附：文中对应的关键源码位置

- 自定义算子导出总表：[`csrc/ops.cu`](/Users/sam/project/github/wall-x/csrc/ops.cu#L11)
- 文本 attention 结构与 `repeat_kv`：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L834)
- `q_proj / k_proj / v_proj` 定义：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L879)
- `num_key_value_heads` 对 MHA/MQA/GQA 的说明：[`configuration_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/configuration_qwen2_5_vl.py#L72)
- 文本分支的 SDPA 调用：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L1203)
- 视觉分支的 FlashAttention 调用：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L266)
- 顶层 `self.visual` / `self.model` 装配：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L1787)
- 图像特征替换进文本序列的位置：[`modeling_qwen2_5_vl.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py#L2107)
- 当前推理配置里文本与视觉 attention 后端的设置：[`infer_config.py`](/Users/sam/project/github/wall-x/wall_x/infer/infer_config.py#L576) 和 [`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L465)
