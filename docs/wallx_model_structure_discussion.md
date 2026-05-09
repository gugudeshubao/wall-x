# wall-x 模型结构讨论

这份文档只整理当前对 `wall-x` 模型结构的讨论结论，不改写系列文章，也不追求正式发文口吻。目标是把几个关键判断先压实，后面如果要写成文章，再基于这里展开。

---

## 一句话判断

如果只看“模型数学本体”，`wall-x` 自己做的结构创新其实不算特别多。  
它更像是：

> 在 `Qwen2.5-VL` 这套现成多模态主干上，叠了一层更偏机器人 / VLA 场景的 `MoE`、`action` 和 runtime 组织。

也就是说，它不是那种“从注意力公式到网络骨架都重写”的模型。

---

## 一、基础主干：大部分还是 Qwen2.5-VL

先看最上层骨架。

在 [`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py) 里，`Qwen2_5_VLMoEForAction` 的核心初始化仍然是：

- `self.visual`
- `self.model`
- `self.lm_head`

对应位置见 [`modeling_qwen2_5_vl_act.py#L952`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L952)。

这意味着：

- 视觉部分还是那套 `ViT` 式的视觉编码器
- 文本主干还是 decoder-only Transformer
- 图像 token 仍然是先编码成 embedding，再回填进统一序列
- 最终的大主干仍然是 decoder

所以如果问题是：

> `wall-x` 是不是另起炉灶发明了一套全新的多模态模型？

答案基本是否定的。

---

## 二、模型结构位置分析

如果只按“源码里的位置”来拆，`wall-x` 的结构可以压成下面这条链：

```text
input_ids / pixel_values / proprioception / moe_token_types
-> token embedding
-> visual(image/video -> visual embeds)
-> 把 visual / proprio / action embeds 回填进统一序列
-> Qwen2_5_VLMoEModel(decoder stack with MoE)
-> 分叉：
   - VQA: lm_head -> logits / generate
   - Flow Action: action_preprocessor + ODE / Euler -> action
```

对应的源码位置大致是：

- 顶层动作模型骨架：[`modeling_qwen2_5_vl_act.py#L773`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L773)
- 顶层三件套 `self.visual / self.model / self.lm_head`：[`modeling_qwen2_5_vl_act.py#L952`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L952)
- MoE 主干 `Qwen2_5_VLMoEModel`：[`modeling_qwen2_5_vl_act.py#L284`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L284)
- Flow Action 路径：[`modeling_qwen2_5_vl_act.py#L1959`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L1959)

这里最值得记住的，不是每个类名，而是三个“位置判断”：

1. 视觉分支不是最终主干，它只是先把图像变成 embedding。
2. 真正的总主干仍然是 decoder，也就是 `self.model`。
3. `VQA` 和 `Flow Action` 不是两套前向系统，而是共用前半段，在顶层任务头处分叉。

---

## 三、对第 3 篇问题的回答

如果对应“视觉主干、文本主干、合流点到底在哪里”这个问题，当前最简洁的回答可以写成：

### 1. 视觉主干在哪里

视觉主干就是 `self.visual`，也就是 `Qwen2_5_VisionTransformerPretrainedModel`。它负责：

- patch / window 级视觉编码
- 视觉位置编码
- blocks 级 attention / MLP
- 最后产出可回填到统一序列中的视觉 embedding

### 2. 文本主干在哪里

文本主干就是 `self.model`。在 `wall-x` 里，它已经不是基础版 `Qwen2_5_VLModel`，而是换成了 `Qwen2_5_VLMoEModel`，但本质上仍然是一条 decoder-only 主干。

### 3. 真正的合流点在哪里

真正的合流点不在模型初始化，而在顶层 `forward()` / `generate_flow_action()` 的 embedding 处理阶段：

- 先做 `input_ids -> token embedding`
- 再让 `visual` 跑出 image/video embedding
- 再把 image/video/proprio/action 这些 embedding 回填进序列里
- 最后统一送入 `self.model`

所以这一篇最核心的判断是：

> `wall-x` 不是“两套模型各跑各的”，而是“视觉先编码成 embedding，再回填进统一序列，最后统一由 decoder 建模”。

---

## 四、对第 4 篇问题的回答

如果对应“MoE 到底加在了哪里”这个问题，当前最准确的回答是：

### 1. MoE 不在视觉分支

视觉分支还是那套 `Qwen2.5-VL` 的视觉编码链，本身没有变成专家网络。

### 2. MoE 直接插在 decoder layer 里

`wall-x` 不是在 decoder 后面外挂一个专家模块，而是把每一层 decoder 换成了 `Qwen2_5_VLDecoderLayer_with_MoE`。

也就是说，MoE 改造发生在：

- layer 内部的 token 路由
- expert-aware FFN 路径
- 以及围绕这些路由信号展开的 attention / norm / position 组织

### 3. 真正支撑 MoE 的不是“一个专家类”，而是一组运行时控制量

最关键的几组信号是：

- `moe_token_types`
- `start_indices`
- `end_indices`
- `positional_masks`

这说明 `wall-x` 的 MoE 不是纯静态结构，而是很依赖输入序列当前的 token 组织方式。

### 4. 第 4 篇最该立住的判断

> `wall-x` 的 MoE，不是一个挂在模型外面的附加头，而是直接长在 decoder layer 里面。

---

## 五、wall-x 真正加的东西主要有三块

### 1. MoE 化 decoder

这是最像“模型增量”的部分。

`wall-x` 不是只在最后加一个专家头，而是把 decoder layer 换成了 `Qwen2_5_VLDecoderLayer_with_MoE` 这一套。顶层模型主干也从基础版的 `Qwen2_5_VLModel` 换成了 `Qwen2_5_VLMoEModel`，对应 [`modeling_qwen2_5_vl_act.py#L284`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L284)。

这部分带来的增量主要包括：

- `moe_token_types`
- `start_indices / end_indices`
- token type 驱动的 expert 路由
- expert-aware 的 FFN 路径
- 以及围绕这些路由信号展开的 attention / norm / position 处理

这一块可以算是 `wall-x` 最明确的源码级改动。

### 2. action 路径

第二块核心增量是动作生成路径。

这里最关键的不是再造一个新 backbone，而是在统一序列主干之上，额外挂上：

- proprioception embedding 注入
- action token 占位
- `ActionProcessor`
- `action_proj_back`
- Flow Action 的 `ODE / Euler` 迭代

这部分让模型从“多模态文本生成器”变成了“机器人动作生成器”。但严格说，它更像是：

> 在现有主干上，接出了一条面向动作任务的生成系统。

### 3. embodiment / robot-specific 输入组织

第三块更偏系统接口，而不是模型结构创新。

比如：

- `dof_mask`
- `agent_pos_mask`
- customized robot config
- 不同 embodiment 的 action/statistics 对齐

这部分不是在改网络数学，而是在解决：

> 不同机器人配置，怎么被统一进同一个动作表示和推理流程里。

---

## 六、真正复杂的地方，不在公式，而在系统组织

如果把注意力从“有没有新公式”移开，会发现 `wall-x` 的复杂度主要不在：

- 新注意力数学
- 新 backbone
- 新 MoE 理论

而在更工程化的地方：

- 多模态输入怎么拼
- 图像 embedding 怎么回填
- MoE token 怎么组织和路由
- action token 怎么进入序列
- prefix / postfix 怎么切
- KV cache 怎么截断和复用
- ODE 循环怎么挂在 decoder 后面
- 不同机器人配置怎么共用同一个 runtime

所以更准确的表述应该是：

> `wall-x` 的创新密度不一定高在“模型结构发明”上，而是高在“把现成主干改造成可服务机器人动作生成的运行系统”上。

---

## 七、为什么这会把讨论重心推向 runtime

也正因为 `wall-x` 不是一个“全新数学结构驱动”的模型，后续优化自然不会一直停在模型结构分析上。

当你把主干拆开以后，会发现真正困难的问题越来越像：

- profiling
- runtime 调度
- C++ 化
- KV cache 管理
- 量化
- 小算子 fusion
- graph capture
- kernel 设计

也就是说，`wall-x` 很适合拿来讨论“系统复杂度”，而不只是“模型增量”。

---

## 八、一个更直白的归纳

如果要把上面的判断压成更短的一句话，那就是：

> `wall-x` 不是一个“模型架构革命”，而更像是一个 `Qwen2.5-VL + MoE + Action/Flow 任务化改造 + 机器人接口工程` 的组合体。

所以“它自己做的创新不多”这个直觉，大体是对的。  
但同时也要补一句：

> 虽然结构创新不算特别多，但运行时复杂度一点都不低。

---

## 九、当前最值得保留的判断

后面如果继续深挖，可以把当前结论保留成下面这几条：

1. `wall-x` 的基础主干大部分仍然是 `Qwen2.5-VL`。
2. 它最明确的模型级增量是 `MoE 化 decoder`。
3. 它最重要的任务级增量是 `action / flow` 路径。
4. 它最难的部分不在新公式，而在“怎么把这些东西组织成一条可运行的机器人推理链”。
5. 这也是为什么，对 `wall-x` 的讨论最后一定会从模型结构转向 runtime。
