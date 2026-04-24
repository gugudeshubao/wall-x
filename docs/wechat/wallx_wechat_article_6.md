# Wall-X 是怎么训练出来的：从 `DataCollator` 到双 Loss，它有没有用 RL？

> 前面几篇把推理和结构讲得差不多了，但很多人真正关心的问题是另一类：它到底怎么训练出来的？是不是基于 Qwen2.5-VL？有没有用 RL？

**TL;DR**
- Wall-X 的训练入口很普通，就是 `forward -> loss -> backward -> AdamW.step()`。
- 它不是基于纯文本 Qwen2，而是基于 `Qwen2.5-VL` 这条多模态基座继续长出来的。
- 训练 batch 里同时有图像、文本、状态、连续动作、MoE 路由标记。
- loss 本质上是 `cross entropy + flow_loss_weight * flow_loss`。
- 按这个公开仓库代码，没有看到 PPO、GRPO、RLHF 这类 RL 训练链。

## 一、训练入口其实很朴素

Wall-X 的训练脚本是：

- `train_qact.py`

真正干活的 trainer 是：

- `QwenVlAct_Trainer`

trainer 里最关键的一句其实非常普通：

```python
outputs = self.model(**batch, mode="train")
```

然后就是标准流程：

- 拿 `loss`
- backward
- clip grad
- AdamW.step()
- lr scheduler.step()

这已经先告诉你一件事：

> 从训练框架外壳看，Wall-X 不是 RL trainer，也不是 preference optimization trainer，而就是一套标准监督训练循环。

## 二、它真的是从 Qwen2.5-VL 长出来的

Wall-X 的动作模型类是：

- `Qwen2_5_VLMoEForAction`

这个类不是凭空来的，它直接继承自 Qwen2.5-VL 那条多模态模型线。

最关键的结构关系可以压成：

`Qwen2.5-VL 基座`
-> 保留视觉编码器
-> 把 decoder 换成带 MoE 的版本
-> 增加 `<|propri|>`、`<|action|>` 等新 token
-> 增加 `ActionProcessor`

也就是说，它不是在纯语言模型上硬加机器人头，而是沿着 Qwen2.5-VL 这条“视觉 + 文本 + 统一序列”的骨架继续长出来的。

## 三、trainer 里实际上有两条模型初始化路径

### 路径 A：从已有 Wall-X checkpoint 继续训练

这条路径最完整，也最稳。

它会直接从已有 Wall-X 权重目录里加载：

- config
- processor
- safetensors

如果你已经有一版 Wall-X，这条路径就是标准 continue training。

### 路径 B：从 Qwen2.5-VL 基座出发构造 Wall-X

这条路径的设计意图很清楚：

1. 先读 `Qwen2_5_VLConfig`
2. 给 tokenizer 增加 `<|propri|>`、`<|action|>`，可选再加 `<|action_token_i|>`
3. 实例化 `Qwen2_5_VLMoEForAction`
4. 再尝试把原始 Qwen 权重迁到新的 Wall-X 结构里

从思路上看，它就是：

> 拿 Qwen2.5-VL 的多模态基础能力，往上接动作和 MoE，再继续训练。

## 四、它原本想怎么迁移原始 Qwen 权重

trainer 里专门有个函数在干这件事：

- `load_qwen_pretrain_weight()`

它的逻辑其实很直白：

### 1. 原始 dense MLP 权重

如果开了 `mlp_moe`，原始的：

- `layers.X.mlp.*`

会被重命名成：

- `layers.X.moe.experts.0.*`

也就是说，原始 Qwen 的 dense MLP 会被塞进 `expert 0`。

### 2. 原始 attention 投影权重

如果开了 `attention_moe`，原始的：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

也会被迁到对应的 `*_experts.0` 上。

这个设计想表达的意思其实非常清楚：

- `expert 0` 保留通用大模型能力
- 新增的动作专家再去学动作相关的东西

## 五、但当前公开仓库里，这条裸 Qwen 初始化路径有个现状问题

这里必须说清楚。

虽然 `load_qwen_pretrain_weight()` 里已经写了权重重命名逻辑，但真正把这些 `renamed_weights` 加进模型的那行 `load_state_dict(...)` 在当前 trainer 里是注释掉的。

所以你要分两层理解：

- 从设计意图看，它显然想“从 Qwen2.5-VL 初始化出 Wall-X”
- 从当前这版 trainer 代码看，这条路径最后一步并没有完全接通

这意味着：

- 公开仓库里最稳的继续训练方式，还是从已有 Wall-X checkpoint 出发
- “从裸 Qwen2.5-VL 权重直接迁到 Wall-X” 这条线，当前实现更像半完成状态

## 六、训练 batch 不是“图像 + 文本”这么简单

`DataCollator` 会把样本拼成一个复合 batch。比较关键的字段有：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`
- `proprioception`
- `agent_pos_mask`
- `action_chunk`
- `dof_mask`
- `moe_token_types`
- `labels`

也就是说，Wall-X 的训练样本从一开始就不只是图文对话，而是：

> 图像 + 指令 + 状态 + 动作监督 + expert 路由标记

这也是它和普通 VLM 最不一样的地方之一。

## 七、训练 prompt 里其实有两层动作占位

Wall-X 训练时的 prompt 里会同时出现两类动作相关符号：

- `<|action_fast|>`
- `<|action|>`

它们分别服务两条不同监督路径。

### 1. `<|action_fast|>`

如果启用了 fast action tokenizer，这一段会被替换成真正的离散 action token 串。

这时交叉熵就能直接监督离散动作 token。

### 2. `<|action|>`

这部分更重要，因为它对应的是连续动作 / flow 路径。

训练时这些 `<|action|>` 不会直接拿去做 CE，而是会被后面的 noisy action embedding 替换掉，成为连续动作建模的位置。

所以 Wall-X 的 prompt 设计天然就给两条动作训练路线都留了入口。

## 八、`labels` 不是全序列监督，只监督 assistant 输出

在标签构造上，Wall-X 也很标准：

- 先把 `labels` 全部初始化成 `-100`
- 再只把 assistant response 区间对应出来
- 其他位置继续保持 `-100`

同时还会额外屏蔽：

- `<|action|>`
- `<|propri|>`
- `pad token`

这说明：

- 连续动作占位 token 本身不参与交叉熵
- 交叉熵主要监督 assistant 文本区域
- 如果 fast tokenizer 开启，离散 action token 也会被监督

## 九、`moe_token_types` 在训练时也是显式生成的

和推理阶段一样，训练时的 `moe_token_types` 也不是模型自己学出来的，而是 collator 直接按 token id 生成的：

```python
inputs.input_ids == action_token_id
```

也就是说，在当前最常见的场景下：

- 普通 token -> 0
- action token -> 1

这也解释了为什么 Wall-X 的 expert 路由可解释性很强。  
它不是“让模型自己在暗处决定哪些 token 去哪个 expert”，而是先用 token 语义把这件事显式标出来。

## 十、连续动作训练不是把真实动作直接塞进去，而是先加噪

这是训练链里最关键的一步。

Wall-X 训练连续动作时，先构造：

```python
noise
time
noisy_action = (1 - t) * noise + t * action_chunk
flow = action_chunk - noise
```

然后把 `noisy_action` 编成 embedding，替换到 `<|action|>` 对应的位置。

这意味着模型训练时真正学的不是“最终动作本体”，而是：

> 给定当前 noisy action 和当前时间 `t`，应该往哪个方向更新，才能从噪声流向真实动作。

这就是 flow matching 的核心思路。

## 十一、总 loss 其实就是双损失

把训练目标压缩一下，非常简单：

### 第一支：交叉熵

如果有 `labels`，就按标准 next-token shift 去算 `cross_entropy_loss`。

### 第二支：flow loss

如果有 `action_chunk`，就从 action token 位置取 hidden states，投回动作空间，得到当前速度预测，再用 MSE 去拟合 `flow`。

### 最后合起来

```text
loss = cross_entropy_loss + flow_loss_weight * flow_loss
```

所以 Wall-X 训练的本质就是：

- 文本/离散 token 世界：交叉熵
- 连续动作世界：flow matching

二者在同一轮前向里一起优化。

## 十二、它有没有用 RL

按这个公开仓库代码，我的判断是：

**没有。**

原因可以从三层证据看。

### 1. 代码搜索层面

没有看到这些 RL 关键词对应的训练实现：

- PPO
- GRPO
- DPO
- RLHF
- reward model
- actor / critic
- advantage

### 2. trainer 层面

训练循环就是：

- forward
- loss
- backward
- AdamW.step()

没有 rollout，没有 reward，没有策略更新分支。

### 3. loss 层面

公开代码里能看到的只有：

- `cross_entropy_loss`
- `flow_loss`

没有 KL penalty，没有 reward shaping，没有 preference loss。

所以如果只根据这个 repo 判断，Wall-X 很明确是：

> 监督式训练，不是 RL。

## 十三、这篇文章真正要立住的判断

如果把这一篇压成一句话，那就是：

> Wall-X 是在 Qwen2.5-VL 多模态基座上继续长出来的：它保留了视觉编码器和统一序列建模范式，把 decoder 改造成带 MoE 和动作 token 的版本，再用 demonstration 数据做监督训练，同时优化交叉熵和 flow loss；按公开仓库代码，没有看到 RL。

到这里，这个系列就闭环了。

