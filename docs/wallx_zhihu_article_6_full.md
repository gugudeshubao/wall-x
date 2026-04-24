# 番外：Wall-X 是怎么训练出来的？从 `DataCollator` 到双 Loss

主线的 5 篇，到上一章其实已经闭环了。

如果你的目标是理解 Wall-X 的推理结构、看清它的多模态主干、知道 MoE 插在什么地方、以及动作是怎么一步步生成出来的，那么前面的内容已经够用。

但如果你还想继续追一个更底层的问题：

> 这套行为到底是怎么训练出来的？

那就必须把视角从“推理路径”切到“训练路径”。

这也是为什么我一直建议把训练单独成篇。因为一旦进入训练，关注点马上就变了：

- 不是先看 `generate_flow_action()` 了，而是先看 `DataCollator`
- 不是先看 KV cache 了，而是先看 `labels`
- 不是先看动作怎么推出来，而是先看 `action_chunk`、`moe_token_types`、`dof_mask` 这些字段是怎么被拼出来的
- 最后要关心的，也不是生成结果，而是 `cross_entropy_loss + flow_loss_weight * flow_loss` 这套双损失到底怎么落到代码里

所以这一篇就只干一件事：  
顺着训练实际调用链，把 Wall-X 的训练路径完整串一遍。

## 先从 trainer 开始：训练入口其实很朴素

Wall-X 训练时最直接的入口，在 `wall_x/trainer/qwen_vl_act_trainer.py`。

核心那句代码其实很简单：

```python
outputs = self.model(**batch, mode="train")
```

然后 trainer 紧接着做的，就是你熟悉的标准套路：

- 取 `outputs.loss`
- backward
- clip grad
- optimizer step
- lr scheduler step
- 记录日志

从这个入口你能先得到一个很重要的判断：

> Wall-X 的训练框架本身并不奇怪，奇怪的是 `batch` 里到底装了什么，以及 `mode="train"` 这条模型内部分支到底做了什么。

所以接下来，训练路径最值得看的其实不是 trainer 外壳，而是中间这两层：

1. `DataCollator` 怎么把样本拼成训练 batch  
2. `train_step_forward()` 怎么把这个 batch 变成 loss  

## 训练 batch 不是“图片 + 文本”这么简单

先看 `DataCollator.__call__()`。

这一层会把数据样本整理成一组训练时真正要喂给模型的字段。比较关键的有：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`
- `proprioception`
- `agent_pos_mask`
- `action_chunk`
- `dof_mask`
- `moe_token_types`
- `dataset_names`
- 以及 tokenizer 生成的 `labels`

这已经说明一件事：

> Wall-X 的训练样本，天然就是“多模态 + 状态 + 动作监督”的复合体。

和普通 VLM 不一样，它这里不是简单做“图文对话监督”，而是同时把：

- 图像观测
- 文本 prompt
- proprioception
- 动作序列

都一起塞进了 batch。

## `DataCollator` 先做两类归一化：状态和动作

`DataCollator.__call__()` 里，最先做的两件事分别对应：

- `agent_pos`
- `action`

### 对状态 `agent_pos`

它先把 batch 里的 `agent_pos` 堆起来，然后：

- 用 `~torch.isnan(...)` 得到 `agent_pos_mask`
- 把 NaN 填成 0
- 用 `normalizer_propri.normalize_data(...)` 做归一化

最后得到：

- `proprioception`
- `agent_pos_mask`

这说明训练时 proprioception 不是“顺便附带一个状态向量”，而是明确被当作需要规范化、需要带 mask 的正式输入。

### 对动作 `action`

对动作也是类似逻辑：

- 先 stack 成 `action`
- 再用 `~torch.isnan(...)` 得到 `dof_mask`
- 把 NaN 填成 0
- 用 `normalizer_action.normalize_data(...)` 归一化

最后得到：

- `action_chunk`
- `dof_mask`

这里的 `action_chunk`，就是训练时连续动作监督的主体；  
而 `dof_mask` 则表示哪些动作维度当前有效。

这两个字段后面会一路进到：

- `scatter_flow_action_embeddings(...)`
- `ActionProcessor.forward(...)`
- `flow_loss(...)`

所以千万别把它们看成只是 collator 里临时生成的小辅助量。

## 训练 prompt 其实有两层动作占位

Wall-X 训练里一个很有意思的设计，是 prompt 本身同时兼容了两种动作监督形式。

在 `wall_x/data/utils.py` 里，动作任务的 assistant 输出模板大致是：

```text
<|im_start|>assistant
<|action_fast|><|im_end|>
<|action|><|action|><|action|>...
```

这里有两个完全不同的符号：

- `<|action_fast|>`
- `<|action|>`

它们分别服务于两种不同训练目标。

### `<|action_fast|>` 是离散动作监督入口

如果训练时启用了 fast action tokenizer，那么 `replace_action_token(...)` 会把：

```text
<|action_fast|><|im_end|>\n
```

替换成真正的离散 action token 串，比如一串 `<|action_token_i|>`。

然后还会把剩下的 `<|action|>` 全部删掉。

这意味着在 fast tokenizer 打开的情况下，训练样本里会真的出现一段离散 action token，后面的交叉熵损失就能直接监督这些 token。

### `<|action|>` 是连续动作 / flow 路径入口

如果没有 action tokenizer，那么 `replace_action_token(...)` 不会填入离散 action token，而只是去掉 `<|action_fast|>` 那段占位。

这时重复的 `<|action|>` 会保留下来。

这件事非常关键，因为它说明：

- fast tokenizer 开启时，动作可以走离散 token 监督
- fast tokenizer 关闭时，序列里保留的是连续动作占位 token
- 后续这些 `<|action|>` 位置会被 noisy action embedding 替换，用来做 flow 路径训练

换句话说，Wall-X 的 prompt 设计从一开始就给两种动作监督留了入口。

## `labels` 不是全序列监督，只监督 assistant 输出

训练里另一个关键点，是 `labels` 怎么构造。

在 `wall_x/data/utils.py` 里，逻辑非常明确：

1. 先整张 `labels` 初始化成 `-100`
2. 再按对话模板去找 assistant response 区间
3. 只把 assistant 输出部分对应的 token 拷进 `labels`
4. 其余位置继续保持 `-100`

这就是一个很标准的“只监督 assistant 响应”的 chat-style loss 构造。

但 Wall-X 这里还多做了几步 masking：

- `<|action|>` 会被设成 `-100`
- `<|propri|>` 会被设成 `-100`
- `pad_token` 会被设成 `-100`

这意味着连续动作占位 token 本身不会参与交叉熵。

这里可以顺带看出 fast tokenizer 路径的一个巧妙点：

- `<|action|>` 被 mask 掉，不参与 CE
- 但真正插进去的 `<|action_token_i|>` 并没有被统一屏蔽

所以如果 fast action tokenizer 开启，交叉熵实际上会监督离散 action token；  
如果没开启，那交叉熵主要监督的还是普通 assistant 文本区域。

这也就解释了为什么 Wall-X 训练里会同时存在：

- 交叉熵监督
- 连续动作的 flow 监督

因为它们在样本层面本来就对着不同对象。

## `moe_token_types` 在训练时也是直接从 token 语义生成的

和推理阶段一样，训练时的 `moe_token_types` 也不是模型自己在线学出来的，而是 collator 直接按 token id 构造的：

```python
additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id
```

这说明训练阶段的 expert routing 信号也同样是显式的。

当前最常见的场景下，它仍然是一个布尔张量：

- 非 action token 为 0
- action token 为 1

所以从训练到推理，MoE 这条链是一致的：

不是先让模型学一个复杂 router，再决定动作 token 去哪里；  
而是先用 token 语义把 expert 类型标出来，再让 decoder 围绕这条 token type 语义来建模。

## 模型进入训练分支后，先做的还是“多模态 embedding 装配”

当 trainer 调到：

```python
self.model(**batch, mode="train")
```

模型内部会走 `train_step_forward(...)`。

这一步一开始做的事情，和推理路径其实非常接近：

- 如果没传 `start_indices / end_indices`，先根据 `moe_token_types` 现算
- 准备 `position_ids`
- 文本 token 做 `embed_tokens`
- 图像走 `self.visual(...)`
- 再把图像 embedding scatter 回文本序列
- proprioception 也 scatter 到 `<|propri|>` 位置

也就是说，训练路径并没有另外起一条“特殊模型结构”，它仍然是在同一条多模态主干上跑。

真正进入训练特有逻辑的地方，是动作 embedding 的注入。

## 连续动作训练不是直接把真实动作塞进去，而是先加噪

这是训练路径最核心的一步。

在 `train_step_forward(...)` 里，动作相关注入走的是：

```python
inputs_embeds, flow, adarms_cond = self.scatter_flow_action_embeddings(...)
```

而这个函数内部实际会调用：

```python
noisy_action_emb, flow, adarms_cond = self.action_preprocessor(action_chunk, ...)
```

换句话说，训练时并不是把 `action_chunk` 原样投影成 embedding 再塞进 `<|action|>`，而是：

1. 先对真实动作加噪
2. 生成 `noisy_action`
3. 再把 noisy action 编成 embedding
4. 替换掉序列里的 `<|action|>` 占位

与此同时，它还会返回 `flow` 这个训练目标。

这和上一篇讲推理时的 flow 路径正好完全对上了。  
推理时模型看到的是“当前 timestep 下的 noisy action embedding”；训练时也一样。

所以这套动作链不是“训练学一套，推理跑一套”，而是从输入形式到目标形式都非常一致。

## `train_step_forward()` 做完主干前向后，才进入 loss 计算

动作和多模态 embedding 都装好以后，`train_step_forward()` 会调用：

```python
outputs = self.model(...)
hidden_states = outputs[0]
logits = self.lm_head(hidden_states)
```

然后统一走：

```python
self.compute_loss(...)
```

这一步非常关键，因为它说明训练里的两类监督：

- 文本 / 离散 token 监督
- 连续动作 flow 监督

最终是汇总在同一个 loss 计算入口里的。

## `compute_loss()` 里其实就是双损失

如果把 `compute_loss()` 压缩一下，它的结构其实很简单：

```text
loss = 0
if labels 存在:
    计算 cross entropy loss
    loss += cross_entropy_loss
if action_chunk 存在:
    计算 flow loss
    loss += flow_loss * flow_loss_weight
```

这就是 Wall-X 训练目标的本质：

> 文本/离散动作部分走交叉熵，连续动作部分走 flow loss，最后按权重加起来。

所以它不是两阶段训练，也不是两个独立优化器，而是在同一轮前向里同时拿两类监督。

## 交叉熵到底监督了什么

先看交叉熵这一支。

`compute_loss()` 里做的是很标准的 next-token shift：

- `shift_logits = logits[..., :-1, :]`
- `shift_labels = labels[..., 1:]`

然后用 `CrossEntropyLoss(reduction="none")` 先拿到逐 token loss，再对非 `-100` 区域求均值。

所以交叉熵真正监督的，仍然是前面构造出来的那些 assistant response 区域。

对应到不同训练设置：

- 如果 fast tokenizer 打开，交叉熵可以监督离散 action token
- 如果 fast tokenizer 不开，交叉熵主要监督普通 assistant 文本

这里还有一个小指标：

如果配置里存在 action token 列表，代码还会额外算一个 `action_accuracy`。  
它的逻辑也不复杂，本质上就是看 action token 位置上的 argmax 预测是否命中标签。

## flow loss 到底监督了什么

再看 flow 那一支。

`compute_loss()` 里会先找到：

```python
action_mask = input_ids == self.action_token_id_set["action_token_id"]
```

然后只取 action token 位置对应的 hidden states。

这些 hidden states 会送进：

```python
self.action_preprocessor.flow_loss(...)
```

在 `flow_loss(...)` 里，逻辑非常直接：

1. 先用 `action_proj_back` 把 action hidden states 投回动作维度
2. 得到 `v_pred`
3. 用 MSE 拟合前面构造好的 `flow`
4. 再乘 `dof_mask`
5. 如果有 `flow_loss_mask`，再继续屏蔽

所以 flow loss 真正监督的是：

- action token 位置上的连续动作向量场预测

而不是整条序列的所有位置。

这一点也很重要，因为它再次说明 Wall-X 的动作训练不是“把整个 decoder 当成回归器”，而是只在专门的 action token 区域上施加连续动作监督。

## 总 loss 为什么写成 `CE + flow_weight * flow`

到这里其实已经很自然了。

Wall-X 把两种监督绑在一起训练，最直接的写法就是：

```python
loss += cross_entropy_loss
loss += flow_loss * self.config.flow_loss_weight
```

其中：

- 交叉熵管的是离散 token 世界
- flow loss 管的是连续动作世界

这两个世界在 Wall-X 里并不是矛盾关系，而是并行关系。  
尤其是在既有文本对话语义、又有动作输出需求的 VLA 场景下，这种双损失其实很自然。

真正需要调的是两者的相对权重，也就是 `flow_loss_weight`。

## trainer 最终消费的其实也是这几个标量

再回到 trainer 视角，你会发现训练循环最后真正关心的输出也就这些：

- `loss`
- `cross_entropy_loss`
- `flow_loss`
- 可选的 `channel_loss_dict`

也就是说，从 trainer 角度看，Wall-X 训练虽然模型内部链路很复杂，但最后暴露出来的训练信号其实很清楚：

- 总 loss
- 文本/离散 token loss
- 连续动作 loss

这也是这套实现比较务实的一点。  
模型里可以有很多多模态、MoE、flow、action head 的细节，但 trainer 外层还是保持了普通训练框架能理解的输出接口。

## 这里有一处源码现状值得单独指出

训练路径里有一个地方，我觉得值得单独说一下。

trainer 侧明显还想记录 per-dataset 的 `channel_loss_dict` 和 `channel_loss_count_dict`。  
但是 `compute_loss()` 里这些字典的初始化逻辑已经被注释掉了，当前实际返回的是：

- `unique_datasets_name = None`
- `channel_loss_dict = None`
- `channel_loss_count_dict = None`

可后面的循环又还保留着按 `unique_datasets_name` 聚合 loss 的代码。

这说明这条 per-dataset loss 统计链目前是半完成状态：

- trainer 还在期待它
- 输出结构也还保留着
- 但核心初始化和实际可用性已经不完整了

如果以后真要整理训练代码，我会优先把这条链处理干净：

- 要么彻底补完
- 要么先把 trainer 侧依赖去掉

不然它会一直处在“看起来支持、实际上没完全接通”的状态。

## 为什么训练值得单独成篇

现在回头看，你大概也能理解为什么我不建议把训练和前 5 篇主线混在一起了。

因为训练关心的是另一套问题：

- 样本怎么变成 batch
- prompt 里两层 action 占位怎么配合
- labels 怎么只监督 assistant 输出
- 连续动作怎么先加噪再喂进模型
- 双损失怎么在同一轮前向里一起算

这些东西如果硬塞进推理结构篇里，节奏会非常怪。  
但单独拿出来以后，反而能让整套 Wall-X 更完整。

## 把这一篇压成一句话

如果把这一篇的核心结论只压成一句话，那就是：

> Wall-X 的训练本质上是“同一条多模态 decoder 主干 + 两类监督同时优化”：样本先被 `DataCollator` 组装成包含图像、状态、动作和 `moe_token_types` 的复合 batch，随后模型在 action token 位置注入 noisy action embedding，再用 `cross_entropy_loss + flow_loss_weight * flow_loss` 同时监督离散 token 和连续动作向量场。

主线到这里就完整了。

如果后面还想继续写，我会更倾向于做两类后续内容：

- 一类是“代码整理视角”，比如 `channel_loss_dict` 这条半完成链怎么修
- 一类是“性能优化视角”，比如 Wall-X 真正值得 profile 的热点到底在哪

但如果只谈“结构理解”，这个系列到这里已经把最关键的东西都讲齐了。

## 附：文中对应的关键源码位置

- trainer 训练主循环：[`qwen_vl_act_trainer.py`](/Users/sam/project/github/wall-x/wall_x/trainer/qwen_vl_act_trainer.py#L377)
- trainer 的 loss 日志消费：[`qwen_vl_act_trainer.py`](/Users/sam/project/github/wall-x/wall_x/trainer/qwen_vl_act_trainer.py#L422)
- `DataCollator.__call__()`：[`load_lerobot_dataset.py`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L328)
- `moe_token_types` 在 collator 里的生成：[`load_lerobot_dataset.py`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L438)
- assistant-only `labels` 构造：[`utils.py`](/Users/sam/project/github/wall-x/wall_x/data/utils.py#L231)
- 动作任务 prompt 里 `<|action_fast|>` 和 `<|action|>` 的双层占位：[`utils.py`](/Users/sam/project/github/wall-x/wall_x/data/utils.py#L499)
- `replace_action_token(...)`：[`utils.py`](/Users/sam/project/github/wall-x/wall_x/data/utils.py#L606)
- 训练分支 `train_step_forward(...)`：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L1287)
- 训练时 action token 替换成 noisy action embedding：[`vla_mixin.py`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L420)
- `ActionProcessor`：[`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L536)
- 训练时 `noisy_action` / `flow` 构造：[`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L668)
- `compute_loss(...)`：[`vla_mixin.py`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L763)
