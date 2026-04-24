# Wall-X 是怎么基于 Qwen2.5-VL 训练出来的，以及它有没有用 RL

## 结论先行

按这个公开仓库当前代码看，可以先下 4 个结论：

1. Wall-X 不是基于纯文本 Qwen2 训出来的，而是基于 **Qwen2.5-VL** 这条多模态基座继续改出来的。
2. 它的训练方式本质上是 **监督学习 / imitation learning**。
3. 训练目标不是单一的 next-token prediction，而是 **`cross entropy + flow matching`** 的双损失。
4. **按公开仓库代码，没有看到 RL。**

更准确地说，这个 repo 里没有看到：

- PPO
- DPO
- GRPO
- RLHF
- reward model
- actor-critic
- rollout 后按 reward 回传的训练逻辑

所以如果只根据这个仓库判断，Wall-X 更像是：

> 基于 Qwen2.5-VL 的多模态监督微调，叠加了 MoE、动作 token、状态 token 和 continuous action 的 flow matching 训练。

## 1. 训练入口是什么

训练入口很直接：

- 启动脚本是 `train_qact.py`
- 实际训练器是 `wall_x/trainer/qwen_vl_act_trainer.py`

`train_qact.py` 本身没有什么花活，主要就是：

1. 读 yaml 配置
2. 初始化 `Accelerator`
3. 构造 `QwenVlAct_Trainer`
4. 调 `trainer.fit()`

也就是说，训练框架本身是标准的 Accelerate + AdamW 路线，不是某种定制 RL trainer。

## 2. 它是怎么从 Qwen2.5-VL 长出来的

Wall-X 的动作模型类是：

- `Qwen2_5_VLMoEForAction`

它直接继承自：

- `Qwen2_5_VLForConditionalGeneration`

这件事本身已经说明了它的基座来源：

- 不是另起一个全新模型
- 而是从 Qwen2.5-VL 这条多模态结构继续往上长

顶层模型结构大致可以压成：

`Qwen2.5-VL 基座`
-> 保留视觉编码器 `self.visual`
-> 把 decoder 替换成 `Qwen2_5_VLMoEModel`
-> 保留 `lm_head`
-> 增加 `ActionProcessor`
-> 增加 `<|propri|>`、`<|action|>`、可选 `<|action_token_i|>`

也就是说，它不是“在 Qwen2.5-VL 外面套一层动作头”，而是：

- 保留原来的视觉入口
- 保留统一序列建模思路
- 把 decoder 改造成适合动作和 MoE 的版本

## 3. trainer 里其实有两条初始化路径

`QwenVlAct_Trainer.load_model()` 里，至少能看到两条路径：

### 路径 A：从已有 Wall-X checkpoint 继续训练

这条路径对应：

- `model_type == "wall-oss"`

它会直接调用：

- `Qwen2_5_VLMoEForAction.from_pretrained(...)`

然后从已有的 `.safetensors` 里把 Wall-X 权重整体加载进来。

这条路径是当前代码里最完整、最稳的一条继续训练路径。

### 路径 B：从 Qwen2.5-VL 基座出发构造 Wall-X

这条路径对应：

- `model_type == "qwen2_5"`

它做的事情是：

1. 从 `qwen_vl_act_config_path` 读 `Qwen2_5_VLConfig`
2. 从 `pretrained_wallx_path` 读 processor
3. 给 tokenizer 加新 token
   - `<|propri|>`
   - `<|action|>`
   - 可选 `<|action_token_i|>`
4. 把机器人配置写回 `model_config`
5. 实例化 `Qwen2_5_VLMoEForAction`
6. 试图把预训练权重迁到新的 Wall-X 结构上

这条路径的设计意图很明确：

> 先拿 Qwen2.5-VL 的多模态基础能力，再把 Wall-X 的动作/MoE 结构接上，然后继续训。

## 4. 它想怎么把原始 Qwen 权重迁进来

trainer 里专门有个函数：

- `load_qwen_pretrain_weight(model, pretrain_weight_path)`

这个函数做的关键事情是“权重重命名”：

### 原始 dense MLP 权重

如果开了 `mlp_moe`，它会把：

- `layers.X.mlp.*`

重命名成：

- `layers.X.moe.experts.0.*`

也就是说，原始 Qwen 的 dense MLP 会被塞进 `expert 0`。

### 原始 attention 投影权重

如果开了 `attention_moe`，它会把：

- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`

重命名成：

- `q_proj_experts.0`
- `k_proj_experts.0`
- `v_proj_experts.0`
- `o_proj_experts.0`

这同样是在表达一个设计意图：

> 原始 Qwen2.5-VL 的能力主要落在 `expert 0`，新增动作专家则交给别的 expert 去学。

## 5. 但当前 trainer 里的这条“裸 Qwen 初始化”路径有个现状问题

这里有一个必须单独说明的源码现状。

虽然 `load_qwen_pretrain_weight()` 已经把权重重命名逻辑写出来了，但当前版本里，真正把这些 `renamed_weights` 加进模型的这行代码被注释掉了：

```python
# err = model.load_state_dict(renamed_weights, strict=False)
```

这意味着：

- 从设计上看，它显然是想“用 Qwen2.5-VL 初始化 Wall-X”
- 但从当前 trainer 代码实现上看，这条 `model_type == "qwen2_5"` 路径是 **未完全接通的**

所以如果你问“公开仓库当前最稳的训练起点是哪条”，答案反而更像是：

- 从已有 Wall-X checkpoint 继续训

而不是从原始 Qwen2.5-VL 权重裸启动。

换句话说：

- **设计意图**：基于 Qwen2.5-VL 迁移出 Wall-X
- **当前这版 trainer 实现**：这条迁移路径最后一步看起来还没彻底写完

## 6. expert0 和 expert1 大致分别在学什么

仓库里没有直接写“expert0 是谁、expert1 是谁”这种显式注释，但从几处实现可以比较稳地推出来：

### 第一，router 很简单

`TokenTypeRouter` 的核心逻辑是：

```python
experts_indices = token_types % num_experts
```

也就是说，expert 分配本质上就是按 token type 来的。

### 第二，当前最常见的 `moe_token_types` 是布尔的

不管在训练还是推理，`moe_token_types` 最常见的来源都是：

- `inputs.input_ids == action_token_id`

也就是：

- 普通 token -> 0
- action token -> 1

### 第三，仓库里的 FLOPs 估算直接写了

`wall_x/model/model_utils.py` 里有一句非常直白的注释：

- `expert0 = language tokens`
- `expert1 = action tokens`

所以在这个仓库的常见两 expert 设置下，最合理的理解就是：

- `expert 0`：承接原始 Qwen2.5-VL 的通用语言/上下文能力
- `expert 1`：承接动作 token 的专门建模能力

这和上面“把原始 Qwen 权重迁到 expert0”的设计也是一致的。

## 7. 训练数据是什么形式

训练 batch 不是简单的“图像 + 文本”，而是一个复合 batch。

`DataCollator` 里会生成这些关键字段：

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
- `labels`

其中最关键的是：

- `proprioception`：归一化后的机器人状态
- `action_chunk`：归一化后的连续动作序列
- `dof_mask`：动作自由度有效位
- `moe_token_types`：action token 路由标记

这说明 Wall-X 的训练从样本层面就已经不是纯 VLM，而是：

> 图像 + 指令 + 状态 + 动作监督 的联合训练。

## 8. 它训练时到底学什么

Wall-X 的训练目标不是单一的 next-token prediction。

它实际上同时在学两件事：

### 第一件事：文本 / 离散 token 预测

这一支走标准交叉熵：

- 只监督 assistant 响应区间
- `<|action|>`、`<|propri|>`、`pad` 都会被 mask 成 `-100`
- 如果启用了 fast tokenizer，则离散 action token 也会进入交叉熵监督

所以这部分更像：

- Chat-style SFT
- 可选地叠加离散 action token 监督

### 第二件事：连续动作的 flow matching

动作这一支不是直接回归最终 action，而是先构造：

- `noise`
- `time`
- `noisy_action = (1 - t) * noise + t * action_chunk`
- `flow = action_chunk - noise`

然后把 `noisy_action` 编成 embedding，替换掉序列里的 `<|action|>` 位置。

最终模型在 action token 位置输出 hidden states，再通过 `action_proj_back` 投回动作空间，用 MSE 去拟合 `flow`。

所以这条连续动作训练更像：

- imitation learning
- flow matching

而不是 RL policy optimization。

## 9. 总 loss 是什么

`compute_loss()` 里最终非常直接：

- 有 `labels` 时，算 `cross_entropy_loss`
- 有 `action_chunk` 时，算 `flow_loss`
- 总损失：

```text
loss = cross_entropy_loss + flow_loss_weight * flow_loss
```

这就是 Wall-X 训练目标的本质。

更准确一点说：

- 文本/离散动作 token 世界，用交叉熵
- 连续动作世界，用 flow matching 的 MSE

这两部分在同一轮前向里一起优化。

## 10. 它有没有用 RL

按这个公开仓库当前代码，我的判断是：

**没有。**

判断依据不是一句猜测，而是几层证据叠起来的：

### 代码搜索层面

仓库里没有看到这些 RL 训练相关实现：

- PPO
- DPO
- GRPO
- RLHF
- reward model
- actor / critic
- advantage
- policy gradient
- TRL trainer

### trainer 层面

训练循环就是标准的：

```text
forward
-> loss
-> backward
-> AdamW.step()
```

没有 rollout、没有 reward、没有策略更新分支。

### loss 层面

公开代码里能看到的只有：

- `cross_entropy_loss`
- `flow_loss`

没有 KL penalty、没有 reward shaping、没有 preference loss。

所以如果只根据这个 repo，下结论应该很谨慎但也很明确：

> Wall-X 在公开代码里体现出来的是监督式训练，不是 RL。

## 11. 最靠谱的一句话总结

如果要把这套训练来源压成一句话，可以写成：

> Wall-X 是在 Qwen2.5-VL 多模态基座上继续长出来的：它保留了视觉编码器和统一序列建模范式，把 decoder 改造成带动作 token 的 MoE 版本，再用机器人 demonstration 数据做监督训练，同时优化文本/离散 token 的交叉熵和连续动作的 flow matching loss；按公开仓库代码，没有看到 RL。

## 12. 关键源码位置

- 训练入口：`train_qact.py`
- trainer：`wall_x/trainer/qwen_vl_act_trainer.py`
- trainer 加载模型主逻辑：`wall_x/trainer/qwen_vl_act_trainer.py:581-657`
- `wall-oss` 继续训练路径：`wall_x/trainer/qwen_vl_act_trainer.py:584-593`
- `qwen2_5` 基座初始化路径：`wall_x/trainer/qwen_vl_act_trainer.py:596-657`
- 原始 Qwen 权重迁移逻辑：`wall_x/trainer/qwen_vl_act_trainer.py:767-831`
- 当前被注释掉的 `load_state_dict`：`wall_x/trainer/qwen_vl_act_trainer.py:825-827`
- 动作模型顶层：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:773-987`
- `from_pretrained()`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:829-926`
- MoE decoder layer：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:82-160`
- `TokenTypeRouter`：`wall_x/model/vla_mixin.py:34-49`
- `SparseMoeBlock`：`wall_x/model/vla_mixin.py:85-120`
- 数据 collator：`wall_x/data/load_lerobot_dataset.py:328-446`
- 训练时 flow/noisy action 构造：`wall_x/model/action_head.py:668-738`
- 推理时 action step：`wall_x/model/action_head.py:740-781`
- flow loss：`wall_x/model/action_head.py:783-808`
- 总 loss：`wall_x/model/vla_mixin.py:763-872`
- 2 expert 语义旁证：`wall_x/model/model_utils.py:175-183`

