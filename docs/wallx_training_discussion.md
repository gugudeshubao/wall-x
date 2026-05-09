# wall-x 训练链讨论

这份文档只整理当前对 `wall-x` 训练链的讨论结论，不改写系列文章，也不追求正式发文口吻。目标是把训练入口、batch 组织、双损失结构和后续最值得继续追的问题先压实。

---

## 一句话判断

按公开仓库当前代码看，`wall-x` 的训练可以先压成一句话：

> 它本质上是“`Qwen2.5-VL` 基座上的监督式 imitation learning”，损失是 `cross entropy + flow matching`，不是 RL。

也就是说，它不是：

- PPO
- DPO
- GRPO
- reward model + rollout
- actor-critic

至少在这个公开仓库里，没有看到这类 RL 训练链。

---

## 一、训练入口是什么

训练入口很直接：

- 启动脚本：`train_qact.py`
- 实际训练器：[`qwen_vl_act_trainer.py`](/Users/sam/project/github/wall-x/wall_x/trainer/qwen_vl_act_trainer.py)

按当前实现看，训练框架本身是标准的：

- `Accelerate`
- `AdamW`
- cosine scheduler

不是某种专门为 RL 写的 trainer。

在 trainer 里，最值得先记住的是两条模型加载路径，见 [`qwen_vl_act_trainer.py#L570`](/Users/sam/project/github/wall-x/wall_x/trainer/qwen_vl_act_trainer.py#L570)。

### 路径 A：从已有 Wall-X checkpoint 继续训练

- `model_type == "wall-oss"`
- 直接走 `Qwen2_5_VLMoEForAction.from_pretrained(...)`

对应 [`qwen_vl_act_trainer.py#L584`](/Users/sam/project/github/wall-x/wall_x/trainer/qwen_vl_act_trainer.py#L584)。

这条路径是当前代码里最稳、最完整的继续训练路径。

### 路径 B：从 Qwen2.5-VL 基座出发构造 Wall-X

- `model_type == "qwen2_5"`
- 先起 `Qwen2_5_VLMoEForAction`
- 再尝试把基础 Qwen 权重迁进来

对应 [`qwen_vl_act_trainer.py#L596`](/Users/sam/project/github/wall-x/wall_x/trainer/qwen_vl_act_trainer.py#L596)。

这条路径的设计意图很清楚：

> 先拿 `Qwen2.5-VL` 的多模态基础能力，再把 `MoE + Action` 结构接上，然后继续训。

但当前公开代码里，这条“裸 Qwen 初始化”路径看起来没有完全接通，所以如果只问“仓库当前最稳的训练起点是什么”，答案仍然更像是：

> 从已有 `Wall-X` checkpoint 继续训。

---

## 二、训练 batch 不是“图像 + 文本”，而是复合 batch

训练链里最关键的一个位置，是 [`DataCollator`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py)。

对应实现见 [`load_lerobot_dataset.py#L246`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L246) 和 [`load_lerobot_dataset.py#L328`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L328)。

它做的不是简单拼 batch，而是先把样本拆成几类训练信号，再统一组装：

### 1. 状态侧

对 `agent_pos`：

- 先堆叠成 batch
- 用 `~torch.isnan(...)` 生成 `agent_pos_mask`
- 把 `nan` 变成 `0`
- 再做 `normalizer_propri.normalize_data(...)`

最后得到：

- `proprioception`
- `agent_pos_mask`

对应 [`load_lerobot_dataset.py#L332`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L332)。

### 2. 动作侧

对 `action`：

- 先堆叠成 batch
- 用 `~torch.isnan(...)` 生成 `dof_mask`
- 把 `nan` 变成 `0`
- 再做 `normalizer_action.normalize_data(...)`

最后得到：

- `action_chunk`
- `dof_mask`

对应 [`load_lerobot_dataset.py#L368`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L368)。

### 3. 文本和图像侧

`DataCollator` 会先把文本里的 action 占位符替换好，再走统一的 `preprocesser_call(...)`：

- `replace_action_token(...)`
- `preprocesser_call(...)`

对应 [`load_lerobot_dataset.py#L416`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L416) 和 [`load_lerobot_dataset.py#L424`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L424)。

### 4. MoE 路由信号

在 tokenizer / image processor 跑完之后，collator 还会显式生成：

- `moe_token_types = inputs.input_ids == action_token_id`

对应 [`load_lerobot_dataset.py#L435`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L435) 和 [`load_lerobot_dataset.py#L438`](/Users/sam/project/github/wall-x/wall_x/data/load_lerobot_dataset.py#L438)。

所以训练时模型实际吃进去的，不是普通 VLM batch，而是：

- `input_ids`
- `pixel_values`
- `labels`
- `proprioception`
- `agent_pos_mask`
- `action_chunk`
- `dof_mask`
- `moe_token_types`

---

## 三、labels 是怎么构造的

文本和图像处理最终落到 [`preprocesser_call(...)`](/Users/sam/project/github/wall-x/wall_x/data/utils.py)。

对应实现见 [`utils.py#L119`](/Users/sam/project/github/wall-x/wall_x/data/utils.py#L119)。

这里最关键的一步是：

- 先把整张 `labels` 初始化为 `-100`
- 只保留 assistant 回复区域的 token 监督
- 再把特殊 token 屏蔽掉

对应代码位置：

- 初始化 `labels`：[`utils.py#L231`](/Users/sam/project/github/wall-x/wall_x/data/utils.py#L231)
- 找 assistant 区域：[`utils.py#L239`](/Users/sam/project/github/wall-x/wall_x/data/utils.py#L239)
- 把 assistant 区域拷进 `labels`：[`utils.py#L269`](/Users/sam/project/github/wall-x/wall_x/data/utils.py#L269)
- 屏蔽 `<|action|>` / `<|propri|>` / pad：[`utils.py#L273`](/Users/sam/project/github/wall-x/wall_x/data/utils.py#L273)

所以文本监督不是“全序列都算 CE”，而是：

> 只监督 assistant 输出，而且 action / proprio 这些特殊 token 本身不会直接参与 CE。

---

## 四、训练时动作监督不是“直接回归 action_chunk”

动作侧最关键的逻辑在 [`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py)。

### 1. 训练前向

`ActionProcessor.forward()` 的核心步骤在 [`action_head.py#L671`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L671)。

它做的是：

1. 对 `action_chunk` 加噪
2. 采样时间 `t`
3. 构造：
   - `noisy_action = (1 - t) * noise + t * action_chunk`
   - `flow = action_chunk - noise`
4. 再把 `noisy_action (+ dof_mask)` 投成 action embedding

也就是说，训练时模型学的不是“直接输出最终动作”，而是：

> 预测从噪声流向真实动作的向量场。

### 2. 推理 step

推理时 `step(...)` 会把当前 `timestep + noisy_action (+ dof_mask)` 投成 action embedding，见 [`action_head.py#L743`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L743)。

所以训练和推理在动作侧是统一的：

- 训练时采样 `t`
- 推理时按 ODE / Euler 迭代 `t`

---

## 五、训练时模型前向到底做了什么

顶层训练前向在 [`modeling_qwen2_5_vl_act.py#L1294`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L1294)。

如果把它压缩一下，大致是：

1. 准备 `position_ids`
2. `input_ids -> token embedding`
3. 把 image/video embedding 回填进统一序列
4. 把 `proprioception` embedding 回填进统一序列
5. 把 `flow/noisy action` embedding 回填进 `<|action|>` 位置
6. 用 `moe_token_types + start_indices + end_indices` 跑 `Qwen2_5_VLMoEModel`
7. 过 `lm_head`
8. 调 `compute_loss(...)`

这里最值得记住的不是代码细节，而是：

> 训练时文本监督和动作监督共用的是同一条 decoder 主干。

---

## 六、双损失到底怎么落地

真正的损失汇总在 [`vla_mixin.py#L769`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L769)。

### 1. 文本侧：cross entropy

如果 `labels` 存在，就做标准 next-token shift：

- `shift_logits = logits[..., :-1, :]`
- `shift_labels = labels[..., 1:]`

然后只对 `labels != -100` 的位置取平均。

对应 [`vla_mixin.py#L805`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L805)。

### 2. 动作侧：flow loss

如果 `action_chunk` 存在：

- 先在输入序列里找到 `<|action|>` 位置
- 从这些位置取 `hidden_states`
- 投回动作空间
- 和 `flow` 做 MSE
- 再乘 `dof_mask`
- 如果有 `flow_loss_mask`，继续屏蔽

对应：

- 取 action hidden states：[`vla_mixin.py#L858`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L858)
- 调 `flow_loss(...)`：[`vla_mixin.py#L863`](/Users/sam/project/github/wall-x/wall_x/model/vla_mixin.py#L863)
- `flow_loss(...)` 本体：[`action_head.py#L786`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L786)

### 3. 总损失

最后就是：

> `loss = cross_entropy_loss + flow_loss_weight * flow_loss`

其中 `flow_loss_weight` 由配置控制。

---

## 七、当前对训练链最重要的判断

到这里可以先保留 5 条判断：

1. `wall-x` 的训练本质是监督式 imitation learning，不是 RL。
2. 它不是只做 next-token prediction，而是文本监督和动作监督并行。
3. `DataCollator` 的重要性非常高，因为训练 batch 不是普通 VLM batch，而是复合 batch。
4. 动作监督学的不是“直接输出 action_chunk”，而是 flow matching 里的向量场。
5. 文本和动作两种监督最终是共用同一条 decoder 主干的。

---

## 八、明天最值得继续追的问题

明天最值得接着追的，不是再泛泛看 trainer，而是只盯一个问题：

> **`labels`、`action_chunk`、`moe_token_types` 三者在同一个样本里到底是怎么对齐的？**

更具体一点，就是：

1. 文本里的 assistant 回复和 `<|action|>` 占位是什么关系？
2. `labels` 为什么要屏蔽 `<|action|>` 和 `<|propri|>`？
3. `action_chunk` 是怎么被塞回 `<|action|>` 位置的？
4. `moe_token_types` 为什么可以直接由 `input_ids == action_token_id` 得到？
5. 这三者对齐之后，模型到底是在同一序列上同时学什么？

这个问题一旦讲透，`wall-x` 的训练链就基本算是吃透了。
