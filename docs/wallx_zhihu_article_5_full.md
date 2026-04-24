# Wall-X 是怎么生成动作的：AR、Flow、KV Cache 和 Euler 积分

前面几篇文章，我一直在做铺垫。

先是把 Wall-X 的整体判断立住：它不是一个“整套靠自定义算子重写”的模型。  
然后把推理入口和输入构造讲清楚。  
接着又把视觉主干、文本主干、多模态合流，以及 decoder 里的 MoE 改造拆开讲了一遍。

铺到这里，终于可以回答 Wall-X 最有“任务味”的那个问题了：

它到底是怎么生成动作的？

这件事很关键，因为 Wall-X 不是一个“最后顺手多接了个 action head”的文本模型。它在动作生成这条链上有自己明确的设计：

- 一条是 AR 路径，走离散 action token
- 一条是 flow 路径，走连续动作
- 而当前仓库里更核心、更完整的推理实现，明显在 flow 这条线上

所以这一篇我会把几个最容易混淆的问题一起说清楚：

- `model.generate()` 和 `batch_decode()` 到底有什么区别
- AR 动作路径到底在做什么
- `ActionProcessor` 在整条动作链里扮演什么角色
- flow 路径为什么不是“一次前向直接出动作”，而是要先加噪、再积分
- prefix prefill、KV cache 截断、postfix-only 迭代，到底分别解决了什么问题

## 先把最容易混的两件事拆开：生成和解码不是一回事

先从一个最小但最容易搞混的点开始。

很多人在看动作生成代码时，看到 `generate(...)`、`batch_decode(...)`、`predict_action` 这些名字混在一起，很容易下意识觉得它们都属于“模型在生成答案”。

其实不是。

在 Wall-X 里，至少要先把两件事分开：

- `model.generate(...)` 是模型真的在做自回归生成
- `processor.batch_decode(...)` 是把 token id 反查成字符串

在 `predict(predict_mode="text" / "fast")` 里，这条关系写得非常清楚：

1. 先调用 `self.generate(...)`
2. 得到 `predict_output_ids`
3. 再用 `self.processor.batch_decode(...)` 变成文本

所以如果只说一句话：

> `generate()` 负责“生成 token”，`batch_decode()` 负责“把 token 翻译成人能看的字符串”。

这一点在 AR 动作路径里尤其重要。因为 AR 模式下，模型先生成的是离散 action token 序列，而不是直接吐出连续动作张量。

## Wall-X 的动作路径，至少有两条

如果把动作生成这件事概括一下，Wall-X 当前仓库里可以看到两类明显不同的思路。

### 第一条：AR 路径

这条路径的关键特征是：

- 模型先像语言模型一样生成离散 token
- 这些 token 里有专门的 action token 区间
- 生成出来后，再通过 action tokenizer / action processor 解码成动作序列

这条路径在 `predict(predict_mode="fast")` 里能看到最完整的可读实现。

### 第二条：flow 路径

这条路径的关键特征是：

- 输入里直接放连续动作占位 token `<|action|>`
- 模型在这些位置上不是预测离散 id，而是预测连续动作相关的向量场
- 推理时从噪声开始，通过多步迭代逐渐把动作“推”出来

这条路径的核心实现就在 `generate_flow_action()`。

如果只看当前仓库的整体重心，其实很明显：

> Wall-X 的动作设计，主要是围绕 flow 这条连续动作路径展开的。

AR 路径更像保留了一条离散动作生成方案，而 flow 路径才是这里真正有较多工程设计投入的部分。

## 先看 AR：它本质上还是“生成 token，再解码成动作”

先把 AR 讲完，因为它更容易理解。

在 `predict(predict_mode="fast")` 里，动作生成的过程大致是这样：

1. 先把输入 prompt 截到 `<|im_start|>assistant` 之前
2. 调用 `self.generate(...)` 自回归往后生成
3. 拿到生成出的 token ids
4. 从中筛出 action token 那一段
5. 把这些离散 action token 解码回连续动作

这条链其实很像“文本生成 + 特殊词表映射”。

也就是说，AR 模式里，动作不是模型内部直接回归出来的，而是先走了一次离散化表达。你可以把它理解成：

- 先让模型“说出动作 token”
- 再让后处理逻辑把这段 token 翻译成真正动作

这条路的优点是直观、容易和语言模型生成框架对齐。  
但它也有一个天然限制：连续动作空间最后还是被离散 token 化了。

而这恰恰是 flow 路径要解决的问题。

## flow 路径为什么更像 Wall-X 的主力方案

如果你看 `ActionProcessor`、`flow_loss`、`generate_flow_action()`、`odeint(...)` 这些实现，就会发现 Wall-X 在连续动作这条线上明显投了更多心思。

原因也不难理解。

机器人动作天然是连续量：

- 末端位置是连续的
- 姿态是连续的
- gripper 开合虽然有时接近离散，但整体动作序列仍然更像连续控制问题

如果全都转成离散 token 去做，会有两个问题：

- 表达精度容易受限
- 时序上的平滑性和连续性也更难自然表达

所以 Wall-X 在 flow 路径里采取的思路是：

> 不直接离散化动作本体，而是把动作生成看成一个“从噪声逐步流到目标动作”的过程。

这时，模型学的就不再是“下一个离散 token 是什么”，而更像是在学“当前这一步，动作应该往哪个方向更新”。

## `ActionProcessor` 是动作链的核心中间层

理解 flow 路径，绕不过 `ActionProcessor`。

它在 Wall-X 里的职责，至少有 4 个：

1. 把 proprioception 投成状态 embedding  
2. 把 noisy action + timestep 投成 action embedding  
3. 把 transformer 输出再投回动作空间  
4. 在训练时构造 flow 目标并计算 flow loss  

也就是说，`ActionProcessor` 不是单一的 action head，它更像是动作侧的一整套接口层。

### 它怎么处理 proprioception

在动作模型里，proprioception 不是当普通文本喂进去的，而是会经过：

```python
proprioception_proj(...)
```

映射到状态 embedding 空间，再散射到 `<|propri|>` 对应的位置。

这一步我前面已经提过，但放到动作生成上下文里再看，会更清楚：

- 视觉给模型提供外部观测
- proprioception 给模型提供内部状态
- action token 给模型提供未来动作的占位或生成位点

这三者最后都是在统一序列里协同工作的。

### 它怎么处理动作

`ActionProcessor` 处理动作时，分训练和推理两个阶段。

训练时用的是 `forward(...)`。  
推理时更常用的是 `step(...)`。

这两个函数共同干的事情，本质上都是把：

- 当前 noisy action
- 当前 timestep
- 可选的 `dof_mask`

映射成一段 action embedding，再塞回统一序列里的 action token 位置。

这一点非常关键。因为它说明 flow 路径不是“模型外部自己在数值域里做扩散”，而是：

> 每一个时间步的 noisy action，都会先被编码成 token 级 embedding，然后再进入 Transformer 主干。

所以 flow 推理并不是绕开了大模型，而是每一步都还在借用同一条多模态 decoder 链。

## 训练时它到底学的是什么

这里可以顺手把训练目标再说清楚一点。

在 `ActionProcessor.forward(...)` 里，代码做了一个非常关键的构造：

```python
noise = torch.randn_like(action_chunk)
time = self.sample_time(...)
noisy_action = (1 - t) * noise + t * action_chunk
flow = action_chunk - noise
```

这几行就是整条 flow 动作路径的数学核心。

它的含义是：

- 先随机采一个时间 `t`
- 再在纯噪声和真实动作之间做线性插值，得到 `noisy_action`
- 同时把目标向量场定义成 `flow = action_chunk - noise`

这意味着模型真正学的，不是“直接输出 action_chunk”，而是：

> 给定当前的 noisy action 和时间 `t`，预测应该往哪个方向走，才能从噪声流向真实动作。

所以 Wall-X 的 flow 路径，本质上学的是一个速度场，或者更准确点说，是一个动作空间里的更新方向。

这也是为什么推理时最后能用 ODE / Euler 这类积分方法。

## `flow_loss` 为什么是 MSE

既然模型学的是速度场，那 loss 也就顺理成章了。

在 `flow_loss(...)` 里，逻辑大致是：

1. 先把 action token 对应的 hidden states 取出来
2. 通过 `action_proj_back` 投回动作维度
3. 得到 `v_pred`
4. 用 MSE 去拟合上面构造出来的 `flow`

也就是说，这里的回归目标不是“最终动作”，而是“当前时刻的速度向量”。

这和很多 diffusion / flow matching 风格的方法是一致的。  
也正因为这样，推理时模型可以从一个随机噪声初值开始，沿着预测的向量场一步一步往前推。

## 推理时为什么不是一次 forward 直接出动作

现在就可以回答一个很核心的问题了：

既然模型最终能输出动作，那为什么不一次 forward 直接预测最终动作？为什么还要搞 noisy action、多步迭代、ODE 积分？

答案就在上一节：

因为模型训练时学到的目标，本来就不是“最终动作本体”，而是“从当前 noisy action 出发，应该怎么更新”。

这就决定了推理时的自然形式不是一次回归，而是迭代积分。

换句话说：

- 如果训练目标是最终动作本体，一次 forward 回归是自然的
- 但 Wall-X 这里学的是向量场
- 那推理就自然要沿着这个向量场一步一步走

这也是 `generate_flow_action()` 存在的根本原因。它不是一个“花哨包装器”，而是训练目标决定下来的推理过程。

## `generate_flow_action()` 的第一阶段：先把整条序列预填充一遍

真正看 `generate_flow_action()`，第一感觉往往是：为什么这么长？

因为它不是只在做一个采样循环，它在做一整套“先 prefill，再复用 cache，再只迭代后缀”的优化版 flow 推理。

第一阶段非常关键，可以概括成：

1. 先构造初始噪声 `noise`
2. 令 `noisy_action = noise`
3. 取 `t=0`
4. 调用 `self.action_preprocessor.step(...)`，把当前 noisy action 编成 action embedding
5. 把这些 action embedding 填进输入序列的 action token 位置
6. 用完整输入跑一次 `self.model(...)`

这一遍完整前向，本质上是一次 prefix prefill。

为什么要这么做？

因为在 flow 迭代里，不是整个序列每次都会变化。大部分前缀上下文其实是固定的，真正会变的是 action token 对应的后缀区域。

所以 Wall-X 在第一步先跑完整条链，是为了把前缀上下文的 key/value cache 预热出来。

## prefill 之后，先做一次 Euler 起步

第一遍 prefill 跑完之后，代码会立刻从 hidden states 里取出 action token 的输出，然后通过：

```python
action_pred = self.action_preprocessor.action_proj_back(...)
```

得到当前的动作预测。

如果 `use_x_pred=False`，它直接把这个结果当成速度 `v_0`。  
如果 `use_x_pred=True`，则还会再减一次初始噪声。

然后代码马上做了一步：

```python
noisy_action = noisy_action + dt * v_0
```

这其实就是一次显式 Euler 更新。

换句话说，Wall-X 在真正进入后面的 ODE 迭代前，先用 prefill 的结果把初始点从 `t=0` 推到了 `t=dt`。

这一步很重要，因为它把：

- 初始完整上下文前向
- 第一个时间步更新

合并在了一起。

## 为什么要截断 KV cache

接下来是整条实现里最值得细看的工程细节之一。

prefill 输出完之后，代码会拿到 `prefix_kv_cache`。  
但它不会把整个序列的 cache 原样留着，而是会根据 `prefix_length` 把 cache 截断到 action 区域之前。

这一步的意义非常直接：

> 后面的迭代里，真正稳定不变的是前缀上下文；动作后缀每一步都在变，所以没必要把它们也当成“可复用 cache”。

所以 Wall-X 的做法是：

- 前缀上下文：保留 KV cache
- 后缀 action 区域：每步重新算

这就把后续迭代从“每次都重跑整条序列”优化成了“只重跑后缀，但还能看到固定前缀”。

这一步如果没有，flow 路径的推理成本会更大。

## postfix-only 迭代才是这条实现真正省算力的地方

cache 截断完之后，`generate_flow_action()` 会把后缀相关的数据全部单独切出来：

- `postfix_position_ids`
- `postfix_inputs_embeds`
- `postfix_attention_mask`
- `postfix_moe_token_types`
- `postfix_input_ids`

同时重新计算一套 postfix 版本的：

- `postfix_start_indices`
- `postfix_end_indices`

然后再构造 `_postfix_attention_mask`。

到这一步为止，Wall-X 已经把“完整序列推理”转换成了一种更高效的模式：

> 前缀只保留 cache，后缀每步动态更新 noisy action embedding 并重新跑一遍。

这就是后面 `step_with_kvcache(...)` 的核心。

在每个时间步里，它做的事情大致是：

1. 根据当前 `timestep` 和当前 `noisy_action` 生成新的 action embedding
2. 用这些 embedding 替换 postfix 里的 action token 位置
3. 带着 `prefix_kv_cache` 跑一次 `self.model(...)`
4. 只从 action token 位置取出 hidden states
5. 投回动作维度，得到当前速度 `v_t`

这条链就是典型的“后缀重算 + 前缀缓存复用”。

## 为什么这里会出现 `odeint(..., method="euler")`

到这里，前面的铺垫就全部接上了。

Wall-X 在 flow 推理里最终调用的是：

```python
odeint(step_with_kvcache, noisy_action, times[1:], method="euler")
```

这句话背后的含义其实不复杂：

- `step_with_kvcache` 给定当前 `t` 和 `noisy_action`，返回当前速度 `v_t`
- `odeint` 负责按时间网格把这个系统往前积分
- `method="euler"` 说明这里用的是最简单直接的欧拉法

为什么 Euler 就够了？

因为这里时间网格本来就是离散切分好的，而且模型每一步给出的也是当前速度估计。对于这种工程场景，Euler 通常已经是一个足够实用、足够稳定的选择。

更重要的是，Wall-X 的优化重点不在“积分器要多高阶”，而在“每一步 Transformer 前向要尽量少重算”。

这也是为什么这段实现的核心不在数值分析技巧，而在：

- prefix prefill
- KV cache 截断
- postfix-only 重算

## `times_cache` 和时间步缓存是个很典型的小优化

`generate_flow_action()` 里还有一个小地方也挺典型：

```python
if num_inference_timesteps not in self.times_cache:
    self.times_cache[num_inference_timesteps] = torch.linspace(...)
```

也就是说，不同的推理步数对应的时间网格会被缓存下来，避免每次推理都重新构造一遍。

这不是决定性优化，但它很符合这段代码的整体风格：

不是去发明一套完全新的推理框架，而是在现有模型链上，把每一处重复开销都尽量削掉。

## `dof_mask` 和 `padding_action` 是动作侧的工程细节

再说一个比较工程化，但在机器人场景里很有必要的点。

动作维度不一定总是全部有效。某些数据集、某些机械臂配置、某些任务场景里，部分自由度可能是无效的或者需要忽略的。

所以在 flow 路径里你会看到：

- `dof_mask`
- 可选的 `padding_action`

如果某些 DOF 不有效，代码会用 `padding_action` 对这些维度做替换，而不是盲目套用模型预测值。

这一步本身不是算法核心，但它说明 Wall-X 的动作生成不是只考虑“论文里的连续控制公式”，而是明确考虑了机器人控制维度在工程里经常不整齐这件事。

## 这条动作链真正特别的地方是什么

如果现在回头总结一下，Wall-X 的动作生成最特别的地方，其实不是“它会输出动作”，而是它把动作生成完整地嵌进了统一多模态序列和 decoder 主干里。

这意味着：

- 动作不是一个完全独立的小头在外面做
- 每个 timestep 的 noisy action 都会被编码成 token embedding
- 这些 action embedding 会和视觉、文本、状态一起进入同一条 Transformer 链
- 然后再通过动作专用投影映射回连续动作空间

所以你如果问“Wall-X 的动作头是接在哪的”，更准确的回答不是“接在最后”，而是：

> 它接在统一序列建模的闭环里。

这也是为什么前面几篇文章必须先讲视觉、文本、MoE 和统一序列，不然这一篇就会显得像凭空冒出来的另外一套系统。

## 把这一篇压成一句话

如果把这一篇的核心结论只压成一句话，那就是：

> Wall-X 的动作生成并不是简单的“最后接个回归头”，而是分成了 AR 和 flow 两条路径；其中更核心的 flow 路径会把 noisy action 在每个时间步编码成 action embedding，塞回统一序列，再利用 prefix prefill、KV cache 截断和 postfix-only Euler 积分，逐步从噪声推到最终动作。

这句话一旦理解了，前面那些实现细节就都能归到正确的位置上：

- 为什么 `ActionProcessor` 同时管状态、动作和 loss
- 为什么 flow 推理不是一次前向
- 为什么 KV cache 在这里依然有价值
- 为什么 Wall-X 看起来像“VLA 模型”，但动作生成这条链又明显是单独设计过的

下一篇就是这个系列的番外了。主线到这里，推理和结构部分已经基本闭环。番外篇我会单独讲训练路径，也就是：样本怎么被拼成训练 batch，`labels`、`action_chunk`、`moe_token_types` 怎么生成，以及 `cross_entropy_loss + flow_loss_weight * flow_loss` 这套双损失是怎么真正落到代码里的。

## 附：文中对应的关键源码位置

- 动作模型里的 `predict(...)`：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L1503)
- AR 路径里 `generate()` 和 `batch_decode()`：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L1753)
- flow 路径的简化版 `predict(predict_mode="diffusion")`：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L1852)
- 主力 flow 推理实现 `generate_flow_action(...)`：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L1959)
- prefix prefill 和第一次 Euler 更新：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L2144)
- prefix KV cache 截断：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L2221)
- postfix-only ODE 迭代：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L2293)
- `forward(mode="predict")` 和 `prepare_inputs_for_generation(...)`：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L2363)
- `ActionProcessor`：[`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L536)
- `sample_time(...)`：[`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L616)
- `proprioception_proj(...)`：[`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L633)
- 训练时的 `noisy_action` / `flow` 构造：[`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L668)
- 推理时的 `step(...)`：[`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L740)
- `flow_loss(...)`：[`action_head.py`](/Users/sam/project/github/wall-x/wall_x/model/action_head.py#L783)
