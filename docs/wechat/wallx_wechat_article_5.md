# Wall-X 是怎么生成动作的：AR、Flow、KV Cache 和 Euler 积分

> Wall-X 不是一个“最后顺手接个 action head”的模型。它的动作生成链，本身就是这套系统里单独设计过的一部分。

**TL;DR**
- Wall-X 至少有两条动作路径：AR 和 flow。
- `generate()` 负责生成 token，`batch_decode()` 负责把 token 翻成人类可读文本。
- AR 路径更像“先生成离散 action token，再解码成动作”。
- flow 路径更像“从噪声出发，沿着向量场一步步推到最终动作”。
- `generate_flow_action()` 的核心不只是采样，而是 `prefix prefill + KV cache 截断 + postfix-only Euler 迭代`。

## 一、先把一个最容易混的点拆开

很多人看动作生成代码时，会把 `generate()`、`batch_decode()`、`predict_action` 这些名字混成一团。

其实它们不是一回事。

最简单的区分是：

- `model.generate(...)`：模型真的在自回归生成 token
- `batch_decode(...)`：把 token id 反查成字符串

所以如果只压成一句话：

> `generate()` 是“生成”，`batch_decode()` 是“翻译”。

这在 AR 动作路径里尤其重要，因为 AR 模式下，模型先生成的是离散 action token，而不是直接吐连续动作张量。

## 二、Wall-X 的动作路径至少有两条

### 1. AR 路径

这条路径的思路很直接：

- 先像语言模型一样生成 token
- 这些 token 里有一段专门的 action token
- 再把它们解码回动作序列

如果你只看这条路，它会很像“文本生成 + 特殊词表映射”。

### 2. flow 路径

这条路径就不一样了。

它不是直接生成离散动作 token，而是：

- 在输入里放连续动作占位 `<|action|>`
- 每个时间步都把当前 noisy action 编成 embedding
- 再送进统一序列
- 让模型预测当前应该往哪个方向更新动作

从代码投入和完整度看，Wall-X 当前明显更重视这条 flow 路径。

## 三、AR 路径到底在做什么

AR 动作路径可以压成这样：

```text
prompt
-> generate()
-> 得到离散 token ids
-> 从中筛出 action token
-> 用 action tokenizer / processor 解码回连续动作
```

它的优点是：

- 非常符合语言模型生成范式
- 训练和推理接口直观

但它也有天然限制：

- 连续动作最后还是被离散 token 化了

这也正是 flow 路径存在的原因。

## 四、为什么 Wall-X 更像在押注 flow 路径

因为机器人动作天然是连续量：

- 位置是连续的
- 姿态是连续的
- 动作序列的平滑性也是连续控制问题的一部分

如果把它们全都塞进离散 token 空间，表达上总会受限。  
而 flow 路径更适合处理“从噪声逐步流到目标动作”这种连续生成问题。

从这个角度看，Wall-X 的动作设计其实很清楚：

> 它不是只想让大模型“说出动作 token”，而是想让模型真正学会连续动作怎么被生成出来。

## 五、`ActionProcessor` 是动作链的核心中间层

理解 flow 路径，绕不过 `ActionProcessor`。

它在 Wall-X 里至少干了 4 件事：

1. 把 proprioception 投成状态 embedding
2. 把 noisy action + timestep 投成 action embedding
3. 把 action hidden states 再投回动作空间
4. 在训练时构造 flow 目标并计算 flow loss

所以它不是一个狭义的 action head，更像动作侧的一整套接口层。

## 六、它训练时到底学什么

Wall-X 动作训练最核心的几行，其实非常简单：

```python
noise = torch.randn_like(action_chunk)
time = sample_time(...)
noisy_action = (1 - t) * noise + t * action_chunk
flow = action_chunk - noise
```

这几行决定了它训练时真正学的不是“最终动作本体”，而是：

> 给定当前 noisy action 和当前时间 `t`，应该往哪个方向更新，才能从噪声流向真实动作。

所以 Wall-X 的 flow 路径，本质上学的是动作空间里的向量场。

这也是为什么后面推理时会自然出现 Euler / ODE 积分。

## 七、`flow_loss` 为什么是 MSE

既然模型学的是“更新方向”，那监督目标也就顺理成章了。

动作 token 位置上的 hidden states 会被投回动作空间，得到当前预测的速度 `v_pred`，然后用 MSE 去拟合上面构造好的 `flow`。

也就是说，flow loss 监督的不是“最终动作”，而是“当前时刻该怎么动”。

这和扩散 / flow matching 这一路的方法是完全一致的。

## 八、推理时为什么不是一次 forward 直接出动作

现在就可以回答一个很关键的问题：

既然最终要的是动作，为什么不一次前向直接回归动作？

答案很简单：

- 因为训练时学的目标，本来就不是最终动作
- 而是向量场

既然学的是向量场，推理时自然就不是一次回归，而是从初值开始迭代积分。

换句话说：

- 如果训练学的是 `action`
- 一次回归很自然
- 但如果训练学的是 `d(action)/dt`
- 推理就自然要一步一步往前推

这就是 `generate_flow_action()` 存在的根本原因。

## 九、`generate_flow_action()` 为什么这么长

很多人第一次看这段函数，最大的感受都是：怎么这么长？

因为它不是只在做一个采样循环，而是在做一整套针对 flow 推理的工程优化。

最关键的结构可以压成这样：

```text
初始化 noise
-> prefix prefill
-> 第一次 Euler 更新
-> 截断 prefix KV cache
-> 只对 postfix action 区域反复迭代
-> odeint(..., method="euler")
```

理解这条链，Wall-X 的动作推理就顺了。

## 十、第一阶段：prefix prefill

flow 推理不是一上来就进入循环，而是先做一次完整前向。

这一步大致是：

1. 令 `noisy_action = noise`
2. 用 `ActionProcessor.step(...)` 把当前 noisy action 编成 action embedding
3. 把这些 embedding 填到 action token 位置
4. 用完整输入跑一遍模型

这一遍完整前向的意义，不只是“看一下当前输出”，更重要的是：

> 把前缀上下文的 KV cache 预热出来。

因为后面真正会变化的，主要是 action token 对应的后缀区域，前缀上下文其实是稳定的。

## 十一、prefill 之后，先走一步 Euler

第一次完整前向跑完之后，代码会立刻从 action token 位置取出 hidden states，再投回动作空间，得到当前速度 `v_0`。

然后马上做一次：

```python
noisy_action = noisy_action + dt * v_0
```

这其实就是一次显式 Euler 更新。

你可以把它理解成：

- prefill 负责拿到第一个可靠方向
- 然后先往前走一小步

这一步之后，模型就可以进入后面的 cache 复用阶段了。

## 十二、为什么要截断 KV cache

这是整条实现里最有工程味的一步。

prefill 之后，代码拿到了整条序列的 KV cache。  
但它不会把整条 cache 原样带进后续循环，而是会把 cache 截断到 action 区域之前。

原因很直接：

- 前缀上下文是稳定的，适合缓存
- 后缀 action 区域每一步都在变，不适合直接复用

所以最合理的做法就是：

- 保留 prefix cache
- 每一步只重算 postfix

这正是 Wall-X 后面能把 flow 推理做得更高效的关键。

## 十三、真正省算力的是 postfix-only 迭代

截断完 cache 之后，代码会把后缀相关的东西单独切出来：

- `postfix_inputs_embeds`
- `postfix_position_ids`
- `postfix_attention_mask`
- `postfix_moe_token_types`

后面的每一步都只干一件事：

1. 用当前 `t` 和当前 `noisy_action` 生成新的 action embedding
2. 把这些 embedding 填回 postfix 的 action token 位置
3. 带着 prefix cache 跑一次模型
4. 只读 action token 位置上的输出
5. 得到当前速度 `v_t`

这就是“前缀缓存，后缀重算”的标准结构。

所以 Wall-X 的 flow 推理重点不是“积分器多高级”，而是“每一步 Transformer 前向尽量少重算”。

## 十四、为什么最后会出现 `odeint(..., method="euler")`

现在前面的铺垫就都能接上了。

模型每一步给出的不是最终动作，而是当前速度 `v_t`。  
那要从噪声走到最终动作，最自然的方法就是积分。

Wall-X 这里调用的是：

```python
odeint(step_with_kvcache, noisy_action, times[1:], method="euler")
```

这说明：

- 数学上它确实把这件事看成一个连续时间系统
- 工程上它用了最朴素、最稳定、也最够用的 Euler 方法

对 Wall-X 这种场景来说，这个选择很务实。  
因为真正的瓶颈不在“高阶积分器”，而在每一步模型前向开销。

## 十五、这条动作链为什么值得单独看

Wall-X 的动作生成最特别的地方，并不是“它会输出动作”，而是：

> 它把动作生成完整嵌进了统一多模态序列和 decoder 主干里。

这意味着：

- 动作不是一个完全独立的小头
- 每个 timestep 的 noisy action 都会先变成 token 级 embedding
- 然后和视觉、文本、状态一起进入同一条 Transformer 链
- 最后再通过动作投影回到连续动作空间

这就是为什么你看起来像是在分析一个 VLM，但最后会碰到一整套连续动作生成工程。

## 十六、这篇文章真正要立住的判断

如果把这篇压成一句话，那就是：

> Wall-X 的动作生成不是“最后接个回归头”这么简单，而是分成了 AR 和 flow 两条路径；其中更核心的 flow 路径会把 noisy action 编成 token embedding 塞回统一序列，再通过 `prefix prefill + KV cache 截断 + postfix-only Euler 迭代`，一步步从噪声推到最终动作。

下一篇就是这个系列的训练篇。到那时，我们会把“这些能力到底是怎么训练出来的”补完整。

