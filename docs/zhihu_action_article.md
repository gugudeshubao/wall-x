# Wall-X 是怎么生成动作的：Flow Matching、ODE 积分与 Action Token 设计

> 前几篇文章全在讲"怎么让推理更快"——FA2、C++、SDPA、量化。但一直没有回答一个根本问题：**Wall-X 到底怎么从一张图片和一条指令，生成机器人的关节动作？** 这篇专门讲 Action。

**TL;DR**
- Wall-X 有两条动作路径：**AR（自回归离散 token）** 和 **Flow（连续流匹配）**。当前主力是 Flow 路径
- Flow 路径不是"transformer 直接回归动作值"，而是 **transformer 预测速度场 v(x,t)，再用 ODE 积分推出最终动作**
- 动作 token 不是独立的 head，而是 **直接嵌入到统一多模态序列中**——和 text、image、proprioception token 一起送进 transformer
- 核心技巧：**Prefix-Postfix KV Cache 分离**——静态上下文只算一次，ODE 每步只重算 action 区域（32 tokens）
- ActionProcessor 有 3 层 MLP（w1→w2→w3）+ 正弦时间编码，负责 `(noisy_action, timestep) → action embedding`
- 归一化系统：per-dataset min/delta 映射到 [-1,1]，支持 DOF mask 选择性控制
- **量化风险点**：ODE 5-10 步积分会累积量化误差，Action Head 建议保持 bf16

---

## 一、两条动作路径：AR vs Flow

Wall-X 的代码里其实有两条动作生成路径：

### 1.1 AR 路径：像生成文本一样生成动作

```
prompt → generate() → 离散 action token ids → decode → 动作序列
```

这条路径复用了标准的自回归文本生成框架（HuggingFace `generate()`）。模型在词表中添加了特殊的 action token，逐个生成后再解码回连续动作。

**优点**：框架简单，复用所有文本生成的基础设施（beam search、sampling strategy 等）
**缺点**：离散化会损失精度；生成长度和动作维度耦合；无法利用动作空间的连续性

### 1.2 Flow 路径：从噪声出发，沿速度场积分

```
noise → ODE 积分 → 最终动作
每一步：transformer 预测 v(x_t, context, t) → Euler 更新 x_{t+dt} = x_t + dt * v_t
```

这是 Wall-X **当前的主力路径**。它不生成离散 token，而是在连续空间中，通过学习一个条件速度场，从随机噪声"推"到目标动作。

下面所有内容都在讲这条 Flow 路径。

---

## 二、Flow Matching：比扩散模型更简单的连续生成

### 2.1 核心思想

Flow Matching 定义了一条从噪声到数据的直线路径：

```
x_t = (1 - t) * x_0 + t * x_1

其中：
  x_0 = 随机噪声（标准正态分布）
  x_1 = 目标动作（训练数据）
  t ∈ [0, 1]
```

这条路径的速度（导数）是常数：

```
v(x_t) = dx_t/dt = x_1 - x_0 = action_chunk - noise
```

**训练目标**：让模型学会预测这个速度场 `v_θ(x_t, context, t) ≈ x_1 - x_0`

**推理过程**：从纯噪声 `x_0` 出发，用学到的速度场做 ODE 积分，一步步推到 `x_1`

### 2.2 和 Diffusion 的区别

| 特性 | Diffusion (DDPM) | Flow Matching |
|------|------------------|---------------|
| 前向过程 | 逐步加高斯噪声 | 线性插值 |
| 路径形状 | 弯曲（需要 ~50 步） | 直线（5-10 步即可） |
| 训练目标 | 预测噪声 ε | 预测速度 v |
| 噪声调度 | 需要手工设计 β schedule | 不需要 |
| 推理步数 | 多（20-100） | **少（5-10）** |

**Flow Matching 最大的实用优势是步数少**。Wall-X 推理只需 **5 步 Euler 积分**，而基于 DDPM 的方法通常需要 20-50 步。这对实时机器人控制至关重要。

### 2.3 时间步采样

训练时 timestep 不是均匀采样，而是用 **Beta(1.5, 1.0)** 分布：

```python
# wall_x/model/action_head.py
self.beta_dist = Beta(alpha=1.5, beta=1.0)
sample = self.beta_dist.sample([batch_size])
time = (1 - sample) * 0.999  # 缩放到 [0, 0.999]
```

Beta(1.5, 1.0) 分布偏向小值——**模型在训练时更多地见到 t 接近 0 的样本**（噪声更多的状态），这让早期去噪步骤更准确。

推理时 timestep 是均匀网格：`[0, 0.2, 0.4, 0.6, 0.8, 1.0]`（5 步 Euler）。

---

## 三、统一序列：Action Token 直接嵌入 Transformer

Wall-X 的一个关键设计是：**动作不是 transformer 之后的附加 head，而是直接作为 token 嵌入到统一多模态序列中。**

### 3.1 序列结构

```
[text tokens] [image tokens] [proprio token] [action tokens × 32]
 ← 普通 token →                               ← action_token_id →

总长度 ≈ 420-488 tokens
其中 action tokens 固定 32 个（= action_horizon）
```

每个 action token 用特殊 token id `<|action|>` 占位。在 forward 之前，这些位置的 embedding 被替换为 ActionProcessor 生成的连续 action embedding。

### 3.2 为什么不用独立的 Action Head？

传统做法是 transformer 出 hidden states 后，接一个独立的 MLP head 映射到动作空间。Wall-X 把 action token 放进序列的好处：

1. **Action 能 attend to 全部上下文**——图像、文本指令、机器人状态，一个 attention 全看到
2. **上下文也能 attend to action**——transformer 知道当前要生成什么动作，可以反向调节理解
3. **复用 MoE 路由**——action token 会被路由到专门的"动作 expert"（intermediate_size=2048），而文本 token 走"语言 expert"（intermediate_size=11008）
4. **ODE 循环复用 KV Cache**——和标准 decode 共享同一套 KV Cache 机制

---

## 四、ActionProcessor：连续动作 ↔ Token Embedding 的桥梁

ActionProcessor 是 action token embedding 的核心模块，负责将 `(noisy_action, timestep)` 编码为 transformer 能理解的 embedding。

### 4.1 架构

```
输入：noisy_action [batch, 32, action_dim]  +  timestep [batch]

noisy_action ──→ [w1 Linear] ──→ action_embed [batch, 32, action_hidden_size]
                                        ↓
timestep ──→ [SinusoidalPosEmb] ──→ time_embed [batch, action_hidden_size]
                                        ↓ repeat × 32
                                        ↓
                              [concat(action_embed, time_embed)]
                                        ↓
                                   [w2 Linear]
                                        ↓
                                     [SiLU]
                                        ↓
                                   [w3 Linear]
                                        ↓
                              action_time_embed [batch, 32, action_hidden_size]
                                        ↓ (padding to hidden_size if needed)
                              → 替换 <|action|> 位置的 embedding
```

**关键参数**：
- `action_dim`：动作维度，由 `dof_config` 配置决定（各关节维度之和）
- `action_hidden_size`：通常 = 2048（和 transformer hidden_size 一致）
- `w1`：`action_dim → action_hidden_size`
- `w2`：`action_hidden_size × 2 → action_hidden_size`（concat 了 time_embed）
- `w3`：`action_hidden_size → action_hidden_size`

### 4.2 正弦时间编码

时间步编码沿用了经典的 sinusoidal positional embedding：

```python
class SinusoidalPosEmb(nn.Module):
    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
```

标量 timestep `t ∈ [0, 1]` 被编码为 2048 维向量，和 action embedding concat 后送入 w2。

### 4.3 DOF Mask：选择性控制

不同机器人有不同的关节配置。`dof_mask` 是一个 `[batch, horizon, action_dim]` 的 binary tensor，指示哪些自由度是活跃的。

当 `proj_with_mask=True` 时，dof_mask 会被 concat 到 noisy_action 上一起送入 w1：

```python
# w1 输入维度变为 action_dim * 2
noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)
action_embed = self.w1(noisy_action)  # w1: (action_dim*2) → action_hidden_size
```

这让模型能够**根据哪些关节被控制来调整 embedding**，而不是无差别地编码所有维度。

### 4.4 AdaRMS：可选的自适应时间调节

除了默认的 concat 时间编码路径，Wall-X 还支持 AdaRMS（Adaptive RMS Normalization）路径：

```python
if self.config.use_adarms:
    # 时间信息不 concat，而是作为独立的条件信号
    time_embed = self.time_mlp_in(time_embed)
    time_embed = self.act_fn(time_embed)
    time_embed = self.time_mlp_out(time_embed)
    adarms_cond = time_embed  # 传给 transformer 层做自适应归一化
```

AdaRMS 的思想是：时间信息不直接混入 action embedding，而是作为**自适应归一化的条件信号**传给 transformer 的每一层。

---

## 五、推理全流程：Prefix-Postfix KV Cache + Euler ODE

这是 Wall-X Flow Action 推理最精巧的部分。

### 5.1 总览

```
Phase 1: Prefill
  [text + image + proprio + action(t=0)] → 完整 forward → 得到 v_0, KV Cache

Phase 2: KV Cache 截断
  保留 prefix（text + image + proprio）的 KV Cache
  丢弃 action 区域的 KV Cache

Phase 3: ODE 循环（5 步）
  for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
    1. ActionProcessor.step(t, noisy_action) → 新 action embedding
    2. 只替换 postfix 中的 action embedding
    3. Transformer forward（复用 prefix KV Cache + 新 postfix）
    4. 投影 → v_t
    5. noisy_action += dt * v_t  (Euler step)

Phase 4: 反归一化
  noisy_action → unnormalize → 机器人关节角度/位置
```

### 5.2 Phase 1：Prefill + 第一步 Euler

```python
# 1. 初始化噪声
noise = torch.randn(batch, action_horizon, action_dim)
noisy_action = noise  # t=0 时就是纯噪声

# 2. t=0 的 action embedding
times = torch.linspace(0, 1, num_timesteps + 1)  # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
action_embed, adarms_cond = action_processor.step(times[0], noisy_action, dof_mask)

# 3. 替换 <|action|> 位置的 embedding
inputs_embeds[action_mask] = action_embed.reshape(-1, hidden_size)

# 4. 完整 prefill forward
output = transformer(inputs_embeds, position_ids, ...)
past_key_values = output.past_key_values

# 5. 提取 action hidden states → 投影出 v_0
action_hs = output.hidden_states[action_mask]
v_0 = action_proj_back(action_hs[:, :action_hidden_size])

# 6. 第一步 Euler 更新
dt = times[1] - times[0]  # 0.2
noisy_action = noisy_action + dt * v_0
```

### 5.3 Phase 2：KV Cache 截断

```python
# 找到第一个 action token 的位置
prefix_length = (input_ids == action_token_id).nonzero()[0, 1]

# 截断 KV Cache：只保留 prefix 部分
for layer in past_key_values:
    layer.key_cache = layer.key_cache[:, :, :prefix_length, :]
    layer.value_cache = layer.value_cache[:, :, :prefix_length, :]

# 切出 postfix 信息
postfix_embeds = inputs_embeds[:, prefix_length:, :]
postfix_position_ids = position_ids[:, prefix_length:]
```

**这是性能优化的关键**：静态上下文（图片 + 文本 + 机器人状态）有 ~420 个 token，只需算一次。ODE 每步只需重算 postfix（~32 个 action token）。

### 5.4 Phase 3：ODE Euler 循环

```python
def velocity_fn(t, noisy_action):
    # 1. 生成当前步的 action embedding
    action_embed, adarms_cond = action_processor.step(t, noisy_action, dof_mask)

    # 2. 替换 postfix 中的 action token
    postfix_embeds_new = postfix_embeds.clone()
    postfix_embeds_new[action_mask_postfix] = action_embed.reshape(-1, hidden_size)

    # 3. Transformer forward（复用 prefix KV Cache）
    output = transformer(
        inputs_embeds=postfix_embeds_new,  # 只有 32 个 token
        past_key_values=prefix_kv_cache,    # 复用 420 个 token 的 cache
        position_ids=postfix_position_ids,
    )

    # 4. 提取速度场
    action_hs = output.hidden_states[action_mask_postfix]
    v_t = action_proj_back(action_hs[:, :action_hidden_size])
    return v_t

# Euler 积分
for i in range(1, len(times) - 1):  # times[1] ~ times[-1]
    dt = times[i+1] - times[i]
    v_t = velocity_fn(times[i], noisy_action)
    noisy_action = noisy_action + dt * v_t
```

### 5.5 Phase 4：反归一化

```python
# 从 [-1, 1] 映射回机器人关节空间
predict_action = normalizer.unnormalize(noisy_action, dataset_name)
# unnormalize: (x + 1) / 2 * delta + min
```

### 5.6 C++ 实现的等价代码

```cpp
// cpp_infer/src/model.cpp
auto [postfix_cos, postfix_sin] = compute_rotary_emb(postfix_position_ids, ...);
auto ode_buf = postfix_embeds.clone();

for (int t = 0; t < num_timesteps; t++) {
    auto action_embed = action_head_.step(timestep[t], noisy_action, dof_mask);
    ode_buf.copy_(postfix_embeds);
    ode_buf.index_put_({action_mask}, action_embed);
    auto h = transformer_forward_postfix(ode_buf, postfix_cos, postfix_sin, ...);
    auto v_t = action_head_.action_proj_back(h);
    noisy_action += dt * v_t;
    kv_cache_.truncate(prefix_len);
}
auto action = action_head_.unnormalize(noisy_action, dataset_name);
```

C++ 版的额外优化：
- **预计算 rotary embedding**——cos/sin 在 ODE 循环外算一次
- **预分配 buffer**——`ode_buf.copy_()` 替代 `clone()`
- **无 Python dispatch 开销**——每步 ODE 从 86.5ms 降到 26.2ms

---

## 六、归一化系统：Per-Dataset Min/Delta

### 6.1 为什么需要归一化

不同机器人的关节范围差异巨大：

```python
# wall_x/utils/constant.py 示例
DOF_STATS = {
    "follow_left_ee_cartesian_pos": {  # 左臂末端位置
        "min": [-0.5, -0.3, 0.0],      # 米
        "delta": [1.0, 0.6, 0.5],
    },
    "follow_left_gripper_position": {   # 夹爪开合
        "min": [0.0],                   # 归一化
        "delta": [1.0],
    },
}
```

如果直接用原始值，不同关节的数值尺度差 100 倍以上，训练不稳定。

### 6.2 归一化/反归一化公式

```
归一化:    y = (x - min) / delta * 2 - 1     → 映射到 [-1, 1]
反归一化:  x = (y + 1) / 2 * delta + min     → 映射回原始空间
```

Wall-X 的 Normalizer 支持 **per-dataset** 的 min/delta 参数，存储为 `nn.ParameterDict`：

```python
class Normalizer(nn.Module):
    def __init__(self):
        self.min = nn.ParameterDict()    # {dataset_name: tensor}
        self.delta = nn.ParameterDict()  # {dataset_name: tensor}
```

这意味着同一个模型可以处理不同机器人的数据——只要归一化参数不同。

### 6.3 DOF Mask 在反归一化中的作用

反归一化时，dof_mask 用于选择性地只还原活跃的自由度：

```python
def unnormalize_data(self, action, dataset_name, dof_mask=None):
    min_val = self.min[dataset_name]
    delta_val = self.delta[dataset_name]
    action = (action + 1) / 2 * delta_val + min_val
    if dof_mask is not None:
        action = action * dof_mask  # 非活跃 DOF 归零
    return action
```

---

## 七、Flow Action 对量化的特殊要求

这一点在前几篇的部署优化中反复提到，值得在 Action 的语境下详细解释。

### 7.1 ODE 积分会放大误差

```
Step 1: x_1 = x_0 + dt * v_θ(x_0, t_0)        误差 ε_1
Step 2: x_2 = x_1 + dt * v_θ(x_1, t_1)        误差 ε_1 + ε_2（x_1 已有误差）
Step 3: x_3 = x_2 + dt * v_θ(x_2, t_2)        误差 ε_1 + ε_2 + ε_3 + 交叉项
...
Step 5: 累积误差 ≈ O(5ε)  ← 线性累积！
```

如果 `action_proj_back` 被量化到 INT8，每步预测的速度 v_t 会有量化噪声。经过 5 步 Euler 积分，这些噪声会**线性累积**。而机器人关节的容差通常很小（毫米级），微小的累积偏差可能导致抓取失败。

### 7.2 建议：Action Head 保持 bf16

```
可以量化的部分：
  ✓ Transformer 36 层 decoder（GEMM 是带宽瓶颈，量化收益大）
  ✓ ViT 32 层 encoder（同理）

不建议量化的部分：
  ✗ ActionProcessor (w1, w2, w3)
  ✗ action_proj_back
  ✗ Normalizer
```

Action Head 的参数量很小（w1 + w2 + w3 + action_proj_back ≈ 几 MB），量化它省不了多少计算，但风险很高。

---

## 八、性能剖析：Action 在推理延迟中的占比

基于 C++ 推理引擎的实测数据（Orin SM 8.7, bf16）：

| 阶段 | 时间 (ms) | 占比 | 备注 |
|------|-----------|------|------|
| ViT encoding | 220.6 | 39.8% | 32 层 ViT，和 action 无关 |
| Prefill | 199.7 | 36.1% | 488 tokens 完整 forward，含初始 action embedding |
| **ODE 5步** | **131.0** | **23.7%** | **action 生成的核心开销** |
| 每步 ODE | 26.2 | — | 32 tokens postfix forward + 投影 |
| **总计** | **553.9** | 100% | — |

**ODE 占 24%，是 action 生成的"价格"。** 相比直接用 AR 生成 32 个动作 token（每个 ~26ms × 32 = ~840ms），Flow 的 5 步 ODE（131ms）快了 **6.4 倍**——这是 Flow Matching "步数少"的直接体现。

---

## 九、总结

Wall-X 的 Action 系统不是一个简单的"regression head"，而是一套完整的设计：

| 组件 | 功能 | 设计理由 |
|------|------|---------|
| **Flow Matching** | 连续动作生成方法 | 比 diffusion 步数少 5-10x，适合实时控制 |
| **统一序列 Action Token** | 动作嵌入 transformer 序列 | 全注意力访问多模态上下文 |
| **ActionProcessor (w1→w2→w3)** | noisy_action + timestep → embedding | 连续动作到 token 空间的桥梁 |
| **Prefix-Postfix KV Cache** | 静态上下文只算一次 | ODE 5 步只需 5 次 postfix forward |
| **Per-dataset Normalizer** | 动作归一化/反归一化 | 支持多机器人多数据集 |
| **DOF Mask** | 选择性关节控制 | 不同机器人不同自由度配置 |
| **Euler ODE 积分** | 5 步从噪声推到动作 | 简单、快速、够用 |

从部署角度看，这套系统的瓶颈很明确：**ODE 循环中 5 次 transformer postfix forward**。我们在 C++ 引擎中已经把这个从 Python 的 432ms 压到 131ms（3.3x 加速）。下一步 INT8 量化将继续压缩 transformer 的 GEMM 时间——但 Action Head 本身会保持 bf16 精度。

---

*测试环境：wall-oss-flow 3B 模型，bfloat16 精度，batch_size=1。Jetson AGX Orin 64GB，JetPack 6.2.1，CUDA 12.6，PyTorch 2.5.0a0。C++ 推理引擎实测数据。Flow Action 配置：action_horizon=32，num_inference_timesteps=5，Euler 积分。*
