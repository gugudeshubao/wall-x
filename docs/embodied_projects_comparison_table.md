# 具身开源项目横向对比表

这份文档的目的，是把当前最值得看的几类具身项目压成一个可快速扫描的对照表。  
重点不在“信息最全”，而在“从 `wall-x` 出发，应该比较什么”。

当前纳入对比的项目：

- `wall-x`
- `openpi`
- `OpenVLA`
- `Diffusion Policy`
- `ACT`
- `Octo`
- `LeRobot`

---

## 一、快速对比表

| 项目 | 主干类型 | 动作表示 | 复杂度主要落点 | runtime 风格 | 在 Orin 上最先会炸的点 |
|------|----------|----------|----------------|--------------|------------------------|
| **wall-x** | `Qwen2.5-VL + MoE + Action` | `Flow Action` 为主，也有 `AR` 路径 | `MoE + multimodal + action loop + runtime` | 更偏底层张量/向量原语，手工 runtime | **小算子 fusion + ODE/postfix loop + KV cache + MoE 路由** |
| **openpi** | 机器人 `VLA` / policy family | `flow` 与 `AR` 并存（如 `π0 / π0-FAST`） | 动作路径设计 + 系统组织 | 可能同时需要模型层和系统层视角 | **先看动作循环怎么组织，再看部署假设** |
| **OpenVLA** | 标准 `VLA` 主干 | 多数是 token / action head 风格 | 模型结构与机器人任务接口 | 更像典型 VLA 推理栈 | **大算子 + multimodal serving，随后是系统工程** |
| **Diffusion Policy** | 纯动作 policy 主干 | `diffusion` | action rollout / denoise loop | 不一定重 VLM，但重 rollout | **迭代动作循环本身** |
| **ACT** | 较轻量动作模型 | `action chunking` | 动作表示本身 | runtime 相对简单 | **不是大算子，而是机器人控制接口/数据链** |
| **Octo** | generalist robot policy | 通常是 policy-style action head | representation + generalization | 更像大一统 robot policy 框架 | **不一定先炸 runtime，可能先炸系统集成和训练/数据规模假设** |
| **LeRobot** | 基础设施 / 机器人 ML 工程层 | 不绑定单一动作表示 | dataset / robot interface / train-eval pipeline | 更偏平台层 | **模型之外的工程复杂度** |

---

## 二、怎么理解这张表

这张表最重要的不是具体名词，而是四个维度：

### 1. 主干类型

先分清它到底是：

- `VLA`
- `generalist robot policy`
- `纯动作 policy`
- 还是“机器人 ML 基础设施”

这个区分决定了你后面是该盯模型结构，还是盯系统组织。

### 2. 动作表示

动作表示会直接决定后续复杂度长什么样：

- `AR` 更像语言生成延伸
- `flow / diffusion` 会把 repeated forward / rollout 问题重新抬起来
- `action chunking` 通常结构更简单，但表达能力和控制方式不同

所以这个维度非常关键。

### 3. 复杂度主要落点

不是所有项目的难点都在同一个地方。

有的难点在：

- 主干模型本身
- 动作生成机制
- rollout / sampling loop
- 多机器人 / 多数据集接口
- runtime / deployment

如果不先判断复杂度主要落点，就很容易把注意力浪费在不值钱的地方。

### 4. runtime 风格

这里重点不是“它用了什么框架”，而是它默认依赖的能力边界。

大致可以分成两种：

- 更偏底层张量/向量原语接口
- 更偏计算图接口 / 编译器接口

这个维度和你前面第五篇结尾里讨论的：

> 向量接口 vs 计算图接口

其实是一回事。

---

## 三、从 wall-x 出发，最有价值的几组对照

### 1. wall-x vs OpenVLA

这组最适合回答：

- 哪些东西是 `VLA` 项目的共性
- 哪些是 `wall-x` 因为 `MoE + Flow Action` 才额外长出来的复杂度

如果你想知道“`wall-x` 到底特殊在哪”，这组对照最直接。

### 2. wall-x vs openpi

这组最适合回答：

- `Flow` 路径和 `AR` 路径到底怎么分化
- 动作生成方式一变，runtime 主战场会怎么变

如果你想把“动作路径”这个问题看透，这组优先级最高。

### 3. wall-x vs Diffusion Policy

这组最适合回答：

- `flow / diffusion / repeated forward` 自己会引入哪些复杂度
- 哪些问题并不是 `VLM` 带来的，而是动作迭代本身带来的

这组能帮你把“动作生成复杂度”和“多模态复杂度”拆开。

### 4. wall-x vs ACT

这组最适合回答：

- 如果不用 `flow / diffusion`，动作模型能简化多少
- 哪些 runtime 问题是复杂动作表示才会放大的

`ACT` 是一个很好的“轻量下限对照”。

### 5. wall-x / OpenVLA / Octo

这组三方对照最适合回答：

- `VLA` 和 `generalist robot policy` 的边界到底在哪里
- 模型复杂度、系统复杂度和泛化目标是怎么互相牵制的

---

## 四、如果只按“下一步值不值得看”排序

如果目标是继续提升你现在的判断框架，而不是“把所有项目都知道一遍”，那推荐优先级还是：

1. `openpi`
2. `OpenVLA`
3. `Diffusion Policy`
4. `ACT`
5. `Octo`
6. `LeRobot`

理由是：

- `openpi / OpenVLA` 和你当前的 `wall-x` 视角最近
- `Diffusion Policy / ACT` 能迅速把动作生成问题单独剥开
- `Octo / LeRobot` 更适合在建立完前面对照后，再抬高系统视角

---

## 五、当前最值得保留的判断

后面如果继续深挖，最值得保留的判断大概有这些：

1. `wall-x` 不是理解具身项目的终点，而是一个很好的起点。
2. 真正有价值的不是单独看某个项目，而是建立跨项目对照视角。
3. 动作表示方式会决定后续 runtime 和 deployment 的主要矛盾。
4. `VLA` 项目和 `policy` 项目，看起来都在“输出动作”，但系统复杂度来源并不一样。
5. 下一步最该建立的，是“模型复杂度、动作复杂度、runtime 复杂度、系统工程复杂度”这四层之间的分界感。
