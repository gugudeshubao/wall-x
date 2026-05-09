# 下一个该看的具身开源项目清单

这份文档的目标不是做“最全项目列表”，而是基于当前对 `wall-x` 的理解，给出一个更高效的下一步阅读顺序。

核心前提是：

> 现在继续只盯 `wall-x`，边际收益已经开始下降。  
> 更有价值的下一步，是通过看别的具身项目，建立“哪些做法是共性，哪些只是 `wall-x` 自己的选择”。

---

## 一、现在为什么适合跳出去看别的项目

当前对 `wall-x` 的理解，已经足够支撑进入对照阅读阶段。

已经比较清楚的部分包括：

- `wall-x` 的主干本质上还是 `Qwen2.5-VL + MoE + action/flow`
- `VQA` 和 `Flow Action` 的分叉位置
- 哪些复杂度来自模型，哪些来自 runtime
- `cpp_infer` 里哪些是标准层，哪些是 fused op，哪些是 quant 路径
- 为什么优化路线会自然从 attention 走到 C++、量化、fusion、graph、CUTLASS

也就是说，下一步再看别的项目时，已经不会再被“类很多、文件很多、模块很多”带着走，而是会直接抓：

- 主干到底是什么
- 动作链怎么接
- runtime 假设是什么
- 真实瓶颈会出在哪里
- 它到底是在改模型，还是在改系统

---

## 二、推荐顺序

如果只给一个推荐顺序，我会建议这样看：

1. `openpi`
2. `OpenVLA`
3. `Diffusion Policy`
4. `ACT`
5. `Octo`
6. `LeRobot`

这个顺序的逻辑是：

- 前两个离 `wall-x` 最近，适合先做“VLA 级对照”
- 中间两个把“动作生成”问题单独剥开
- 最后两个帮你建立更大的具身系统全景图

---

## 三、第一梯队：直接对照 wall-x

### 1. openpi

- 仓库：<https://github.com/Physical-Intelligence/openpi>
- 最适合对照的问题：
  - 它的动作生成主线到底更像 `Flow`、`AR`，还是混合系统
  - `π0` 和 `π0-FAST` 这两条路分别在解决什么
  - 和 `wall-x` 比，哪些复杂度在模型，哪些复杂度在 runtime

为什么优先级高：

`openpi` 最大的价值，在于它把不同风格的 VLA 路线都摆在了一个项目体系里。你现在刚好对 `wall-x` 里的 `Flow Action` 非常敏感，拿它和 `π0 / π0-FAST` 去对照，会很快建立“动作生成路径”的比较框架。

### 2. OpenVLA

- 仓库：<https://github.com/openvla/openvla>
- 最适合对照的问题：
  - 它的 VLA 主干到底有多“标准”
  - Action head 是怎么接到 VLM 主干上的
  - 它的复杂度主要落在模型结构，还是落在部署 / serving

为什么第二个看：

`OpenVLA` 和 `wall-x` 的相似点非常高，都是把通用视觉语言主干拉到机器人动作任务上。所以它非常适合帮你区分：

- 哪些东西是“VLA 项目都会有的共性”
- 哪些是 `wall-x` 自己的额外复杂度

---

## 四、第二梯队：把动作生成问题单独剥开

### 3. Diffusion Policy

- 仓库：<https://github.com/real-stanford/diffusion_policy>
- 最适合对照的问题：
  - 如果把大语言/视觉主干拿掉，只盯动作 diffusion，本体到底长什么样
  - rollout / denoising / action horizon 的复杂度分别来自哪里
  - 哪些 runtime 问题是动作 diffusion 天生带来的

为什么重要：

`wall-x` 里很多复杂度看起来像“模型很大所以复杂”，但其实未必。  
用 `Diffusion Policy` 做对照之后，你会更清楚：

- 哪些问题来自 `flow / ODE / repeated forward`
- 哪些问题来自 `VLM + multimodal + MoE`

### 4. ACT

- 仓库：<https://github.com/tonyzhaozh/act>
- 最适合对照的问题：
  - `action chunking` 和 `flow / diffusion` 的结构差异到底在哪里
  - 如果把动作路径做得更简单，系统复杂度会下降多少
  - `chunk-level action prediction` 和连续迭代式动作生成，在 runtime 上各自意味着什么

为什么值得看：

`ACT` 是一个非常好的“下限对照”。  
它能帮助回答一个特别重要的问题：

> 很多复杂度到底是不是 VLA / flow 带来的，而不是机器人动作模型天然就必须这么复杂？

---

## 五、第三梯队：看 generalist policy 和工程基座

### 5. Octo

- 仓库：<https://github.com/octo-models/octo>
- 最适合对照的问题：
  - generalist robot policy 怎么组织 observation / task / action
  - 它在“模型统一性”和“系统工程复杂度”之间怎么权衡
  - 它的重点更偏 representation，还是更偏部署

为什么放在后面：

`Octo` 的价值不在于它和 `wall-x` 完全同构，而在于它能帮你把视野从“单个 VLA 项目”抬到“generalist robot policy”。

### 6. LeRobot

- 仓库：<https://github.com/huggingface/lerobot>
- 文档：<https://huggingface.co/docs/lerobot/main/en/index>
- 最适合对照的问题：
  - 机器人 ML 的复杂度有多少其实不在模型里
  - dataset、robot interface、训练/eval pipeline 会怎样反过来塑造模型设计
  - 如果把“模型研究”和“系统工程”分开看，哪个才是当前阶段的主战场

为什么最后看：

`LeRobot` 更像基础设施层。它不是单纯给你一个模型，而是给你一个更完整的机器人 ML 工程环境。看它的目的，不是替代 `wall-x / OpenVLA` 这类模型阅读，而是帮你把“模型之外的复杂度”看得更清楚。

---

## 六、每个项目都该带着哪些问题去看

不管看哪个项目，建议都带着同一组问题去扫：

1. 它的主干到底是什么？  
   是 `VLA`、`policy model`，还是只是“带语言条件的动作模型”？

2. 动作表示是什么？  
   是 `AR`、`chunking`、`flow`，还是 `diffusion`？

3. 它的复杂度主要在哪？  
   在模型结构，还是在 runtime、rollout、robot interface？

4. 它默认依赖的是什么接口？  
   更偏底层张量/向量原语，还是更偏计算图接口？

5. 如果把它搬到 Orin，这个项目最先炸的会是什么？  
   是大算子、还是小算子 fusion、还是系统工程本身？

这组问题的价值在于：  
它们能把阅读重心从“这个项目有多少文件”拉回到“这个项目到底在解决什么问题”。

---

## 七、当前最值得建立的对照视角

如果要把后面的阅读目标压成更短的一句话，那就是：

> 单看一个项目，你只能知道“它是怎么做的”；看过两个以上，你才会开始知道“哪些做法是共性，哪些只是它自己的偶然选择”。

对当前阶段最有价值的几组对照是：

- `wall-x` vs `OpenVLA`
  看 VLA 主干和动作路径的共性

- `wall-x` vs `openpi`
  看 `Flow / AR` 风格的动作生成路线怎么分化

- `wall-x` vs `Diffusion Policy`
  看动作循环本身引入的复杂度

- `wall-x` vs `ACT`
  看更简单动作表示下，模型和 runtime 会轻多少

- `wall-x` / `OpenVLA` / `Octo`
  看“VLA”与“generalist robot policy”之间的边界

---

## 八、当前结论

后面如果继续深挖，当前最值得保留的判断可以压成下面几条：

1. 现在跳去看别的具身项目，是高收益动作。
2. 优先级最高的是 `openpi` 和 `OpenVLA`。
3. `Diffusion Policy` 和 `ACT` 适合用来把动作生成问题单独剥开。
4. `Octo` 和 `LeRobot` 适合用来建立更大的系统视角。
5. 阅读重点不该放在“项目大不大”，而该放在“它到底把复杂度放在模型、runtime，还是系统工程里”。
