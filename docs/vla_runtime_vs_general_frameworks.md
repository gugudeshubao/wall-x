# 为什么具身 VLA 仍然需要自定义 runtime

## 1. 先说结论

即使已经有了 `llama.cpp`、`TensorRT-LLM`、`TensorRT` 这样的成熟推理框架，**具身 VLA 里“自己做一层推理 runtime / pipeline”的需求仍然存在**。

但这里的“自己做”不应该理解成：

> 所有 kernel 都自己手写。

更准确的说法是：

> **标准算子尽量复用成熟 backend，自定义的是整体 runtime、控制流、状态管理和任务 pipeline。**

也就是：

- **不是重复造 cuBLAS / cuDNN**
- 而是要自己掌控 **感知 -> 融合 -> 动作生成 -> 控制输出** 这一整条实时链路

---

## 2. 通用推理框架本来擅长什么

现成框架并不是没用。相反，它们在自己的目标问题上非常强。

### 2.1 `llama.cpp`

最擅长：

- 纯 LLM / 近似纯 LLM 推理
- 相对规则的 token-by-token decode
- 简单多模态扩展
- 单机部署和工程可移植性

它的前提是：

- 模型结构比较标准
- 推理过程近似固定
- 输入输出语义仍然接近“token 流”

### 2.2 `TensorRT-LLM`

最擅长：

- 标准 autoregressive decode
- 规则的 KV cache
- LLM 典型子图的 kernel 选择与融合
- 面向吞吐和低延迟的固定生成路径

它的前提是：

- 模型主干接近标准 decoder-only LLM
- 推理流程可抽象成“prefill + decode”
- 动态控制流尽量少

### 2.3 `TensorRT`

最擅长：

- 静态图或相对稳定的动态图
- 标准层的图优化
- kernel fusion
- FP16 / INT8 等自动化部署优化

它的前提是：

- 图可导出
- 图结构稳定
- 非标准算子数量有限
- 中间张量不要在运行时频繁被外部逻辑改写

---

## 3. 具身 VLA 为什么会超出这些抽象

VLA 不是“多模态 chatbot 再多一个动作头”这么简单。

它和通用 LLM 推理最根本的不同在于：

> **VLA 是闭环控制系统的一部分，不只是一个 token 生成器。**

下面几类特征，正是通用框架经常吃不干净的地方。

### 3.1 输出不一定是标准 AR decode

很多具身模型不是只靠 `generate()` 吐 token。

除了纯文本 VQA，动作生成常见还有：

- flow matching
- diffusion
- ODE / SDE 积分
- latent action rollout
- 多步 refinement

这意味着推理不再是：

```text
prefill -> decode -> eos
```

而更像：

```text
prefill -> 初始化状态 -> 迭代更新动作/latent -> 中间状态复用 -> 输出控制量
```

### 3.2 运行时控制流更复杂

具身模型推理里经常会出现：

- 动态更新 action embedding
- prefix/postfix 分段重算
- 截断并复用 KV cache
- runtime scatter proprioception / state token
- 多相机输入的动态拼接
- 不同步长或多阶段控制逻辑

这些逻辑不是一个单纯的“图”就能优雅承载的。

### 3.3 输入不再只是 text/image

真实机器人系统里，模型输入经常包括：

- 多视角 RGB 图像
- depth / point cloud
- proprioception
- dof mask
- gripper state
- robot pose
- 历史动作 / 历史状态
- 外部 planner / safety 模块的条件信号

这些量很多都带有明显的**运行时状态性质**，不是固定 prompt 的自然延伸。

### 3.4 模型里会出现非标准结构

具身 VLA 为了把动作建模做好，经常会引入非标准部件：

- 自定义 MoE 路由
- 多模态 RoPE
- 特殊 attention pattern
- 动作专用 embedding / projection
- ODE 相关 action head
- 自定义 CUDA op

一旦这些结构进入主热路径，通用框架的抽象边界就会开始松动。

### 3.5 优化目标也变了

云端 LLM 典型关注：

- tok/s
- 吞吐
- 平均延迟

具身系统典型关注：

- 控制频率 `Hz`
- 尾延迟
- jitter
- CPU/GPU 同步次数
- 整个闭环的确定性

换句话说，机器人更关心：

> **这一步动作能不能在 300-500ms 内稳定出来，而不是 benchmark 上 token 吞吐是不是更漂亮。**

---

## 4. 关键区别：VLA 要的是 runtime，不只是 engine

一个很容易被忽略的问题是：

> 推理引擎和 runtime 不是一回事。

### 4.1 推理引擎做什么

推理引擎主要负责：

- 执行某个计算图
- 选择 kernel
- 做 layer fusion
- 管理部分显存
- 调度一次 forward / decode

### 4.2 VLA runtime 还要做什么

VLA runtime 往往还要负责：

- 多模态输入整理
- 多模型 / 多模块协同
- 状态缓存
- 动作采样循环
- 中间张量重写
- prefix / postfix cache 策略
- 感知-决策-执行节拍控制
- 和机器人驱动 / planner / safety checker 的接口

所以在具身系统里，真正缺的常常不是“更强的 engine”，而是：

> **一个能把多个 engine / backend / model block 串成实时闭环的 runtime。**

---

## 5. `wall-x` 是一个很典型的例子

`wall-x` 当前这条线，已经把这个问题暴露得很明显了。

### 5.1 现成框架能吃掉的部分

`wall-x` 里很多底层计算其实已经复用成熟 backend：

- attention: `cuDNN SDPA`
- GEMM: `cuBLAS / cublasLt`
- INT8 GEMM: `CUTLASS / cublasLt`
- Tensor 基础操作：`libtorch / ATen`

也就是说，当前路线并不是“完全自研底层算子栈”。

### 5.2 必须自己掌控的部分

真正需要自己实现的，是下面这些 runtime 级逻辑：

- `Flow Action` 的 ODE 采样循环
- action embedding 的每步更新
- prefix/postfix KV cache 截断复用
- 多模态 token 拼装
- proprioception / dof mask 注入
- TokenTypeRouter 风格的 MoE 路由
- 整个感知到动作输出的端到端调度

这也是为什么：

- `TRT-LLM` 对 `VQA` 理论上还能做一些事
- 但对 `Flow Action` 就很难直接承接

不是因为它“不够快”，而是因为：

> **任务结构已经超出了它原本面向的推理范式。**

---

## 6. 所以“自定义推理框架”到底指什么

在具身 VLA 里，这个词最好拆成两层理解。

### 6.1 不推荐的理解

错误理解：

> 从零写自己的 attention、GEMM、allocator、kernel library，一切都不用现成 backend。

这条路通常成本过高，而且大多数团队没有必要这样做。

### 6.2 推荐的理解

更现实的理解是：

> **自定义的是 runtime 和 pipeline，复用的是标准 kernel/backend。**

也就是：

- attention 复用 `SDPA` / `FlashInfer` / `TensorRT` 子图都可以
- GEMM 复用 `cuBLASLt` / `CUTLASS`
- 但动作采样循环、cache 策略、状态管理、调度自己掌控

这才是具身部署里最常见、也最合理的形态。

---

## 7. 一个更有用的三层划分

可以把整件事拆成三层来看。

### 7.1 第一层：标准计算层

这里优先复用现成实现：

- GEMM
- attention
- norm
- activation
- quantized linear

目标：

- 别重复造轮子
- 直接吃成熟 backend 的优化

### 7.2 第二层：模型 runtime 层

这里往往要自己做：

- KV cache 策略
- prefix/postfix 分离
- 多模态输入拼装
- 动作采样循环
- 状态注入
- 子模块协同

目标：

- 把“模型结构的真实控制流”表达出来

### 7.3 第三层：机器人系统层

这里更不可能靠通用 LLM 框架解决：

- 相机采集
- 状态同步
- planner / safety / controller 协同
- 控制周期管理
- 异常恢复
- 实时调度

目标：

- 让模型真正成为机器人闭环的一部分

---

## 8. 哪些情况下通用框架已经够了

不是所有具身项目都一定要很早做自定义 runtime。

如果你做的是下面这些任务，通用框架通常已经够用：

- 只跑 VQA / caption / scene description
- 只做离线评测
- 动作不是实时输出，只做高层 planning
- 模型结构接近标准 VLM
- 输入没有复杂状态注入

这种情况下：

- `TensorRT-LLM`
- `vLLM`
- `llama.cpp`
- `TensorRT`

都可能是更划算的起点。

---

## 9. 哪些情况下自定义 runtime 几乎不可避免

一旦同时出现下面几条，自定义 runtime 的必要性就会迅速上升：

- 动作生成不是纯 AR decode
- 推理里有多步迭代采样
- 中间张量要在 runtime 被改写
- 需要细粒度管理 KV cache
- 有自定义 MoE / 自定义 CUDA op
- 目标是实时闭环控制而不是离线生成
- 你关心的是 `Hz`、抖动、确定性，而不是单次 benchmark

这时问题通常不是“要不要自定义”，而是：

> **自定义到哪一层最划算。**

---

## 10. 更务实的工程策略

对大多数具身 VLA 团队，更务实的路线通常是：

1. **先用通用框架把标准路径跑通**
2. **找出真正的热路径和控制流断点**
3. **只把最关键的 runtime 层拿回来自己做**
4. **底层算子尽量继续复用成熟 backend**

所以真正有价值的能力不是：

> “我能不能完全脱离通用框架。”

而是：

> **“我能不能在复用成熟 backend 的同时，把 runtime 主导权拿回来。”**

---

## 11. 一句话总结

在具身 VLA 里，`llama.cpp`、`TensorRT-LLM`、`TensorRT` 这样的框架依然非常有价值，但它们更像是：

- **标准子图的高性能执行器**

而不是：

- **整个机器人智能闭环的最终 runtime**

所以需求并没有消失，只是形式变了：

> **不是所有 kernel 都要自己写，但“自定义 runtime / pipeline”这层需求，仍然长期存在。**

如果再压缩成一句更短的话，就是：

> **通用框架能吃掉标准算子，吃不掉任务结构；而具身系统真正难的，往往正是后者。**

