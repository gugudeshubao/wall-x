# 07. VLA Runtime 的最小执行图：Input Adapter、Planner、Kernel Backend、Output Adapter

如果继续把 `VLA Runtime` 往下落，迟早要把一件事说清楚：

> 它到底不是一堆脚本，而应该是一张怎样的执行图？

很多项目一开始都只有：

- 一个推理脚本
- 一个模型加载函数
- 一个 `forward`
- 一点前处理和后处理代码

这当然可以跑起来。  
但只要你开始认真做：

- benchmark
- fused path
- 低精度
- 端侧适配
- 局部 `megakernel`

很快就会发现，没有清楚的执行图，很多工作都会互相打架。

所以这篇文章不讲具体某个 kernel，而是想把 `VLA Runtime` 的最小执行图先画出来。

我会把它压缩成四个核心模块：

1. `Input Adapter`
2. `Planner`
3. `Kernel Backend`
4. `Output Adapter`

如果这四层关系能理顺，后面很多优化工作才真正有位置可放。

## 一、先说结论：执行图比“函数列表”更重要

很多工程在早期都容易把系统理解成：

> 有哪些函数、哪些脚本、哪些模块

但 runtime 真正关心的不是“有哪些代码文件”，而是：

> 一次请求进入系统之后，信息怎样流动，决策怎样发生，执行怎样落到硬件上。

这就是为什么我更愿意用“执行图”这个词。

执行图要回答的是：

- 输入在哪里变成 runtime 可消费的结构
- 哪一层决定当前请求走哪条执行路径
- 哪一层真正承接 fused op 或低精度 kernel
- 输出在哪一层重新变成控制系统可消费的结果

如果这些问题不先回答，后面很容易出现这样的混乱：

- 前处理逻辑混进 kernel 调度
- benchmark 逻辑混进模型代码
- 低精度开关散落在各个 Python 调用点
- 同一套逻辑既像 planner 又像后处理器

所以，先画清执行图，本身就是 `VLA Runtime` 的基础工作。

## 二、为什么 VLA 的执行图比普通 LLM 更复杂

如果是普通 `LLM`，你可以把执行图相对简单地理解成：

- tokenize
- prefill
- decode
- detokenize

即使内部很复杂，整体主线仍然比较清楚。

但 `VLA` 的问题在于，它经常同时接：

- 图像
- 文本
- 状态
- 可能还有动作历史

同时还可能带着：

- token 类型
- expert 分支
- 动作 horizon
- 端侧/云侧不同执行模式

所以 `VLA Runtime` 如果没有显式执行图，最后很容易退化成：

> 一大坨“能跑”的 glue code。

这也是为什么我觉得，即使还没做成完整 runtime，先把最小执行图定义清楚也很值。

## 三、第一层：Input Adapter 负责什么

我会把 `Input Adapter` 理解成：

> 把外部世界的原始输入，变成 runtime 可规划、可度量、可执行的内部描述。

注意，这层不是简单的“做预处理”。

它至少要处理三件事。

### 1. 输入归一化

把不同来源的输入整理成统一结构，例如：

- 图像
- 文本
- 状态
- 动作历史
- 可选的控制上下文

它的目标不是做模型逻辑，而是让后面各层都不必再关心输入格式的混乱。

### 2. shape 信息显式化

这点很重要。

`VLA Runtime` 后面要不要：

- 走固定 shape 路径
- 做 graph capture
- 走局部 `megakernel`
- 使用某个特定低精度后端

很多时候都依赖输入 shape 是否稳定。

所以 `Input Adapter` 应该尽量早地把这些信息显式化，例如：

- 图像尺寸
- prompt 长度
- 状态维度
- action horizon
- batch 是否恒为 1

### 3. 输入成本可度量

这层还必须天然支持 benchmark。

否则你后面根本无法回答：

- 图像预处理占了多少时间
- 文本模板组织占了多少时间
- 哪些成本不属于核心模型前向

所以 `Input Adapter` 不是附属工具，而是执行图里第一个正式模块。

## 四、第二层：Planner 为什么是 runtime 的大脑

我觉得很多系统里最容易缺失的，就是这一层。

如果没有 `Planner`，系统通常会退化成：

- 遇到一个请求
- 直接调一串固定函数
- 所有决策写死在调用代码里

这在 demo 阶段可以接受，但一旦要做 runtime，就远远不够了。

`Planner` 最核心的职责，是把“这个请求应该怎么跑”显式决定出来。

它至少要决定下面这些事：

### 1. 走哪条执行路径

比如：

- 通用高精度路径
- fused path
- 固定 shape 路径
- 低精度路径
- 端侧特化路径

### 2. 当前请求的执行特征

例如：

- 是否固定 shape
- 是否满足 graph capture 条件
- 当前 expert 路由信息是否可缓存
- 当前部署目标是 edge 还是 cloud

### 3. 给 Kernel Backend 提供结构化 metadata

这一步特别关键。

`Kernel Backend` 不应该自己从零推断所有东西，而应该拿到 planner 已经整理好的信息，例如：

- layout 选择
- weight pack 选择
- precision mode
- routing metadata
- output layout 目标

一句话说，`Planner` 的存在意义是：

> 把“系统层面的选择”从“算子层面的执行”里剥离出来。

## 五、第三层：Kernel Backend 不是“一个大库”，而是一组可替换执行单元

很多人一提 backend，容易直接想到一个巨大的一体化模块。

但对 `VLA Runtime` 来说，我更推荐把它理解成：

> 一组带统一接口、但可按路径替换的执行单元集合。

这一层才是真正承接：

- 普通高精度路径
- fused op
- 低精度变体
- 后续局部 `megakernel`

的地方。

它至少应该有三个特征。

### 1. 接口统一，但实现可替换

比如同一段 `MLP MoE` 路径，可以有：

- 普通 PyTorch 版本
- `BF16` fused 版本
- 低精度版本
- 后续局部 `megakernel`

runtime 上层不应该被这些差异直接撕裂。

### 2. 承接 planner 的决策，而不是替 planner 做决策

`Kernel Backend` 的任务是执行，而不是重新理解整个请求。

这意味着：

- 它应该拿到明确的输入布局
- 拿到明确的 precision mode
- 拿到明确的 output layout 目标

而不是在内部猜测“我该怎么跑”。

### 3. 天然支持 benchmark 和 profiling

如果 backend 是完全黑盒的，后面很多优化就很难做。

所以它至少应该在设计上允许：

- 每个 stage 单独计时
- 记录 kernel 路径
- 区分不同 precision/backend 变体

## 六、第四层：Output Adapter 为什么不能只是“后处理”

很多系统一到输出阶段，就容易简单写成：

> 模型出结果了，做点解析就结束。

但在 `VLA` 场景里，这层远比“普通后处理”重要。

因为这里面对的不是纯文本，而是要真正进入控制链路的结果。

`Output Adapter` 至少负责：

### 1. 输出结构整理

例如：

- 动作块整理
- 动作维度裁剪或映射
- 可选的语言输出解码

### 2. 控制链路衔接

它要解决的问题是：

- 结果什么时候算“可消费”
- 哪个时间点才算 camera-to-action 的真正终点
- 输出格式怎样和机器人控制接口对齐

### 3. 输出阶段的可度量

如果这层不纳入 benchmark，你最后得到的 latency 往往只是“模型前向时间”，而不是系统真正关心的“动作可用时间”。

所以 `Output Adapter` 不是附属层，它决定 benchmark 是否真正覆盖系统目标。

## 七、这四层之间的数据流应该长什么样

如果把这四层压缩成一张最小执行图，它大致应该是：

```text
External Input
  -> Input Adapter
  -> Planner
  -> Kernel Backend
  -> Output Adapter
  -> Control / Serving Consumer
```

但真正有用的不是这行箭头，而是每层之间交换的信息类型应该明确。

### Input Adapter -> Planner

这里不该只传原始张量，而应该传：

- 归一化后的输入结构
- shape metadata
- 可缓存信息
- 输入阶段 benchmark 数据

### Planner -> Kernel Backend

这里应该传：

- 选定的执行路径
- precision mode
- layout metadata
- routing / expert metadata
- output layout 目标

### Kernel Backend -> Output Adapter

这里应该传：

- 输出张量
- 输出语义类型
- 当前阶段 benchmark 数据
- 可选的中间 metadata

如果这些边界是清楚的，runtime 才真正具备可维护性。

## 八、为什么这张执行图正好能承接后面的优化

我觉得这张最小执行图最大的价值，在于它不是静态架构图，而是天然适合继续往下长。

### 1. benchmark 有位置放

可以放在 `Input Adapter`、`Kernel Backend` 和 `Output Adapter` 各阶段，不需要到处打补丁。

### 2. fused path 有位置放

放在 `Kernel Backend`，由 `Planner` 决定是否启用。

### 3. 低精度有位置放

作为 `Kernel Backend` 的后端变体，而不是一堆散落开关。

### 4. 局部 megakernel 也有位置放

不是替代整个 runtime，而是替代某个具体执行 stage。

这点很重要，因为它让 `megakernel` 成为一种可控增强，而不是一开始就把系统做死。

## 九、如果现在只做 MVP，这四层应该做到什么程度

如果只做最小可行版本，我会建议：

### Input Adapter

- 能稳定整理图像、文本、状态输入
- 能显式提供 shape metadata
- 能独立计时

### Planner

- 至少能在“普通路径”和“fused 路径”之间做选择
- 至少能区分 cloud / edge 两类配置
- 能把 precision/layout/routing metadata 传下去

### Kernel Backend

- 至少有一条高精度 baseline path
- 至少有一条 fused path
- 接口上为低精度预留空间

### Output Adapter

- 能把模型输出整理成控制链路可消费结构
- 能纳入 camera-to-action benchmark

只要这四件事成立，这个 runtime 就已经不是“脚本集合”，而是一个真正的系统雏形。

## 十、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的最小执行图，不该是一堆杂糅的推理脚本，而应该至少拆成 `Input Adapter`、`Planner`、`Kernel Backend` 和 `Output Adapter` 四层，让 benchmark、fused path、低精度和局部 `megakernel` 都有明确的位置可放。

这张执行图看起来不复杂，但它的意义很大。

因为一旦这四层关系清楚了，后面的很多事情才真正开始变得可设计、可比较、可迭代。

而这，也正是从“能跑的推理代码”走向“真正的 `VLA Runtime`”时，最应该先迈出的那一步。
