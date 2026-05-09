# 怎么把 INT8 收益真正兑现：在 Orin 上给 wall-x 补 CUDA Graph 和量化算子融合

> 前四篇走到这里，wall-x 在 Orin 上的部署路线已经很清楚了：第一篇把问题摆出来，第二篇证明 attention 不是主要瓶颈，第三篇先把 Python/HuggingFace 的框架空转拿掉，第四篇再用 INT8 把真正的大头 GEMM 压下去。做到第四篇结尾，Flow Action 已经从最初的 Python `~912ms` 压到 `480ms`，VQA（20 tokens）压到 `~1.0s`。而我们在写这篇的时候，又连续做了几轮 runtime 级优化：decoder `rmsnorm + residual+rmsnorm`、`gate/up + SiLU * mul`、Flow Action ODE `CUDA Graph`、VQA decode 去同步、再到 Vision `1280` norm fusion 和 VisionAttention 的 equal-block no-mask fast path。结果很有代表性：**Flow Action 最终压到 `290.7ms`，VQA 压到 `851.2ms`。** 这说明量化不是终点，**量化压掉大头之后，新的瓶颈会立刻浮出来：kernel launch、逐元素短链、以及还没被融合掉的残余路径。** 这一篇，不再继续追“还有哪些层没量化”，而是专门讲清楚：为什么量化走到最后，一定会逼你进入 CUDA Graph、算子融合和 runtime 设计。

**TL;DR**
- 这篇写作时的最新 baseline：Flow Action `290.7ms`，VQA（20 tokens）`851.2ms`
- 到这一步，主瓶颈已经不再是“哪些大 Linear 还没量化”，而是 **kernel launch + elementwise 路径 + quant/dequant 前后处理**
- 现在在 Orin 上，**我们手里已经明确有两条自定义算子路线**：
  `CUTLASS / 手写 CUDA` 是稳定主线，`Triton` 是已验证可行但工程成熟度更低的备选线
- **CUDA Graph** 先解决的是 CPU 侧 dispatch gap，尤其适合 Flow Action 里 shape 固定的 ODE/postfix 循环
- **算子融合** 解决的是中间结果反复写回全局内存的问题，重点不在“重写所有 GEMM”，而在“把 GEMM 前后那串短链合并”
- 对 wall-x 来说，更值得优先做的是 `residual + rmsnorm`、`quantize + layout transform`、`gate/up + SiLU/mul` 这类高频短路径
- **CUTLASS 不是拿来替代 cuBLAS 全家桶的**；它真正有价值的地方，是让你在 Tensor Core GEMM 的主体不变的前提下，自定义 epilogue、layout 和数据流
- 量化写到最后，问题已经不再只是“模型能不能转成 INT8”，而是在补一层端侧 runtime 基础设施
- 这一轮我们已经先落地了 `rmsnorm + residual+rmsnorm`、`fused_silu_mul`、ODE `CUDA Graph`、VQA decode 去同步、`Vision 1280` norm fusion，以及 VisionAttention 的 equal-block no-mask fast path，验证了这条路确实还能继续往下榨，但不同任务对同一条优化的收益并不对称

> **本文环境（Orin 实机）**：JetPack `6.2.1`，Python `3.10`，CUDA Toolkit `12.6.11`，cuDNN `9.3.0.75`，cuBLAS `12.6.1.4`，TensorRT `10.3.0.30`，TensorRT-LLM `0.12.0`，Triton `3.6.0`，PyTorch `2.5.0a0+872d972e41.nv24.08`，flash-attn `v2.8.3`（源码编译，SM 8.7 gencode），`CUTLASS` 使用仓库内置子树，当前 checkout 为 `v4.1.0`，本文涉及的 `EVT / GemmWithVisitor` 路径属于 `CUTLASS 4.x` 接口；profiling 工具为 `nsys 2024.7.1` 与 `ncu`。

---

## 一、为什么第五篇不再继续写“量化覆盖率”

第四篇把一个重要事实讲透了：量化不是天然有用，只有当覆盖率真正打到主热路径时，收益才会释放出来。把 `vision_mlp` 的漏网层补齐，把 216 个 `MoE expert projection` 也拖进 INT8 之后，Flow Action 才终于从 `568ms` 压到 `480ms`。而第五篇开工后的 runtime 优化，又把它进一步压到了 `290.7ms`。

但量化做到这一步之后，新的问题也变得很明确了。

第一，**继续补量化覆盖率的边际收益已经在下降**。因为真正的大头已经基本打到了，后面剩下的很多链路，不再是大矩阵本体，而是：

- 每个 kernel 之间的 CPU dispatch
- quant/dequant 之间的格式转换
- residual、norm、SiLU、mul、copy 这类逐元素算子
- KV Cache 读写、shape 固定但重复很多次的短路径

第二，**量化越成功，非 GEMM 部分越显眼**。这是一个很典型的“优化后瓶颈转移”过程：当大矩阵没那么贵了，原来不显眼的 0.02ms、0.05ms、0.1ms kernel 会开始排队冒出来。单个看都不大，但一层一层叠上去，一步一步迭代下来，最后就会重新长成几十毫秒。

所以第五篇的问题，不再是“INT8 还能不能再多打几个点”，而是：

> 当大矩阵已经被量化压下去之后，剩下的时间到底卡在什么地方？又应该用什么办法把这些时间继续榨掉？

在进入具体优化之前，先把一个容易混淆的判断立住：

> **现在不是“只有 CUTLASS 能写自定义算子”，而是“CUTLASS 和 Triton 两条路都能走，但成熟度不一样”。**

具体来说：

- **CUTLASS / 手写 CUDA**：
  已经稳定支撑了 `INT8 GEMM`、`decoder norm fusion`、`fused_silu_mul`、`CUDA Graph` 这条主线，是当前生产级路线
- **Triton**：
  在 Orin 上已经明确证明 kernel 能写、能编、能 benchmark，甚至部分 cubin 能 load 进 C++ runtime；但 `AOT / launcher ABI / metadata` 这层还没有完全收口，所以当前更适合作为快速原型和备选线

所以第五篇后面的很多取舍，核心不是“选哪条路才对”，而是：

> **先用哪条路更快拿到稳定收益，再决定哪条路更适合长期自动化。**

---

## 二、先做 CUDA Graph，不是因为它最性感，而是因为它最顺手

这一阶段最先该做的，不是马上去写一堆融合 kernel，而是 **先把固定 shape 的循环用 CUDA Graph 抓起来**。

这里说的是**工程优先级**，不是严格的时间顺序。真实推进过程中，我们前后其实交错做了 `decoder norm fusion`、`fused_silu_mul`、`ODE CUDA Graph`、`Vision 1280 norm fusion` 和 `VisionAttention` fast path；但如果只从“哪一类动作最值得先做”来排序，Graph 仍然是最顺手、最稳妥的一步。

原因很现实。CUDA Graph 不改变数学，不改权重格式，不改 kernel 本体，它做的事只有一件：

> 把原来“CPU 一次次发射 kernel”的过程，录成一段可 replay 的执行图，之后直接整段回放。

这对 Orin 这种 CPU 比较弱、而 GPU 端 kernel 已经不算慢的平台特别有意义。因为当你把 Python 层拿掉、又把 GEMM 压到 INT8 以后，CPU dispatch gap 会变成一个更扎眼的残余项。

Flow Action 尤其适合先做这一步。原因也很直接：

- batch 基本固定
- postfix 长度固定
- ODE 步数固定
- 每一步执行的 kernel 序列高度重复

换句话说，**ODE/postfix 这条链天然就是 CUDA Graph 的好材料**。你不需要先解决一堆数学或精度问题，只要把 buffer 预分配好，把 capture 边界理清楚，把每轮 replay 需要更新的输入槽位留出来，就能先把 CPU 侧那部分多余调度拿掉。

所以从工程优先级看，CUDA Graph 更像是一个“低风险先收一波收益”的动作：

- 不改模型行为
- 不碰量化格式
- 不碰精度闭环
- 但能先吃掉一部分 launch overhead

而从现在的结果看，这个判断是成立的：**Graph 这一步已经先把 Flow Action 从 `416.8ms` 压到了 `389.9ms`。**

这也是为什么第五篇标题里先写 `CUDA Graph`，而不是一上来就写 `CUTLASS`。

### 2.1 CUDA Graph 落地后的实际收益

这里我们已经不是停留在“理论上适合 graph capture”，而是真的把 Flow Action 里剩余 4 步 ODE 循环抓进了 `CUDA Graph`。

做法很明确：

- Prefill 仍然走普通 eager 路径
- 截断 KV Cache 到 prefix 后
- 把剩余 4 步 postfix forward + velocity 预测 + Euler update 录成一段可 replay 的 graph
- 后续同 shape 调用直接 replay，不再逐个 launch 这一整串 kernel

放到当前 baseline 上看，收益非常干净：

| 配置 | Flow Action 总时间 | Prefill | ODE | VQA 总时间 |
|------|-------------------|---------|-----|------------|
| `+ fused_silu_mul` | 416.8 ms | 99.3 ms | 104.5 ms | 958.5 ms |
| **+ ODE CUDA Graph** | **389.9 ms** | **99.6 ms** | **74.1 ms** | **956.7 ms** |

也就是说，Graph 这一步几乎只打中了它该打中的地方：

- **Flow Action ODE**：`104.5ms → 74.1ms`
- **Flow Action 总时延**：`416.8ms → 389.9ms`
- **VQA**：基本不变，因为当前 graph fast path 只接在 Flow Action 的 fixed-shape ODE 循环上

这其实比“所有任务都一起快一点”更说明问题。因为它证明了：

> **CUDA Graph 的收益不是抽象的，它强依赖那条链是否真的 fixed-shape、是否真的高频 replay。**

Flow Action 是天然适合 graph capture 的；VQA decode 不是。这正是第五篇想讲的重点之一。

### 2.2 VQA decode 去同步：收益有，但不像 Flow Action 那么剧烈

Flow Action 这条线吃到了非常纯粹的 `CUDA Graph` 红利；VQA 则不是。它的问题不在 fixed-shape ODE，而在 **每步 decode 都要把 token 从 GPU 拿回 CPU**。

原始写法里，关键路径是：

```cpp
next_token = step_logits.argmax().item<int64_t>();
```

这会在每个 decode step 上触发一次 GPU->CPU 同步。第五篇这一轮做的改动是：

- token 全程保留在 GPU 上
- decode loop 内部只更新 GPU tensor
- 循环结束后再一次性把 `generated_ids` 搬回 CPU，统一扫描 EOS

放到当前 baseline 上，收益是有的，但没有 Flow Action 那么夸张：

| 配置 | VQA 总时间 (20 tok) | Prefill | Decode | tok/s |
|------|----------------------|---------|--------|------|
| `+ ODE CUDA Graph` | 956.7 ms | 93.6 ms | 653.4 ms | 20.91 |
| **+ GPU-only decode loop** | **949.4 ms** | **94.3 ms** | **641.3 ms** | **21.07** |

这组数据也很典型：

- **收益存在**：`956.7ms → 949.4ms`
- **但远没有 Flow Action Graph 那么猛**

原因并不复杂：VQA decode 虽然有同步问题，但它每步真正的主体仍然是整段 decoder forward，不是单个 `.item()`。所以这一步更像是在“把最扎眼的串行同步拿掉一块”，而不是一次性翻盘。

### 2.3 把同一套 norm fusion 扩到 Vision 1280：ViT 终于也开始掉

前面几轮优化里，一个很明显的现象是：Flow Action 和 VQA 都在降，但 **ViT 段一直还是最大的单段时间之一**。这很容易让人误以为“Vision 这块已经没有太多可做的了”。

实际上，我们前面写出来的 `rmsnorm + residual+rmsnorm` kernel 一开始只接在 decoder `hidden_size=2048` 路径上。把它进一步泛化到 `Vision hidden_size=1280` 后，收益立刻就出来了：

| 配置 | Flow Action 总时间 | ViT | VQA 总时间 (20 tok) | ViT |
|------|-------------------|-----|----------------------|-----|
| `+ GPU-only decode loop` | 389.9 ms | 212.8 ms | 949.4 ms | 212.6 ms |
| **+ Vision 1280 norm fusion** | **369.1 ms** | **192.8 ms** | **925.9 ms** | **190.8 ms** |

这一步很有代表性，因为它说明：

- `norm` 这种看起来“不像大头”的短链，一旦在高频模块里重复几十层，累计收益仍然很可观
- 之前只接 decoder 不够，**同一类 fusion 往 Vision 主干扩展**，仍然能继续吃到收益

也正因为 ViT 真的开始掉了，第五篇的结论又更清楚了一层：

> **短链融合不是只服务于 decoder，也不是只服务于量化路径。它本质上是在把整条数据流里的高频小开销一段段收紧。**

### 2.4 VisionAttention 的 equal-block no-mask fast path：ViT 再次断崖式下降

做到这一步之后，ViT 仍然是 Action 和 VQA 里最大的单段之一。进一步读代码会发现一个很扎眼的点：`VisionAttention` 原来每次都会根据 `cu_seqlens` 在 host 侧构一个 block-diagonal `attn_mask`，再把它传给 SDPA。

这有两个问题：

1. host 侧 mask 构造本身有额外开销
2. 一旦传了显式 mask，底层 attention 路径就不一定还能走到最舒服的 fused/no-mask 分支

而在 wall-x 当前的 vision window 场景里，很多 block 实际上是 **等长分块**。只要 `cu_seqlens` 的每段长度一致，就没必要再构显式 block-diagonal mask，完全可以：

- 直接把这些 block reshape 到 batch 维
- 用一次 **无 mask 的 batched SDPA** 跑完

这一步落地后的收益比预想还大：

| 配置 | Flow Action 总时间 | ViT | VQA 总时间 (20 tok) | ViT |
|------|-------------------|-----|----------------------|-----|
| `+ Vision 1280 norm fusion` | 369.1 ms | 192.8 ms | 925.9 ms | 190.8 ms |
| **+ equal-block no-mask VisionAttention** | **290.7 ms** | **115.0 ms** | **851.2 ms** | **115.9 ms** |

这已经不是“再抠几毫秒”的级别，而是直接把 ViT 段砍掉了接近 40%。

这里最值得记住的不是这个具体数字，而是它说明了另一个很重要的经验：

> **很多时候，真正拖慢 attention 的不是算 QKV 本体，而是你为了喂给框架一个“通用表达”，额外塞进去的 mask / layout / host-side 组织逻辑。**

这也是为什么第五篇会越来越偏向 runtime 视角：到后面，优化的关键已经不是“再写一个更快的单算子”，而是**重新组织数据，让底层库能走到它本来就擅长的路径。**

---

## 三、但只做 Graph 还不够，因为 Graph 不会减少全局内存往返

CUDA Graph 能解决的，是 **kernel 之间的 CPU dispatch gap**；它解决不了的，是 **中间结果在全局内存里来回写回/读回**。

这点必须分清。

如果一条链原来长这样：

```text
kernel 1: residual add
-> kernel 2: rmsnorm
-> kernel 3: quantize
-> kernel 4: int8 gemm
-> kernel 5: dequant / bias / act
```

那 CUDA Graph 做的是：

- 不再让 CPU 一次次发这 5 个 kernel
- 而是一次 `cudaGraphLaunch` 整段 replay

但这 5 个 kernel 还是 5 个 kernel。  
中间结果还是会在全局内存里落地，还是会被下一步再读出来。

这就是为什么做完 Graph 之后，下一步一定会自然转到 **算子融合**。因为到了这个阶段，最值得省下来的，往往已经不是“再少发几个 kernel”，而是：

- 少一次全局内存写回
- 少一次格式转换
- 少一段独立的逐元素 pass

也就是说，Graph 是把 **发射开销** 打掉，Fusion 是把 **数据通路上的浪费** 打掉。两者不是替代关系，而是顺序关系。

---

## 四、什么样的融合最值得优先做

这里有一个很容易走偏的点：**不是所有能 fusion 的地方都值得先做。**

如果一个 kernel 本身已经是 cuBLAS/cuDNN/FA2 这类被打磨得很深的主体，那“为了 fusion 而完全重写主体”通常不划算。更现实的策略是：

> 不去碰已经很强的主算子本体，而去吃掉它前后那串短小但高频的辅助链路。

对 wall-x 当前这条链，优先级大概是这样的。

### 4.1 `residual + rmsnorm`

这是最典型的短链。  
如果拆开跑：

1. 先做 residual add
2. 写回 global memory
3. 再读回来做 RMSNorm

这种链在 decode / ODE 里会被疯狂重复。单次收益可能不夸张，但高频出现，累计效果很稳定。

### 4.2 `quantize + layout transform`

量化之后，很多路径会多出一段“先算 scale，再量化，再搬成目标 layout”的前处理。如果这些步骤是散的：

- 一个 kernel 算 `absmax`
- 一个 kernel 做量化
- 一个 kernel 做 pack / transpose / layout transform

那 launch 和内存读写都会很碎。

这类链很适合往一起收。不是因为某一步单独特别慢，而是因为它们天然属于同一个“进入 INT8 GEMM 之前的准备动作”。

### 4.3 `gate_proj / up_proj + SiLU + mul`

这类 MLP/MoE 支路也很适合 fusion。  
如果 `gate_proj` 和 `up_proj` 出来以后，再单独做：

- SiLU
- 元素乘
- 甚至后续再接一层 down_proj

那中间结果会很重。这里的关键不是“把所有投影都自己重写”，而是判断：

- 哪些输出可以不落地
- 哪些激活可以留在寄存器或 shared memory 里继续消费

这条链也是第五篇下一步最直接的目标。当前 `fused_silu_mul` 的 Triton kernel 本身已经有实现，但 AOT 编译和 C++ loader 这条链还没完全打通：`rmsnorm` 和 `fused_add_rmsnorm` 的 cubin 已经能编出来并 load 进 runtime，`fused_silu_mul` 则还卡在 Triton 3.6 的编译接口/launcher 适配上。也就是说，这不是“要不要做”的问题，而是**先用哪条工程路径把它尽快落地**的问题。

这也是为什么，第二轮真正拿到结果的 `fused_silu_mul` 最后先走了 **原生 CUDA kernel**，而不是继续等 Triton AOT 完整跑通。这里的判断非常工程化：

- 要验证这条 fusion 有没有端到端价值，先上最可控的实现
- Triton / compiler / loader 这条链，是下一层“可维护性”和“自动化”的问题，不该反过来阻塞收益验证

### 4.4 不要优先碰已经很成熟的大 GEMM 主体

这点反而最重要。  
第五篇如果要给读者一个硬结论，我会更愿意写成：

> **算子融合的重点不是“用自定义 kernel 替代所有库”，而是识别哪些短链路值得贴着成熟 GEMM 主体做融合。**

因为对 Orin 这种平台来说，重写一个打不过 cuBLAS 的大 GEMM，通常只是把问题从“launch 多”换成“主算子也慢”。

### 4.5 两轮融合的实测结果：先吃 `rmsnorm`，再吃 `SiLU * mul`

第五篇开工后的第一轮落地，不是 CUDA Graph，也不是更激进的 GEMM epilogue，而是先把 decoder 里的两条高频短链收掉：

1. **pre-attention `rmsnorm`**
2. **post-attention `residual add + rmsnorm`**

实现方式也很克制：没有去碰 attention / GEMM 主体，只是给 `hidden_size=2048` 的 decoder 路径补了一版原生 CUDA fused kernel。第一轮结果比我预期更干净：

| 配置 | Flow Action 总时间 | Prefill | ODE | VQA 总时间 (20 tok) | Prefill | Decode |
|------|-------------------|---------|-----|----------------------|---------|--------|
| Stage 3 INT8（第四篇结尾） | 480.0 ms | 120.2 ms | 139.9 ms | 1012.6 ms | 114.5 ms | 682.7 ms |
| **+ decoder norm fusion** | **426.8 ms** | **103.0 ms** | **105.2 ms** | **951.5 ms** | **97.5 ms** | **637.2 ms** |

也就是说，这一轮只动 `norm` 和 `residual` 相关的短链，就拿到了：

- **Flow Action**：`480.0ms → 426.8ms`，再降 `53.2ms`
- **VQA**：`1012.6ms → 951.5ms`，再降 `61.1ms`

这里最重要的不是“某个 kernel 单独快了多少”，而是它真的回到了端到端链路里，并且同时压低了：

- Flow Action 的 **Prefill**
- Flow Action 的 **ODE**
- VQA 的 **Prefill**
- VQA 的 **Decode**

这也验证了第五篇最核心的判断：**当量化把大 GEMM 打下来以后，继续往下抠最值得做的，不是再去替换已经很成熟的主矩阵乘，而是把 decoder 里这些高频短链一点点收紧。**

第二轮我又把 `gate/up + SiLU * mul` 这条链也收掉了，接到了 `MoE` 和 `VisionMLP` 路径里。结果就更有意思了：

| 配置 | Flow Action 总时间 | Prefill | ODE | VQA 总时间 (20 tok) | Prefill | Decode |
|------|-------------------|---------|-----|----------------------|---------|--------|
| `+ decoder norm fusion` | 426.8 ms | 103.0 ms | 105.2 ms | 951.5 ms | 97.5 ms | 637.2 ms |
| **+ fused_silu_mul** | **416.8 ms** | **99.3 ms** | **104.5 ms** | **958.5 ms** | **95.4 ms** | **652.2 ms** |

这组数据说明了一件很重要的事：

- **Flow Action** 继续受益，说明 `gate/up + SiLU * mul` 在 postfix / prefill 这种高频短链里是值钱的
- **VQA** 没有继续同步下降，甚至有轻微波动，说明同一条 fusion 在不同任务结构下不一定等价生效

也就是说，第五篇讲的不是“只要 fusion 就一定更快”，而是：

> **Fusion 的收益强依赖调用频率、链路位置和任务结构。**

---

## 五、为什么会自然走到 CUTLASS

一说到写自定义 GEMM，很多人会本能地想到“是不是得从 PTX / MMA 指令自己起手”。理论上可以，但在大多数工程场景里，这条路既慢也没必要。

更现实的做法是把 CUTLASS 当成一个中间层去理解：

- 主体 GEMM 的 tile、warp、mma pipeline，它已经帮你搭好了
- 你真正要下手改的，通常是：
  - 输入输出 layout
  - iterator
  - epilogue
  - 有没有把 bias / dequant / activation 顺手并进去

这也是 CUTLASS 在这条路线上的真正价值。  
它不是拿来和 cuBLAS 正面拼“通用 GEMM 谁更快”的，而是让你能在 **保留 Tensor Core 主体效率** 的前提下，把自己关心的数据流逻辑接进去。

第四篇里其实已经碰到这个思路了。朴素 INT8 慢，不是因为 INT8 本身错了，而是因为：

- INT32 中间结果落地
- dequant 太晚
- 数据流被切成了多个 kernel

后来换成 CUTLASS EVT，把 dequant 融进 epilogue 以后，问题才真正被解决。

第五篇沿着这个思路再往前走，重点就不是“CUTLASS 能不能跑”，而是：

> **如何基于 CUTLASS，把那些本来散在 GEMM 前后的小链路继续往主算子里收。**

---

## 六、基于 CUTLASS 写融合算子，真正该关心什么

如果把第五篇写成一篇方法论文章，我觉得最值得讲清楚的，不是罗列一堆 template 参数，而是告诉读者：**在端侧 VLA 这类场景里，什么问题才是写融合算子时真正决定成败的。**

至少有四件事特别关键。

### 6.1 先看 shape，再决定值不值得写

不是所有 shape 都值得自定义。

- 大 GEMM，如果 cuBLAS 已经很好，重写通常不划算
- 小而重复的固定 shape，反而更值得做
- decode / postfix 这种 batch 小、token 数固定、调用次数极高的路径，最适合先下手

所以第一个判断永远不是“能不能写”，而是“这个 shape 的调用频率和累计占比值不值得写”。

### 6.2 layout 比位宽更决定真实收益

量化经常给人一种错觉：位宽下来了，带宽压力就自然下来了。  
但真正决定能不能吃到收益的，往往不是“存成了 INT8”，而是：

- A/B/C 的 layout 怎么排
- tile 怎么切
- shared memory 怎么摆
- register fragment 怎么接
- 中间结果要不要落地

也就是说，**位宽只是入场券，layout 才是兑现性能的关键。**

### 6.3 epilogue 往往比 mainloop 更值得先改

很多实际收益都不是来自“把 mainloop 改得比 cuBLAS 更强”，而是来自：

- 把 bias 融进去
- 把 dequant 融进去
- 把 activation 融进去
- 把后处理链条截断在 epilogue 里

这也是为什么 CUTLASS 对这类工作特别顺手。因为你真正要做的，经常不是重新设计 Tensor Core 主体，而是把结果在“离开 GEMM 之前”就处理成下一步能直接消费的形式。

### 6.4 wall-x 这种模型，不能只盯单个 kernel

如果只测单个 kernel，很容易得到一个错误结论：

- 某个 fused kernel 很快
- 但放回端到端链路没明显收益

原因通常是：

- 调用次数没你想的多
- 前后还有更大的瓶颈
- 或者新的 layout 让下游链路变复杂了

所以第五篇一个很重要的视角，应该始终是：

> **不是单个 kernel 快不快，而是它放回整个 Flow Action / VQA 路径之后，端到端到底省了多少。**

---

## 七、再往前走，Fusion 其实就是编译器问题

写第五篇时，我们先踩到了一个很现实的坑：**Triton kernel 写出来了，不等于你就能在 C++ runtime 里稳定把它 launch 起来。** 这次在 Orin 上，问题不在 kernel 本身，而在 AOT 接口和 launcher ABI。它说明了一件事：

> **Fusion 不是只发生在 kernel 代码里，它还发生在 compiler/runtime 的交界处。**

所以如果再往上抬一层，会得到一个很自然的结论：

> **当你开始系统性地把算子链和计算图融合自动化时，本质上就在碰编译器。**

TensorRT、TVM、Triton/MLIR，做法不同，但核心都一样：先识别哪些链路值得 fuse，再把它们 lower 成目标硬件上的高效实现。也就是说，**自动 fusion 从来不是一个单独的技巧，而是一套编译链条的结果。**

所以我们的思路一直不是“为了学编译器而学编译器”，而是：

> **学编译器，是为了写算子；写算子，是为了反过来理解怎样把这些经验固化成更好的 compiler pass。**

所以从现在这条路径往后看，成长顺序其实也很自然：

1. 先在 `cpp_infer` 里手工把热路径打透  
2. 搞清楚真正值钱的 fusion 是哪些  
3. 再把这些经验抽象成 pass / pattern  
4. 最后才有可能长成 TensorRT / TVM 那种自动化系统  

如果没有前两步，后面的自动化很容易变成“能 fuse，但不值钱”。

再往极端一点看，**megakernel** 也是同一条逻辑的延伸。它当然难，但在固定 shape、固定流程、重复执行极多的端侧场景里，依然值得尝试。因为它追求的，本质上还是同一件事：**让数据在真正有用的计算路径里停留得更久，绕路更少。**

## 八、第五篇真正想立住的判断

写到这里，其实可以把第五篇压成一句话：

> **量化把大矩阵压下去之后，优化的主战场就会从“算得够不够快”转成“数据怎么流得更短、更紧、更少绕路”。**

这就是 CUDA Graph、Fusion、CUTLASS 会在这个阶段同时出现的原因。

- CUDA Graph 先把 CPU dispatch gap 收掉
- Fusion 再把中间结果的全局内存往返收掉
- CUTLASS 则提供了一个能让你在 Tensor Core 主体旁边动手的工程抓手

所以第五篇并不是“第四篇做完之后，顺手再补几个优化技巧”。  
它其实是在回答另一个更底层的问题：

> 当量化已经把最显眼的大头压下去之后，端侧 VLA 的最后一公里应该怎么走？

---

## 九、从量化走到 runtime

量化写到这里，边界其实已经很清楚了。真正难的，不是把模型从 bf16 改成 INT8，也不是把某个 benchmark 跑通，而是怎么让低比特计算在端侧真正变成一条高效、稳定、可复用的执行路径。容量问题相对直接，性能问题却远没有那么简单。权重位宽降下来了，不代表总数据搬运就按比例下降；激活、KV Cache、中间 buffer、dequant/requant 以及 kernel launch，都会继续参与整条链路的成本构成。

所以一旦开始认真追求收益，问题就会立刻下沉到更底层的实现细节：数据块在 register 和 shared memory 里怎么排，dequant 能不能和 matmul 融成一个 kernel，MoE 路由前后的 permute/unpermute 值不值得继续做 fused kernel，CUTLASS 提供的算子模板能不能刚好覆盖当前这些 shape，覆盖不了的部分是不是就要自己补。到这一步，量化就已经不是“换个格式跑工具”的问题了，而是在逼你重写一部分 runtime。

而具身 VLA 又把这个问题进一步放大了。普通 LLM 的主要矛盾，很多时候还能集中在 attention、GEMM 和 KV Cache 这些大算子上；但具身 VLA 在 LLM 主干之外，又叠了 vision、MoE 路由、action head、ODE/postfix 循环、state/action token 注入这些更碎的链路。结果就是：**大算子的优化不会失效，但小算子的 fusion 问题会被重新抬起来。**

也正因为如此，这条路线的分界线不再只是“要不要用 TensorRT”，而更像是：**你依赖的是向量/张量原语接口，还是计算图接口。** 前者对应的是 `CUDA / cuBLAS / cuDNN / CUTLASS / 手写 kernel` 这类能力，重点是把大算子的性能控制权先拿回自己手里；后者对应的是 `TensorRT / TVM` 这类系统，重点是自动做图级融合、schedule、lowering、memory planning 和 codegen。两条路不是互斥的，但回答的问题不同：前者先解决“单个主算子怎么跑到上限”，后者再解决“哪些小链路值得整体融合、如何自动把它们编下去”。

从这个角度看，`TVM` 真正补的是中间那层“图到 kernel”的能力：哪些 pattern 值得 fuse、怎样做 shape specialization、怎样把融合后的子图 lower 成目标硬件上的实现。`TensorRT` 也是在做这类事，只是工程形态更偏 engine builder。对我们现在这条线来说，关键不是非得绑定哪一个系统，而是先把值钱的热路径和 fusion 规律搞清楚，再决定哪些部分值得沉淀到图级自动化里。

如果把这条路再往前推一步，下一层自然会逼近**显式的图级量化表达**：不是把 `quant/dequant` 藏在某个 `LinearOp` 里，而是让 `QuantizeLinear / DequantizeLinear` 这类边界直接出现在 IR 里。只有这样，后端才有机会跨算子识别真正值钱的 pattern，比如 `quantize + layout transform + matmul + bias + dequant`，再进一步决定哪些 `dequant` 可以后移、哪些 `requant` 可以省掉、哪些短链值得整体 lower 成一个 kernel。换句话说，今天我们在 `cpp_infer` 里手工打通的，更像是一批局部的 `QDQ island`；而更远的方向，是把这些 island 总结成图级 pattern，再交给 `TensorRT / TVM` 一类系统自动做 fusion。

而且这条路并不是停留在理论上。在 Orin 上，`Triton` 已经证明能跑，`TVM` 也同样是可行路线。这意味着社区技术完全有可能在不绑定 TensorRT 的前提下，先基于底层向量/张量原语把大算子闭环，再逐步补上自己的图级融合能力。

这也是为什么，前面几篇虽然表面上分别在讲 profiling、C++ 重写、量化和算子融合，但它们最后都会收敛到同一个判断：机器人端侧真正缺的，不只是一个更快的模型，而是一层能把模型、kernel、cache、路由和调度组织起来的执行基础设施。继续往前走，这个问题自然会逼近 AI OS。

按原来的计划，这个系列到这里其实已经可以把 `wall-x` 这条 `C++ + 量化 + runtime` 主线先收住了。但后来我发现，`TensorRT-Edge-LLM` 这条 NVIDIA 官方路线，还是很有必要拿来和我们当前这套实现正面对比一下。所以后面会额外补一篇，专门分析基于 TRT Edge LLM 来部署 VLA 的方案，也顺手补一轮对 NVIDIA 官方思路的学习。

至于更大一层的 runtime / AI OS 抽象，我已经在另一个项目里开始做源码落地了，但不在这个系列里展开。这个系列先收到 wall-x 的部署、量化、runtime 优化，以及后面那篇 TRT Edge LLM 对照分析为止；等那边整理完整，再单独把实现思路和代码结构拿出来讲。
