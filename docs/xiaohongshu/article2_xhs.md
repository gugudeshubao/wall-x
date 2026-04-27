# Orin 上 GPU 利用率只有 32.7%！FA2 编译 + 深度 Profiling 全记录

📌 本文为精华摘要版，完整版请搜索知乎同名文章《当算子逼近硬件极限：一次 Orin Profiling 引发的具身智能实时系统思考》

---

上一篇在 Orin 上部署 wall-x 3B 时留了一个悬念：Flash Attention 编译不过，只能用 SDPA。profiling 显示 Orin 上 SDPA 的 Softmax 没有融合，独立 kernel 比 5090 慢 57 倍。这一篇，把 FA2 在 Orin 上编译通了，做了完整 benchmark，然后一路 profiling 挖到了真正的瓶颈。

## FA2 在 Orin 上怎么编译通的

flash-attn v2.8.3，需要两个补丁：
1. setup.py 加 SM 8.7 gencode（`TORCH_CUDA_ARCH_LIST="8.7"`）
2. triton rotary kernel 替换成纯 PyTorch 实现（Orin 上 triton 虽然能装但不稳定）

编译 73 个 .cu 文件，耗时约 50 分钟。关键点：必须生成 SM 8.7 的原生 SASS 指令，不能用 SM 8.0 的 PTX 前向兼容——PTX 的 JIT 编译会导致次优的寄存器分配和指令调度。

## VQA 实测：FA2 比 SDPA 快 27%

测试条件：1 张 640×480 真实图片，VQA 文本生成 64 tokens，3 次 warmup + 10 次正式计时，Orin GPU 锁频 1300.5 MHz。

核心数据：
- SDPA：8,681ms（std 109），7.4 tok/s
- FA2：6,360ms（std 25），10.1 tok/s
- FA2 快 27%，内存持平（8.14 vs 8.19 GB），方差只有 SDPA 的 1/4

跨平台对比：与 5090 的差距从 8.0x 缩小到 5.9x。

⚠️ 为什么本篇显存是 8.1 GB 而第一篇是 15.7 GB？因为加载方式不同。第一篇用 .to("cuda").bfloat16()（fp32 先上 GPU，峰值 12.4 GB + 激活 3.3 GB = 15.7 GB），本篇用 .to(device, dtype=torch.bfloat16) 一步到位（bf16 直接上 GPU，6.2 GB + 激活 1.9 GB = 8.1 GB）。模型权重公式：3.1B × 4 bytes = 12.4 GB (fp32) 或 3.1B × 2 bytes = 6.2 GB (bf16)。

## 为什么只快 27%，而不是 A100 上的 2-3x？

深挖 FA2 源码发现：tile size 是在 SM 8.6/8.9（82-128 SM 的 GPU）上调优的，源码注释写明 *"For sm86 or sm89"*。Orin 的 16 SM + 4MB L2 + 204 GB/s 带宽组合完全不在设计范围内。

NCU 实测验证：
- ViT attention：L2 read hit 96%（大矩阵，tile 能复用）
- Prefill attention：L2 read hit 65%（中等矩阵）
- Decode attention：L2 read hit 仅 11.5%（M=1，单次遍历无复用）

Decode 的低 L2 hit 是 batch=1 时 KV cache 单次遍历的固有特征，缩小 tile size 也帮不了。但 Prefill 的 L2 miss 有优化空间——给 SM 8.7 加专属 tile size 分支可能让 FA2 提速从 27% 涨到 50%+。

## GEMM 已经到带宽天花板

对 GEMM kernel 做了逐个 profiling：
- 52.9% 的 GEMM 时间集中在 64×64 小矩阵 decode GEMV（batch=1）
- cuBLAS 实测带宽 168 GB/s，Orin 理论峰值 204 GB/s，利用率 ~72%
- kernel 时间 ≈ 纯内存读取时间

结论：这不是算力瓶颈，是带宽瓶颈。cuBLAS + CUTLASS 已经是最优 bf16 实现了，只有量化（减少数据搬运量）能进一步加速。

## 最大发现：GPU 利用率只有 32.7%

退一步看全局数据：

每个 decode step 的时间分解：
- Wall clock：97.2ms（100%）
- GPU kernel：29.7ms（30.6%）← 真正在计算
- 框架开销：67.5ms（69.4%）← Python + HuggingFace 在"空转"

67.5ms 的开销来自：HuggingFace generate() 循环的 Python dispatch、52,280 次 kernel launch 的调度延迟（20.7μs/launch）、动态 KV cache 内存管理、Python GIL 和 GC。

64 步 decode 总共浪费 4,320ms（67.5ms × 64 步）在框架开销上——这比 GEMM 全部时间（1,643ms）都多！

## 结论

优化 GEMM 最多省 1,643ms；消除框架开销可以省 4,320ms。

三个发现共同指向一个结论：GEMM 已触达带宽天花板、框架空转占 67%、算子优化空间见顶——具身智能的瓶颈不在 model，而在 runtime。C++ 推理改造不只是性能优化，而是把具身智能从"跑模型"推向"实时系统"的第一步。

下一步：用 C++ libtorch 消灭那 67% 的框架空转。

#FlashAttention #JetsonOrin #GPU优化 #Profiling #具身智能 #CUDA #机器人 #深度学习 #性能优化 #nsys #Amdahl定律 #GEMM #带宽瓶颈 #实时系统
