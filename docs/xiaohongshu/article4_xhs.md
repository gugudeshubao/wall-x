# INT8 量化实战：理论 2x 加速遇上 Amdahl 定律的铁壁

📌 本文为精华摘要版，完整版请搜索知乎同名文章《INT8 量化实战：从理论 2× 加速到 Amdahl 定律的铁壁》

---

前三篇做完了：环境部署 → FA2 深挖 → C++ 推理框架，Orin 上推理从 Python 的 13.9 秒降到 C++ 的 554ms。这篇终于动刀量化：bf16 GEMM → INT8 GEMM。

## 量化方案

W8A8：Per-channel 权重 INT8 + Per-token 动态激活 INT8。Python 离线量化导出权重和 scale，C++ 在线推理时做动态激活量化。

wall-x 3B 模型有 ~723 个 Linear 层（36 层 decoder × 7 + 32 层 ViT × 5 + 杂项），GEMM 是计算的绝对主体，量化收益天花板很高。

## 三条路线的探索

❌ 路线一：朴素 INT8（torch._int_mm）
cuBLAS INT8 GEMM 输出 INT32（4 bytes/element），带宽比 bf16（2 bytes）翻倍。dequant 还需要额外 kernel，两个瓶颈叠加。
结果：871ms vs bf16 的 554ms，反而慢了 1.6 倍。

❌ 路线二：cublasLt 手动调用
cublasLtMatmul 可以直接输出 bf16，但 Orin 的 Ampere 架构上 cublasLt 的 INT8 tile 选择不佳，matmul + epilogue 组合没比分开快。
结果：还是比 bf16 慢。

✅ 路线三：CUTLASS EVT（Epilogue Visitor Tree）
核心思路：把 INT32→bf16 dequant 融合进 GEMM 的 epilogue 阶段。INT32 累加器从不写回全局内存，直接在寄存器里做 scale × accumulator → bf16 输出。零额外内存访问。

CUTLASS 的 EVT 让你用 C++ 模板定义 epilogue 计算树：
Compute0（乘 scale_row）→ Compute1（乘 scale_col）→ 写回 bf16

Ampere 架构上 INT8 MMA 指令是 m16n8k32，每条指令执行 8192 次 INT8 乘加运算——是 bf16 m16n8k16（4096 ops）的 2 倍。理论上 INT8 就应该快 2 倍。

## GEMM 微基准测试

M=32, K=N=2048（ODE 步的典型 shape）：
- bf16 cuBLAS：163.4μs
- CUTLASS INT8 融合：70.0μs → 2.33x 加速

全矩阵尺寸覆盖：1.43-1.53x 加速。

## 端到端结果——Amdahl 来了

bf16 C++：577.5ms（ViT 225.9 + Prefill 201.8 + ODE 146.7）
INT8 CUTLASS：557.3ms（ViT 223.1 + Prefill 198.3 + ODE 133.4）

GEMM 快了 2.3x，端到端只快了 3.5%。

为什么？nsys profiling 给出了答案：

量化后 Linear GEMM 只占 GPU 总时间的 6.5%。其余是：
- MoE GEMM：28%（自定义 CUDA op，未量化）
- Elementwise 操作：20%（不可量化）
- cudaLaunchKernel 开销：33,552 次调用 × 21.7μs/次
- Attention、ViT 等：剩余

Amdahl 定律：当你优化的部分只占 6.5% 时，即使优化到 0，总体也只能快 6.5%。我们把它快了 2.3x（省了约 60%），换算成端到端就是 ~3.5%。

## 教训

1. Profiling 先行：不 profiling 就量化，可能把精力花在只占 6.5% 的地方
2. 朴素方案不一定更快：INT32 输出带宽翻倍 + 额外 dequant kernel 可以吃掉所有理论收益
3. Epilogue 融合是关键：CUTLASS EVT 零额外内存访问，是 INT8 GEMM 真正能兑现理论收益的原因

## 下一步

当前 1.79 Hz，目标 2 Hz（500ms），还差 57ms：
- CUDA Graph：减少 33K kernel launch 开销 → 预估 20-40ms
- ViT INT8 量化：ViT 占 40% 但仍是 bf16 → 预估 15-30ms
- MoE INT8：占 GPU 28% → 预估 5-10ms

CUDA Graph + ViT 量化组合拳大概率能过 2 Hz 门槛。

#INT8量化 #CUTLASS #Amdahl定律 #JetsonOrin #GPU优化 #具身智能 #CUDA #机器人 #W8A8 #性能优化 #MMA指令 #Ampere #EVT #GEMM
