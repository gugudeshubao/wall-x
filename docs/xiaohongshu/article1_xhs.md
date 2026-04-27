# 把 3B 大模型塞进机器人！RTX 5090 vs Jetson Orin 部署实录

📌 本文为精华摘要版，完整版请搜索知乎同名文章《把 3B 大模型塞进机器人：RTX 5090 与 Jetson Orin 边缘端产品部署踩坑全记录》

---

我们在做一个叫 wall-x 的机器人大模型——3B 参数量的 VLA（Vision-Language-Action），能看图、能对话、能输出机械臂动作。为了让它跑在真实的机器人上，需要同时部署两个平台：

- RTX 5090（Blackwell 架构）：32GB VRAM，桌面端快速迭代
- Jetson AGX Orin（Ampere 架构）：64GB 统一内存，真正装在机器人身上

这篇记录完整的部署过程和性能数据。

## 部署踩坑

5090 的坑主要是版本兼容：
- Flash Attention 没有 SM 12.0 预编译 wheel，必须源码编译
- 编译时 nvcc 并行度开太高会 OOM，需要 MAX_JOBS=2 限制
- RTX 5090 太新，很多库还没适配

Orin 的坑更多：
- 标准 PyTorch wheel 不能用，必须用 NVIDIA 提供的 JetPack 专用版
- nvidia-smi 在 Orin 上几乎全是 N/A——因为是 SoC 统一内存架构，没有独立 VRAM
- Flash Attention 编译不过，SM 8.7 缺 gencode（下一篇解决）
- HuggingFace 的 SDPA 被 attention_mask 参数退化成 math backend，Softmax 没有融合

显存监控的坑也值得说：Orin 上只能用 torch.cuda.max_memory_allocated()，nvidia-smi 看不到有用信息。

## VQA 推理性能实测

测试条件：8 张真实机器人桌面图片（640×480），每张图 3 个 VQA 问题（"描述你看到的"、"桌上有什么"、"机器人应该做什么"），共 24 个测试用例。生成长度 max_new_tokens=128，这是脚本参数不是模型限制（Qwen2.5 上下文窗口 32K+），128 能覆盖大部分机器人场景的 VQA 回答。

核心数据：
- RTX 5090：平均 1,747ms，59.8 tok/s
- Orin：平均 13,898ms，7.5 tok/s
- 端到端差距 8 倍
- 两个平台 Peak GPU 显存均为 15.74 GB

⚠️ 显存 15.74 GB 偏高的原因：测试脚本用了 .to("cuda").bfloat16() 链式调用，fp32 权重（3.1B × 4 bytes = 12.4 GB）先完整上了 GPU，再原地转 bf16。正确做法是 .to("cuda", dtype=torch.bfloat16) 一步到位，峰值只有 ~8.1 GB。第二篇修正了这个问题。

## Nsight Systems 深度分析

在两个平台上用 nsys profile 抓了完整的 VQA 推理过程，28 万+ kernel：

- GPU 纯算时间差距其实是 13.2 倍（不是表面的 8 倍）
- 差值被 CPU/Python 框架开销"拉近"了
- GEMM/GEMV 占 54-69% GPU 时间——量化的直接目标
- Orin 上 SDPA 的 Softmax 是独立 kernel，耗时 746ms，比 5090 慢 57 倍（13ms）——这是因为 attention_mask 导致 SDPA 退化

两个平台跑的 kernel 完全一样，性能差距纯靠硬件规格碾压（5090: 170 SM, 1.2 TB/s 带宽 vs Orin: 16 SM, 204 GB/s 带宽）。

## 结论

Orin 虽然比 5090 慢 8 倍，但它只有巴掌大小，可以直接装在机器人身上。后续优化方向明确：
1. 编译通 Flash Attention → 解决 Softmax 57 倍瓶颈
2. INT8 量化 → 砍掉 54-69% 的 GEMM 时间
3. 消除 Python 框架开销 → 提升 GPU 利用率

这是系列第一篇，后面还有 FA2 编译、C++ 推理、INT8 量化的完整实战。

#具身智能 #Jetson #大模型部署 #RTX5090 #VLA #机器人 #边缘计算 #CUDA #PyTorch #深度学习 #Orin #NsightSystems #VQA #Profiling
