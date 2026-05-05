# TRT vs `cpp_infer` 对照结论

这份文档只回答一个问题：

> **到目前为止，`TensorRT / TRT-LLM` 这条路线，相比当前 `cpp_infer` 主线，到底在哪些地方已经赢了，哪些地方还没有。**

---

## 1. 结论先说

### 1.1 VQA

当前证据下，**TRT 路线还没有在端到端上明显赢过 `cpp_infer`**。

原因不是数学不成立，而是：

- 当前 TRT VQA 只完整跑通了 `5 token`
- 需要多个超大 decode engine
- runtime 组织成本很重

按现有 `5 token` 结果外推到 `20 token`：

- TRT VQA 预估：`~900 ms`
- `cpp_infer` 当前最佳：`851.2 ms`

所以当前更稳的判断是：

> **VQA 上，TRT 还没形成明确优势。**

### 1.2 Flow Action

当前证据下，**TRT 路线已经在端到端上明显赢过 `cpp_infer`**。

在当前 dummy Flow 条件下：

- TRT 两阶段 Flow：`175.9 ms`
- `cpp_infer` 当前最佳：`290.7 ms`

速度比大约：

- `290.7 / 175.9 ≈ 1.65x`

所以当前更稳的判断是：

> **Flow Action 是 TRT 路线真正已经证明有价值的主战场。**

### 当前 Orin 环境前提

本轮 `trt_spike` 对应的 Orin 实机环境已经确认：

- **JetPack**：`6.2.1+b38`
- **L4T / Jetson Linux**：`R36.4.4`
- **nvidia-l4t-core**：`36.4.4-20250616085344`
- **TensorRT-LLM**：`0.12.0`
- **当前实际安装的 Python 包**：`tensorrt-llm`
  - `pip show` 的 `Home-page` 指向：`https://github.com/NVIDIA/TensorRT-LLM`
  - `tensorrt_edgellm` **未安装**

这里还要特别区分两条线：

- **当前实机正在使用的**：`TensorRT-LLM`
- **另一个单独仓库**：`TensorRT-Edge-LLM`

根据 `TensorRT-Edge-LLM` 官方文档：

- 它是面向 embedded 平台的 LLM / VLM runtime
- `Jetson Orin + JetPack 6.2.x` 当前属于 **compatible / experimental**
- 官方支持的平台重点仍然更偏新的 Thor / 后续 JetPack 版本
- 官方工作流是三段式：
  - **Python export pipeline**（通常在 x86 host 上）
  - **Engine builder**
  - **C++ runtime**（在 edge 设备上）
- 官方文档明确写的是：
  - **“Set up the Python export pipeline on your x86 host and build the C++ runtime on your edge device”**

所以当前更准确的环境判断是：

> **这台 Orin 的 JP 版本已经满足 NVIDIA 口中的“JP 6.2+”门槛，但当前本机实际安装并用于实验的是 `TensorRT-LLM 0.12.0`，不是 `TensorRT-Edge-LLM`。**

这意味着：

> **如果 NVIDIA 那边给出的支持门槛是“JetPack 6.2+”，那这台 Orin 在版本条件上已经满足。**

也就是说，当前后续问题已经不是：

- “是不是 JetPack 太老不支持”

而是更具体的：

- `TRT-LLM` 当前版本和 plugin 能力，到底能覆盖 `wall-x` 的哪一部分

补充一条已经实测过的事实：

> **`TensorRT-Edge-LLM` 源码已经在这台 Orin 上按 `EMBEDDED_TARGET=jetson-orin` 成功完成 `cmake + ninja` 编译。**

当前已确认：

- `cmake` 配置通过
- `ninja` 全量编译通过
- 生成的可执行文件包括：
  - `examples/llm/llm_inference`
  - `examples/llm/llm_bench`
  - `examples/llm/llm_stream`
  - `examples/omni/qwen3_tts_inference`
- `./examples/llm/llm_inference --help` 可正常运行

也就是说，至少从：

- 工具链
- TensorRT / CUDA 依赖
- `jetson-orin` 目标构建链

这几个维度看，**当前这台 Orin 已经具备源码编译并运行 `TensorRT-Edge-LLM` runtime binary 的基础条件。**

进一步说，当前这条线还有三个很关键的事实：

1. **`jetson-orin` 构建目标在源码里是真实存在的**
   - 虽然公开安装文档主要写的是 `jetson-thor`
   - 但源码里的 `CMakeLists.txt` 和 `cmake/aarch64_linux_toolchain.cmake` 都已经明确支持：
     - `-DEMBEDDED_TARGET=jetson-orin`

2. **`Qwen2.5-VL` 在 `TensorRT-Edge-LLM` 的支持矩阵里是官方支持模型**
   - 支持矩阵里明确列了：
     - `Qwen/Qwen2.5-VL-3B-Instruct`
     - `Qwen/Qwen2.5-VL-7B-Instruct`
   - 也就是说，对 `wall-x` 的 **VQA 主干** 来说，`TensorRT-Edge-LLM` 理论上比当前手工拼的 `TensorRT-LLM` 路线更接近官方现成支持

3. **源码里已经有 `action_build` 工具，但动作模型侧不是直接对 `wall-x Flow` 开箱**
   - `examples/multimodal/action_build`
   - `tensorrt_edgellm/action_models/`
   - 当前代码里更明显地出现的是：
     - `alpamayo_r1`
     - 以及一套独立的 action expert 导出 / build 流程

所以对 `wall-x` 的现实意义可以先压成：

> **`TensorRT-Edge-LLM` 对 `Qwen2.5-VL` 这条 VQA/VLM 主干更像“官方支持路径”；但对 `wall-x Flow Action` 这类自定义动作控制链，还不能直接等同于“开箱即用”。**

再补两条已经实际验证过的事实：

1. **独立 Python export 环境已经安装完成**
   - 独立目录：
     - `/data/wy/wall-x/workspace/edge_llm_exp/venv_edge`
   - 已成功安装：
     - `tensorrt-edgellm 0.7.0`
     - `torch 2.10.0`
     - `transformers 5.3.0`
   - `tensorrt-edgellm-export-llm --help`
   - `tensorrt-edgellm-export-visual --help`
     都已经可以正常运行

2. **`Qwen2.5-VL-3B-Instruct` 的 LLM export 已经开始**
   - 直连 `huggingface.co` 在这台 Orin 上会超时
   - 改用：
     - `HF_ENDPOINT=https://hf-mirror.com`
   - 并把 cache 明确重定向到：
     - `/data/wy/hf_cache`
   - 当前后台任务已经在下载 `Qwen2.5-VL-3B-Instruct` 权重并准备导出到：
     - `/data/wy/wall-x/workspace/edge_llm_exp/work_onnx/qwen2.5-vl-3b`

这意味着，`TensorRT-Edge-LLM` 这条线的当前状态已经不是：

- 只看过文档
- 只编过 C++ runtime

而是已经进一步走到了：

> **“runtime 编过 + export 环境装好 + Qwen2.5-VL-3B 实际开始导出”**

现在这一步已经进一步推进为：

> **`Qwen2.5-VL-3B-Instruct` 的 LLM ONNX export 已经在 Orin 上实际完成。**

已确认的细节：

- export 工具：
  - `tensorrt-edgellm-export-llm`
- 导出环境：
  - 独立 venv：`/data/wy/wall-x/workspace/edge_llm_exp/venv_edge`
  - `torch 2.10.0`
  - `transformers 5.3.0`
  - `tensorrt-edgellm 0.7.0`
- 解决过的问题：
  - `huggingface.co` 直连超时
  - 改用：
    - `HF_ENDPOINT=https://hf-mirror.com`
  - Hugging Face cache 默认会落到根分区
  - 改成：
    - `HF_HOME=/data/wy/hf_cache`
    - `HF_HUB_CACHE=/data/wy/hf_cache/hub`

实际导出结果：

- ONNX 输出目录：
  - `/data/wy/wall-x/workspace/edge_llm_exp/work_onnx/qwen2.5-vl-3b`
- 关键产物包括：
  - `model.onnx`
  - `onnx_model.data`
  - `config.json`
  - `embedding.safetensors`
  - `tokenizer.json`
  - `processor_config.json`
  - `processed_chat_template.json`

日志里的关键完成信息：

- `ONNX export completed`
- `ONNX post-processing completed`
- `Config saved`
- `Tokenizer saved`
- `Processor saved`
- `LLM model export completed successfully`

这意味着：

> **`TensorRT-Edge-LLM` 对 `Qwen2.5-VL-3B` 这条 VLM/VQA 主干，不再只是“支持矩阵里写了支持”，而是已经在这台 Orin 上实际走通了 LLM ONNX 导出。**

同一条路线上，视觉分支也已经进一步走通：

- `tensorrt-edgellm-export-visual`
  - 已成功导出：
    - `/data/wy/wall-x/workspace/edge_llm_exp/work_onnx/qwen2.5-vl-3b/visual_enc_onnx/model.onnx`
- `visual_build`
  - 初次失败的原因是：
    - `ViTAttentionPlugin` 没有被 loader 找到
  - 带上：
    - `EDGELLM_PLUGIN_PATH=/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so`
    之后，插件已经成功注册，visual engine 已实际生成：
    - `/data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen2.5-vl-3b/visual/visual.engine`

所以到这里，`TensorRT-Edge-LLM` 对 `Qwen2.5-VL-3B` 这条线已经至少完成了：

1. **LLM export**
2. **Visual export**
3. **Visual engine build**

当前只剩下：

4. **LLM engine build**

也就是说，这条官方 VLM 路线已经不再只是“文档上说支持”，而是已经非常接近一条完整可运行的 Qwen2.5-VL-3B engine 目录。

现在这一步也已经完成：

- `llm_build` 最终成功生成：
  - `/data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen2.5-vl-3b/llm.engine`
- 该目录下同时还有：
  - `visual/visual.engine`
  - `visual/config.json`
  - `visual/preprocessor_config.json`

所以现在可以把这条线明确收束为：

> **LLM export → visual export → visual engine → LLM engine**

这意味着 `TensorRT-Edge-LLM` 对 `Qwen2.5-VL-3B` 的官方 VLM 路线，在这台 Orin 上已经完整闭环。

再往前一步，我们还拿到了一个最小 VQA 的真实 wall-clock benchmark：

- 输入：
  - `/data/wy/wall-x/test_images/fruits_on_table.png`
  - prompt: `Please describe the image.`
- 推理入口：
  - `./build/examples/llm/llm_inference`
- engine：
  - `llm.engine`
  - `visual/visual.engine`

3 次正式运行结果：

- `run_1_ms = 7704.467`
- `run_2_ms = 7266.907`
- `run_3_ms = 7626.541`

统计值：

- `mean_ms = 7532.639`
- `std_ms = 190.575`
- `min_ms = 7266.907`
- `max_ms = 7704.467`

这说明：

> **`TensorRT-Edge-LLM` 这条官方 VLM 路线已经不只是能导出、能 build、能推理一次，而是已经能在 Orin 上稳定跑出真实的 wall-clock 性能数字。**

为了避免只看一个模型，我们又补了第二个官方 VLM：

- `Qwen/Qwen3-VL-2B-Instruct`

这条线也已经完整跑通：

- `export-llm`
- `export-visual`
- `llm.engine`
- `visual.engine`
- 最小 VQA inference

最小 wall-clock benchmark（3 次正式运行）：

- `run_1_ms = 5954.077`
- `run_2_ms = 5650.564`
- `run_3_ms = 5333.013`
- `mean_ms = 5645.885`

这组结果说明：

> **`TensorRT-Edge-LLM` 的官方 VLM 路线不是只对单一 `Qwen2.5-VL-3B` 样例成立，而是至少在第二个 `Qwen3-VL-2B` 模型上也已经完整跑通，并且端到端时延随着模型变小而明显下降。**

并且这条线已经进一步跑通了一个最小 VQA 推理样例：

- 推理入口：
  - `./build/examples/llm/llm_inference`
- engine：
  - `edge_engines/qwen2.5-vl-3b/llm.engine`
  - `edge_engines/qwen2.5-vl-3b/visual/visual.engine`
- 输入：
  - 图像：`/data/wy/wall-x/test_images/fruits_on_table.png`
  - prompt：`Please describe the image.`

最终输出文件：

- `/data/wy/wall-x/workspace/edge_llm_exp/output_vlm_qwen25vl3b.json`

实际 `output_text` 已生成，例如：

> `The image depicts a simple, cartoon-style illustration of four fruits...`

为了更接近 `wall-x` 的真实接入方式，我们又换成了同图同 prompt 的对照：

- 图像：`fruits_on_table.png`
- prompt：`Describe what you see in this image.`

`wall-x` baseline 参考答案：

> `The image features a minimalist composition with a brown background. In the center, there are three distinct shapes`

`TensorRT-Edge-LLM` 的实际输出则是：

- `Qwen2.5-VL-3B-Instruct`
  - `run_1_ms = 7703.721`
  - 输出：`The image depicts four colorful objects placed on a flat surface...`
  - 文本相似度：`0.3083`
- `Qwen3-VL-2B-Instruct`
  - `run_1_ms = 5457.292`
  - 输出：`This image is a simple, stylized illustration of a face...`
  - 文本相似度：`0.3169`

这说明：

> **`TensorRT-Edge-LLM` 能把 `wall-x` 风格的 VQA 输入跑起来，但在同图同 prompt 下，它并没有自然复现 wall-x 的 baseline 语义输出。**

这意味着，现在可以把这条线的状态从：

- **“能编过”**

升级成：

- **“已经在 Orin 上用 `TensorRT-Edge-LLM` + `Qwen2.5-VL-3B` 真跑出了一次 VQA 推理”**

---

## 2. VQA 对照

### 2.1 `cpp_infer` 当前最佳

来自：

- `docs/zhihu_orin_fusion_article5.md`

当前最佳结果：

- VQA（20 tokens）：`851.2 ms`

### 2.2 TRT 当前已跑通结果

来自：

- `workspace/trt_spike/docs/results.md`

当前完整已跑通的是：

- VQA-specialized
- `5 token`
- `prefill = 103.818 ms`
- `decode mean = 41.897 ms`
- `total = 271.408 ms`

并且 token 前缀对齐。

### 2.3 `20 token` 结果

最早这里只能根据 `5 token` 结果去外推：

- `total(N tokens) ≈ prefill + (N - 1) * decode_mean`
- `103.818 + 19 * 41.897 ≈ 899.9 ms`

但现在已经不只是估算，而是直接跑出了新的 `2 engine` 动态 decode 版本：

- `prefill.engine`
- `decode_dynamic.engine`

完整 `20 token` 实测结果：

- `prefill = 99.937 ms`
- `decode mean = 43.451 ms`
- `total = 925.512 ms`
- 生成 token 与 reference **完全一致**

而且当前 `2 engine` 目录已经真正收敛成：

- `prefill.engine`
- `decode_dynamic.engine`

总大小约 `12G`（相比原来 `vqa36` 的 `29G` 已明显下降）。

`run-only` 复用也已经验证通过：

- `prefill = 100.222 ms`
- `decode mean = 44.358 ms`
- `total = 943.018 ms`
- `token_match_prefix = True`

### 2.4 当前判断

这说明：

- TRT VQA 这条线数学上成立
- 但端到端上还没有明显压过 `cpp_infer`

更重要的是，它的 runtime 成本很重：

- `prefill.engine` 约 `5.8G`
- 每个 `decode_past*.engine` 约 `5.8G`
- 一整套 `vqa36` engine 目录约 `29G`

所以当前 VQA 路线的问题不是“子图不对”，而是：

> **engine 数量、engine 体积和 session / I/O 组织代价太高。**

---

## 3. Flow Action 对照

### 3.1 `cpp_infer` 当前最佳

来自：

- `docs/zhihu_orin_fusion_article5.md`

当前最佳结果：

- Flow Action：`290.7 ms`

### 3.2 Python baseline

来自：

- `scripts/bench_flow_action.py`
- `workspace/trt_spike/docs/results.md`

当前 dummy 条件下：

- Average total：`1417.1 ms`
- Prefill：`251.8 ms`
- ODE：`769.9 ms`

### 3.3 TRT 当前最佳

来自：

- `workspace/trt_spike/docs/results.md`

当前最稳的 Flow TRT 结果是：

- 两阶段 Flow TRT runner
- `prefetch` TRT
- `postfix step` TRT
- `ActionProcessor.step()` 留 host
- `action_proj_back` 已并进 engine

结果：

- `prefetch_ms = 110.864`
- `postfix_mean_ms = 16.493`
- `total_ms = 176.837`
- `predict_action cosine = 0.98511314`

### 3.4 当前判断

这说明：

- 相比 Python baseline，TRT 路线已经是大幅下降
- 相比 `cpp_infer` 当前最佳，也已经形成明确优势

直接比：

- `cpp_infer`: `290.7 ms`
- TRT: `176.8 ms`
- 速度比：`~1.65x`

所以当前 Flow 路线已经可以下硬判断：

> **TRT 在 Flow Action 上已经不是“可能有价值”，而是“已经证明有价值”。**

---

## 4. 为什么会出现这种分化

### 4.1 VQA 更吃 runtime 组织

VQA 当前的问题不是单层 block 慢，而是：

- 需要 prefill + 多步 decode
- decode 按 `past_len` 切成多个 engine
- 多个超大 engine 的 load / session / I/O 成本很高

所以它更容易被：

- engine 体积
- engine 数量
- runtime orchestration

卡住。

### 4.2 Flow 更适合当前专门化路线

Flow 在当前 dummy 条件下有两个天然优势：

1. prefix/postfix 划分非常清楚
   - `prefix = 456`
   - `postfix = 32`
2. `moe_token_types` 也是连续块
   - 前 `456` 个全 `0`
   - 后 `32` 个全 `1`

这使得当前 Flow TRT 很容易做出专门化：

- prefetch：`expert0 + expert1` 连续切片
- postfix：`expert1-only`

所以 Flow 路线比 VQA 更早吃到“专用 engine”的收益。

---

## 5. 到这里该怎么判断路线

### 5.1 对 VQA

当前判断是：

- 继续做可以
- 但优先级不高
- 因为还没有端到端明确赢 `cpp_infer`

如果继续做，重点不该是继续抠 layer 数学，而该是：

- 减少 decode engine 数量
- 减少 engine 体积
- 降低 session / I/O 成本

### 5.2 对 Flow

当前判断是：

- 值得继续做
- 因为已经在端到端上明确赢了 `cpp_infer`

但下一步也不该再盯小块数学，而该转向：

- engine 体积
- engine 数量
- runtime 组织
- 是否值得做成更像专用 runtime 的东西

---

## 6. 最终判断

压成一句话：

> **VQA 上，TRT 目前“能跑通，但还没赢”；Flow Action 上，TRT 目前“已经跑通，而且已经赢了”。**

再压缩一点：

- `VQA`: 先别指望 TRT 立刻替代 `cpp_infer`
- `Flow`: TRT 已经证明值得继续投

---

## 7. 为什么 `TRT-LLM` 能，而裸 `TensorRT` 不够

这个问题非常关键，因为它决定了后面该不该继续在 `trt_spike` 上投入。

先把一句话说死：

> **`TRT-LLM` 之所以“能跑标准 LLM”，不是因为裸 `TensorRT` 天然什么都能做，而是因为 NVIDIA 已经在 `TensorRT` 上面额外写好了一层标准 LLM runtime。**

### 7.1 裸 `TensorRT` 负责什么

如果只看 `TensorRT` 本体，它主要负责的是：

- load / execute engine
- engine 内部的 kernel / tactic / workspace
- shape profile
- 推理执行本身

也就是说，`TensorRT` 更像是：

> **底层执行后端**

如果你的任务只是：

- 一个固定输入
- 一个固定输出
- 一次 forward 结束

那很多时候，裸 `TensorRT` 已经够了。

### 7.2 `TRT-LLM` 多做了什么

标准 LLM 真正麻烦的不是“一次 forward”，而是：

- prefill / decode
- KV cache 管理
- sampling
- 多步生成循环
- batch / request 调度
- 常见模型骨架适配

这些东西，裸 `TensorRT` 不会替你自动做。

`TRT-LLM` 真正额外提供的是：

- 标准 decoder-only 模型模板
- prefill + decode loop
- KV cache 管理
- sampling
- 部分 paged/block cache 抽象
- 常见 plugin / fused kernel

所以更准确地说：

> **`TRT-LLM = TensorRT 执行后端 + 一层面向标准 LLM 的专用 runtime`**

这也是为什么：

- 对 `Llama / Qwen / Mistral` 这类标准 decoder-only LLM，`TRT-LLM` 很容易直接接住
- 而对 `wall-x` 这种具身 VLA，就没有那么顺

### 7.3 为什么 `wall-x` 尤其是 Flow Action 还得自己管 runtime

因为 `wall-x` 已经明显超出了标准 LLM 抽象，尤其是 `Flow Action`：

- 不是普通 token-by-token decode
- 有 `ActionProcessor.step()`
- 有 `prefix / postfix` 切分
- 有 `ODE / Euler` 循环
- 有动态 action embedding 注入
- 有 `moe_token_types`
- 有具身控制特有的 mask / routing / control flow

这些都不是 `TRT-LLM` 现成 runtime 会替你处理好的内容。

所以在 `wall-x` 上，更合理的职责拆分是：

- **TensorRT**：负责 engine 执行
- **TRT-LLM / 你自己的 builder 脚本**：负责把特定子图拼成 engine
- **你自己的 runtime**：负责
  - KV 生命周期
  - prefill / decode / postfix orchestration
  - 动作循环
  - ODE / Euler
  - task-specific sampling / control logic

### 7.4 三层职责表

| 层 | 负责什么 | 对 `wall-x` 的意义 |
|---|---|---|
| `TensorRT` | engine 执行、kernel/tactic、workspace | 底层执行后端 |
| `TRT-LLM` | 标准 LLM runtime：decode loop、KV、sampling、常见 plugin | 对标准 LLM 很有用；对 `wall-x` 只部分有用 |
| 你自己的 runtime | prefill / decode / postfix / ODE / action orchestration | 在 `wall-x` 尤其 `Flow Action` 上不可避免 |

### 7.5 当前路线该怎么理解

所以现在这条 `trt_spike` 的意义，不是要证明：

- “只用 `TensorRT` 就能把整条 `wall-x` 推理链自动接管”

而是要证明：

- 哪些块值得交给 `TensorRT`
- 哪些块 `TRT-LLM` 可以直接吃
- 哪些块最后必须由你自己的 runtime 组织起来

压成一句最后的判断：

> **标准 LLM 之所以看起来“TRT-LLM 直接就能跑”，是因为 NVIDIA 已经替它们写好了 runtime；而 `wall-x` 尤其 `Flow Action` 这类具身 VLA，最终仍然需要你自己接管那层 runtime。**

### 7.6 为什么有的 VLA 能直接用 `TRT-LLM`

这件事要分清楚：

- **“能用 `TRT-LLM`”**
- 和
- **“整条 VLA 推理链都天然适合 `TRT-LLM`”**

不是一回事。

很多看起来“能直接上 `TRT-LLM`”的 VLA，往往满足下面这些条件：

1. **视觉前端可以放在外面**
   - vision encoder 单独跑
   - 产出 image embeddings
   - 再把这些 embeddings 当成 prompt / prefix 喂给一个标准 decoder

2. **语言主干仍然是标准 decoder-only LLM**
   - Llama / Qwen / Mistral 这类结构
   - 没有额外的动作循环
   - 没有特殊的 runtime 控制流

3. **动作头很薄**
   - 可能只是从最后几个 token 上接一个小 head
   - 而不是像 `Flow Action` 那样反复跑 ODE / Euler

4. **没有显式的 `MoE token routing + prefix/postfix` 这种自定义流程**

这类 VLA 本质上更像：

> **“一个标准 LLM + 一个外接 vision encoder + 一个薄动作头”**

所以它们经常能把最值钱的主干直接交给 `TRT-LLM`：

- vision 在 host / PyTorch
- decoder 在 `TRT-LLM`
- action head 继续留在外面

这样对外看就像“这个 VLA 能直接用 `TRT-LLM`”。

但 `wall-x` 尤其 `Flow Action` 不一样。它的难点不是“有没有视觉前端”，而是：

- `Qwen2.5-VL` 多模态主干本身就不是当前 `TRT-LLM 0.12.0` 的现成模型入口
- `VQA` 还需要你自己管理多模态 embedding 组织和 decode runtime
- `Flow Action` 更进一步：
  - `ActionProcessor.step()`
  - `prefix / postfix`
  - `ODE / Euler`
  - `moe_token_types`
  - 动态 action embedding 注入

这些都让它不再像“标准 decoder-only LLM + 薄 head”。

所以更准确的判断应该是：

> **有些 VLA 能“直接用 `TRT-LLM`”，往往是因为它们真正交给 `TRT-LLM` 的部分，本来就足够像标准 LLM。`wall-x` 的 `Flow Action` 则已经明显超出了这个边界。**

### 7.7 哪类 VLA 更适合直接上 `TRT-LLM`

把这个问题再压得更工程一点，其实可以直接分成两类。

#### A. `TRT-LLM` 友好型 VLA

这类模型通常具备下面几个特征：

1. **视觉前端和语言主干解耦**
   - vision encoder 可以单独跑
   - image embeddings 可以直接作为 prefix / prompt 喂给 decoder

2. **主干仍然是标准 decoder-only LLM**
   - Llama / Qwen / Mistral 这一类骨架
   - 不需要特殊的 layer 级控制流

3. **动作头很薄**
   - 例如最后几层 hidden state 接一个小 MLP
   - 或少量 action token 回归
   - 没有复杂的迭代生成流程

4. **生成流程仍然像普通 autoregressive decode**
   - prefill
   - decode
   - sampling
   - KV cache

这类模型的核心特点是：

> **你真正想加速的 80% 主干，本来就还是标准 LLM decode。**

所以它们通常很适合：

- vision 留在外面
- decoder 交给 `TRT-LLM`
- 薄动作头继续留 host / PyTorch

对外看起来就像：

> **“这个 VLA 可以直接用 `TRT-LLM`。”**

#### B. 必须自己写 runtime 型 VLA

这类模型的典型特征是：

1. **不只是“生成 token”，而是有任务特有的循环**
   - ODE / Euler
   - diffusion / flow matching loop
   - chunk-level rollout

2. **decoder 前后还有大量控制逻辑**
   - prefix / postfix
   - 动态 embedding 注入
   - task-specific cache trim / reuse

3. **动作生成不是薄头，而是完整控制链**
   - 例如 `ActionProcessor.step()`
   - `action_proj_back`
   - 连续动作状态反复回灌

4. **模型内部路由或 mask 不是标准 LLM 假设**
   - `moe_token_types`
   - 非标准 attention mask
   - token/block 级的 task routing

这类模型的核心特点是：

> **你真正想加速的部分，已经不再只是标准 LLM decode，而是一整套 task-specific runtime。**

这时 `TRT-LLM` 最多只能吃其中一部分 block，
但整条链最后还是会逼你自己管理：

- engine 切分
- KV 生命周期
- 动作循环
- task-specific orchestration

#### `wall-x` 属于哪类

`wall-x` 的两条路径其实分别落在这两类的不同位置：

- **VQA**
  - 更接近 `TRT-LLM` 友好型
  - 因为它大体还像标准 decoder decode
  - 只是当前被多模态 glue 和 engine 组织拖住了

- **Flow Action**
  - 明显属于“必须自己写 runtime 型”
  - 因为它最值钱的部分根本不是标准 decode，而是：
    - `prefix / postfix`
    - `ActionProcessor.step()`
    - `ODE / Euler`
    - 动作状态回灌

所以最后最准确的总结是：

> **`wall-x` 不是完全不能借 `TRT-LLM`，而是只能借它加速局部 block；整条 `Flow Action` 路线，最终仍然会逼你自己写 runtime。**

### 7.8 以后看别的具身项目，怎么快速判断值不值得走 TRT 路线

这个分类不只是帮你理解 `wall-x`，也可以直接拿去看别的具身开源项目。

以后遇到一个新的 VLA / policy 项目，其实可以先问下面这几件事：

#### 1. 视觉前端能不能和语言主干解耦

如果答案是：

- **能**
  - vision encoder 自己跑
  - image embeddings 喂给 decoder

那它更可能适合：

- `TRT-LLM` 吃主干
- 你自己只管外层 glue

如果答案是：

- **不能**
  - 视觉和主干强耦合
  - 视觉中间态频繁参与后续控制逻辑

那它更可能会逼你自己写 runtime。

#### 2. 主干是不是标准 decoder-only LLM

如果主干还是标准：

- decoder-only
- autoregressive
- 标准 KV cache

那它通常更接近 `TRT-LLM` 友好型。

如果主干已经有：

- 非标准路由
- 非标准 attention mask
- 复杂的 token-type 控制

那它就会更快滑向“自己写 runtime 型”。

#### 3. 动作头是薄头还是完整控制链

如果动作头只是：

- 最后 hidden state 接一个小 MLP
- 或少量 action token projection

那通常还比较容易把大部分价值交给 `TRT-LLM`。

如果动作头本身已经变成：

- ODE / Euler
- diffusion / flow loop
- chunk rollout
- action state 回灌

那说明真正值钱的部分已经不再只是 LLM decode 了。

#### 4. 推理是不是“单次 forward 完成”

如果一次推理大体是：

- image -> decoder -> answer / action

那 `TensorRT / TRT-LLM` 的价值通常比较直接。

如果一次推理其实是：

- decoder
-> 外层状态更新
-> 再 decoder
-> 再更新
-> 连续很多步

那你最后很大概率还是得自己写 runtime orchestration。

#### 5. runtime 真正的热点在哪里

如果热点主要在：

- attention
- GEMM
- KV cache

那 TRT 路线通常比较有戏。

如果热点主要在：

- 多 engine orchestration
- prefix / postfix
- host 控制流
- 动作循环
- 小块张量组织

那即使局部 block 能进 TRT，端到端也不一定自动值。

### 一个最短的判断公式

以后你可以粗暴用这句去筛：

> **越像“标准 LLM + 外挂 vision + 薄动作头”，越适合先上 `TRT-LLM`；越像“带控制环的专用机器人 runtime”，越应该准备自己写 runtime。**

### 放到你后面要看的项目上

所以你后面再看别的具身项目时，先别急着问：

- “它能不能用 TRT？”

而先问：

- “它真正值钱的 80% 计算，像不像标准 LLM runtime？”

如果答案是“像”，那就值得先试 `TRT-LLM`。  
如果答案是“不像”，那就别对 `TRT-LLM` 抱太高期望，直接把重点放在：

- block 级别加速
- 自己的 runtime 组织
- 或者更彻底的系统设计

这也是当前 `wall-x` spike 最值得带走的经验之一。

---

## 8. 能不能收成 1 个 engine 文件

这个问题需要分开看。

### 8.1 VQA

对 VQA 来说，**收成 1 个 engine 文件是有可能的**。

当前 VQA 之所以变成：

- `prefill.engine`
- `decode_past420.engine`
- `decode_past421.engine`
- `decode_past422.engine`
- `decode_past423.engine`

不是因为它本质上必须这样，而是因为当前 spike 用的是：

> **每个 `past_len` 单独静态编译一份 engine**

更合理的演化应该是：

#### 当前 spike 形态

```text
prefill.engine
+ decode_past420.engine
+ decode_past421.engine
+ decode_past422.engine
+ decode_past423.engine
```

#### 第一阶段收敛

```text
prefill.engine
+ decode_dynamic.engine
```

也就是：

- prefill 单独一份
- decode 用一个支持动态 `past_len` 的 engine

#### 第二阶段极限形态

```text
unified_vqa.engine
```

也就是：

- prefill / decode 共用同一个 engine
- runtime 只负责切 profile / 控制 cache_position / sampling

这一层不是完全不可能，但工程上明显更难，所以正常顺序应该是：

> **先把 5 个 engine 收成 2 个，再决定要不要进一步收成 1 个。**

### 8.2 Flow Action

对 Flow 来说，**理论上并不是绝对不能做成 1 个 engine**，但我判断：

> **它不是合理的工程目标。**

因为 Flow 当前最自然的结构本来就是两段：

- `prefetch`
- `postfix`

而且这两段的职责差异很大：

- `prefetch`：吃完整输入序列，产出 prefix KV 和第一次动作预测
- `postfix`：围绕固定长度 postfix 和 ODE / Euler 循环反复执行

所以 Flow 更自然的演化不是：

```text
很多小 engine -> 1 个 engine
```

而是：

```text
prefetch.engine
+ postfix.engine
```

这已经非常接近合理终态。

如果硬追 1 个 engine，通常只会变成：

- 图更大
- build 更慢
- debug 更难
- engine 更重
- runtime 灵活性更差

而端到端不一定更快。

### 8.3 最现实的目标

所以从当前 spike 往可用系统演化，更现实的目标其实是：

#### VQA

```text
5 个 engine
-> 2 个 engine
-> （可选）1 个 engine
```

#### Flow

```text
2-3 个 engine
-> 2 个 engine
```

这里要特别强调：

> **真正该追求的不是“必须 1 个文件”，而是“最少且合理的 engine 数量”。**

因为最终决定复杂度和性能的，不只是文件数，还有：

- engine 体积
- 动态 shape 能力
- KV cache 管理方式
- runtime orchestration 成本

### 8.4 当前判断

把这个问题压成一句话：

> **VQA 有机会最终收成 1 个 engine，但先收成 2 个更现实；Flow 理论上也许能追 1 个 engine，但工程上基本不值得，2 个 engine 已经接近合理终态。**

---

## 9. `TensorRT-Edge-LLM` 的最小 VQA benchmark 启发

在 `TensorRT-Edge-LLM` 这条官方 edge 路线上，我们已经把 `Qwen2.5-VL-3B` 的 VQA 最小样例跑通了：

- `llm.engine`
- `visual.engine`
- `llm_inference`

最小 wall-clock benchmark（3 次正式运行）：

- `run_1_ms = 7704.467`
- `run_2_ms = 7266.907`
- `run_3_ms = 7626.541`
- `mean_ms = 7532.639`

这组结果的意义在于：

1. **它证明 `TensorRT-Edge-LLM` 不是只在文档里支持，而是真的在 Orin 上闭环了。**
2. **但也说明默认形态下的端到端时延并不低。**

所以对 `wall-x` 的判断就更清楚了：

> **如果首要目标是“把端到端时延压到最好”，当前手工 `cpp_infer` / 自建 TRT runtime 仍然更有吸引力；如果首要目标是“对齐 NVIDIA 官方 edge runtime 形态并保持更高可维护性”，`TensorRT-Edge-LLM` 值得继续研究。**

后续我们又把 `wall-x` 自己现有的 `16` 个 VQA case 全量跑了一遍，两个官方模型的平均结果分别是：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`7488.74 ms`
  - 平均文本相似度：`0.3636`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`5452.96 ms`
  - 平均文本相似度：`0.4041`

这说明：

> **`TensorRT-Edge-LLM` 的官方 VLM 路线在 wall-x 现有 VQA case 集上是可跑的，但当前默认输出分布仍然和 wall-x baseline 有明显差异；更小的模型更快，也略更接近 baseline，但还远没到“直接替换 wall-x VQA backend”的程度。**

我们还做了一个更关键的输入组织消融：

- 去掉 system prompt
- 将 `max_generate_length` 对齐到 wall-x reference token 数

6-case 代表性 batch 的结果变成：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`9536.22 ms`
  - 平均文本相似度：`0.4285`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`6604.15 ms`
  - 平均文本相似度：`0.5216`

这让判断更细了一层：

> **`TensorRT-Edge-LLM` 对 wall-x 的对齐不是只有“模型权重是否合适”的问题，输入组织本身也会明显影响语义输出；但即使把 prompt 和 token 长度调到更像 wall-x，它也还没有自动变成 wall-x baseline。**

再按问题类型看 best preset 的 full16 结果：

- `Describe what you see in this image.`
  - `Qwen2.5-VL-3B`: `6723.48 ms / 0.4652`
  - `Qwen3-VL-2B`: `5250.01 ms / 0.4410`
- `What objects are on the table?`
  - `Qwen2.5-VL-3B`: `6838.53 ms / 0.3988`
  - `Qwen3-VL-2B`: `5046.15 ms / 0.5041`

这说明：

> **如果后面要先挑一个 Edge-LLM 官方 engine 去承接 wall-x VQA，`Qwen3-VL-2B` 更适合作为第一候选，尤其是在“对象枚举”类问题上。**

最后，我们把这件事推进到了主工程入口级别：

- `scripts/vqa_backend_switch.py`
- `scripts/vqa_backend_compare.py`

在 Orin 上，同图同题的单 case 对照已经可以直接从主工程入口跑出：

- `Edge-LLM / Qwen3-VL-2B`
  - `latency_ms = 4970.847`
  - 输出：`This image is a simple, minimalist graphic composed of several geometric shapes on a two-tone background. The`
- `wall-x`
  - 输出：`The image features a minimalist composition with a brown background. In the center, there are three distinct shapes`
- 文本相似度：`0.4732`

这说明：

> **现在已经有了一个主工程入口层面的“可替换前对照”：同一张图同一个问题，可以切 wall-x / Edge-LLM 两个 backend 直接比较。**

我们还专门做了 `propri/action` 的字面串 probe：

- `What objects are on the table?`
  - 加 `Proprioception: <|propri|>` / `<|action|>` 后，Edge-LLM 仍然保持对象枚举风格
- `Predict the next action in robot action.`
  - 加 `Proprioception: <|propri|> <|action|>` 后，Edge-LLM 才开始输出动作描述

这说明：

> **`propri/action` 作为文本 prompt 的字面内容不会报错，但并不会在 Edge-LLM 里自动变成 wall-x 的专用动作 token 体系。**

随后我们又把这个主工程入口扩成了 6-case 的 wall-x vs Edge 双后端对照：

- 图片：
  - `blocks_and_plates`
  - `dual_arm_robot`
  - `fruits_on_table`
- 问题：
  - `Describe what you see in this image.`
  - `What objects are on the table?`

结果：

- `mean_edge_latency_ms = 5508.13`
- `mean_similarity = 0.5216`

这说明：

> **从主工程入口层看，Edge-LLM 已经足够成为 wall-x VQA 的可替换候选；但它目前更像“对象枚举类问题优先可替换”，而不是所有 VQA 问题都已经无缝等价。**

进一步地，我们把两套 Edge backend 做成了一个规则路由器：

- `Describe ...` -> `Qwen2.5-VL-3B`
- `What objects ...` -> `Qwen3-VL-2B`

这个 `edge_router` 在完整 `16-case` 上的结果是：

- `mean_latency_ms = 5707.43`
- `mean_similarity = 0.4846`

这说明：

> **如果只比较 Edge 家族内部，按问题类型做路由，已经能比单一 `Qwen2.5-VL-3B` 或单一 `Qwen3-VL-2B` 更接近 wall-x baseline。**

而且这条路由已经不是离线 benchmark 了：

- `launch_vqa_serving.py --backend edge_router`
- `VQAClient`

在 Orin 上已经验证 websocket 服务层会把对象问题路由到 `qwen3`，把描述问题路由到 `qwen25`。

再往前一步，我们已经把它推进到了 websocket serving 层：

- `launch_vqa_serving.py --backend edge`
- 客户端发送 numpy 图像 + prompt
- 服务端返回 `vqa` 结果和 `server_timing.infer_ms`

这意味着：

> **`TensorRT-Edge-LLM` 对 wall-x 的接入已经不只是脚本级验证，而是已经达到服务层可用的程度。**

再往下看静态模板和 token 体系，边界也很清楚：

- `wall-x`
  - `processor_class = Qwen2_5_VLProcessor`
  - 有 `You are a helpful assistant.` fallback
  - 有 `tools` 分支
  - 有 `propri` / `action`
  - 有 `2048` 个 `action_token_*`
- `Edge-LLM / Qwen2.5-VL-3B`
  - 默认 system prompt 为 `You are a helpful assistant.`
  - 没有 `propri` / `action`
- `Edge-LLM / Qwen3-VL-2B`
  - 默认 system prompt 为空
  - 没有 `propri` / `action`

这意味着：

> **VQA 的问题还主要集中在 prompt / 生成长度 / 输出风格，但 Flow 不是，因为 Flow 对应的 token 体系在官方 VLM engine 里根本就不存在。**

后面我们没有停在单一最小样例上，而是继续把 `wall-x` 现有 VQA case 抽成了 6 个代表性样本，在同一条 Edge-LLM 路线上批量跑：

- `blocks_and_plates`
- `dual_arm_robot`
- `fruits_on_table`

每张图配两种问法：

- `Describe what you see in this image.`
- `What objects are on the table?`

批量结果是：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`7542.18 ms`
  - 平均文本相似度：`0.3611`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`5436.60 ms`
  - 平均文本相似度：`0.3944`

这进一步说明：

> **`TensorRT-Edge-LLM` 对 wall-x 风格的 VQA case 是可跑的，但当前默认官方路径和 wall-x baseline 的语义分布仍然有明显差异；更小的模型更快，但并没有自动更贴近 wall-x 的回答风格。**

---

## 9. 问题最后到底变成了什么

做到这里，其实问题已经不再只是：

- 某个算子能不能进 `TensorRT`
- 某个 block 能不能进 `TRT-LLM`

而是更本质的两个问题：

1. **`TensorRT` 能不能接住标准 LLM 的执行后端需求**
2. **`TRT-LLM` 这套“标准 LLM runtime 抽象”，到底能不能接住 VLA**

当前这轮 spike 给出的答案其实已经比较明确了。

### 9.1 `TensorRT` 本身能不能推 LLM

答案是：

> **能，但它解决的是“执行后端”问题，不是“完整推理系统”问题。**

也就是说：

- attention
- GEMM
- KV cache 相关张量
- decode block

这些东西，本身都可以交给 `TensorRT` 去算。

所以就“LLM 的核心计算块能不能落到 `TensorRT`”这个问题，答案基本已经不是障碍。

### 9.2 `TRT-LLM` 能不能直接推 VLA

答案要分清楚：

- **标准 LLM / 接近标准 LLM 的 VLA**
  - 很多时候可以
- **控制流很重的 VLA**
  - 通常不行，或者说：**不能只靠它**

`TRT-LLM` 本质上提供的是：

- 标准 decoder-only LLM 模板
- prefill / decode loop
- KV cache
- sampling
- 常见 plugin / fused path

这套东西能很好接住的是：

> **“标准 LLM runtime 问题”**

但 `VLA` 尤其是 `Flow Action` 这类东西，真正麻烦的地方已经不是：

- 普通 token decode

而是：

- action state 回灌
- `prefix / postfix`
- `moe_token_types`
- `ActionProcessor.step()`
- ODE / Euler
- task-specific 控制流

也就是说，问题已经从：

> “标准 LLM runtime”

变成了：

> **“面向具身控制的专用 runtime”**

### 9.3 为什么“把 VLA 变成 VLM + action”不总是现实

这也是这轮实验最重要的认识之一。

在一些比较轻的 VLA 里，把问题拆成：

- VLM 主干
- 再接一个薄动作头

是现实的，所以这些模型常常显得更 `TRT-LLM` 友好。

但对 `wall-x` 尤其 `Flow Action` 来说，这样拆并不自然，因为它的动作生成不是一个薄头，而是：

- 反复迭代的动作状态
- task-specific runtime 循环
- decoder 与动作状态双向耦合

这类模型本质上更像：

> **“控制系统 + 神经网络执行后端”**

而不是：

> **“一个标准 VLM，再外挂一个小 action head”**

所以如果硬把它重新表述成 “VLM + action head”，很多真正值钱的复杂度其实只是被你藏起来了，并没有消失。

### 9.4 当前最稳的系统判断

所以到这里，比较硬的系统判断应该是：

> **端侧最终还是要有一层自己的 runtime。**

只是这层 runtime 不一定什么都自己算，它更可能是：

- `TensorRT` 负责底层执行
- `TRT-LLM` 能吃掉一部分标准 LLM 风格的块
- 你自己的 runtime 负责把具身任务真正组织起来

也就是说，端侧真正成熟的形态更像：

> **`TensorRT / TRT-LLM` 作为加速后端 + 一层面向 VLA 的专用 runtime**

而不是：

> **“只要上了 `TRT-LLM`，VLA 推理系统问题就自动解决了。”**

### 9.5 最短结论

压成一句话：

> **问题最后确实变成了：`TensorRT` 能不能接住 LLM 计算，`TRT-LLM` 能不能接住标准 LLM runtime，而 VLA 尤其控制流很重的 VLA，最终仍然会逼你在端侧自己写一层 runtime。**

### 9.6 Edge-LLM Flow Action bridge

我们后来又把 `wall-x` 的 `Flow Action` 单独桥接到了 `TensorRT-Edge-LLM action_build`。

Orin 上已经实际完成：

- wall-x flow-action step ONNX export
- `action_build` 生成 `action.engine`
- 自定义 runner 跑单步 denoise
- host Euler loop 跑完整 flow

结果：

- `action_step_ms = 31.435`
- `denoised_vs_ref cosine = 0.99999797`
- `flow_final_cosine = 0.98523772`
- `flow_step_ms_mean = 31.347`

热 benchmark（`--warmup 2 --iters 5`）下：

- `action_step_ms_mean = 31.329`
- `action_step_ms_std = 0.018`
- `flow_step_ms_mean = 31.335`
- `flow_step_ms_std = 0.032`
- `flow_total_ms_mean = 157.661`
- `flow_total_ms_std = 0.138`

这说明 `TensorRT-Edge-LLM` 不只是 VQA/VLM 可用；在 wall-x 上，它已经能通过自定义 bridge 承接 Flow Action。

边界也很清楚：

- 这是 `custom bridge`
- 不是 stock `llm_inference` 直接吃 wall-x Flow
- host 侧仍然保留 Euler loop/runtime 组织

### 9.7 Flow Action 三方数据对比

把当前 `wall-x Flow Action` 的三条路线放在一起，数据如下：

| 路线 | Flow Action latency | 相对 `cpp_infer` |
|---|---:|---:|
| `cpp_infer` | `290.7 ms` | `1.00x` |
| 手工 TRT / TRT-LLM | `176.837 ms` | `1.65x faster` |
| `TensorRT-Edge-LLM` custom bridge | `157.661 ms` | `1.85x faster` |

其中：
- `cpp_infer` 是当前可控基线
- 手工 TRT / TRT-LLM 是我们在 `workspace/trt_spike/` 里做出来的两阶段 Flow 路线
- `TensorRT-Edge-LLM custom bridge` 是我们把 wall-x flow-action step 导出后通过 `action_build` 跑起来的桥接版

所以最后的硬判断可以压成：

> **Flow Action 上，`cpp_infer` 已经被 TRT 路线显著压过；`TensorRT-Edge-LLM` 还能再压一点，但它当前的优势建立在 custom bridge 上，不是 stock 默认 VLM 入口。**

这里的 `custom bridge` 不是“新写了一堆 plugin”。

实际情况是：

- 没有新写 TensorRT/CUDA plugin
- 新写的是 wall-x Flow Action 的 ONNX 导出桥接层和 runner
- 复用的是 Edge-LLM 自带的 `action_build` / builder / plugin runtime

兼容性关键点：

- ONNX opset 23 会引入 native `RMSNormalization`
- 当前 Edge-LLM action build 路径在 Orin 上不接受这个 op
- 把导出降到 opset 22 以后，图保持为标准 primitive ops，engine build 成功

还有一个需要单独记住的现实边界：

- 当前这条 `Edge-LLM custom bridge` **没有做额外量化**
- `157.661 ms` 不是 INT8 / W8A8 结果
- 按当前 `TensorRT-Edge-LLM` 源码，`export_action` / `action_export` 这条 action-expert 路线目前仍然是 **FP16-only**

所以更准确的说法是：

> **VQA/VLM 的 Edge-LLM 路线有量化故事；我们当前这条 wall-x Flow custom bridge 还没有。**

## 10. 最终建议

### 10.1 VQA

- 保留 `cpp_infer` 基线
- Edge-LLM 作为可切 backend / 路由候选

补充一条现实边界：

- 我们已经用统一脚本把未量化 `Qwen2.5-VL-3B-Instruct` 重跑了一遍，得到更干净的 VQA 基线：
  - `mean_ms = 7664.731`
- 但 `Qwen2.5-VL-3B-Instruct-AWQ` 目前还没拿到最终 wall-clock
  - CPU export 会卡在 `aten::_convert_weight_to_int4pack_for_cpu` 无法导 ONNX
  - CUDA export 又被当前 `venv_edge` 的 CPU-only torch 卡住

所以到现在为止：

> **Edge-LLM 的 VQA 量化路线还在“明确阻塞点已定位”的阶段，还没有进入“量化后确实更快”的阶段。**

补一条最新状态：

- 切到推荐的 `experimental/llm_loader` 路线之后
- `Qwen2.5-VL-3B-Instruct-AWQ` 已经能走到：
  - model type 识别
  - checkpoint load
  - pre-quantized AWQ tensor load
- 我们随后继续把这条线往前推到了：
  - `model.onnx` 成功落盘
  - 通过 `workspace/patch_awq_onnx_dynamic_shapes.py` 把静态 `1x1` 输入改回动态符号轴
  - `llm_build` 最终成功生成 `llm.engine`
  - 运行时也能成功加载 AWQ `llm.engine` 与现有 visual engine

但当前新的硬边界是：

- 这份 AWQ `llm.engine` 实际仍然是 **decode-only profile**
- 为了让 `llm_build` 通过，我们当前只能用：
  - `--maxInputLen=2`
- 结果运行时配置里 `maxSupportedInputLength = 2`
- 一旦跑完整 VQA，请求 prefill 长度约 `417`，就会直接失败：
  - `The max input length (417) exceeds the max supported input length (2) of the LLM Engine.`
- 我们随后又直接尝试了更接近真实 VQA prefill 的 builder 配置：
  - `--maxInputLen=417`
- 这次不再是 runtime 长度检查失败，而是 builder 在 profile 0 就报：
  - `IShuffleLayer /_model/model/layers.0/self_attn/Reshape: reshaping failed for tensor: /_model/model/layers.0/self_attn/AttentionPlugin_output_0 reshape would change volume 425984 to 2048`
- 这说明：
  - 不是简单把 `maxInputLen` 调大就能让 AWQ 路线承接完整 prefill
  - 当前导出的 AWQ LLM 图在结构上仍然只适合 decode-only
- 我们随后又继续做了更激进的 ONNX 后处理：
  - 把所有 `self_attn` 的 reshape 常量从 `[1, 1, 2048]` 改成 `[0, 0, 2048]`
  - 用这份 patched ONNX 重新 build 后，`llm.engine` 成功生成
  - 运行时也能成功加载 AWQ `llm.engine` 与现有 visual engine
- 但新的现实边界是：
  - 纯文本 prompt 和完整 VQA prompt 都能跑完 request
  - `output_text` 仍然为空字符串
  - 我们又补做了一条最小纯文本对照：
    - FP16 `Qwen2.5-VL-3B` 同一请求会生成 `8` 个 token，输出 `2+2 is 4.`
    - FP16 的 token ids 是：`[17, 10, 17, 374, 220, 19, 13, 151645]`
  - AWQ 路线只记录到 `generated_tokens = 1`
  - AWQ 的 token ids 只有：`[151645]`
  - profile 里也没有正常的 `llm_generation` stage
  - `special_tokens_map.json` 显示：
    - `eos_token = <|im_end|>`
  - 用 tokenizer 直接 decode `151645` 的结果是空字符串
  - `id_to_token(151645) = <|im_end|>`
  - 我们还加了 runtime 级 `DEBUG_TOP1`：
    - FP16 在 prefill / decode 中正常生成正文 token，最后才落到 `151645`
    - AWQ 在 prefill 阶段首 token 就直接是 `151645`
    - 具体日志里：
      - FP16 prefill 首 token = `17`，score = `29.203125`
      - AWQ prefill 首 token = `151645`，score = `16.937500`
    - 也就是说 AWQ 不是“decode 过程慢慢塌掉”，而是第一步采样就把 EOS 顶成 top1
  - 我们又加了 runtime 级 `DEBUG_TOPK=5`：
    - FP16 prefill top-5：
      - `(17,29.2031), (785,27.4062), (11613,24.875), (19,24.6719), (1249,24.1094)`
    - AWQ prefill top-5：
      - `(151645,16.9375), (151644,16.0469), (151657,13.0469), (17,12.5625), (785,12.4922)`
    - 这说明问题不在 sampling，而在 AWQ prefill logits 分布本身：`<|im_end|>` 已经稳居第一
  - 我们还做了 checkpoint 侧的第一层 `q/k/v` 对照：
    - 原始 dequant 权重和 FP16 权重的 cosine 都接近 `0.99`
    - 把同一条 prompt 的 embedding 喂进第一层 `q_proj/k_proj/v_proj` 后：
      - `q_proj cosine = 0.999968`
      - `k_proj cosine = 1.0`
      - `v_proj cosine = 0.999745`
    - 说明问题不在第一层输入投影本身
  - 我们又比较了第一层 `LayerNorm + MLP`：
    - `input_layernorm cosine = 0.987499`
    - `post_attention_layernorm cosine = 0.974186`
    - 但 `mlp_out` 在 AWQ 路线里已经直接塌成全 0
    - 这说明问题已经明确收缩到第一层 MLP 路线
  - 我们还把 AWQ 改成随机采样模式再测了一次（`temperature=1.0, top_k=50`）：
    - `DEBUG_TOP1` 仍然直接给出 `token=151645`
    - `score=0.000000`
    - `output_text` 仍然为空
    - 这说明问题不是 greedy-only，而是 prefill 后 logits 本身已经塌到 EOS
  - 说明这条路已经从“不能 prefill”推进到了“能 prefill、能 load、能跑”，但 AWQ 首 token 直接塌成了 `eos`，生成链本身还没有成立

所以现在最准确的判断是：

> **AWQ VLM 不再只是环境问题，也不再只是 repack 问题。当前已经能导出 ONNX、能 build `llm.engine`、能加载 runtime、也能承接 prefill；但 runtime 级 `DEBUG_TOP1/DEBUG_TOPK`、随机采样，以及 checkpoint 侧第一层 `q/k/v` 对照已经共同说明：问题不在 sampling，也不在第一层输入投影，而是在更深层的量化后 logits 生成链。**

### 10.2 Flow Action

- 保留手工 TRT / TRT-LLM
- 保留 Edge-LLM custom bridge
- 当前性能最强的是 Edge-LLM custom bridge，但形态仍是 custom bridge

### 10.3 Stock vs custom

- stock runtime 负责官方可维护性
- custom bridge 负责把 wall-x 的控制流硬接进去
