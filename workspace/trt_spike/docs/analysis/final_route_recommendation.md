# Final Route Recommendation

这份文档只回答一个问题：

> **基于当前 Orin 上已经跑出来的事实，`wall-x` 后续主线该怎么走。**

## 1. 直接结论

### 1.1 VQA

建议：

- **保留 `cpp_infer` 作为当前默认基线**
- **保留 `TensorRT-Edge-LLM` 作为可切换 backend / router 候选**

原因：

- `cpp_infer` 当前 VQA 端到端是 `851.2 ms`
- 手工 `TRT / TRT-LLM` 动态 decode 版本当前约 `943.018 ms`
- `TensorRT-Edge-LLM` 官方 VLM 路线当前在 `wall-x` 风格输入上是秒级
- `Edge-LLM` 的 AWQ VLM benchmark 当前仍未闭环：
  - 未量化 clean baseline 已拿到：`7664.731 ms`
  - AWQ 路线的阻塞点已定位，但还没拿到最终时延
  - 当前已经从 legacy export 的环境/ONNX 问题，推进到：
    - `llm_loader` 导出 ONNX 成功
    - `llm.engine` build 成功
    - runtime load 成功
  - 当前新的主阻塞是：
    - reshape 常量补成动态后，已经能承接 prefill
    - 但纯文本和 VQA 请求的 `output_text` 仍然是空
    - 纯文本对照里 FP16 会生成 `8` 个 token，而 AWQ 只生成 `1` 个 token
    - runtime 级 `DEBUG_TOP1` 已证明 AWQ 在 prefill 第一拍就把 EOS 顶成 top1
    - runtime 级 `DEBUG_TOPK=5` 进一步确认：`<|im_end|>` 在 AWQ 的 prefill top-5 里也稳居第一
    - 所以 AWQ 路线还不能当成“可用的量化 VQA backend”
- Edge-LLM 在 `VQA` 上已经能接进主工程入口和 serving 层，但更适合作为：
  - 对象枚举类问题的候选 backend
  - 官方 edge runtime 路线的参考实现

所以：

> **VQA 现在不该整体切离 `cpp_infer`，而应该把 Edge-LLM 当作“可选 backend / router 路线”保留。**

### 1.2 Flow Action

建议：

- **保留手工 `TRT / TRT-LLM` 路线**
- **保留 `TensorRT-Edge-LLM custom bridge` 路线**
- **`cpp_infer` 继续保留为最稳的全控制链基线**

原因：

- `cpp_infer`: `290.7 ms`
- 手工 `TRT / TRT-LLM`: `176.837 ms`
- `TensorRT-Edge-LLM custom bridge`: `157.661 ms`

也就是说：

> **Flow Action 上，性能最强的是 `TensorRT-Edge-LLM custom bridge`，其次是手工 TRT / TRT-LLM，再其次是 `cpp_infer`。**

但边界同样清楚：

- `TensorRT-Edge-LLM` 这里不是 stock `llm_inference`
- 而是：
  - wall-x ONNX export bridge
  - Edge-LLM `action_build`
  - custom runner
  - host Euler loop

所以：

> **Flow Action 现在最值得继续投的是“TRT/Edge custom runtime”方向，而不是期待 stock VLM runtime 直接吃掉 wall-x 控制链。**

补充一条当前已经实测过的边界：

- Flow 的第一版 custom `INT8-SQ` 路线已经能：
  - 导出 ONNX
  - `action_build`
  - 跑通 runner
- 但当前结果是：
  - `flow_total_ms_mean = 77.866`
  - `flow_final_cosine = 0.27758002`

所以：

> **Flow INT8-SQ 现在更像“证明这条量化链理论上可走”的实验，而不是能替代 FP16 custom bridge 的可用方案。**

## 2. 不要混淆的三条线

### 2.1 `cpp_infer`

定位：

- 当前最完整、最可控、最稳的 wall-x 自研 runtime 基线

适合：

- 保住当前可用系统
- 做精度基准
- 做 fallback

### 2.2 手工 `TRT / TRT-LLM`

定位：

- 已经证明对 wall-x 主干和 Flow 有真实收益
- 是“最接近正式专用 runtime”的加速路线

适合：

- 继续做 Flow
- 继续压控制链时延
- 和 `cpp_infer` 互相验证

### 2.3 `TensorRT-Edge-LLM`

定位：

- 官方 edge runtime 路线
- 对 VLM/VQA 支持最自然
- 对 wall-x Flow 目前需要 custom bridge

适合：

- 做 VQA backend 候选
- 做官方 edge runtime 形态参考
- 做 wall-x Flow 的 action-engine bridge 实验

## 3. 当前推荐主线

### 3.1 如果目标是“尽快做成能用系统”

推荐：

- **VQA：默认继续走 `cpp_infer`**
- **VQA 的 Edge-LLM 侧优先做成 `edge_router`**
  - `Describe ...` → `Qwen2.5-VL-3B`
  - `What objects ...` → `Qwen3-VL-2B`
- **Flow：默认继续走 `cpp_infer` 或手工 TRT / TRT-LLM`**
- **Edge-LLM：作为可切 backend / router 保留，不整体替换默认路径**

对主工程服务层来说，当前已经可以明确：

- `wall_x.serving.VQAPolicy(backend=edge)` 已经可切到 Edge-LLM VQA
- `wall_x.serving` 里的 `WallXPolicy` 也已经能在 `backend=edge` 下把观测里的 `image` / `camera_key[0]` 交给 Edge-LLM
- 这说明 **VQA 的 Edge-LLM 路线已经足够进入主工程入口**
- 但 `Flow` 还没有同样的服务层入口收口，当前更自然的下一步仍然是把 `edge_llm_wallx_flow` 这条 custom bridge 接入 service / policy 层，而不是继续深挖 VQA 量化

### 3.2 如果目标是“继续压端到端性能”

推荐：

- **Flow 优先继续投**
  - `TensorRT-Edge-LLM custom bridge`
  - 手工 `TRT / TRT-LLM`
- **VQA 暂时不继续重投性能**
  - 因为当前最值钱的问题不是再抠几毫秒
  - 而是它该不该用官方 edge runtime 去承接部分问题类型

### 3.3 如果目标是“拥抱 NVIDIA 官方 edge runtime 路线”

推荐：

- **先把 Edge-LLM 明确定位成 VQA/VLM backend**
- **不要假设 stock `llm_inference` 能直接吞掉 wall-x Flow**
- **Flow 继续接受 custom bridge / self-runtime 是必要现实**

### 3.4 当前已经落地的服务层事实

- `wall_x.serving.VQAPolicy(backend=edge)` 已经可切到 Edge-LLM VQA
- `wall_x.serving.FlowPolicy` 已经可以挂上 `edge_llm_wallx_flow`
- `wall_x.serving.launch_flow_serving` 在 Orin 上已经完成最小 websocket smoke

所以现在更准确的判断是：

> **VQA 的 Edge-LLM 侧已经进入主工程入口，Flow 的 Edge-LLM custom bridge 也已经进入 `wall_x.serving` 服务层。**

## 4. 最短版本

> **VQA 不要急着替换掉 `cpp_infer`；Flow 应该继续投 TRT / Edge custom runtime；`TensorRT-Edge-LLM` 值得用，但它当前在 wall-x 上最合理的角色是“VQA 的官方 edge backend 候选 + Flow 的 custom bridge 后端”，不是直接替代整套 wall-x runtime。**

## 5. 量化路线补充

AWQ 这条线现在可以停掉，不继续深挖：

- 首 token 已经直接塌到 `<|im_end|>`
- 第一层 `q/k/v` 没坏
- 第一层 `MLP` 已经塌成 0

后续更值得试的是官方量化主路：

- `fp8`
- `nvfp4`
- `mxfp8`
- `int8_sq`

而不是继续在 `Qwen2.5-VL-3B-Instruct-AWQ` 上修补。

## 6. int8_sq 结果

`Qwen3-VL-2B-Instruct + int8_sq` 已经在 Orin 上实测跑通：

- 量化 checkpoint 成功导出
- `llm_loader` LLM / visual ONNX 成功导出
- `llm_build` / `visual_build` 成功
- `llm_inference` 成功
- 单次 VQA wall-clock: **6245.258 ms**

这说明：

> **官方量化主路在 Orin 上是可用的，但它仍然更像官方 edge runtime 路线，而不是直接把 wall-x 的 VQA 压成最优时延。**

同口径继续补一条：

- `Qwen2.5-VL-3B-Instruct + int8_sq`
  - 单次 VQA wall-clock: **8234.588 ms**

所以这一轮同口径结果是：

- `Qwen3-VL-2B + int8_sq`: `6245.258 ms`
- `Qwen2.5-VL-3B + int8_sq`: `8234.588 ms`

也就是说：

> **更大的官方 VLM 模型在 Orin 上只会把量化 VQA 时延继续拉高，而不会自然逼近 wall-x 当前 VQA baseline。**
