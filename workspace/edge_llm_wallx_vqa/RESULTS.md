# Edge-LLM wall-x VQA Adapter Results

当前已经跑完两轮 wall-x VQA batch：

## 1. 代表性 6-case

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`7542.18 ms`
  - 平均相似度：`0.3611`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`5436.60 ms`
  - 平均相似度：`0.3944`

几个直接观察：

- `Qwen3-VL-2B` 比 `Qwen2.5-VL-3B` 更快
- 但两个模型在 wall-x 的 VQA case 上都没有自然贴近 wall-x baseline
- `dual_arm_robot` 和 `fruits_on_table` 的相似度都不高，说明这不是单个 case 偶发偏差，而是整体输出分布差异

## 2. 完整 16-case

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`7488.74 ms`
  - 平均相似度：`0.3636`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`5452.96 ms`
  - 平均相似度：`0.4041`

这说明前 6 个 case 的结论不是采样偏差，而是全量 `16` 个 case 也基本成立：

- `Qwen3-VL-2B` 更快
- `Qwen3-VL-2B` 的平均相似度也略高
- 但两个模型都没有自然贴近 `wall-x` baseline 的语义分布

## 3. 当前结论

如果只看 `wall-x` 现有 VQA case 集：

- `TensorRT-Edge-LLM` **能跑**
- 但当前默认官方 `Qwen-VL` 路线的输出分布**并不能直接当作 wall-x VQA backend**
- 下一步必须进入：
  - `wall-x VQA` 输入组织适配
  - `Qwen2.5-VL -> wall-x VQA` 差异定位
  - 必要时补 plugin / 前处理层

## 4. 输入组织消融

我们进一步做了一个更直接的消融：

- 去掉 system prompt
- 将 `max_generate_length` 对齐到 wall-x reference 的 token 数

在 6-case 代表性 batch 上：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`9536.22 ms`
  - 平均相似度：`0.4285`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`6604.15 ms`
  - 平均相似度：`0.5216`

这说明：

- 输入组织会明显影响输出风格
- 去掉 system prompt 后，相似度明显回升
- 但它仍然没有把 Edge-LLM 自动变成 wall-x baseline

当前最适合 wall-x VQA 的运行方式已经被收敛成一个 preset：

- `--compat-mode wallx_vqa`

它等价于：

- 去掉 system prompt
- 使用 wall-x reference token 数作为 `max_generate_length`

## 5. 模板与 token 体系差异

静态模板对比结果在：

- [prompt_template_comparison.json](/Users/sam/project/github/wall-x/workspace/edge_llm_wallx_vqa/prompt_template_comparison.json)

里面现在已经确认：

- `wall-x tokenizer_config.json`
  - `processor_class = Qwen2_5_VLProcessor`
  - 模板里有 `You are a helpful assistant.` fallback
  - 模板里有 `tools` 分支
  - 有 `image_pad / video_pad / vision_start / vision_end`
  - 还有 `propri` 和 `action` 这套 wall-x 专用 token
  - `action_token_count = 2048`
- `TensorRT-Edge-LLM / Qwen2.5-VL-3B`
  - `default_system_prompt = You are a helpful assistant.`
  - 没有 `propri` / `action`
- `TensorRT-Edge-LLM / Qwen3-VL-2B`
  - `default_system_prompt = ""`
  - 没有 `propri` / `action`

## 6. 最终 best preset 的 full16 结果

在 `--compat-mode wallx_vqa` 下，完整 `16-case` 的最终均值是：

- `Qwen2.5-VL-3B-Instruct`
  - 平均时延：`6781.01 ms`
  - 平均相似度：`0.4320`
- `Qwen3-VL-2B-Instruct`
  - 平均时延：`5148.08 ms`
  - 平均相似度：`0.4726`

按问题类型拆开后：

- `Describe what you see in this image.`
  - `Qwen2.5-VL-3B`: `6723.48 ms / 0.4652`
  - `Qwen3-VL-2B`: `5250.01 ms / 0.4410`
- `What objects are on the table?`
  - `Qwen2.5-VL-3B`: `6838.53 ms / 0.3988`
  - `Qwen3-VL-2B`: `5046.15 ms / 0.5041`

这说明：

- `Qwen3-VL-2B` 在 wall-x VQA 上整体更快
- 在 “What objects are on the table?” 这类对象枚举问题上，它也更容易贴近 wall-x baseline
- 在 “Describe what you see in this image.” 上，`Qwen2.5-VL-3B` 的相似度略高，但时延更大
- 也就是说：
  - `Qwen3-VL-2B` 更像“速度更好、对象枚举更强”
  - `Qwen2.5-VL-3B` 更像“描述类问题稍稳一点，但更慢”

这意味着：

- **VQA**：问题主要集中在输入组织、模板默认值和输出风格
- **Flow**：问题不只在 prompt，还在 token 体系本身就不一致

## 7. backend smoke

我们已经把这条线收成了一个可直接调用的 backend：

- [edge_llm_vqa_backend.py](/Users/sam/project/github/wall-x/workspace/edge_llm_wallx_vqa/edge_llm_vqa_backend.py)
- [smoke_test_backend.py](/Users/sam/project/github/wall-x/workspace/edge_llm_wallx_vqa/smoke_test_backend.py)

在 Orin 上的最小 smoke 结果：

- 模型：`Qwen3-VL-2B-Instruct`
- 图像：`fruits_on_table.png`
- 问题：`Describe what you see in this image.`
- `system_prompt = ""`
- `max_new_tokens = 20`
- `latency_ms = 4989.377`
- `formatted_complete_request`：
  - `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe what you see in this image.<|im_end|>\n<|im_start|>assistant\n`

这说明：

- 这条线已经不是只有 benchmark 脚本
- 而是已经有一个可以被 wall-x 外层直接调用的 VQA backend 雏形

## 8. drop-in wrapper smoke

我们又把 backend 封成了一个更像 wall-x `VQAWrapper.generate()` 的接口：

- [vqa_wrapper_edge_llm.py](/Users/sam/project/github/wall-x/workspace/edge_llm_wallx_vqa/vqa_wrapper_edge_llm.py)
- [run_wrapper_edge_llm.py](/Users/sam/project/github/wall-x/workspace/edge_llm_wallx_vqa/run_wrapper_edge_llm.py)

在 Orin 上的最小 wrapper smoke：

- 模型：`Qwen3-VL-2B-Instruct`
- 图像：`fruits_on_table.png`
- 问题：`Describe what you see in this image.`
- `latency_ms = 5018.217`
- `output_text` 正常返回
- `formatted_complete_request` 仍然是：
  - `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe what you see in this image.<|im_end|>\n<|im_start|>assistant\n`

这说明：

- 我们已经有了一个可直接被 wall-x 外层调用的 Edge-LLM VQA wrapper
- 后续只需要把 wall-x 原来的 VQA backend 切换到这个 wrapper，就能继续做接入对比

## 9. unified backend switch runner

我们又补了一个统一入口：

- [run_vqa_backend_switch.py](/Users/sam/project/github/wall-x/workspace/edge_llm_wallx_vqa/run_vqa_backend_switch.py)

它的目标不是再造一个 benchmark，而是提供一个可切换的外层入口：

- `--backend edge`
- 未来也可以扩成 `--backend wallx`

在 Orin 上，`backend=edge` 已经 smoke 通过：

- 模型：`Qwen3-VL-2B-Instruct`
- 图像：`fruits_on_table.png`
- 问题：`Describe what you see in this image.`
- `latency_ms = 5020.174`
- 输出正常返回

这说明：

- 我们现在不只是有 backend 和 wrapper
- 而是已经有了一个可以切后端的统一入口雏形

## 10. main-repo `scripts/` entrypoint

最后我们把统一入口也抬到了主工程 `scripts/`：

- [scripts/vqa_backend_switch.py](/Users/sam/project/github/wall-x/scripts/vqa_backend_switch.py)

在 Orin 上，这个主工程入口已经成功用 `--backend edge` 跑通：

- 模型：`Qwen3-VL-2B-Instruct`
- 图像：`fruits_on_table.png`
- 问题：`Describe what you see in this image.`
- `latency_ms = 4970.847`
- 输出正常返回

这意味着：

- 这条线已经不只是 workspace 内部实验脚本
- 而是已经能从主工程 `scripts/` 入口切到 Edge-LLM backend

## 11. unified wall-x vs Edge compare

我们已经把同一个主工程入口跑成了双后端对照：

- `backend=edge`
- `backend=wallx`

在 Orin 上，同图同题 `fruits_on_table.png + Describe what you see in this image.` 的单 case 对照是：

- `Edge-LLM / Qwen3-VL-2B`
  - `latency_ms = 4970.847`
  - 输出：`This image is a simple, minimalist graphic composed of several geometric shapes on a two-tone background. The`
- `wall-x`
  - 输出：`The image features a minimalist composition with a brown background. In the center, there are three distinct shapes`
- 文本相似度：`0.4732`

这说明：

- 现在已经能通过同一条 `scripts/vqa_backend_switch.py` / `scripts/vqa_backend_compare.py` 入口，把 wall-x 原生 backend 和 Edge backend 放到同一张图同一问题上比较
- `wall-x` backend 在这个调用链里需要 `/data/wy/wall-x/venv/bin/python`
- `Edge-LLM` backend 则仍然通过 `edge_llm_vqa_backend.py` 驱动 `llm_inference`
- 这已经是“可替换前对照”，不是单纯的 benchmark

## 12. 统一入口 6-case 双后端对照

我们又把这个统一入口扩成了 6 个代表性 case 的双后端对照：

- 图片：
  - `blocks_and_plates.png`
  - `dual_arm_robot.png`
  - `fruits_on_table.png`
- 问题：
  - `Describe what you see in this image.`
  - `What objects are on the table?`

结果：

- `mean_edge_latency_ms = 5508.13`
- `mean_similarity = 0.5216`

按 case 看：

- `blocks_and_plates / Describe`：`0.4579`
- `blocks_and_plates / Objects`：`0.6489`
- `dual_arm_robot / Describe`：`0.4507`
- `dual_arm_robot / Objects`：`0.5257`
- `fruits_on_table / Describe`：`0.4732`
- `fruits_on_table / Objects`：`0.5732`

这说明：

- 通过主工程统一入口，Edge backend 已经能在多 case 上和 wall-x 原生 backend 做直接对照
- 当前最稳定的趋势仍然是：
  - “对象枚举”类问题更容易贴近 wall-x
  - “整体描述”类问题还存在明显风格差异

## 13. serving-layer policy smoke

我们已经把后端能力收进了 `wall_x.serving.vqa_policy.VQAPolicy`：

- `backend=edge`
  - 在 Orin 上能直接返回 `vqa` 结果
- `backend=wallx`
  - 在 Orin 上也能直接返回 `vqa` 结果

最新 smoke 结果：

- `EdgeLLMVQABackend / Qwen3-VL-2B`
  - `latency_ms = 5338.702`
- `WallXVQABackend / wall-oss-flow`
  - `latency_ms = 3461.166`

这说明：

- `VQAPolicy` 这个服务层对象已经能把两条 backend 统一起来
- `backend=edge` 和 `backend=wallx` 现在已经可以在同一入口层切换

## 14. websocket serving smoke

我们还把 VQA backend 接到了真正的 websocket serving 层：

- [wall_x/serving/launch_vqa_serving.py](/Users/sam/project/github/wall-x/wall_x/serving/launch_vqa_serving.py)

在 Orin 上，`backend=edge` 的 websocket server 已经成功启动并完成一次真实 client 往返：

- `META` 能收到：
  - `backend=edge`
  - `max_new_tokens=20`
  - `default_prompt='Describe what you see in this image.'`
- 客户端发送 numpy 图像 + prompt 后：
  - 成功收到 `vqa` response
  - `server_timing.infer_ms = 5015.150`

这意味着：

- `TensorRT-Edge-LLM` 已经不只是脚本级调用
- 它已经能挂到 wall-x 的 websocket serving 层，作为真实 backend 返回结果

## 15. edge router full16

我们又把两套 Edge backend 做成了一个规则路由器：

- `Describe ...` -> `Qwen2.5-VL-3B`
- `What objects ...` -> `Qwen3-VL-2B`

全量 `16-case` 的结果是：

- `mean_latency_ms = 5707.43`
- `mean_similarity = 0.4846`

这比单一模型更好：

- 好于 `Qwen2.5-VL-3B` 的 `0.4320`
- 也好于 `Qwen3-VL-2B` 的 `0.4726`

代价是：

- 时延高于纯 `Qwen3-VL-2B`
- 但仍低于纯 `Qwen2.5-VL-3B`

## 16. edge router websocket smoke

我们还把这个路由器挂到了 websocket 服务层：

- `launch_vqa_serving.py --backend edge_router`

在 Orin 上，客户端发送：

- 图像：`fruits_on_table.png`
- 问题：`What objects are on the table?`

服务端返回：

- `backend = edge_router`
- `routed_backend = qwen3`
- `server_timing.infer_ms ≈ 4908.498`

这说明：

- 规则路由不只是 benchmark 逻辑
- 它已经能在服务层按问题类型真正切换到不同 Edge backend

## 18. special token probe

我们又专门试了 `propri/action` 作为**字面字符串**塞进 Edge-LLM prompt 里会发生什么。

### Qwen3-VL-2B

- `What objects are on the table?`
  - `4930.1 ms`
  - 输出基本不变
- `What objects are on the table? + Proprioception: <|propri|>`
  - `4896.4 ms`
  - 输出基本不变
- `What objects are on the table? + <|action|>`
  - `4930.1 ms`
  - 输出基本不变
- `Predict the next action in robot action. + Proprioception: <|propri|> + <|action|>`
  - `4862.8 ms`
  - 输出变成了动作描述：`move the robot's head to the right`

### Qwen2.5-VL-3B

- `What objects are on the table?`
  - `6548.0 ms`
  - 输出是常规对象枚举
- `What objects are on the table? + Proprioception: <|propri|>`
  - `6857.0 ms`
  - 输出风格开始变化，但仍然不是 wall-x 的动作 token 语义
- `What objects are on the table? + <|action|>`
  - `6561.7 ms`
  - 输出对象数目和表述风格发生变化
- `Predict the next action in robot action. + Proprioception: <|propri|> + <|action|>`
  - `6528.9 ms`
  - 输出变成动作预测句：`move its arm towards the red apple`

这说明：

- `propri/action` 作为**文本 prompt 的字面内容**不会报错
- 但它们并没有在 Edge-LLM 里变成 wall-x 的专用动作 token
- 更像是 prompt 风格扰动，而不是 token 体系对齐

## 15. websocket client smoke

我们又补了主包内的 websocket 客户端：

- [wall_x.serving.VQAClient](/Users/sam/project/github/wall-x/wall_x/serving/vqa_client.py)

这意味着：

- `TensorRT-Edge-LLM` 现在不只是有 VQA server
- 还可以通过仓库主包内的 client 发起同样的 observation 请求
- server / client 两边都已经能在同一协议下工作

## 16. edge router full16

我们把两套 Edge backend 做成了一个规则路由器：

- `Describe ...` -> `Qwen2.5-VL-3B`
- `What objects ...` -> `Qwen3-VL-2B`

全量 `16-case` 的结果：

- `mean_latency_ms = 5707.43`
- `mean_similarity = 0.4846`

这比单一模型更好：

- 好于 `Qwen2.5-VL-3B` 的 `0.4320`
- 也好于 `Qwen3-VL-2B` 的 `0.4726`

代价是：

- 时延高于纯 `Qwen3-VL-2B`
- 但仍低于纯 `Qwen2.5-VL-3B`

## 17. edge router websocket smoke

我们还把这个路由器挂到了 websocket 服务层：

- `launch_vqa_serving.py --backend edge_router`

在 Orin 上，客户端发送：

- 图像：`fruits_on_table.png`
- 问题：`What objects are on the table?`

服务端返回：

- `backend = edge_router`
- `routed_backend = qwen3`
- `server_timing.infer_ms ≈ 4908.498`

再测 `Describe what you see in this image.` 时，路由到了：

- `routed_backend = qwen25`

这说明：

- 规则路由不只是 benchmark 逻辑
- 它已经能在服务层按问题类型真正切换到不同 Edge backend

## 18. special token probe

我们又专门试了 `propri/action` 作为**字面字符串**塞进 Edge-LLM prompt 里会发生什么。

### Qwen3-VL-2B

- `What objects are on the table?`
  - `4930.1 ms`
  - 输出基本不变
- `What objects are on the table? + Proprioception: <|propri|>`
  - `4896.4 ms`
  - 输出基本不变
- `What objects are on the table? + <|action|>`
  - `4930.1 ms`
  - 输出基本不变
- `Predict the next action in robot action. + Proprioception: <|propri|> + <|action|>`
  - `4862.8 ms`
  - 输出变成了动作描述：`move the robot's head to the right`

### Qwen2.5-VL-3B

- `What objects are on the table?`
  - `6548.0 ms`
  - 输出是常规对象枚举
- `What objects are on the table? + Proprioception: <|propri|>`
  - `6857.0 ms`
  - 输出开始偏向“对象列表”风格，但仍然不是 wall-x 的动作 token 语义
- `What objects are on the table? + <|action|>`
  - `6561.7 ms`
  - 输出对象数目和表述风格发生变化
- `Predict the next action in robot action. + Proprioception: <|propri|> + <|action|>`
  - `6528.9 ms`
  - 输出变成动作预测句：`move its arm towards the red apple`

这说明：

- `propri/action` 作为**文本 prompt 的字面内容**不会报错
- 但它们并没有在 Edge-LLM 里变成 wall-x 的专用动作 token
- 更像是 prompt 风格扰动，而不是 token 体系对齐
