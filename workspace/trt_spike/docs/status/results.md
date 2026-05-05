# TensorRT Spike Results

## 2026-04-29

### 1. 最小 TRT-LLM FMHA（prefill 子图）

环境：

- Orin
- TensorRT `10.3.0.30`
- TensorRT-LLM `0.12.0`
- bf16
- `batch=1`
- `heads=16`
- `kv_heads=2`
- `head_dim=128`
- `seq=420`

结果：

- engine build：`~2.3s`
- `TRT-LLM FMHA prefill`：
  - `Mean 0.079 ms`
  - `Std 0.008 ms`
  - `Min 0.072 ms`
  - `Max 0.126 ms`

脚本参考值：

- `SDPA prefill`: `0.127 ms`
- `FA2 prefill`: `0.306 ms`

当前结论：

- 标准 attention 子图上，`TRT-LLM FMHA` 已经显示出明确性能优势
- 但当前脚本仍有 `past_key_value_0` 地址未绑定的 `enqueueV3` 警告
- 所以这组结果可作为“量级正确”的先验证据，但还不算完全干净

### 2. 当前路线判断

- 下一步优先做 **完整 VQA TensorRT 路线**
- `Flow Action` 暂不作为第一落地点

原因：

- `VQA` 没有 ODE / Euler 循环
- 没有每步动态 action embedding 替换
- 没有 proprioception 动态 scatter
- 存在做成 `VQA-specialized` 路线、固定 `moe_token_types=0`、只走 `expert0` 的简化空间

### 3. Phase A Python 参考底座验证

已完成：

- `export_vqa_decoder_reference.py`
  从 Python `wall-x` 模型导出：
  - `inputs_embeds`
  - `position_ids`
  - `moe_token_types`
  - `prefill_last_logits`
  - greedy `token_ids`
- `phase_a_decoder_wrapper.py`
  实现 `expert0-only` 的 `VQA-specialized decoder`
- `test_phase_a_decoder_wrapper.py`
  对 `prefill last logits` 做 smoke test

对比结果（`real_tabletop_1.jpg`）：

- `last-logits cosine`: `1.00000000`
- `mean abs`: `0.0`
- `max abs`: `0.0`

当前结论：

> **Phase A 的核心假设已经成立：对 VQA 路线做 `expert0-only` 裁剪后，decoder 主体在 Python 参考实现上可以和原始 wall-x 路径做到精确对齐。**

### 4. Phase A ONNX 导出尝试

已尝试：

- `export_phase_a_decoder_onnx.py`
- 先走原始 `sdpa` attention
- 再切到 `force_manual_attention=True`

结果：

- 两条路都没有成功导出 ONNX
- 最终都卡在 `torch.onnx.export` 的图优化阶段
- 错误为：
  - `RuntimeError: ScalarType ComplexDouble is an unexpected tensor scalar type`

当前判断：

- 这说明 **继续深挖 ONNX 导出这条线的投入产出比已经很低**
- 现阶段不该继续在 `ONNX` 上死磕

### 5. 当前路线结论更新

Orin 当前环境里：

- `torch_tensorrt`：**未安装**
- `onnx`：已安装
- `onnxruntime`：已安装

所以从现实条件看，下一步最该切换到：

> **TensorRT Python API 直接构图**

而不是：

- 继续抠 `torch.onnx.export`
- 或者等待 `torch_tensorrt`

### 6. Phase A layer0 TRT-LLM block

已完成：

- `layer0_weights.safetensors`
  - 导出 `qkv fused / o_proj / expert0 gate-up-down / norm / lm_head`
- `layer0_ref.safetensors`
  - 导出 `hidden_in / cos_mrope / sin_mrope / causal_mask_4d / layer0_out`
- `build_phase_a_layer0_trtllm.py`
  - 用 `TRT-LLM` 现成层 + functional 手工拼出：
    - `RmsNorm`
    - `qkv + dense`
    - manual `multimodal_rope`
    - manual attention
    - `expert0-only` MLP
    - residual

对比结果（`bfloat16`）：

- `layer0_out cosine`: `0.99994922`
- `mean abs`: `4.97436523e-03`
- `max abs`: `2.50000000e-01`
- 输出无 `NaN`

当前结论：

> **Phase A 的最小单层 TRT-LLM block 已经跑通，而且数值和 Python reference 基本对齐。**

这说明当前路线不是停留在设计层，而是已经真正拿到：

- 可运行的 `TensorRT / TRT-LLM` 单层 block
- 可量化的数值误差
- 后续扩成多层的可行底座

### 7. Phase A multi-layer prefill decoder

进一步扩到多层后，当前已经跑到：

#### 2 层（prefill-only, hidden_out 对齐）

- `cosine`: `0.99999464`
- `mean abs`: `5.92574710e-03`
- `max abs`: `7.42553711e-01`
- 输出无 `NaN`

#### 4 层（prefill-only, hidden_out 对齐）

- `cosine`: `1.00009239`
- `mean abs`: `7.22634047e-03`
- `max abs`: `1.27404785e+00`
- 输出无 `NaN`
- `BENCH mean`: `11.370 ms`

#### 8 层（prefill-only, hidden_out 对齐）

- `cosine`: `1.00008547`
- `mean abs`: `9.28657502e-03`
- `max abs`: `2.05580139e+00`
- 输出无 `NaN`
- `BENCH mean`: `21.244 ms`

#### 16 层（prefill-only, hidden_out 对齐）

- `cosine`: `1.00011027`
- `mean abs`: `1.23471329e-02`
- `max abs`: `1.72486572e+01`
- 输出无 `NaN`
- `BENCH mean`: `42.622 ms`

#### 36 层（prefill-only, final norm + lm_head）

- 对内部 torch reference：
  - `cosine`: `0.99998784`
  - `mean abs`: `2.19913032e-02`
  - `max abs`: `1.51231766e-01`
- 对原始 `wall-x` 导出的 `prefill_last_logits`：
  - `cosine`: `0.99994069`
  - `mean abs`: `4.11631577e-02`
  - `max abs`: `4.06250000e-01`
- 输出无 `NaN`
- `BENCH mean`: `103.170 ms`

当前结论：

> **多层扩展没有破坏数值稳定性。`expert0-only + manual attention + TRT-LLM layers` 这条路线在 2 / 4 / 8 / 16 层上都已经能稳定对齐，并且在 36 层完整 prefill 上已经能和原始 `wall-x` 的 `prefill_last_logits` 直接对齐。**

这意味着当前阶段最重要的不再是“能不能做多层”，而是：

- 开始做 **decode / KV cache** 这半条链
- 再把 prefill + decode 串成第一版完整 `VQA-specialized` TRT 路线

### 8. Decode reference 已补齐

`decoder_reference.safetensors` 现在已经包含：

- `decode_input_ids`
- `decode_attention_mask`
- `decode_inputs_embeds`
- `decode_position_ids`
- `decode_moe_token_types`
- `decode_last_logits`
- `past_key_0 ... past_key_35`
- `past_value_0 ... past_value_35`

当前结论：

> **Phase A 的 decode 参考接口已经准备好了。**

这意味着下一步可以直接开始做：

- 单步 `decode` engine
- KV cache 输入/输出对齐
- 与原始 `wall-x` 第一步 decode logits 对比

### 9. Phase A single-step decode（2 层）

已完成：

- `build_phase_a_decode_trtllm.py`
  - 吃：
    - `decode_inputs_embeds`
    - `decode_cos_mrope / decode_sin_mrope`
    - `prefill_past_key_i / prefill_past_value_i`
  - 出：
    - `decode_last_logits`
    - `present_past_key_i / present_past_value_i`

当前结果（`2` 层，`bfloat16`）：

- 对内部 torch reference：
  - `cosine`: `0.99556613`
  - `mean abs`: `5.67788064e-01`
  - `max abs`: `4.02746391e+00`
- `BENCH mean`: `5.852 ms`

对原始 `wall-x` 导出的 `decode_last_logits`：

- `cosine`: `-0.18914956`
- `mean abs`: `1.07606573e+01`
- `max abs`: `3.80000000e+01`

这个结果是符合预期的，因为这里对比原始 `wall-x` 时，层数并不一致：

- 当前 TRT decode：`2` 层
- 原始 `wall-x` decode：`36` 层

所以当前阶段真正有意义的判断是：

> **decode 这半条链已经能跑通，而且能和同层数的 torch reference 对齐。**

下一步不再停在 `2` 层，而应该直接冲 `36` 层完整 decode。

### 10. Phase A single-step decode（36 层）

当前结果（`36` 层，`bfloat16`）：

- 对内部 torch reference：
  - `cosine`: `0.99998027`
  - `mean abs`: `3.68854813e-02`
  - `max abs`: `7.76718140e-01`
- `BENCH mean`: `40.360 ms`

对原始 `wall-x` 导出的 `decode_last_logits`：

- `cosine`: `0.99996674`
- `mean abs`: `3.81305739e-02`
- `max abs`: `7.50000000e-01`

当前结论：

> **完整 36 层的单步 decode 主体已经在 TRT-LLM/TensorRT 路线上跑通，并且能和原始 `wall-x` 的 decode logits 直接对齐。**

这意味着，当前 `VQA-specialized` 路线里真正最值钱的两部分：

- `prefill decoder`
- `single-step decode decoder`

都已经有了可运行、可对齐、可 benchmark 的 TensorRT 版本。

### 11. 第一版完整 VQA-specialized TRT runner（5 token）

已完成：

- `run_phase_a_vqa_trtllm.py`
  - 串起：
    - `36` 层 prefill engine
    - `36` 层 single-step decode engine
    - 外层 greedy loop

当前结果（`5` 个 token，`bfloat16`）：

- `prefill`: `102.445 ms`
- `decode mean`: `41.697 ms`
- `total`: `269.232 ms`
- TRT 生成 token：
  - `[1925, 279, 2168, 13, 671]`
- 原始 `wall-x` 参考 token 前缀：
  - `[1925, 279, 6422, 11, 1052, ...]`

当前判断：

- **完整 runner 已经能跑通**
- **前两个 token 能对上**
- **第 3 个 token 开始分叉**

结合当前 `KV` 对比结果，更像是：

> **prefill 产出的 KV 小漂移，在多步 decode 里被逐步放大。**

而不是：

> decode engine 本身完全错误

### 12. 轻量 session 管理后，5 token 完整 runner 已对齐

在把完整 runner 改成：

- `prefill session` 跑完即释放
- `decode session` 按 `past_len` 按需创建/释放

之后，`5 token` 的完整 `VQA-specialized TRT` 路线已经成功跑通，而且 token 前缀与当前参考一致：

- `prefill`: `103.818 ms`
- `decode mean`: `41.897 ms`
- `total`: `271.408 ms`

TRT 生成 token：

- `[1925, 279, 6422, 11, 1052]`

参考 token 前缀：

- `[1925, 279, 6422, 11, 1052, ...]`

当前结论：

> **完整的 `VQA-specialized TRT` 路线已经在 Orin 上跑通，并且在前 5 个 token 上与当前参考 token 序列对齐。**

这说明当前这条路线已经跨过了“只能跑子图 / 只能跑单步”的阶段，开始具备真正拿来和 `cpp_infer` 做完整性能对比的价值。

### 13. 当前工程瓶颈：不是数值，而是 engine/session 组织

当前 `5 token` 的 `VQA-specialized TRT` 路线已经拿到一组可用结果：

- `prefill`: `103.818 ms`
- `decode_times_ms`: `[40.564, 44.143, 41.527, 41.355]`
- `total`: `271.408 ms`
- `generated`:
  - `[1925, 279, 6422, 11, 1052]`
- `reference_prefix`:
  - `[1925, 279, 6422, 11, 1052]`

也就是说：

> **当前 TRT 路线在 Orin 上已经拿到了完整 `5 token` 的性能和文本一致性数据。**

但同时也暴露了一个新的工程瓶颈：

- `prefill.engine`: `5.8G`
- `decode_past420.engine`: `5.8G`
- `decode_past421.engine`: `5.8G`
- `decode_past422.engine`: `5.8G`
- `decode_past423.engine`: `5.8G`
- 整个 `engines/vqa36/` 目录约 `29G`

当前判断：

- 路线已经不是“能不能跑通”的问题
- 数值和 token 前缀已经对齐
- 真正剩下的问题是：
  - **怎么把这组超大 engine 组织成 Orin 能稳定承受的 runtime**

进一步观察到：

- 当 `run-only` 试图重新从磁盘按需加载这些超大 engine 时，进程可能长时间停在 `wait_on_page_bit_common`
- 这更像是 **engine 体积 + 页缓存 / I/O / session 生命周期** 的问题
- 而不是新的模型精度问题

### 13.1 VQA `5 engine -> 2 engine` 动态 decode 原型已跑通

在上面这组 `5 engine` VQA 结果之后，我们又继续往真正可用的 runtime 形态迈了一步：

- [build_phase_a_decode_dynamic_trtllm.py](/Users/sam/project/github/wall-x/workspace/trt_spike/build_phase_a_decode_dynamic_trtllm.py)
- 更新后的 [run_phase_a_vqa_trtllm.py](/Users/sam/project/github/wall-x/workspace/trt_spike/run_phase_a_vqa_trtllm.py)（支持 `--dynamic-decode`）

目标非常直接：

> **把原来每个 `past_len` 一份的 decode engine，先收成一个 `decode_dynamic.engine`。**

最关键的 smoke test 已经通过：

- `past_len = 420`
  - 输出 `present_k0 = (1, 2, 421, 128)`
  - bench `~40.3 ms`
- `past_len = 421`
  - 输出 `present_k0 = (1, 2, 422, 128)`
  - bench `~41.1 ms`

也就是说：

> **同一个 `decode_dynamic.engine` 已经能同时吃 `420` 和 `421` 两个长度，而不是再拆成两份静态 engine。**

在这个基础上，新的 `2 engine` 版 `5 token` VQA runner 也已经跑通：

- `prefill.engine`
- `decode_dynamic.engine`

结果：

- `prefill = 100.928 ms`
- `decode mean = 45.133 ms`
- `total = 281.460 ms`
- `generated = [1925, 279, 6422, 11, 1052]`
- `token_match_prefix = True`

这组结果的意义不在于它现在比 `5 engine` 版更快了多少，而在于：

> **VQA 这条 TRT 路线已经不再只是“每个 `past_len` 编一份”的 spike 形态，而是开始真正进入 “少量 engine + runtime 管理 KV” 的落地阶段。**

当前判断：

- `5 engine` 方案证明了数学可行
- `2 engine` 原型证明了 runtime 方向可行
- 下一步如果继续做 VQA，就不该再围绕 `decode_past420/421/422/423` 这套静态拆法打转了

### 13.2 VQA `20 token` 动态 decode 端到端结果

在 `decode_dynamic.engine` 跑通以后，进一步直接用：

- `prefill.engine`
- `decode_dynamic.engine`

跑了完整 `20 token` 的 VQA 路线。

结果：

- `prefill = 99.937 ms`
- `decode mean = 43.451 ms`
- `total = 925.512 ms`

生成 token：

- `[1925, 279, 6422, 11, 1052, 374, 458, 1787, 2311, 448, 31620, 19281, 11, 892, 374, 264, 2526, 6548, 2311, 13]`

reference：

- `[1925, 279, 6422, 11, 1052, 374, 458, 1787, 2311, 448, 31620, 19281, 11, 892, 374, 264, 2526, 6548, 2311, 13]`

也就是：

- `token_match_prefix = True`

当前结论：

> **VQA 的 `2 engine` 形态已经不只是 5 token 原型，而是完整 20 token 端到端已经跑通，并且文本序列完全对齐。**

但同时也更明确地暴露了当前边界：

- `2 engine` 动态 decode 版本：`925.5 ms`
- `cpp_infer` 当前最佳：`851.2 ms`

所以当前更准确的判断是：

> **VQA 在 TRT 上已经完成了从 “5 engine spike” 到 “2 engine runtime 原型” 的跨越，但端到端性能还没有压过当前 `cpp_infer` 最佳实现。**

补充两个更工程化的事实：

1. 当前 `vqa36_dyn` 目录里只有：
   - `prefill.engine`
   - `decode_dynamic.engine`
   总大小约 `12G`

2. `run-only` 路径也已经稳定复用：
   - `prefill = 100.222 ms`
   - `decode mean = 44.358 ms`
   - `total = 943.018 ms`
   - `token_match_prefix = True`

也就是说：

> **`2 engine` 版本不只是一次 build 成功，而是已经能作为可复用的 runtime 形态稳定跑起来。**

所以当前 `trt_spike` 的下一步，已经不再是继续证明 decoder 数值是否正确，而是优先解决下面这些 runtime 级问题：

- 减少 engine 数量
- 减少单个 engine 体积
- 避免每步重新读入 `5.8G` 级别的 serialized engine
- 在“常驻 session 占内存”和“按需加载占 I/O”之间找到更合理的折中

### 14. 为什么现在会拆成多个 engine，以及它的真实代价

当前这条 TRT 路线在逻辑上还是 **一个 `wall-x VQA-specialized` 模型**，不是多个不同模型。

但在工程实现上，它被拆成了：

- `prefill.engine`
- `decode_past420.engine`
- `decode_past421.engine`
- `decode_past422.engine`
- `decode_past423.engine`

原因不是权重不同，而是当前这条构图路线是 **静态 shape 专门化**：

- `prefill` 和 `decode` 本来就不是同一张图
- `decode` 的 `past_len` 也被写死进了 engine
- 所以当前 runtime 只能按 `past_len` 选择不同的 decode engine

这意味着：

> **现在不是多个模型，而是一个模型被拆成了多个按阶段、按 KV 长度专门化的 TensorRT engine。**

这件事本身不一定等于性能差。

如果这些 engine：

- 体积不大
- 提前建好
- 常驻内存
- session 切换成本低

那多 engine 甚至可能因为 shape 更专门化，拿到更激进的 kernel 选择。

但在当前 Orin 实现里，真正的问题已经不是“多 engine”这三个字，而是：

- 每个 engine 都很大（约 `5.8G`）
- 多个 engine 合起来约 `29G`
- 一旦不常驻，就会引入：
  - 磁盘读入
  - 页缓存抖动
  - 反序列化
  - session 创建/销毁
- 一旦全常驻，又会把 Orin 的内存压力迅速拉高

所以当前的性能损失更准确地说来自：

- `engine` 体积过大
- `session` 生命周期管理过重
- 外层 runtime 需要在多个大 engine 之间做 orchestration

而不是：

- “因为有多个 engine，所以算子本身就更慢”

当前阶段的更硬结论应该是：

> **算子层性能已经基本成立；真正拖后腿的是多个超大 engine 带来的 I/O、session 切换和内存管理成本。**

这也解释了为什么后面的优化重点，不该再放在 layer 数学本身，而应该转向：

- 减少 `decode_past*.engine` 的数量
- 尽量合并成更少的 profile / 更动态的 engine
- 降低单个 engine 的体积
- 重新设计“常驻 session”和“按需加载”的折中策略

### 15. Flow Action dummy reference 已导出

为了开始做 `Flow Action` 的 TRT 路线，先没有直接写 engine，而是先按当前 Python / C++ benchmark 条件导出了一份 dummy reference。

对应脚本：

- [export_flow_dummy_reference.py](/Users/sam/project/github/wall-x/workspace/trt_spike/export_flow_dummy_reference.py)

这份 reference 对齐的是：

- [`scripts/bench_flow_action.py`](/Users/sam/project/github/wall-x/scripts/bench_flow_action.py)
- `seq_len = 488`
- `action_horizon = 32`
- `action_dim = 20`
- `num_inference_timesteps = 5`
- `attn_impl = sdpa`

当前导出的结构已经说明，Flow 很适合先按 “prefetch + postfix” 两段切开：

- `prefix_length = 456`
- `postfix_length = 32`

也就是说，在当前 dummy 条件下：

- 前 `456` 个 token 可以看作 prefix
- 最后的 `32` 个 action token 构成 postfix

导出的关键参考张量包括：

- `inputs_embeds_t0`: `[1, 488, 2048]`
- `prefetch_hidden_states`: `[1, 488, 2048]`
- `prefix_past_key_0`: `[1, 2, 456, 128]`
- `postfix_inputs_embeds`: `[1, 32, 2048]`
- `postfix_attention_mask_3d`: `[1, 32, 488]`
- `action_embed_t0`: `[1, 32, 2048]`
- `action_embed_t1`: `[1, 32, 2048]`
- `v_t0`: `[1, 32, 20]`
- `v_t1`: `[1, 32, 20]`
- `predict_action`: `[1, 32, 20]`

单次参考导出里的时间分布是：

- `embed_processing_ms`: `585.9`
- `position_encoding_ms`: `13.0`
- `action_initialization_ms`: `96.5`
- `prefetch_forward_ms`: `163.4`
- `cache_preprocessing_ms`: `15.1`
- `first_postfix_step_ms`: `112.6`

当前结论：

> **Flow TRT 路线不应该一上来就追完整 ODE 全进 engine，而应该先把 `prefetch decoder` 和 `postfix step` 拆开。**

原因很明确：

- `prefetch_forward` 本身就是一段重量级前向
- `first_postfix_step` 也不是轻量尾巴
- 两段都足够大，单独拿出来进入 TRT 才有价值

所以当前最合理的下一步不是直接做完整 Flow ODE runner，而是：

1. 先做 `prefetch decoder` TRT spike
2. 再做 `postfix step` TRT spike
3. 最后再决定 ODE loop 放 host、放 graph，还是继续切 engine

### 16. Flow prefetch TRT 最小验证（1 层）

基于上面的 dummy Flow reference，又补了一个专门化的 prefetch builder：

- [build_flow_prefetch_trtllm.py](/Users/sam/project/github/wall-x/workspace/trt_spike/build_flow_prefetch_trtllm.py)

当前这版不是完整 Flow runtime，只覆盖：

- `prefetch decoder`
- shared attention
- shared norm
- MLP MoE 的一个专门化假设：
  - 前 `456` 个 token 走 `expert0`
  - 后 `32` 个 token 走 `expert1`

这个假设之所以成立，是因为当前 dummy Flow 输入里：

- `moe_token_types` 恰好是连续块
  - 前 `456` 个全 `0`
  - 后 `32` 个全 `1`

所以第一版可以先不处理通用 `permute / unpermute`，直接按连续切片把 expert0/expert1 分开做。

当前 `1` 层结果（`bfloat16`）：

- build time:
  - `42.6s`
- `TRT vs torch_ref`:
  - `cosine = 0.99990577`
  - `mean_abs = 2.79e-02`
  - `max_abs = 1.91`
- `KV0 prefix slice`:
  - `cosine = 1.00000930`
  - `mean_abs = 4.94e-03`
  - `max_abs = 5.00e-01`
- bench:
  - `3.149 ms`

当前结论：

> **Flow prefetch 这条“shared attention + contiguous expert0/expert1 MLP”专门化路线已经在 1 层上跑通，而且和 torch reference 基本对齐。**

这说明当前 Flow TRT 路线不是停留在分析层，而是已经具备继续往多层放大的基础。

下一步最自然的推进顺序是：

1. 先把 `prefetch decoder` 从 `1` 层扩到更多层
2. 再开始做 `postfix step`
3. 最后再决定是否值得把完整 ODE loop 串起来

### 17. Python Flow Action baseline（dummy 条件）

为了给后面的 TRT `Flow` 路线提供端到端对照，也补跑了当前 Python baseline：

- [`scripts/bench_flow_action.py`](/Users/sam/project/github/wall-x/scripts/bench_flow_action.py)

条件：

- `seq_len = 488`
- `action_horizon = 32`
- `num_inference_timesteps = 5`
- `attn_impl = sdpa`
- 与 dummy Flow reference 对齐

结果：

- Average total:
  - `1417.1 ms`
- Avg `embed+ViT`:
  - `378.9 ms`
- Avg `prefill`:
  - `251.8 ms`
- Avg `ODE`:
  - `769.9 ms`
- Avg `other`:
  - `16.6 ms`
- Throughput:
  - `0.71 infer/s`
- Peak GPU memory:
  - `8.08 GB`

当前判断：

> **在 dummy Flow 条件下，`prefill` 本身已经是一个很值得先吃下的子问题；但更大的总成本仍然在 ODE/postfix 循环。**

所以当前的 Flow TRT 优先级仍然成立：

1. 先做 `prefetch decoder`
2. 再做 `postfix step`
3. 最后再看完整 ODE runtime

### 18. Flow prefetch TRT 多层扩展

在 `1` 层最小验证之后，继续把同一条专门化路线扩到了 `4` 层和 `36` 层。

#### 4 层（bfloat16）

- build time:
  - `44.5s`
- `TRT vs torch_ref`:
  - `cosine = 0.99992013`
  - `mean_abs = 2.84e-02`
  - `max_abs = 1.63`
- `KV0 prefix slice`:
  - `cosine = 1.00000930`
  - `mean_abs = 4.94e-03`
  - `max_abs = 5.00e-01`
- bench:
  - `12.263 ms`

#### 36 层（bfloat16）

- build time:
  - `70.5s`
- `torch vs exported_ref`:
  - `cosine = 0.99860704`
  - `mean_abs = 4.19e-02`
  - `max_abs = 17.30`
- `TRT vs torch_ref`:
  - `cosine = 0.99822533`
  - `mean_abs = 4.03e-02`
  - `max_abs = 43.37`
- `TRT vs exported_ref`:
  - `cosine = 0.99640810`
  - `mean_abs = 4.65e-02`
  - `max_abs = 52.00`
- `KV0 prefix slice`:
  - `cosine = 1.00000930`
  - `mean_abs = 4.94e-03`
  - `max_abs = 5.00e-01`
- bench:
  - `114.735 ms`

当前结论：

> **Flow prefetch 的 TRT 路线已经不只是单层成立，而是已经扩到完整 36 层，并且数值上仍然基本对齐。**

把它和当前 Python Flow baseline 的平均 `prefill` 对比：

- Python Flow prefill:
  - `251.8 ms`
- TRT Flow prefetch:
  - `114.7 ms`

也就是说，在当前 dummy Flow 条件下：

> **单看 prefetch 这一段，TRT 已经拿到了大约 `2.2x` 的量级收益。**

这也让下一步的优先级更清楚了：

- `prefetch decoder` 已经足够值得继续
- 下一步最有价值的就是开始做 `postfix step`

### 19. Flow postfix-step TRT 多层扩展

在 Flow prefetch 站住之后，又继续把 `postfix step` 单独拆出来做了 TRT 最小验证。

当前这版对应的是：

- `step_with_kvcache()` 里的 decoder 主体
- 输入是：
  - `postfix_inputs_embeds_t1`
  - `postfix_cos_mrope / postfix_sin_mrope`
  - `postfix_attention_mask_additive_4d`
  - `prefix_past_key/value_*`
- MLP 路径按当前 dummy 条件专门化成：
  - postfix 全部走 `expert1`

#### 1 层（bfloat16）

- build time:
  - `30.6s`
- `TRT vs torch_ref`:
  - `cosine = 0.99997747`
  - `mean_abs = 1.81e-02`
  - `max_abs = 3.12e-01`
- bench:
  - `0.511 ms`

#### 4 层（bfloat16）

- build time:
  - `38.2s`
- `TRT vs torch_ref`:
  - `cosine = 0.99993670`
  - `mean_abs = 2.49e-02`
  - `max_abs = 2.97e-01`
- bench:
  - `1.814 ms`

#### 36 层（bfloat16）

- build time:
  - `45.9s`
- `torch vs exported_ref`:
  - `cosine = 0.99995577`
  - `mean_abs = 6.49e-03`
  - `max_abs = 1.71e-01`
- `TRT vs torch_ref`:
  - `cosine = 0.99995244`
  - `mean_abs = 6.75e-03`
  - `max_abs = 2.14e-01`
- `TRT vs exported_ref`:
  - `cosine = 0.99994779`
  - `mean_abs = 6.72e-03`
  - `max_abs = 1.25e-01`
- bench:
  - `16.304 ms`

当前结论：

> **Flow 的 `postfix step` 也已经在完整 36 层上跑通，并且数值对齐明显好于 prefetch。**

把它和当前 Python Flow baseline 的 ODE 主体对比看：

- Python Flow `ode` 平均:
  - `769.9 ms`
- TRT `postfix step` 单次:
  - `16.3 ms`

这两者不能直接简单等同，因为：

- Python `ode` 里还包含：
  - `ActionProcessor.step()`
  - host 侧 Euler/ODE loop
  - 每步外层张量组织
- 当前 TRT `postfix step` 只覆盖 decoder 主体

但至少已经非常清楚：

> **Flow 的两大主耗时块 `prefetch decoder` 和 `postfix decoder step` 都已经能在 TRT 上独立跑通，而且都拿到了明显快于当前 Python 基线的量级。**

所以当前 `Flow` 路线的下一步已经不是再证明某个 block 可不可行，而是：

1. 用现有 `prefetch engine`
2. 用现有 `postfix step engine`
3. 在 host 侧保留 `ActionProcessor.step()` 和 Euler loop
4. 先串出第一版两阶段 Flow TRT runner

### 20. 第一版两阶段 Flow TRT runner（36 层）

在上面的两个 block 都跑通之后，又补了第一版两阶段 Flow runner：

- [run_flow_two_stage_trtllm.py](/Users/sam/project/github/wall-x/workspace/trt_spike/run_flow_two_stage_trtllm.py)

当前这版的结构是：

- `prefetch decoder` 走 TRT
- `postfix step decoder` 走 TRT
- `ActionProcessor.step()` 保留在 host
- `action_proj_back` 保留在 host
- ODE / Euler loop 保留在 host

也就是说，这是一版：

> **decoder 主体进 TRT，动作时间步投影和积分循环仍留在 Python host 侧的 hybrid Flow runtime。**

当前 `36` 层结果（dummy Flow 条件）：

- `prefetch_ms`:
  - `110.144`
- `postfix_times_ms`:
  - `[16.516, 16.419, 16.367, 16.437]`
- `postfix_mean_ms`:
  - `16.435`
- `total_ms`:
  - `175.883`

最终动作对比：

- `predict_action cosine`:
  - `0.98512065`
- `predict_action mean_abs`:
  - `7.03e-02`
- `predict_action max_abs`:
  - `2.98e-01`

对应缓存 engine：

- `flow_prefetch_36.engine`:
  - `6.1G`
- `flow_postfix_step_36.engine`:
  - `1.5G`

当前判断：

> **Flow 的第一版两阶段 TRT 路线已经实际跑通了，而且最终动作已经和原始 Python reference 达到 `cosine ≈ 0.985` 的量级。**

把它和当前 Python Flow baseline 对比：

- Python total:
  - `1417.1 ms`
- two-stage Flow TRT total:
  - `175.9 ms`

即便考虑到当前这版 still 是 hybrid runtime：

- `ActionProcessor.step()` 还在 host
- `action_proj_back` 还在 host
- ODE loop 还在 host

也已经非常明确地说明：

> **Flow Action 这条 TRT 路线不只是子图成立，而是已经能在 dummy 条件下把完整主链跑到明显快于 Python baseline 的量级。**

下一步最自然的优化方向已经很清楚：

1. 减少 `flow_prefetch_36.engine` 的体积
2. 评估 `ActionProcessor.step()` 是否值得继续下沉
3. 评估 `action_proj_back` 是否直接并进 `postfix step` engine
4. 再考虑把 host 侧 ODE loop 进一步收进 runtime

### 21. `action_proj_back` 已并进 Flow engine

在上一版两阶段 Flow runner 里：

- `prefetch decoder` 走 TRT
- `postfix step decoder` 走 TRT
- `ActionProcessor.step()` 留在 host
- `action_proj_back` 也留在 host

这次又进一步把：

- `action_proj_back`

直接并进了：

- `flow_prefetch_36.engine`
- `flow_postfix_step_36.engine`

也就是说，现在两阶段 Flow runner 里 host 侧真正剩下的主逻辑只有：

- `ActionProcessor.step()`
- Euler / ODE loop

重新跑出的 `36` 层结果是：

- `prefetch_ms`:
  - `110.864`
- `postfix_mean_ms`:
  - `16.493`
- `total_ms`:
  - `176.837`
- `predict_action cosine`:
  - `0.98511314`
- `predict_action mean_abs`:
  - `7.03e-02`
- `predict_action max_abs`:
  - `2.99e-01`

和上一版相比，这组数据基本没有变化，说明：

> **把 `action_proj_back` 收进 engine 在架构上是对的，但它不是当前 Flow 路线里的主瓶颈。**

这也进一步收窄了下一步真正值得继续下沉的部分：

> **如果继续压 Flow，最该盯的是 `ActionProcessor.step()`，而不是 `action_proj_back`。**

### 22. `ActionProcessor.step()` 单独 TRT 化与整链影响

在确认 `action_proj_back` 不是主瓶颈之后，又继续把 `ActionProcessor.step()` 里真正有权重的部分单独拆成了一个小的 TRT engine：

- [build_action_step_trtllm.py](/Users/sam/project/github/wall-x/workspace/trt_spike/build_action_step_trtllm.py)

当前这版的拆法是：

- host 只保留无权重的 `SinusoidalPosEmb`
- engine 内部完成：
  - `concat(noisy_action, dof_mask)`
  - `w1`
  - `concat(action_embed, time_embed)`
  - `w2`
  - `SiLU`
  - `w3`

也就是说：

> **`ActionProcessor.step()` 的主要 MLP 部分已经能独立进 TRT。**

#### 单独 block 验证

`t0`：

- `TRT vs torch_ref`:
  - `cosine = 0.99999607`
  - `mean_abs = 7.11e-04`
  - `max_abs = 4.77e-03`
- `TRT vs exported_ref`:
  - `cosine = 0.99999654`
  - `mean_abs = 6.90e-04`
  - `max_abs = 4.82e-03`
- bench:
  - `0.263 ms`

`t1`：

- `TRT vs torch_ref`:
  - `cosine = 0.99999630`
  - `mean_abs = 6.47e-04`
  - `max_abs = 4.31e-03`
- `TRT vs exported_ref`:
  - `cosine = 0.99999660`
  - `mean_abs = 6.20e-04`
  - `max_abs = 4.40e-03`
- bench:
  - `0.266 ms`

这说明：

> **`ActionProcessor.step()` 这块本身非常适合做成一个小 TRT block，而且数值几乎完全对齐。**

#### 但把它作为“第三个独立 engine”挂回两阶段 Flow runner 之后

新的两阶段结果是：

- `prefetch_ms`:
  - `116.331`
- `postfix_mean_ms`:
  - `17.238`
- `total_ms`:
  - `185.283`
- `predict_action cosine`:
  - `0.98509818`
- `predict_action mean_abs`:
  - `7.03e-02`

和上一版相比：

- 质量几乎没变
- 速度反而略慢

当前结论非常明确：

> **`ActionProcessor.step()` 作为一个独立小 engine 是“局部可行”的，但“系统上不划算”。**

也就是说，这一步真正暴露出的不是数学问题，而是 runtime 组织问题：

- 小 engine 本身很快
- 但多一个 engine，就多一层 session 调度 / launch / host orchestration
- 最后端到端未必更快

所以当前更合理的方向不是：

- 把 `ActionProcessor.step()` 长期保留为第三个独立 engine

而是二选一：

1. **继续让它留在 host**
2. **以后直接把它和 `postfix step` 融成一个更大的 engine**

当前阶段更稳的判断是：

> **`ActionProcessor.step()` 已经证明“能进 TRT”，但下一步该追求的是“融合进去”，而不是“单独挂一个 engine”。**

### 23. `action_step.engine + postfix.engine + prefetch.engine` 的端到端结果

在验证完 `ActionProcessor.step()` 这个小 block 后，又把它真正挂回了两阶段 Flow runner，形成了一条新的三段式 runtime：

- `action_step.engine`
- `flow_prefetch_36.engine`
- `flow_postfix_step_36.engine`

同时：

- `action_proj_back` 仍然保留在 `prefetch` / `postfix` engine 内部
- host 侧只保留：
  - 无权重时间嵌入
  - Euler / ODE loop

新的 `36` 层端到端结果是：

- `prefetch_ms`:
  - `111.372`
- `postfix_mean_ms`:
  - `16.600`
- `total_ms`:
  - `177.773`
- `predict_action cosine`:
  - `0.98509818`
- `predict_action mean_abs`:
  - `7.03e-02`

这组结果和上一版相比几乎没有本质变化：

- 之前（`action_step` 在 host）：
  - `total_ms ≈ 176.8`
- 现在（`action_step` 作为第三个独立 engine）：
  - `total_ms ≈ 177.8`

所以最新结论更明确了：

> **`ActionProcessor.step()` 进 TRT 本身没有问题，但只要它还是“第三个独立 engine”，端到端几乎不会因此更快。**

这再次说明，下一步真正值得做的不是继续保留三个独立 engine，而是：

1. **让 `action_step` 留在 host**
2. 或者更激进地：
   **把 `action_step + postfix step` 融成一个更大的单一 engine**

当前更偏向的方向是后者，因为：

- block 级别精度已经证明成立
- 现在真正损耗的是多 engine orchestration
- 再多一个小 engine，不会自动带来端到端收益

### 24. `action_step + postfix step` 融合版验证

在确认第三个独立 `action_step.engine` 不划算之后，又继续做了更接近正确方向的一步：

- [build_flow_postfix_fused_trtllm.py](/Users/sam/project/github/wall-x/workspace/trt_spike/build_flow_postfix_fused_trtllm.py)
- [run_flow_two_stage_postfix_fused_trtllm.py](/Users/sam/project/github/wall-x/workspace/trt_spike/run_flow_two_stage_postfix_fused_trtllm.py)

也就是把：

- `action_step`
- `postfix decoder`
- `action_proj_back`

真正融合成一个更大的 `postfix fused engine`。

#### block 级别

`1` 层：

- `TRT hidden vs torch_ref`:
  - `cosine = 0.99997425`
  - `mean_abs = 1.92e-02`
- `TRT action_pred vs torch_ref`:
  - `cosine = 0.99997061`
  - `mean_abs = 1.98e-02`
- bench:
  - `0.795 ms`

`36` 层：

- `TRT hidden vs torch_ref`:
  - `cosine = 0.99995053`
  - `mean_abs = 6.87e-03`
- `TRT action_pred vs torch_ref`:
  - `cosine = 0.99997282`
  - `mean_abs = 6.28e-03`
- `TRT action_pred vs exported_ref`:
  - `cosine = 0.99997222`
  - `mean_abs = 6.25e-03`
- bench:
  - `16.483 ms`

这说明：

> **融合版 postfix engine 在数学上是成立的，而且精度很好。**

#### 端到端两阶段 runner

基于这个 fused postfix engine，新的两阶段 Flow runner 结果是：

- `prefetch_ms`:
  - `114.289`
- `postfix_mean_ms`:
  - `16.680`
- `total_ms`:
  - `181.009`
- `predict_action cosine`:
  - `0.98502570`
- `predict_action mean_abs`:
  - `7.05e-02`

和前一版两阶段 runner 对比：

- `action_step` 留 host：
  - `total_ms ≈ 176.8`
- `action_step` 作为第三个独立 engine：
  - `total_ms ≈ 177.8`
- `action_step + postfix` 融合：
  - `total_ms ≈ 181.0`

当前判断：

> **融合版在 block 级别是对的，但在当前 dummy Flow 条件下，端到端仍然没有明显优于“host action_step + TRT postfix”的版本。**

所以现在最现实的工程结论是：

1. `Flow` 这条 TRT 路线已经证明可行
2. `prefetch` 和 `postfix` 两大块都已经能独立或融合后跑通
3. 但当前端到端的主要收益已经拿到
4. 再继续往下抠，收益开始明显变成 runtime 组织和 engine 体积问题，而不再是算子数学问题

### 26. VQA 为什么会变成 5 个 engine

当前 `VQA-specialized` 路线里，完整 `5 token` runner 用到的是：

- `prefill.engine`
- `decode_past420.engine`
- `decode_past421.engine`
- `decode_past422.engine`
- `decode_past423.engine`

这不是因为 `VQA` 天生就需要这么多模型，而是因为：

> **当前这版 spike 把 decode 的 `past_len` 写死进了 engine。**

而 autoregressive decode 的特点正是：

- 每生成一个 token
- KV cache 长度就会加一

所以如果当前 engine 只能吃静态 `past_len`：

- 第一步 decode 用 `past_len=420`
- 第二步 decode 用 `past_len=421`
- 第三步 decode 用 `past_len=422`
- 第四步 decode 用 `past_len=423`

就自然会变成多份 decode engine。

这件事要特别强调两点：

1. **这不是 LLM/TRT 的正常终态**
2. **这是当前 spike 为了最快验证数学和性能关系，故意采用的最笨静态化方案**

也就是说，当前 VQA 的 `5 engine` 更像是：

> **“为了快速验证，先把每个 `past_len` 单独编一份”的工程妥协。**

而不是：

> **“TensorRT 做 LLM 就必须一 token 一 engine”。**

真正可用的 LLM TRT runtime，一般会继续往下面这些方向演化：

- 减少 engine 数量
- 用动态 shape / profile 覆盖一个 `past_len` 区间
- 用 runtime 管理 KV cache
- 用更少的 engine 承接整个 decode loop

所以当前更准确的判断是：

> **`5 engine` 是 spike 产物，不是推荐终态；它证明了 TRT 数学可行，但也把真正的下一个问题暴露得很清楚：VQA 想继续走下去，就必须进入 runtime 设计，而不是继续堆静态 engine。**

### 25. 裁掉无用输出后的融合版复测

在上一轮融合版验证之后，又进一步做了一次 runtime 级小收口：

- `flow_prefetch` engine 不再输出无用的 `hidden_out`
- `flow_postfix_fused` engine 不再输出无用的 `hidden_out`
- 只保留 runner 真正需要的：
  - `action_pred`
  - `prefix KV`

新的 `36` 层融合版结果是：

- `prefetch_ms`:
  - `116.169`
- `postfix_mean_ms`:
  - `16.763`
- `total_ms`:
  - `183.221`
- `predict_action cosine`:
  - `0.98502576`
- `predict_action mean_abs`:
  - `7.05e-02`

这和前一版融合版：

- `total_ms ≈ 181.0`

几乎没有本质区别，甚至还略慢一点。

当前更稳的结论是：

> **当前 Flow 路线的瓶颈已经不是“少传一个 hidden_out tensor”这种局部问题了。**

也就是说，到这里基本可以确认：

- `prefetch` 数学成立
- `postfix` 数学成立
- `action_step` 数学成立
- 三者独立、三者融合、以及裁掉无用输出这些工程尝试都已经走过

但端到端上，真正继续要收益，已经不太像是“再抠一个小算子”，而更像是：

- engine 体积
- engine 数量
- session / I/O 组织
- 以及是否值得继续把这条链完全写成更像专用 runtime 的东西
