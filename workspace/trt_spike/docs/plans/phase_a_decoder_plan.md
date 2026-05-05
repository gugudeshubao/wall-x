# Phase A：VQA-specialized Decoder TRT 路线

这份文档只回答一个问题：

> 如果第一阶段先让 `VQA-specialized` 的 **decoder 主体** 进入 TensorRT，那么它的边界、输入输出接口和对比指标应该怎么定？

这里的目标不是一步到位完成最终纯 TRT 全链路，而是：

> **先把最值钱的 decoder 主体做成一个可测、可比、可扩展的 TensorRT 版本。**

---

## 一、Phase A 的目标和非目标

### 目标

Phase A 要先回答三件事：

1. `expert0-only decoder + lm_head` 进入 TensorRT 后，**单纯 decoder 主体**到底能快多少？
2. 在不改模型语义的前提下，TensorRT 输出和当前参考实现（Python / `cpp_infer`）的 **logits 与 greedy token** 差多少？
3. 这条 decoder TRT 路线值不值得继续往下收多模态 glue 和视觉前端？

### 非目标

Phase A 不试图一次性解决：

- vision encoder 全量进 TRT
- `multimodal_rope` plugin
- 视觉侧 `rot_pos_emb / get_window_index` plugin
- 完整纯 TRT decode runtime
- `Flow Action`

这些都属于后续阶段。

---

## 二、Phase A 的结构边界

Phase A 的核心裁剪是：

> **先把 VQA 路径里的 decoder 主体单独拿出来，按 `expert0-only` 重写成标准 dense 路线。**

也就是说，Phase A 关心的子图是：

```text
inputs_embeds
-> decoder layers (attention + expert0-only MLP)
-> final norm
-> lm_head
-> logits
```

而下面这些东西暂时还留在外层：

- tokenizer
- image processor
- vision encoder
- image embedding scatter
- `position_ids / rope_deltas` 计算
- greedy decode loop 的调度壳

所以 Phase A 的语义更准确地说，是：

> **先把 wall-x 的 VQA 主体从“多模态大系统”切成一个“吃统一序列 embedding 的 decoder engine”。**

---

## 三、为什么 Phase A 要从 `inputs_embeds` 开始，而不是 `input_ids`

当前 `wall-x` 的 VQA 前向里，真正进入 decoder 之前已经发生了几件事：

1. `input_ids -> token embedding`
2. `visual(pixel_values) -> image_embeds`
3. `image_embeds` scatter 回统一序列
4. `position_ids / rope index` 组织完成

这说明对 decoder 主体来说，真正关心的起点其实不是原始 token，而是：

- `inputs_embeds`
- `position_ids`
- `attention_mask / cache info`

如果 Phase A 还强行从 `input_ids` 开始，那第一阶段就会被下面这些东西绑住：

- 视觉 encoder
- image scatter
- 多模态 glue

这会让“先验证 decoder 主体值不值”这个问题变得不纯。

所以 Phase A 的最干净入口应该是：

> **直接把 `inputs_embeds` 作为 engine 输入。**

---

## 四、Phase A 从 checkpoint 里真正要提哪些权重

如果按 `expert0-only` 路线裁剪，Phase A 只需要提下面这些权重：

### 1. decoder attention

每层：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

因为当前配置里：

- `attention_moe = false`

所以这部分本来就是标准 attention，不需要额外做 expert 裁剪。

### 2. decoder MLP

每层只提 `expert0`：

- `moe.experts.0.gate_proj`
- `moe.experts.0.up_proj`
- `moe.experts.0.down_proj`

不提：

- `expert1`
- `router`
- `permute / unpermute` 路径

### 3. norm

- 每层 pre-attn norm
- 每层 post-attn norm
- final norm

### 4. 输出头

- `lm_head`

### 5. 不在 Phase A 里处理的权重

- `self.visual` 相关权重
- `expert1`
- `ActionProcessor`
- `proprioception_proj`
- 任何和 `Flow Action` 相关的头

---

## 五、Phase A engine 的建议输入

第一版建议把 engine 输入压到最少：

### Prefill engine

- `inputs_embeds`: `[B, S, H]`
- `position_ids`: `[3, B, S]`
- `attention_mask` 或可等价表达的 causal/padding 信息

### Decode engine

- `inputs_embeds`: `[B, 1, H]`
- `position_ids`: `[3, B, 1]`
- `past_key / past_value`
- decode 所需的 cache length / mask 信息

这里最重要的不是一次性把两个 engine 都做得多优雅，而是先把接口定清楚：

> **Prefill 和 Decode 最好从一开始就分开看。**

原因很简单：

- prefill 是长序列
- decode 是 `q_len=1`
- 它们的性能瓶颈和 TensorRT tactic 选择完全不同

---

## 六、Phase A engine 的建议输出

第一阶段最值得直接输出的是：

### Prefill

- `last_hidden_state`
- 可选：`context logits`
- 可选：`present KV`

### Decode

- `last token logits`
- `present KV`

如果第一阶段还输出整段 hidden states，会增加调试便利性，但也会让接口变重。  
所以一个更现实的策略是：

- **调试版**：保留 hidden states 输出
- **benchmark 版**：只保留 logits 和 KV

---

## 七、Phase A 外层暂时负责什么

即便 Phase A 的 engine 已经吃掉了 decoder 主体，外层仍然要先做几件事：

1. tokenizer
2. image processor
3. vision encoder
4. image embedding scatter
5. `position_ids / rope index` 组织
6. greedy decode loop 外壳

这意味着 Phase A 的完整 VQA 路线更像是：

```text
host-side:
  tokenizer + image processor + vision + scatter + rope index
-> TRT decoder prefill engine
-> host-side decode loop
-> TRT decoder decode engine
-> text output
```

这还不是终局，但已经足够回答：

> **如果先只把 decoder 主体拿进 TRT，整体值不值。**

---

## 八、Phase A 最先该比什么

第一组数据不需要太贪，建议先比：

### 性能

- prefill latency
- per-step decode latency
- 20 token 总时延
- token/s

### 精度

- prefill 最后一个位置的 logits cosine / max diff
- decode 前 20 个 token 的 greedy 一致性
- 如果 greedy 不一致，记录从第几步开始分叉

也就是说，第一组数据不一定要立刻对齐到“整段文本 BLEU/ROUGE”这种层面。  
对当前目标更重要的是：

> **decoder 主体进入 TRT 后，数值漂不漂，漂得会不会立刻把 greedy decode 拉歪。**

---

## 九、当前最值得保留的判断

1. Phase A 的核心不是“完整 wall-x 一次性进 TRT”，而是“先吃最值钱的 decoder 主体”。
2. Phase A 最干净的入口是 `inputs_embeds`，不是原始 `input_ids`。
3. 当前 `attention_moe = false, mlp_moe = true`，所以 decoder attention 是标准块，MLP 才需要做 `expert0-only` 裁剪。
4. Phase A 最先需要的不是大而全 plugin，而是一个清楚的 `inputs_embeds / position_ids / KV` 接口。
5. 第一组性能和精度数据，应该围绕 `prefill / decode / logits / greedy token` 来拿。

---

## 十、下一步最直接的动作

如果继续推进，最直接的一步是：

> **把 Phase A 的 reference runner 写出来。**

也就是先在 `trt_spike/` 里做一份“只保留 VQA-specialized decoder 主体”的参考实现，哪怕最开始只是：

- 从 Python `wall-x` 模型里提取 `inputs_embeds`
- 提取 `position_ids`
- 提取 reference logits

先把接口和对齐流程跑通，再去 build TRT engine。

### 当前进度

这一步已经完成：

- `export_vqa_decoder_reference.py` 已能导出
  - `inputs_embeds`
  - `position_ids`
  - `prefill_last_logits`
  - greedy `token_ids`
- `phase_a_decoder_wrapper.py` 已实现 `expert0-only` 的 Python 参考 decoder
- `test_phase_a_decoder_wrapper.py` 已验证：
  - `prefill last logits` 与原始 `wall-x` 路径 **完全一致**

所以从 Phase A 的角度看，下一步已经不再是“证明裁剪是否正确”，而是：

> **开始把这个 wrapper 变成可 export / 可 build 的 TensorRT 目标。**

### 当前新增结论

这一步里我们已经试过：

- 直接对 `phase_a_decoder_wrapper` 做 `torch.onnx.export`
- 也试过在导出时强制切到 `manual attention`

结果都没有成功，都会在图优化阶段撞到：

- `RuntimeError: ScalarType ComplexDouble is an unexpected tensor scalar type`

同时，Orin 当前环境里：

- `torch_tensorrt` 没有安装

所以 Phase A 的下一步已经可以明确收敛为：

> **不要再继续投入 ONNX 导出，直接切到 TensorRT Python API 构图。**

### 当前进一步进展

这一步之后，`Phase A` 又进一步往前走了一步：

- `layer0_weights.safetensors` 已经导出
- `layer0_ref.safetensors` 已经导出
- `build_phase_a_layer0_trtllm.py` 已经用 `TRT-LLM` 现成层 + functional 手工拼出单层 block

并且当前单层对齐结果已经拿到：

- `cosine = 0.99994922`
- `mean abs = 4.97e-03`
- `max abs = 2.5e-01`

这说明：

> **Phase A 不再只是“可行性讨论”，而是已经有一个真正跑通并且数值基本对齐的单层 TRT-LLM block 底座。**
