# Flow Action TRT 路线（第一版）

当前 `VQA-specialized` 这条 TRT 路线已经拿到了：

- 完整 `5 token` 文本前缀对齐
- `prefill ~103.8 ms`
- `decode ~41-44 ms`
- `total ~271.4 ms`

所以 `trt_spike` 的下一步不再是继续证明 VQA 子图能不能跑，而是开始进入：

> **Flow Action 在 TRT / TRT-LLM 里应该先切哪一刀。**

---

## 1. 第一版目标

第一版不直接追“完整真实机器人样本”，而是先对齐现有的 Python/C++ benchmark 条件：

- 使用 [`scripts/bench_flow_action.py`](/Users/sam/project/github/wall-x/scripts/bench_flow_action.py) 的 dummy 输入
- `seq = 488`
- `action_horizon = 32`
- `num_inference_timesteps = 5`
- attention backend 仍按当前 Python Flow 基线的 `sdpa`

这样做的原因很简单：

- 这是当前最接近 `cpp_infer` Flow benchmark 的条件
- 输入结构固定，更适合先把 TRT runtime 切开
- 不会一开始就被数据集和 serving pipeline 绑住

---

## 2. Flow 和 VQA 的本质差别

`VQA` 的 TRT 路线之所以能先跑通，是因为可以裁成：

- `expert0-only`
- `prefill decoder`
- `single-step decode`
- 外层 greedy loop

但 `Flow Action` 的主干不一样。它不是 token-by-token decode，而是：

1. 先把 `<|action|>` 位置填入 `t=0` 的 `action_embed`
2. 做一次完整 prefetch forward
3. 截断成 `prefix KV`
4. 只对 postfix 子序列反复 forward
5. 每一步都用 `ActionProcessor.step()` 重新生成 action embedding
6. 用 decoder 输出回归 `v_t`
7. 在外层做 Euler / ODE 积分

所以 `Flow Action` 的核心不是 `lm_head`，而是：

- `ActionProcessor.step()`
- `prefix KV trim`
- `postfix attention mask`
- `mixed expert0/expert1 MLP`
- `action_proj_back`
- ODE loop

---

## 3. 第一版最现实的切分

第一版不要一上来就做完整 Flow TRT engine，而是分成三段：

### 3.1 段 A：导出 Python 参考底座

目标：

- 固定 dummy 输入
- 固定随机种子
- 把 `generate_flow_action()` 中间真正值钱的张量都导出来

对应脚本：

- [export_flow_dummy_reference.py](/Users/sam/project/github/wall-x/workspace/trt_spike/export_flow_dummy_reference.py)

第一版参考里至少要有：

- `inputs_embeds_after_scatter`
- `inputs_embeds_t0`
- `position_ids`
- `start_indices / end_indices`
- `noise`
- `times`
- `action_embed_t0`
- `prefetch_hidden_states`
- `v_t0`
- `prefix_length`
- `prefix_past_key_i / prefix_past_value_i`
- `postfix_inputs_embeds`
- `postfix_position_ids`
- `postfix_attention_mask_3d`
- `postfix_moe_token_types`
- `action_embed_t1`
- `postfix_hidden_states_t1`
- `v_t1`
- `predict_action`

这个阶段的目标不是提速，而是建立：

> **Flow TRT 路线的标准答案。**

### 3.2 段 B：先做 prefetch decoder

这一步和当前 VQA 路线最接近：

- 输入是完整序列
- 输出是 `prefetch_hidden_states + prefix KV`

但和 VQA 的区别是：

- 不能再走 `expert0-only`
- 需要处理 `moe_token_types` 的 mixed expert 路径
- 还需要保留 `adarms_cond`

### 3.3 段 C：再做 postfix step engine

这一步才是 Flow 真正难的地方。

输入是：

- trimmed `prefix KV`
- `postfix_inputs_embeds`
- `postfix_position_ids`
- `postfix_attention_mask_3d`
- `postfix_moe_token_types`
- 每一步新的 `action_embed_t`

输出是：

- postfix hidden states
- action token hidden states
- 回归得到 `v_t`

如果这一步能和 `v_t1` 对齐，就说明：

> **Flow 的 decoder 主体已经能开始进 TRT 了。**

---

## 4. 第一版为什么先不追“完整 ODE 全进 TRT”

因为完整 ODE 一上来就会把问题混在一起：

- decoder 数值对不对
- `ActionProcessor.step()` 要不要进 TRT
- ODE loop 放 host 还是放 graph
- `prefix KV` 生命周期怎么管
- postfix 输入每步怎么改写

第一版更稳的策略是：

- 先让 ODE loop 留在 host
- 先证明：
  - prefetch decoder 可进 TRT
  - postfix decoder step 可进 TRT
  - `v_t` 能对齐

等这三件事成立，再决定：

- `ActionProcessor.step()` 是进 TRT、进 plugin，还是先留 host
- ODE loop 是否值得继续 graph 化

---

## 5. 当前判断

所以当前最合理的推进顺序是：

1. `VQA` 路线继续作为完整文本基线保留
2. `Flow` 先导出 dummy reference
3. 再做 `prefetch decoder`
4. 再做 `postfix step`
5. 最后再决定是否做完整 Flow ODE runner

这条路的好处是：

- 不会一开始就掉进“全链路都要进 TRT”的大坑
- 还能保证每一步都有可对齐的 reference
- 和现在的 `cpp_infer` benchmark 条件最接近
