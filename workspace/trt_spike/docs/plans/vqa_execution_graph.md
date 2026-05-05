# VQA-Specialized 执行图拆解

这份文档只回答一个问题：

> 如果目标是先把 `wall-x` 的 **VQA** 路线尽快跑起来，再拿到第一组性能和精度数据，那么当前最现实的分层和推进顺序是什么？

这里需要区分两个概念：

1. **最终目标**
   - 全链路都进入 `TensorRT / TRT-LLM`
   - 进不去的部分一律补 plugin

2. **第一阶段里程碑**
   - 先拿到一条可以整体跑通的 VQA 路线
   - 先拿到第一组性能和精度数据
   - 再决定哪些地方值得继续 plugin 化

也就是说：

> **终局目标不变，但第一阶段不能卡死在“必须一次性纯 TRT 全做完”。**

---

## 一、VQA-specialized 的最小执行图

先把 `wall-x` 的 VQA 路线压成一张图：

```text
文本 prompt
  + 图像
  -> tokenizer / image processor
  -> token embedding
  -> vision encoder
  -> image embedding scatter 到统一序列
  -> position_ids / multimodal rope 组织
  -> decoder (expert0-only)
  -> lm_head
  -> autoregressive decode loop
  -> 文本输出
```

如果再按“最终进 TRT 的难度”拆，可以分成三层：

### A. 很像标准块

- token embedding
- decoder self-attention
- decoder MLP（在 `expert0-only` 裁剪后）
- final norm
- lm_head

这些部分最像 TensorRT / TRT-LLM 自然擅长的东西。

### B. 多模态 glue 层

- image embedding scatter
- position_ids 组织
- multimodal rope

这些部分不一定都是 plugin，但已经明显超出“纯 LLM 现成套路”。

### C. 视觉自定义块

- `rot_pos_emb`
- `get_window_index`
- vision window/full attention 组织

这一层是当前 VQA 路线里最像“自定义图块”的部分。

---

## 二、VQA-specialized 裁剪后，decoder 其实会明显变干净

在 `wall-x` 当前实现里，VQA 路径有一个非常关键的结构性简化：

> `moe_token_types` 可以固定成全 0。

这意味着：

- 没有 action token
- `expert1` 不会被访问
- decoder 可以裁成 `expert0-only`

这个裁剪的直接收益是：

- 不必先支持 `permute / unpermute`
- 不必先支持 `asym_dual_gmm`
- 不必先支持 action token 路由相关 mask

所以 VQA-specialized 的 decoder 更接近：

```text
hidden
-> q_proj / k_proj / v_proj
-> attention
-> o_proj
-> gate_proj_0 / up_proj_0 / down_proj_0
```

也就是说，VQA-specialized 这条线的主要难点不是 MoE，而是：

- 视觉前端
- 多模态 rope
- 多模态 embedding 组织

---

## 三、为什么“先整体跑起来拿数据”和“最终纯 TRT”不是一件事

如果直接按最终目标来做：

- vision 进 TRT
- multimodal rope 进 TRT
- scatter 进 TRT
- decoder 进 TRT
- decode loop 进 TRT runtime
- 所有自定义块 plugin 化

那第一组数据会很慢才出来。

而如果先按“尽快拿整体数据”的思路来做，更现实的第一阶段是：

### 里程碑 0：整体可跑，先拿性能/精度对比

目标：

- 先把一条完整的 **VQA-specialized** 路线跑起来
- 拿到：
  - 总时延
  - token/s
  - 输出文本一致性 / logits 差异

这里不要求：

- 所有块都已经 plugin 化
- 所有 glue 层都进入 engine

这里先回答的是：

> **TensorRT 这条路线如果接进真实 VQA 链路，整体上到底值不值。**

### 里程碑 1：把真正值钱的 glue/plugin 化

在里程碑 0 拿到第一组整体数据之后，再看：

- `multimodal_rope` 值不值得优先 plugin
- 视觉侧 `rot_pos_emb / get_window_index` 值不值得 plugin
- image scatter 是否值得并入 engine

也就是说，plugin 不是不要做，而是：

> **先拿到整体数据，再决定插件化的优先级。**

---

## 四、当前最现实的阶段划分

基于现有约束，我建议把 VQA-specialized TensorRT 路线分成 3 个阶段。

### Phase A：先让 decoder 主体进 TRT

这一阶段的目标是：

- 固定 `expert0-only`
- 把 decoder + lm_head 先作为主体加速对象
- 先回答“最值钱的大头能不能先吃进去”

这一步最重要，因为它能最快给你：

- decode 主体速度
- logits 精度
- token 生成差异

### Phase B：再把多模态 glue 层往里收

主要是：

- image embedding scatter
- position_ids / rope 组织
- `multimodal_rope`

这是从“LLM TRT”走向“VLM TRT”的分水岭。

### Phase C：最后再看视觉前端是否值得进 TRT

视觉前端是：

- `patch_embed`
- `rot_pos_emb`
- `get_window_index`
- vision blocks
- `merger`

这块当然可以追求也进 TRT，但它不是第一步最该做的。

因为在当前 `wall-x` 结构里：

> **先把 decoder 主体吃进去，通常比先把 vision 全吃进去更容易给你第一组有意义的数据。**

---

## 五、当前最可能的第一组对比数据怎么拿

如果目标是“先有一组整体性能和精度数据”，那最现实的是：

### 1. 性能

记录：

- 总时延
- prefill
- decode
- token/s

### 2. 精度

至少对比：

- greedy 生成文本
- 最后 token logits 差异
- 多步 decode 后的 token id 是否一致

如果能做到，再加：

- 前几步 hidden_states cosine similarity

这样第一组数据就已经足够回答：

> TRT 这条线到底是“值得继续插件化”，还是“接进去就已经不值”。

---

## 六、当前最值得保留的判断

1. 最终目标仍然是纯 `TensorRT / TRT-LLM` 路线。
2. 但第一阶段最重要的是先拿到**整体性能和精度数据**，而不是一上来就追求所有块都 plugin 化。
3. `VQA-specialized` 最大的结构优势是 `expert0-only`，这会显著简化 decoder。
4. 第一阶段最现实的顺序是：
   - 先吃 decoder 主体
   - 再收多模态 glue
   - 最后再决定视觉前端是否值得全部进 TRT
5. plugin 不是不要做，而是应该在第一组整体数据出来之后，再按收益排序去做。
