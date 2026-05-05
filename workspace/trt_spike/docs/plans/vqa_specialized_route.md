# VQA-Specialized TensorRT 路线

这份文档的目标，是把 `wall-x` 在当前约束下最现实的第一条完整 TensorRT 路线讲清楚。

这里的约束已经固定：

> **不是做 hybrid runtime，也不是把进不去的东西留在外层兜底。**  
> **能进 TensorRT / TRT-LLM 的都进去；进不去的部分，就写 plugin。**

---

## 一、先下结论

在当前阶段，最应该先做的不是 `Flow Action`，而是：

> **VQA-specialized 纯 TensorRT 路线。**

更具体地说，不是直接把完整 `wall-x` 原样塞进 TensorRT，而是先做一个：

> **只服务 VQA 场景、固定 `moe_token_types=0`、只走 `expert0` 的专用 engine。**

这是当前最有机会先完整跑通、再拿去和 `cpp_infer` 做性能/精度对比的路线。

---

## 二、为什么不是先做 TRT-LLM 版完整 wall-x

先看 Orin 上当前安装的 `TRT-LLM 0.12.0` 能力边界。

实际枚举结果说明：

- 有 `QWenForCausalLM`
- 有 `CogVLMForCausalLM`
- **没有 `Qwen2.5-VL` 对应的现成模型类**

也就是说：

> **TRT-LLM 当前并没有“现成的 Qwen2.5-VL / wall-x 多模态模型入口”可直接复用。**

这不代表不能用 TRT-LLM 的 attention / plugin 能力，但意味着：

- 想直接走“TRT-LLM 完整 wall-x”这条线，前面会多出一层模型适配成本
- 在当前阶段，这条路不会比“纯 TensorRT 定制 engine”更简单

所以第一条完整路线更现实的选择是：

> **先做 VQA-specialized 纯 TensorRT engine。**

不过这里有一个非常关键的边界条件：`TRT-LLM 0.12.0` 的 `QWenForCausalLM.forward()` 虽然参数上有：

- `hidden_states`
- `prompt_embedding_table`

但继续往下看 `QWenModel.forward()` 的实现会发现：

> **在 first pipeline rank 上，它仍然强制走 `vocab_embedding(input_ids)`，并不会把 `hidden_states` 当成 `inputs_embeds` 的替代入口。**

这意味着：

- `hidden_states` 不是一个通用的“直接喂多模态 embedding”入口
- `prompt_embedding_table` 更偏 p-tuning / soft prompt，不是 `wall-x` 这种任意 image/proprio embedding scatter 的直接替代

所以当前不能指望“先把多模态 embedding 在外层拼好，再把 decoder 原样塞进现成 TRT-LLM Qwen 类”这条路自然成立。  
如果后面还想走 TRT-LLM，仍然需要更深的定制接法；而作为第一条完整 VQA 路线，**纯 TensorRT engine** 依然更现实。

---

## 三、为什么 VQA 比 Flow Action 更容易先落地

`Flow Action` 天然不适合做第一条完整 TRT 路线，因为它比 `VQA` 多出太多静态图不舒服的东西：

- ODE / Euler 循环
- prefix KV truncate
- postfix-only repeated forward
- 每步动态替换 `<|action|>` embedding
- proprioception 动态 scatter
- timestep 条件化 action head

而 `VQA` 没有这些。

更关键的是，在 `wall-x` 的当前实现里，`VQA` 路径还有一个非常大的结构性简化空间：

> **`predict_mode == "text"` / `generate()` 路径里，`moe_token_types` 会被直接初始化成全 0。**

这意味着：

- `VQA` 场景下没有 action token 路由
- `expert1` 不会被走到
- 从功能上看，decoder 可以被专门裁成 **只保留 `expert0` 的 dense 路径**

这个判断非常关键。它意味着我们不必一开始就把：

- `permute / unpermute`
- `asym_dual_gmm`
- action token 路由

这些全部拉进第一版 VQA engine。

---

## 四、VQA-specialized 路线的核心思想

这条路线不是“把完整 wall-x 原封不动搬进去”，而是：

> **先针对 VQA 场景把它裁成一个结构等价、但执行图更干净的专用模型。**

最重要的裁剪有两条：

### 1. 固定 `moe_token_types = 0`

也就是：

- 文本 token -> 0
- image token -> 0
- 没有 action token

在这个前提下，`VQA` 路径根本不会走到 action expert。

### 2. 把每层 MoE 改写成 `expert0-only` 的 dense MLP

因为 `expert1` 根本不会被访问，所以对 `VQA-specialized` engine 来说：

- 不必保留 `permute / unpermute`
- 不必保留 `asym_dual_gmm`
- 不必保留 action route 相关 mask / positional 逻辑

每层可以直接退化为：

```text
hidden
-> gate_proj_0
-> up_proj_0
-> SiLU(gate) * up
-> down_proj_0
```

也就是说：

> **VQA-specialized 路线的关键不是“把 MoE 也完整支持进去”，而是承认在当前 VQA 场景里它本来就退化成了 expert0-only 的 dense 路径。**

这里还有一个当前配置层面非常关键的事实：

- `attention_moe = false`
- `mlp_moe = true`

也就是说，`wall-x` 当前的复杂度并不是“attention 和 MLP 都 MoE 化了”，而是：

> **attention 基本还是标准 attention，真正的 MoE 复杂度主要集中在 MLP 路径。**

这会直接影响第一版 TensorRT 路线的难度判断：

- decoder attention 这边更接近 TensorRT / TRT-LLM 天然擅长的标准块
- decoder MLP 这边才是需要做 `expert0-only` 裁剪的主战场

---

## 五、这样做之后，还剩哪些必须处理的 TensorRT 问题

在做了 `expert0-only` 裁剪以后，第一版 VQA engine 需要面对的难点会明显收敛。

### 1. 视觉侧

最可能需要单独处理的是：

- `ops.rot_pos_emb`
- `ops.get_window_index`

这两块属于视觉编码链内部的自定义逻辑。

不过这里还存在进一步简化空间：

- 如果第一版只支持固定图像分辨率 / 固定 `image_grid_thw`
- 那 `get_window_index` 和部分视觉位置组织逻辑，理论上可以先做成常量或 host-side 预计算

这意味着：

> 第一版 VQA engine 不一定一上来就要把视觉所有自定义步骤都做成复杂 plugin。

### 2. decoder attention

decoder 侧最显眼的自定义点是：

- `ops.multimodal_rope`

这块是当前最像“必须补 plugin”的地方。

### 3. 多模态输入组织

还有一类问题不一定是 plugin，但一定要明确怎么处理：

- image embedding scatter 到统一序列
- position_ids / rope index 的组织
- decode 时 KV cache 管理

这些属于“完整链路怎么接”的问题。即便不全是 plugin，也必须在第一版设计里说清楚。

### 4. 当前最可能的第一批 plugin 候选

基于当前结构，第一批最可能挡路、最值得优先准备的点大致是：

1. **decoder 侧 `multimodal_rope`**
   这是当前最像“绕不过去”的自定义算子。
2. **vision 侧 `rot_pos_emb`**
   视觉位置编码本身就是自定义逻辑。
3. **vision 侧 `get_window_index`**
   如果坚持“视觉主干也完整进入 engine”，这一块大概率也要被补进去。

而下面这些东西，第一版不一定非要立刻写成 plugin：

- `permute / unpermute`
- `asym_dual_gmm`
- action/proprio 相关动态 scatter

原因不是它们不重要，而是 **VQA-specialized 路线在第一版里本来就可以先把这些东西裁掉。**

### 5. 哪些东西有机会先用 TensorRT 原生层表达

第一版不必默认所有非标准逻辑都要写 plugin。像下面这些，更应该先确认 TensorRT 原生层能不能表达：

- token embedding
- image embedding 回填（scatter / masked replace）
- position_ids 组织
- 标准 attention / MLP / lm_head

只有当原生层表达不了，或者表达后性能明显太差，才值得把它们升级成 plugin。

---

## 六、第一版最现实的目标

如果目标是尽快拿到一条“完整可跑、可对比”的 TensorRT 路线，那第一版目标应该定得很具体：

> **单图 VQA，固定 image_grid_thw，固定一组 profile，expert0-only，纯 TensorRT engine，先把 prefill + decode 跑起来。**

这版不追求：

- 通用 wall-x 全模式兼容
- Flow Action
- 多机器人 action 路由
- 训练图复用

它只回答一个问题：

> **在裁掉 `Flow` 和 `expert1` 之后，TensorRT 能不能把 `wall-x` 的 VQA 路径完整跑起来，而且跑得比当前 `cpp_infer` 更值。**

---

## 七、当前最值得立住的判断

到这里，可以先把当前路线压成下面几条：

1. 当前第一优先级不是 `Flow Action`，而是 `VQA-specialized`。
2. 当前第一条完整路线不应该优先走 `TRT-LLM 模型封装`，而应该优先走 `纯 TensorRT engine`。
3. `VQA` 最大的结构性优势是：`moe_token_types` 可固定为全 0，因此 decoder 可以裁成 `expert0-only`。
4. 这个裁剪一旦成立，第一版 VQA engine 就不必先解决 `permute / unpermute / asym_dual_gmm`。
5. 当前最可能必须补的关键 plugin，是 decoder 侧的 `multimodal_rope`，以及视觉侧的 `rot_pos_emb / get_window_index`。

---

## 八、下一步要做什么

下一步如果继续推进，最合理的顺序应该是：

1. 先把 `wall-x VQA-specialized` 的执行图画出来  
2. 标出哪些节点是：
   - TensorRT 直接支持
   - 可以裁掉
   - 必须 plugin
3. 再决定第一批到底先补哪个 plugin

也就是说，下一步不是直接开写 plugin，而是：

> **先把 VQA-specialized 这条图裁干净，再选第一刀。**
