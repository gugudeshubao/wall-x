# Wall-X 的 MoE 到底加在了哪里：从 Decoder Layer 到 Token 路由

> 很多人一听到 MoE，就会自动把它想成“把 FFN 换成专家 MLP”。Wall-X 的做法比这个更深入一点。

**TL;DR**
- Wall-X 的 MoE 不在视觉分支，而在 decoder 里。
- 它不是只改 MLP，而是把 decoder layer 整体做成了可路由版本。
- `moe_token_types` 是核心路由信号，当前常见场景下基本就是“普通 token”和“action token”两类。
- `start_indices / end_indices` 表示 expert 重排后连续 token 段的边界。
- `attention_moe` 改的是 expert-specific 的投影，不是另起一套 attention 数学。

## 一、先把结论说清楚：MoE 不在视觉分支

Wall-X 的视觉主干还是上一篇讲的那套东西：

- `patch_embed`
- `rot_pos_emb`
- `get_window_index`
- `vision blocks`
- `merger`

真正被 MoE 化的不是视觉编码器，而是 decoder。

顶层模型里最关键的一句其实很朴素：

```python
self.model = Qwen2_5_VLMoEModel(...)
```

这句话已经说明了大方向：

- 视觉分支保留
- decoder 被替换成 MoE 版本

所以你如果后面分析 Wall-X 的 MoE 行为，重点应该一直放在统一序列进入 decoder 之后发生了什么。

## 二、MoE 不是挂在 decoder 后面，而是长在 layer 里面

Wall-X 对基础 decoder 的改法，不是“外面再接个专家模块”，而是直接把每一层的定义换掉了：

- 原版：`Qwen2_5_VLDecoderLayer`
- Wall-X：`Qwen2_5_VLDecoderLayer_with_MoE`

这意味着什么？

意味着 MoE 不是在 decoder 之外补一层，而是直接进入 layer 内部计算流程。

一个带 MoE 的 decoder layer，大致执行顺序是：

```text
输入
-> 第一次 norm MoE
-> attention（可能是 joint attention）
-> gated residual
-> 第二次 norm MoE
-> MLP MoE
-> gated residual
-> 输出
```

这说明 Wall-X 的 MoE 不只是“专家 MLP”，而是围绕 token 路由把 norm、attention、MLP 一起改了。

## 三、`moe_token_types` 才是这套路由的核心

Wall-X 的 router 其实没有大家想象得那么玄。

它的路由核心信号是：

- `moe_token_types`

而这个东西在训练和推理里，通常都来自同一个简单规则：

```python
inputs.input_ids == action_token_id
```

也就是说，在当前最常见的两 expert 场景下：

- 普通 token -> 0
- action token -> 1

再往下看 router：

```python
experts_indices = token_types % num_experts
```

你会发现它并不是那种经典 top-k learned router，而更像显式 token 语义路由。

这其实很符合 Wall-X 的任务特点。因为这里 token 的语义分工本来就比较明确：

- 普通上下文 token
- 动作 token

既然语义已经很清楚，就没有必要再硬做一个复杂到难解释的在线路由器。

## 四、`start_indices / end_indices` 到底是什么

第一次看到这两个名字时，很多人会觉得抽象。  
其实它们的含义很简单：

> 如果把 token 按 expert 分组重排，那么每个 expert 在这条重排后序列里会占一个连续片段，`start_indices / end_indices` 就是在标这个片段的左右边界。

顶层模型里如果这两个值没有传进来，会根据 `moe_token_types` 现算每个 expert 的 token 数量，再做 cumulative sum。

所以这两个变量并不神秘，它们只是 MoE 重排后的“段描述符”。

一旦理解了这一点，后面很多实现细节都会自然很多：

- 为什么 `mot_opt` 里要先 `permute`
- 为什么 MoE block 里总在按 expert 分段处理
- 为什么最后还要 `unpermute`

## 五、`mot_opt=True` 时，Wall-X 会先把 token 按 expert 排好

这是 Wall-X MoE 实现里最工程化、也最实用的一步。

如果不开 `mot_opt`，模型仍然可以在原序列布局里用 mask 去挑 token。  
但如果开了 `mot_opt`，它会先把 token 按 expert 分组重排：

```python
hidden_states = permute(hidden_states, moe_token_types)
```

这么做的好处非常直接：

- 同一个 expert 的 token 变成连续段
- norm、MLP、输出投影都更容易按连续内存块处理
- `start_indices / end_indices` 的作用也会变得非常直观

最后再用 `unpermute(...)` 把顺序还原。

所以 `mot_opt` 本质上是一种执行优化，而不是建模逻辑变化。

## 六、Wall-X 的 MoE 不只改 MLP

这是最值得反复强调的一点。

很多人默认 MoE = 专家 MLP。  
在 Wall-X 里，这个理解只对了一部分。

### 1. `norm_moe`

如果开了 `norm_moe`，模型会给每个 expert 准备独立的 norm，而不是所有 token 共用一套 RMSNorm。

这说明 Wall-X 允许不同语义类型的 token，在进入 attention / MLP 之前就走不同的归一化路径。

### 2. `mlp_moe`

这一部分最像大家熟悉的 MoE：多个 expert MLP，各自处理自己的 token 段。

也正是这里，原始 dense MLP 和新增动作专家的区别最明显。

### 3. `attention_moe`

这是 Wall-X 比较特别的地方。

如果开了 `attention_moe`，attention 不再共用一套 QKV/O 投影，而是每个 expert 自己维护：

- `q_proj_experts`
- `k_proj_experts`
- `v_proj_experts`
- `o_proj_experts`

注意，这里变的是投影生成方式，不是 attention 数学本身。  
后面的 attention 仍然可以落到：

- `scaled_dot_product_attention`
- 或 `flash_attn`

所以它不是 MLA，也不是完全自定义的新 attention，而是 expert-aware 的 QKV/O。

## 七、mask 和 position 也跟着 token 语义一起变了

一旦你把专家路由真正放进统一序列里，MoE 就不可能只影响线性层。

Wall-X 这里还改了两类东西：

### 1. 某些 action token 区域不再严格纯因果

在某些设置下，type-1 token 区域会被从纯 causal 约束里放开，让它们局部双向可见。

这说明动作 token 不是简单塞进一个标准 next-token prediction 框架里，而是为了动作建模需求调整过可见性规则。

### 2. 位置编号也会重对齐

如果序列里混了 AR token 和 flow token，而且它们语义上属于不同阶段，那么原始的简单顺序位置就不一定最合适。

所以 Wall-X 还会根据 token 类型去修正一部分 position ids。

这一点非常重要，因为它说明 Wall-X 的复杂度不只是“多了几个专家”，而是：

> 专家路由、统一序列和动作语义被绑在了一起。

## 八、这篇文章真正要立住的判断

如果把这篇压成一句话，那就是：

> Wall-X 的 MoE 不是只改 FFN，而是直接长进了 decoder layer：token 先按 `moe_token_types` 被分到不同 expert，随后 norm、attention、MLP、mask 和 position 都会围绕这套路由一起变化。

下一篇我会继续沿着统一序列这条线往下讲：Wall-X 到底是怎么生成动作的。

