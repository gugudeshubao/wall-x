# wall-x 量化讨论

这份文档只整理当前对 `wall-x` 量化问题的讨论结论，不改写系列文章，也不追求正式发文口吻。目标是把“哪些部分适合量化、为什么、风险主要来自哪里”先压实，后面如果要写成文章，再基于这里展开。

---

## 一句话判断

对 `wall-x` 来说，量化的第一价值是容量，第二价值才是速度。  
而从精度角度看，最重要的分层判断是：

> `MoE expert` 可以量化到 `INT8`，而且在当前项目里这是相对安全、相对值得做的一块；但 `Action Head` 不应该轻易量化，因为它直接进入 `ODE / Euler` 的动作更新链。

---

## 一、当前项目里 MoE 到底能不能上 INT8

答案是：

> **能，但不是把原来的 `asym_dual_gmm` 直接改成 INT8。**

当前项目已经走通的是：

- `MoE expert projection` 可以上 `INT8`
- 原来的自定义 bf16 双专家 GEMM `asym_dual_gmm` 本身不支持 `INT8`
- 所以运行时会切成双路径：
  - bf16 checkpoint：继续走 `dual_gemm`
  - INT8 checkpoint：保留 `permute / unpermute / route`，但 expert 内部改走 per-expert `LinearOp`

也就是说，当前量化的是：

- `gate_proj`
- `up_proj`
- `down_proj`

而不是把 `MoE` 这整套自定义算子直接统一替换成一个新的 INT8 kernel。

---

## 二、为什么 MoE expert 比 Action Head 更适合量化

这个判断的关键，不是“MoE 更高级”，而是误差路径长短不同。

### 1. MoE expert 输出的是中间隐状态

MoE expert 的 `gate/up/down projection` 输出，后面还会继续经过：

- 残差
- 后续 decoder 层
- 主干上下文建模

所以它的量化误差还有被后续主干“消化”的空间。

### 2. Action Head 直接进入动作更新链

`Action Head` 不一样。它直接参与：

- `noisy_action` 的 embedding 构造
- `action_proj_back`
- `ODE / Euler` 迭代

这里的误差路径更短，而且会跨步累积。

所以从精度角度，应该优先保守地处理：

- `Action Head`：bf16 保持
- `MoE expert`：INT8 可做

---

## 三、MoE expert INT8 的主要精度风险到底来自哪里

如果只从当前结构看，最值得先立住的判断是：

> **MoE INT8 的主要精度风险更可能来自 expert 内部的 `gate / up / down` 三段线性层，尤其是 `gate_proj` 和 `down_proj`，而不是 token 排布本身。**

下面把这件事拆开说。

### 1. `permute / unpermute` 本身不是主要精度风险

`permute / unpermute` 做的是重排，不是近似计算。

只要实现没 bug：

- token 顺序会变
- 数值不会因为“重排本身”产生量化误差

所以这部分的主要风险是：

- 实现错误
- dtype 不匹配
- contiguous / index 对不上
- route 恢复错误

但它不是量化噪声的主要来源。

### 2. `gate_proj` 最值得警惕

原因有三个：

- 它后面马上接 `SiLU`
- `SiLU(gate) * up` 会把门控分支误差通过非线性和乘法继续传播
- 它更像“调制系数”而不是纯内容通道

换句话说：

- `up_proj` 算错一点，更像 feature 内容有噪声
- `gate_proj` 算错一点，可能会改变哪些通道该被放大、哪些通道该被抑制

所以如果后面真做 layerwise 精度分析，我会优先盯：

- `gate_proj` 输出本身
- `SiLU(gate) * up` 之后的中间张量漂移

### 3. `down_proj` 对端到端影响也很关键

虽然 `gate_proj` 更值得怀疑，但从端到端影响看，`down_proj` 也很关键。

原因是：

- 它负责把 intermediate 空间投回 hidden size
- 它的输出直接写回主干 residual 路径

也就是说：

- `up/gate` 更像在中间空间里产生误差
- `down_proj` 更像把误差直接写回主干

所以如果按工程优先级排：

1. 先看 `down_proj` 的端到端影响
2. 再重点盯 `gate_proj` 的中间表示漂移
3. `up_proj` 通常不是第一个爆精度问题的点

### 4. `up_proj` 往往相对稳一点

`up_proj` 更像在提供被门控的内容向量。

它当然也会受 INT8 影响，但通常比 `gate_proj` 少一层“控制路径”的敏感性。所以在没有 layerwise 数据之前，默认优先级上可以把它放在 `gate_proj` 和 `down_proj` 后面看。

---

## 四、为什么 wall-x 的 MoE 比通用 learned router MoE 更稳一点

`wall-x` 这里还有一个天然优势：

> 它的 `MoE` 路由不是那种完全靠 learned top-k router 在线决定的脆弱路径。

当前这条路里，`moe_token_types` 很大程度上是显式给定的 token type 信号，例如：

- 普通 token -> 0
- action token -> 1

这意味着：

- 量化后的主要风险不是“选错 expert”
- 而是“进了正确 expert 以后，expert 内部那三段 linear 有没有把表示拉歪”

和很多大语言模型里的 learned top-k router 相比，这种结构对量化更友好一些。

---

## 五、动作 expert 和语言 expert，谁更值得小心

虽然当前项目里两边都可以量化：

- `MoE 语言 expert`
- `MoE 动作 expert`

但如果只从精度敏感性看，我会更担心 **动作 expert** 一点。

原因不是它更难量化，而是：

- 它处理的是 action token
- action token 后面离动作监督更近
- 虽然还不是最终 `Action Head`
- 但比纯语言/视觉 token 的误差路径更短

所以一个很实用的判断是：

- `语言 expert`：更容易“看起来没事”
- `动作 expert`：更需要单独做 layerwise / taskwise 验证

---

## 六、当前最值得保留的工程判断

如果把现在的讨论压成一组可直接拿去用的结论，大概是这些：

1. `MoE expert` 可以量化到 `INT8`，当前项目里也已经这么做了。
2. 量化的不是原始 `asym_dual_gmm` 本体，而是 expert 内部的 `gate/up/down` 线性层路径。
3. `permute / unpermute` 不是主要精度风险，主要风险在 expert 内部线性层本身。
4. 三段里最值得优先警惕的是 `gate_proj`，最该验证端到端影响的是 `down_proj`。
5. `Action Head` 不适合轻易量化到 `INT8`，因为它直接进入 ODE / Euler 动作更新链。
6. `动作 expert` 比 `语言 expert` 更值得额外小心。

---

## 七、后面最值得继续追的问题

下一步如果继续深挖，最值得只盯一个问题：

> **在 `wall-x` 里，`MoE expert INT8` 的精度风险到底主要来自哪一段：`gate_proj`、`up_proj`、`down_proj`，还是它们和 `SiLU * mul` 组合后的中间激活？**

更具体一点，可以继续追：

1. 哪一层的 layerwise cosine similarity 最先明显掉下来？
2. `gate_proj` 和 `down_proj` 谁对最终动作成功率更敏感？
3. `动作 expert` 和 `语言 expert` 的敏感度差异到底有多大？
4. 只量化 `up/down`、保留 `gate` 为 bf16，是否会是一个更稳的折中点？
