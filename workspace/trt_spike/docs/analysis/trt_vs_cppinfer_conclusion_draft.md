# TRT vs `cpp_infer` 结论稿

基于 **2026-04-29** 当前 Orin 实测，这次 `TensorRT / TRT-LLM` spike 可以先收成一个比较清楚的结论：

> **VQA 上，TRT 路线已经证明“能跑通”，但还没有在端到端上明显赢过 `cpp_infer`；Flow Action 上，TRT 路线已经证明“能跑通，而且已经赢了”。**

如果只看最重要的几个数字：

- `cpp_infer` 当前最佳 VQA（20 tokens）：`851.2 ms`
- TRT VQA 当前完整跑通的是 `5 token`，按现有结果外推到 `20 token` 约 `899.9 ms`
- `cpp_infer` 当前最佳 Flow Action：`290.7 ms`
- TRT 当前最稳的两阶段 Flow：`176.8 ms`

这意味着两条路线已经出现了很明确的分化。

## 一句话判断

- **VQA**：子图成立、整链能跑，但 TRT 还没有形成比 `cpp_infer` 更硬的端到端优势。
- **Flow Action**：TRT 已经在端到端上形成明确优势，当前是最值得继续投的主战场。

## 正式结论

这轮 spike 之后，关于 `TensorRT` 这条路线，至少有三件事已经可以说得比较硬。

第一，`TensorRT / TRT-LLM` 在 `wall-x` 上不是“理论上可能有用”，而是已经被实际验证过。`VQA-specialized` 路线已经完整跑通，`prefill decoder` 和 `single-step decode` 都能和原始 Python reference 对齐；`Flow Action` 这边，`prefetch decoder`、`postfix step`，以及第一版两阶段 hybrid runtime 也都已经跑通。这说明这条路不存在“图根本拼不起来”或者“数值根本对不齐”的基础性障碍。

第二，`VQA` 和 `Flow Action` 对 `TensorRT` 的响应完全不同。`VQA` 目前的问题不是单层 block 慢，而是它的 runtime 组织成本太重。当前这条线为了支持 decode，不得不拆成多个超大 engine，单单 `vqa36` 目录就接近 `29G`。这使得 `TensorRT` 在 VQA 上虽然数学成立，但端到端收益被 engine 体积、engine 数量、session 生命周期和 I/O 成本吃掉了。按当前 `5 token` 结果外推到 `20 token`，TRT VQA 约 `900 ms`，还没有压过 `cpp_infer` 当前的 `851.2 ms`。所以对 VQA 来说，TRT 现在的状态更准确地说是“可行，但还不够值”。

第三，`Flow Action` 已经给出了完全不同的答案。当前 dummy 条件下，Python baseline 约 `1417.1 ms`，`cpp_infer` 当前最佳约 `290.7 ms`，而 TRT 两阶段 Flow 已经稳定跑到 `176.8 ms` 左右，最终动作对齐也能维持在 `cosine ≈ 0.985` 的量级。也就是说，TRT 在 Flow Action 上不是只在某个 attention block 或某个单层 MLP 上跑得快，而是已经在完整主链上把时延压到了明显优于 `cpp_infer` 的区间。这件事本身就足以说明，TRT 这条路线在 Flow 上已经具备继续投入的价值。

但第四件事同样重要：这轮实验越往后做，越能看清楚真正的瓶颈已经不在“再抠一个算子”。无论是把 `action_proj_back` 单独收进去，还是把 `ActionProcessor.step()` 单独做成第三个小 engine，甚至把 `action_step + postfix step` 融成一个更大的 fused postfix engine，最终端到端收益都没有继续明显拉开。这说明到这个阶段，决定上限的已经不是单个 kernel 的数学快慢，而是更上层的 runtime 组织：engine 体积、engine 数量、session / I/O 调度，以及你愿不愿意把整条链真正写成更像专用 runtime 的东西。

所以如果要把当前结论压成一句最短的话，就是：

> **TRT 在 `wall-x` 上已经证明可行；在 VQA 上，它现在更像“有潜力但还不划算”；在 Flow Action 上，它已经是“端到端上明确赢过 `cpp_infer` 的路线”。**

## 写作提纲

### 标题候选

- `TRT 在 wall-x 上到底值不值得继续做：VQA 还没赢，Flow Action 已经赢了`
- `为什么 TRT 在 VQA 上还没形成优势，但在 Flow Action 上已经把 cpp_infer 压下去了`
- `从 VQA 到 Flow Action：TensorRT 在 wall-x 上真正证明了什么`

### 1. 背景

- 前几篇已经把 `wall-x` 在 Orin 上的主线优化做到了比较深的位置。
- `cpp_infer` 已经有很强的基线：
  - Flow Action `290.7 ms`
  - VQA（20 tokens）`851.2 ms`
- 所以这轮 spike 不是为了“证明 TensorRT 很强”，而是为了回答：
  - **它相对当前 `cpp_infer` 主线，到底还有没有继续做的价值。**

### 2. 实验方法

- 所有 TRT 实验都放在 `workspace/trt_spike/`，不碰 `cpp_infer/`
- `VQA` 先做成 `VQA-specialized`
- `Flow` 先按 dummy 条件拆成：
  - `prefetch`
  - `postfix step`
  - host-side Euler loop

### 3. VQA 结果

- `prefill decoder` 跑通
- `single-step decode` 跑通
- `5 token` 完整 runner 跑通
- 但按现有结果外推到 `20 token` 约 `900 ms`
- 对比 `cpp_infer 851.2 ms`
- 当前还没有形成端到端优势

这里的重点不是“TRT 不行”，而是：

- VQA 被多个超大 engine 卡住了
- 问题已经转到 runtime 组织，而不是子图数学

### 4. Flow Action 结果

- `prefetch decoder` TRT 已跑通
- `postfix step` TRT 已跑通
- 第一版两阶段 runner 已跑通
- 当前最稳结果约 `176.8 ms`
- 对比 `cpp_infer 290.7 ms`

这里要把结论说硬：

- **Flow Action 是 TRT 在 `wall-x` 上真正已经证明有价值的路线。**

### 5. 为什么会分化

- VQA 更吃：
  - decode engine 数量
  - engine 体积
  - session / I/O 成本
- Flow 更适合当前专门化路径：
  - prefix/postfix 划分清楚
  - `moe_token_types` 在 dummy 条件下是连续块
  - 更容易做专用 engine

### 6. 后续判断

- VQA：
  - 可以继续做
  - 但优先级不高
  - 真正要突破，必须动 runtime 组织
- Flow：
  - 明确值得继续
  - 但继续收益也已经越来越依赖 runtime 设计，而不是单个小算子

### 7. 收束句

- `TensorRT` 在 `wall-x` 上已经不是“能不能做”的问题
- 真正的问题已经变成：
  - **在哪些任务上它真的值**
  - **以及为了把价值兑现出来，需要付出多大 runtime 复杂度**

## 适合直接摘出去的短结论

### 版本 A

> 这一轮 spike 之后，我对 TensorRT 在 wall-x 上的判断已经很明确了：VQA 这条线现在是“能跑通，但还没赢”；Flow Action 这条线则是“已经跑通，而且已经赢了”。问题不再是 TRT 能不能拼出图，而是端到端收益到底值不值得你继续为 runtime 组织买单。

### 版本 B

> 如果只看结果，TRT 在 wall-x 上已经给出了分化非常明显的答案。VQA 还没有稳定压过 `cpp_infer`，因为它更容易被多个超大 engine 的组织成本拖住；Flow Action 则已经在端到端上跑到了明显优于 `cpp_infer` 的区间。这说明 TRT 不是普适银弹，但在对的任务结构上，价值已经非常实在。

### 版本 C

> 现在再问“TensorRT 值不值得继续做”，答案不能一刀切。对 VQA，当前证据是“可行，但还不够值”；对 Flow Action，当前证据已经是“值得，而且已经证明能赢”。下一步真正决定上限的，也不再是某个 kernel，而是整套 runtime 怎么组织。 
