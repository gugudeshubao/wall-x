# Phase A：Direct TensorRT Python API 路线

这份文档只回答一个问题：

> 既然 `ONNX` 导出已经被 `ComplexDouble` 卡住，`torch_tensorrt` 也没装，那么 Phase A 下一步应该怎么转成 **TensorRT Python API 直接构图**？

---

## 一、先下结论

当前 Phase A 的现实路线已经收敛成：

> **Python 里用 `wall-x` 做 reference，对齐 `expert0-only decoder`；真正 build engine 时，不再依赖 ONNX，而是直接用 TensorRT Python API 按模块构图。**

也就是说：

- `wall-x` 负责给我们参考值
- `phase_a_decoder_wrapper` 负责告诉我们“要实现的数学到底是什么”
- TensorRT Python API 负责把这套数学变成 engine

---

## 二、为什么现在就该转 Direct TensorRT API

当前已经确认了三件事：

1. `expert0-only` 裁剪在 Python 参考实现上是精确对齐的
2. `torch.onnx.export` 已经在图优化阶段稳定卡住
3. `torch_tensorrt` 在 Orin 上没有现成环境

所以继续抠 ONNX 的价值已经很低。  
下一步更合理的是：

> **直接把已经验证过的那条 `expert0-only decoder` 数学，用 TensorRT API 重写出来。**

---

## 三、Direct TensorRT API 的第一版目标

第一版不要贪。

先只做：

- **prefill-only**
- **decoder-only**
- **expert0-only**
- **输出 logits**

不做：

- decode loop
- KV cache
- vision encoder
- image scatter
- multimodal glue 的完整自动化

也就是说，第一版 engine 先回答：

> **给定 `inputs_embeds + position_ids + attention_mask`，TensorRT 能不能把 `expert0-only decoder + lm_head` 跑出来，并和 Python reference 对齐。**

---

## 四、Direct TensorRT API 第一版最小输入输出

### 输入

- `inputs_embeds [1, S, 2048]`
- `position_ids [3, 1, S]`
- `attention_mask [1, S]` 或导出的 causal mask

### 输出

- `logits [1, S, vocab_size]`

这是最小但完整的 Phase A prefill 验证接口。

---

## 五、第一版实现顺序

建议的顺序不是一上来就搭 36 层，而是：

1. 先搭 **单层 decoder block**
2. 先验证：
   - norm
   - q/k/v/o
   - manual attention
   - `expert0` MLP
   - residual
3. 再扩成多层
4. 最后接 `final norm + lm_head`

理由很简单：

- 一层就能看出 attention / rope / MLP 这几个关键块能不能正确复现
- 如果一层都对不齐，直接上 36 层只会更难排查

---

## 六、当前最值得保留的判断

1. Phase A 的参考数学已经有了，不再是“猜怎么裁剪”。
2. ONNX 路线当前不值得继续投入。
3. 下一步最直接的工作就是：**把 `expert0-only decoder` 按模块用 TensorRT Python API 直接搭出来。**
4. 第一版最该做的不是全链路，而是 **prefill-only decoder engine**。
