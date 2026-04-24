# Wall-X 模型结构梳理

本文档从实际调用顺序出发，先说明推理时数据和模型是如何被装配起来的，再分析 `Wall-X` 在 `Qwen2.5-VL` 基础上增加了哪些结构。

文档主要基于以下代码路径：

- `scripts/infer_robochallenge.py`
- `wall_x/infer/infer_config.py`
- `wall_x/model/model_utils.py`
- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py`
- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py`
- `wall_x/model/action_head.py`
- `csrc/ops.cu`

## 1. 从调用顺序看整体推理链

如果按实际推理路径来看，入口基本是：

```text
WallxModelWrapper.__init__()
  -> get_model_and_processor()
  -> _register_normalizers()

WallxModelWrapper.predict_action_rtc()
  -> preprocess()
  -> _construct_input()
  -> generate_ar_action() 或 generate_flow_action()
  -> 模型内部 AR 或 flow 动作生成逻辑
```

对应代码位置：

- `WallxModelWrapper` 定义在 `scripts/infer_robochallenge.py:372`
- 模型与 processor 加载在 `scripts/infer_robochallenge.py:379`
- 单次推理入口 `predict_action_rtc()` 在 `scripts/infer_robochallenge.py:970`

这条链可以先理解成两大阶段：

1. 外层脚本把运行时输入整理成模型能接受的 batch。
2. 模型把图像、文本、状态、动作 token 混合进一个统一序列，再走视觉编码器和文本/MoE 解码器。

## 2. 数据是如何进入模型的

### 2.1 运行时输入不是直接 dataset sample，而是现场 observation

在 `predict_action_rtc()` 里，外部传入的是：

- `state`
- `views`
- `instruction`
- `valid_action_dim`

然后先进入：

```text
predict_action_rtc()
  -> preprocess(state, views, valid_action_dim)
```

`preprocess()` 在 `scripts/infer_robochallenge.py:822`，它主要做三件事：

1. 按 `dof_config` 把机械臂状态拼成统一的 `agent_pos`
2. 生成 `agent_pos_mask` 和 `dof_mask`
3. 把相机输入重命名成统一视角名，组装成 `observation`

最终形成的 `observation` 至少包含：

- 各视角图像，例如 `face_view`、`left_wrist_view`
- `agent_pos`
- `agent_pos_mask`
- `dof_mask`

### 2.2 输入构造分成文本、图像和附加控制信号三部分

随后 `predict_action_rtc()` 调用 `_construct_input()`，位置在 `scripts/infer_robochallenge.py:731`。

这一步会把 observation 变成真正送进模型的 batch：

```text
observation
  -> proprioception 归一化
  -> prompt 文本构造
  -> 图像 resize + image_processor
  -> tokenizer
  -> BatchFeature
  -> 增加 moe_token_types / dataset_names / masks
```

具体展开如下。

#### a. 状态归一化

`agent_pos` 会先用 `normalizer_propri` 做归一化，代码在：

- `scripts/infer_robochallenge.py:748`
- `wall_x/model/action_head.py:112`

这里的归一化逻辑是把每个机器人数据集对应的连续量压到 `[-1, 1]` 区间。

#### b. 文本 prompt 构造

AR 模式走 `get_text_ar()`，位置在 `scripts/infer_robochallenge.py:581`。  
Flow 模式走 `get_text_flow()`，位置在 `scripts/infer_robochallenge.py:629`。

这一步会把以下内容拼进一段多模态 prompt：

- system prompt
- instruction
- 图像占位符 `<|vision_start|><|image_pad|><|vision_end|>`
- proprioception 信息
- assistant 前缀

可以把它理解成：Wall-X 仍然沿用了 Qwen-VL 风格的“对话式多模态输入模板”，只是在用户输入中额外塞入了状态和动作相关 token。

#### c. 图像预处理

图像在 `resize_images()` 中处理，位置在 `scripts/infer_robochallenge.py:685`。

这里会做：

- PIL resize
- `smart_resize(...)`
- 调用 `self.processor.image_processor(...)`

最终得到：

- `pixel_values`
- `image_grid_thw`

其中 `image_grid_thw` 很重要，后面视觉分支的 RoPE 和窗口划分都会用到它。

#### d. tokenizer 和 batch 组装

文本 tokenize 发生在：

- `scripts/infer_robochallenge.py:805`

图像与文本最终在这里合并：

- `scripts/infer_robochallenge.py:808`

```python
inputs = BatchFeature(data={**text_inputs, **image_inputs})
```

随后脚本又额外添加了几个 Wall-X 特有字段：

- `moe_token_types`
- `dataset_names`
- `dof_mask`
- `agent_pos_mask`
- `proprioception`

其中：

- `moe_token_types` 在 `scripts/infer_robochallenge.py:810-812`
- 它的初始定义是：哪些 token 属于 `<|action|>` 段

这一步非常关键，因为后面的 MoE 路由就是靠这个字段来区分 token 类别的。

## 3. 模型是如何加载的

### 3.1 脚本路径下的加载顺序

`WallxModelWrapper.get_model_and_processor()` 是推理脚本里的主加载流程，位置在 `scripts/infer_robochallenge.py:379`。

实际顺序是：

```text
读取 config.yml
  -> 加载 processor
  -> 加载 action tokenizer
  -> 注册 normalizer
  -> 选择 ModelClass / ConfigClass
  -> ConfigClass.from_pretrained(...)
  -> update_model_config(...)
  -> 指定 attention backend
  -> 实例化 Qwen2_5_VLMoEForAction
  -> 加载 model.safetensors
  -> 设置 normalizer
  -> to(cuda) / bf16
```

对应关键位置：

- 读取 `config.yml`：`scripts/infer_robochallenge.py:380-386`
- 加载 `processor`：`scripts/infer_robochallenge.py:399-403`
- 加载 action tokenizer：`scripts/infer_robochallenge.py:411-430`
- 注册 normalizer：`scripts/infer_robochallenge.py:443` 和 `scripts/infer_robochallenge.py:515`
- 选择模型类：`scripts/infer_robochallenge.py:447-455`
- 读取配置：`scripts/infer_robochallenge.py:457`
- 更新配置：`scripts/infer_robochallenge.py:458`
- 设置 attention backend：`scripts/infer_robochallenge.py:465-466`
- 实例化模型：`scripts/infer_robochallenge.py:477-483`
- 加载权重：`scripts/infer_robochallenge.py:485-494`

### 3.2 通用推理配置路径

除了脚本路径，仓库里还有一个更通用的配置入口 `InferConfig`，定义在 `wall_x/infer/infer_config.py:390`。

它把推理配置拆成两部分：

- `model_config`：由 `_load_model_config()` 加载，位置在 `wall_x/infer/infer_config.py:531`
- `data_config`：由 `_load_data_config()` 加载，位置在 `wall_x/infer/infer_config.py:581`

其中 `X2RDataConfig` 定义在 `wall_x/infer/infer_config.py:12`，它负责整理：

- 图像分辨率与采样
- camera mapping
- action horizon
- padding 策略
- instruction 相关配置

### 3.3 配置注入不是只靠 config.json

基础的 `Qwen2_5_VLConfig` 并不包含所有 Wall-X 需要的动作字段。  
很多字段是通过 `update_model_config()` 动态补进去的，位置在 `wall_x/model/model_utils.py:8`。

这里至少会补充：

- `dof_config`
- `agent_pos_config`
- `use_state_string_representation`
- `flow_loss_weight`
- `action_horizon_flow`

也就是说，Wall-X 的完整模型配置其实是：

```text
基础 Qwen2.5-VL 配置
  + 训练/推理 yaml 中的机器人动作配置
  + 脚本里强制指定的 attention backend
```

## 4. 顶层模型结构

Wall-X 的核心模型类是：

- `Qwen2_5_VLMoEForAction`，定义在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:773`

它的构造函数里最关键的几行在：

- `self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(...)`
- `self.model = Qwen2_5_VLMoEModel(...)`
- `self.lm_head = nn.Linear(...)`
- `self.action_preprocessor = ActionProcessor(config)`

对应位置：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:952-975`

所以从结构上可以直接分成四块：

1. 视觉编码器 `self.visual`
2. 文本/MoE 解码器 `self.model`
3. 语言输出头 `lm_head`
4. 动作预处理与 flow action 头 `action_preprocessor`

## 5. 视觉主干

### 5.1 视觉主干是独立的一套 Vision Transformer

视觉编码器类是：

- `Qwen2_5_VisionTransformerPretrainedModel`
- 定义在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:487`

它内部包含：

- `patch_embed`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:499`
- `blocks`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:509`
- `merger`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:515`

这说明视觉部分不是“顺手做个 embedding”，而是一整套独立的视觉 Transformer 编码器。

### 5.2 视觉 forward 的核心步骤

视觉分支 forward 在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:598`。

逻辑可以压成：

```text
pixel_values
  -> patch_embed
  -> rot_pos_emb
  -> get_window_index
  -> 多层 VisionBlock
  -> merger
  -> visual embeddings
```

关键代码：

- `patch_embed`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:611`
- `ops.rot_pos_emb(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:612`
- `ops.get_window_index(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:616`
- vision block 迭代：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:654-675`
- `merger`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:677`

### 5.3 视觉 attention 后端

视觉 attention 类映射在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:407`

推理脚本里显式设置了：

- `model_config.vision_config._attn_implementation = "flash_attention_2"`
- 位置在 `scripts/infer_robochallenge.py:466`

因此当前推理时，视觉主干默认走的是 `FlashAttention2` 路线。

## 6. 文本主干

### 6.1 文本主干本质上是 Qwen decoder 栈

基础文本模型类是：

- `Qwen2_5_VLModel`，定义在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:1317`

其核心成员：

- `embed_tokens`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:1323`
- `layers`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:1326`
- `norm`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:1333`

这就是标准 decoder-only LLM 主干。

### 6.2 文本 attention 不是 MLA，而是标准 QKV + GQA

基础 attention 类在：

- `Qwen2_5_VLAttention`
- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:839`

从实现上看它是：

- `q_proj`
- `k_proj`
- `v_proj`
- `repeat_kv(...)`

关键位置：

- `q_proj/k_proj/v_proj`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:870-878`
- `repeat_kv`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:923-925`

配置里也明确写了 `num_key_value_heads` 用于 GQA/MQA/MHA，而不是 MLA：

- `wall_x/model/qwen2_5_based/configuration_qwen2_5_vl.py:183-184`
- 说明文字在 `wall_x/model/qwen2_5_based/configuration_qwen2_5_vl.py:72-77`

因此，这个仓库的文本主干不是 MLA。

### 6.3 文本 attention 后端

decoder 层的 attention 映射在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:1211-1215`

推理脚本里显式设置了：

- `model_config._attn_implementation = "sdpa"`
- 位置在 `scripts/infer_robochallenge.py:465`

所以当前推理时，文本主干默认走：

- `torch.nn.functional.scaled_dot_product_attention`
- 具体调用在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:1194`

## 7. 视觉与文本是如何汇合的

顶层 `Qwen2_5_VLForConditionalGeneration.forward()` 在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2040`。

它的关键逻辑是：

```text
input_ids -> embed_tokens -> inputs_embeds
pixel_values -> self.visual(...) -> image_embeds
把 image_embeds / video_embeds 替换进 inputs_embeds
inputs_embeds -> self.model(...)
hidden_states -> lm_head -> logits
```

对应代码：

- 文本 token embedding：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2107`
- 图像编码：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2110`
- 图像特征替换：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2126`
- 视频编码：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2130`
- 视频特征替换：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2146`
- 送入文本模型：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2184`
- 输出 logits：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2198`

所以“文本主干”和“视觉主干”不是口头上的比喻，而是代码里确实存在的两套模块：

- `self.visual`
- `self.model`

## 8. Wall-X 相比基础 Qwen2.5-VL 的增量

Wall-X 不是简单把 Qwen2.5-VL 拿来直接做机器人控制，它至少做了三层扩展。

### 8.1 用 `Qwen2_5_VLMoEModel` 替换了基础 decoder

Wall-X 的 decoder 不是基础 `Qwen2_5_VLModel`，而是：

- `Qwen2_5_VLMoEModel`
- 定义在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:284`

它最明显的变化是：

- 每一层都换成了 `Qwen2_5_VLDecoderLayer_with_MoE`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:310-319`

### 8.2 Decoder 层被改造成 MoE 版本

`Qwen2_5_VLDecoderLayer_with_MoE` 定义在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:82`

这层支持三种 MoE 化开关：

- `attention_moe`
- `mlp_moe`
- `norm_moe`

相关代码：

- `attention_moe` 分支：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:102-109`
- `norm_moe` 分支：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:116-145`
- `mlp_moe` 分支：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:147-157`

也就是说，Wall-X 的 MoE 不是只在 MLP 上做专家路由，而是允许：

- attention 专家化
- norm 专家化
- mlp 专家化

### 8.3 `moe_token_types` 是运行时路由的核心信号

外层脚本在 `_construct_input()` 中生成 `moe_token_types`：

- `scripts/infer_robochallenge.py:810-812`

模型前向时，如果没有传 `start_indices/end_indices`，会根据 `moe_token_types` 统计每个 expert 的 token 数量，然后现算 expert 段边界：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1336-1346`

在 `Qwen2_5_VLMoEModel.forward()` 内部，这些索引会继续用于：

- token permutation
- joint attention mask
- 各 expert 的局部处理

相关位置：

- MoE 前向要求 `moe_token_types`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:395-400`
- `mot_opt` 下调用 `ops.permute(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:459-465`
- 遍历 decoder 层时显式传入 `moe_token_types/start_indices/end_indices`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:489-520`

需要特别注意一件事：

- 当前推理脚本里构造的 `moe_token_types` 实际上是一个布尔张量，来源是 `inputs.input_ids == action_token_id`
- 位置在 `scripts/infer_robochallenge.py:811`

这意味着在当前这条推理路径里，实际最常用的是两类 token：

- `0`：普通文本/视觉/状态上下文 token
- `1`：`<|action|>` 对应的动作 token

虽然配置里允许 `num_experts > 2`，但如果上游没有构造更丰富的 `moe_token_types`，那运行时真正活跃的通常还是这两组。

## 9. 状态和动作 token 是如何注入序列的

这是 Wall-X 最容易被忽略，但实际上非常关键的一层。

在 `Qwen2_5_VLMoEForAction.forward()` 里，除了图像 embedding 之外，还会把状态和动作相关 embedding 散射进输入序列：

- `scatter_proprioception_embeddings(...)`
- `scatter_flow_action_embeddings(...)`

它们的调用位置在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1416-1422`

这意味着对 Wall-X 来说，输入序列不只是：

- 文本 token
- 图像 token

还包括：

- proprioception 对应的 token 位
- action 对应的 token 位

换句话说，Wall-X 不是“先做视觉语言，再单独接一个动作头”，而是先把机器人状态与动作占位一起混入统一 token 序列，再交给 MoE decoder 处理。

## 10. 动作头和两条预测路径

### 10.1 `ActionProcessor` 是动作侧核心模块

动作处理模块定义在：

- `ActionProcessor`
- `wall_x/model/action_head.py:536`

它负责：

- 维护动作维度和 proprio 维度
- 状态投影
- flow action 的时间嵌入
- 把隐藏特征映射回动作空间
- 挂 normalizer

`Qwen2_5_VLMoEForAction` 在初始化时直接持有：

- `self.action_preprocessor = ActionProcessor(config)`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:975`

### 10.2 AR 路径

外层脚本中：

- `predict_action_rtc()` 根据模式调用 `generate_ar_action()`，位置在 `scripts/infer_robochallenge.py:1008-1011`
- 脚本层 `generate_ar_action()` 会继续调用 `self.model.generate_ar_action(...)`，位置在 `scripts/infer_robochallenge.py:1121-1137`

在 `modeling_qwen2_5_vl_act.py` 中，能够直接看到的 AR/fast 生成主逻辑体现在 `predict(..., predict_mode="fast")` 这一支，其核心流程是：

1. 截出 `<|im_start|>assistant` 之前的 prompt
2. 调用 `self.generate(...)`
3. 得到 `predict_output_ids`
4. 再用 `batch_decode(...)` 还原成文本
5. 如果是 fast action token，就把 token 序列再 decode 成连续动作

关键位置：

- `predict_mode == "text" or "fast"`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1727`
- `self.generate(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1775`
- `batch_decode(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1790-1799`
- action token decode：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1804-1835`

这里也能看出：

- `model.generate(...)` 是模型继续生成 token
- `batch_decode(...)` 只是把 token ID 变成人能读的字符串

### 10.3 Flow / diffusion 路径

外层脚本的 flow 入口在：

- `scripts/infer_robochallenge.py:1047`

它最终调用：

- `self.model.generate_flow_action(...)`
- 位置在 `scripts/infer_robochallenge.py:1057`

模型内部的主函数在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1959`

这条路径不会走 `generate()` 产出文本，而是：

1. 先得到多模态条件特征
2. 初始化 noisy action
3. 做若干步 flow / diffusion 迭代
4. 输出连续动作

因此，Wall-X 同时支持两种动作预测范式：

- 离散 action token 自回归生成
- 连续动作 flow/diffusion 生成

## 11. 自定义 CUDA 算子在整套结构里的位置

导出入口在：

- `csrc/ops.cu:11`

当前导出的核心接口包括：

- `asym_dual_gmm`
- `permute`
- `unpermute`
- `unpermute_bwd`
- `rope`
- `rope_bwd`
- `rope_index`
- `rot_pos_emb`
- `get_window_index`

其中比较关键的分工如下。

### 11.1 视觉相关

- `rot_pos_emb`
- `get_window_index`

分别在视觉分支中被调用：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:612`
- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:616`

### 11.2 多模态位置编码相关

在 action 版模型里，3D RoPE index 的生成走了自定义 op：

- `ops.get_rope_index(...)`
- 调用点在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1692`

### 11.3 MoE 相关

当 `mot_opt` 打开时，会先对 token 按 expert 进行重排：

- `ops.permute(...)`
- 调用点在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:463`

因此，这个项目里的自定义算子主要可以分成两类：

1. 视觉 / RoPE 相关
2. MoE token 排布与高性能 kernel 相关

## 12. 最后总结

如果压成一句话，Wall-X 的结构可以概括为：

```text
运行时 observation
  -> prompt + 图像 + proprio + action token 构造
  -> 视觉编码器 self.visual
  -> 文本/MoE 解码器 self.model
  -> lm_head 或 flow action head
  -> 离散动作 / 连续动作输出
```

从工程角度看，Wall-X 不是一套完全重新设计的基础模型，而是：

- 以 `Qwen2.5-VL` 为 backbone
- 保留独立视觉编码器和文本 decoder
- 在 decoder 侧加入 MoE 路由
- 在输入序列中注入 proprioception / action token
- 同时支持 AR action token 和 flow/diffusion action
- 用少量自定义 CUDA 算子加速视觉位置编码和 MoE 路由相关热点

如果继续往下拆，下一步最值得单独展开的是两块：

1. `Qwen2_5_VLMoEModel.forward()` 里的 MoE token 路由细节
2. `ActionProcessor` 与 `generate_flow_action()` 里的 flow action 采样过程

## 13. `Qwen2_5_VLMoEModel.forward()` 的 MoE 路由细节

这一节只看模型内部，不再看外层脚本。

`Qwen2_5_VLMoEModel.forward()` 定义在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:356`

它接收的关键额外输入有：

- `moe_token_types`
- `start_indices`
- `end_indices`
- `positional_masks`
- `adarms_conds`

可以把这层 forward 理解成：

```text
多模态 embeddings
  -> 按 expert 规则准备位置与 mask
  -> 可选 token permutation
  -> 多层 MoE decoder block
  -> 可选 unpermute
  -> 输出统一 hidden_states
```

### 13.1 `start_indices / end_indices` 的作用

如果外部没有传 `start_indices` 和 `end_indices`，模型会根据 `moe_token_types` 统计每个 expert 的 token 数量，然后现场计算每个 expert 段的边界：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1336-1346`

逻辑是：

```text
group_size[i] = expert i 的 token 数
start_indices[i] = 该 expert 在 permutation 后的起点
end_indices[i] = 该 expert 在 permutation 后的终点
```

这两个向量的意义是：

- 如果 token 已经按 expert 聚拢，那么每个 expert 的 token 会落在一个连续区间
- 后续 norm / MLP / residual gate 都可以直接基于这个连续片段处理

### 13.2 路由并不复杂，核心是 token type 到 expert 的映射

最基本的路由器是 `TokenTypeRouter`，定义在：

- `wall_x/model/vla_mixin.py:34`

它的逻辑非常直接：

```python
experts_indices = token_types % self.num_experts
```

位置在：

- `wall_x/model/vla_mixin.py:39-49`

也就是说，这里的 MoE routing 不是 learned router 打分 top-k，而是更偏“规则路由”：

- token 类型先由上游定义
- 再用简单映射分配到 expert

因此 Wall-X 的 MoE 更接近“按 token 语义类型分专家”，而不是标准 LLM MoE 里那种“每个 token 动态选择 top-k expert”。

### 13.3 forward 开头先处理 mask、position_ids 和 position embedding

`Qwen2_5_VLMoEModel.forward()` 开头主要做三件事：

1. 构造 causal mask 或 joint attention mask
2. 必要时修正 `position_ids`
3. 生成共享的 `position_embeddings`

关键代码：

- `_update_causal_mask(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:584`
- forward 内部调用 `_update_causal_mask(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:434-442`
- `_update_position_ids(...)`：`wall_x/model/vla_mixin.py:445-472`
- `_update_joint_attention_mask_2d(...)`：`wall_x/model/vla_mixin.py:474-549`
- 生成共享位置编码：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:456-457`

其中有几个关键点。

#### a. `_update_causal_mask()` 会把 type-1 token 区域改成双向可见

在 `_update_causal_mask()` 里，如果 `moe_token_types` 不为空，就会把 `moe_token_types == 1` 的区域从普通 causal mask 改成全连接可见：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:662-672`

这说明当前实现对“动作 token 区域”有一个显式特判。

#### b. `_update_position_ids()` 会对 flow token 的位置做重对齐

当 `attention_moe=True` 且不是 FlashAttention2 时，模型会调用 `_update_position_ids(...)`：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:448-454`

这个函数会根据：

- `ar_predict_token_positions`
- `flow_mask`

来对 flow token 的位置编码做偏移修正，位置在：

- `wall_x/model/vla_mixin.py:456-470`

它的目标可以理解成：

- 保持 AR 预测段和 flow 动作段在位置编码上的相对关系合理
- 避免两种 token 段共享同一套绝对位置后产生冲突

#### c. `_update_joint_attention_mask_2d()` 会按 token 语义修正 2D attention

当 attention 类型需要 2D mask 且 `attention_moe=True` 时，会走：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:477-487`

这个函数会先构造下三角 causal mask，然后再按 token 类型改写：

- `wall_x/model/vla_mixin.py:483-491`

接着处理几类特例：

- `moe1` 区域全开或保留因果约束：`wall_x/model/vla_mixin.py:508-516`
- AR 预测 token 对 flow token 的屏蔽：`wall_x/model/vla_mixin.py:518-528`
- 无效 flow action 位置的屏蔽：`wall_x/model/vla_mixin.py:530-547`

这部分说明：Wall-X 的 attention 规则不是统一一张标准 causal mask，而是“按 token 角色重写可见性”。

### 13.4 `mot_opt=True` 时会先把 token 重排

如果打开 `mot_opt`，模型会在进入 decoder 之前先对 token 做一次 permutation：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:459-465`

对应调用的是自定义算子：

- `ops.permute(...)`

其作用是把同一个 expert 的 token 放到连续区间里。这样后面的 expert 处理就能直接按 `[start:end]` 片段处理，而不必反复做稀疏 gather/scatter。

forward 结束后，如果 `mot_opt=True`，还会再做一次：

- `ops.unpermute(...)`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:566-568`

所以 `mot_opt` 的本质就是：

```text
先 permute 聚拢 expert token
  -> expert 内部连续计算
  -> 再 unpermute 恢复原顺序
```

### 13.5 每一层 decoder 的执行顺序

`Qwen2_5_VLDecoderLayer_with_MoE.forward()` 定义在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:161`

它的主流程是：

```text
residual = hidden_states
  -> 输入 norm_moe
  -> self_attn
  -> gated residual
  -> post-attn norm_moe
  -> mlp_moe
  -> gated residual
```

关键代码：

- 第一段 norm：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:208-217`
- self attention：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:219-247`
- 第一次 gated residual：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:249-251`
- 第二段 norm：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:256-265`
- MLP/MoE：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:267-269`
- 第二次 gated residual：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:271-273`

### 13.6 `norm_moe` 和 gated residual 的细节

MoE-aware norm 在 `ActionModelMixMin._apply_norm_moe(...)` 里，定义在：

- `wall_x/model/vla_mixin.py:179`

它分两种情况：

- `mot_opt=True`：按 `[start:end]` 连续区间处理 expert token，位置在 `wall_x/model/vla_mixin.py:208-266`
- `mot_opt=False`：按 token mask 做选择和 scatter 回写，位置在 `wall_x/model/vla_mixin.py:271-333`

如果 `norm_moe=False`，就退化成共享 norm：

- `wall_x/model/vla_mixin.py:337-352`

更特别的是 residual 不是简单的 `x + y`，而是走 `_gated_residual(...)`：

- 定义在 `wall_x/model/vla_mixin.py:356-385`

这里当 `gate` 存在时，会只对 action expert 对应的那段 hidden state 乘 gate，再加回 residual：

- 位置在 `wall_x/model/vla_mixin.py:375-383`

这意味着：

- gate 主要是对 action expert 的输出强度做调制
- 普通 token 路径并不一定经历同样的门控

### 13.7 `mlp_moe` 的实际执行方式

`_apply_mlp_moe(...)` 定义在：

- `wall_x/model/vla_mixin.py:170-177`

如果 `mlp_moe=True`，就调用 `SparseMoeBlock`。否则退化成普通 `self.mlp(...)`。

`SparseMoeBlock` 定义在：

- `wall_x/model/vla_mixin.py:85`

它的执行模式是：

1. 如果还没 permute，就先用 `ops.permute(...)` 把 token 按 expert 聚拢
2. 遍历每个 expert
3. 对该 expert 的连续 token 片段做 MLP
4. 只处理该 expert 对应的 `dim_input`
5. 必要时再 `ops.unpermute(...)`

关键代码：

- 输入 permute：`wall_x/model/vla_mixin.py:115-124`
- expert 循环：`wall_x/model/vla_mixin.py:128-139`
- 输出 unpermute：`wall_x/model/vla_mixin.py:140-144`

这里有一个很重要的工程细节：

- 每个 expert 不一定处理完整 hidden dim
- `dim_inputs[expert_idx]` 决定了该 expert 实际吃进去多少维

这意味着 Wall-X 的 expert 不是简单复制完整 FFN，而是允许不同 expert 拥有不同输入维度。

## 14. Flow Action 采样过程

这一节只看连续动作那条路径，也就是：

- 外层 `scripts/infer_robochallenge.py:1047-1078`
- 内层 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1959-2360`

### 14.1 `ActionProcessor` 在训练时做什么

`ActionProcessor.forward()` 定义在：

- `wall_x/model/action_head.py:668`

训练阶段它做的是“构造 noisy action token embedding + flow target”：

1. 采样噪声 `noise`
2. 采样时间 `time`
3. 构造 `noisy_action = (1 - t) * noise + t * action_chunk`
4. 构造监督目标 `flow = action_chunk - noise`
5. 用 `w1 / w2 / w3` 和时间嵌入把 noisy action 投影到 hidden space

关键代码：

- `sample_time(...)`：`wall_x/model/action_head.py:616-631`
- 构造 `noisy_action` / `flow`：`wall_x/model/action_head.py:684-690`
- 时间嵌入：`wall_x/model/action_head.py:691-693`
- 动作投影：`wall_x/model/action_head.py:697-738`

这一步的输出是：

- `action_time_embed`
- `flow`
- `adarms_cond`

其中 `action_time_embed` 会被塞回输入 token 序列的 `<|action|>` 位置。

### 14.2 推理时不是直接走 `ActionProcessor.forward()`，而是走 `step()`

连续动作推理使用的是：

- `ActionProcessor.step(...)`
- 定义在 `wall_x/model/action_head.py:740-781`

它的作用是：

- 给定当前 `timestep`
- 给定当前 `noisy_action`
- 产出当前步对应的 action embedding
- 同时返回 `adarms_cond`

这相当于“把一个连续动作状态编码成当前时间步下的 action token embedding”。

### 14.3 `generate_flow_action()` 的第一阶段是前缀预填充

`generate_flow_action()` 定义在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1959`

它开头仍然会先准备：

- 文本 embedding
- 图像 embedding
- proprio embedding
- 位置编码
- `start_indices / end_indices`

对应关键位置：

- 多模态 embedding 处理：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2017-2080`
- 位置编码与 group index：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2084-2128`

然后它做一件非常关键的事：

1. 初始化 `noise`
2. 令 `noisy_action = noise`
3. 用 `ActionProcessor.step(t=0, noisy_action)` 得到初始 action embedding
4. 把这些 action embedding 填进 `<|action|>` token 位

对应代码：

- 初始化噪声：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2139-2145`
- 构造时间网格：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2147-2157`
- 初始 action embedding：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2158-2166`

接着它会先跑一次完整的 transformer forward：

- `prefetch_output = self.model(...)`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2171-2186`

这一步的意义是：

- 先把“静态上下文 + 初始动作 token”整体 prefill 一次
- 得到 `prefix_kv_cache`
- 同时得到第一步的动作速度场预测 `v_0`

对应代码：

- 取出 `past_key_values`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2187-2188`
- 从 action hidden state 投影出 `action_pred`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2190-2197`
- 用 `v_0` 更新 `noisy_action`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2204-2206`

这一段从思想上看，很像“先做一次 prefix prefill，再缓存上下文”，只是它不是标准文本生成，而是连续动作 ODE 采样前的预计算。

### 14.4 之后会把 KV cache 截成 prefix-only

完成第一次 prefetch 之后，代码会找到第一个 action token 位置，作为 `prefix_length`：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2212-2217`

然后把 KV cache 截到 prefix 部分：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2219-2234`

再把后缀信息单独切出来：

- `postfix_position_ids`
- `postfix_inputs_embeds`
- `postfix_attention_mask`
- `postfix_moe_token_types`

位置在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2236-2240`

同时还会为 postfix 段单独重算：

- `postfix_start_indices`
- `postfix_end_indices`

位置在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2242-2250`

### 14.5 ODE 迭代时只重算后缀 action 区域

接下来 `generate_flow_action()` 定义了一个内部函数：

- `step_with_kvcache(...)`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2293`

它每一步做的是：

1. 用当前 `timestep` 和 `noisy_action` 得到新的 action embedding
2. 只替换 postfix 段中的 action token
3. 复用 `prefix_kv_cache`
4. 再跑一次 `self.model(...)`
5. 从新的 action hidden states 投影得到 `v_t`

关键代码：

- 当前步 action embedding：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2298-2305`
- 复用 prefix cache 的 transformer forward：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2306-2320`
- 投影出 `v_t`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2322-2331`

然后主函数用：

- `odeint(..., method="euler")`

来沿时间网格推进：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2333-2335`

最后取最后一步轨迹作为预测动作：

- `predict_action = action_trajectory[-1]`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2340`

再按数据集做反归一化：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2341-2346`

### 14.6 flow 路径的本质

如果压成一句话，Wall-X 的 flow action 推理其实是：

```text
先把连续动作变量编码成 action token embedding
  -> 和文本/图像/状态一起送进 transformer
  -> transformer 预测当前动作流场 v_t
  -> 用 ODE/Euler 更新 noisy_action
  -> 重复这个过程直到得到最终动作
```

所以它不是“transformer 直接回归动作值”，而是：

- transformer 负责建模多模态条件下的动作流场
- `ActionProcessor` 负责动作变量和 token 表示之间的转换
- `odeint` 负责把局部流场积分成最终动作轨迹

## 15. 补充结论

把本文档和前一部分合在一起，可以得到一个更完整的认识：

1. Wall-X 的骨架仍然是 `Qwen2.5-VL`，即视觉编码器加 decoder-only 文本主干。
2. 它最核心的增量不只是“加了动作头”，而是把状态和动作都变成 token 级条件，混入统一序列里建模。
3. `moe_token_types` 决定了 token 该走哪条专家路径；在当前推理脚本里，它基本就是“普通 token”和“动作 token”两组。
4. `mot_opt + permute/unpermute` 的目的不是改模型语义，而是让 expert token 聚成连续片段，方便高效计算。
5. flow action 路径本质上是“用 transformer 反复估计动作流场，再做 ODE 积分”，而不是一次性直接输出连续动作。

如果还要继续往下拆，下一层最值得单独写的是：

1. `JointQwen2VLAttention` 与 `attention_moe=True` 时的联合注意力细节
2. `scatter_proprioception_embeddings()` 和 `scatter_flow_action_embeddings()` 对 token 序列布局的具体影响

## 16. `JointQwen2VLAttention` 在 `attention_moe=True` 时做了什么

当 `Qwen2_5_VLDecoderLayer_with_MoE` 发现：

- `config.attention_moe == True`

时，它不会使用普通的 `Qwen2_5_VLAttention`，而是使用 joint attention 版本：

- 选择逻辑在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:102-109`
- attention 类映射在 `wall_x/model/joint_attention.py:652-656`

这里要注意一件事：

- `JOINT_QWEN_ATTENTION_CLASSES["sdpa"] = JointQwen2VLAttention`
- 也就是说，虽然配置名字叫 `sdpa`，但在 attention-moe 场景下用的是 `JointQwen2VLAttention` 这个类
- 只是这个类内部最终仍然调用 `scaled_dot_product_attention(...)`

### 16.1 joint attention 的核心思路

`JointQwen2VLAttention` 定义在：

- `wall_x/model/joint_attention.py:48`

它和普通 attention 的最大区别不是 attention 公式本身，而是：

- Q/K/V/O 都变成了“按 expert 分组的独立线性层”

具体看初始化：

- `q_proj_experts`：`wall_x/model/joint_attention.py:82-87`
- `k_proj_experts`：`wall_x/model/joint_attention.py:88-95`
- `v_proj_experts`：`wall_x/model/joint_attention.py:96-103`
- `o_proj_experts`：`wall_x/model/joint_attention.py:104-109`

这意味着它的结构更像：

```text
expert 0: q_proj_0 / k_proj_0 / v_proj_0 / o_proj_0
expert 1: q_proj_1 / k_proj_1 / v_proj_1 / o_proj_1
...
```

而不是一套共享 QKV 投影。

### 16.2 每个 expert 只处理自己的输入维度

joint attention 还有一个很重要的特征：

- 每个 expert 不一定吃完整 hidden size
- 而是只处理 `dim_inputs[expert_idx]` 那一段

关键位置：

- `self.dim_inputs = config.dim_inputs`：`wall_x/model/joint_attention.py:75`
- `_generate_qkv()` 中截取 `selected_hidden[:, :dim_input]`：`wall_x/model/joint_attention.py:313-325`
- `_generate_output()` 中只写回 `:dim_input`：`wall_x/model/joint_attention.py:457-471`

也就是说，joint attention 的专家化不是“不同 token 走不同共享层”，而是：

- token 先按类型选 expert
- 每个 expert 用自己的 QKV/O 投影
- 每个 expert 只负责 hidden state 的一个前缀子空间

### 16.3 `mot_opt=False` 时的 joint attention

当 `mot_opt=False` 时，joint attention 会先根据 `token_types == expert_idx` 构造 mask：

- `wall_x/model/joint_attention.py:174-180`

然后 `_generate_qkv()` 会：

1. 为整批 token 初始化空的 `query_states / key_states / value_states`
2. 对每个 expert，用 mask 选出属于它的 token
3. 对这些 token 应用对应 expert 的 `q_proj / k_proj / v_proj`
4. 再 scatter 回统一的 Q/K/V 张量

代码在：

- `wall_x/model/joint_attention.py:279-335`

这一版最接近“按 token mask 选专家”的直觉实现。

### 16.4 `mot_opt=True` 时的 joint attention

当 `mot_opt=True` 时，joint attention 会走 `_generate_qkv_mot_opt(...)`：

- 调用点：`wall_x/model/joint_attention.py:161-172`
- 实现：`wall_x/model/joint_attention.py:337-427`

这时输入的 `hidden_states` 已经是按 expert 排好序的 token 流。

函数内部会：

1. 按 `[start:end]` 切出每个 expert 的连续 token 片段
2. 对每个 expert 的 token 段做 Q/K/V 投影
3. 把结果先写到 permuted buffer
4. 再用 `ops.unpermute(...)` 恢复到原始 token 顺序

关键代码：

- 逐 expert 处理连续段：`wall_x/model/joint_attention.py:384-408`
- `ops.unpermute(...)` 恢复顺序：`wall_x/model/joint_attention.py:410-414`

从工程角度看，这一版的目的就是减少稀疏索引开销，让同一 expert 的 token 连续计算。

### 16.5 attention 核本身没有变，变的是 QKV/O 的生成方式

无论 joint 还是非 joint，attention 的核心核函数仍然是标准那套：

- `scaled_dot_product_attention(...)`：`wall_x/model/joint_attention.py:258-265`
- Flash 版走 `flash_attn_func(...)`：`wall_x/model/joint_attention.py:631-638`

所以 joint attention 的重点不是“发明了新 attention 公式”，而是：

- 在 attention 前后，把投影层改成 expert-aware
- 再配合专门的 mask 和 token 路由规则

可以把它概括成：

```text
专家化的是 Q/K/V/O 投影与 token 分组
attention 内核本身仍然是标准 SDPA / FlashAttention
```

### 16.6 joint attention 输出如何还原

如果 `mot_opt=False`，输出通过 `_generate_output()` scatter 回原始序列：

- `wall_x/model/joint_attention.py:447-473`

如果 `mot_opt=True`，输出会先保留在 expert 排序空间，再由外层 model forward 的 `ops.unpermute(...)` 做整体恢复：

- `_generate_output_mot_opt()`：`wall_x/model/joint_attention.py:475-527`
- 外层整体 `ops.unpermute(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:566-568`

这也解释了为什么：

- 在 `mot_opt=True` 场景里，很多中间张量都会临时脱离原始 token 顺序
- 但最终返回给上层的 hidden states 仍然会恢复成正常 `[B, S, H]`

## 17. Token 序列布局：prompt、图像占位、`<|propri|>`、`<|action|>`

这一节把外层 prompt 文本和模型里实际被替换的 embedding 对应起来。

### 17.1 AR 模式的 prompt 结构

AR 文本由 `get_text_ar()` 构造：

- `scripts/infer_robochallenge.py:581-627`

它的大致结构是：

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
Observation:
  camera_1: <|vision_start|><|image_pad|><|vision_end|>
  camera_2: <|vision_start|><|image_pad|><|vision_end|>
Instruction: ...
Proprioception: ...
<|im_end|>
<|im_start|>assistant
```

几个关键点：

- AR prompt 里没有预先放 `<|action|>` 占位
- 它只放到 `assistant` 起始位置
- 后续动作 token 是靠 `generate()` 自回归生成出来的

### 17.2 Flow 模式的 prompt 结构

Flow 文本由 `get_text_flow()` 构造：

- `scripts/infer_robochallenge.py:629-683`

它和 AR 最大的区别是最后多了一段：

```python
action = f"{action_symbol * action_chunk_size}"
text = prologue + user_message + assistant_message + action
```

位置在：

- `scripts/infer_robochallenge.py:679-681`

这意味着 flow 模式不是“生成 action token”，而是：

- 先在输入序列里明确放入固定长度的 `<|action|>` token 段
- 再把这段 token 的 embedding 替换成连续动作对应的 latent/action embedding

### 17.3 `state_str` 会改变 `Proprioception` 的表示方式

无论 AR 还是 flow，`Proprioception:` 后面的内容都有两种模式。

如果 `state_str=True`：

- 会把归一化状态离散化成 0 到 255 的字符串序列
- 位置在 `scripts/infer_robochallenge.py:595-615` 和 `scripts/infer_robochallenge.py:653-670`

如果 `state_str=False`：

- 文本里放的是 `<|propri|>` 特殊 token
- 位置在 `scripts/infer_robochallenge.py:618-621` 和 `scripts/infer_robochallenge.py:673-676`

这两种模式的差别很大：

- `state_str=True` 时，状态作为普通文本 token 参与建模
- `state_str=False` 时，状态作为一个专用占位 token，后续会被真正的 proprio embedding 替换

### 17.4 `<|image_pad|>` 不会只对应一个 token

在 `_construct_input()` 里，图像占位会根据 `image_grid_thw` 被扩展成多个 `<|image_pad|>`：

- `scripts/infer_robochallenge.py:789-803`

关键逻辑是：

```python
"<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
```

位置在：

- `scripts/infer_robochallenge.py:797-800`

这说明：

- prompt 里写一个 `<|image_pad|>` 只是模板写法
- 真正 tokenize 之前，它会被展开成与视觉 token 数量匹配的多个占位

所以图像 token 数量不是固定的 1，而是由：

- `image_grid_thw`
- `merge_size`

共同决定。

### 17.5 `<|propri|>` 和 `<|action|>` 的 embedding 替换

模型初始化时会记录这些 special token 的 ID：

- `action_token_id` 和 `propri_token_id`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1007-1015`

后续在 forward 里：

- `<|propri|>` 位置会被 `scatter_proprioception_embeddings(...)` 替换
- `<|action|>` 位置会被 `scatter_flow_action_embeddings(...)` 替换

关键位置：

- `scatter_proprioception_embeddings(...)`：`wall_x/model/vla_mixin.py:387-418`
- `scatter_flow_action_embeddings(...)`：`wall_x/model/vla_mixin.py:420-442`

替换方式本质上都是：

```text
先找到对应 token 的 mask
  -> 计算真实 embedding
  -> 用 masked_scatter / 直接赋值写回 inputs_embeds
```

### 17.6 当前推理脚本里 `moe_token_types` 如何和 token 布局对应

在 `_construct_input()` 的最后：

- `moe_token_types = inputs.input_ids == action_token_id`
- 位置在 `scripts/infer_robochallenge.py:810-811`

因此对于当前推理脚本来说：

- 只有 `<|action|>` 段会被标成 `1`
- 其他 token 都是 `0`

也就是说，当前默认布局下：

- 图像 token 不是 expert-1
- `<|propri|>` token 不是 expert-1
- 只有 flow prompt 末尾那串 `<|action|>` token 属于动作 expert

这正好和前面的观察一致：

- 当前推理路径里，真正的 expert 分流主要就是“普通上下文” vs “动作 token”

### 17.7 AR 生成时 token 布局和 cache 的一个细节

`prepare_inputs_for_generation()` 会在缓存增量生成时同步切 `moe_token_types`：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2478-2495`

同时在 continuation step 里会关闭视觉输入重复传递：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2497-2500`

这说明生成阶段的布局规则是：

- 初次 prefill 时把图像、状态、文本全部送进去
- 后续 decode 只增量喂新 token 和对应的 `moe_token_types`

从结构角度看，这和普通 LLM generation 的 cache 逻辑一致，只是多了：

- 视觉输入的首步专用处理
- `moe_token_types` 的同步切片

## 18. 当前文档覆盖范围

到这里，这份文档已经覆盖了 Wall-X 里最核心的四层结构：

1. 外层推理调用顺序
2. 输入 batch 的构造与模型加载
3. 视觉主干、文本主干、MoE 解码器
4. 连续动作 flow 采样和 token 布局

如果还要继续往下拆，下一批最值得单独成文的是：

1. 训练路径，也就是 `forward(train)` 时的 loss 组成
2. `flow_loss`、cross-entropy loss、channel-wise loss 在训练中怎么组合
3. `rot_pos_emb`、`get_window_index`、`rope_index` 这几个 CUDA kernel 的输入输出张量语义

## 19. 训练路径：从 trainer 到 `train_step_forward()`

如果把训练这条线也压成调用顺序，大致是：

```text
QwenVlAct_Trainer.__init__()
  -> load_normalizer()
  -> load_model()
  -> load_qact_data()

fit()
  -> train_loop()
  -> outputs = self.model(**batch, mode="train")
  -> Qwen2_5_VLMoEForAction.forward(mode="train")
  -> train_step_forward(...)
  -> compute_loss(...)
```

关键位置：

- trainer 初始化：`wall_x/trainer/qwen_vl_act_trainer.py:130`
- `load_model()`：`wall_x/trainer/qwen_vl_act_trainer.py:570`
- `load_qact_data()`：`wall_x/trainer/qwen_vl_act_trainer.py:746`
- 训练主循环：`wall_x/trainer/qwen_vl_act_trainer.py:300`
- 训练时 forward 调用：`wall_x/trainer/qwen_vl_act_trainer.py:377`
- 模型 `forward()` 的 mode 分发：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2363-2395`

也就是说，训练时不会走 `predict()` 或 `generate_flow_action()`，而是统一走：

- `mode="train"` -> `train_step_forward(use_cache=False, **kwargs)`

### 19.1 trainer 实际消费哪些返回值

trainer 在训练循环里直接取：

- `loss = outputs.loss`

位置在：

- `wall_x/trainer/qwen_vl_act_trainer.py:380`

然后调用：

- `self.accelerator.backward(loss)`

位置在：

- `wall_x/trainer/qwen_vl_act_trainer.py:391`

除了总 loss，trainer 还会尝试记录：

- `outputs.cross_entropy_loss`
- `outputs.flow_loss`
- `outputs.channel_loss_dict`
- `outputs.channel_loss_count_dict`

对应日志逻辑在：

- `wall_x/trainer/qwen_vl_act_trainer.py:420-483`

验证集也复用同一条训练前向路径，只是包在 `torch.no_grad()` 下：

- `wall_x/trainer/qwen_vl_act_trainer.py:555-558`

## 20. 训练 batch 是如何构造的

### 20.1 数据加载入口

trainer 通过：

- `load_lerobot_data(...)`

来加载训练数据，位置在：

- `wall_x/trainer/qwen_vl_act_trainer.py:757-764`

真正把样本拼成训练 batch 的是 `DataCollator`：

- 定义在 `wall_x/data/load_lerobot_dataset.py:246`
- `__call__()` 主逻辑在 `wall_x/data/load_lerobot_dataset.py:328`

### 20.2 collator 生成的关键字段

`DataCollator.__call__()` 会先从原始样本里抽出并归一化：

- `proprioception`
- `agent_pos_mask`
- `action_chunk`
- `dof_mask`

关键代码：

- 处理 `agent_pos`：`wall_x/data/load_lerobot_dataset.py:332-367`
- 处理 `action`：`wall_x/data/load_lerobot_dataset.py:368-400`

随后它会构造文本，并调用：

- `replace_action_token(...)`

位置在：

- `wall_x/data/load_lerobot_dataset.py:416-422`

最后再调用：

- `preprocesser_call(...)`

位置在：

- `wall_x/data/load_lerobot_dataset.py:424-433`

最终 batch 至少会包含：

- `input_ids`
- `attention_mask`
- `pixel_values` / `image_grid_thw`
- `labels`
- `action_chunk`
- `proprioception`
- `dof_mask`
- `agent_pos_mask`
- `moe_token_types`
- `dataset_names`

其中 `moe_token_types` 的生成位置在：

- `wall_x/data/load_lerobot_dataset.py:435-438`

逻辑和推理脚本一致，仍然是：

- `inputs.input_ids == action_token_id`

### 20.3 训练 prompt 里的 action token 有两层

训练时构造完整多模态 prompt 的函数在：

- `wall_x/data/utils.py:465`

正常动作训练分支会生成：

```text
<|im_start|>assistant
<|action_fast|><|im_end|>
<|action|><|action|><|action|>...
```

关键代码：

- `assistant_output = ... <|action_fast|> ... {action_symbol * action_chunk_size}`
- 位置在 `wall_x/data/utils.py:549`

这里其实有两层动作表示：

1. `<|action_fast|>`：给离散 action token 监督留的位置
2. `<|action|>` 重复若干次：给连续 flow action expert 留的位置

### 20.4 `replace_action_token()` 决定离散动作监督是否存在

`replace_action_token()` 定义在：

- `wall_x/data/utils.py:606`

如果提供了 fast action tokenizer：

- 会把 `<|action_fast|><|im_end|>\n` 替换成真实的 `<|action_token_i|>` 序列
- 再把剩余的 `<|action|>` 删除

关键代码：

- 替换 fast action token：`wall_x/data/utils.py:646-652`
- 删除剩余 `<|action|>`：`wall_x/data/utils.py:654-655`

如果没有 fast action tokenizer：

- 只会去掉 `<|action_fast|><|im_end|>\n`
- 连续 `<|action|>` 占位会保留

对应代码：

- `wall_x/data/utils.py:656-659`

这会直接影响后面的 loss 形式：

- 有 fast tokenizer 时，训练里会有离散 action token 的交叉熵监督
- 没有 fast tokenizer 时，动作段主要走 flow loss，而不是 CE

### 20.5 `labels` 是怎么生成的

`preprocesser_call()` 在：

- `wall_x/data/utils.py:128`

它会只保留 assistant 响应部分的 label，其他位置都设成 `-100`：

- 生成 `labels = -100`：`wall_x/data/utils.py:231-232`
- 只标注 assistant response 区域：`wall_x/data/utils.py:269-271`

然后再额外 mask 掉：

- `<|action|>`
- `<|propri|>`
- `pad_token`

对应代码：

- `wall_x/data/utils.py:273-278`

这点非常重要：

- 连续动作占位 `<|action|>` 默认不会参与交叉熵
- `<|propri|>` 也不会参与交叉熵
- CE 主要监督的是普通 assistant 文本，以及 fast tokenizer 替换进来的离散 action token

如果一个样本最后所有 label 都是 `-100`，那就直接：

- `labels = None`

位置在：

- `wall_x/data/utils.py:280-284`

### 20.6 `flow_loss_mask` 在默认数据管线里不是必备字段

`train_step_forward()` 和 `compute_loss()` 都支持：

- `flow_loss_mask`

但从默认的 `DataCollator` 构造路径来看，没有看到它被直接生成。  
按当前源码，`flow_loss_mask` 更像是一个可选额外字段，而不是这条基础数据管线的必备输出。

## 21. `train_step_forward()` 和 loss 组成

### 21.1 `train_step_forward()` 做了什么

`train_step_forward()` 定义在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1287`

它的大框架和推理 forward 很像：

1. 准备 `start_indices / end_indices`
2. 构造多模态 `inputs_embeds`
3. 注入图像、proprio、flow action embedding
4. 调用 `self.model(...)`
5. 取 `hidden_states` 和 `logits`
6. 调 `compute_loss(...)`

关键位置：

- 生成 `start_indices / end_indices`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1336-1346`
- 注入 proprio / flow action：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1416-1422`
- 调 decoder：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1427-1443`
- 取 `hidden_states` / `logits`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1445-1446`
- 进入 `compute_loss(...)`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1448-1464`

最终返回的是：

- `Qwen2_5_VLACausalLMOutputWithPast`

里面显式保留了：

- `loss`
- `cross_entropy_loss`
- `flow_loss`
- `channel_loss_dict`
- `channel_loss_count_dict`

定义在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:60-72`

### 21.2 `compute_loss()` 的总结构

`compute_loss()` 定义在：

- `wall_x/model/vla_mixin.py:763`

它的逻辑非常直接：

```text
loss = 0
if labels is not None:
    计算 cross_entropy_loss
    loss += cross_entropy_loss

if action_chunk is not None:
    计算 flow_loss
    loss += flow_loss * flow_loss_weight
```

关键代码：

- 初始化：`wall_x/model/vla_mixin.py:779-780`
- CE 分支：`wall_x/model/vla_mixin.py:799-850`
- flow 分支：`wall_x/model/vla_mixin.py:852-866`

所以从源码看，总 loss 是：

```text
total_loss = cross_entropy_loss + flow_loss_weight * flow_loss
```

其中：

- `flow_loss_weight` 来自配置
- 会存到 `self.config.flow_loss_weight`

### 21.3 交叉熵 loss 的计算方式

CE 部分是标准 next-token 语言建模写法：

- `shift_logits = logits[..., :-1, :]`
- `shift_labels = labels[..., 1:]`

位置在：

- `wall_x/model/vla_mixin.py:802-805`

再用 unreduced 的：

- `CrossEntropyLoss(reduction="none")`

这个 loss object 在模型初始化时创建：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:962`

最后只对 `shift_labels != -100` 的位置取平均：

- `wall_x/model/vla_mixin.py:808-814`

因此：

- 普通文本 assistant response 会进入 CE
- fast action tokenizer 产生的 `<|action_token_i|>` 也会进入 CE
- `<|action|>` 和 `<|propri|>` 不会进入 CE，因为在 data preprocessing 已经被 mask 成 `-100`

### 21.4 action accuracy 是如何算的

如果 `action_token_id_set["action_token_list"]` 非空，也就是启用了 fast action tokenizer，那么还会额外计算 action token 的 top-1 准确率：

- `action_preds = shift_logits.argmax(dim=-1)`
- `action_mask = shift_labels > first_action_token_id`

位置在：

- `wall_x/model/vla_mixin.py:839-850`

最终它会被塞到：

- `channel_loss_dict["action_accuracy"]`

trainer 侧只有在 `use_fast_tokenizer=True` 时才会记录这个指标：

- `wall_x/trainer/qwen_vl_act_trainer.py:470-483`

### 21.5 flow loss 的计算方式

flow 部分首先会找到输入序列里所有 `<|action|>` token 的位置：

- `action_mask = input_ids == self.action_token_id_set["action_token_id"]`
- 位置在 `wall_x/model/vla_mixin.py:853`

然后从 decoder 输出里抽出这些位置对应的 hidden states：

- `action_hidden_states = hidden_states[action_mask].to(torch.float32)`
- 位置在 `wall_x/model/vla_mixin.py:855`

再把训练阶段先前构造好的 `flow` target reshape 成同样的 token 维度：

- `flow = flow.reshape(-1, flow.shape[-1])`
- 位置在 `wall_x/model/vla_mixin.py:856`

最后调用：

- `self.action_preprocessor.flow_loss(...)`
- 位置在 `wall_x/model/vla_mixin.py:857-858`

### 21.6 `ActionProcessor.flow_loss()` 内部做了什么

`flow_loss()` 定义在：

- `wall_x/model/action_head.py:783`

它的逻辑是：

1. 把 `action_hidden_states` 先投影回动作空间
2. 得到 `v_pred`
3. 对 `v_pred` 和 `flow` 做逐元素 MSE
4. 用 `dof_mask` 过滤无效动作维度
5. 可选再乘 `flow_loss_mask`

关键代码：

- `action_proj_back(...)`：`wall_x/model/action_head.py:792-795`
- `self.mse_loss(v_pred, flow)`：`wall_x/model/action_head.py:796`
- `dof_mask`：`wall_x/model/action_head.py:797-799`
- `flow_loss_mask`：`wall_x/model/action_head.py:801-807`

然后在 `compute_loss()` 里：

- 先对 `_flow_loss` 取 mean 得到标量 `flow_loss`
- 再乘 `self.config.flow_loss_weight`

对应代码：

- `wall_x/model/vla_mixin.py:860-862`

### 21.7 per-dataset `channel_loss` 统计的源码现状

`compute_loss()` 里本来显然是打算统计：

- 每个 dataset 的 CE loss 累积和
- 每个 dataset 的 token count

但当前源码里这部分初始化被注释掉了，转而把：

- `unique_datasets_name`
- `channel_loss_dict`
- `channel_loss_count_dict`

都直接设成了 `None`：

- `wall_x/model/vla_mixin.py:782-797`

后面却仍然保留了：

- `for dataset_name_i in unique_datasets_name:`

这段统计逻辑，位置在：

- `wall_x/model/vla_mixin.py:816-830`

从纯源码阅读角度看，这一段当前是未完成状态，和 trainer 侧希望读取 `channel_loss_dict/channel_loss_count_dict` 的日志逻辑之间存在不一致。

也就是说：

- 总 loss、`cross_entropy_loss`、`flow_loss` 这三项逻辑是清晰的
- `channel_loss_dict` 这条 per-dataset 统计分支按当前源码看并不完整

## 22. 训练路径的一句总结

如果把训练部分压成一句话，就是：

```text
数据侧先把 assistant 文本监督、离散 action token 监督、连续 action chunk 监督一起构造成 batch，
模型前向再把它们拆成两条 loss：
  1. 文本/离散动作 token 的交叉熵
  2. 连续动作 `<|action|>` 段的 flow MSE
最后按 `cross_entropy_loss + flow_loss_weight * flow_loss` 汇总成总 loss。
```

如果继续往下拆，下一步最值得单独写的是：

1. 训练 prompt 在 fast tokenizer 和非 fast tokenizer 两种配置下的精确差异
2. `flow` target、`noisy_action`、`v_pred` 三者的数学关系
3. 代码里当前未完成的 `channel_loss_dict` 分支应该原本想统计什么

## 23. Flow 训练目标的数学关系

这一节把前面训练和推理里反复出现的几个量统一一下：

- `action_chunk`
- `noise`
- `time`
- `noisy_action`
- `flow`
- `v_pred`

### 23.1 训练时采样的基本公式

训练阶段，`ActionProcessor.forward()` 会先采样：

- `noise ~ N(0, I)`
- `time ~ Beta(alpha, beta)` 再做缩放

对应代码：

- `sample_time(...)`：`wall_x/model/action_head.py:616-631`
- `noise = torch.randn_like(action_chunk)`：`wall_x/model/action_head.py:685`

然后构造：

```text
noisy_action = (1 - t) * noise + t * action_chunk
flow = action_chunk - noise
```

代码位置：

- `wall_x/model/action_head.py:686-690`

如果写成路径插值，可以记成：

```text
x_t = (1 - t) * x_0 + t * x_1
其中:
  x_0 = noise
  x_1 = action_chunk
```

于是：

```text
dx_t / dt = x_1 - x_0 = action_chunk - noise = flow
```

这就是当前代码里 `flow` 监督信号的来源。

### 23.2 模型真正学的不是 action 本身，而是速度场

训练时，模型不会直接回归 `action_chunk`，而是：

1. 把 `noisy_action` 编码成 `<|action|>` token embedding
2. 让 transformer 输出对应 action token 的 hidden state
3. 再用 `action_proj_back` 把 hidden state 投影回动作空间

关键代码：

- 训练时 action embedding 生成：`wall_x/model/action_head.py:697-738`
- flow loss 前的投影：`wall_x/model/action_head.py:792-795`

于是模型输出的是：

```text
v_pred = action_proj_back(action_hidden_states)
```

再用它去拟合：

```text
flow = action_chunk - noise
```

MSE 定义在：

- `wall_x/model/action_head.py:796`

所以这个训练目标更准确地说是：

```text
让 transformer 学到条件速度场 v_theta(x_t, cond, t)
并逼近真实 flow = x_1 - x_0
```

### 23.3 推理时为什么可以用 ODE/Euler 积分

因为训练学到的是 `v_theta(x_t, cond, t)`，所以推理时就可以从纯噪声出发：

```text
x_0 = noise
```

然后迭代：

```text
x_{t+dt} = x_t + dt * v_theta(x_t, cond, t)
```

在代码里：

- 第一步更新：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2204-2206`
- 后续 ODE 迭代：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2333-2335`

因此当前实现本质上是：

- 训练时学速度场
- 推理时做数值积分

### 23.4 `use_x_pred` 是什么

代码里还支持一个可选分支：

- `use_x_pred`

当它打开时，会把模型输出重新解释为另一种形式：

- 在 prefetch 阶段：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2194-2197`
- 在 ODE 步进阶段：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2327-2330`

当前文档先不深入这个分支，只需要记住：

- 默认可理解为直接预测速度 `v_t`
- `use_x_pred=True` 时，会对输出再做一次与 `noise` 的组合变换

## 24. CUDA 算子的张量语义

这一节只讲三个和视觉/位置编码最相关的自定义算子：

- `rot_pos_emb`
- `get_window_index`
- `rope_index`

它们都在 `ops.cu` 中导出：

- `csrc/ops.cu:11-20`

### 24.1 `rot_pos_emb`

导出名：

- `rot_pos_emb`
- 对应实现：`fused_rot_pos_emb_cuda(...)`
- 声明在 `csrc/rot_pos.h:3`

#### 输入

- `inv_freq`: shape `[dim/2]`，float32，CUDA
- `grid_thw`: shape `[num_grids, 3]`，每行是 `(T, H, W)`，int32 或 int64，CUDA
- `spatial_merge_size`: int

对应检查在：

- `csrc/rot_pos.cu:192-201`
- `csrc/rot_pos.cu:256-265`

#### 输出

输出是：

- shape `[total_tokens, dim]`
- dtype `float32`

其中：

- `dim = 2 * len(inv_freq)`
- 前半部分对应 `h_pos * inv_freq`
- 后半部分对应 `w_pos * inv_freq`

关键代码：

- 输出 shape 创建：`csrc/rot_pos.cu:225-226` 和 `csrc/rot_pos.cu:289-290`
- 写出 `h_pos` / `w_pos`：`csrc/rot_pos.cu:71-74` 与 `csrc/rot_pos.cu:142-145`

#### `total_tokens` 的含义

`total_tokens` 并不是简单的 `sum(T*H*W)`，而是：

```text
sum over grids:
  T * (H / spatial_merge_size) * (W / spatial_merge_size) * spatial_merge_size^2
```

对应 token count 计算：

- `csrc/rot_pos.cu:159-164`
- `csrc/rot_pos.cu:178-183`

从实现上看，它保留了 merged block 内部的细粒度 token 次序，然后为每个 token 生成：

- 它所属的 `h_pos`
- 它所属的 `w_pos`

#### 在 Python 侧怎么用

视觉分支里：

- `rotary_pos_emb = ops.rot_pos_emb(...)`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:612-614`

随后它会被 reshape、按 window 重排，再拼成：

```python
emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
position_embeddings = (emb.cos(), emb.sin())
```

对应代码：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:630-636`

所以 `rot_pos_emb` 输出的不是最终旋转后的 Q/K，而是：

- 用于后续 `cos/sin` 构造的二维位置相位值

### 24.2 `get_window_index`

导出名：

- `get_window_index`
- 对应实现：`get_window_index_cuda(...)`
- 声明在 `csrc/window_index.h:4`

#### 输入

- `grid_thw`: `[num_grids, 3]`，int32，CUDA
- `spatial_merge_size`
- `vit_merger_window_size`
- `patch_size`
- `spatial_merge_unit`

#### 输出

返回两个张量：

1. `window_indices`
2. `cu_window_seqlens`

定义在：

- `csrc/window_index.cu:208-285`

#### `window_indices` 是什么

`window_indices` 是一个一维 int tensor，里面存的是：

- 每个 token 在“按窗口顺序重排”之后，对应原始 LLM token 序列中的索引

最终写值的位置在：

- `csrc/window_index.cu:200-204`

这里的 `value` 是：

```text
t_element_base + abs_h * llm_w + abs_w
```

也就是：

- 时间维 `t`
- 合并后的空间网格 `llm_h, llm_w`

展开后得到的原始 token 线性下标。

#### `cu_window_seqlens` 是什么

`cu_window_seqlens` 是每个窗口 token 数量的前缀和：

- shape `[total_windows + 1]`

构造位置在：

- `csrc/window_index.cu:104-131`

每个窗口的长度先存到 `window_counts`，再乘 `spatial_merge_unit`，然后做 cumulative sum。

这类张量通常用来给：

- varlen attention
- window 内分段处理

提供边界信息。

#### 在 Python 侧怎么用

视觉前向里：

- `window_index, cu_window_seqlens = ops.get_window_index(...)`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:616-622`

然后：

- 用 `window_index` 重排 `hidden_states`
- 用 `cu_window_seqlens` 决定每一层视觉 attention 的窗口边界

对应代码：

- 重排 hidden states：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:624-629`
- 选择 `cu_seqlens_now` / `max_seqlen_now`：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:654-675`

所以这个算子的核心作用是：

- 把视觉 token 从“原图网格顺序”映射到“窗口 attention 顺序”

### 24.3 `rope_index`

导出名：

- `rope_index`
- 对应实现：`get_rope_index(...)`
- 声明在 `csrc/rope_index.h:3`

#### 输入

- `input_ids`
- `image_grid_thw`
- `video_grid_thw`
- `second_per_grid_ts`
- `attention_mask`
- `spatial_merge_size`
- `image_token_id`
- `video_token_id`
- `vision_start_token_id`
- `tokens_per_second`

host 接口定义在：

- `csrc/rope_index.cu:464-474`

#### 输出

返回两个张量：

1. `position_ids`
2. `mrope_deltas`

从 kernel 语义看：

- `position_ids` shape 是 `[3, batch_size, seq_len]`
- `mrope_deltas` shape 是 `[batch_size]`

写出位置：

- `position_ids` 写入：`csrc/rope_index.cu:355-357`
- `mrope_deltas` 写入：`csrc/rope_index.cu:376-379`

#### `position_ids` 的三维含义

对于每个 token，`rope_index` 会输出三条位置轴：

- `pos_t`
- `pos_h`
- `pos_w`

如果 token 是普通文本 token：

- 三个位置都设成同一个一维偏移值

对应代码：

- `csrc/rope_index.cu:337-341`

如果 token 属于视觉 patch：

- 会先把 patch 序号还原成 `(t, h, w)`
- 再分别写出三条位置坐标

对应代码：

- `get_3d_coords(...)`：`csrc/rope_index.cu:40-48`
- 视觉 token 写 `pos_t/pos_h/pos_w`：`csrc/rope_index.cu:344-352`

因此它的语义是：

```text
文本 token: 1D 位置编码，复制到 3 个轴
视觉 token: 真正的 3D 时空位置编码
```

#### `mrope_deltas` 是什么

`mrope_deltas` 记录的是：

```text
max_position + 1 - seq_len
```

位置在：

- `csrc/rope_index.cu:376-379`

它本质上是在描述：

- 当视觉 token 的 3D 位置跨度大于纯文本序列长度时，需要补偿多少位置偏移

在模型侧，`mrope_deltas` 会被缓存起来，用于后续增量生成时恢复正确的 `position_ids`：

- 基础 VL 模型 forward 中缓存：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2161-2168`
- 基础 VL 模型后续 step 复用：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py:2171-2182`
- action 版模型 forward 中同样缓存：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1692-1705`

### 24.4 这三个算子的分工

把它们放在一起看，职责很清楚：

- `rot_pos_emb`
  - 为视觉分支生成 2D 空间位置相位值
  - 主要服务视觉 transformer 的 `cos/sin`

- `get_window_index`
  - 为视觉分支生成窗口重排索引和窗口边界
  - 主要服务视觉 window attention

- `rope_index`
  - 为整条多模态序列生成 3D/1D 混合的 `position_ids`
  - 主要服务 LLM 主干的多模态 RoPE

也就是说：

```text
视觉局部位置: rot_pos_emb + get_window_index
整条多模态序列位置: rope_index
```

## 25. 这一轮补充后的整体结论

到这里，这份文档已经覆盖了三层不同粒度：

1. 系统级
   - 推理入口、训练入口、数据和模型加载
2. 模型级
   - 视觉主干、文本主干、MoE decoder、joint attention、flow action
3. 算子级
   - `rot_pos_emb`、`get_window_index`、`rope_index` 的输入输出语义

如果继续往下拆，下一步最值得写的是两种方向：

1. 代码清理/修复视角
   - `channel_loss_dict` 那段未完成逻辑怎么修
   - AR 路径和 `predict(..., fast)` 的接口边界如何统一
2. 数学视角
   - `use_x_pred` 分支到底在学什么
   - flow ODE 路径和 diffusion policy 的对应关系

## 26. 代码清理/修复视角：当前最明显的两处不一致

这一节不讨论“理论上应该怎么做”，只讨论当前源码里已经能看见的结构性不一致。

最值得优先梳理的有两处：

1. `channel_loss_dict/channel_loss_count_dict` 的统计分支未完成
2. AR/fast/generate 相关接口存在并行实现，边界不够清晰

### 26.1 `channel_loss_dict` 分支的现状

trainer 侧显然期望模型返回：

- `outputs.channel_loss_dict`
- `outputs.channel_loss_count_dict`

并且会按数据集名聚合日志：

- `wall_x/trainer/qwen_vl_act_trainer.py:440-483`

而模型输出结构体里也显式定义了这两个字段：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:71-72`

`train_step_forward()` 也会把它们塞回输出：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1470-1482`

但真正负责生成它们的 `compute_loss()` 里，当前初始化是：

- `unique_datasets_name = None`
- `channel_loss_dict = None`
- `channel_loss_count_dict = None`

位置在：

- `wall_x/model/vla_mixin.py:793-797`

后面却继续保留了：

- `for dataset_name_i in unique_datasets_name:`

以及对 `channel_loss_dict[dataset_name_i]` 的写入逻辑：

- `wall_x/model/vla_mixin.py:819-830`

这说明当前状态更像是：

- 原本设计过 per-dataset loss 统计
- 后来中途注释掉了初始化
- 但后续使用路径还没一起删干净或补完整

#### 这带来的工程风险

从结构上说，这里至少有三个风险：

1. 模型侧和 trainer 侧契约不一致  
   trainer 把它当成稳定输出字段，但模型内部并没有完整地构造这两个 dict。

2. 日志语义不稳定  
   即使代码在某些分支下没直接报错，这两个字段也无法代表可靠的 per-dataset 统计。

3. 后续维护者容易误判  
   因为表面上看 trainer 已经“支持 channel loss logging”，但实际上模型端这条链路并未真正闭环。

#### 一个更清晰的修复方向

如果从重构角度看，这部分最简单的选择只有两种：

方案 A：彻底补全

- 在 `compute_loss()` 中恢复 `unique_datasets_name`
- 初始化 `channel_loss_dict` / `channel_loss_count_dict`
- 明确只统计哪些 loss
- 保证 trainer 读到的字段始终存在且语义稳定

方案 B：暂时删掉这条统计链

- 模型不再返回 `channel_loss_dict`
- trainer 也不再读取这两个字段
- 等设计清楚后再单独恢复

当前源码最不利于维护的状态，其实就是现在这种：

- 类型定义保留
- trainer 使用保留
- 关键构造逻辑却处于半删除状态

### 26.2 AR / fast / generate 接口的现状

第二个问题是 AR 路径现在存在多套并行接口，阅读成本比较高。

先列出当前可见的几条路径。

#### 路径 1：`predict(..., predict_mode="fast")`

模型内部有一套清晰的 fast 路径：

- `predict()` 定义在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1503`

这条路径会：

1. 截 prompt 到 `<|im_start|>assistant`
2. 调用 `self.generate(...)`
3. `batch_decode(...)`
4. 把 fast action token 解码成连续动作

关键逻辑在：

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1727-1835`

从设计上看，这条链已经足够完整。

#### 路径 2：脚本层 `self.model.generate_ar_action(...)`

但推理脚本实际走 AR 时调用的是：

- `scripts/infer_robochallenge.py:1121-1137`

也就是：

- `self.model.generate_ar_action(...)`

然而从当前仓库可见代码检索看：

- 直接定义为 `def generate_ar_action(...)` 的实现并不在 `modeling_qwen2_5_vl_act.py` 里显式出现
- 可见的主要 AR 生成逻辑反而集中在 `predict(..., predict_mode="fast")`

这会带来一个阅读上的问题：

- 使用者看到脚本调用的是 `generate_ar_action`
- 但源码里最完整的 AR/fast 逻辑却写在 `predict()`

#### 路径 3：`forward(mode="predict")`

模型 `forward()` 的 mode 分发里还有一条：

- `mode == "predict"` 时，直接 `return self.generate_flow_action(predict_mode=predict_mode, **kwargs)`
- 位置在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:2389-2390`

这里又有一个值得警惕的点：

- `generate_flow_action()` 的函数签名并没有 `predict_mode` 参数
- 它的入口是 `input_ids, action_horizon, action_dim, ...`
- 定义在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py:1959`

也就是说，单从源码阅读上看：

- `forward(mode="predict")` 这条分发和 `generate_flow_action()` 的签名并不自然匹配
- 这更像是历史演化中遗留的一处接口重叠

### 26.3 当前 AR 接口最清晰的统一方向

如果只从可维护性出发，AR 相关接口最好压成单一路径。

最自然的统一方式是：

```text
外部统一调用一个显式的 action prediction API
  -> 内部根据 mode 选择 fast / diffusion
  -> fast 只保留一套实现
  -> diffusion 只保留一套实现
```

例如，从结构上更清晰的形态会是：

- `predict_action(predict_mode="fast" | "diffusion")`
- 或者 `predict(predict_mode=...)`

而不是同时保留：

- `predict(...)`
- `predict_action(...)`
- `generate_ar_action(...)`
- `generate_flow_action(...)`
- `forward(mode="predict")`

这些接口本身并不是都错，但它们叠在一起之后，会导致两个问题：

1. 使用者不清楚哪一个才是主入口
2. 维护者修改一条路径时，很容易漏掉另一条平行路径

### 26.4 这两类问题的共性

`channel_loss_dict` 和 AR 接口问题看起来不一样，但本质很像：

- 都是功能本身并不难理解
- 问题主要出在“历史迭代后留下了并行路径或半完成分支”

从工程治理角度，这类问题通常不需要大改模型，而是要做两件事：

1. 明确唯一主路径  
   哪个函数是正式入口，哪个只是兼容层，必须在代码层说清楚。

2. 清理半完成状态  
   对外暴露的字段和 trainer 依赖的字段，要么完整可用，要么暂时删掉。

## 27. 如果下一步要真正动代码，优先级建议

如果从“最小代价获得最大清晰度”的角度排优先级，我会建议：

1. 先处理 `channel_loss_dict`  
   因为它是训练日志契约问题，最容易误导后续实验结果解释。

2. 再统一 AR 接口  
   因为它是推理入口语义问题，最容易在新功能接入时造成路径分叉。

3. 最后再考虑更深层的数学或架构重构  
   比如 `use_x_pred`、flow/diffusion 命名统一、joint attention 路径抽象等。

这样做的好处是：

- 先把“看起来能用但语义不稳定”的部分收紧
- 再去处理“可以用但入口不统一”的部分
- 最后才碰真正高风险的结构改造
