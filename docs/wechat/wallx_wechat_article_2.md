# 从 `infer_robochallenge.py` 开始：Wall-X 的推理入口、数据加载和模型加载

> 理解一个模型项目，最省时间的方法通常不是先翻模型定义，而是先找真实入口。Wall-X 就很适合这么看。

**TL;DR**
- Wall-X 推理时接收的不是训练样本原样，而是现场 observation。
- 入口不是裸 `forward()`，而是脚本层的 `WallxModelWrapper`。
- `_construct_input()` 是最关键的一步，它把状态、图像、prompt、MoE 路由标记拼成最终 batch。
- 模型加载也不只是 `from_pretrained()`，而是 `processor + config 注入 + tokenizer 扩展 + checkpoint` 一整套流程。

## 一、为什么要从脚本入口开始看

Wall-X 真正的推理入口，不是模型类里某个抽象的 `forward()`，而是脚本：

- `scripts/infer_robochallenge.py`

这里有一个很关键的包装器：

- `WallxModelWrapper`

这意味着模型不是直接接收“已经整理好的 `input_ids` 和 `pixel_values`”，而是先接收更原始的现场输入：

- `state`
- `views`
- `instruction`

然后再由包装器把这些东西组织成模型真正能吃的 batch。

如果你跳过这一层，直接去看模型内部，很容易把很多运行时拼出来的字段误以为是“模型天然就有的”。

## 二、推理时的数据不是训练样本原样复用

这是个很容易忽略的点。

训练时你会想到 dataset、collator、batch sampler。  
但推理时，Wall-X 接收的是一份 observation，而不是训练样本对象本身。

在 `predict_action_rtc()` 里，一次推理的起点其实是：

- `state`
- `views`
- `instruction`

然后脚本先调用 `preprocess(...)`，把它们变成新的 observation 字典。

这个 observation 里至少包含：

- 多路相机图像
- `agent_pos`
- `agent_pos_mask`
- `dof_mask`

这里后面这三项非常重要，因为它们会一路进入动作侧模型，而不是“只给推理做个参考”。

换句话说，Wall-X 的推理输入从一开始就不是“图像 + 文本”这么简单，而是：

> 图像 + 指令 + 机器人状态 + 动作自由度约束

## 三、一次完整推理的调用顺序是什么

把脚本侧逻辑压缩一下，大致是这样：

```text
state / views / instruction
-> preprocess(...)
-> _construct_input(...)
-> generate_ar_action(...) 或 generate_flow_action(...)
-> 反归一化
-> 输出后处理
```

这里有两个特别关键的判断。

第一，Wall-X 对外最终要返回的是动作序列，不是单纯文本。  
第二，脚本层已经把动作路径分叉做掉了：

- AR 路径
- flow / diffusion 路径

也就是说，后面不管你看哪条动作生成链，前面的输入装配逻辑其实是共享的。

## 四、`_construct_input()` 才是推理输入装配的核心

如果说 `predict_action_rtc()` 是总入口，那么 `_construct_input()` 就是整条推理链里最值得认真看的函数。

它基本上在做 4 件事：

1. 处理并归一化状态信号
2. 构造文本 prompt
3. 预处理图像并展开视觉占位符
4. 把文本和图像拼成 batch，再挂上额外字段

这 4 件事拼起来，才是模型真正看到的输入。

## 五、状态信号是怎么进来的

`_construct_input()` 一上来先取：

- `agent_pos`
- `agent_pos_mask`
- `dof_mask`

然后如果存在 normalizer，就先对状态做归一化，并把它放进：

- `proprioception`

这里有两个层面。

第一层是数值层面：状态先归一化，避免不同机器人、不同量纲直接混在一起。  
第二层是语义层面：状态不仅会作为数值张量传入模型，还可能被写进 prompt。

这就引出 Wall-X 里一个很有意思的设计：

- 如果开启 `state_str`，状态会被离散化成字符串，拼到 `Proprioception:` 后面
- 如果不开，就直接在文本里放一个 `<|propri|>` 占位符，后面再替换成 embedding

这说明 Wall-X 对状态至少支持两种表达：

- 文本化的状态
- token/embedding 化的状态

## 六、prompt 不是一段文本，而是一套模板

Wall-X 推理时的 prompt 不是“把 instruction 塞进去”这么简单，而是脚本按模板动态拼出来的。

最主要的两条路径是：

- `get_text_ar(...)`
- `get_text_flow(...)`

它们的共同点是：

- 都有 system / user / assistant 结构
- 都会把图像写成 `Observation: ... <|vision_start|><|image_pad|><|vision_end|>`
- 都会显式写出“预测下一步机器人动作”
- 都会带 `Proprioception:` 这段

区别是：

- AR 路径更像语言模型继续生成
- flow 路径会在 assistant 段后面直接补一串 `<|action|>`

这个区别非常重要。因为它说明 action horizon 不只是模型内部的事，它从 prompt 这层就已经开始影响序列布局了。

## 七、`<|image_pad|>` 为什么最后会展开成一大段

很多人第一次看多模态 prompt 会疑惑：

文本里明明只有一个 `<|image_pad|>`，为什么最后模型里会占那么长一段 token？

答案就在 `_construct_input()`。

脚本会先把图像送进 image processor，拿到一个很关键的字段：

- `image_grid_thw`

然后根据这个网格大小，把 `<|image_pad|>` 按实际视觉 token 数量展开。

所以 `<|image_pad|>` 在 prompt 里只是一个书写占位符，不代表“图像只占一个 token”。  
真正送进模型前，它会按 patch / merge 后的视觉网格被扩成一长段。

这也是为什么理解 `image_grid_thw` 很重要。因为它直接决定视觉 token 数量，也直接影响后面 decoder 看到的序列长度。

## 八、batch 拼完以后，脚本还会再挂一层额外字段

文本和图像经过 tokenizer / image processor 之后，会被合成一个 `BatchFeature`。  
但这还没完。

Wall-X 还会在这个阶段继续往 batch 里加：

- `proprioception`
- `agent_pos_mask`
- `dof_mask`
- `moe_token_types`
- `dataset_names`

其中最关键的是：

```python
moe_token_types = inputs.input_ids == action_token_id
```

这句话非常有信息量。

它说明当前推理里，MoE 路由信号并不神秘，而是脚本直接按 token 语义标出来的：

- 普通 token
- action token

这也为后面 decoder 里的 expert 路由埋下了明确入口。

## 九、模型加载并不只是 `from_pretrained()` 一句话

再看 `get_model_and_processor()`，你会发现模型加载流程远比一句 `from_pretrained()` 丰富。

大致顺序是：

1. 读 `config.yml`
2. 加载 `AutoProcessor`
3. 可选加载 action tokenizer
4. 注册 normalizer
5. 根据 `model_type` 选模型类和配置类
6. 读 config
7. 用训练/推理配置覆盖基础 config
8. 显式设置 attention backend
9. 实例化模型
10. 加载 safetensors checkpoint

这里至少有三点值得记住。

### 第一，processor 和 model 是分开加载的

processor 来自：

- `processor_path`

而模型结构配置来自：

- `qwen_vl_act_config_path`

它们不一定是同一个目录。

### 第二，config 不是照抄 checkpoint

脚本会在加载完 config 之后，再跑一遍配置注入。  
所以最终生效的模型配置，其实是“基础 config + 上层 yaml 覆盖”的结果。

### 第三，attention 后端是在推理时被显式指定的

当前常见设置是：

- 文本：`sdpa`
- 视觉：`flash_attention_2`

这也解释了为什么 Wall-X 不是“整套统一后端”的模型。

## 十、这一篇真正要立住的判断

如果把这篇文章压成一句话，那就是：

> Wall-X 的推理不是把一段文本和一张图直接喂给模型，而是先由脚本把 observation 组装成一个带视觉、状态、动作语义和 MoE 路由标记的复合 batch，再交给动作模型去做后续生成。

这个入口链看明白以后，下一篇再看视觉主干和文本主干怎么汇合，就会顺很多。

