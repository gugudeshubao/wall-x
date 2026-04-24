# 从 `infer_robochallenge.py` 开始：Wall-X 的推理入口、数据加载和模型加载

上一篇我先做了一件更基础的事：把 Wall-X 的结构判断立住。结论很简单，Wall-X 不是一个“整套推理主干都靠自定义 CUDA 算子重写”的模型。它的自定义算子主要集中在 MoE、RoPE 和视觉 window/index，主体仍然是 PyTorch + Transformers 这条技术栈。

但只知道这一点还不够。

因为你后面不管是分析模型结构、排查推理瓶颈，还是考虑做服务化，都会遇到一个更实际的问题：Wall-X 运行时到底是怎么把外部输入变成模型张量的？

这个问题如果不先讲清楚，后面很多讨论都会飘。你会分不清：

- 哪些字段来自现场 observation
- 哪些字段是 prompt 构造出来的
- 哪些字段是视觉分支额外需要的
- 哪些字段是 MoE 和动作预测链自己附加进去的

所以这一篇不急着进 decoder layer，也不急着讲 MoE。先顺着实际推理调用顺序，把 3 件事讲清楚：

1. 推理入口是谁在调谁  
2. 运行时 observation 是怎么变成模型输入的  
3. 模型类和配置是怎么被加载起来的  

## 为什么从 `infer_robochallenge.py` 开始看

理解一个项目最省时间的方式，通常不是先去翻模型类定义，而是先找真实入口。

Wall-X 在这个仓库里最直接的入口，不是某个抽象的 `forward()`，而是脚本层的 `scripts/infer_robochallenge.py`。这里有一个很关键的包装器类：`WallxModelWrapper`。

这意味着什么？

意味着模型并不是直接接收“已经准备好的 `input_ids` 和 `pixel_values`”，而是先经过一层运行时包装逻辑。外部进来的其实是更原始的东西：

- 机器人状态 `state`
- 多路相机画面 `views`
- 自然语言指令 `instruction`

而 `WallxModelWrapper` 负责把这些东西整理成模型真正能吃的 batch。

所以如果你一开始跳过这个包装层，直接去看模型类，很容易误以为很多张量是“模型天然就会有的”。其实不是。这里面相当一部分字段，都是脚本在推理时现组的。

## 运行时输入不是训练样本原样复用

这是一个很容易被忽略的点。

很多人看到“数据加载”四个字，脑子里会先想到训练时的 dataset、collator、batch sampler。但推理时不是这条链。Wall-X 在推理阶段接收的不是训练样本原样复用，而是现场 observation。

在 `predict_action_rtc(...)` 里，真正进来的参数是：

- `state`
- `views`
- `instruction`
- `valid_action_dim`

然后脚本先调用 `preprocess(...)`，把这些原始输入整理成一个新的 `observation` 字典。

这个 `observation` 至少包含几类东西：

- 多路相机视图，对应不同 camera name
- `agent_pos`
- `agent_pos_mask`
- `dof_mask`

这里最值得注意的是后 3 个字段。它们不是“顺手带一下”的辅助信息，而是后面整条动作预测链都会依赖的输入。

尤其是：

- `agent_pos` 代表状态向量本体
- `agent_pos_mask` 代表哪些状态维度当前有效
- `dof_mask` 代表动作自由度范围，在 action horizon 维度上也会被保留

换句话说，Wall-X 的推理输入从一开始就不是“只有图片和文本”，而是“图像 + 指令 + 机器人状态 + 动作自由度约束”的组合体。

## `predict_action_rtc()` 才是一次完整推理的真正起点

在脚本层，一次完整的动作预测，是从 `predict_action_rtc(...)` 开始的。

这条调用链可以压缩成这样：

```text
state / views / instruction
-> preprocess(...)
-> _construct_input(...)
-> generate_ar_action(...) 或 generate_flow_action(...)
-> 反归一化
-> 输出后处理
```

这里有两个特别重要的判断。

第一个判断是，Wall-X 在推理时并不是直接“吐 token”，而是最终要返回动作序列。也就是说，文本生成只是中间能力，真正的对外输出是动作。

第二个判断是，脚本层已经在这里把模式分叉做掉了：

- 如果 `action_predict_mode == "ar"`，走 AR action 路径
- 否则走 flow / diffusion 路径

也就是说，你后面分析模型时必须始终记住：Wall-X 顶层有两种动作生成方式，但它们共享前面的输入构造链。

## `_construct_input()` 是推理输入装配的核心

如果说 `predict_action_rtc()` 是总入口，那么 `_construct_input()` 就是最关键的一步。因为它完成了从 observation 到模型 batch 的真正转化。

这个函数大致在做 4 件事：

1. 处理并归一化状态信号  
2. 构造文本 prompt  
3. 预处理图像并展开视觉占位符  
4. 生成 tokenizer / image processor 最终 batch，并附加额外控制字段  

这 4 件事串起来，才是模型实际看到的输入。

## 第一步：状态信号先被归一化，再决定是否进 prompt

`_construct_input()` 开头先从 `observation` 里取出：

- `agent_pos`
- `agent_pos_mask`
- `dof_mask`

然后如果存在 proprioception normalizer，就先对 `agent_pos` 做归一化，并把它保存在 `additional_inputs["proprioception"]` 里。

这一步有两个层次。

第一个层次是数值层面：状态信号先归一化，避免模型接收到的物理量尺度太乱。  
第二个层次是语义层面：状态信号既可能作为数值张量直接传给模型，也可能进一步被写进 prompt。

这一点在 `get_text_ar(...)` 和 `get_text_flow(...)` 里体现得很明显。

如果脚本参数打开了 `state_str`，归一化后的状态还会被离散化成 `0-255` 区间的字符串，然后拼进：

```text
Proprioception: ...
```

如果没打开 `state_str`，则直接在 prompt 里放一个 `<|propri|>` 占位符，后面再由模型侧把它替换成真正的 proprio embedding。

这个设计很有意思，因为它说明 Wall-X 对状态信号其实支持两种表达方式：

- 一种是“把状态写成文本”
- 一种是“把状态当成专门的控制 token / embedding”

这也是后面理解 `<|propri|>` 很重要的前提。

## 第二步：prompt 不是一段文本，而是一套模板

Wall-X 推理时的 prompt 不是“把 instruction 塞进去”这么简单，而是由脚本动态拼出来的一套模板。

仓库里至少能看到两条主要路径：

- `get_text_ar(...)`
- `get_text_flow(...)`

它们有共同点，也有区别。

共同点是：

- 都会以 system / user / assistant 这种 chat template 风格组织
- 都会把视觉输入写成 `Observation: ... <|vision_start|><|image_pad|><|vision_end|>`
- 都会在 prompt 里显式要求“预测下一步机器人动作”
- 都会带上 `Proprioception:` 这一段

区别是：

- AR 路径的 assistant 开头后面不直接补 action 占位符
- flow 路径会在 assistant 段后面拼上连续的 `<|action|>`，数量等于 `action_chunk_size`

这意味着 flow 路径在文本层面已经提前把“未来要预测多少个 action token”编码进 prompt 结构里了。  
所以动作 horizon 不是只在模型内部出现，它从 prompt 这层就已经开始影响序列布局。

## 第三步：图像不是直接喂进去，先要做 resize 和占位符展开

接下来是视觉输入。

`_construct_input()` 会调用 `resize_images(observation)`，把 observation 里的多路图像读出来，转成 `PIL.Image`，然后经历两层处理：

第一层是目标分辨率约束。  
第二层是 `smart_resize(...)`，用 `IMAGE_FACTOR`、`MIN_PIXELS`、`MAX_PIXELS` 这些参数把分辨率调整到模型能接受的范围。

然后图像会送进：

```python
self.processor.image_processor(images=image_inputs, videos=None, return_tensors="pt")
```

这一步的关键输出不是只有 `pixel_values`，还有一个特别重要的字段：`image_grid_thw`。

为什么它重要？

因为 Wall-X 的文本里本来只是写了一个 `<|image_pad|>` 占位符，但真正进模型时，一个图像通常不会只对应一个 token。脚本会根据 `image_grid_thw` 计算每张图像在 LLM 序列里实际应该展开成多少个视觉 token。

对应逻辑大致是：

```python
"<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
```

这里的 `merge_length` 来自 `merge_size ** 2`。这说明视觉 token 的数量不是拍脑袋定的，而是和图像经过 patch / merge 之后的网格大小直接相关。

这一点非常值得记住。因为它解释了一个很多人第一次看多模态 prompt 时会困惑的问题：

为什么 prompt 里看上去只有一个 `<|image_pad|>`，但最后 `input_ids` 里视觉 token 占了很长一段？

答案就是：文本模板里的 `<|image_pad|>` 只是一个书写占位符，真正送进模型前，脚本会按 `image_grid_thw` 把它展开。

## 第四步：tokenizer 之后，batch 还会再挂一层额外字段

文本和图像都准备好以后，脚本会分别经过 tokenizer 和 image processor，然后合并成一个 `BatchFeature`。

这时你可能以为输入构造结束了，但其实还没有。Wall-X 在这个阶段还会往 batch 里再塞一批额外字段：

- `proprioception`
- `agent_pos_mask`
- `dof_mask`
- `moe_token_types`
- `dataset_names`

其中最关键的是 `moe_token_types`。

脚本的做法非常直接：

```python
action_token_id = tokenizer.convert_tokens_to_ids("<|action|>")
moe_token_types = inputs.input_ids == action_token_id
```

这意味着在当前推理脚本里，`moe_token_types` 本质上就是一个布尔张量：哪些位置是 action token，哪些位置不是。

这个判断后面会直接影响 MoE 路由。也就是说，MoE 的输入分类不是某种很玄的隐式机制，而是脚本层已经根据 token 语义显式标出来了。

到这里，模型真正拿到的就不再只是“文本和图像”，而是一组更完整的输入：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`
- `proprioception`
- `agent_pos_mask`
- `dof_mask`
- `moe_token_types`
- 以及一些数据集和动作相关的辅助字段

这就是为什么我前面说，Wall-X 的推理输入不是简单的多模态对话，而是一个带动作控制语义的复合 batch。

## 模型加载也不只是 `from_pretrained()` 一句话

讲完输入装配，再看模型怎么被加载。

`WallxModelWrapper.get_model_and_processor()` 这段逻辑做的事，远比一句 `from_pretrained()` 丰富。

它的顺序大致是：

1. 先读 `config.yml`
2. 初始化 `AutoProcessor`
3. 视情况初始化 action tokenizer
4. 注册 proprio / action normalizer
5. 根据 `model_type` 选择 `ModelClass` 和 `ConfigClass`
6. 从 act config 路径加载模型配置
7. 用训练配置对模型配置做增量更新
8. 强制设置文本和视觉的 attention 后端
9. 实例化模型
10. 加载 checkpoint 的 `model.safetensors`
11. 设置 normalizer、迁移设备、转部分参数到 bfloat16

这里至少有 3 个地方值得单独强调。

### 第一，processor 和 model 是分开加载的

processor 通过：

```python
AutoProcessor.from_pretrained(config["processor_path"], use_fast=True)
```

加载。

而模型配置则是从 `qwen_vl_act_config_path` 走 `ConfigClass.from_pretrained(...)` 加载的。

这说明 tokenizer / image processor 的来源，和模型结构配置的来源，不必是同一个目录。

### 第二，配置不是照抄 checkpoint，而是会被二次注入

脚本里在加载完基础 config 之后，还会调用：

```python
model_config = update_model_config(config, model_config)
```

也就是说，最终生效的模型配置不是单靠 `config.json` 决定，而是“基础配置 + 训练/推理 yaml 注入”的结果。

这件事对排查问题很重要。因为你如果只盯着 checkpoint 自带的配置文件看，很可能会漏掉脚本层的覆盖逻辑。

### 第三，attention 后端是在推理时被显式改掉的

脚本紧接着就做了两句非常关键的设置：

```python
model_config._attn_implementation = "sdpa"
model_config.vision_config._attn_implementation = "flash_attention_2"
```

这也是为什么上一篇我会说：Wall-X 当前推理配置下，文本走 SDPA，视觉走 FlashAttention2。

这个结论不是靠猜出来的，是推理脚本直接写死的。

## 这里还有一条更通用的配置加载路径

如果你不只看 `infer_robochallenge.py`，还会发现仓库里还有一条更通用的推理配置入口：`wall_x/infer/infer_config.py`。

它的 `_load_model_config()` 逻辑说明了两件事：

- 优先使用 checkpoint 目录里的 `config.json`
- 如果没有，再回退到训练配置里的 `qwen_vl_act_config_path`

然后同样会做：

- `update_model_config(...)`
- 文本 `sdpa`
- 视觉 `flash_attention_2`

所以从工程角度看，Wall-X 的配置加载思路是统一的：  
先拿到基础模型配置，再用上层配置做覆盖，最后在推理阶段明确 attention backend。

## 模型实例化之后，还要补一层动作侧能力

模型类选定之后，脚本不是简单地 `from_pretrained()` 一个现成权重对象，而是先直接实例化：

```python
model = ModelClass(model_config, ...)
```

这里的 `ModelClass` 在当前仓库里通常就是 `Qwen2_5_VLMoEForAction`。

这个名字其实已经很说明问题了。它不是纯文本模型，也不是纯视觉模型，而是一个“为 action 预测改造过的多模态模型”。

实例化之后，脚本还会做几件和动作侧很相关的事：

- `resize_token_embeddings(...)`
- 从 `model.safetensors` 加载参数
- `set_normalizer(...)`
- `to_bfloat16_for_selected_params()`

这些动作都说明，推理阶段真正跑起来的对象，并不是一个裸的 Qwen2.5-VL，而是“Qwen2.5-VL 主干 + Wall-X 自己的动作头和归一化体系”。

## 把这一篇压成一句话

如果把这一篇的核心结论只压成一句话，那就是：

> Wall-X 的推理不是“把一段文本和一张图送进模型”这么简单，而是脚本先把 observation 组装成一个带视觉、状态、动作语义和 MoE 路由标记的复合 batch，再交给 `Qwen2_5_VLMoEForAction` 去做后续生成。

你只有先把这个入口链看清楚，后面分析模型结构时才不会误判。

因为到了这个阶段你已经知道：

- 哪些信息来自运行时 observation
- 哪些信息来自 prompt 模板
- 哪些信息来自 image processor 的网格展开
- 哪些信息是 MoE 和动作链额外附加进去的

这就为下一篇铺好了路。

下一篇我会正式进入模型骨架本身，专门讲 Wall-X 的视觉主干和文本主干是怎么汇合的。到那时再回头看这一篇里的 `image_grid_thw`、`<|image_pad|>`、`<|propri|>`，很多细节会一下子变得很顺。

## 附：文中对应的关键源码位置

- `WallxModelWrapper` 和模型加载入口：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L372)
- `processor`、`ModelClass`、`ConfigClass`、`update_model_config`：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L400)
- 文本/视觉 attention 后端在脚本里被显式设置：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L465)
- AR prompt 构造：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L581)
- flow prompt 构造：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L629)
- 运行时输入装配 `_construct_input(...)`：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L731)
- 图像占位符按 `image_grid_thw` 展开：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L788)
- `moe_token_types` 的生成：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L811)
- 推理总入口 `predict_action_rtc(...)`：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L970)
- flow / AR 分支调用：[`infer_robochallenge.py`](/Users/sam/project/github/wall-x/scripts/infer_robochallenge.py#L1047)
- 通用推理配置加载 `_load_model_config()`：[`infer_config.py`](/Users/sam/project/github/wall-x/wall_x/infer/infer_config.py#L531)
- `Qwen2_5_VLMoEForAction` 的顶层模型定义：[`modeling_qwen2_5_vl_act.py`](/Users/sam/project/github/wall-x/wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py#L773)
