# 02. 测试 WALL-X 开源具身多模态模型：第二篇，训练链路、数据集适配与模型结构梳理

上一篇已经把 `WALL-X` 的推理链路跑通了，这一篇不急着直接堆训练命令，而是先把三个最关键的问题讲清楚：

1. 这个仓库的训练链路到底是怎么串起来的  
2. 当前使用的数据集和模型结构之间是怎么对齐的  
3. 真正开始训练前，哪些地方必须先修，哪些地方可以先保守跑通

如果第一篇回答的是：

> 这个仓库能不能跑推理？

那第二篇更关注的是：

> 这个仓库的训练到底要怎么理解，怎么改，怎么安全地起第一轮？

## 一、这篇文章的目标

这一篇不是“训练跑分贴”，而是一篇训练前的工程梳理文章。

目标有三层：

- 从源码结构上看清训练入口
- 从数据集结构上看清输入输出
- 从模型结构上看清训练时真正发生了什么

这也是具身项目和很多普通视觉/文本项目不太一样的地方。

对大多数开源仓库来说，训练难点常常只是“机器够不够大”。  
但对 `WALL-X` 这种具身多模态仓库来说，真正的难点更接近：

- 数据维度和动作空间能不能对齐
- 模型内部动作分支到底期待什么格式
- 配置文件里那套 `dof_config` / `agent_pos_config` 到底和原始数据是不是一个空间

也就是说，训练前最重要的不是立刻开跑，而是先把“链路理解”做对。

## 二、训练入口到底在哪里

从仓库结构看，训练主入口其实很清晰：

- 顶层脚本：`train_qact.py`
- 训练器：`wall_x/trainer/qwen_vl_act_trainer.py`
- 数据加载：`wall_x/data/load_lerobot_dataset.py`
- 主模型：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py`
- 动作头：`wall_x/model/action_head.py`

如果把训练链路压缩成一句话，就是：

> 配置文件 -> Trainer -> LeRobot 数据集 -> DataCollator -> Qwen2.5-VL Action 模型 -> loss -> optimizer

这条链路在源码里是显式存在的，不是散在一堆 notebook 里。

### 1. 顶层脚本做什么

`train_qact.py` 主要做四件事：

1. 读取 YAML 配置  
2. 初始化 `Accelerator`
3. 初始化日志
4. 构造 `QwenVlAct_Trainer` 并调用 `fit()`

也就是说，这个脚本本身并不复杂，复杂度集中在 Trainer、Data 和 Model 三层。

### 2. Trainer 做什么

`QwenVlAct_Trainer` 是训练主控。

它主要负责：

- 加载 normalizer
- 加载模型
- 加载数据
- 创建 optimizer / scheduler
- 执行 train loop

这意味着如果训练起不来，最常见的问题其实也会集中在这三层：

- `load_normalizer`
- `load_model`
- `load_qact_data`

## 三、这次训练实际选了什么硬件

这个系列文章的硬件路线是分阶段的：

- 第一篇推理：`RTX 4090 24GB`
- 第二篇训练：`A800 80G`
- 第三篇再考虑推理加速和更稳定的服务化路径

为什么训练切到 `A800 80G` 很重要？

因为推理和训练对资源的要求不是一个数量级。

在仓库自己的说明里，`lerobot/aloha_mobile_cabinet` 单卡训练显存实测大约就是 `40GB+` 级别。  
所以：

- `4090 24GB` 适合先做推理验证
- `A800 80G` 才更像是真正能把训练链路稳定拉起来的硬件

## 四、这次实际使用的数据集是什么

训练例子用的是：

- `lerobot/aloha_mobile_cabinet`

这个数据集的特点很重要，因为它决定了为什么训练前必须先做结构检查。

这里需要先澄清一个很容易被误解的问题：

> `WALL-X` 不是“只用 aloha_mobile_cabinet 训练出来的模型”。

更准确地说：

- `WALL-X` / `WALL-OSS` 是一个更大的具身多模态模型体系
- 当前仓库给出的训练示例，选用了 `lerobot/aloha_mobile_cabinet`
- 这个数据集是“测试训练链路”的起点，不是整个模型体系训练来源的全部

换句话说，`aloha_mobile_cabinet` 在这里的角色更像：

- 一个仓库内可复现的样例数据集
- 一个足够具体、足够真实的训练测试入口

而不是：

- 整个 `WALL-X` 体系唯一的数据来源

这个区别很重要，因为它直接影响你对仓库定位的理解。

如果把 `WALL-X` 误解成“单数据集模型”，很多后续分析都会偏掉；  
但如果把它理解成“具身基础模型工程栈 + 一个可复现训练样例”，那仓库结构和配置设计就更好理解了。

### 1. 数据集里到底有什么

从下载后的 `meta/info.json` 可以直接看到：

- 图像：
  - `observation.images.cam_high`
  - `observation.images.cam_left_wrist`
  - `observation.images.cam_right_wrist`
- 状态：
  - `observation.state`
- 动作：
  - `action`

这说明它是一个标准的具身数据组织方式：

- 多相机视觉
- 机器人状态
- 对应动作轨迹

### 2. 原始 action/state 实际是多少维

这是这次训练前排查里最关键的发现之一。

从数据集元信息看：

- `observation.state` 是 `14` 维
- `action` 也是 `14` 维

并且每一维的含义都明确写在 `info.json` 里，属于 ALOHA 电机空间：

- `left_waist`
- `left_shoulder`
- `left_elbow`
- `left_forearm_roll`
- `left_wrist_angle`
- `left_wrist_rotate`
- `left_gripper`
- `right_waist`
- `right_shoulder`
- `right_elbow`
- `right_forearm_roll`
- `right_wrist_angle`
- `right_wrist_rotate`
- `right_gripper`

这件事非常关键，因为它直接决定了：

> 训练配置里的动作空间如果不是 14 维原生空间，就一定要做映射或者 padding。

## 五、为什么当前示例配置不能直接无脑训练

这次训练前最大的真实问题，不是缺依赖，也不是模型下载，而是：

> 示例配置的动作空间和数据集原始动作空间不完全一致。

### 1. 配置里的训练空间是 20 维

在 `workspace/lerobot_example/config_qact.yml` 这类配置里，可以看到训练时使用的是：

- `follow_left_ee_cartesian_pos`
- `follow_left_ee_rotation`
- `follow_left_gripper`
- `follow_right_ee_cartesian_pos`
- `follow_right_ee_rotation`
- `follow_right_gripper`
- `head_actions`
- `height`
- `car_pose`

把这些维度加起来是 `20` 维。

### 2. 数据集原始空间是 14 维

而 `aloha_mobile_cabinet` 的原始 `action/state` 是 `14` 维电机空间。

这就带来一个结构性问题：

- 模型训练配置期待的是 `20` 维输入
- 数据集给的是 `14` 维原始电机值

如果直接硬塞进去，训练就不可能稳定起。

### 3. 仓库里其实已经留了这个痕迹

在 `wall_x/data/load_lerobot_dataset.py` 的 `DataCollator` 里，可以看到原作者其实已经留了“把维度 pad 到 20”的代码片段，只是默认被注释掉了。

这说明什么？

说明这个问题不是偶发问题，而是仓库本身就已经意识到：

> 原始具身数据和统一动作空间之间，需要一层适配。

## 六、这次训练前实际做了哪些修复

为了让训练链路先“跑起来”，这次做的是最小必要修复，而不是一步追求完美动作语义。

### 1. 在 DataCollator 里补了 14 -> 20 维 padding

这一步的目的很简单：

- 对 `action`
- 对 `agent_pos`
- 以及对应的 mask

统一 pad 到模型配置期待的 20 维。

这样做的好处是：

- 不改变原始数据顺序
- 让缺失的维度通过 mask 屏蔽
- 先保证训练图能过

### 2. 单独生成了一份 A800 训练用的 norm stats

训练配置里需要：

- `norm_stats_path`

但 `LeRobot` 原始下载结果里并没有一个可以直接拿来给 `wall-x` 用的 `stats.json`。

所以这次额外做了一步：

- 基于 `aloha_mobile_cabinet` 的 `action/state`
- 生成一份适配 `wall-x` 配置键的 `norm_stats`

路径是：

- `/root/autodl-tmp/norm_stats/aloha_mobile_cabinet_stats.json`

### 3. 修了训练配置里缺失的 `processor_path`

如果直接走 `wall-oss` 路径加载模型，Trainer 最终会调用 `load_wallx_processors(train_config)`。  
而这一层需要配置里有：

- `processor_path`

所以这次也做了最小修复：

- 把 `processor_path` 指向和 `pretrained_wallx_path` 一样的模型目录

## 七、模型结构该怎么理解

如果只看名字，很多人会以为这就是一个“Qwen2.5-VL 加一点动作头”的模型。

但从源码看，它其实比这复杂一些。

### 1. 主干是什么

主干基于：

- `Qwen2.5-VL`

也就是说它天然具备：

- 图像理解
- 文本建模
- 多模态输入处理

### 2. 额外加了什么

在 `modeling_qwen2_5_vl_act.py` 和相关 mixin 里，可以看到它又叠了几层：

- 动作 token 相关处理
- proprioception 注入
- 动作专家 / MoE 路由
- action preprocessor
- flow / fast 两种动作分支逻辑

所以它不是把机器人动作单纯拼接到输入后面，而是显式地把动作空间变成模型内部的一等公民。

### 3. ActionProcessor 的作用

`wall_x/model/action_head.py` 里的 `ActionProcessor` 是训练理解的关键。

它负责的事情包括：

- 处理动作 chunk
- 处理 proprioception
- 动作归一化 / 反归一化
- 在 flow 分支里构造动作相关隐表示

如果不理解这一层，就很容易误以为：

> 训练只是把图片和文字喂进去，然后随便回归动作

但真实情况是，这个仓库对动作空间做了明确建模。

## 八、训练数据链是怎么走的

这一层通常比模型本身更容易踩坑。

训练时的主要链路是：

1. `LeRobotDataset` 读取 parquet + 视频  
2. `PreprocessedDataset` 负责样本级视觉和文本整理  
3. `DataCollator` 负责 batch 级对齐、归一化、mask 构造  
4. `preprocesser_call()` 交给 processor 做真正的 token 化

### 1. 图像怎么处理

图像不是原尺寸直接喂进去，而是：

- 先根据视角分辨率配置做 resize
- 再走 `smart_resize`
- 最后变成 Qwen2.5-VL 能接受的视觉输入

### 2. 文本怎么处理

文本不是死模板，而是根据：

- instruction
- frame index
- action horizon

动态构造出来的。

然后再由 processor 加上多模态模板，形成真正送入模型的输入。

### 3. 动作和状态怎么处理

动作和状态在进入模型前都要经过：

- mask 生成
- NaN 处理
- padding 到目标维度
- normalizer 归一化

这一层就是训练能不能跑通的关键之一。

## 九、这次实际采用的训练方案是什么

如果直接照仓库原始长配置去跑，训练时间会非常夸张。

按这次在 A800 上观察到的稳定单步速度，完整长跑版会是“天级”而不是“小时级”实验。  
这显然不适合作为系列第二篇的主实验。

所以在真正进入训练阶段后，最终采用的不是原始长跑版，而是一套更适合验证的短实验配置：

- 单卡 `A800 80G`
- 单 epoch
- `num_training_steps = 10000`
- 只取少量 episode
- 单卡、离线日志

可以把它理解成：

> 一个更适合“第二篇训练测试”定位的 1 小时级版本

这套配置的核心目标不是追求最终最优结果，而是：

- 让训练链路完整成立
- 控制实验时长
- 让 loss、吞吐和资源占用都能被观察到

## 十、训练 smoke run 和 1 小时版分别说明了什么

为了不一上来就烧完整训练时长，最开始先做了一次非常克制的 smoke run：

- 单卡 `A800 80G`
- 只取少量 episode
- `batch_size_per_gpu = 1`
- `gradient_accumulation_steps = 1`
- `WANDB_MODE=offline`

目的只有一个：

> 证明训练图不是只能初始化，而是真的能进入迭代。

### Smoke run 的结果

这次 smoke run 最终不是停在初始化，而是已经连续跑到了：

- `iter 102`

单步耗时大约在：

- `0.41s` 左右

日志里大致能看到：

- `forward-compute`: `~130ms`
- `backward-compute`: `~117-125ms`
- `optimizer`: `~74-77ms`

这说明对这次训练测试来说，最重要的一句话已经成立：

> 训练链路已经被真正打通了。

这一步回答的是：

> 这套训练图到底能不能真跑起来？

答案是肯定的。

但 smoke run 的角色到这里其实就够了。

它的价值主要在于：

- 验证链路
- 提前暴露结构问题
- 证明训练不是只能初始化

真正更适合作为第二篇主实验结果的，是那套更短的 1 小时级配置。

### 1 小时版为什么更适合作为主实验

在 A800 上，1 小时版最终的训练规模被收敛到：

- 总 step 数约 `9975`
- 单步时间约 `0.41s`

据此估算，总训练时长约为：

- `9975 × 0.41s ≈ 68 分钟`

这也是为什么第二篇真正应该展示的主实验不是长跑版，而是这一版：

- 时间可控
- 训练真实发生
- 足够体现源码、数据和模型结构之间的关系
- 更适合后续复现实验和文章展示

### 训练后 checkpoint 的最小验证

到这里还不能只停在“训练在跑”，因为还有一个很实际的问题：

> 训练结束后产出的 checkpoint，能不能重新拿来推理？

这次也补了一步最小验证：

- 把 `1h` 训练产出的 checkpoint 目录整理成可推理目录
- 补齐 `config.json` 和 tokenizer / processor 相关文件
- 再用它重新跑 fake inference 和一条最小 VQA

这个验证的意义很直接：

- 证明训练不只是产出中间文件
- 证明新的 checkpoint 已经可以重新进入推理链路

为了让这个验证过程可复用，也额外整理了两个直接可执行的脚本：

```bash
bash scripts/run_fake_ft_1h.sh
bash scripts/run_vqa_ft_1h.sh
```

它们默认都会指向：

```text
/root/autodl-tmp/outputs/wall-x-train-1h/0
```

更重要的是，这一步也确认了当前训练的本质：

- 它是以原始 `wall-oss-flow` 权重为初始化继续训练
- 不是随机初始化
- 也不是从另一个中断训练状态恢复 optimizer 后继续跑

也就是说，这次训练更准确的性质是：

> 从基础模型出发做一轮具身适配微调

### `base` 和 `ft_1h` 的最小对比说明了什么

为了避免只看 loss，这次还做了一次最小对比：

- `base`: 原始 `wall-oss-flow`
- `ft_1h`: 1 小时训练后整理出的 checkpoint

在同一张图片、同一个问题上，二者都能正常生成回答。  
这说明：

- 训练后的 checkpoint 没有把推理能力直接训坏
- 权重更新已经能够反映到生成行为上

但这一步更应该被理解成：

- “训练结果已经可以被调用”

而不是：

- “已经能证明新模型整体优于原模型”

这也是第二篇在实验边界上必须保持克制的地方。

## 十一、这次训练测试最大的经验是什么

如果只总结一个训练前最重要的经验，那就是：

> 具身训练仓库最容易卡住的地方，不是“卡不够大”，而是“动作空间和原始数据空间没有对齐”。

这次最大的真实工作量，不是在改 optimizer，也不是在改学习率，而是在处理：

- 原始 14 维 ALOHA 电机空间
- 训练期 20 维统一动作空间
- normalizer 格式
- collator 对齐逻辑

这也是为什么这篇文章不把重点放在“训练命令是什么”，而是先放在：

- 源码结构
- 数据结构
- 模型结构

因为具身项目一旦这三层没理解清楚，后面的训练只会变成重复报错。

## 十二、第二篇接下来的方向

到这一步，训练已经不是“能不能起”的问题，而是“怎么把它变成真正可用的训练实验”的问题。

接下来的训练测试，重点会放在：

1. 是否继续保留当前 14 -> 20 的适配方式  
2. 是否把 `customized_dof_config` 调整到更严格一致  
3. 更正式的训练配置怎么设  
4. checkpoint、恢复训练和更长时间运行是否稳定

换句话说，这一篇解决的是：

> 训练链路是否成立

下一篇真正进入的会是：

> 训练链路如何从“能跑”变成“能用”

## 十三、这一篇的结论

如果第一篇推理测试回答的是：

> WALL-X 能不能在 4090 上跑起来？

那么这一篇回答的是：

> WALL-X 这套训练链路，在 A800 80G 上是不是已经具备继续深入实验的条件？

答案是肯定的。

因为这次已经完成了：

- `lerobot` 安装
- 数据集下载
- 模型迁移
- CUDA 扩展重编
- 数据维度适配
- norm stats 生成
- 训练 smoke run 跑通
- 1 小时级主实验配置落地

这就意味着，后面已经可以从“训练前准备”进入“真正的训练实验”阶段了。
